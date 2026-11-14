#pragma once
#include "strix/env_flags.hpp"
#include "hip_mtp_decode_bridge.hpp"
#include <cstdlib>

namespace strix::decode_cli {
// One request of an MTP cohort: its own target parents (state memory, scratch, PLE prefetch),
// session, serial dispatcher and window factory over the shared model.
struct MtpCohortRequest {
    std::shared_ptr<MtpDecodeTargetParents> parents;
    SerialSessionContext* session = nullptr;
    DirectLaunchDispatcher* dispatcher = nullptr;
    std::vector<std::uint32_t> tokens;
    std::string golden, golden_sha256;
    MtpDecodeWindowFactory make_window;
};

// Several independent requests in one process (CLI benchmark mode). Every request is prepared as
// the single-request route prepares it: serial prompt over its own state, complete independent
// replay against its own golden (the declared error policy, 0 by default), qualification and
// activation of its own draft plan, then generation verified by its own target. lockstep runs one
// cycle of every active request at a time and verifies their drafts with cohort windows
// (hip_mtp_greedy_cohort_cycle); sequential runs each request to completion before the next. The
// MTP sidecar assets are loaded once; each request has its own MTP runtime and draft state.
inline void run_mtp_cohort_bridge(const Options& options, const ModelContract& model,
    const CompiledSerialExecution& compiled, std::vector<MtpCohortRequest>& requests,
    AllocationGenerationSource& generations, bool deferred_status, std::string_view resolved_profile,
    std::ostream& output) {
    validate_resolved_mtp_cli(options);
    const auto& m = options.mtp;
    if (!m.enabled || requests.size() < 2U || requests.size() > 4U)
        throw std::invalid_argument("MTP cohort requires two to four requests");
    const auto pair = numerical_profile_pair(options);
    if (pair.split()) throw std::invalid_argument("MTP cohort requires one numerical profile for prompt and decode");
    const auto profile_identity = numerical_profile_identity(pair.decode);
    const auto width = mtp_cli_mul(model.hidden_size, model.hc_count);
    const auto& actual0 = *requests.front().parents->experiment;
    HipMtpAssetLoadRequest load; load.sidecar_path = m.sidecar; load.sidecar_logical_name = m.sidecar_logical_name;
    load.expected_target_file_sha256 = mtp_cli_digest(m.target_sha256); load.expected_target_content = mtp_cli_digest(m.target_content);
    auto& evidence = load.expected_sidecar; evidence.repository = m.repository; evidence.revision = m.revision;
    evidence.relative_path = m.source_path;
    evidence.declared_file_sha256 = mtp_cli_digest(m.sidecar_sha256); evidence.declared_file_bytes = m.sidecar_bytes;
    evidence.conversion = MtpSidecarConversionProfile::effective_norm_gain_embedding_first_packed_v1;
    evidence.converter_reference_sha256 = mtp_cli_digest(m.converter_sha256);
    evidence.norm_converter_reference_sha256 = mtp_cli_digest(m.norm_converter_sha256);
    evidence.converter_reference_revision = m.converter_revision;
    load.metadata_read_budget_bytes = m.header_bytes;
    load.residency.mode = m.sidecar_residency == "copy" ? HipModelResidencyMode::resident_copy : HipModelResidencyMode::registered_file_mapping;
    load.residency.reservation_limits = {m.sidecar_retained_bytes, m.sidecar_setup_bytes};
    const auto assets = HipMtpAssets::load(model, load, actual0.mapped, actual0.manifest, actual0.model, generations);
    HipMtpBudget native_budget; native_budget.retained_bytes = m.runtime_bytes; native_budget.context_capacity = options.context_tokens;
    native_budget.maximum_candidate_depth = m.candidate_depth;
    native_budget.normalized_rounding = m.normalized_rounding == "fp32" ? HipMtpInputRounding::fp32 : HipMtpInputRounding::bf16_rne;
    native_budget.projection_rounding = m.projection_rounding == "fp32" ? HipMtpInputRounding::fp32 : HipMtpInputRounding::bf16_rne;
    const MtpForwardReplayPolicy policy{m.policy_identity, m.max_hidden_error, m.max_logit_error, 2U, true};
    (void)assess_mtp_forward_replay(policy, {});
    const std::array<std::uint32_t, 1> eos{options.eos_token.value_or(static_cast<std::uint32_t>(model.ple_eos_token_id))};

    struct Live {
        std::shared_ptr<HipMtpResources> native;
        std::shared_ptr<HipMtpRuntime> runtime;
        std::shared_ptr<HipMtpTargetHidden> target;
        std::shared_ptr<HipMtpWindowVerifier> window;
        std::unique_ptr<HipMtpGreedyRuntime> greedy;
        HipMtpTargetHiddenTicket pending;
        HipMtpNativeForwardReplayResult replay;
        std::vector<std::uint32_t> generated;
        std::string stop = "token-budget", draft_vocabulary;
        std::uint64_t cycles = 0, proposed = 0, accepted = 0;
        std::array<std::uint64_t, 9> chosen_rows{};
        double finish_ms = 0.0, start_ms = 0.0;
        bool done = false;
    };
    std::vector<Live> live(requests.size());
    try {
        for (std::size_t q = 0; q < requests.size(); ++q) {
            auto& request = requests[q];
            auto& l = live[q];
            const auto& parents = request.parents;
            const auto& r = parents->resources;
            const auto& actual = *parents->experiment;
            auto& session = *request.session;
            if (!parents || !request.session || !request.dispatcher || !request.make_window || session.next_position ||
                session.committed_tokens || session.health != SerialSessionHealth::ready ||
                session.execution_identity != compiled.execution_identity || request.tokens.size() < 3U)
                throw std::invalid_argument("MTP cohort request requires a fresh actual target session, prompt and window factory");
            const auto golden = read_mtp_forward_golden_file(request.golden, mtp_cli_digest(request.golden_sha256), m.golden_max_bytes);
            if (golden.target_decode_profile_identity && (golden.target_prefill_profile_identity != profile_identity ||
                golden.target_decode_profile_identity != profile_identity))
                throw std::invalid_argument("MTP cohort golden was dumped under another numerical profile");
            if (golden.rows.size() + 1U != request.tokens.size())
                throw std::invalid_argument("MTP cohort golden must cover every prompt predictor pair");
            for (std::size_t i = 0; i < golden.rows.size(); ++i) {
                const auto& row = golden.rows[i];
                if (row.source_position != i || row.source_input_token != request.tokens[i] || row.next_input_token != request.tokens[i + 1U] ||
                    row.target_hidden_bf16.size() != width || row.next_hidden_bf16.size() != width || row.full_logits.size() != model.vocab_size)
                    throw std::invalid_argument("MTP cohort golden prompt chain or full-row geometry differs before target execution");
            }
            l.native = HipMtpResources::create(model, assets->compiled(), assets->sidecar_manifest(), assets->sidecar_residency(),
                assets->target_manifest(), assets->target_residency(), assets->shared_target(), native_budget, actual.allocator, model.rotary_identity());
            l.runtime = std::make_shared<HipMtpRuntime>(l.native);
            if (golden.target_content != assets->shared_target().target_model_content_identity ||
                golden.sidecar_content != assets->shared_target().sidecar_model_content_identity ||
                golden.structural_plan_identity != assets->compiled().plan_identity ||
                golden.semantic_identity != assets->compiled().forward_semantics.contract_identity ||
                golden.numerical_profile_identity != l.runtime->numerical_profile_identity())
                throw std::invalid_argument("MTP cohort golden does not bind actual verified assets, semantics and native cast profile");
            auto replay_payload = mtp_cli_add(mtp_cli_mul(width, 4U), mtp_cli_mul(model.vocab_size, 4U));
            replay_payload = mtp_cli_add(replay_payload, mtp_cli_mul(golden.rows.size(), sizeof(HipMtpForwardReplayRowReceipt) + 2U * sizeof(std::uint32_t)));
            replay_payload = mtp_cli_add(replay_payload, mtp_cli_mul(golden.rows.size(), mtp_cli_add(mtp_cli_mul(width, 4U), mtp_cli_mul(model.vocab_size, 4U))));
            replay_payload = mtp_cli_add(replay_payload, mtp_cli_mul(golden.rows.size() + 1U, 192U));
            if (replay_payload > m.replay_host_bytes)
                throw std::invalid_argument("MTP cohort independent replay exceeds host payload budget before target mutation");
            const HipMtpForwardReplayRequest replay_request{policy, hip_mtp_forward_golden_payload_identity(golden), golden.rows.size(), m.replay_host_bytes};
            const auto retained_rows = static_cast<std::uint32_t>(std::max<std::size_t>(request.tokens.size(),
                static_cast<std::size_t>(m.candidate_depth) + 3U));
            HipMtpTargetDeferredConfig d1;
            if (deferred_status) d1 = {r.session_state_layout, &generations};
            l.target = HipMtpTargetHidden::create(model, compiled, actual.memory, actual.model, actual.allocator,
                retained_rows, m.target_capture_bytes, d1);
            l.window = request.make_window(l.target);
            if (!l.window || l.window->target_owner() != l.target || l.window->logical_rows() != m.window_rows)
                throw std::invalid_argument("MTP cohort window factory returned incompatible actual owner");
            assets->validate_assets_unchanged();
            std::vector<HipMtpReplayRow> replay_rows; replay_rows.reserve(golden.rows.size());
            for (std::size_t i = 0; i < request.tokens.size(); ++i) {
                auto current = l.target->execute_serial(session, r, *request.dispatcher, request.tokens[i]);
                if (l.pending.valid()) replay_rows.push_back({l.pending, request.tokens[i]});
                l.pending = std::move(current);
            }
            l.replay = replay_hip_mtp_forward(*l.runtime, replay_rows, golden, replay_request);
            if (!l.replay.assessment.passes_declared_policy) {
                std::ostringstream error; error << std::setprecision(17)
                    << "MTP cohort request " << q << " replay failed its declared policy; hidden_max_abs_error="
                    << l.replay.observation.maximum_hidden_abs_error << "; logit_max_abs_error=" << l.replay.observation.maximum_logit_abs_error;
                throw std::runtime_error(error.str());
            }
            const auto qualified = qualify_mtp_draft_plan(assets->compiled(), policy, l.replay.observation);
            require_qualified_mtp_draft_plan(qualified);
            const auto activated = activate_mtp_draft_plan(assets->compiled(), policy, l.replay.observation);
            if (l.runtime->next_source_position() + 1U != session.next_position)
                throw std::logic_error("MTP cohort replay did not prime the actual final prompt frontier");
            replay_rows.clear();
            HipMtpDraftHeadConfig draft; draft.full = m.draft_vocab.empty() && !m.draft_context_vocab;
            if (!m.draft_vocab.empty()) {
                const auto spec = HipMtpDraftHead::resolve_vocabulary_spec(m.draft_vocab, request.tokens,
                    static_cast<std::uint32_t>(model.vocab_size), l.draft_vocabulary);
                if (spec == m.draft_vocab) l.draft_vocabulary = spec;
                draft.base_tokens = HipMtpDraftHead::read_token_file(spec, static_cast<std::uint32_t>(model.vocab_size));
            }
            draft.context_capacity = m.draft_context_vocab; draft.q4 = m.draft_q4;
            l.runtime->enable_draft_chain(std::move(draft), m.draft_exact); l.runtime->observe_draft_context(request.tokens);
            l.greedy = std::make_unique<HipMtpGreedyRuntime>(l.runtime, qualified, activated, l.target, session, r,
                *request.dispatcher, parents);
            l.greedy->use_window_verifier(l.window);
            l.greedy->set_draft_min_probability(mtp_draft_min_probability(m));
            l.generated.reserve(static_cast<std::size_t>(options.generate_tokens));
            l.generated.push_back(l.pending.predicted_token());
        }
        const bool lockstep = m.cohort_mode.empty() || m.cohort_mode == "lockstep";
        const auto done = [&](std::size_t q) {
            auto& l = live[q];
            if (l.done) return true;
            if (l.generated.back() == eos.front()) l.stop = "eos";
            else if (l.generated.size() >= options.generate_tokens) l.stop = "token-budget";
            else if (requests[q].session->next_position >= options.context_tokens) l.stop = "context-capacity";
            else return false;
            return l.done = true;
        };
        const auto apply = [&](std::size_t q, const HipMtpGreedyCycleResult& result) {
            auto& l = live[q];
            if (result.emitted_tokens.empty()) throw std::runtime_error("MTP cohort generation made no completed target progress");
            l.generated.insert(l.generated.end(), result.emitted_tokens.begin(), result.emitted_tokens.end());
            l.pending = result.final_target; ++l.cycles; l.proposed += result.proposed_candidates; l.accepted += result.accepted_candidates;
            ++l.chosen_rows[std::min<std::uint32_t>(result.target_window_rows, 8U)];
        };
        HipMtpGreedyCohortStats stats;
        const auto start = std::chrono::steady_clock::now();
        const auto now_ms = [&] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(); };
        if (!lockstep) {
            for (std::size_t q = 0; q < live.size(); ++q) {
                auto& l = live[q];
                l.start_ms = now_ms();
                while (!done(q)) {
                    const auto remaining = static_cast<std::uint32_t>(options.generate_tokens - l.generated.size());
                    apply(q, l.greedy->cycle(l.pending, m.candidate_depth, remaining, eos));
                }
                l.finish_ms = now_ms();
            }
        } else {
            for (;;) {
                std::vector<std::size_t> active;
                for (std::size_t q = 0; q < live.size(); ++q) {
                    if (!live[q].done && done(q)) live[q].finish_ms = now_ms();
                    if (!live[q].done) active.push_back(q);
                }
                if (active.empty()) break;
                if (active.size() == 1U) {
                    auto& l = live[active[0]];
                    const auto remaining = static_cast<std::uint32_t>(options.generate_tokens - l.generated.size());
                    apply(active[0], l.greedy->cycle(l.pending, m.candidate_depth, remaining, eos));
                    continue;
                }
                std::vector<HipMtpGreedyCohortMember> members;
                for (const auto q : active) {
                    auto& l = live[q];
                    members.push_back({l.greedy.get(), &l.pending, m.candidate_depth,
                        static_cast<std::uint32_t>(options.generate_tokens - l.generated.size()), eos});
                }
                const auto results = hip_mtp_greedy_cohort_cycle(members, &stats);
                for (std::size_t k = 0; k < active.size(); ++k) apply(active[k], results[k]);
            }
        }
        const auto wall = now_ms();
        assets->validate_assets_unchanged();
        std::uint64_t total_tokens = 0;
        for (const auto& l : live) total_tokens += l.generated.size() - 1U;
        output << "{\n" << prefill_identity_fields(prefill_options_active()) << ",\n  \"mode\":\"native-mtp-cohort-independent-replay-then-target-verified-greedy\",\n"
            << "  \"mtp_native_hardware_qualified\":false,\n  \"mtp_performance_qualified\":false,\n"
            << "  \"optimization_profile_selected\":\"" << decode_receipt::escape_json_string(resolved_profile) << "\",\n"
            << "  \"target_numerical_profile\":\"" << numerical_profile_name(pair.decode) << "\",\n"
            << "  \"mtp_cohort_mode\":\"" << (lockstep ? "lockstep" : "sequential") << "\",\n"
            << "  \"mtp_cohort_shared_steps\":" << (strix::envcfg::flag("STRIX_COHORT_SHARED", true) ? "true" : "false") << ",\n"
            << "  \"mtp_cohort_requests\":" << live.size() << ",\n"
            << "  \"mtp_cohort_windows\":" << stats.cohort_windows << ",\n  \"mtp_cohort_window_sessions\":" << stats.cohort_sessions
            << ",\n  \"mtp_cohort_window_rows\":" << stats.cohort_rows << ",\n  \"mtp_cohort_solo_windows\":" << stats.solo_windows << ",\n  \"mtp_cohort_chunks\":" << stats.cohort_chunks << ",\n"
            << "  \"mtp_cohort_wall_ms\":" << wall << ",\n  \"mtp_cohort_generated_tokens\":" << total_tokens << ",\n"
            << "  \"mtp_cohort_aggregate_tokens_per_second\":" << (wall > 0.0 ? 1000.0 * static_cast<double>(total_tokens) / wall : 0.0) << ",\n"
            << "  \"mtp_timing_scope\":\"post-prompt-and-independent-replay-cycles-only-host-completed-wall-not-GPU-kernel-time\",\n"
            << "  \"requests\":[";
        for (std::size_t q = 0; q < live.size(); ++q) {
            const auto& l = live[q];
            const auto tokens = l.generated.size() - 1U;
            const auto elapsed = l.finish_ms - l.start_ms;
            output << (q ? ",\n" : "\n") << "    {\"request\":" << q << ",\"prompt_tokens\":" << requests[q].tokens.size()
                << ",\"draft_vocabulary\":\"" << decode_receipt::escape_json_string(l.draft_vocabulary) << "\""
                << ",\"mtp_replay_rows\":" << l.replay.rows.size()
                << ",\"mtp_replay_maximum_hidden_abs_error\":" << l.replay.observation.maximum_hidden_abs_error
                << ",\"mtp_replay_maximum_logit_abs_error\":" << l.replay.observation.maximum_logit_abs_error
                << ",\"mtp_cycles\":" << l.cycles << ",\"mtp_proposed_candidates\":" << l.proposed
                << ",\"mtp_accepted_candidates\":" << l.accepted << ",\"mtp_chosen_window_rows_histogram\":{";
            for (std::size_t t = 0, first = 1; t < l.chosen_rows.size(); ++t)
                if (l.chosen_rows[t]) { output << (first ? "" : ",") << "\"" << t << "\":" << l.chosen_rows[t]; first = 0; }
            output << "},\"start_ms\":" << l.start_ms << ",\"finish_ms\":" << l.finish_ms << ",\"cycle_tokens\":" << tokens
                << ",\"tokens_per_second\":" << (elapsed > 0.0 ? 1000.0 * static_cast<double>(tokens) / elapsed : 0.0)
                << ",\"final_position\":" << requests[q].session->next_position
                << ",\"stop_reason\":\"" << l.stop << "\",\"generated_tokens\":[";
            for (std::size_t i = 0; i < l.generated.size(); ++i) output << (i ? "," : "") << l.generated[i];
            output << "]}";
        }
        output << "\n  ]\n}\n";
    } catch (...) {
        for (auto& l : live) if (l.runtime) l.runtime->cancel();
        throw;
    }
}
} // namespace strix::decode_cli
