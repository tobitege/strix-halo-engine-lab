#pragma once
// Cohort requests on the tiled MTP prompt (--mtp-cohort-request with the tiled known-prefix prefill): the
// batched runner with one slot per request. lockstep admits every request (prompts one after another), then
// decodes them together until all end; sequential admits each request after the previous one ended, each
// exactly as alone (the in-process reference). The receipt keeps the cohort receipt's fields.
#include "strix/env_flags.hpp"
#include "hip_decode_options.hpp"
#include "hip_decode_receipt_json.hpp"
#include "hip_mtp_decode_bridge.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <memory>
#include <ostream>
#include <string>
#include <vector>

namespace strix::decode_cli {

inline MtpDecodeBatchServe make_cohort_tiled_run(const Options& options, const ModelContract& model, std::ostream& output) {
    return [&options, &model, &output](MtpBatchRunner& runner) {
        const auto& m = options.mtp;
        std::vector<std::vector<std::uint32_t>> prompts{options.tokens};
        for (const auto& request : m.cohort) prompts.push_back(request.tokens);
        if (prompts.size() > runner.slots()) throw std::logic_error("cohort on the tiled prompt has fewer slots than requests");
        const bool lockstep = m.cohort_mode.empty() || m.cohort_mode == "lockstep";
        const std::vector<std::uint32_t> stops{options.eos_token.value_or(static_cast<std::uint32_t>(model.ple_eos_token_id))};
        struct Record {
            std::size_t slot = 0;
            std::vector<std::uint32_t> generated;
            MtpServeResult result;
            std::shared_ptr<TokenSampler> sampler;
            std::string prompt_receipt, draft_vocabulary;
            std::array<std::uint64_t, 9> chosen_rows{};
            std::uint64_t final_position = 0;
            double start_ms = 0.0, finish_ms = 0.0;
            bool done = false;
        };
        std::vector<Record> records(prompts.size());
        auto start = std::chrono::steady_clock::now();
        double decode_ms = 0.0; // sequential: the requests' own decode phases
        const auto now_ms = [&] { return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count(); };
        const auto admit = [&](std::size_t q) {
            auto& record = records[q];
            MtpBatchRequest request;
            request.prompt = prompts[q];
            request.max_tokens = options.generate_tokens;
            request.stop_tokens = stops;
            if (mtp_sampling(m)) {
                auto policy = mtp_sampling_policy(m);
                policy.seed += q;
                record.sampler = request.sampler = std::make_shared<TokenSampler>(policy);
            }
            request.on_tokens = [&record](std::span<const std::uint32_t> tokens) {
                record.generated.insert(record.generated.end(), tokens.begin(), tokens.end());
                return true;
            };
            request.on_finish = [&record, &runner, &now_ms](const MtpServeResult& result) {
                record.result = result;
                record.done = true;
                record.finish_ms = now_ms();
                const auto& slot = runner.slot(record.slot);
                record.chosen_rows = slot.chosen_rows;
                record.prompt_receipt = slot.prompt_receipt;
                record.final_position = slot.session->next_position;
            };
            // The slot of the next admission (the first free one) is known before the call.
            for (std::size_t s = 0; s < runner.slots(); ++s)
                if (!runner.slot(s).active) { record.slot = s; break; }
            runner.admit(std::move(request));
        };
        double prompts_ms = 0.0; // lockstep with interleaved admission: from admission to the end of the last prompt
        if (lockstep) {
            for (std::size_t q = 0; q < prompts.size(); ++q) admit(q);
            // Interleaved admission (STRIX_BATCH_INTERLEAVE=1): the prompts advance in the steps (in groups with
            // STRIX_PROMPT_BATCH=1), with the decode cycles of the requests already past theirs.
            if (runner.interleaved()) {
                const auto admitted = std::chrono::steady_clock::now();
                while (runner.prefilling()) runner.step();
                prompts_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - admitted).count();
            }
            start = std::chrono::steady_clock::now();
            while (runner.active()) runner.step();
            decode_ms = now_ms();
        } else {
            for (std::size_t q = 0; q < prompts.size(); ++q) {
                admit(q);
                start = std::chrono::steady_clock::now();
                records[q].start_ms = 0.0;
                while (runner.active()) runner.step();
                if (!records[q].done) throw std::logic_error("sequential cohort request did not finish");
                decode_ms += records[q].finish_ms;
            }
        }
        const auto& stats = runner.stats();
        std::uint64_t total_tokens = 0;
        for (const auto& r : records) total_tokens += r.generated.size() - 1U;
        const auto stop_name = [](const std::string& finish) {
            return finish == "stop" ? "eos" : finish == "length" ? "token-budget" : finish == "context" ? "context-capacity" : finish.c_str();
        };
        output << "{\n" << prefill_identity_fields(prefill_options_active()) << ",\n  \"mode\":\"native-mtp-cohort-tiled-prompt-target-verified-" << (mtp_sampling(m) ? "sampled" : "greedy") << "\",\n"
            << "  \"mtp_native_hardware_qualified\":false,\n  \"mtp_performance_qualified\":false,\n"
            << "  \"mtp_cohort_mode\":\"" << (lockstep ? "lockstep" : "sequential") << "\",\n"
            << "  \"mtp_cohort_shared_steps\":" << (strix::envcfg::flag("STRIX_COHORT_SHARED", true) ? "true" : "false") << ",\n"
            << "  \"mtp_cohort_requests\":" << records.size() << ",\n"
            << "  \"mtp_calibration_rows\":" << runner.calibration.rows << ",\n  \"mtp_calibration_ms\":" << runner.calibration.ms << ",\n"
            << "  \"mtp_cohort_windows\":" << stats.cohort.cohort_windows << ",\n  \"mtp_cohort_window_sessions\":" << stats.cohort.cohort_sessions
            << ",\n  \"mtp_cohort_window_rows\":" << stats.cohort.cohort_rows << ",\n  \"mtp_cohort_solo_windows\":" << stats.cohort.solo_windows << ",\n  \"mtp_cohort_chunks\":" << stats.cohort.cohort_chunks
            << ",\n  \"mtp_cohort_steps\":" << stats.steps << ",\n  \"mtp_cohort_cohort_steps\":" << stats.cohort_steps << ",\n"
            << "  \"prompt_batch\":{\"enabled\":" << (runner.prompt_batching() ? "true" : "false") << ",\"interleaved\":"
            << (runner.interleaved() ? "true" : "false") << ",\"prompt_steps\":" << stats.prompt_steps << ",\"groups\":" << stats.prompt_groups
            << ",\"grouped_blocks\":" << stats.grouped_blocks << ",\"grouped_rows\":" << stats.grouped_rows
            << ",\"group_ms\":" << stats.prompt_group_ms << ",\"prompts_ms\":" << prompts_ms << "},\n"
            << "  \"mtp_cohort_wall_ms\":" << decode_ms << ",\n  \"mtp_cohort_generated_tokens\":" << total_tokens << ",\n"
            << "  \"mtp_cohort_aggregate_tokens_per_second\":" << (decode_ms > 0.0 ? 1000.0 * static_cast<double>(total_tokens) / decode_ms : 0.0) << ",\n"
            << "  \"mtp_timing_scope\":\"post-prompt-cycles-only-host-completed-wall-not-GPU-kernel-time\",\n"
            << "  \"requests\":[";
        for (std::size_t q = 0; q < records.size(); ++q) {
            const auto& r = records[q];
            const auto tokens = r.generated.size() - 1U;
            const auto elapsed = r.finish_ms - r.start_ms;
            output << (q ? ",\n" : "\n") << "    {\"request\":" << q << ",\"slot\":" << r.slot << ",\"prompt_tokens\":" << prompts[q].size()
                << ",\"tiled_prompt\":{" << r.prompt_receipt << "}"
                << ",\"prompt_ms\":" << r.result.prompt_ms << ",\"restored_tokens\":" << r.result.restored_tokens
                << ",\"mtp_replay_maximum_hidden_abs_error\":" << runner.calibration.maximum_hidden_abs_error
                << ",\"mtp_replay_maximum_logit_abs_error\":" << runner.calibration.maximum_logit_abs_error
                << ",\"mtp_cycles\":" << r.result.cycles << ",\"mtp_proposed_candidates\":" << r.result.proposed
                << ",\"mtp_accepted_candidates\":" << r.result.accepted << ",\"mtp_chosen_window_rows_histogram\":{";
            for (std::size_t t = 0, first = 1; t < r.chosen_rows.size(); ++t)
                if (r.chosen_rows[t]) { output << (first ? "" : ",") << "\"" << t << "\":" << r.chosen_rows[t]; first = 0; }
            output << "},\"start_ms\":" << r.start_ms << ",\"finish_ms\":" << r.finish_ms << ",\"cycle_tokens\":" << tokens
                << ",\"tokens_per_second\":" << (elapsed > 0.0 ? 1000.0 * static_cast<double>(tokens) / elapsed : 0.0)
                << ",\"final_position\":" << r.final_position << ",\"stop_reason\":\"" << stop_name(r.result.finish) << "\"";
            // Diagnostic only (STRIX_FINAL_STATE_DIGEST=1 or STRIX_PREFIX_CACHE_DIGEST=1), absent otherwise.
            // final_target_state_digest: the committed target state alone; it depends on the request's tokens only,
            // so it is the one to compare across runs, cohort sizes and against a solo request.
            // final_state_digest: target state plus the MTP proposer cache; with the default mixed proposer cache
            // profile that part also depends on where the cycles started, which follows wall-clock admission.
            if (!r.result.final_state_digest.empty())
                output << ",\"final_state_position\":" << r.result.final_state_position << ",\"final_state_digest\":\""
                       << r.result.final_state_digest << "\"";
            if (!r.result.final_target_state_digest.empty())
                output << ",\"final_target_state_digest\":\"" << r.result.final_target_state_digest << "\"";
            // With STRIX_PREFIX_CACHE_DIGEST_COMPONENTS=1: one digest per registered component (target components and
            // the MTP predictor's mtp.kv, mtp.index and mtp.held), so the target state can be compared on its own.
            if (!r.result.final_state_components.empty())
                output << ",\"final_state_components\":" << r.result.final_state_components;
            output << ",\"sampling\":" << mtp_sampling_receipt(m, r.sampler.get()) << ",\"generated_tokens\":[";
            for (std::size_t i = 0; i < r.generated.size(); ++i) output << (i ? "," : "") << r.generated[i];
            output << "]}";
        }
        output << "\n  ]\n}\n";
    };
}

} // namespace strix::decode_cli
