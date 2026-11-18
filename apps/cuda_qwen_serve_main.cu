// Qwen3.8-Flash-Next serving on an RTX card: the node's OpenAI/Anthropic API front end (strix::serve: chat template,
// GGUF BPE tokenizer, SSE, /tokenize, /health) over the CUDA engine (qwen_prefill::Prefill). Greedy decode; every
// committed token is bit-identical to the Strix Halo node's dense-A8 decode, with or without MTP drafts.
//
//   strix-cuda-qwen-serve --model M.gguf --tables DIR --rope-table rope_cs.bin [--mtp-sidecar S.gguf --draft-vocab V.bin]
//       [--max-positions 32768] [--bind 127.0.0.1] [--port 8090] [--api-key-file F] [--model-id ID]
//       [--mtp-depth 7] [--min-p 0.7] [--max-queue 64] [--http-slots 4] [--prefill-chunk 1024|auto] [--prefill-overlap-rows 1024]
//       [--admit-grace-ms 2] [--cohort-depth fair|max] [--slot-margin-mib 2048] [--block-rows N]
//       [--expert-residency auto|full|tiered] [--vram-budget-gib N] [--print-effective-config]
//       [--prefix-cache on|off] [--prefix-cache-block 256] [--prefix-cache-stride 16384]
//       [--prefix-cache-host-mb N] [--prefix-cache-vram-mb 0]
// Context cap: --max-positions P is the largest session (KV/index/pooled + MTP cache, 31,616 B per position + 233 MB).
// Each request reserves its own session: prompt + max_tokens + 16 positions, rounded to --session-granule (256), capped
// at P (cuda_qwen_serve_policy.hpp over strix/serve/session_reservation.hpp; --session-reserve max: every session at P).
// Admission (plan_admission): an idle session that holds the request is reused as is; else a new session is built while
// cudaMemGetInfo free VRAM >= its bytes + --slot-margin-mib (env STRIX_QRTX_SLOT_MARGIN_MIB, default 2048: the WDDM
// fallback engaged at ~1.4 GiB free), else an idle session is re-sized or idle sessions are released; else the request
// waits (FIFO, head of line) for a running one to end (never a WDDM system-memory spill). So the slot count follows the
// free memory and the requests' sizes; --http-slots is only its upper bound. Every engine allocation also refuses past
// the card (strix/cuda/vram_guard.hpp). --block-rows N: one engine block size of N
// rows instead of 1024/2048/4096/8192 (whole-prompt prefill when nothing decodes; less activation memory).
// Prefill while others decode (decode first): chunks of --prefill-chunk rows (default 1024) between the decode steps.
// STRIX_QRTX_PREFILL_OVERLAP=1 runs the chunks on a second stream beside the steps (opt-in); --prefill-chunk auto sizes
// them from a fitted chunk time model (STRIX_QRTX_PREFILL_AUTO_K: target chunk time in decode steps, default 1).
// --http-slots N (default 16; env STRIX_QRTX_HTTP_SLOTS): up to N requests decode together. Each scheduler step gathers
// every decoding request into cohort verify windows (<= 8 rows: one row per request first, the spare rows to the MTP
// draft chains, cuda_qwen_cohort_plan.hpp); a new request is prefilled in chunks of --prefill-chunk rows between the
// steps (whole prompt blocks when nothing decodes). Every request's tokens equal it running alone.
// --http-slots 1 or STRIX_QRTX_COHORT=0: the old serialized path (one request at a time, generate() to completion).
// Token-block prefix checkpoints restore exact prompt state before the remaining prefill rows.
#include "strix/device/cuda_enumerator.hpp"
#include "strix/deploy/memory_plan.hpp"
#include "cuda_qwen_prefix_cache.hpp"
#include "strix/mtp_depth_host.hpp"
#include "cuda_qwen_serve_policy.hpp"
#include "strix/env_flags.hpp"
#include "strix/cuda/qwen_cohort.hpp"
#include "strix/cuda/qwen_options.hpp"
#include "strix/cuda/qwen_prefill.hpp"
#include "strix/residency/residency_cli.hpp"
#include "strix/serve/api.hpp"
#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/engine.hpp"
#include "strix/serve/http_server.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "strix/preflight.hpp"
#include "strix/qwen_weight_packs.hpp"

// The node's preflight (src/preflight.cpp) is POSIX-only; it is reached only through ApiConfig::health_record, which this
// server does not set.
std::string strix::render_preflight(const PreflightObservation&, const PreflightAssessment&) { return "{}"; }

namespace qp = strix::cuda::qwen_prefill;
namespace qc = strix::cuda::qwen_cohort;
namespace px = strix::cuda::qwen_prefix;
namespace qo = strix::cuda::qwen_options;
namespace serve = strix::serve;
using Clock = std::chrono::steady_clock;

namespace {

// The bound device's memory budget (strix/device/budget.hpp: --device-profile NAME or the entry's profile /
// memory_gib; 0 = no cap). Free memory is what the budget leaves after everything already on the card, so a smaller
// profile sees the slot count and the refusals a card of that size would.
std::uint64_t g_budget_bytes = 0;


std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop.store(true); }

struct Options {
    qp::Config engine;                          // from the shared parser (qwen_options.hpp), serve role applied
    std::string effective_config;               // --print-effective-config: JSON, printed instead of serving
    px::Settings prefix;
    std::string bind = "127.0.0.1", api_key_file, model_id = "qwen3.8-flash-next";
    int port = 8090;
    std::uint32_t mtp_depth = 7, max_queue = 64;
    float min_p = 0.7F;
    std::uint32_t http_slots = 16, admit_grace_ms = 2;
    std::uint32_t prefill_chunk = 1024;         // rows per prefill chunk while others decode; 0 = auto (fitted, see chunk_done)
    std::uint32_t prefill_overlap_rows = 1024;  // largest chunk on the prefill stream (overlap); its scratch is sized to it
    bool prefill_overlap = false;               // STRIX_QRTX_PREFILL_OVERLAP=1: chunks on a second stream beside the steps
    bool prefill_poll = false;                  // STRIX_QRTX_PREFILL_POLL=1 (overlap): decode steps keep running until the chunk
                                                // completes (cudaEventQuery), then it joins at the next step boundary
    double prefill_auto_k = 1.0;                // auto: target chunk time = k decode steps
    std::string cohort_depth = "fair";
    std::uint32_t slot_margin_mib = 2048;       // free VRAM that must remain after a new slot's session
    bool print_effective_config = false;        // resolve (GGUF header + live card/host numbers), print, exit: no model load
    cuda_qwen_serve::SessionPolicy session;     // per-request session capacity (one resolver for both serve paths)
};

Options parse(int argc, char** argv) {
    Options o;
    if (const char* v = std::getenv("STRIX_QRTX_PREFIX_CACHE")) o.prefix.enabled = v[0] != '0';
    qo::Request request;  // common engine + profile options; serve differences are named in resolve()
    if (const char* v = std::getenv("STRIX_QRTX_HTTP_SLOTS")) o.http_slots = static_cast<std::uint32_t>(std::stoul(v));
    if (const char* v = std::getenv("STRIX_QRTX_SLOT_MARGIN_MIB")) o.slot_margin_mib = static_cast<std::uint32_t>(std::stoul(v));
    strix::residency::CliOptions residency_cli;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
            return argv[++i];
        };
        const auto mib = [&]() -> std::uint64_t {
            const auto value = next();
            std::size_t end = 0;
            if (value.empty() || value[0] == '-') throw std::invalid_argument(a + " requires nonnegative MiB");
            const auto n = std::stoull(value, &end);
            if (end != value.size() || n > (std::numeric_limits<std::uint64_t>::max() >> 21U))
                throw std::invalid_argument(a + " budget out of range");
            return n << 20U;
        };
        const auto cache_count = [&]() -> std::uint32_t {
            const auto value = next();
            std::size_t end = 0;
            if (value.empty() || value[0] == '-') throw std::invalid_argument(a + " requires nonnegative tokens");
            const auto count = std::stoull(value, &end);
            if (end != value.size() || count > UINT32_MAX) throw std::invalid_argument(a + " token count out of range");
            return static_cast<std::uint32_t>(count);
        };
        if (qo::take(argc, argv, i, request)) continue;
        if (a == "--prefix-cache") {
            const auto value = next();
            if (value != "on" && value != "off") throw std::invalid_argument("--prefix-cache is on or off");
            o.prefix.enabled = value == "on";
        }
        else if (a == "--prefix-cache-block") o.prefix.block = cache_count();
        else if (a == "--prefix-cache-stride") o.prefix.stride = cache_count();
        else if (a == "--prefix-cache-host-mb") o.prefix.host_bytes = mib();
        else if (a == "--prefix-cache-vram-mb") o.prefix.vram_bytes = mib();
        else if (a == "--bind") o.bind = next();
        else if (a == "--port") o.port = std::stoi(next());
        else if (a == "--api-key-file") o.api_key_file = next();
        else if (a == "--model-id") o.model_id = next();
        else if (a == "--mtp-depth") o.mtp_depth = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--min-p") o.min_p = std::stof(next());
        else if (a == "--max-queue") o.max_queue = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--session-granule") o.session.granule = std::stoull(next());
        else if (a == "--session-reserve") {
            const auto v = next();
            if (v != "request" && v != "max") throw std::invalid_argument("--session-reserve request|max");
            o.session.per_request = v == "request";
        }
        else if (a == "--http-slots") o.http_slots = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--prefill-chunk") { const auto v = next(); o.prefill_chunk = v == "auto" ? 0U : static_cast<std::uint32_t>(std::stoul(v)); }
        else if (a == "--prefill-overlap-rows") o.prefill_overlap_rows = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--admit-grace-ms") o.admit_grace_ms = static_cast<std::uint32_t>(std::stoul(next()));
        else if (a == "--cohort-depth") o.cohort_depth = next();
        else if (strix::residency::consume_cli_arg(a, next, residency_cli)) {}  // --expert-residency, --vram-budget-gib, --print-effective-config
        else if (a == "--slot-margin-mib") o.slot_margin_mib = static_cast<std::uint32_t>(std::stoul(next()));
        else throw std::invalid_argument("unknown option " + a);
    }
    const auto resolved = qo::resolve(request, qo::Entry::serve);
    o.engine = resolved.config;
    if (!request.print_effective_config) {
        for (const auto& n : resolved.resolution.notices()) std::cerr << n << std::endl;
        std::cerr << "strix-cuda-qwen-serve: arithmetic " << resolved.contract << std::endl;
    }
    if (request.print_effective_config) {
        auto config = serve::parse_json(qo::effective_config_json(resolved));
        if (!o.engine.model_gguf.empty()) {
            const auto admission = strix::qwen_packs::admit_model(o.engine.model_gguf,
                strix::qwen_packs::expert_arithmetic_name(o.engine.expert_arithmetic));
            config.set("weight_pack", strix::weight_pack::to_json(admission));
        }
        o.effective_config = config.dump();
    }
    o.engine.residency = strix::residency::finish_cli(residency_cli);  // flags win over STRIX_EXPERT_RESIDENCY / STRIX_VRAM_BUDGET_GIB
    o.print_effective_config = request.print_effective_config;
    if (const char* v = std::getenv("STRIX_QRTX_COHORT"); v && v[0] == '0') o.http_slots = 1;
    if (o.http_slots == 0) throw std::invalid_argument("--http-slots must be >= 1");
    if (const char* v = strix::envcfg::text("STRIX_QRTX_SESSION_RESERVE")) o.session.per_request = std::string(v) != "max";
    if (const int g = strix::envcfg::integer("STRIX_QRTX_SESSION_GRANULE", 0); g > 0) o.session.granule = static_cast<std::uint64_t>(g);
    if (o.session.granule == 0) throw std::invalid_argument("--session-granule must be >= 1");
    o.session.max_positions = o.engine.max_positions;
    o.engine.session_positions = static_cast<std::uint32_t>(cuda_qwen_serve::initial_positions(o.session));
    if (const char* v = std::getenv("STRIX_QRTX_PREFILL_OVERLAP")) o.prefill_overlap = v[0] == '1';
    if (const char* v = std::getenv("STRIX_QRTX_PREFILL_POLL")) o.prefill_poll = v[0] == '1';
    if (const char* v = std::getenv("STRIX_QRTX_PREFILL_AUTO_K")) o.prefill_auto_k = std::stod(v);
    if (!(o.prefill_auto_k > 0)) throw std::invalid_argument("STRIX_QRTX_PREFILL_AUTO_K must be > 0");
    if (o.prefill_overlap_rows < 16) throw std::invalid_argument("--prefill-overlap-rows must be >= 16");
    if (o.cohort_depth != "fair" && o.cohort_depth != "max") throw std::invalid_argument("--cohort-depth is fair or max");
    px::validate_block(o.prefix.block);
    if (o.prefix.stride && o.prefix.stride % o.prefix.block)
        throw std::invalid_argument("--prefix-cache-stride must be 0 or a multiple of --prefix-cache-block");
    return o;
}

std::string read_key(const std::string& path) {
    if (path.empty()) return {};
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read --api-key-file");
    std::string key;
    std::getline(f, key);
    while (!key.empty() && (key.back() == '\r' || key.back() == ' ')) key.pop_back();
    return key;
}

class CudaQwenEngine final : public serve::GenerationEngine {
public:
    CudaQwenEngine(qp::Prefill& engine, const Options& o, px::Cache& cache) : e_(engine), o_(o), cache_(cache) {}

    serve::GenerationStats generate(serve::GenerationJob& job) override {
        serve::GenerationStats st;
        st.prompt_tokens = job.prompt.size();
        const auto t0 = Clock::now();
        const std::uint64_t n = job.prompt.size();
        const std::uint64_t cap = o_.engine.max_positions;
        if (n < 2) throw serve::GenerationRequestError(st, "prompt needs at least 2 tokens");
        if (n >= cap) { st.error_status = 400; st.error_code = "context_length_exceeded"; throw serve::GenerationRequestError(st, "prompt longer than the context"); }
        {
            // the same resolver and planner as the cohort admission (one slot: session 0, pinned)
            const auto need = cuda_qwen_serve::request_positions(o_.session, n, job.max_tokens);
            std::size_t free = 0, total = 0;
            if (cudaMemGetInfo(&free, &total) != cudaSuccess) { (void)cudaGetLastError(); free = 0; }
            const std::vector<serve::SlotSession> one{{false, true, true, e_.capacity(0), e_.session_bytes(0)}};
            const auto plan = serve::plan_admission(one, need, [&](std::uint64_t p) { return e_.session_bytes_at(p); }, free,
                                                    std::uint64_t{o_.slot_margin_mib} << 20U);
            if (plan.action == serve::AdmitAction::resize) e_.reserve_session(0, need);
            else if (plan.action != serve::AdmitAction::reuse) {
                st.error_status = 503;
                st.error_code = "insufficient_device_memory";
                throw serve::GenerationRequestError(st, "request does not fit device memory: " + plan.why);
            }
        }
        if (job.on_progress) job.on_progress({serve::GenerationPhase::prefill, 0, 0, n});
        double prefix_restore_ms = 0, prefix_capture_ms = 0;
        if (!cache_.enabled()) e_.run(job.prompt, n - 1, [](const qp::Piece&) {});
        else {
            auto prefilled = cache_.restore(e_, 0, job.prompt, prefix_restore_ms);
            st.cached_tokens = prefilled;
            const auto points = cache_.plan(n, prefilled);
            while (prefilled < n - 1) {
                const auto rows = px::clip_chunk(prefilled, UINT32_MAX, points);
                prefilled = e_.prefill_chunk(0, job.prompt, n - 1, prefilled, rows);
                if (std::binary_search(points.begin(), points.end(), prefilled)) prefix_capture_ms += cache_.capture(e_, 0, job.prompt, prefilled);
            }
        }
        const auto t1 = Clock::now();
        st.prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (job.on_progress) job.on_progress({serve::GenerationPhase::decode, n, 0, n});
        const bool mtp = !o_.engine.mtp_sidecar_gguf.empty() && o_.mtp_depth > 0;
        std::uint32_t depth = o_.mtp_depth;
        if (job.draft_depth_max) depth = std::min(depth, job.draft_depth_max);
        std::uint32_t pending = job.prompt[n - 1];
        std::uint64_t pos = n - 1;
        st.finish = "length";
        bool first = true;
        std::vector<std::uint32_t> emitted;
        // Measured-cost depth controller (STRIX_MTP_CONTROLLER=depth, shared with the HIP engine): picks the draft depth
        // and the verified rows per window; committed ids are unchanged (greedy verify, row-count invariant).
        strix::MtpDepthHost dh(o_.mtp_depth, o_.min_p, mtp && strix::mtp_depth_controller_enabled());
        while (st.generated_tokens < job.max_tokens) {
            if (pos + 1 >= cap) { st.finish = "context"; break; }
            const auto tw = Clock::now();
            std::vector<std::uint32_t> rows{pending};
            std::vector<float> probs;
            if (mtp) {
                const std::uint32_t want = dh.on() ? dh.plan(depth) : depth;
                const auto td = Clock::now();
                const auto d = e_.draft(pending, want, dh.on() ? dh.min_p() : o_.min_p);
                if (dh.on()) {
                    dh.note_draft(want,
                                  std::chrono::duration<double, std::milli>(Clock::now() - td).count());
                    probs = d.probability;
                    if (probs.size() > d.tokens.size()) probs.resize(d.tokens.size());
                }
                rows.insert(rows.end(), d.tokens.begin(), d.tokens.end());
                st.proposed += d.tokens.size();
            }
            const auto room = static_cast<std::uint32_t>(std::min<std::uint64_t>(rows.size(), cap - pos));
            rows.resize(room);
            bool selected = false;
            if (dh.on()) {
                probs.resize(std::min<std::size_t>(probs.size(), rows.size() - 1));
                if (probs.size() + 1 == rows.size()) {
                    const auto before = rows.size();
                    rows.resize(dh.select(probs));
                    st.proposed -= before - rows.size();
                    selected = true;
                }  // 1 = serial step
            }
            const auto r = e_.verify(rows.data(), static_cast<std::uint32_t>(rows.size()), pos);
            std::uint32_t keep = 1;
            while (keep < rows.size() && rows[keep] == r.argmax[keep - 1]) ++keep;
            e_.commit(keep);
            if (dh.on()) {
                if (selected) dh.finish(static_cast<std::uint32_t>(rows.size()), keep - 1,
                    std::chrono::duration<double, std::milli>(Clock::now() - tw).count());
                else dh.skip();
            }
            ++st.cycles;
            st.accepted += keep - 1;
            pos += keep;
            pending = r.argmax[keep - 1];
            // emit the verified tokens up to max_tokens and the first stop token (included)
            std::vector<std::uint32_t> out;
            bool stopped = false;
            for (std::uint32_t j = 0; j < keep && st.generated_tokens + out.size() < job.max_tokens; ++j) {
                out.push_back(r.argmax[j]);
                if (std::find(job.stop_tokens.begin(), job.stop_tokens.end(), r.argmax[j]) != job.stop_tokens.end()) { stopped = true; break; }
            }
            if (first && job.observed) st.first_token_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            first = false;
            st.generated_tokens += out.size();
            emitted.insert(emitted.end(), out.begin(), out.end());
            if (job.on_tokens && !job.on_tokens(out)) { st.finish = "cancelled"; break; }
            if (stopped) { st.finish = "stop"; break; }
        }
        st.decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - t1).count();
        char head[200];
        std::snprintf(head, sizeof(head), "{\"engine\":\"cuda-qwen-a8\",\"mtp_depth\":%u,\"tokens_per_cycle\":%.3f,\"prefill_ms\":%.2f,\"decode_ms\":%.2f,\"ids\":[",
                      mtp ? depth : 0U, st.cycles ? static_cast<double>(st.generated_tokens) / static_cast<double>(st.cycles) : 0.0,
                      st.prefill_ms, st.decode_ms);
        std::string extra = head;
        for (std::size_t i = 0; i < emitted.size() && i < 4096; ++i) extra += (i ? "," : "") + std::to_string(emitted[i]);
        extra += "],\"prefix_cached\":" + std::to_string(st.cached_tokens) + ",\"prefix_restore_ms\":" + std::to_string(prefix_restore_ms)
            + ",\"prefix_capture_ms\":" + std::to_string(prefix_capture_ms) + "}";
        st.engine_extra.emplace_back("rtx", extra);
        return st;
    }
    [[nodiscard]] std::uint64_t context_tokens() const override { return o_.engine.max_positions; }

private:
    qp::Prefill& e_;
    const Options& o_;
    px::Cache& cache_;
};

// Continuous batching over up to http_slots requests (one engine session each): between scheduler steps it admits
// queued requests into free slots; each step runs one cohort decode step over every decoding request (qc::step: drafts,
// packed windows, greedy acceptance), then one prefill chunk of the oldest request still prefilling (decode first).
// The tokens a request emits, its stop rule and its receipts follow CudaQwenEngine::generate() exactly.
class CohortServer {
public:
    CohortServer(qp::Prefill& engine, const Options& o, serve::GenerationQueue& queue, px::Cache& cache) : e_(engine), o_(o), queue_(queue), cache_(cache) {
        mtp_ = !o_.engine.mtp_sidecar_gguf.empty() && o_.mtp_depth > 0;
        dh_ = std::make_unique<strix::MtpDepthHost>(std::min<std::uint32_t>(o_.mtp_depth, qc::kWindowRows - 1U), o_.min_p, mtp_ && strix::mtp_depth_controller_enabled());
        if (o_.prefill_overlap) e_.enable_prefill_overlap(o_.prefill_overlap_rows);
        // slot 0 is the engine's session 0 (re-sized per request, never released); the others get theirs at admission
        for (std::uint32_t i = 0; i < o.http_slots; ++i) {
            slots_.push_back(Slot{});
            slots_.back().has_session = i == 0;
        }
        const auto vr = vram();
        const auto at = [&](std::uint64_t p) { return e_.session_bytes_at(p) >> 20U; };
        std::cerr << "strix-cuda-qwen-serve: VRAM after load " << (vr.first >> 20U) << " MiB free of " << (vr.second >> 20U)
                  << " MiB; sessions " << cuda_qwen_serve::describe(o_.session) << ": " << at(1280) << " MiB at 1280 positions, "
                  << at(o_.engine.max_positions) << " MiB at " << o_.engine.max_positions << "; slot margin " << o_.slot_margin_mib
                  << " MiB; up to " << o_.http_slots << " slots, as many as the free memory holds" << std::endl;
        max_chunk_ = o_.prefill_overlap ? e_.prefill_overlap_rows() : 1024U;
        if (o_.prefill_overlap && o_.prefill_chunk > max_chunk_)
            std::cerr << "strix-cuda-qwen-serve: --prefill-chunk " << o_.prefill_chunk << " capped at " << max_chunk_
                      << " rows (--prefill-overlap-rows) for overlapped chunks" << std::endl;
    }

    void run() {
        for (;;) {
            admit();
            if (queue_.stopping()) {
                if (inflight_) {  // the chunk in flight completes before its request is finished
                    try { (void)e_.prefill_wait(); } catch (...) {}
                    inflight_ = nullptr;
                }
                for (auto& s : slots_) if (s.job) finish(s, "server shutting down", "cancelled");
                if (held_) {
                    serve::GenerationStats st;
                    st.finish = "cancelled";
                    queue_.finish(*held_, st, "server shutting down");
                    held_.reset();
                }
                return;
            }
            bool any = false;
            for (auto& s : slots_) any = any || s.job.has_value();
            if (!any) continue;  // take() woke for a control request
            // decode first. Overlap: the oldest prefilling request's chunk runs on the prefill stream (no shared buffer; the
            // request is in no window) while decode steps run. Wait mode: one decode step, then wait for the chunk. Poll
            // mode: decode steps keep running and the chunk is polled after each; it joins at the next step boundary once
            // complete; the loop blocks on it only when no session can decode.
            if (!inflight_ && e_.prefill_overlap() && decoding()) {
                Slot* slot = oldest_prefill();
                if (slot && slot->job->ticket->cancelled()) { finish(*slot, {}, "cancelled"); slot = nullptr; }
                if (slot) {
                    try {
                        const auto rows = px::clip_chunk(slot->prefilled, next_chunk_rows(true), slot->capture_points);
                        e_.prefill_chunk_async(slot->session, slot->job->job.prompt, slot->n - 1, slot->prefilled, rows);
                        inflight_ = slot;
                        inflight_steps_ = 0;
                    } catch (const std::exception& ex) {
                        finish(*slot, ex.what(), "length");
                    }
                }
            }
            const bool stepped = decoding();
            if (stepped) {
                decode_step();
                if (inflight_) ++inflight_steps_;
            }
            if (inflight_) {
                std::optional<qp::PrefillChunk> r;
                const bool block = !o_.prefill_poll || !decoding();
                const auto tw = Clock::now();
                try {
                    r = block ? std::optional<qp::PrefillChunk>(e_.prefill_wait()) : e_.prefill_poll();
                } catch (const std::exception& ex) {
                    Slot* slot = inflight_;
                    inflight_ = nullptr;
                    finish(*slot, ex.what(), "length");
                    continue;
                }
                if (block) inflight_->prefill_wait_ms += std::chrono::duration<double, std::milli>(Clock::now() - tw).count();
                if (r) {
                    Slot* slot = inflight_;
                    inflight_ = nullptr;
                    slot->overlap_steps += inflight_steps_;
                    chunk_done(*slot, r->length, r->gpu_ms, true);
                }
            } else if (!stepped || !e_.prefill_overlap()) {
                prefill_step();
            }
        }
    }

private:
    using Clock = std::chrono::steady_clock;
    enum class Phase { prefill, decode };
    struct Slot {
        qp::SessionId session = 0;
        bool has_session = false, refused = false;  // refused: unused (admission plans memory per request)
        double admit_free_mib = 0;                  // cudaMemGetInfo free VRAM when the request was admitted
        std::optional<serve::TakenJob> job;
        Phase phase = Phase::prefill;
        std::uint64_t n = 0, prefilled = 0;
        std::vector<std::uint64_t> capture_points;
        double prefix_restore_ms = 0, prefix_capture_ms = 0;
        qc::Stream stream;
        std::unique_ptr<qc::ContextDraft> ctx;  // context copy/n-gram drafter of this request (STRIX_QRTX_CONTEXT_DRAFT)
        serve::GenerationStats st;
        Clock::time_point t0, t_decode;
        double prefill_gpu_ms = 0;   // overlapped chunks: GPU time on the prefill stream (events)
        double prefill_sync_ms = 0;  // chunks between steps / whole blocks: host wall time
        std::uint32_t prefill_chunks = 0, overlapped_chunks = 0, overlap_steps = 0;
        double prefill_wait_ms = 0;  // overlap: host time blocked waiting for a chunk (wait mode, or nothing could decode)
        std::vector<std::uint32_t> emitted;
        std::uint64_t rows = 0, window_sessions = 0, max_window_sessions = 0, window_rows = 0;
        bool first = true;
    };

    // Free/total VRAM. Tiered residency: the engine was sized to a budget, so the server never fills the card beyond it:
    // free = min(card free, budget - what this process holds (engine + extra sessions)), total = the budget.
    std::pair<std::uint64_t, std::uint64_t> vram() const {
        std::size_t free = 0, total = 0;
        if (cudaMemGetInfo(&free, &total) != cudaSuccess) { (void)cudaGetLastError(); return {0, 0}; }
        const unsigned live = live_slots();
        const std::uint64_t used = e_.engine_device_bytes() + (live > 1U ? live - 1U : 0U) * e_.session_bytes();
        const auto view = strix::residency::live_budget_view({total, free}, g_budget_bytes, e_.card_budget_bytes(), used);
        return {view.free, view.total};
    }
    unsigned live_slots() const {
        unsigned n = 0;
        for (const auto& x : slots_) n += x.has_session ? 1U : 0U;
        return n;
    }

    // Requests that never reach a slot (the checks start() makes, before any memory is planned for them).
    bool reject_early(serve::TakenJob& t) {
        serve::GenerationStats st;
        st.prompt_tokens = t.job.prompt.size();
        std::string error;
        if (t.ticket && t.ticket->cancelled()) {
            st.finish = "cancelled";
            queue_.finish(t, st);
            return true;
        }
        if (t.job.prompt.size() < 2) error = "prompt needs at least 2 tokens";
        else if (t.job.prompt.size() >= o_.engine.max_positions) {
            st.error_status = 400;
            st.error_code = "context_length_exceeded";
            error = "prompt longer than the context";
        }
        if (error.empty()) return false;
        queue_.finish(t, st, error);
        return true;
    }

    void refuse(serve::TakenJob& t, const std::string& why) {
        serve::GenerationStats st;
        st.prompt_tokens = t.job.prompt.size();
        st.error_status = 503;
        st.error_code = "insufficient_device_memory";
        queue_.finish(t, st, why);
    }

    // Place `t` by the shared planner (strix/serve/session_reservation.hpp): reuse an idle session that holds it, build
    // one, or re-size one, after releasing idle sessions when needed. false: it waits for a running request to end.
    bool place(serve::TakenJob& t) {
        const auto need = cuda_qwen_serve::request_positions(o_.session, t.job.prompt.size(), t.job.max_tokens);
        std::vector<serve::SlotSession> view(slots_.size());
        for (std::size_t i = 0; i < slots_.size(); ++i) {
            const auto& x = slots_[i];
            view[i] = {x.job.has_value(), x.has_session, i == 0, x.has_session ? e_.capacity(x.session) : 0U,
                       x.has_session ? e_.session_bytes(x.session) : 0U};
        }
        const auto plan = serve::plan_admission(view, need, [&](std::uint64_t p) { return e_.session_bytes_at(p); },
                                                vram().first, std::uint64_t{o_.slot_margin_mib} << 20U);
        if (plan.action == serve::AdmitAction::wait) {
            if (!held_logged_)
                std::cerr << "strix-cuda-qwen-serve: request waits (" << plan.why << "); " << live_slots() << " sessions" << std::endl;
            held_logged_ = true;
            return false;
        }
        held_logged_ = false;
        if (plan.action == serve::AdmitAction::reject) {
            refuse(t, "request does not fit device memory: " + plan.why);
            return true;
        }
        auto& slot = slots_[plan.slot];
        try {
            for (const auto r : plan.release) {
                e_.destroy_session(slots_[r].session);
                slots_[r].has_session = false;
            }
            if (plan.action == serve::AdmitAction::resize) e_.reserve_session(slot.session, need);
            else if (plan.action == serve::AdmitAction::build) {
                slot.session = e_.create_session(need);
                slot.has_session = true;
            }
        } catch (const std::exception& ex) {
            // an allocation the observation allowed failed: a re-sized slot keeps no session (session 0: the smallest)
            std::cerr << "strix-cuda-qwen-serve: session of " << need << " positions failed (" << ex.what() << ")" << std::endl;
            if (plan.action == serve::AdmitAction::resize) {
                if (plan.slot == 0) {
                    try { e_.reserve_session(0, cuda_qwen_serve::initial_positions(o_.session)); } catch (...) {}
                } else {
                    try { e_.destroy_session(slot.session); } catch (...) {}
                    slot.has_session = false;
                }
            }
            bool busy = false;
            for (const auto& x : slots_) busy = busy || x.job.has_value();
            if (busy) return false;
            refuse(t, std::string("session allocation failed: ") + ex.what());
            return true;
        }
        if (plan.action != serve::AdmitAction::reuse) {
            const auto after = vram();
            std::cerr << "strix-cuda-qwen-serve: slot " << plan.slot << " " << serve::admit_action_name(plan.action) << ": "
                      << plan.why << "; " << live_slots() << " sessions, VRAM free " << (after.first >> 20U) << " MiB" << std::endl;
        }
        start(slot, std::move(t));
        return true;
    }

    // A request taken but not placed yet (no memory for its session) heads the line: nothing is taken before it (FIFO).
    void admit() {
        bool woke = false;
        for (;;) {
            bool free_slot = false, busy = false;
            for (const auto& x : slots_) {
                free_slot = free_slot || !x.job;
                busy = busy || x.job.has_value();
            }
            if (!free_slot) return;
            if (!held_) {
                auto taken = queue_.take(!busy);  // block only when nothing runs
                if (!taken && woke && o_.admit_grace_ms > 0) {
                    // requests that arrive together (a client burst) join the same first step instead of waiting behind
                    // the first one's whole-prompt prefill
                    std::this_thread::sleep_for(std::chrono::milliseconds(o_.admit_grace_ms));
                    woke = false;
                    taken = queue_.take(false);
                }
                if (!taken) return;
                woke = woke || !busy;
                if (reject_early(*taken)) continue;
                held_ = std::move(taken);
            } else if (held_->ticket && held_->ticket->cancelled()) {
                serve::GenerationStats st;
                st.prompt_tokens = held_->job.prompt.size();
                st.finish = "cancelled";
                queue_.finish(*held_, st);
                held_.reset();
                held_logged_ = false;
                continue;
            }
            if (!place(*held_)) return;
            held_.reset();
        }
    }
    std::optional<serve::TakenJob> held_;
    bool held_logged_ = false;

    void start(Slot& s, serve::TakenJob taken) {
        s = Slot{s.session, s.has_session, s.refused};
        s.admit_free_mib = static_cast<double>(vram().first) / 1048576.0;
        s.job = std::move(taken);
        auto& job = s.job->job;
        s.t0 = Clock::now();
        s.st.prompt_tokens = job.prompt.size();
        s.n = job.prompt.size();
        const std::uint64_t cap = o_.engine.max_positions;
        std::string error;
        if (s.n < 2) error = "prompt needs at least 2 tokens";
        else if (s.n >= cap) {
            s.st.error_status = 400;
            s.st.error_code = "context_length_exceeded";
            error = "prompt longer than the context";
        }
        if (!error.empty()) {
            queue_.finish(*s.job, s.st, error);
            s.job.reset();
            return;
        }
        s.prefilled = cache_.restore(e_, s.session, job.prompt, s.prefix_restore_ms);
        s.st.cached_tokens = s.prefilled;
        s.capture_points = cache_.plan(s.n, s.prefilled);
        if (job.on_progress) job.on_progress({serve::GenerationPhase::prefill, s.prefilled, 0, s.n});
    }

    bool decoding() const {
        for (const auto& s : slots_) if (s.job && s.phase == Phase::decode) return true;
        return false;
    }
    Slot* oldest_prefill() {
        Slot* pick = nullptr;
        for (auto& s : slots_)
            if (s.job && s.phase == Phase::prefill && (!pick || s.job->started < pick->job->started)) pick = &s;
        return pick;
    }
    // Rows of the next chunk while others decode: --prefill-chunk N, or auto. Auto fits chunk_ms = a + b * rows (the fixed
    // per-chunk cost a matters: a per-row rate measured on small chunks would shrink the chunks without end) by least squares
    // with exponential forgetting over the chunks that ran beside decode steps, and solves a + b * rows = k * step_ms.
    // Sizes: first 256, second 512 (two sizes for the fit), then the solution, a multiple of 16 in
    // [max(256, the first size), max_chunk_].
    std::uint32_t next_chunk_rows(bool overlap) const {
        const std::uint32_t cap = overlap ? max_chunk_ : std::numeric_limits<std::uint32_t>::max();
        return std::min(cap, o_.prefill_chunk ? o_.prefill_chunk : auto_rows_);
    }
    void fit_chunk_time(double rows, double ms) {
        constexpr double keep = 0.8;  // forgetting factor per observation
        fit_[0] = keep * fit_[0] + 1.0;
        fit_[1] = keep * fit_[1] + rows;
        fit_[2] = keep * fit_[2] + rows * rows;
        fit_[3] = keep * fit_[3] + ms;
        fit_[4] = keep * fit_[4] + rows * ms;
        ++fit_n_;
        const std::uint32_t floor_rows = 256U, top = std::max(floor_rows, max_chunk_);
        if (fit_n_ == 1) { auto_rows_ = std::min(512U, top); return; }  // a second size for the fit
        const double det = fit_[0] * fit_[2] - fit_[1] * fit_[1];
        if (!(det > 1e-6 * fit_[0] * fit_[2]) || step_ms_ <= 0) return;
        const double b = (fit_[0] * fit_[4] - fit_[1] * fit_[3]) / det;
        const double a = (fit_[3] - b * fit_[1]) / fit_[0];
        fit_a_ = a;
        fit_b_ = b;
        if (!(b > 0)) return;
        const double want = (o_.prefill_auto_k * step_ms_ - a) / b;
        const double lo = std::max<double>(floor_rows, first_rows_), hi = std::max<double>(lo, top);
        auto_rows_ = static_cast<std::uint32_t>(std::clamp(want, lo, hi)) / 16U * 16U;
    }
    void chunk_done(Slot& s, std::uint64_t length, double ms, bool overlapped) {
        const auto rows = static_cast<double>(length - s.prefilled);
        if (o_.prefill_chunk == 0 && rows > 0 && ms > 0 && decoding()) {  // auto learns from chunks beside decode steps
            if (first_rows_ == 0) first_rows_ = static_cast<std::uint32_t>(rows);
            fit_chunk_time(rows, ms);
        }
        s.prefilled = length;
        if (std::binary_search(s.capture_points.begin(), s.capture_points.end(), length))
            s.prefix_capture_ms += cache_.capture(e_, s.session, s.job->job.prompt, length);
        (overlapped ? s.prefill_gpu_ms : s.prefill_sync_ms) += ms;
        ++s.prefill_chunks;
        s.overlapped_chunks += overlapped ? 1U : 0U;
        auto& job = s.job->job;
        if (s.prefilled < s.n - 1) {
            if (job.on_progress) job.on_progress({serve::GenerationPhase::prefill, s.prefilled, 0, s.n});
            return;
        }
        s.t_decode = Clock::now();
        s.st.prefill_ms = std::chrono::duration<double, std::milli>(s.t_decode - s.t0).count();
        if (job.on_progress) job.on_progress({serve::GenerationPhase::decode, s.n, 0, s.n});
        s.phase = Phase::decode;
        s.ctx.reset();
        // request field "ngram_draft" (shared serve API): false = no context drafter for this request, true = the
        // configured mode (first when the server's is off); unset = the server's mode
        const qc::ContextMode mode = !job.ngram_draft ? context_mode_
            : !*job.ngram_draft ? qc::ContextMode::off : context_mode_ != qc::ContextMode::off ? context_mode_ : qc::ContextMode::first;
        if (mtp_ && mode != qc::ContextMode::off) {
            // the drafter's context is what the session holds; the pending (last prompt) token joins at the first step
            s.ctx = std::make_unique<qc::ContextDraft>(mode);
            s.ctx->drafter.reset(std::span<const std::uint32_t>(job.prompt.data(), s.n - 1));
        }
        s.stream = qc::Stream{s.session, job.prompt[s.n - 1], s.n - 1, job.draft_depth_max, s.ctx.get(), job.copy_rows, job.window_rows};
        s.st.finish = "length";
        if (job.max_tokens == 0) finish(s, {}, "length");
    }

    // One chunk of the oldest prefilling request after the decode step: next_chunk_rows() while others decode, else
    // whole blocks.
    void prefill_step() {
        Slot* pick = oldest_prefill();
        if (!pick) return;
        Slot& s = *pick;
        if (s.job->ticket->cancelled()) { finish(s, {}, "cancelled"); return; }
        const std::uint32_t chunk = decoding() ? next_chunk_rows(false) : std::numeric_limits<std::uint32_t>::max();
        const auto tc = Clock::now();
        std::uint64_t length = 0;
        try {
            length = e_.prefill_chunk(s.session, s.job->job.prompt, s.n - 1, s.prefilled, px::clip_chunk(s.prefilled, chunk, s.capture_points));
        } catch (const std::exception& ex) {
            finish(s, ex.what(), "length");
            return;
        }
        chunk_done(s, length, std::chrono::duration<double, std::milli>(Clock::now() - tc).count(), false);
    }

    void decode_step() {
        std::vector<Slot*> live;
        std::vector<qc::Stream> streams;
        for (auto& s : slots_) {
            if (!s.job || s.phase != Phase::decode) continue;
            live.push_back(&s);
            streams.push_back(s.stream);
        }
        if (live.empty()) return;
        qc::StepOptions opt;
        opt.mtp = mtp_;
        opt.depth = std::min<std::uint32_t>(o_.mtp_depth, qc::kWindowRows - 1U);
        opt.min_p = dh_->on() ? dh_->min_p() : o_.min_p;
        const bool lone = dh_->on() && mtp_ && live.size() == 1 && !inflight_;
        if (lone) opt.depth = dh_->plan(opt.depth);
        opt.policy = qc::depth_policy_of(o_.cohort_depth);
        opt.max_positions = o_.engine.max_positions;
        std::vector<qc::StreamStep> r;
        const auto ts = Clock::now();
        try {
            r = qc::step(e_, streams, opt);
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - ts).count();
            step_ms_ = step_ms_ > 0 ? 0.5 * step_ms_ + 0.5 * ms : ms;
        } catch (const std::exception& ex) {
            e_.abort_cohort();
            for (auto* s : live) finish(*s, std::string("engine failure: ") + ex.what(), "length");
            return;
        }
        const std::uint64_t cap = o_.engine.max_positions;
        for (std::size_t i = 0; i < live.size(); ++i) {
            Slot& s = *live[i];
            auto& job = s.job->job;
            const auto& step = r[i];
            s.stream = streams[i];
            if (step.failed) {  // only this request: its window was undone and the other sessions re-ran alone
                std::cerr << "strix-cuda-qwen-serve: request failed in decode: " << step.error << std::endl;
                finish(s, step.error, "length");
                continue;
            }
            // Context windows do not measure predictor acceptance or its whole-step cost.
            if (lone && step.drafted > 0 && step.context_proposed == 0) dh_->finish_fixed(opt.depth, step.mtp_accepted, std::chrono::duration<double, std::milli>(Clock::now() - ts).count(),
                                        step.drafted == opt.depth && step.mtp_proposed == opt.depth && step.rows == opt.depth + 1U);
            ++s.st.cycles;
            s.st.proposed += step.mtp_proposed;
            s.st.accepted += step.mtp_accepted;
            s.st.ngram_rows += step.context_proposed;
            s.st.ngram_accepted += step.context_accepted;
            s.rows += step.rows;
            s.window_sessions += step.window_sessions;
            s.max_window_sessions = std::max<std::uint64_t>(s.max_window_sessions, step.window_sessions);
            s.window_rows += step.window_rows;
            std::vector<std::uint32_t> out;
            bool stopped = false;
            for (const auto t : step.tokens) {
                if (s.st.generated_tokens + out.size() >= job.max_tokens) break;
                out.push_back(t);
                if (std::find(job.stop_tokens.begin(), job.stop_tokens.end(), t) != job.stop_tokens.end()) { stopped = true; break; }
            }
            if (s.first && job.observed)
                s.st.first_token_ms = std::chrono::duration<double, std::milli>(Clock::now() - s.job->submitted).count();
            s.first = false;
            s.st.generated_tokens += out.size();
            s.emitted.insert(s.emitted.end(), out.begin(), out.end());
            if (job.on_tokens && !job.on_tokens(out)) { finish(s, {}, "cancelled"); continue; }
            if (stopped) { finish(s, {}, "stop"); continue; }
            if (s.st.generated_tokens >= job.max_tokens) { finish(s, {}, "length"); continue; }
            if (s.stream.position + 1 >= cap) { finish(s, {}, "context"); continue; }
        }
    }

    void finish(Slot& s, const std::string& error, const char* why) {
        auto& st = s.st;
        st.finish = why;
        if (s.phase == Phase::decode)
            st.decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - s.t_decode).count();
        const double queue_ms = std::chrono::duration<double, std::milli>(s.job->started - s.job->submitted).count();
        const double steps = st.cycles ? static_cast<double>(st.cycles) : 1.0;
        char head[1280];
        std::snprintf(head, sizeof(head),
            "{\"engine\":\"cuda-qwen-a8\",\"mtp_depth\":%u,\"tokens_per_cycle\":%.3f,\"prefill_ms\":%.2f,\"decode_ms\":%.2f,"
            "\"queue_ms\":%.2f,\"http_slots\":%u,\"prefill_chunks\":%u,\"overlapped_chunks\":%u,\"overlap_steps\":%u,\"prefill_wait_ms\":%.2f,\"prefill_overlap_gpu_ms\":%.2f,\"prefill_sync_ms\":%.2f,\"steps\":%llu,"
            "\"rows_per_step\":%.3f,\"cohort_sessions\":%.3f,\"cohort_sessions_max\":%llu,\"window_rows\":%.3f,\"prefill_auto_rows\":%u,\"prefill_fit_a_ms\":%.3f,\"prefill_fit_b_ms_per_row\":%.5f,"
            "\"max_positions\":%u,\"session_positions\":%llu,\"session_mib\":%.1f,\"admit_free_vram_mib\":%.1f,\"live_slots\":%u,\"ids\":[",
            mtp_ ? o_.mtp_depth : 0U, st.cycles ? static_cast<double>(st.generated_tokens) / static_cast<double>(st.cycles) : 0.0,
            st.prefill_ms, st.decode_ms, queue_ms, o_.http_slots, s.prefill_chunks, s.overlapped_chunks, s.overlap_steps, s.prefill_wait_ms, s.prefill_gpu_ms, s.prefill_sync_ms,
            static_cast<unsigned long long>(st.cycles), static_cast<double>(s.rows) / steps, static_cast<double>(s.window_sessions) / steps,
            static_cast<unsigned long long>(s.max_window_sessions), static_cast<double>(s.window_rows) / steps, o_.prefill_chunk ? o_.prefill_chunk : auto_rows_, fit_a_, fit_b_,
            o_.engine.max_positions, static_cast<unsigned long long>(e_.capacity(s.session)),
            static_cast<double>(e_.session_bytes(s.session)) / 1048576.0, s.admit_free_mib, live_slots());
        std::string extra = head;
        for (std::size_t i = 0; i < s.emitted.size() && i < 4096; ++i) extra += (i ? "," : "") + std::to_string(s.emitted[i]);
        extra += "]";
        if (s.ctx) extra += std::string(",\"context_draft\":\"") + qc::context_mode_name(s.ctx->mode) + "\",\"copy_first_steps\":" +
            std::to_string(s.ctx->copy_first_steps) + ",\"context_extended_steps\":" + std::to_string(s.ctx->extended_steps) +
            ",\"context_rows\":" + std::to_string(s.ctx->context_rows) + ",\"copy_first_accepted\":" + std::to_string(s.ctx->context_accepted) +
            ",\"context_host_us\":" + std::to_string(static_cast<long long>(s.ctx->host_us)) +
            ",\"copy_wide_steps\":" + std::to_string(s.ctx->wide_steps) + ",\"copy_wide_rows\":" + std::to_string(s.ctx->wide_rows) +
            ",\"copy_wide_accepted\":" + std::to_string(s.ctx->wide_accepted);
        extra += ",\"prefix_cached\":" + std::to_string(st.cached_tokens) + ",\"prefix_restore_ms\":" + std::to_string(s.prefix_restore_ms)
            + ",\"prefix_capture_ms\":" + std::to_string(s.prefix_capture_ms) + "}";
        st.engine_extra.emplace_back("rtx", extra);
        queue_.finish(*s.job, std::move(st), error);
        s.job.reset();
        s.phase = Phase::prefill;
    }

    qp::Prefill& e_;
    const Options& o_;
    serve::GenerationQueue& queue_;
    px::Cache& cache_;
    std::vector<Slot> slots_;
    bool mtp_ = false;
    const qc::ContextMode context_mode_ = qc::context_mode_default();
    std::unique_ptr<strix::MtpDepthHost> dh_;  // measured-cost lone-stream depth (STRIX_MTP_CONTROLLER=depth)
    std::uint32_t max_chunk_ = 1024, auto_rows_ = 256, first_rows_ = 0, fit_n_ = 0, inflight_steps_ = 0;
    Slot* inflight_ = nullptr;  // overlap: the slot whose chunk is on the prefill stream
    double step_ms_ = 0, fit_[5] = {0, 0, 0, 0, 0}, fit_a_ = 0, fit_b_ = 0;
};

}  // namespace

int main(int argc, char** argv) {
    // --help answers before any device or file is touched (post-install smoke: the binary and its DLLs load).
    {
        if (strix::device::wants_help(argc, argv)) {
            std::cout << "usage: strix-cuda-qwen-serve --model FIRST_PART.gguf --tables DIR --rope-table FILE\n"
                         "         [--mtp-sidecar FILE] [--draft-vocab FILE] [--bind ADDR] [--port N]\n"
                         "         [--api-key-file FILE] [--prefix-cache on|off] [--devices FILE | --device-id ID]\n"
                         "Serves Qwen3.8-Flash-Next from one CUDA GPU (OpenAI and Anthropic APIs, /health).\n"
                         "Every option and default: docs/models/qwen3.8/rtx-serve.md\n";
            return 0;
        }
    }
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared card, restricts the process to the chosen one and prints the "device:" receipt line.
    try {
        const auto device_cli = strix::device::take_command_line(argc, argv);
        auto o = parse(argc, argv);
        strix::device::Request device_request;
        device_request.select = device_cli.select;
        const auto& bound = strix::device::bind_cuda(std::move(device_request), device_cli.devices_path);
        // ONE budget resolver (strix/device/budget.hpp), as every route: the bound profile caps the admission view
        // (budget_view) and, through qp::plan_residency (the same route_budget), the residency plan, so the engine,
        // --print-effective-config (vram_cap_bytes / vram_cap_source) and admission plan to one cap; a tighter
        // --vram-budget-gib still wins. This route has a host / file expert tier, so the residency plan, not a
        // weights-only check, decides whether the model fits.
        const auto budget = strix::device::route_budget(bound, {});
        g_budget_bytes = budget.device_bytes;
        if (budget.device_capped()) std::cerr << "strix-cuda-qwen-serve: memory budget " << budget.json() << std::endl;
        if (o.print_effective_config) {
            // No model load: the residency decision from the GGUF headers and the live card/host numbers.
            const auto plan = qp::plan_residency(o.engine);
            std::cout << o.effective_config.substr(0, o.effective_config.size() - 1) << ",\"model\":\"" << o.engine.model_gguf << "\",\"max_positions_requested\":" << o.engine.max_positions
                      << ",\"residency\":" << plan.json << "}" << std::endl;
            return plan.decision.ok ? 0 : 3;
        }
        std::cerr << "strix-cuda-qwen-serve: loading " << o.engine.model_gguf << std::endl;
        if (!o.engine.mtp_sidecar_gguf.empty() && o.mtp_depth > 0) std::cerr << "strix-cuda-qwen-serve: " << strix::mtp_depth_state_line(o.min_p) << ", mtp_depth=" << o.mtp_depth
            << ", context_draft=" << qc::context_mode_name(qc::context_mode_default())
            << ", copy_rows=" << qc::copy_rows_default() << ", copy_wide=" << (qc::copy_wide_switch() < 0 ? "auto" : qc::copy_wide_switch() ? "on" : "off")
            << " (effective " << ((qc::copy_wide_switch() < 0 ? qc::copy_wide_pays(qc::copy_rows_default(), qc::copy_wide_model()) : qc::copy_wide_switch() == 1) ? "on" : "off")
            << ": wide " << qc::copy_wide_model().wide_ms << " ms vs 8-row " << qc::copy_wide_model().narrow_a_ms + 8 * qc::copy_wide_model().narrow_b_ms
            << " ms), copy_wide_accept=" << qc::copy_wide_accept_default()
            << ", context_draft_effective=" << qc::context_mode_name(o.http_slots > 1 ? qc::context_mode_default() : qc::ContextMode::off)
            << ", context_draft_applicable=" << (o.http_slots > 1 ? "cohort" : "no (single-slot loop)") << std::endl;
        qp::Prefill engine(o.engine);
        o.engine.max_positions = engine.max_positions();  // resolved limit for admission, cache and capabilities
        std::cerr << "strix-cuda-qwen-serve: loaded " << engine.describe() << std::endl;
        if (budget.device_capped()) {
            // Measured after the load: what is on the card + one session + margin must fit the budget (the shared plan).
            std::size_t free = 0, total = 0;
            if (cudaMemGetInfo(&free, &total) != cudaSuccess) throw std::runtime_error("cudaMemGetInfo failed");
            strix::deploy::RouteNeeds needs;
            needs.route = "qwen-serve";
            needs.session_fixed = engine.session_bytes();
            // weights + context, as measured (session 0 is already allocated and is counted as a slot below)
            needs.weights_resident = total - free > needs.session_fixed ? total - free - needs.session_fixed : 0;
            needs.slots = o.http_slots;
            needs.margin_bytes = std::uint64_t{o.slot_margin_mib} << 20U;
            auto view = budget;
            view.physical_device_bytes = total;
            const auto plan = strix::deploy::plan_memory(view, needs);
            std::cerr << "strix-cuda-qwen-serve: memory plan " << plan.json() << std::endl;
            if (plan.refused) {
                std::cerr << "strix-cuda-qwen-serve: refusing to start: " << plan.refusal << std::endl;
                return 2;
            }
        }
        const auto tokenizer = serve::BpeTokenizer::load_gguf(o.engine.model_gguf);
        serve::ApiConfig config;
        config.model_id = o.model_id;
        config.http_slots = o.http_slots;
        config.api_key = read_key(o.api_key_file);
        config.engine_description = o.http_slots > 1
            ? "cuda-qwen-a8 (RTX, node-exact dense A8 decode, MTP verify windows, multi-session cohort windows)"
            : "cuda-qwen-a8 (RTX, node-exact dense A8 decode, MTP verify windows)";
        config.engine_progress = true;
        config.stop_tokens.push_back(tokenizer->eos_token());
        if (const auto end = tokenizer->find("<|endoftext|>"); end != UINT32_MAX) config.stop_tokens.push_back(end);
        std::sort(config.stop_tokens.begin(), config.stop_tokens.end());
        config.stop_tokens.erase(std::unique(config.stop_tokens.begin(), config.stop_tokens.end()), config.stop_tokens.end());
        serve::GenerationQueue queue(o.max_queue);
        px::Cache prefix_cache(o.engine, o.prefix);
        config.state_cache_stats = [&prefix_cache] { return prefix_cache.stats_json(); };
        // /api/status names the bound device (id, UUID, PCI, attach) next to the prefix cache counters.
        const std::string device_json = strix::device::receipt_json(strix::device::bind_cuda());
        config.status_extra = [&prefix_cache, device_json, &engine](serve::Json& status) {
            status.set("expert_residency", serve::parse_json(engine.residency_json()));
            status.set("expert_residency_receipt", serve::parse_json(engine.residency_receipt_json()));
            status.set("prefix_cache", serve::parse_json(prefix_cache.stats_json()));
            status.set("device", serve::parse_json(device_json));
        };
        CudaQwenEngine adapter(engine, o, prefix_cache);
        std::optional<CohortServer> cohort;  // --http-slots > 1: continuous batching; 1: the serialized queue.run path
        if (o.http_slots > 1) {
            cohort.emplace(engine, o, queue, prefix_cache);
            std::cerr << "strix-cuda-qwen-serve: " << o.http_slots << " slots (cohort decode), sessions "
                      << cuda_qwen_serve::describe(o.session) << ", prefill chunk "
                      << (o.prefill_chunk ? std::to_string(o.prefill_chunk) + " rows" : std::string("auto"))
                      << (o.prefill_overlap ? ", overlapped on a prefill stream (max " + std::to_string(o.prefill_overlap_rows) + " rows, " +
                                                  (o.prefill_poll ? "polled: decode steps run until it completes)" : "one step then wait)")
                                            : std::string(", between steps"))
                      << ", draft depth "
                      << o.cohort_depth << std::endl;
        }
        serve::ApiService api(config, tokenizer, queue, o.engine.max_positions, false);
        serve::HttpServerConfig http;
        http.bind_address = o.bind;
        http.port = static_cast<std::uint16_t>(o.port);
        serve::HttpServer server(http, [&api](const serve::HttpRequest& request, serve::HttpResponseWriter& response) {
            api.handle(request, response);
        });
        server.listen();
        api.set_state("ready");
        std::cerr << "strix-cuda-qwen-serve: listening on " << o.bind << ':' << server.port() << " model " << config.model_id << std::endl;
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
        std::atomic<bool> finished{false};
        std::thread http_thread([&server] { server.serve(); });
        std::thread watcher([&] {
            while (!finished.load() && !g_stop.load()) std::this_thread::sleep_for(std::chrono::milliseconds(100));
            server.stop();
            queue.shutdown();
        });
        if (cohort) cohort->run();
        else queue.run(adapter);
        finished.store(true);
        watcher.join();
        http_thread.join();
        std::cerr << "strix-cuda-qwen-serve: stopped" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "strix-cuda-qwen-serve: " << e.what() << std::endl;
        return 1;
    }
}
