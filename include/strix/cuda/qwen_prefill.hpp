// Exactness comments below describe the default FP32 route target. The explicit
// BF16 route-storage candidate is a separate, unqualified numerical target.
// CUDA prefill of Qwen3.8-Flash-Next, bit-identical to the HIP tiled prefill under the single A8 profile
// (hc-splitk-f32+q51-q80-a8-down+q5k-a8-gateup+dense-a8-v1) of a Strix Halo node, for hand-off to that node.
//
// The prefill computes prompt rows [0, p) and leaves the node's decode state for position p: per row component
// (12 attention KV slices, 12 raw index slices, MTP KV and raw index with shift 1) and the fixed blob (GDN recurrent
// matrices, GDN conv histories, PLE conv history, n-gram history, position word, held wide row h_{p-1}). Each piece is
// handed to the caller (`Sink`) on the host as soon as it is final, so it can be streamed while later rows compute.
// The node then runs row p itself (first token, MTP adoption). Under this profile the bits do not depend on the row
// blocking (verified on the node: T = 1024 and T = 256 give identical state), so the blocks here are free.
#pragma once
#include "strix/weight/admission.hpp"

#include "strix/residency/expert_residency.hpp"
#include "strix/qwen_prefill_trace.hpp"
#include "strix/qwen_prefill_schedule.hpp"
#include "strix/cuda/qwen_route_bf16.hpp"
#include "strix/cuda/qwen_mtp_diagnostic.hpp"
#include "strix/qwen_weight_packs.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace strix::cuda::qwen_prefill {

struct Config {
    std::string model_gguf;            // production UD-Q4_K_XL merged GGUF
    std::string mtp_sidecar_gguf;      // MTP sidecar (distillG/D Q8_0); empty = no MTP rows (non-MTP node route)
    std::string tables_dir;            // gfx1151 math tables (amd_math_tables.hpp)
    std::string rope_table;            // cs[position][32] float2 from the node (rope_cs_262144.bin)
    std::uint32_t max_positions = 32768;   // context cap: the largest session; rope/QSA workspaces are sized to it
    // Positions of session 0 and of create_session() without an argument (0: max_positions). A server sets it small
    // and sizes each request's session with reserve_session() (strix/serve/session_reservation.hpp).
    std::uint32_t session_positions = 0;
    std::vector<std::uint32_t> block_schedule{1024, 2048, 4096, 8192};  // rows per block; the last entry repeats
    bool tail_large = true;           // rotate final remainder to front; same count and capacity (--block-schedule default = off)
    std::uint32_t final_block_rows = 0; // optional tail reservation within loaded max block
    // Fit the block capacity to the VRAM left after weights, decode buffers and session 0 (strix/rows_fit.hpp): the block
    // workspace is allocated last and steps down by 1024 rows until it fits with block_fit_reserve_mib free; the schedule is
    // clamped to the fitted capacity. Bitwise (block rows never change the per-row arithmetic). Off: allocate the schedule
    // maximum before the decode buffers and refuse when anything later does not fit (the pre-fit behaviour).
    bool block_fit = false;
    std::uint32_t block_fit_reserve_mib = 512;
    int device = 0;
    bool ple_host_pinned = true;       // per_layer_token_embd (28.8 GB) in pinned host memory, gathered on the host
    // Release per layer: every block's KV/index rows of layer l, and in the last block each GDN layer's recurrent/conv
    // slice (final once that layer ran over the last rows), are copied on a second stream and handed to the sink while
    // the next layers compute, from a host thread (the sink must then be thread-safe; pieces of one component still
    // arrive in ascending order). Same bytes; only when each piece is released changes. false: after each block.
    // Default ON (gate 2026-10-24 05:33: all state bit-identical; net3: faster at every length 1K-32K).
    bool release_by_layer = true;
    // Decode: also upload the LM head (output.weight, 675 MB) and the final hyper mix, and enable step().
    bool decode = false;
    // Decode with a sidecar: the MTP draft head covers these token ids (little-endian uint32 file, e.g.
    // data/draft-vocab/core64k.bin); empty = the whole vocabulary. Drafts never change committed tokens.
    std::string draft_vocab;
    // Diagnostic: append every block's router selections (all rows, all layers) to this EXTR01 file
    // (scripts/bench/residency/residency_sim.py). Forces eager windows (no graphs); empty = off, no change.
    std::string expert_trace;
    // Expert residency (docs/engine/memory/expert-residency.md): automatic = fully resident when the model fits the card
    // budget (today's route, nothing new happens), else tiered (hot experts on the card, the rest computed on the host,
    // bitwise equal). The caller applies strix::residency::with_environment before constructing the engine.
    strix::residency::Options residency;
    // false: run() keeps the state on the device only (no D2H staging, no pieces): a decode session that never hands off.
    bool export_state = true;
    // Explicit unqualified CUDA A8 route-storage candidate. Omitted: environment;
    // false: individual OFF. Baseline/defaults/exact masters always suppress it.
    std::optional<bool> moe_route_bf16;
    // Frozen by the common resolver; direct API callers resolve once at construction.
    std::optional<qwen_route::Policy> route_policy;
    // Explicit source diagnostic only: does not admit ordinary product MTP.
    bool diagnostic_bf16_mtp = false;
    // TURBO onfly weights (--profile turbo --turbo-weights onfly, resolved by strix/cuda/qwen_turbo.hpp): convert the
    // dense Q8_0 roles to NVFP4 once at load (onfly-nvfp4-rtn-g16, nvfp4-a8-v1) and read them in decode windows.
    // false (default, exact, FAST): nothing converted, every route and byte unchanged.
    bool turbo_onfly_nvfp4 = false;
    // TURBO cache namespace (strix::profile::turbo::cache_namespace); empty outside turbo. Part of the prefix identity.
    std::string turbo_namespace;
    // Prefill arithmetic forms of the FAST profile (CUDA numerical identities, include/strix/profiles/qwen_cuda.hpp).
    // All false = the exact CUDA path. Set only by the profile resolver (or its named overrides), never by getenv here.
    struct Forms {
        bool hc_mma = false;   // hyper-connection projection / output on FP16 tensor cores, FP32 accumulation
        bool mtp_mma = false;  // MTP fused projection on FP16 tensor cores, FP32 accumulation
        bool qsa_mma = false;  // QSA prefill attention (scores and output) on FP16 tensor cores, FP32 accumulation
        bool moe_mma = false;  // routed-MoE prefill project/down: int8 IMMA block dots, ONE FP32 FMA chain per output (not the HIP four)
    } forms;
    qwen_packs::ExpertArithmetic expert_arithmetic=qwen_packs::ExpertArithmetic::incumbent;
};

// Diagnostic snapshot, not the node wire or a portable session/cache format.
// Same engine instance and target only. Default snapshots invalidate derived pools;
// the armed MTP diagnostic preserves and checks their actual retained prefixes.
struct CanonicalState {
    std::uint64_t target = 0, engine_binding = 0, length = 0;
    std::vector<std::uint8_t> rows, fixed, held;
    qwen_mtp_diagnostic::Binding mtp_binding;
    std::vector<std::uint8_t> mtp_rows; // committed pairs [0,length-1), never provisional tail
    std::vector<std::vector<float>> pools; // actual retained prefixes, before any repair
    std::vector<std::uint32_t> pool_valid;
};

// One finished piece of the node state (host memory owned by the prefill until the next run() starts).
struct Piece {
    bool fixed = false;
    std::string component;             // node component name: kv.L3, index.L3, mtp.kv, mtp.index | recurrent_matrix, ...
    std::uint64_t first = 0;           // first row (row component) or byte offset inside the component (fixed)
    std::uint64_t count = 0;           // rows (row component), 0 for fixed
    const std::uint8_t* data = nullptr;
    std::uint64_t bytes = 0;
    std::uint64_t trace_id = 0;
};
struct RunOptions {
    // Omitted fields inherit configured defaults; overrides do not resize workspaces.
    std::optional<bool> tail_large;
    std::optional<std::uint32_t> final_block_rows;
    qwen_prefill_trace::Trace* trace = nullptr;
};
using Sink = std::function<void(const Piece&)>;

struct Timing {
    double total_s = 0, first_block_s = 0;
    std::vector<double> block_s;       // per block, device time
    std::uint64_t rows = 0;
    // Forms dispatched by this run, reported only after successful device synchronization.
    Config::Forms executed_forms;
    bool executed_moe_route_bf16 = false;
};

// One decode/verify step: rows consecutive tokens at positions [position, position + rows) continuing the session
// state left by run() (prompt rows [0, position)) or by earlier steps. Every row's bits are those of the node A8 decode
// of that row: the families never depend on the row count, so a window of T rows equals T single-row steps.
struct Draft {
    std::vector<std::uint32_t> tokens;
    std::vector<float> probability;
};
struct StepResult {
    std::vector<std::uint32_t> argmax;  // per row: first maximum of the FP32 logits (lowest index wins)
    std::vector<float> logits;          // [rows][248320] when requested
    // verify_cohort: false = this entry's rows cannot be trusted (a non-finite row, or a routed-MoE failure word in the
    // window, which affects every entry of it); commit it with keep 0 (undo) or drop the session. verify() throws.
    bool ok = true;
    std::string error;
    std::uint32_t moe_status = 0;       // MoE row status bits of this entry's rows (diagnostics)
};
struct PrefillChunk {
    std::uint64_t length = 0;           // the session length after the chunk
    double gpu_ms = 0;                  // the chunk's time on its stream (events)
};
struct GraphCounters {
    std::size_t cached = 0, cap = 0;
    std::uint64_t captures = 0, replays = 0, evictions = 0, eager_windows = 0;
};

// Multi-session decode. A session is the per-request state: QSA KV / raw index / pooled keys per layer, the GDN recurrent
// double buffer and conv histories, PLE conv + n-gram word, the MTP predictor cache and held row, the length and the
// window bookkeeping. Weights and the activation scratch are shared. Session 0 is the implicit session of the
// single-session API (run/step/verify/commit/draft without an id), created with the engine.
using SessionId = std::uint32_t;
struct RowSegment {
    std::uint64_t first = 0, count = 0;
    const void* data = nullptr;
};
// One session's draft chain request (draft_cohort).
struct DraftRequest {
    SessionId session = 0;
    std::uint32_t pending = 0;
    std::uint32_t depth = 0;
    float min_probability = 0.7F;
};
// One session's rows of a cohort window: `rows` consecutive tokens at positions [position, position + rows).
struct CohortRows {
    SessionId session = 0;
    const std::uint32_t* tokens = nullptr;
    std::uint32_t rows = 0;
    std::uint64_t position = 0;
};

// The residency decision of a configuration, resolved from the GGUF headers and the live card/host numbers without
// loading the model (the same function the engine calls before its upload).
struct ResidencyPlan {
    strix::residency::ModelBytes model;
    strix::residency::CardBytes card;
    strix::residency::SessionBytes session;
    strix::residency::HostBytes host;
    strix::residency::Decision decision;
    strix::residency::Options options; // what the decision used: config.residency + the bound device profile's cap
    std::uint32_t block_rows = 0;    // rows per prefill block in the decided mode (tiered: the exchange's rows)
    std::string json;                 // decision_json + the numbers it used
};
[[nodiscard]] ResidencyPlan plan_residency(const Config& config);

class Prefill {
public:
    explicit Prefill(const Config& config);
    ~Prefill();
    const weight::WeightMap& weight_map() const noexcept;
    const std::vector<weight::LegacyConversion>& weight_conversions() const noexcept;
    Prefill(const Prefill&) = delete;
    Prefill& operator=(const Prefill&) = delete;

    // Computes rows [0, position) of `tokens` (position <= tokens.size()). Pieces go to `sink` in block order; the
    // fixed pieces come last. Returns after every piece was handed over.
    Timing run(const std::vector<std::uint32_t>& tokens, std::uint64_t position, const Sink& sink, const RunOptions& options = {});

    // Decode step (Config::decode): computes rows [position, position + rows) of the session, 1 <= rows <= 8,
    // and their next-token logits. position must equal the session length (run()'s position plus the
    // rows of earlier steps); MTP cache rows are not advanced by step().
    StepResult step(const std::uint32_t* tokens, std::uint32_t rows, std::uint64_t position, bool want_logits = false);
    // Verify window: as step() but leaves the window open; commit(keep) then keeps rows [position, position + keep)
    // (1 <= keep <= rows) and rolls the GDN recurrent/conv, PLE conv and n-gram state back to that row. The session
    // length becomes position + keep. step() = verify() + commit(rows).
    StepResult verify(const std::uint32_t* tokens, std::uint32_t rows, std::uint64_t position, bool want_logits = false);
    void commit(std::uint32_t keep);
    // MTP draft chain (Config::decode + sidecar) after the committed rows: from (h_{L-1}, pending) at position L-1,
    // up to `depth` (<= 7) drafts for positions L+1 ..; stops after the first draft whose probability is below
    // min_probability. Draft arithmetic is free (it only picks what the exact verify checks).
    Draft draft(std::uint32_t pending, std::uint32_t depth, float min_probability);

    // ---- multi-session decode (Config::decode)
    // A new session (fresh state, length 0): allocates session_bytes_at(P) = ~P * 31.6 KB + 0.23 GB of device memory
    // (P = Config::session_positions, else max_positions; or `positions`).
    SessionId create_session();
    SessionId create_session(std::uint64_t positions);
    void destroy_session(SessionId session);        // not session 0; drops the graphs that captured it
    // New position buffers of 1..max_positions for an idle session (session 0 too): its state is discarded (length 0)
    // and the graphs that captured it are dropped. Every write past capacity() is refused (run, chunks, verify, drafts).
    void reserve_session(SessionId session, std::uint64_t positions);
    [[nodiscard]] std::uint64_t capacity(SessionId session) const;
    [[nodiscard]] std::uint64_t session_bytes() const;  // device bytes of session 0's state
    [[nodiscard]] std::uint64_t session_bytes(SessionId session) const;
    [[nodiscard]] std::uint64_t session_bytes_at(std::uint64_t positions) const;  // exact bytes a session of P allocates
    [[nodiscard]] std::uint64_t length(SessionId session) const;
    // Component-major rows: 12 KV arrays (count * 1024 BF16 each), 12 raw-index arrays
    // (count * 128 BF16 each), then MTP KV and raw index when a sidecar is loaded.
    // At an intermediate prefill boundary P < position, all MTP pairs [0,P) are complete.
    // At the final prefill boundary, the last MTP pair is not yet written.
    [[nodiscard]] std::uint64_t row_bytes() const;
    // Fixed layout: current GDN matrices (FP32), GDN conv (BF16), PLE conv (BF16),
    // ngram[2] (uint32). No padding, pooled keys, held row or decode-window scratch.
    [[nodiscard]] std::uint64_t fixed_bytes() const;
    // Synchronous copies on a dedicated stream; no open window or chunk in this session.
    // Destinations and import sources may be pinned host or device pointers (UVA).
    void export_rows(SessionId session, std::uint64_t first, std::uint64_t count, void* dst);
    void export_fixed(SessionId session, void* dst);
    // One producer fence and one completion fence for all requested destinations.
    // Segments may omit already cached rows; fixed_dst may be null.
    void export_segments(SessionId session, const std::vector<RowSegment>& segments, void* fixed_dst);
    // Segments must cover [0,length) in order. Resume prefill with at least one row;
    // the caller must ensure the MTP boundary pair belongs to the resumed prompt.
    void import_state(SessionId session, std::uint64_t length, const std::vector<RowSegment>& segments, const void* fixed);
    // run() into a session (the same computation; the implicit-session run() is run(0, ...)).
    Timing run(SessionId session, const std::vector<std::uint32_t>& tokens, std::uint64_t position, const Sink& sink,
               const RunOptions& options = {});
    // Prefill of rows [from, from + n) of a decode session (Config::export_state == false), n = min(max_rows, position -
    // from, block capacity): one block, so a scheduler can interleave decode windows between the chunks. from == 0
    // starts a fresh state; otherwise from must equal the session length. After the chunk reaching `position` the
    // session is as after run(tokens, position) (same bits: the profile does not depend on the row blocking).
    // Returns the new length.
    std::uint64_t prefill_chunk(SessionId session, const std::vector<std::uint32_t>& tokens, std::uint64_t position,
                                std::uint64_t from, std::uint32_t max_rows);
    // Prefill overlap (decode-first serving): a second set of block buffers and its own stream, so a prefill chunk of a
    // session that is in no window computes concurrently with the decode windows of the others (nothing is shared:
    // exactness holds). max_rows: the largest overlapped chunk. prefill_chunk_async enqueues one chunk (as
    // prefill_chunk, without waiting); prefill_wait() waits for it, sets the session length and returns the chunk's GPU
    // time. One chunk in flight at a time; the session can do nothing else until prefill_wait().
    void enable_prefill_overlap(std::uint32_t max_rows);
    [[nodiscard]] bool prefill_overlap() const;
    [[nodiscard]] std::uint32_t prefill_overlap_rows() const;
    void prefill_chunk_async(SessionId session, const std::vector<std::uint32_t>& tokens, std::uint64_t position,
                             std::uint64_t from, std::uint32_t max_rows);
    PrefillChunk prefill_wait();
    // Non-blocking completion check of the chunk in flight (cudaEventQuery): nullopt while it runs (the session stays
    // excluded; decode windows of the other sessions may keep running), else finalized as prefill_wait().
    std::optional<PrefillChunk> prefill_poll();
    [[nodiscard]] bool prefill_pending() const;
    // Cohort verify window: the sessions' rows concatenated session-major (total 1..16 rows, <= 8 distinct sessions;
    // windows of 9..16 rows, also one session's copy window, run the row-independent families prefill-shaped). The
    // row-independent families (dense A8, HC, router, shared expert, routed MoE, LM head, argmax) run once over all rows;
    // GDN cores, QSA attention, PLE tails and conv histories run per session on its own state and positions. Every row's
    // bits equal the same session's verify() alone. Results are per cohort entry. commit_cohort(keep[i]) (1 <= keep[i] <=
    // rows[i], same order) must follow before any other run/verify/draft; each session rolls back as commit() does.
    // keep[i] == 0 undoes the window for that session exactly (its state is as before verify_cohort). Each entry's
    // position must equal its session's length (> 0: prefilled). A failure inside verify_cohort/commit_cohort leaves the
    // affected sessions at length 0 (a new prefill is needed); per-row failures come back as StepResult::ok == false.
    std::vector<StepResult> verify_cohort(const std::vector<CohortRows>& cohort, bool want_logits = false);
    void commit_cohort(const std::vector<std::uint32_t>& keep);
    // Drops an open cohort window after a failure: its sessions are left needing a new prefill (length 0).
    void abort_cohort() noexcept;
    Draft draft(SessionId session, std::uint32_t pending, std::uint32_t depth, float min_probability);
    // The draft chains of several sessions batched by step: step k of every chain still running (depth > k) is one
    // multi-row MTP forward (HC mixes, q/k/v/o, router, shared expert, draft head once over the rows); the MTP cache rows,
    // draft attention (each session's MTP KV/index/pooled cache and position), routed Q8 experts and the selection run per
    // session. Every chain equals draft(session, ...) bit for bit (tokens and probabilities). Results in request order.
    std::vector<Draft> draft_cohort(const std::vector<DraftRequest>& requests);
    // CUDA graphs of the verify windows (default: on unless STRIX_QRTX_GRAPHS=0). Graphs are keyed by the cohort layout
    // (session, rows, GDN buffer parity per entry), one eager warm-up per key first, LRU-capped (STRIX_QRTX_GRAPH_CACHE).
    void set_graphs(bool on);
    // Descriptor-table graphs (default on unless STRIX_QRTX_GRAPH_TABLE=0): keyed by rows per entry only; the sessions'
    // GDN/conv/PLE buffers and GDN parity are read from a per-segment device table written before each replay.
    void set_graph_table(bool on);
    // Multi-segment cores (default on unless STRIX_QRTX_MULTISEG=0): in a window of >= 2 sessions every GDN core step and QSA
    // core step is one launch across the sessions (each block one session's head / row set, the same per-row arithmetic),
    // instead of one launch per session.
    void set_multiseg(bool on);
    void set_graph_cap(std::size_t cap);            // LRU cap (>= 1); evicts down to it now
    [[nodiscard]] std::string graph_stats() const;
    [[nodiscard]] GraphCounters graph_counters() const;
    [[nodiscard]] std::uint32_t max_positions() const;

    // Diagnostics for the bitwise gates: the wide residual rows after layer 47 of the last run (BF16 [rows][10240]).
    [[nodiscard]] std::string describe() const;
    // Residency: the decision JSON (mode, reason, pool, slots, positions) and, tiered only, the receipt JSON (hit rate,
    // promotions, host time); the receipt of a fully resident engine is {"expert_residency":"full"}.
    [[nodiscard]] bool tiered() const;
    [[nodiscard]] std::string residency_json() const;
    [[nodiscard]] std::string residency_receipt_json() const;
    // Resolved card budget (tiered or an explicit cap, including full mode; 0 = full mode without an explicit cap) and the
    // device bytes it holds once loaded (measured): a server admits sessions against budget - engine bytes.
    [[nodiscard]] std::uint64_t card_budget_bytes() const;
    [[nodiscard]] std::uint64_t engine_device_bytes() const;
    [[nodiscard]] qwen_route::Identity numerical_identity() const;
    [[nodiscard]] qwen_route::Policy route_storage_policy() const;
    // No window/chunk in flight. Armed MTP includes committed predictor pairs and
    // retained pooled bytes; shared draft/window scratch is overwritten before use.
    // Includes canonical KV/index, GDN matrices,
    // histories, ngram, held row and length; excludes derived pooled caches.
    CanonicalState capture_canonical(SessionId session = 0);
    void restore_canonical(const CanonicalState&, SessionId session = 0);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::cuda::qwen_prefill
