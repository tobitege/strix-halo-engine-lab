#pragma once
// Qwen3.8-27B dense family (`qwen35`) on one CUDA device: reusable engine over the shared cuda_core kernels.
// Home: specific (model layer composition); every arithmetic kernel lives in src/cuda_core (shared).
//
// Drafting contract (Gemma-style target):
//   forward_rows(tokens[R]) runs R rows at positions committed() .. committed()+R-1 and leaves per-row logits / argmax /
//   post-final-norm hidden on the device. NOTHING is committed: the GDN recurrent state and conv state are written to a
//   shadow copy, the KV rows are written past the committed length. commit(k) then accepts the first k rows of the LAST
//   window (0..R): k == R flips the state parity (free); 0 < k < R re-applies the first k rows of the window to the
//   pre-window state in one state pass (the window's per-row GDN inputs are kept on the device); k == 0 discards.
//   Rows 1..R are bitwise equal to row-by-row decode (every reduction order is independent of the row count).
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "strix/cuda_core/attention_policy.hpp"
#include "strix/cuda_core/dense.hpp"
#include "strix/qwen35/config.hpp"
#include "strix/qwen35/engine_types.hpp"

namespace strix::qwen35 {

enum class PackFormat : std::uint8_t { bf16, q8_0 };
// TURBO weight source (owner requirement): off = exact pack as is; native = TURBO policy without weight change;
// onfly = BF16 2-D weights rounded at load to NVFP4 (RTN per 16, kept only in VRAM). converted = the same NVFP4 bytes read from a pack written by save_converted_pack (no BF16 source,
// no conversion; the pack decides which tensors are NVFP4, e.g. lm_head).
enum class TurboWeights : std::uint8_t { off, native, onfly, converted };
// Prefill arithmetic policy for CudaEngine::prefill (the large-M path). exact: dense on the large-M tensor-core GEMM with the decode
// reduction order + decode attention in 32-row windows + sequential GDN: bitwise equal to forward_rows windows + commit.
// flash: exact dense, causal flash attention (bf16 P, online softmax; FAST attention, differs from decode attention).
// fast: flash + activation split of ONE bf16 term (mma1; nvfp4 weights are unchanged: their activation is f16 already).
// fast2: flash + two bf16 terms (mma2).
// fast8: fast + nvfp4 weights on the e4m3 x e4m3 mma (activation per-row e4m3, weight e2m1*e4m3 rounded to e4m3): TURBO/FAST, quality-gated.
enum class PrefillArith : std::uint8_t { exact, flash, fast, fast2, fast8 };
[[nodiscard]] const char* prefill_arith_name(PrefillArith a);

struct EngineOptions {
    cuda_core::DenseArith dense_arithmetic = cuda_core::dense_arith();
    std::uint32_t max_positions = 4096;  // KV capacity per attention layer (and for the MTP layer)
    bool use_graphs = true;              // CUDA graph per (rows, logits mode)
    bool multi_gemv = true;              // one launch for GEMVs sharing x (q|k|v, qkv|z|beta|alpha); bitwise equal either way
    bool pdl = false;                    // (measured +0.27 ms loss at C1: off) programmatic dependent launch + weight L2 prefetch between kernels
    bool load_mtp = true;                // read blk.<layers> (MTP) tensors when present
    bool quiet = false;
    // TURBO levers (default off = exact BF16/q8_0 identity). Each lever is individually switchable.
    TurboWeights turbo_weights = TurboWeights::off;
    std::uint32_t turbo_bits = 4;   // 4 = NVFP4 (3/2 later)
    bool turbo_lm_head = true;      // onfly: lm_head follows the source (true) or stays BF16 (false)
    // multi_gemv (above) is also the TURBO multi-launch lever.
    // Large-M prefill (CudaEngine::prefill). Default exact = bitwise equal to the windowed path.
    PrefillArith prefill_arith = PrefillArith::exact;
    std::uint32_t prefill_chunk = 2048;  // rows per large-M chunk (multiple of 128, <= 8192); buffers are allocated on first use
    // Wave-aligned steps (strix/wave_fit.hpp, scheduling only, bitwise): prompts of >= this many tokens step by the largest
    // rows <= prefill_chunk whose flash grid fills whole waves (4096 -> 3968 on 188 SMs: flash +16 % at 128K, bench). 0 = off
    // (opt-in until the 128K/256K wall ABAB; then the measured crossover becomes the default).
    std::uint32_t prefill_wave_min_tokens = 0;
    std::uint32_t prefill_slots = 2;     // chunks in flight (1..2): 2 overlaps the sequential GDN kernel of one chunk with the GEMMs of the next
    // bounded submission (cuda_core/submit_window.hpp): the host queues at most this many chunks ahead of the GPU (waits on
    // chunk i - N before queuing chunk i). Scheduling only, bitwise. 0 = unbounded (whole prefill queued before one sync).
    std::uint32_t prefill_inflight = 3;
    std::uint32_t prefill_ramp = 0;      // > 0: first chunks grow ramp, 2*ramp, ... (< prefill_chunk) from idle; 0 = off
    bool prefill_big = true;             // false: prefill() runs forward_rows windows of 32 + commit (reference / fallback)
    // Long-context prefill attention (non-exact prefill policies only; the exact policy keeps attention_decode).
    // flash_pipe: shared-memory pipeline of the flash kernel (scheduling only, bitwise equal across pipes).
    cuda_core::FlashPipe flash_pipe = cuda_core::FlashPipe::pad2;
    // flash_tail: wave-tail split of the flash grid (scheduling only, bitwise equal; measured +4.5 % flash at 4096-row chunks). off = old grid.
    cuda_core::FlashTail flash_tail = cuda_core::FlashTail::split;
    // attn8: e4m3 flash attention over an e4m3 shadow of the KV cache (bit-changing, own identity suffix "-a8"; FAST gate required).
    bool prefill_attn8 = false;
};

// Named identity of the arithmetic + weight source; also the prefix-cache key (caches never mix identities).
[[nodiscard]] inline std::string engine_identity(const EngineOptions& o) {
    static const char* names[] = {"fp32", "mma3", "mma2", "mma1"};
    static const char* pf_names[] = {"exact", "flash", "fast", "fast2", "fast8"};
    const std::string a = names[static_cast<unsigned>(o.dense_arithmetic)];
    // a non-exact prefill policy is a different arithmetic identity (prefix caches never mix them)
    const std::string pf = o.prefill_arith == PrefillArith::exact
        ? ""
        : std::string("-pf-") + pf_names[static_cast<unsigned>(o.prefill_arith)] + (o.prefill_attn8 ? "-a8" : "");
    if (o.turbo_weights == TurboWeights::off) return "qwen35-cuda-exact-" + a + "-v1" + pf;
    if (o.turbo_weights == TurboWeights::native) return "qwen35-turbo-native-" + a + "-v1" + pf;
    // converted packs hold the onfly bytes: same arithmetic identity as onfly (prefix caches may be shared)
    return "qwen35-turbo-nvfp4-onfly-rtn16-v2-retained-" + a + (o.turbo_lm_head ? "" : "-lmhead-bf16") + pf;
}
// Writes the onfly NVFP4 weights of the BF16 gguf `src` as a converted pack at `out` (use with TurboWeights::converted). Returns bytes.
std::uint64_t save_converted_pack(const std::string& src, const std::string& out, const EngineOptions& opt);
// Effective-config text (identity, prefix-cache key, every lever).
void print_effective_config(const EngineOptions& o, const char* dense_arith, std::FILE* f);

// Outputs of the last forward (device pointers, valid until the next forward on this engine).
struct RowsOut {
    const float* logits = nullptr;    // [rows][vocab] (all) or [1][vocab] (last), null for none
    const std::uint32_t* top = nullptr;  // [rows] argmax (lowest index on ties) (all) or [1] (last)
    const float* hidden_post = nullptr;  // [rows][hidden] after the final norm (always)
    const float* hidden_pre = nullptr;   // [rows][hidden] residual stream before the final norm
};

// Multi-session decode (slots): one batch = groups; a group is `rows` consecutive rows of ONE session slot (1 = a decode lane,
// more = a verify / prefill window). Total rows over all groups <= 32.
struct SlotRows {
    std::uint32_t slot = 0;
    std::uint32_t rows = 1;
};

class CudaEngine {
public:
    using Rows = RowsOut;  // shared MtpChain binding (strix/spec/mtp_chain.hpp)
    using LogitRows = qwen35::LogitRows;
    ~CudaEngine();
    CudaEngine(const CudaEngine&) = delete;
    CudaEngine& operator=(const CudaEngine&) = delete;

    static std::unique_ptr<CudaEngine> from_gguf(const std::string& path, const EngineOptions& opt);
    // Real dimensions of `cfg` (defaults = Qwen3.8-27B) with weights generated on the device by a counter-based RNG
    // (deterministic in seed, scaled so activations stay finite). No disk. Timing equals the real model (memory bound).
    static std::unique_ptr<CudaEngine> synthetic(const Config& cfg, PackFormat pack, const EngineOptions& opt,
                                                 std::uint64_t seed);

    [[nodiscard]] const Config& config() const;
    [[nodiscard]] const EngineOptions& options() const;
    [[nodiscard]] std::string identity() const;  // engine_identity(options(), current dense arith)
    [[nodiscard]] std::uint32_t nvfp4_tensors() const;  // number of tensors stored as NVFP4
    [[nodiscard]] bool has_mtp() const;
    [[nodiscard]] cudaStream_t stream() const;
    [[nodiscard]] std::uint32_t committed() const;             // committed sequence length
    [[nodiscard]] std::uint64_t weight_bytes() const;          // bytes read per pass by the trunk + lm_head
    [[nodiscard]] std::uint64_t state_bytes_per_pass() const;  // GDN state read + write per pass

    void reset();  // committed length 0, all recurrent state zero, window discarded

    // rows 1..32 (cuda_core::kMaxRows). tokens: host array. Async on stream(); read outputs after synchronizing.
    RowsOut forward_rows(const std::uint32_t* tokens, std::uint32_t rows, LogitRows want = LogitRows::all);
    // Prefill n tokens (any n >= 1) at positions committed() .. committed()+n-1 and COMMIT them (same committed state as forward_rows
    // windows + commit(rows) under PrefillArith::exact). Returns the logits / argmax / hidden of the LAST token (want = last; none
    // skips the lm_head). Runs the large-M tensor-core path in chunks of options().prefill_chunk rows; falls back to windows when
    // prefill_big is off, dense arith is fp32 or a weight shape is unsupported. Synchronises the host at the end (own streams); no CUDA graphs.
    RowsOut prefill(const std::uint32_t* tokens, std::uint32_t n, LogitRows want = LogitRows::last);
    [[nodiscard]] bool prefill_is_big() const;  // whether prefill() takes the large-M path for this engine
    // Accepts the first k rows of the last forward_rows window (0 <= k <= rows). Async on stream().
    void commit(std::uint32_t k);

    // MTP layer (single full-attention block with its own KV; positions are explicit, no hidden state to commit).
    // Row r runs at position pos0 + r with input token next_tokens[r] (the token FOLLOWING the hidden's position) and
    // hidden_dev[r] = the trunk hidden of position pos0 + r (use RowsOut::hidden_post or hidden_pre: the convention is the
    // caller's choice). Rows are causal inside the call; earlier positions must have been written by earlier calls.
    struct MtpOut {
        const float* logits = nullptr;  // [rows][vocab]
        const std::uint32_t* top = nullptr;
        const float* hidden_pre = nullptr;   // MTP block output before mtp.norm (feed back for chains, convention 'pre')
        const float* hidden_post = nullptr;  // after mtp.norm
    };
    MtpOut mtp_forward(const std::uint32_t* next_tokens, const float* hidden_dev, std::uint32_t rows, std::uint32_t pos0,
                    bool want_top = true);  // want_top=false: skip lm_head + argmax (logits/top null): KV/hidden only

    // Host copies (synchronize the stream).
    void read_logits(const RowsOut& o, std::uint32_t row, float* host_vocab) const;
    void read_top(const RowsOut& o, std::uint32_t rows, std::uint32_t* host) const;
    void read_hidden(const float* dev, std::uint32_t rows, float* host) const;
    void sync() const;

    // ---- Multi-session (slots). Home: shared mechanism (row map + slot tables live in cuda_core attention/rope/gdn); the layer
    // composition below is model-specific. Slots have their own KV, GDN/conv state, parity and length; the single-session API above
    // is a separate session (do not interleave its pending window with a slot batch: both reuse the window buffers).
    // Arithmetic per row is bitwise equal to running that session alone (every reduction is independent of the row count/mix).
    void configure_slots(std::uint32_t n);  // allocate state for n sessions (once, before graphs); n >= 1
    [[nodiscard]] std::uint32_t slot_count() const;
    [[nodiscard]] std::uint32_t slot_committed(std::uint32_t slot) const;
    void reset_slot(std::uint32_t slot);
    void copy_slot(std::uint32_t dst, std::uint32_t src);  // dst becomes a copy of src (KV prefix, GDN state, length)
    // tokens: host [sum rows] in group order; group g row i runs at position slot_committed(slot)+i. want: none/all (last == all for
    // 1-row groups; for wider groups it also returns all rows). auto_commit: every row is accepted (state committed inside the graph);
    // false keeps the window until commit_slots. Logits / top / hidden are indexed by batch row.
    RowsOut forward_slots(const SlotRows* groups, std::uint32_t n, const std::uint32_t* tokens, LogitRows want = LogitRows::all,
                          bool auto_commit = true);
    // After forward_slots(..., auto_commit=false): accept[g] rows of group g (0..rows) are committed, the rest discarded.
    void commit_slots(const std::uint32_t* accept);

    // Diagnostics (no effect on arithmetic).
    void set_graphs(bool on);
    [[nodiscard]] std::uint64_t launches_last_forward() const;

private:
    CudaEngine();
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace strix::qwen35
