#pragma once
// Qwen3.8-27B dense family (`qwen35`) on one HIP device: window engine over the shared HIP cores (dense_linear,
// attn_core dense_decode, hip_gdn_decode_step_dims, ops/norm). Same drafting contract as CudaEngine:
//   forward_rows(tokens[R]) runs R rows at positions committed() .. committed()+R-1 (R <= max_rows()) and leaves per-row
//   logits / argmax on the device. Nothing is committed: the GDN matrix + conv state are snapshotted first, the KV rows are
//   written past the committed length (dead until committed). commit(k) accepts the first k rows of the LAST window:
//   k == R keeps the advanced state (free); 0 < k < R restores the snapshot and replays the first k rows of every GDN layer
//   from the saved raw inputs; k == 0 restores only. Rows are bitwise equal to row-by-row decode (every reduction order is
//   independent of the row count).
// Home: specific (model layer composition: qwen35 GDN/attention interleave); every kernel is shared (backend core).
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "strix/qwen35/config.hpp"
#include "strix/qwen35/engine_types.hpp"

namespace strix::qwen35 {

struct HipEngineOptions {
    std::uint32_t max_positions = 4096;  // KV capacity per full-attention layer
    bool quiet = false;
    // forward_commit window width. 16 = exact (the serial arithmetic). > 16 = FAST policy dense-q8a8-f16-v1 (shared
    // core::f16_dense_tall::q8a8: A8 activations x Q8_0 weights, FP16 WMMA; IU4 weights: dense-iu4a16-f16-v1 of
    // core::f16_dense_tall::iu4a16, FP16 activations x decoded IU4) for the dense projections of those wide windows;
    // GDN and attention keep their exact kernels. Decode windows (forward_rows) stay <= 16 rows.
    std::uint32_t prefill_rows = 16;
};

struct HipRowsOut {
    const float* logits = nullptr;       // device [rows][vocab] (all) or [1][vocab] (last); null for none
    const std::uint32_t* top = nullptr;  // device [rows] argmax (lowest index on ties) (all) or [1] (last)
    // Only with an MTP layer loaded (load_mtp): per-row trunk hidden of EVERY row of the window (valid until the next forward).
    const float* hidden_post = nullptr;  // device [rows][hidden] after the final norm
    const float* hidden_pre = nullptr;   // device [rows][hidden] residual stream before the final norm
};

// MTP layer outputs (device pointers valid until the next mtp_forward); same contract as the CUDA engine's MtpOut.
struct HipMtpOut {
    const float* logits = nullptr;  // [rows][vocab] (want_top)
    const std::uint32_t* top = nullptr;
    const float* hidden_pre = nullptr;   // MTP block output before shared_head_norm
    const float* hidden_post = nullptr;  // after shared_head_norm
};

class HipEngine {
public:
    using Rows = HipRowsOut;   // shared MtpChain binding (strix/spec/mtp_chain.hpp)
    using MtpOut = HipMtpOut;
    using LogitRows = qwen35::LogitRows;
    ~HipEngine();
    HipEngine(const HipEngine&) = delete;
    HipEngine& operator=(const HipEngine&) = delete;
    static std::unique_ptr<HipEngine> from_gguf(const std::string& path, const HipEngineOptions& opt);

    [[nodiscard]] const Config& config() const;
    [[nodiscard]] std::uint32_t vocab() const;
    [[nodiscard]] static constexpr std::uint32_t max_rows() { return 16U; }
    [[nodiscard]] std::uint32_t prefill_rows() const;  // widest forward_commit window (options)
    [[nodiscard]] std::uint32_t committed() const;
    [[nodiscard]] std::uint64_t weight_bytes() const;          // bytes read per pass (all weights but the embedding table)
    [[nodiscard]] std::uint64_t state_bytes_per_pass() const;  // GDN state read + write per pass
    [[nodiscard]] std::uint32_t kv_bytes_per_position() const;
    void reset();  // committed length 0, recurrent state zero, window discarded

    // rows 1..max_rows(). tokens: host array. Async on the stream; read outputs after read_top / sync.
    HipRowsOut forward_rows(const std::uint32_t* tokens, std::uint32_t rows, LogitRows want = LogitRows::all);
    void commit(std::uint32_t k);  // accepts the first k rows of the last forward_rows window (0 <= k <= rows)
    // Same arithmetic, every row accepted immediately: no snapshot, no replay inputs (prefill chunks, serial decode).
    HipRowsOut forward_commit(const std::uint32_t* tokens, std::uint32_t rows, LogitRows want = LogitRows::all);

    void read_top(const HipRowsOut& o, std::uint32_t rows, std::uint32_t* host) const;  // synchronizes
    void read_logits(const HipRowsOut& o, std::uint32_t rows, float* host) const;       // [rows][vocab], synchronizes
    void sync() const;

    // MTP layer (blk.<layers>.* of a side GGUF: attn_norm, post_attention_norm, ffn_*, attn_q/k/v/output, q/k norms, nextn.enorm /
    // hnorm / eh_proj / shared_head_norm; bf16 or q8_0 matrices, F32 norms). The embedding and lm_head are the trunk's.
    // Call once after from_gguf; afterwards forward_rows / forward_commit also return hidden_pre / hidden_post (one extra final
    // norm launch per pass; the logits path is unchanged). Without load_mtp nothing changes.
    void load_mtp(const std::string& path);
    [[nodiscard]] bool has_mtp() const;
    [[nodiscard]] static constexpr std::uint32_t mtp_max_rows() { return 16U; }
    // Row r runs at position pos0 + r with input token next_tokens[r] (the token FOLLOWING the hidden's position) and
    // hidden_dev[r] = the trunk hidden of position pos0 + r. Rows are causal inside the call; earlier positions must have been
    // written by earlier calls (own bf16 KV, positions explicit). want_top=false skips lm_head + argmax.
    HipMtpOut mtp_forward(const std::uint32_t* next_tokens, const float* hidden_dev, std::uint32_t rows, std::uint32_t pos0,
                          bool want_top = true);
    // Device helpers for the shared chain glue (all on the engine stream).
    [[nodiscard]] float* alloc_f32(std::size_t n) const;
    void free_f32(float* p) const;
    void copy_f32_async(float* dst, const float* src, std::size_t n) const;
    // Timing only: set the committed length to `pos` WITHOUT computing (KV / state keep their contents; positions below pos are
    // whatever the cache holds). Used to sample window cost at long context without running the whole prefill.
    void bench_seek(std::uint32_t pos);

private:
    HipEngine();
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace strix::qwen35
