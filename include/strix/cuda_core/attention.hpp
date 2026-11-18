#pragma once
#include <cuda_bf16.h>

#include <cstddef>
#include <cstdint>

#include "strix/cuda_core/types.hpp"
#include "strix/cuda_core/attention_policy.hpp"  // FlashPipe, FlashTail

namespace strix::cuda_core {

struct AttentionDims {
    std::uint32_t heads = 0, kv_heads = 0, head_dim = 256;  // GQA: q head h reads kv head h / (heads / kv_heads)
    float scale = 1.0f;
    std::uint32_t max_positions = 0;  // KV cache capacity per kv head
};

// KV cache BF16, layout [kv_head][max_positions][head_dim] (one buffer each for K and V).
// pos0 is a DEVICE scalar (graph capture: the step's start position changes without re-capture).
// Stores rows k/v ([rows][kv_heads][head_dim] FP32, contiguous) at positions pos0 + r, rounding RNE to BF16.
cudaError_t kv_store(const AttentionDims& d, const float* k, const float* v, __nv_bfloat16* k_cache,
                     __nv_bfloat16* v_cache, const std::uint32_t* pos0_dev, std::uint32_t rows, cudaStream_t stream, const RowMap& rm = {});

constexpr std::uint32_t kAttnChunk = 256;  // positions per reduction chunk
[[nodiscard]] __host__ __device__ inline std::uint32_t attn_max_chunks(const AttentionDims& d) { return (d.max_positions + kAttnChunk - 1) / kAttnChunk; }

// Floats of scratch attention_decode needs for `rows`: scores + chunk max + chunk denominators + chunk value partials.
inline std::size_t attention_scratch_floats(const AttentionDims& d, std::uint32_t rows) {
    const std::size_t rh = static_cast<std::size_t>(rows) * d.heads, mc = attn_max_chunks(d);
    return rh * d.max_positions + 2 * rh * mc + rh * mc * d.head_dim;
}

// Row r (position pos0 + r) attends to cache positions 0 .. pos0 + r (causal inside the window; the window's own K/V
// must have been stored first). q: FP32 view (rows x heads x head_dim). gate (optional, same shape, its own strides):
// out *= sigmoid(gate). ORDER per (row, head), positions split into 256-position chunks by absolute position:
// score_p = fl(dot) * scale, dot = lane owns 8 consecutive dims sequentially then xor butterfly 16..1; m = max over all positions
// (exact); e_p = expf(s_p - m); chunk denominator = block tree (thread t holds position chunk*256 + t, zero past the length) of norms.hpp;
// chunk value partial_d = sum over the chunk positions ascending of fl(e_p * v[p][d]); denom = chunk denominators added ascending,
// acc = chunk partials added ascending, out = acc / denom. Independent of rows and launch shape. head_dim 256, group <= 8.
cudaError_t attention_decode(const AttentionDims& d, const float* q, const GroupedView& qv, const float* gate,
                             const GroupedView& gv, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache,
                             const std::uint32_t* pos0_dev, std::uint32_t rows, float* out /*[rows][heads][head_dim]*/,
                             float* scratch, cudaStream_t stream, const RowMap& rm = {});

// PREFILL attention (FAST class, NOT bitwise equal to attention_decode): causal flash attention over the BF16 KV cache for a chunk of
// `rows` query rows at positions pos0 .. pos0 + rows - 1 (the chunk's own K/V must already be stored by kv_store). Q is rounded to bf16,
// scores and the online softmax are FP32, P is rounded to bf16 for the PV mma (FP32 accumulate); out = o * sigmoid(gate) like
// attention_decode. head_dim 256, GQA group <= 8. q/gate/out use the same grouped views as attention_decode.
// FlashPipe = shared-memory pipeline of the K/V tiles only (scheduling): every pipe gives the SAME bits (same tiles, same order per row).
//   pad2: padded rows, 2 stages, two barriers per tile (the original form, kept as the off switch);
//   swz3: XOR-swizzled dense rows, 3 stages (prefetch distance 2), one barrier per tile.
// FlashTail = wave-tail split (scheduling only, SAME bits): with T = ceil(rows/128) * heads CTAs on S SMs (1 CTA/SM), the last
// T mod S tiles form a partial wave that bills a whole CTA time. split: those tiles run as jobs of w active warps (w = 1/2/4, the
// smallest with (T mod S) * 8 / w <= S, or tail_warps if non-zero), so the partial wave spreads over the idle SMs. Every warp keeps its
// 16 rows, its head and the tile count of its 128-row block, so each output row runs the same instructions on the same tiles.
cudaError_t attention_prefill_flash(const AttentionDims& d, const float* q, const GroupedView& qv, const float* gate, const GroupedView& gv,
                                    const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache, std::uint32_t pos0, std::uint32_t rows,
                                    float* out, cudaStream_t stream, FlashPipe pipe = FlashPipe::pad2, FlashTail tail = FlashTail::off,
                                    std::uint32_t tail_warps = 0);
// Launch plan of attention_prefill_flash (host arithmetic; exposed for tests and the effective-config print).
struct FlashPlan {
    std::uint32_t tiles = 0, full = 0, tail_tiles = 0, tail_warps = 8, jobs = 0;
};
FlashPlan flash_plan(std::uint32_t rows, std::uint32_t heads, std::uint32_t sms, FlashTail tail, std::uint32_t tail_warps = 0);
// Launch geometry of attention_prefill_flash for the wave planner (strix/wave_fit.hpp): q rows per CTA, CTAs per q-row tile
// (one per q head), SMs, resident CTAs per SM from the occupancy of the real kernel. Keep it in step with the kernel.
struct FlashGrid {
    std::uint32_t tile_rows = 0, ctas_per_tile = 0, sms = 0, ctas_per_sm = 0;
};
cudaError_t attention_prefill_flash_grid(const AttentionDims& d, FlashPipe pipe, int device, FlashGrid* out);
// FAST8 prefill attention (bit-changing; under the FAST8 prefill policy and its quality gate only). An e4m3 SHADOW of the BF16 KV cache
// (the BF16 cache stays the source of truth for decode and for every other path): per kv head K8 [P8][head_dim] e4m3 + ks [P8] FP32
// (k = K8 * ks, ks = amax / 448 per position), V8T [head_dim][P8] e4m3 transposed + vs [P8] (v = V8T * vs); P8 = max_positions rounded up
// to kAttn8Align. Allocate kv_shadow_e4m3_bytes and zero it once (positions never written must hold finite values).
constexpr std::uint32_t kAttn8Align = 64;
struct KvShadowE4m3 {
    std::uint8_t* k8 = nullptr;   // [kv_heads][P8][head_dim]
    float* ks = nullptr;          // [kv_heads][P8]
    std::uint8_t* v8t = nullptr;  // [kv_heads][head_dim][P8], mma key order inside 32-key groups
    float* vs = nullptr;          // [kv_heads][P8]
};
[[nodiscard]] std::uint32_t attn8_capacity(const AttentionDims& d);
[[nodiscard]] std::size_t kv_shadow_e4m3_bytes(const AttentionDims& d);  // k8 + ks + v8t + vs
// Re-quantises cache positions pos_begin .. pos_begin + count - 1 (all kv heads) from the BF16 cache into the shadow.
cudaError_t kv_shadow_e4m3(const AttentionDims& d, const __nv_bfloat16* k_cache, const __nv_bfloat16* v_cache, const KvShadowE4m3& sh,
                           std::uint32_t pos_begin, std::uint32_t count, cudaStream_t stream);
// Same contract as attention_prefill_flash (rows at pos0 .., causal, gated output) with Q quantised per (row, head) to e4m3, S = Q8 K8^T
// and O += P'' V8 on e4m3 tensor cores (FP32 accumulate), online softmax FP32. Keys 0 .. pos0 + rows - 1 must be in the shadow.
cudaError_t attention_prefill_flash8(const AttentionDims& d, const float* q, const GroupedView& qv, const float* gate, const GroupedView& gv,
                                     const KvShadowE4m3& sh, std::uint32_t pos0, std::uint32_t rows, float* out, cudaStream_t stream);
// p[i] = base + step * i for i < n (small device helper used to address attention windows).
cudaError_t iota_u32(std::uint32_t* p, std::uint32_t n, std::uint32_t base, std::uint32_t step, cudaStream_t stream);

}  // namespace strix::cuda_core
