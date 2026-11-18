#pragma once
// Fast prefill form of the deepseek41 CSA2 attention (window ring + selected compressed entries, sinks, RoPE in, inverse
// RoPE out) for calls of many rows. NOT the declared arithmetic of cuda_ds41_attention.hpp: tensor-core products
// (FP16 inputs, FP32 accumulation; the dequantised cache entries are exact in FP16), base-2 online softmax over tiles
// of 32 keys in a free order, approximate exp2. Same inputs, selection and output layout as enqueue_attention (the
// caches through their FP16 rows, enqueue_fast_kv), so it replaces the attention calls of a prefill chunk
// (docs/models/deepseek-v4.1/cuda-fast-attention.md). Only mma.sync / ldmatrix / cp.async (sm_80 and later: built for sm_89 and
// sm_120).
#include "strix/cuda/ds41_attention.hpp"
#include "strix/cuda/ds41_common.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace strix::ds41::cuda::attn {

// FP16 rows (512 values) of cache entries, the values of the dequantised entries (exact but for window scales outside
// 2^-14 .. 2^15): window positions [first, first + count) of the ring -> win16[(pos - first) * 512], compressed entries
// [0, kv_count) -> kv16[e * 512]. A prefill converts once per layer, after the layer's window / compressor writes.
struct FastKvView {
    const std::uint8_t* ring = nullptr;
    std::uint32_t capacity = 0, first = 0, count = 0;
    std::uint16_t* win16 = nullptr;
    const std::uint8_t* kv_cache = nullptr;
    std::uint32_t kv_count = 0;
    std::uint16_t* kv16 = nullptr;
};
void enqueue_fast_kv(const FastKvView& v, cudaStream_t stream);

struct FastAttentionView {
    std::uint32_t rows = 0, topk = kTopK;
    const float* q = nullptr;               // R x heads x 512, attn_q_b before RoPE
    const float* sink = nullptr;            // heads
    const RopeRow* rope = nullptr;          // table of the layer's kind, by position
    const std::uint64_t* pos = nullptr;     // R
    const std::uint16_t* win16 = nullptr;   // FP16 window entries by position - win_first (every row's last 128)
    std::uint32_t win_first = 0;
    const std::uint16_t* kv16 = nullptr;    // FP16 compressed entries (null for a window-only layer)
    const std::uint32_t* idx = nullptr;     // R x topk (null for a window-only layer)
    const std::uint32_t* idx_count = nullptr;
    float* o = nullptr;                     // optional: R x heads x 512, after the inverse RoPE
    A8Rows a8{};                            // optional (codes != null): quantize_a8 of the o rows (width heads x 512)
    std::uint32_t heads = kHeads;           // multiple of 32
};
void enqueue_fast_attention(const FastAttentionView& v, cudaStream_t stream);

// ---- index scores on FP8 tensor cores (src/cuda/models/ds41/ds41_fast_index.cu) -------------------------------------------------
// The indexer's E2M1 products as E4M3 values relative to each vector's largest block scale (exact unless a block is
// more than 2^8 below it), FP32 accumulation in a free order; the selection (enqueue_select) runs on these scores.
// Needs compute capability 8.9 or later (fast_index_supported).
[[nodiscard]] bool fast_index_supported(int device);

// Index keys [0, count) of a KV source as E4M3 rows of 128 and their factors 2^ck (once per layer, after the writes).
struct FastIndexKeysView {
    const std::uint8_t* idx_cache = nullptr;
    std::uint32_t count = 0;
    std::uint8_t* k8 = nullptr;    // count x 128
    float* kscale = nullptr;       // count
};
void enqueue_fast_index_keys(const FastIndexKeysView& v, cudaStream_t stream);

inline constexpr unsigned kFastIdxPrepBytes = kIdxHeads * kIdxDim + kIdxHeads * 4;
struct FastIndexView {
    std::uint32_t rows = 0, ratio = 0, max_count = 0;  // max_count >= every row's (pos + 1) / ratio (the scores' stride)
    const float* q = nullptr;            // R x 32 x 128, indexer attn_q_b(qr)
    const float* w = nullptr;            // R x 32, indexer proj(x)
    const RopeRow* rope = nullptr;       // table of the layer's kind, by position
    const std::uint64_t* pos = nullptr;  // R
    const std::uint8_t* k8 = nullptr;    // enqueue_fast_index_keys of the KV source (>= max_count entries)
    const float* kscale = nullptr;
    float* scores = nullptr;             // R x max_count (positions t < (pos + 1) / ratio written)
    void* qprep = nullptr;               // scratch R x kFastIdxPrepBytes
};
void enqueue_fast_index_scores(const FastIndexView& v, cudaStream_t stream);

}  // namespace strix::ds41::cuda::attn
