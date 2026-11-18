// FAST lever ds41-attn-bf16flash-v1 (STRIX_DS41_FAST_ATTN=2, default off; prefill only). NOT bit-exact: a declared
// arithmetic behind the FAST profile. Included by src/hip_ds41_attention.hip inside its anonymous namespace.
// DS41 as a policy of the shared sparse flash core (attn_core/sparse_flash_wmma.hpp, the GLM geometry: 32 heads x
// 512): BF16 WMMA, one pass per product. Keys: BF16 mirrors of the FP8 entries (E4M3 x 2^k and E2M1 x E4M3 have at
// most 7 significant bits and BF16 has the FP32 exponent range, so the mirror is exact and needs no exponent bands).
// Queries (after RoPE, FP32) and probabilities are rounded once to BF16. The work-group writes ONE (m, l, acc)
// partial into chunk slot 0 and the unchanged combine finishes the row (sink, 1 / L, inverse RoPE, A8 of o).
__global__ __launch_bounds__(32) void flash_mirror_kernel(KeyMirrorView v) {
    const unsigned lane = threadIdx.x;
    const std::size_t e = v.window ? static_cast<std::size_t>(v.pos[blockIdx.x] % v.capacity) : blockIdx.x;
    float x[16];
    if (v.window) {
        const std::uint8_t* s = v.src + e * kWindowEntryBytes;
        const uint4 c = *reinterpret_cast<const uint4*>(s + 16U * lane);
        const float sc = e8m0_value(s[512U + lane / 2U]);
        const std::uint32_t w[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) x[i] = e4m3_value(static_cast<std::uint8_t>(w[i / 4U] >> (8U * (i % 4U)))) * sc;
    } else {
        const std::uint8_t* s = v.src + e * kKvEntryBytes;
        const std::uint32_t* c = reinterpret_cast<const std::uint32_t*>(s + 8U * lane);
        const std::uint32_t w[2] = {c[0], c[1]};
        const float sc = e4m3_value(s[256U + lane]);
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) x[i] = e2m1_value(static_cast<std::uint8_t>((w[i / 8U] >> (4U * (i % 8U))) & 15U)) * sc;
    }
    std::uint32_t o[8];
#pragma unroll
    for (unsigned i = 0; i < 8; ++i) o[i] = bf_rn(x[2 * i]) | (static_cast<std::uint32_t>(bf_rn(x[2 * i + 1])) << 16U);
    uint4* d = reinterpret_cast<uint4*>(v.dst + e * kHeadDim + 16U * lane);
    d[0] = make_uint4(o[0], o[1], o[2], o[3]);
    d[1] = make_uint4(o[4], o[5], o[6], o[7]);
}

struct FlashAttentionPolicy {
    using Args = AttentionView;
    static constexpr unsigned kHeads = 32, kDim = kHeadDim;
    static constexpr strix::attn_core::tile::Operand kOp = strix::attn_core::tile::Operand::bf16;
    static constexpr float kScaleLog2 = kAttnScale * 1.4426950408889634F;
    // work-group = row * (heads / 32) + head group
    __device__ static __forceinline__ unsigned row(const Args& v, unsigned wg) { return wg / (v.heads / 32U); }
    __device__ static __forceinline__ unsigned head0(const Args& v, unsigned wg) { return (wg % (v.heads / 32U)) * 32U; }
    __device__ static __forceinline__ unsigned count(const Args& v, unsigned wg) {
        unsigned nw;
        return visible_keys(v, row(v, wg), nw);
    }
    // cache position: a ring slot, or capacity + the selected entry
    __device__ static __forceinline__ unsigned key(const Args& v, unsigned wg, unsigned j) {
        const unsigned r = row(v, wg);
        const std::uint64_t p = v.pos[r];
        const unsigned nw = p + 1U < kWindow ? static_cast<unsigned>(p + 1U) : kWindow;
        if (j < nw) {
            unsigned slot = static_cast<unsigned>((p + 1U - nw) % v.capacity) + j;
            if (slot >= v.capacity) slot -= v.capacity;
            return slot;
        }
        return v.capacity + v.idx[static_cast<std::size_t>(r) * v.topk + j - nw];
    }
    __device__ static __forceinline__ const std::uint16_t* key_row(const Args& v, unsigned, unsigned pos) {
        return pos < v.capacity ? v.ring16 + static_cast<std::size_t>(pos) * kHeadDim : v.kv16 + static_cast<std::size_t>(pos - v.capacity) * kHeadDim;
    }
    __device__ static __forceinline__ const std::uint16_t* value_row(const Args& v, unsigned wg, unsigned pos) { return key_row(v, wg, pos); }
    __device__ static __forceinline__ void query16(const Args& v, unsigned wg, unsigned head, unsigned chunk, float (&x)[16]) {
        const unsigned r = row(v, wg);
        const float* src = v.q + (static_cast<std::size_t>(r) * v.heads + head0(v, wg) + head) * kHeadDim + 16U * chunk;
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) x[i] = src[i];
        rope512(x, v.rope[r], chunk, false);
    }
    __device__ static __forceinline__ float* part(const Args& v, unsigned wg, unsigned head) {
        return v.partial + (static_cast<std::size_t>(row(v, wg)) * v.heads + head0(v, wg) + head) * kMaxChunks * (2 + kHeadDim);
    }
    // the combine's units: m of the scaled score in natural-log units
    __device__ static __forceinline__ void emit_state(const Args& v, unsigned wg, unsigned head, float m, float l) {
        float* pp = part(v, wg, head);
        pp[0] = m * 0.6931471805599453F; pp[1] = l;
    }
    __device__ static __forceinline__ void emit(const Args& v, unsigned wg, unsigned head, unsigned dim, float o) { part(v, wg, head)[2U + dim] = o; }
};
// The same policy reading the q_b epilogue's BF16 image of q after RoPE (AttentionView::q16; query_op of the shared
// core): the image is to_bf16(rope512(q)) of the FP32 q query16 loads, so the fragments and every result are the
// same bits; no FP32 prologue, half the query bytes (STRIX_DS41_FAST_ATTN_Q16=0 keeps the FP32 q).
struct FlashAttentionQ16Policy : FlashAttentionPolicy {
    __device__ static __forceinline__ const std::uint16_t* query_op(const Args& v, unsigned wg) {
        return v.q16 + (static_cast<std::size_t>(row(v, wg)) * v.heads + head0(v, wg)) * kHeadDim;
    }
};

// policy ds41-decoder-replay-bounded-v1: the window keys from the layer's key_floor (visible_keys_floor); the ring slots of
// the real positions. Its own instantiation of the flash core: FlashAttentionPolicy's code is untouched.
struct FlashAttentionFloorPolicy : FlashAttentionPolicy {
    __device__ static __forceinline__ unsigned count(const Args& v, unsigned wg) {
        unsigned nw;
        return visible_keys_floor(v, row(v, wg), nw);
    }
    __device__ static __forceinline__ unsigned key(const Args& v, unsigned wg, unsigned j) {
        const unsigned r = row(v, wg);
        const std::uint64_t p = v.pos[r], span = p + 1U - v.key_floor;
        const unsigned nw = span < kWindow ? static_cast<unsigned>(span) : kWindow;
        if (j < nw) {
            unsigned slot = static_cast<unsigned>((p + 1U - nw) % v.capacity) + j;
            if (slot >= v.capacity) slot -= v.capacity;
            return slot;
        }
        return v.capacity + v.idx[static_cast<std::size_t>(r) * v.topk + j - nw];
    }
};
// both: the floored window keys and the q_b epilogue's BF16 query image (same bits as each form alone)
struct FlashAttentionFloorQ16Policy : FlashAttentionFloorPolicy {
    __device__ static __forceinline__ const std::uint16_t* query_op(const Args& v, unsigned wg) {
        return FlashAttentionQ16Policy::query_op(v, wg);
    }
};
// the row's single partial (slot 0) through the unchanged combine
__global__ __launch_bounds__(32) void attention_flash_combine_kernel(AttentionView v) {
    attention_combine_head(v, blockIdx.y, blockIdx.x, threadIdx.x, 1U);
}
