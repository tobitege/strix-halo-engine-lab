// FAST lever ds41-attn-f16wmma-v1 (STRIX_DS41_FAST_ATTN=1, default off; prefill only). NOT bit-exact: a new declared
// arithmetic, qualified behind the FAST profile. Included by src/hip_ds41_attention.hip inside its anonymous namespace.
//
// One block per (key chunk of 64, 16 heads, row), 8 waves, in two tiles of 32 keys:
// - Q: the row's 16 heads after RoPE (FP32), rounded once to f16. Keys: the dequantised K/V values in f16, compressed
//   entries times 2^-7 (exact), a window key times 2^-ek (ek > 0 only when its largest E8M0 scale is above 134); the
//   factor is undone in FP32 on its scores and folded into its probabilities, so nothing overflows.
// - QK^T on WMMA f16 x f16 -> f32 (the 512 dims split into four quarters over the waves, quarters summed in a fixed tree).
// - Softmax statistics in FP32 (scale, max, exp_f, row sums in a fixed lane tree), online over the two tiles with the
//   ordered rescale of acc; the probabilities rounded once to f16.
// - PV on WMMA f16 x f16 -> f32. The chunk's (m, l, acc) partial and the combine are today's (FP32, chunk order, sink,
//   1 / L, inverse RoPE, A8 of o).
// Keys sit in LDS as pairs (dword = key 2p and 2p + 1 at one dim), 16-byte vectors XOR-swizzled by the pair, so the QK
// operand (16 dims of one key per lane) and the PV operand (16 keys of one dim per lane) both read without conflicts.
constexpr unsigned kFastHeads = 16U;
constexpr int kFastPScale = 15;   // f16 probabilities carry 2^15 (see the PV comment in the kernel)
using fhalf2 = _Float16 __attribute__((ext_vector_type(2)));

// dword index of (pair p, dim d) in the key tile
__device__ __forceinline__ unsigned fast_kslot(unsigned p, unsigned d) { return strix::attn_core::pair_slot<kHeadDim>(p, d); }
__device__ __forceinline__ unsigned fast_pk_mul(unsigned x, unsigned f) {
    return __builtin_bit_cast(unsigned, __builtin_bit_cast(fhalf2, x) * __builtin_bit_cast(fhalf2, f));
}
// f16 bits of 2^e for e in [-14, 15], in both halves
__device__ __forceinline__ unsigned fast_pow2x2(int e) { const unsigned b = static_cast<unsigned>(e + 15) << 10U; return b | (b << 16U); }
// (byte d of a, byte d of b) as E4M3 -> f16 pair, times 2^-8 (exact, subnormals included)
__device__ __forceinline__ unsigned fast_e4m3x2(unsigned a, unsigned b, unsigned d) {
    const unsigned x = __builtin_amdgcn_perm(b, a, 0x0c000c00U | ((4U + d) << 16U) | d);
    return ((x & 0x007f007fU) << 7U) | ((x & 0x00800080U) << 8U);
}
// (nibble i of a, nibble i of b) as E2M1 -> f16 pair, times 2^-14 (exact)
__device__ __forceinline__ unsigned fast_e2m1x2(unsigned a, unsigned b, unsigned i) {
    const unsigned x = ((a >> (4U * i)) & 15U) | (((b >> (4U * i)) & 15U) << 16U);
    return ((x & 0x00070007U) << 9U) | ((x & 0x00080008U) << 12U);
}
// (byte k of a, byte k of b) -> their low (j = 0) or high (j = 1) E2M1 nibbles as an f16 pair times 2^-14 (exact)
__device__ __forceinline__ unsigned fast_e2m1x2_byte(unsigned a, unsigned b, unsigned k, unsigned j) {
    const unsigned x = __builtin_amdgcn_perm(b, a, 0x0c000c00U | ((4U + k) << 16U) | k);   // [a.k, 0, b.k, 0]
    return j == 0U ? ((x & 0x00070007U) << 9U) | ((x & 0x00080008U) << 12U) : ((x & 0x00700070U) << 5U) | ((x & 0x00800080U) << 8U);
}
// The thread's 32 dims (32 part ..) of keys a = 2p, b = 2p + 1 into the tile as pairs. ek: a window key's exponent
// shift. Stored values: window keys times 2^-ek, compressed keys times 2^-7 (kfac undoes it on the scores; the PV
// passes go by exponent band, see the kernel).
// The three key-pair kinds take separate branches (uniform across a pair's 16 threads; a wave holds two pairs).
__device__ __forceinline__ void fast_store_pair(unsigned* kt, unsigned p, unsigned part, const KeyRaw& ka, const KeyRaw& kb, bool wa,
                                                bool wb, bool va, bool vb, int eka, int ekb) {
    unsigned out[32];
    const std::uint32_t ca[8] = {ka.c0.x, ka.c0.y, ka.c0.z, ka.c0.w, ka.c1.x, ka.c1.y, ka.c1.z, ka.c1.w};
    const std::uint32_t cb[8] = {kb.c0.x, kb.c0.y, kb.c0.z, kb.c0.w, kb.c1.x, kb.c1.y, kb.c1.z, kb.c1.w};
    // compressed: per half of 16 dims, the E4M3 scale times 2^7 as f16 (exact: at most 448 x 128 = 57344)
    const auto cscale = [](const KeyRaw& k, unsigned hh) {
        const unsigned s = (k.sc >> (8U * hh)) & 0xFFU;
        return fast_pk_mul(((s & 0x7FU) << 7U) | ((s & 0x80U) << 8U), 0x7800U) & 0xFFFFU;   // (s x 2^-8) x 2^15
    };
    // window: 2^(b - 119 - ek) in one or two f16 powers of two (the E4M3 bit form is the value times 2^-8)
    const auto wscale = [](const KeyRaw& k, int ek, unsigned& g1, unsigned& g2) {
        const int e = static_cast<int>(k.sc & 0xFFU) - 119 - ek, e1 = e < -14 ? -14 : e > 15 ? 15 : e;
        const int r = e - e1, e2 = r < -14 ? -14 : r > 15 ? 15 : r;
        g1 = fast_pow2x2(e1) & 0xFFFFU; g2 = fast_pow2x2(e2) & 0xFFFFU;
    };
    if (!wa && !wb) {
#pragma unroll
        for (unsigned hh = 0; hh < 2; ++hh) {
            const unsigned f = cscale(ka, hh) | (cscale(kb, hh) << 16U);
#pragma unroll
            for (unsigned i = 16 * hh; i < 16 * hh + 16; i += 2) {   // dims i, i + 1: byte (i % 8) / 2 of word i / 8
                out[i] = fast_pk_mul(fast_e2m1x2_byte(ca[i / 8], cb[i / 8], (i % 8) / 2, 0U), f);
                out[i + 1] = fast_pk_mul(fast_e2m1x2_byte(ca[i / 8], cb[i / 8], (i % 8) / 2, 1U), f);
            }
        }
    } else if (wa && wb) {
        unsigned a1, a2, b1, b2;
        wscale(ka, eka, a1, a2);
        wscale(kb, ekb, b1, b2);
        const unsigned f1 = a1 | (b1 << 16U), f2 = a2 | (b2 << 16U);
#pragma unroll
        for (unsigned i = 0; i < 32; ++i) out[i] = fast_pk_mul(fast_pk_mul(fast_e4m3x2(ca[i / 4], cb[i / 4], i % 4), f1), f2);
    } else {   // a window key next to a compressed one (the window / selected boundary)
        unsigned a1 = 0, a2 = 0, b1 = 0, b2 = 0;
        if (wa) wscale(ka, eka, a1, a2);
        if (wb) wscale(kb, ekb, b1, b2);
#pragma unroll
        for (unsigned i = 0; i < 32; ++i) {
            const unsigned ga1 = wa ? a1 : cscale(ka, i / 16), ga2 = wa ? a2 : 0x3C00U, gb1 = wb ? b1 : cscale(kb, i / 16), gb2 = wb ? b2 : 0x3C00U;
            const unsigned xa = wa ? fast_e4m3x2(ca[i / 4], 0U, i % 4) : fast_e2m1x2(ca[i / 8], 0U, i % 8);
            const unsigned xb = wb ? fast_e4m3x2(cb[i / 4], 0U, i % 4) : fast_e2m1x2(cb[i / 8], 0U, i % 8);
            out[i] = fast_pk_mul(fast_pk_mul((xa & 0xFFFFU) | (xb << 16U), ga1 | (gb1 << 16U)), ga2 | (gb2 << 16U));
        }
    }
#pragma unroll
    for (unsigned i = 0; i < 32; ++i) {
        if (!va) out[i] &= 0xFFFF0000U;
        if (!vb) out[i] &= 0x0000FFFFU;
    }
#pragma unroll
    for (unsigned v4 = 0; v4 < 8; ++v4)
        *reinterpret_cast<uint4*>(kt + fast_kslot(p, 32U * part + 4U * v4)) = make_uint4(out[4 * v4], out[4 * v4 + 1], out[4 * v4 + 2], out[4 * v4 + 3]);
}

// Test-only ablations (benchmarks of where the time goes; results are then wrong): bit 0 no key loads, 1 no key
// dequantisation, 2 no QK products, 3 no PV products, 4 no Q loads.
__device__ unsigned g_fast_attn_ablate = 0U;

// Cpb chunks per block, one after the other: the next tile's key bytes are always in flight (also across a chunk
// boundary) and Q is staged once for all of them. The block's online softmax runs over all of its keys and writes ONE
// (m, l, acc) partial (slot blockIdx.x): the per-chunk partials and their combine reads were the largest cost (an
// ablation with no loads, dequantisation or products kept 2/3 of the time). The combine merges the blocks in order.
// H heads per block (16, or 32: each key is loaded and dequantised once for 32 heads; dequantisation was the next cost).
constexpr unsigned kFastChunk = kChunk, kFastMaxChunks = kMaxChunks;
__device__ __forceinline__ int fast_log2_ceil(float x) { return log2_ceil(x); }
__device__ __forceinline__ float fast_wave_max(float x) { return wave_max(x); }
struct FastAttentionPolicy {
    using Args = AttentionView;
    using Raw = KeyRaw;
    static constexpr unsigned kDim = kHeadDim, kChunk = kFastChunk;
    static constexpr unsigned kMaxChunks = kFastMaxChunks, kGroups = kAttnGroups;
    static constexpr int kProbabilityShift = kFastPScale;
    __device__ static __forceinline__ unsigned ablation() { return g_fast_attn_ablate; }
    __device__ static __forceinline__ float exp(float x) { return exp_f(x); }
    __device__ static __forceinline__ float wave_max(float x) { return fast_wave_max(x); }
    __device__ static __forceinline__ int log2_ceil(float x) { return fast_log2_ceil(x); }
    __device__ static __forceinline__ unsigned visible(const Args& v, unsigned r, unsigned& nw) { return visible_keys(v, r, nw); }
    __device__ static __forceinline__ unsigned max_visible(const Args& v) { return v.max_visible; }
    __device__ static __forceinline__ unsigned capacity(const Args& v) { return v.capacity; }
    __device__ static __forceinline__ unsigned heads(const Args& v) { return v.heads; }
    __device__ static __forceinline__ float* partial(const Args& v) { return v.partial; }
    __device__ static __forceinline__ std::uint32_t* tickets(const Args& v) { return v.tickets; }
    __device__ static __forceinline__ std::uint64_t position(const Args& v, unsigned r) { return v.pos[r]; }
    __device__ static __forceinline__ std::uint32_t index(const Args& v, unsigned r, unsigned j) { return v.idx[static_cast<std::size_t>(r) * v.topk + j]; }
    __device__ static __forceinline__ void query(const Args& v, unsigned r, unsigned h, unsigned lane, unsigned ablate, float (&q)[16]) {
        const float* src = v.q + (static_cast<std::size_t>(r) * v.heads + h) * kHeadDim + 16 * lane;
#pragma unroll
        for (unsigned i = 0; i < 16; ++i) q[i] = (ablate & 16U) != 0U ? 0.01F : src[i];
        rope512(q, v.rope[r], lane, false);
    }
    __device__ static __forceinline__ void load(const Args& v, unsigned j, unsigned end, unsigned nw, unsigned first_slot,
                                               unsigned part, unsigned index, unsigned ablate, Raw& k) {
        k = KeyRaw{};
        if (j >= end || (ablate & 1U) != 0U) return;
        if (j < nw) {
            unsigned slot = first_slot + j;
            if (slot >= v.capacity) slot -= v.capacity;
            const std::uint8_t* e = v.ring + static_cast<std::size_t>(slot) * kWindowEntryBytes;
            const uint4* cp = reinterpret_cast<const uint4*>(e + 32 * part);
            k.c0 = cp[0]; k.c1 = cp[1];
            k.sc = e[512 + part];
        } else {
            const std::uint8_t* e = v.kv_cache + static_cast<std::size_t>(index) * kKvEntryBytes;
            k.c0 = *reinterpret_cast<const uint4*>(e + 16 * part);
            k.sc = *reinterpret_cast<const std::uint16_t*>(e + 256 + 2 * part);
        }
    }
    __device__ static __forceinline__ void stage(unsigned* kt, unsigned pr, unsigned part, const Raw& ra, const Raw& rb,
                                                unsigned t0, unsigned nw, unsigned cnt, int* kexp, float* kfac, unsigned ablate) {
            const unsigned ka = t0 + 2U * pr, kb = ka + 1U;
            const bool wa = ka < nw, wb = kb < nw, va = 2U * pr < cnt, vb = 2U * pr + 1U < cnt;
            int ba = wa && va ? static_cast<int>(ra.sc & 0xFFU) : 0, bb = wb && vb ? static_cast<int>(rb.sc & 0xFFU) : 0;
            const auto nan_of = [](const KeyRaw& k, bool win) {
                if (!win) return ((k.sc & 0x7FU) == 0x7FU) || (((k.sc >> 8U) & 0x7FU) == 0x7FU);
                const std::uint32_t c[8] = {k.c0.x, k.c0.y, k.c0.z, k.c0.w, k.c1.x, k.c1.y, k.c1.z, k.c1.w};
                std::uint32_t any = 0;
#pragma unroll
                for (unsigned i = 0; i < 8; ++i) any |= ((c[i] & 0x7F7F7F7FU) + 0x01010101U) & 0x80808080U;
                return any != 0U || (k.sc & 0xFFU) == 0xFFU;
            };
            int na = va && nan_of(ra, wa) ? 1 : 0, nb = vb && nan_of(rb, wb) ? 1 : 0;
#pragma unroll
            for (unsigned d = 1; d < 16; d *= 2) {
                ba = max(ba, __shfl_xor(ba, static_cast<int>(d), 32)); bb = max(bb, __shfl_xor(bb, static_cast<int>(d), 32));
                na |= __shfl_xor(na, static_cast<int>(d), 32); nb |= __shfl_xor(nb, static_cast<int>(d), 32);
            }
            const int eka = ba > 134 ? ba - 134 : 0, ekb = bb > 134 ? bb - 134 : 0;
            if ((ablate & 2U) == 0U) fast_store_pair(kt, pr, part, ra, rb, wa, wb, va, vb, eka, ekb);
            if (part == 0U) {
                const int xa = wa ? eka : 7, xb = wb ? ekb : 7;   // the stored value is the key times 2^-x
                kexp[2U * pr] = va ? xa : 0; kexp[2U * pr + 1U] = vb ? xb : 0;
                kfac[2U * pr] = na != 0 ? __builtin_nanf("") : ldexpf(kAttnScale, xa);
                kfac[2U * pr + 1U] = nb != 0 ? __builtin_nanf("") : ldexpf(kAttnScale, xb);
            }
    }
    __device__ static __forceinline__ void combine(const Args& v, unsigned r, unsigned h, unsigned lane, unsigned blocks) {
        attention_combine_head(v, r, h, lane, blocks);
    }
};

template <unsigned Cpb, unsigned H>
__global__ __launch_bounds__(256) void attention_f16wmma_kernel(AttentionView v) {
    DS41_TIMED(5U);
    strix::attn_core::banded_flash_prefill<FastAttentionPolicy, Cpb, H>(v);
}

// Combine without tickets: the row's blocks (ceil(chunks / Cpb) partials) in order.
template <unsigned Cpb>
__global__ __launch_bounds__(32) void attention_f16wmma_combine_kernel(AttentionView v) {
    unsigned nw;
    const unsigned n = visible_keys(v, blockIdx.y, nw), chunks = (n + kChunk - 1) / kChunk;
    attention_combine_head(v, blockIdx.y, blockIdx.x, threadIdx.x, (chunks + Cpb - 1) / Cpb);
}
