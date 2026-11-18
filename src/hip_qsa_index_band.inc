// Qwen adapter of shared attn_core idx WMMA + band. Included inside qsa_flash,
// after block_score/fused_exact/ScoreGroup; no second scoring core lives here.
namespace idx = strix::attn_core::idx;

struct BandRow {
    const float* query = nullptr;
    const void* keys = nullptr;
    const uint2* ranges = nullptr;
    const float* kmax = nullptr;
    unsigned scored_blocks = 0, head_dim = kDim;
    bool packed = false;
};
struct BandState { int lo, hi; float delta; };


// Called by the 32 lanes of one wave (every lane returns the same state): lane l reads features 4l .. 4l + 3 of
// each head. The norm only feeds an upper bound (1.01 margin), so its summation order is free; the exponent range
// is order-free (min / max), equal to score_kernel's.
__device__ __forceinline__ BandState band_state(const BandRow& row) {
    const unsigned lane = threadIdx.x % 32U;
    unsigned lo = 0xffffffffU, hi = 0U, unsafe = 0U;
    float error = 0.0F;
    const float kmax = *row.kmax;
#pragma unroll
    for (unsigned h = 0; h < kHeads; ++h) {
        const float4 x = *reinterpret_cast<const float4*>(row.query + h * kDim + 4U * lane);
        magnitude_range(x.x, lo, hi); magnitude_range(x.y, lo, hi); magnitude_range(x.z, lo, hi); magnitude_range(x.w, lo, hi);
        unsafe |= !(fabsf(x.x) < idx::kF16Safe && fabsf(x.y) < idx::kF16Safe && fabsf(x.z) < idx::kF16Safe && fabsf(x.w) < idx::kF16Safe);
        float q2 = __builtin_fmaf(x.x, x.x, __builtin_fmaf(x.y, x.y, __builtin_fmaf(x.z, x.z, x.w * x.w)));
#pragma unroll
        for (unsigned d = 16U; d >= 1U; d >>= 1U) q2 += __shfl_xor(q2, static_cast<int>(d), 32);
        error += idx::f16_dot_error(sqrtf(q2), kmax);
    }
    lo = wave_min_u(lo); hi = wave_max_u(hi); unsafe = wave_max_u(unsafe);
    const float delta = 1.01F * error / sqrtf(static_cast<float>(row.head_dim));
    return {lo_exponent(lo), hi_exponent(hi), unsafe == 0U && delta < INFINITY ? delta : INFINITY};
}
// Pooled key b as the floats score_kernel holds (packed: BF16 bits widened, the identical float).
__device__ __forceinline__ void band_load_key(const void* keys, bool packed, unsigned b, float (&k)[kDim]) {
    if (packed) {
        const auto* key = reinterpret_cast<const uint4*>(static_cast<const std::uint16_t*>(keys) + static_cast<std::uint64_t>(b) * kDim);
#pragma unroll
        for (unsigned i = 0; i < kDim / 8U; ++i) {
            const uint4 x = key[i];
            const unsigned words[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) {
                k[8U * i + 2U * j] = __uint_as_float(words[j] << 16U);
                k[8U * i + 2U * j + 1U] = __uint_as_float(words[j] & 0xffff0000U);
            }
        }
    } else {
        const auto* key = reinterpret_cast<const float4*>(static_cast<const float*>(keys) + static_cast<std::uint64_t>(b) * kDim);
#pragma unroll
        for (unsigned i = 0; i < kDim / 4U; ++i) {
            const float4 x = key[i];
            k[4U * i] = x.x; k[4U * i + 1U] = x.y; k[4U * i + 2U] = x.z; k[4U * i + 3U] = x.w;
        }
    }
}

// One bound per score call: Kmax = max |k_b|_2 over the call's scored prefix, read from THIS pool's keys (the
// key-range / norm arenas are process-wide and filled incrementally, so their entries below pool->valid may belong
// to another layer's pool during chunked prefill: never a bound). Infinity when any element is non-finite or
// >= kF16Safe (the f16 operand could overflow): the selection then rescores every block exactly. One work-group,
// <= 8 MB read at 128K (~50 us). Deferred per-group selections of the same call read it before the next call's
// reduction (stream order; the select flush precedes the next layer's scoring); prefixes only grow within a call.
__global__ __launch_bounds__(1024) void key_norm_max_kernel(const void* keys, bool packed, unsigned blocks, float* bound) {
    __shared__ float slots[32];
    float m = 0.0F;
    for (unsigned b = threadIdx.x; b < blocks; b += blockDim.x) {
        float k[kDim];
        band_load_key(keys, packed, b, k);
        float n2 = 0.0F;
        bool safe = true;
#pragma unroll
        for (unsigned d = 0; d < kDim; ++d) { safe = safe && fabsf(k[d]) < idx::kF16Safe; n2 = __builtin_fmaf(k[d], k[d], n2); }
        m = safe ? fmaxf(m, sqrtf(n2)) : INFINITY;  // fmaxf keeps an earlier INFINITY
    }
    m = idx::group_reduce(m, slots, [](float x, float y) { return fmaxf(x, y); });
    if (threadIdx.x == 0U) *bound = m;
}
// Exact score of block b by one whole wave (every lane calls it with the same b; every lane returns the value),
// bitwise equal to score_kernel's block_score<Fused>(q, k, head_dim) for that block:
//   * Fused: fused_exact(row exponent range, range of b's 32-block score tile [32 (b / 32), + 32) clipped to the
//     LAUNCH's scored prefix - score_kernel's wave tile; lanes past the prefix reuse the last block's range there,
//     which changes no extremum);
//   * lane l computes leaf<Fused>(q, k, l) (the same four products / FMAs, contraction off);
//   * head_dot's tree is a[i] = leaf(i) + leaf(16 + i) (i < 16, both leaf groups), then a[l] + a[l + 8],
//     a[l] + a[l + 4], a[l] + a[l + 2], a[0] + a[1]: the shuffle-down steps 16, 8, 4, 2, 1 below form exactly these
//     sums (IEEE addition is commutative, so only the grouping matters, and it is the same);
//   * block_score's sum over heads in order h = 0..3 of fmaxf(dot, 0), then / sqrtf(head_dim).
// Registers: 4 key and 4 query values per lane (the per-thread form held 128 keys and spilled in the selector).
__device__ __forceinline__ float band_exact_score_wave(const BandRow& row, const BandState& state, unsigned b) {
#pragma clang fp contract(off)
    const unsigned lane = threadIdx.x % 32U, first = b & ~31U;
    const uint2 r = row.ranges[first + lane < row.scored_blocks ? first + lane : row.scored_blocks - 1U];
    const bool fused = fused_exact(state.lo, state.hi, lo_exponent(wave_min_u(r.x)), hi_exponent(wave_max_u(r.y)));
    float k[4];
    const auto base = static_cast<std::uint64_t>(b) * kDim + lane;
#pragma unroll
    for (unsigned j = 0; j < 4U; ++j)
        k[j] = row.packed ? __uint_as_float(static_cast<unsigned>(static_cast<const std::uint16_t*>(row.keys)[base + 32U * j]) << 16U)
                          : static_cast<const float*>(row.keys)[base + 32U * j];
    float total = 0.0F;
#pragma unroll
    for (unsigned h = 0; h < kHeads; ++h) {
        const float* q = row.query + h * kDim + lane;
        const float q0 = q[0], q1 = q[32], q2 = q[64], q3 = q[96];
        float x;
        if (fused) {
            const float p2 = q2 * k[2], p3 = q3 * k[3];
            x = __builtin_fmaf(q0, k[0], p2) + __builtin_fmaf(q1, k[1], p3);
        } else {
            const float p0 = q0 * k[0], p1 = q1 * k[1], p2 = q2 * k[2], p3 = q3 * k[3];
            x = (p0 + p2) + (p1 + p3);
        }
        x = x + __shfl_down(x, 16U, 32);
        x = x + __shfl_down(x, 8U, 32);
        x = x + __shfl_down(x, 4U, 32);
        x = x + __shfl_down(x, 2U, 32);
        x = x + __shfl_down(x, 1U, 32);
        total += fmaxf(__shfl(x, 0, 32), 0.0F);
    }
    return total / sqrtf(static_cast<float>(row.head_dim));
}
// Exact scores of every block below `blocks` (wave-cooperative), then a barrier.
__device__ __forceinline__ void band_rescore_all(float* scores, unsigned blocks, const BandRow& row, const BandState& state) {
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    for (unsigned b = wave; b < blocks; b += kSelectWaves) {
        const float v = band_exact_score_wave(row, state, b);
        if (lane == 0U) scores[b] = v;
    }
    __syncthreads();
}

template <unsigned Rows, bool Batch> struct BandScorePolicy {
    static constexpr idx::Operand kOp = idx::Operand::f16;
    struct Args {
        QsaRowsScoreViews<Rows> single;
        const void* keys;
        unsigned blocks, head_dim, maxima_stride;
        float* maxima;
        const ScoreGroup<Rows>* groups;
        bool packed;
    };
    struct Ctx { const QsaRowsScoreViews<Rows>* views; unsigned r0, blocks, stride; float* maxima; };
    struct State {};
    __device__ static bool begin(const Args& a, Ctx& c) {
        c.r0 = blockIdx.y * 32U;
        if constexpr (Batch) {
            const auto& g = a.groups[c.r0 / Rows];
            c.r0 %= Rows; c.views = &g.views; c.blocks = g.blocks; c.stride = g.maxima_stride; c.maxima = g.maxima;
        } else {
            c.views = &a.single; c.blocks = a.blocks; c.stride = a.maxima_stride; c.maxima = a.maxima;
        }
        return c.blocks != 0U;
    }
    __device__ static void stage_queries(const Args&, const Ctx& c, std::uint16_t* tile) {
        auto* out = reinterpret_cast<uint4*>(tile);
        for (unsigned e = threadIdx.x; e < idx::kQueries * idx::kPieces; e += idx::kThreads) {
            const unsigned v = e / idx::kPieces, piece = e % idx::kPieces, r = c.r0 + v / kHeads;
            uint4 x = make_uint4(0U, 0U, 0U, 0U);
            if (r < Rows && c.views->blocks[r]) {
                const float* q = c.views->index_query[r] + (v % kHeads) * kDim + piece * 8U;
                x = idx::pack8<kOp>(*reinterpret_cast<const float4*>(q), *reinterpret_cast<const float4*>(q + 4));
            }
            out[idx::piece_slot(v, piece)] = x;
        }
    }
    __device__ static void init(const Args&, const Ctx&, State&) {}
    __device__ static unsigned tiles(const Args&, const Ctx& c) { return (c.blocks + idx::kKeys - 1U) / idx::kKeys; }
    __device__ static bool live(const Args&, const Ctx&, unsigned) { return true; }
    __device__ static uint4 key_piece(const Args& a, const Ctx& c, unsigned kt, unsigned key, unsigned piece) {
        const unsigned b = kt * idx::kKeys + key;
        if (b >= c.blocks) return make_uint4(0U, 0U, 0U, 0U);
        const auto at = static_cast<std::uint64_t>(b) * kDim + piece * 8U;
        if (a.packed) {  // 8 BF16 values = one 16-byte load, widened exactly to float, then rounded to FP16
            const uint4 w = *reinterpret_cast<const uint4*>(static_cast<const std::uint16_t*>(a.keys) + at);
            const auto lo16 = [](unsigned v) { return __uint_as_float(v << 16U); };
            const auto hi16 = [](unsigned v) { return __uint_as_float(v & 0xffff0000U); };
            return idx::pack8<kOp>(make_float4(lo16(w.x), hi16(w.x), lo16(w.y), hi16(w.y)),
                make_float4(lo16(w.z), hi16(w.z), lo16(w.w), hi16(w.w)));
        }
        const auto* x = reinterpret_cast<const float4*>(static_cast<const float*>(a.keys) + at);
        return idx::pack8<kOp>(x[0], x[1]);
    }
    __device__ static void tile_begin(const Args&, const Ctx&, State&, unsigned) {}
    __device__ static void fragment(const Args& a, const Ctx& c, State&, unsigned kt, unsigned kf, const idx::Acc (&acc)[idx::kFrags]) {
        const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
        const unsigned b = kt * idx::kKeys + kf * 16U + lane % 16U;
#pragma unroll
        for (unsigned f = 0; f < idx::kFrags; ++f) {
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) {
                const unsigned r = c.r0 + wave * 8U + f * 4U + j;
                float s = fmaxf(acc[f][2U * j], 0.0F) + fmaxf(acc[f][2U * j + 1U], 0.0F);
                s += __shfl_xor(s, 16, 32);
                const float score = s / sqrtf(static_cast<float>(a.head_dim));
                const unsigned blocks = r < Rows ? c.views->blocks[r] : 0U;
                if (lane < 16U && b < blocks) c.views->block_scores[r][b] = score;
                float m = b < blocks ? score : -INFINITY;
                m = fmaxf(m, __shfl_xor(m, 1, 32)); m = fmaxf(m, __shfl_xor(m, 2, 32));
                if (c.maxima && lane < 16U && (lane & 3U) == 0U && b < blocks)
                    c.maxima[static_cast<std::uint64_t>(r) * c.stride + b / kGroup] = m;
            }
        }
    }
    __device__ static void tile_end(const Args&, const Ctx&, State&, unsigned) {}
};
template <unsigned Rows, bool Batch>
void launch_index_band(QsaRowsScoreViews<Rows> views, const void* keys, bool packed, unsigned blocks,
    unsigned head_dim, float* maxima, unsigned stride, const ScoreGroup<Rows>* groups, unsigned count, hipStream_t stream) {
    using P = BandScorePolicy<Rows, Batch>;
    idx::launch<P>({views, keys, blocks, head_dim, stride, maxima, groups, packed},
        idx::wmma_split((blocks + idx::kKeys - 1U) / idx::kKeys), ((Rows + 31U) / 32U) * count, stream);
}
