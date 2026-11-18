// qpf2k/attn (STRIX_QPF2K_QSA=1): exact indexer block scores of prefill groups with one lane per block.
// Included by hip_indexed_attention_launcher.hip after hip_qsa_prefill_index.inc (needs QsaRowsScoreViews).
//
// block_score_parked_kernel computes, per (row, block, head), lane l's partial (p0 + p2) + (p1 + p3) of the
// features l + 32j (pj = q * k, no contraction), then the xor reduce-scatter over the 32 lanes (distances
// 16, 8, 4, 2, 1; each step adds the partials of lanes L and L ^ D, one addition commutes), then lane i's
// relu sum over the heads from head 0 and the division. Here one lane owns one block (its 128 pooled key
// features in registers) and computes that whole tree itself, in the same pairing: leaf l is
// (p_l + p_{l+64}) + (p_{l+32} + p_{l+96}), then a_l = leaf_l + leaf_{l+16} (l < 16), a_l + a_{l+8} (l < 8),
// + a_{l+4}, + a_{l+2}, a_0 + a_1. The query of the row is wave-uniform (scalar loads); the relu sum and
// the division are the parked kernel's statements.
//
// Products of two BF16 values (the pooled keys and the normed index queries are BF16 values held in FP32)
// have at most 16 significant bits, so p = q * k is exact whenever it is a normal FP32 number; then
// fma(q_l, k_l, p_{l+64}) rounds exactly p_l + p_{l+64}, as the separate product and sum do. The kernel takes
// that form only when every product of the (row, block tile) is provably normal and finite: the sum of the
// largest exponents <= 125 and the sum of the smallest exponents of non-zero values >= -120 (subnormal values
// count as -127; key ranges from key_range_kernel, row ranges per CTA), and only when neither side holds an
// infinity or NaN (largest exponent 128), checked before the shortcut for an all-zero side (zero times a
// non-finite value is not zero). Otherwise it runs the separate products. Same bits either way.
// Stats (fixtures only): counts of fused / separate (row, block tile) iterations and of selection paths.
namespace qsa_flash {
constexpr unsigned kHeads = 4U, kDim = 128U;
constexpr unsigned kScoreRows = 16U, kScoreWaves = 8U; // engine geometry: rows per CTA, waves (32-block tiles) per CTA
// STRIX_QPF2K_SCORE_ROWS=32 / 64 / auto (stage-batch scorer only, unset 16; decoder injects auto): each CTA keeps its wave's 32 keys in
// registers across 32 / 64 query rows instead of 16. Every score is the same lane's block_score of the same row and key
// with the same guard (row range per row, key range per 32-block wave tile), so the bits do not depend on RowsPerCta.
// auto: 32 rows once a call scores more than 4096 blocks (deep chunks, where the keys of a group
// outgrow the cache), 16 below (measured: 32 rows lose 5-10 % on calls of <= 4096 blocks); 64 rows from
// 65536 (beyond the native context; tests/qsa_score_rows.hip, T8192: 65536 blocks -7 % vs 32 rows,
// 129024 / 262144 blocks -6 % / -8 %), unless STRIX_QSA_SCORE_ROWS64=0.
constexpr unsigned kScoreRowsAuto = 1U;
inline unsigned score_rows_mode() {
    static const unsigned rows = [] {
        const char* v = std::getenv("STRIX_QPF2K_SCORE_ROWS");
        if (v && std::strcmp(v, "auto") == 0) return kScoreRowsAuto;
        const int n = v ? std::atoi(v) : 16;
        return n == 32 || n == 64 ? static_cast<unsigned>(n) : kScoreRows;
    }();
    return qpf2k::suppressed(qpf2k::Item::score_rows) ? kScoreRows : rows;
}
inline unsigned score_rows(std::uint32_t max_blocks) {
    const unsigned mode = score_rows_mode();
    if (mode != kScoreRowsAuto) return mode;
    return qsa_auto_score_rows(max_blocks, qpf2k::long_context_item_enabled("STRIX_QSA_SCORE_ROWS64"));
}
// STRIX_QSA_SCORE_SPLIT_LOADS (default on; 0 = the previous merged-load form): score_kernel's SplitLoads (see
// head_dot). The requested value is startup-cached; shared baseline/master/model
// suppression is applied after the cache, as for the other exact organization items.
inline bool score_split_loads() {
    static const bool on = strix::envcfg::flag("STRIX_QSA_SCORE_SPLIT_LOADS", true);
    return on && !qpf2k::long_context_item_suppression("STRIX_QSA_SCORE_SPLIT_LOADS");
}
inline bool index_band() {
    // STRIX_QSA_INDEX_BAND (default off; 1 = f16 WMMA scores on attn_core + band-exact selection, bitwise-identical
    // selected set and logits, gate PASS 32K / 128K 11-16). Measured SLOWER today (rocprofv3 128K: scorer 2,967 ->
    // 2,742 ms, select 718 -> 1,635 ms, bound reduction +232 ms; QLONG-RESULT.md section 3), so not default-on yet.
    static const bool on = strix::envcfg::flag("STRIX_QSA_INDEX_BAND", false);
    if (!on) return false; // OFF does not inspect the device
    static const bool supported = [] {
        int device = 0;
        hipDeviceProp_t p{};
        if (hipGetDevice(&device) != hipSuccess || hipGetDeviceProperties(&p, device) != hipSuccess)
            throw std::runtime_error("QSA index band: cannot inspect WMMA device");
        const std::string_view arch(p.gcnArchName);
        // The shared WMMA core implements these wave32 targets only.
        return p.warpSize == 32 && (arch.starts_with("gfx1151") || arch.starts_with("gfx1150") ||
            arch.starts_with("gfx1100") || arch.starts_with("gfx1101") || arch.starts_with("gfx1102"));
    }();
    return supported; // startup-cached; restart between arms/devices
}
constexpr unsigned kGroup = 4U; // blocks per selection group (score_kernel maxima, select_kernel)
typedef __attribute__((address_space(4))) const float* uniform_ptr;

// Exponent range of non-zero values: lo / hi accumulate the smallest / largest magnitude bits shifted left by
// one (sign dropped; a zero becomes 0xffffffff for the minimum and 0 for the maximum). exponent_of turns them
// into exponents (field 0, i.e. subnormal, -127; field 255, i.e. non-finite, 128; an all-zero set gives lo >
// hi).
__device__ __forceinline__ void magnitude_range(float x, unsigned& lo, unsigned& hi) {
    const unsigned m = __float_as_uint(x) << 1U;
    hi = max(hi, m);
    lo = min(lo, m - 1U);
}
__device__ __forceinline__ int exponent_of(unsigned shifted) { return static_cast<int>(shifted >> 24U) - 127; }
// lo exponent from the minimum (m - 1 of a non-zero m keeps its exponent field unless m is a power of two
// times 2: then the field can drop by one, which only makes the guard stricter); all zero -> 1 << 20.
__device__ __forceinline__ int lo_exponent(unsigned lo) { return lo == 0xffffffffU ? (1 << 20) : exponent_of(lo); }
__device__ __forceinline__ int hi_exponent(unsigned hi) { return hi == 0U ? -(1 << 20) : exponent_of(hi); }
__device__ __forceinline__ unsigned wave_min_u(unsigned v) {
#pragma unroll
    for (unsigned d = 16U; d > 0U; d >>= 1U) v = min(v, static_cast<unsigned>(strix::qwen_dpp::xor32(static_cast<int>(v), d)));
    return v;
}
__device__ __forceinline__ unsigned wave_max_u(unsigned v) {
#pragma unroll
    for (unsigned d = 16U; d > 0U; d >>= 1U) v = max(v, static_cast<unsigned>(strix::qwen_dpp::xor32(static_cast<int>(v), d)));
    return v;
}
// Per pooled block (first .. first + count - 1): its magnitude range (lo, hi) for score_kernel's guard. Launched
// after every block_pool_kernel launch over the same blocks.
template <bool Batch = false>
__global__ __launch_bounds__(256) void key_range_kernel(const float* pooled_keys, std::uint32_t first, std::uint32_t count, uint2* ranges,
    const uint2* bounds = nullptr, std::uint32_t groups = 0U, float* key_norm = nullptr) {
    auto i = static_cast<std::uint32_t>(blockIdx.x) * 256U + threadIdx.x;
    if (i >= count) return;
    if constexpr (Batch) {
        const auto absolute = first + i;
        const auto group = qsa_prefill_block_group(bounds, groups, absolute);
        if (group == groups) return;
        const auto range = bounds[group];
        if (absolute < range.x) return;
        i = absolute - range.x;
        first = range.x; count = range.y;
    }
    if (i >= count) return;
    const auto* key = reinterpret_cast<const float4*>(pooled_keys + static_cast<std::uint64_t>(first + i) * kDim);
    unsigned lo = 0xffffffffU, hi = 0U;
    float norm2 = 0.0F;
    bool safe = true;
#pragma unroll 8
    for (unsigned j = 0; j < kDim / 4U; ++j) {
        const float4 x = key[j];
        magnitude_range(x.x, lo, hi); magnitude_range(x.y, lo, hi); magnitude_range(x.z, lo, hi); magnitude_range(x.w, lo, hi);
        if (key_norm) {
            safe = safe && fabsf(x.x) < strix::attn_core::idx::kF16Safe && fabsf(x.y) < strix::attn_core::idx::kF16Safe &&
                fabsf(x.z) < strix::attn_core::idx::kF16Safe && fabsf(x.w) < strix::attn_core::idx::kF16Safe;
            norm2 = __builtin_fmaf(x.x, x.x, norm2); norm2 = __builtin_fmaf(x.y, x.y, norm2);
            norm2 = __builtin_fmaf(x.z, x.z, norm2); norm2 = __builtin_fmaf(x.w, x.w, norm2);
        }
    }
    if (key_norm) key_norm[first + i] = safe ? sqrtf(norm2) : INFINITY;
    ranges[first + i] = make_uint2(lo, hi);
}

template <bool Fused, class Q>
__device__ __forceinline__ float leaf(Q q, const float (&k)[kDim], unsigned l) {
#pragma clang fp contract(off)
    if constexpr (Fused) {
        const float p2 = q[l + 64U] * k[l + 64U], p3 = q[l + 96U] * k[l + 96U];
        return __builtin_fmaf(q[l], k[l], p2) + __builtin_fmaf(q[l + 32U], k[l + 32U], p3);
    } else {
        const float p0 = q[l] * k[l], p1 = q[l + 32U] * k[l + 32U], p2 = q[l + 64U] * k[l + 64U], p3 = q[l + 96U] * k[l + 96U];
        return (p0 + p2) + (p1 + p3);
    }
}
// The tree in leaf groups of 8 (l0 .. l0 + 7 need the query features l0 + 32j .. + 7: four 8-float scalar loads),
// in the order 0, 16 (a_0..7), 8, 24 (a_8..15), so at most two groups of query values are live in SGPRs; the
// scheduling fences keep the compiler from hoisting every query load of the row ahead of the arithmetic.
template <bool Fused, class Q, bool Fence = true>
__device__ __forceinline__ void leaf_pair_group(Q q, const float (&k)[kDim], unsigned l0, float (&a)[16], unsigned at) {
#pragma clang fp contract(off)
    float x[8];
#pragma unroll
    for (unsigned i = 0; i < 8U; ++i) x[i] = leaf<Fused, Q>(q, k, l0 + i);
    if constexpr (Fence) __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (unsigned i = 0; i < 8U; ++i) a[at + i] = x[i] + leaf<Fused, Q>(q, k, l0 + 16U + i);
    if constexpr (Fence) __builtin_amdgcn_sched_barrier(0);
}
// SplitLoads: an opaque copy of the wave-uniform query pointer before the second leaf group and before every head,
// so the scalar-load combiner cannot merge the 8-float query loads of neighbouring groups / heads into 16-float
// loads (s_load_b512). Merged, the fused form keeps all 128 query values of a head live (more than the SGPR file:
// spills to VGPR lanes, ~64 v_writelane/v_readlane per row and block tile on gfx1151). Only the SGPR allocation and
// the load widths change; every product, sum and their order are the same, so the bits are the same.
template <class Q> __device__ __forceinline__ Q opaque_uniform(Q q) {
    if constexpr (std::is_same_v<Q, uniform_ptr>) {
        unsigned zero; // an SGPR zero the compiler cannot see through: the next loads are not provably adjacent
        asm volatile("s_mov_b32 %0, 0" : "=s"(zero));
        return q + zero;
    } else {
        return q;
    }
}
template <bool Fused, class Q, bool Fence = true, bool SplitLoads = false>
__device__ __forceinline__ float head_dot(Q q, const float (&k)[kDim]) {
#pragma clang fp contract(off)
    float a[16];
    leaf_pair_group<Fused, Q, Fence>(q, k, 0U, a, 0U);
    leaf_pair_group<Fused, Q, Fence>(SplitLoads ? opaque_uniform(q) : q, k, 8U, a, 8U);
#pragma unroll
    for (unsigned l = 0; l < 8U; ++l) a[l] = a[l] + a[l + 8U];
#pragma unroll
    for (unsigned l = 0; l < 4U; ++l) a[l] = a[l] + a[l + 4U];
    a[0] = a[0] + a[2]; a[1] = a[1] + a[3];
    return a[0] + a[1];
}
template <bool Fused, class Q, bool Fence = true, bool SplitLoads = false>
__device__ __forceinline__ float block_score(Q q, const float (&k)[kDim], std::uint32_t head_dim) {
#pragma clang fp contract(off)
    float total = 0.0F;
#pragma unroll
    for (unsigned h = 0; h < kHeads; ++h)
        total += fmaxf(head_dot<Fused, Q, Fence, SplitLoads>(SplitLoads ? opaque_uniform(q + h * kDim) : q + h * kDim, k), 0.0F);
    return total / sqrtf(static_cast<float>(head_dim));
}

// grid (ceil(ceil(scored_blocks / 32) / W), Rows / RowsPerCta), block W * 32. Wave w of CTA x owns blocks
// 32 (x W + w) + lane; the CTA's rows are [y RowsPerCta, + RowsPerCta). Writes exactly the parked kernel's
// outputs (block_scores[r][b] for b < blocks[r]).
// maxima (optional): row r's maximum score of every 4-block group g (blocks 4g .. 4g + 3 below blocks[r]) at
// maxima[r * maxima_stride + g], for select_kernel.
// The fused-form guard of a (row, block tile): exponent ranges of the row's query and the tile's keys.
__device__ __forceinline__ bool fused_exact(int row_lo, int row_hi, int key_lo, int key_hi) {
    if (row_hi >= 128 || key_hi >= 128) return false;        // an infinity or NaN on either side
    if (row_lo > row_hi || key_lo > key_hi) return true;      // one side all zero (and both finite): zero products
    return row_hi + key_hi <= 125 && row_lo + key_lo >= -120; // every product normal and finite
}
enum Stat : unsigned { kFused, kSeparate, kSelectCandidates, kSelectFewMaxima, kSelectCapacity, kSelectDense, kCandidateGroups, kCandidateMax, kBandSize, kBandOverflow, kStats };
// QueryLds: the CTA stages its rows' queries in LDS (32 KB for 16 rows) and the waves read them as wave-uniform
// LDS broadcasts instead of scalar loads (the same values, so the same arithmetic).
template <unsigned Rows> struct ScoreGroup {
    QsaRowsScoreViews<Rows> views;
    std::uint32_t blocks, maxima_stride;
    float* maxima;
};
// Packed: pooled_keys holds the pool's packed BF16 copy (u16 per feature, block_pool_kernel's packed_keys); each key is
// widened back to the identical float, so every product, tree and guard is unchanged and the key reads halve.
template <unsigned Rows, unsigned RowsPerCta, unsigned W, bool Stats = false, bool QueryLds = false, bool Fence = true, unsigned RowUnroll = 1U, bool Batch = false,
    bool Packed = false, bool SplitLoads = false>
__global__ __launch_bounds__(W * 32U) void score_kernel(QsaRowsScoreViews<Rows> single, const void* pooled_keys,
    std::uint32_t scored_blocks, std::uint32_t head_dim, float* maxima, std::uint32_t maxima_stride, const uint2* key_ranges,
    std::uint32_t* stats = nullptr, const ScoreGroup<Rows>* groups = nullptr) {
    const auto& views = [&]() -> const QsaRowsScoreViews<Rows>& {
        if constexpr (Batch) return groups[blockIdx.z].views;
        else return single;
    }();
    if constexpr (Batch) {
        const auto& group = groups[blockIdx.z];
        scored_blocks = group.blocks;
        maxima = group.maxima; maxima_stride = group.maxima_stride;
        // Uniform CTA exit, before the row-range barriers; later groups may have a larger grid.
        if (blockIdx.x >= ((scored_blocks + 31U) / 32U + W - 1U) / W) return;
    }
    static_assert(Rows % RowsPerCta == 0U);
    __shared__ int row_lo[RowsPerCta], row_hi[RowsPerCta];
    __shared__ __align__(16) float q_lds[QueryLds ? RowsPerCta : 1U][QueryLds ? kHeads * kDim : 1U];
    const unsigned lane = threadIdx.x % 32U, w = threadIdx.x / 32U;
    const unsigned r0 = blockIdx.y * RowsPerCta;
    if constexpr (QueryLds) {
        for (unsigned i = threadIdx.x; i < RowsPerCta * kHeads * kDim / 4U; i += W * 32U) {
            const unsigned rr = i / (kHeads * kDim / 4U), j = i % (kHeads * kDim / 4U);
            *reinterpret_cast<float4*>(&q_lds[rr][4U * j]) = reinterpret_cast<const float4*>(views.index_query[r0 + rr])[j];
        }
    }
    // Row exponent ranges, once per CTA: wave w takes rows w, w + W, ...
    for (unsigned rr = w; rr < RowsPerCta; rr += W) {
        const float* q = views.index_query[r0 + rr];
        unsigned lo = 0xffffffffU, hi = 0U;
#pragma unroll
        for (unsigned i = 0; i < kHeads * kDim / 128U; ++i) {
            const float4 x = *reinterpret_cast<const float4*>(q + 128U * i + 4U * lane);
            magnitude_range(x.x, lo, hi); magnitude_range(x.y, lo, hi); magnitude_range(x.z, lo, hi); magnitude_range(x.w, lo, hi);
        }
        lo = wave_min_u(lo); hi = wave_max_u(hi);
        if (lane == 0U) { row_lo[rr] = lo_exponent(lo); row_hi[rr] = hi_exponent(hi); }
    }
    __syncthreads();
    const unsigned base = (blockIdx.x * W + w) * 32U;
    if (base >= scored_blocks) return;
    const unsigned b = base + lane;
    float k[kDim];
    if constexpr (Packed) {
        const auto* key = reinterpret_cast<const uint4*>(static_cast<const std::uint16_t*>(pooled_keys) +
            static_cast<std::uint64_t>(b < scored_blocks ? b : scored_blocks - 1U) * kDim);
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
        const auto* key = reinterpret_cast<const float4*>(static_cast<const float*>(pooled_keys) +
            static_cast<std::uint64_t>(b < scored_blocks ? b : scored_blocks - 1U) * kDim);
#pragma unroll
        for (unsigned i = 0; i < kDim / 4U; ++i) {
            const float4 x = key[i];
            k[4U * i] = x.x; k[4U * i + 1U] = x.y; k[4U * i + 2U] = x.z; k[4U * i + 3U] = x.w;
        }
    }
    const uint2 range = key_ranges[b < scored_blocks ? b : scored_blocks - 1U];
    const int klo = lo_exponent(wave_min_u(range.x)), khi = hi_exponent(wave_max_u(range.y));
#pragma unroll RowUnroll
    for (unsigned rr = 0; rr < RowsPerCta; ++rr) {
        const unsigned r = r0 + rr;
        const unsigned blocks = views.blocks[r];
        if (base >= blocks) continue;
        const bool fused = fused_exact(row_lo[rr], row_hi[rr], klo, khi);
        if constexpr (Stats) if (lane == 0U) atomicAdd(&stats[fused ? kFused : kSeparate], 1U);
        float score;
        if constexpr (QueryLds) {
            const float* q = &q_lds[rr][0];
            score = fused ? block_score<true, const float*, Fence>(q, k, head_dim) : block_score<false, const float*, Fence>(q, k, head_dim);
        } else {
            const auto q = reinterpret_cast<uniform_ptr>(reinterpret_cast<std::uintptr_t>(views.index_query[r]));
            score = fused ? block_score<true, uniform_ptr, true, SplitLoads>(q, k, head_dim) : block_score<false>(q, k, head_dim);
        }
        if (b < blocks) views.block_scores[r][b] = score;
        if (maxima) {
            float m = b < blocks ? score : -INFINITY;
            m = fmaxf(m, strix::qwen_dpp::xor32<1>(m));
            m = fmaxf(m, strix::qwen_dpp::xor32<2>(m));
            if ((lane & 3U) == 0U && b < blocks) maxima[static_cast<std::uint64_t>(r) * maxima_stride + b / kGroup] = m;
        }
    }
}

#include "hip_qsa_index_band.inc"

// Selection of a prefill row (select_rows_kernel's outputs) from the group maxima. select_blocks_parallel_body
// emits the set of the block_topk blocks that come first in the order (select_key descending, block
// ascending) among the blocks whose score is > -inf, in ascending block order (each block's 4 tokens), then
// the row's partial-block tail tokens, and leaves -inf at the selected scores (block 0 too on the zero fill).
// Every selected block b lies in a group whose maximum M has select_key(M) >= the block_topk-th largest key
// among the groups' maxima: those block_topk groups hold block_topk distinct blocks with keys >= it, so the
// block_topk-th largest score key is >= it, and b's group maximum is >= b's score. So this kernel finds that
// key with the same radix passes over the maxima, gathers the groups at or above it (ascending, their 4
// scores each) into LDS, and runs the radix passes and the ordered compaction over those candidates only:
// the same set, in the same order, and the same -inf writes. Rows with fewer than block_topk finite maxima, or
// more candidate groups than fit, run select_blocks_parallel_body itself.
constexpr unsigned kCandidates = 6144U;
template <bool Band> struct SelectBandStorage {};
template <> struct SelectBandStorage<true> { BandState state; float slots[kSelectWaves]; };
template <bool Stats, bool Band = false>
__device__ __forceinline__ void select_row_body(float* const scores, std::uint32_t* const selected, const std::uint32_t visible,
    const float* const row_max, std::uint32_t compress_ratio, std::uint32_t block_topk, std::uint32_t* stats, bool preserve_scores = false, BandRow band = {}) {
    const auto tid = static_cast<std::uint32_t>(threadIdx.x);
    const auto complete = visible / compress_ratio;
    if (complete <= block_topk) {
        if constexpr (Stats) if (tid == 0U) atomicAdd(&stats[kSelectDense], 1U);
        for (std::uint32_t index = tid; index < visible; index += kSelectThreads) selected[index] = index;
        return;
    }
    // Band identity: exact selected set/order; non-selected score values outside
    // the band remain approximate. Union-cap consumers request preserve_scores,
    // so selected values are restored exactly before they leave this selector.
    [[maybe_unused]] __shared__ SelectBandStorage<Band> band_storage;
    // Rows scored by the VALU kernel (no band query) take the incumbent path inside the band instantiation: one
    // copy of the selector's LDS per kernel.
    [[maybe_unused]] const bool band_on = Band && band.query != nullptr;
    if constexpr (Band) if (band_on) {
        if (tid < 32U) {  // wave 0 cooperates (band_state is a wave function)
            const BandState s = band_state(band);
            if (tid == 0U) band_storage.state = s;
        }
        __syncthreads();
        if (!(band_storage.state.delta < INFINITY)) {
            band_rescore_all(scores, complete, band, band_storage.state);
            select_blocks_parallel_body(scores, complete, block_topk, compress_ratio, visible, selected, GraphSelectionArguments{}, preserve_scores);
            return;
        }
    }
    const auto groups = (complete + kGroup - 1U) / kGroup;
    __shared__ std::uint32_t histogram[2048];
    __shared__ std::uint32_t waves[kSelectWaves];
    __shared__ std::uint32_t found_digit, found_need, found_count;
    __shared__ float cand[kCandidates];
    __shared__ std::uint32_t cand_group[kCandidates / kGroup];
    // The radix passes of select_blocks_parallel_body over n values value(i): (prefix, mask) of the need-th
    // largest key; false when fewer than need values are > -inf.
    const auto radix = [&](auto value, std::uint32_t n, std::uint32_t& prefix, std::uint32_t& mask, std::uint32_t& need) {
        prefix = 0; mask = 0;
        for (unsigned pass = 0; pass < 3U; ++pass) {
            const unsigned shift = pass == 0 ? 21U : pass == 1 ? 10U : 0U;
            const auto digits = pass == 2 ? 1024U : 2048U;
            for (auto i = tid; i < 2048U; i += kSelectThreads) histogram[i] = 0;
            __syncthreads();
            for (std::uint32_t base = 0; base < n; base += kSelectThreads) {
                const auto i = base + tid;
                const float v = i < n ? value(i) : -INFINITY;
                const auto key = select_key(v);
                const bool counted = v > -INFINITY && (key & mask) == prefix;
                const auto digit = (key >> shift) & (digits - 1U);
                const auto votes = __ballot(counted);
                if (votes) {
                    const auto leader = static_cast<unsigned>(__builtin_ctzll(votes));
                    const auto lead = __shfl(digit, static_cast<int>(leader), 32);
                    const auto same = __ballot(counted && digit == lead);
                    if (counted && digit != lead) atomicAdd(&histogram[digit], 1U);
                    else if (tid % 32U == leader) atomicAdd(&histogram[lead], static_cast<std::uint32_t>(__popcll(same)));
                }
            }
            __syncthreads();
            const auto high = 2U * tid + 1U < digits ? histogram[2U * tid + 1U] : 0U;
            const auto low = 2U * tid < digits ? histogram[2U * tid] : 0U;
            std::uint32_t counted_total = 0;
            const auto above = [&] { const auto inclusive = select_scan(high + low, waves, counted_total); return counted_total - inclusive; }();
            if (pass == 0 && counted_total <= need) return counted_total == need; // prefix = mask = 0: every counted value
            if (above < need && need <= above + high) { found_digit = 2U * tid + 1U; found_need = need - above; found_count = high; }
            else if (above + high < need && need <= above + high + low) { found_digit = 2U * tid; found_need = need - above - high; found_count = low; }
            __syncthreads();
            prefix |= found_digit << shift; mask |= (digits - 1U) << shift; need = found_need;
            if (found_count == need) break;
        }
        return true;
    };
    std::uint32_t prefix = 0, mask = 0, need = block_topk;
    const bool enough = radix([&](std::uint32_t g) { return row_max[g]; }, groups, prefix, mask, need);
    float group_cut = -INFINITY;
    if constexpr (Band) if (band_on) {
        float m = INFINITY;
        if (enough) for (unsigned g = tid; g < groups; g += kSelectThreads)
            if (row_max[g] > -INFINITY && (select_key(row_max[g]) & mask) == prefix) m = fminf(m, row_max[g]);
        // Radix may stop at a whole bucket: its minimum is the actual kth value.
        group_cut = idx::group_reduce(m, band_storage.slots, [](float x, float y) { return fminf(x, y); }) - 2.0F * band_storage.state.delta;
    }
    bool exact = enough;
    // Candidate groups: maxima > -inf with (key & mask) >= prefix, compacted in ascending group order.
    std::uint32_t taken = 0;
    for (std::uint32_t base = 0; exact && base < groups; base += kSelectThreads) {
        const auto g = base + tid;
        const float m = g < groups ? row_max[g] : -INFINITY;
        const bool keep = m > -INFINITY && (band_on ? m >= group_cut : (select_key(m) & mask) >= prefix);
        std::uint32_t tile = 0;
        const auto at = taken + select_scan(keep ? 1U : 0U, waves, tile) - (keep ? 1U : 0U);
        if (keep && at < kCandidates / kGroup) cand_group[at] = g;
        taken += tile;
        if (taken > kCandidates / kGroup) exact = false; // uniform: tile is the workgroup total
    }
    if constexpr (Stats) if (tid == 0U) {
        atomicAdd(&stats[!enough ? kSelectFewMaxima : !exact ? kSelectCapacity : kSelectCandidates], 1U);
        if (exact) { atomicAdd(&stats[kCandidateGroups], taken); atomicMax(&stats[kCandidateMax], taken); }
    }
    if (!exact) {
        if constexpr (Band) if (band_on) {
            // Fewer finite maxima need not mean fewer finite blocks. Restore all
            // scores before the incumbent fallback, including its zero padding.
            band_rescore_all(scores, complete, band, band_storage.state);
            if constexpr (Stats) if (tid == 0U && enough) atomicAdd(&stats[kBandOverflow], 1U);
        }
        select_blocks_parallel_body(scores, complete, block_topk, compress_ratio, visible, selected, GraphSelectionArguments{}, preserve_scores);
        return;
    }
    __syncthreads();
    const std::uint32_t n = taken * kGroup;
    for (std::uint32_t i = tid; i < n; i += kSelectThreads) {
        const auto block = cand_group[i / kGroup] * kGroup + i % kGroup;
        cand[i] = block < complete ? scores[block] : -INFINITY;
    }
    __syncthreads();
    if constexpr (Band) if (band_on) {
        need = block_topk;
        radix([&](std::uint32_t i) { return cand[i]; }, n, prefix, mask, need);
        float t = INFINITY;
        for (unsigned i = tid; i < n; i += kSelectThreads)
            if (cand[i] > -INFINITY && (select_key(cand[i]) & mask) == prefix) t = fminf(t, cand[i]);
        t = idx::group_reduce(t, band_storage.slots, [](float x, float y) { return fminf(x, y); });
        // Wave w tests candidates [32 c, + 32) for c = w, w + 32, ...; every band hit is rescored by the whole wave.
        for (unsigned c0 = (tid / 32U) * 32U; c0 < n; c0 += kSelectThreads) {
            const unsigned i = c0 + tid % 32U;
            const unsigned b = i < n ? cand_group[i / kGroup] * kGroup + i % kGroup : 0U;
            const bool hit = i < n && b < complete && idx::in_band(cand[i], t, band_storage.state.delta);
            for (unsigned long long m = __ballot(hit); m; m &= m - 1ULL) {
                const unsigned l = static_cast<unsigned>(__builtin_ctzll(m)), bi = __shfl(b, static_cast<int>(l), 32);
                const float v = band_exact_score_wave(band, band_storage.state, bi);
                if (tid % 32U == l) { cand[i] = v; scores[b] = v; }
                if constexpr (Stats) if (tid % 32U == 0U) atomicAdd(&stats[kBandSize], 1U);
            }
        }
        __syncthreads();
    }
    need = block_topk;
    radix([&](std::uint32_t i) { return cand[i]; }, n, prefix, mask, need); // >= block_topk counted: one per group
    std::uint32_t greater_before = 0, equal_before = 0;
    for (std::uint32_t base = 0; base < n; base += kSelectThreads) {
        const auto i = base + tid;
        const float value = i < n ? cand[i] : -INFINITY;
        const auto masked = select_key(value) & mask;
        const unsigned kind = !(value > -INFINITY) ? 0U : masked > prefix ? 1U : masked == prefix ? 2U : 0U;
        std::uint32_t tile = 0;
        const auto packed = select_scan(((kind == 1U ? 1U : 0U) << 16U) | (kind == 2U ? 1U : 0U), waves, tile);
        const auto g = greater_before + (packed >> 16U) - (kind == 1U ? 1U : 0U);
        const auto e = equal_before + (packed & 0xffffU) - (kind == 2U ? 1U : 0U);
        if (kind == 1U || (kind == 2U && e < need)) {
            const auto block = cand_group[i / kGroup] * kGroup + i % kGroup, slot = g + min(e, need);
            for (std::uint32_t offset = 0; offset < compress_ratio; ++offset)
                selected[slot * compress_ratio + offset] = block * compress_ratio + offset;
            if (!preserve_scores) scores[block] = -INFINITY;
        }
        greater_before += tile >> 16U; equal_before += tile & 0xffffU;
    }
    if constexpr (Band) if (band_on && preserve_scores) {
        // Score readers (FAST union cap) get the exact value of every selected block (outside the band the stored
        // value was the WMMA approximation): wave-cooperative rescore over the selected slots.
        __syncthreads();
        for (unsigned s = tid / 32U; s < block_topk; s += kSelectWaves) {
            const unsigned block = selected[s * compress_ratio] / compress_ratio;
            const float v = band_exact_score_wave(band, band_storage.state, block);
            if (tid % 32U == 0U) scores[block] = v;
        }
    }
    const auto complete_tokens = complete * compress_ratio, selected_complete = block_topk * compress_ratio;
    for (auto token = complete_tokens + tid; token < visible; token += kSelectThreads)
        selected[selected_complete + token - complete_tokens] = token;
}
template <unsigned Rows, bool Stats = false>
__global__ __launch_bounds__(kSelectThreads) void select_kernel(QsaRowsSelectViews<Rows> views, const float* maxima,
    std::uint32_t maxima_stride, std::uint32_t compress_ratio, std::uint32_t block_topk, std::uint32_t* stats = nullptr) {
    const auto r = blockIdx.x;
    select_row_body<Stats>(views.block_scores[r], views.selected[r], views.visible[r], maxima + static_cast<std::uint64_t>(r) * maxima_stride,
        compress_ratio, block_topk, stats, views.preserve_scores);
}
// STRIX_QPF2K_QSA bit 4: the selections of every prefill group of a call in one launch (one workgroup per row, the same
// select_row_body), run after the call's last group was scored and before the batched union / attend.
template <unsigned Rows, bool Stats = false>
__global__ __launch_bounds__(kSelectThreads) void select_band_kernel(QsaRowsSelectViews<Rows> views,
    QsaRowsScoreViews<Rows> queries, const float* maxima, unsigned stride, unsigned ratio, unsigned topk, BandRow band, std::uint32_t* stats = nullptr) {
    const unsigned r = blockIdx.x;
    band.query = queries.index_query[r];
    select_row_body<Stats, true>(views.block_scores[r], views.selected[r], views.visible[r],
        maxima + static_cast<std::uint64_t>(r) * stride, ratio, topk, stats, views.preserve_scores, band);
}
// Keep the OFF descriptor and kernel unchanged; band metadata is a parallel
// extension with the same pinned-slot lifetime, allocated only for the ON arm.
struct SelectRow { float* scores; std::uint32_t* selected; const float* maxima; std::uint32_t visible; std::uint32_t pad; };
__global__ __launch_bounds__(kSelectThreads) void select_batch_kernel(const SelectRow* rows, std::uint32_t compress_ratio, std::uint32_t block_topk) {
    const SelectRow row = rows[blockIdx.x];
    select_row_body<false>(row.scores, row.selected, row.visible, row.maxima, compress_ratio, block_topk, nullptr, row.pad != 0U);
}
__global__ __launch_bounds__(kSelectThreads) void select_band_batch_kernel(const SelectRow* rows, const BandRow* bands,
    std::uint32_t compress_ratio, std::uint32_t block_topk) {
    const SelectRow row = rows[blockIdx.x];
    const BandRow band = bands[blockIdx.x];
    select_row_body<false, true>(row.scores, row.selected, row.visible, row.maxima, compress_ratio, block_topk, nullptr, row.pad != 0U, band);
}
// One serial decode row (STRIX_QSA_DECODE_SELECT): the 4-block group maxima of a finished score row, with score_kernel's
// expression fmaxf(fmaxf(s0, s1), fmaxf(s2, s3)) and -inf past the row's blocks, then select_row_body over them.
__global__ __launch_bounds__(256) void row_maxima_kernel(const float* scores, std::uint32_t blocks, float* maxima) {
    const auto g = static_cast<std::uint32_t>(blockIdx.x) * 256U + threadIdx.x;
    if (g >= (blocks + kGroup - 1U) / kGroup) return;
    float s[kGroup];
#pragma unroll
    for (unsigned i = 0; i < kGroup; ++i) s[i] = kGroup * g + i < blocks ? scores[kGroup * g + i] : -INFINITY;
    maxima[g] = fmaxf(fmaxf(s[0], s[1]), fmaxf(s[2], s[3]));
}
__global__ __launch_bounds__(kSelectThreads) void select_one_kernel(float* scores, std::uint32_t* selected, std::uint32_t visible,
    const float* maxima, std::uint32_t compress_ratio, std::uint32_t block_topk) {
    select_row_body<false>(scores, selected, visible, maxima, compress_ratio, block_topk, nullptr);
}
// STRIX_QPF2K_QSA_TRACE=1 (diagnostic, independent of STRIX_QPF2K_QSA): order-independent 64-bit hashes of every prefill
// row's block scores before and after selection, its selected token list and its attention output, summed over the
// rows and calls of the process (acc[kind]), so an OFF and an ON process can be compared on every QSA intermediate.
template <unsigned Rows> struct TraceViews {
    const std::uint32_t* data[Rows];
    bool compact[Rows]{};
    std::uint32_t count[Rows];
    std::uint64_t position[Rows];
};
__device__ __forceinline__ std::uint64_t mix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31U);
}
// grid Rows, block 256: row r's hash = sum of mix64(value bits, index), folded with (kind, call, position).
template <unsigned Rows>
__global__ __launch_bounds__(256) void trace_kernel(TraceViews<Rows> views, std::uint32_t kind, std::uint32_t call, unsigned long long* acc) {
    __shared__ unsigned long long partial[256];
    const auto r = blockIdx.x, t = threadIdx.x;
    std::uint64_t h = 0;
    for (std::uint32_t i = t; i < views.count[r]; i += 256U) h += mix64((static_cast<std::uint64_t>(views.compact[r] ? static_cast<unsigned>(reinterpret_cast<const std::uint16_t*>(views.data[r])[i]) << 16U : views.data[r][i]) << 32U) | i);
    partial[t] = h;
    __syncthreads();
    for (unsigned d = 128U; d > 0U; d >>= 1U) { if (t < d) partial[t] += partial[t + d]; __syncthreads(); }
    if (t == 0U) atomicAdd(&acc[kind], static_cast<unsigned long long>(mix64(partial[0] ^ mix64(views.position[r] * 0x100000001b3ULL ^
        (static_cast<std::uint64_t>(call) << 40U) ^ kind) ^ views.count[r])));
}
// STRIX_QPF2K_QSA bit 2: qsa_index::norm_rope_group_kernel with one wave per (row, head) and no workgroup barriers.
// Lane l holds features l + 32j (j < dim / 32). The LDS tree reduction[f] += reduction[f + stride] (stride dim / 2 .. 1)
// is, for strides >= 32, the in-lane pairs (j, j + stride / 32) and, for 16 .. 1, lane l plus lane l + stride
// (__shfl_down): the same pairs in the same order (own + upper). mean, inv_rms, the weighted BF16 rounding and the
// rotation (partner feature f + half or f - half, half = rotary_dim / 2 = 32 a multiple of 32: the same lane) are
// norm_rope_group_kernel's statements. grid (ceil(waves / 8)), block 256; waves = Rows x (heads[0] + heads[1] + heads[2]).
// The squares are rounded before the tree (production stores them to LDS): no contraction here. Every lane gets the sum.
__device__ __forceinline__ float wave_square_sum(const float (&input)[8], unsigned J) {
#pragma clang fp contract(off)
    float sq[8];
#pragma unroll
    for (unsigned j = 0; j < 8U; ++j) sq[j] = input[j] * input[j];
    // strides head_dim / 2 .. 32 in lane
#pragma unroll
    for (unsigned d = 4U; d > 0U; d >>= 1U)
        if (d < J)
#pragma unroll
            for (unsigned j = 0; j < d; ++j) sq[j] += sq[j + d];
    float x = sq[0];
#pragma unroll
    for (unsigned d = 16U; d > 0U; d >>= 1U) x += __shfl_down(x, d, 32);
    return __shfl(x, 0, 32);
}
template <SmallWeightKind Kind, unsigned Rows, bool Batch = false>
__global__ __launch_bounds__(256) void norm_rope_wave_kernel(qsa_index::NormGroupViews<Rows> single, float epsilon, std::uint32_t rotary_dim,
    const qsa_index::NormGroupViews<Rows>* groups = nullptr) {
    const auto& views = [&]() -> const qsa_index::NormGroupViews<Rows>& {
        if constexpr (Batch) return groups[blockIdx.y];
        else return single;
    }();
    const unsigned lane = threadIdx.x % 32U;
    const unsigned heads_total = views.heads[0] + views.heads[1] + views.heads[2];
    const unsigned wave = blockIdx.x * 8U + threadIdx.x / 32U;
    if (wave >= Rows * heads_total) return;
    const unsigned row = wave / heads_total;
    unsigned head = wave % heads_total;
    unsigned s = 0;
    while (s < 2U && head >= views.heads[s]) head -= views.heads[s++];
    const auto head_dim = views.dim[s];
    float* values = views.values[s][row];
    const auto base = static_cast<std::uint64_t>(head) * views.stride[s];
    const unsigned J = head_dim / 32U; // 1 .. 8 (head_dim a power of two, 32 .. 256; checked by the caller)
    float input[8];
#pragma unroll
    for (unsigned j = 0; j < 8U; ++j) input[j] = j < J ? values[base + lane + 32U * j] : 0.0F;
    const float total = wave_square_sum(input, J);
    const float mean = total / static_cast<float>(head_dim) + epsilon;
    const float inv_rms = rsqrtf(mean);
    float src[8];
#pragma unroll
    for (unsigned j = 0; j < 8U; ++j) src[j] = j < J ? round_bf16(input[j] * inv_rms * load_small_weight<Kind>(views.weights[s], lane + 32U * j)) : 0.0F;
    const auto half = rotary_dim / 2U;
#pragma unroll
    for (unsigned j = 0; j < 8U; ++j) {
        if (j >= J) continue;
        const unsigned feature = lane + 32U * j;
        float result = src[j];
        if (feature < rotary_dim) {
            const auto pair = feature % half;
            const float2 cs = views.table[static_cast<std::uint64_t>(row) * half + pair];
            const float cosine = cs.x;
            const float sine = cs.y;
            const unsigned hj = half / 32U;
            const auto at = [&](unsigned k) { float v = 0.0F;
#pragma unroll
                for (unsigned i = 0; i < 8U; ++i) if (i == k) v = src[i];
                return v; };
            if (feature < half) result = src[j] * cosine - at(j + hj) * sine;
            else result = src[j] * cosine + at(j - hj) * sine;
        }
        values[base + feature] = round_bf16(result);
    }
}
// STRIX_QPF2K_QSA bit 3: qsa_index::rows_kernel (indexer q/k projections of a prefill group, BF16 weights) with RB = 16
// rows per workgroup instead of 4, so each weight read from L2 serves 4x more rows. The rows are staged in LDS in slices
// of 512 features (2 x 256: the chain step), one output per wave; lane l owns chains l + 32j (j < 8) of every row, each
// the ascending sum over the 256-feature steps k of the contracted a * w (the same expression as rows_kernel), and the
// same tree (in-lane strides 128 / 64 / 32, lane_down 16 .. 1) into lane 0. grid (ceil(outputs / W), Rows / RB),
// block W * 32; width % 512 == 0 and 16-byte aligned rows (checked by the caller).
template <unsigned Rows, unsigned RB, unsigned W>
__global__ __launch_bounds__(W * 32U) void index_rows_kernel(QsaRowsInputViews<Rows> views, std::uint32_t width) {
    static_assert(Rows % RB == 0U && RB % 4U == 0U);
    constexpr unsigned Slice = 512U;
    __shared__ __align__(16) float act[Slice * RB];
    const unsigned tid = threadIdx.x, lane = tid % 32U, wave = tid / 32U, r0 = blockIdx.y * RB;
    const unsigned outputs = views.first[views.count];
    const unsigned g = blockIdx.x * W + wave;
    std::uint32_t mrow = 0;
    const auto m = qsa_rows_matrix(views, min(g, outputs - 1U), mrow);
    const std::uint16_t* weight = static_cast<const std::uint16_t*>(views.weights[m]) + static_cast<std::uint64_t>(mrow) * width + lane;
    float sum[RB][8];
#pragma unroll
    for (unsigned r = 0; r < RB; ++r)
#pragma unroll
        for (unsigned j = 0; j < 8U; ++j) sum[r][j] = 0.0F;
    for (unsigned base = 0; base < width; base += Slice) {
        __syncthreads();
        // unit u = (row u % RB, features 8 (u / RB) .. + 7 of the slice)
        for (unsigned u = tid; u < RB * Slice / 8U; u += W * 32U) {
            const unsigned r = u % RB, f = u / RB * 8U;
            const uint4 x = *reinterpret_cast<const uint4*>(views.input[r0 + r] + base + f);
            const unsigned words[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
            for (unsigned j = 0; j < 4U; ++j) {
                act[(f + 2U * j) * RB + r] = __uint_as_float(words[j] << 16U);
                act[(f + 2U * j + 1U) * RB + r] = __uint_as_float(words[j] & 0xffff0000U);
            }
        }
        __syncthreads();
#pragma unroll
        for (unsigned kk = 0; kk < Slice / 256U; ++kk) {
#pragma unroll
            for (unsigned j = 0; j < 8U; ++j) {
                const unsigned f = kk * 256U + 32U * j + lane;
                const float w = bf16_to_float(weight[base + kk * 256U + 32U * j]);
                float a[RB];
#pragma unroll
                for (unsigned q = 0; q < RB / 4U; ++q) {
                    const float4 x = *reinterpret_cast<const float4*>(&act[f * RB + 4U * q]);
                    a[4U * q] = x.x; a[4U * q + 1U] = x.y; a[4U * q + 2U] = x.z; a[4U * q + 3U] = x.w;
                }
#pragma unroll
                for (unsigned r = 0; r < RB; ++r) sum[r][j] += a[r] * w;
            }
        }
    }
#pragma unroll
    for (unsigned r = 0; r < RB; ++r) {
        float (&v)[8] = sum[r];
#pragma unroll
        for (unsigned j = 0; j < 4U; ++j) v[j] = v[j] + v[j + 4U];
        v[0] = v[0] + v[2]; v[1] = v[1] + v[3];
        float x = v[0] + v[1];
        x = x + qsa_index::lane_down<16U>(x);
        x = x + qsa_index::lane_down<8U>(x);
        x = x + qsa_index::lane_down<4U>(x);
        x = x + qsa_index::lane_down<2U>(x);
        x = x + qsa_index::lane_down<1U>(x);
        if (lane == 0U && g < outputs) views.output[m][r0 + r][mrow] = round_bf16(x);
    }
}
} // namespace qsa_flash
