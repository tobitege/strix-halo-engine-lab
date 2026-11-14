// Fast prefill profile, qsa component (included by hip_indexed_attention_launcher.hip; needs round_bf16).
//
// Sparse attention of a prefill group (Rows > 8 rows, head_dim 256) over the union of the selections of
// TR consecutive rows. union_kernel turns the rows' selected token lists into the ascending list of
// 4-token blocks any of them selected, each with a TR-bit row mask, and the multiplicity of tokens 0..3
// (which the selection may repeat). attend_kernel runs one tile of TR rows and one KV head over a slice of
// the union in 16-token chunks; a query column is a (row, head) pair, 16 columns per WMMA tile (TR rows
// times 16 / TR heads of that KV head). Scores S^T = K x Q^T and values O^T += V^T x P^T use
// v_wmma_f32_16x16x16_f16 with FP32 accumulation. K and V go from BF16 to FP16 (exact in the FP16 normal
// range, zero past the tile's visible prefix); Q is rounded once to FP16 after a power-of-two scale per
// (row, head) that puts its largest |q| in [2^13, 2^14), undone with 1 / sqrt(256) on the FP32 scores. A
// (row, token) pair outside the row's selection or causal prefix gets score -inf; the softmax is the
// online one in FP32 and base 2 (running maximum and sum per column, P rounded once to FP16 for the
// product, O rescaled only when a column's maximum grows). combine_kernel merges the slices (splits) of
// each (row, head), divides by the sum and applies the BF16 roundings and the BF16 sigmoid gate of
// attention_output_body; attend_kernel does it itself when there is one slice.
// Scratch (per row: its attention_scores buffer, heads x maximum_selected floats): slice partials
// [split][head][m, l, O[256]] of the row, then for the first row of each tile, at union_at, the
// tile's union: u32 [padded block count, 0, 0, 0, TR packed token-0..3 counts], uint2 (block, mask)[].
namespace qsa_fast {
typedef _Float16 f16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
constexpr unsigned kDim = 256U, kPitch = kDim + 8U, kChunk = 16U, kRangeBlocks = 8192U, kPartial = kDim + 2U;

template <unsigned Rows> struct Views {
    const float* q_gate[Rows];
    const std::uint32_t* selected[Rows];
    float* scratch[Rows];
    float* output[Rows];
    std::uint32_t count[Rows];
    std::uint32_t visible[Rows];
    bool compact_output = false;
    const float* block_scores[Rows]{}; // cap only: valid until the score-ring lease ends
    std::uint32_t score_blocks[Rows]{}; // dense rows were not scored; their score ties at -inf
    __device__ __forceinline__ RoundedStageImage image(unsigned r) const { return {output[r], compact_output}; }
};
// Floats of a row's scratch the fast path needs (union_at + union) for TR-row tiles.
inline std::uint64_t scratch_floats(unsigned tile_rows, unsigned splits, unsigned heads, unsigned maximum_selected) {
    const auto blocks = static_cast<std::uint64_t>(tile_rows) * (maximum_selected / 4U + 2U) + 4U;
    return static_cast<std::uint64_t>(splits) * heads * kPartial + 4U + tile_rows + 2U * blocks;
}

__device__ __forceinline__ void lds_barrier() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory"); }
// 16-byte global (not flat) load, so the loads in flight do not end at lds_barrier.
__device__ __forceinline__ u32x4 load16(const void* p) {
    return *reinterpret_cast<const __attribute__((address_space(1))) u32x4*>(reinterpret_cast<std::uintptr_t>(p));
}
// Two BF16 (one word) as two FP16: exact for values in the FP16 normal range (BF16 has fewer significand
// bits); the conversion truncates, so magnitudes beyond the FP16 range saturate at 65504.
__device__ __forceinline__ unsigned bf16x2_to_f16x2(unsigned w) {
    return __builtin_bit_cast(unsigned, __builtin_amdgcn_cvt_pkrtz(__uint_as_float(w << 16U), __uint_as_float(w & 0xffff0000U)));
}
// LDS row of feature d in the transposed V tile: rows 8p + j of one store instruction (p = 0..7) and rows
// 16n + c of one fragment read fall in different bank groups.
__device__ __forceinline__ unsigned v_row(unsigned d) { return d ^ ((d >> 3U) & 7U); }
__device__ __forceinline__ float swap16(float v) { // lane L ^ 16
    return __uint_as_float(__builtin_amdgcn_permlanex16(__float_as_uint(v), __float_as_uint(v), 0x76543210U, 0xfedcba98U, false, false));
}
// Exponent s with peak * 2^s in [2^13, 2^14) (0 for a zero or non-finite peak), 2^+-s normal.
__device__ __forceinline__ int q_exponent(float peak) {
    if (!(peak > 0.0F) || !isfinite(peak)) return 0;
    int e; (void)frexpf(peak, &e);
    return min(120, max(-120, 14 - e));
}
[[maybe_unused]] __device__ unsigned block_exclusive(unsigned v, unsigned* waves, unsigned& total) {
    const unsigned lane = threadIdx.x % 32U, wave = threadIdx.x / 32U;
    unsigned x = v;
    for (unsigned d = 1U; d < 32U; d <<= 1U) { const unsigned y = __shfl_up(x, d, 32); if (lane >= d) x += y; }
    if (lane == 31U) waves[wave] = x;
    __syncthreads();
    unsigned before = 0U;
    total = 0U;
    for (unsigned w = 0; w < blockDim.x / 32U; ++w) { if (w < wave) before += waves[w]; total += waves[w]; }
    __syncthreads();
    return before + x - v;
}

// First index of the ascending block list blk[0, n) whose block is >= b.
__device__ __forceinline__ unsigned lower_bound_block(const std::uint32_t* blk, unsigned n, unsigned b) {
    unsigned lo = 0U, hi = n;
    while (lo < hi) {
        const unsigned mid = (lo + hi) / 2U;
        if (blk[mid] < b) lo = mid + 1U; else hi = mid;
    }
    return lo;
}
// Merge form of the union (STRIX_QSA_UNION_MERGE, default on): the same (block, mask) pairs in the same order,
// with work bounded by the selections instead of the block range. The rows' selections are staged in LDS as block
// IDs (token / 4). Every union block has one owner: its first token in the lowest row that selected it. A block's
// place in the union is the number of owners of smaller blocks, sum over rows r' of the owners before
// lower_bound_r'(b), from per-row owner prefix counts. Needs non-decreasing selections of at most kMergeCap tokens;
// returns false (nothing written, weights untouched) otherwise, and the caller runs the range form. Only tiles of up
// to four rows (the attend batch's TR) fit their lists and prefix table; larger tiles always run the range form.
constexpr unsigned kMergeCap = 2052U;
// The merge form costs about the same at any depth; the range form costs one pass per kRangeBlocks window
// (tests/qsa_union_merge_tests.hip, T8192 layer-chunk: dense 0.15 vs 1.30 ms, ~75K blocks 8.7 vs 4.7 ms, ~262K
// blocks 19.4 vs 5.2 ms). The merge form runs from kMergeMinBlocks (six windows) up.
constexpr unsigned kMergeMinBlocks = 6U * kRangeBlocks;
template <unsigned TR> constexpr bool kMergeFits = TR <= 4U;
template <unsigned TR> constexpr unsigned kUnionLdsWords = kMergeFits<TR> && TR * kMergeCap > kRangeBlocks ? TR * kMergeCap : kRangeBlocks;

constexpr unsigned kUnionStatsBins = 16U * kMergeCap + 1U;
struct UnionStatsResult {
    unsigned long long rows = 0, tiles = 0, per_row_blocks = 0, blocks = 0, chunks = 0, padded = 0;
    unsigned p50 = 0, p90 = 0, maximum = 0;
};
struct UnionStats { UnionStatsResult result; unsigned histogram[kUnionStatsBins]; };
// Only diagnostic storage is written. Quantiles are nearest ranks of unpadded tile sizes.
template <unsigned TR>
__device__ void union_stats_add(const std::uint32_t* header, UnionStats* stats) {
    const auto* pairs = reinterpret_cast<const uint2*>(header + 4U + TR);
    unsigned n = header[0];
    while (n && !pairs[n - 1U].y) --n;
    unsigned memberships = 0U;
    for (unsigned i = 0; i < n; ++i) memberships += static_cast<unsigned>(__popc(pairs[i].y));
    atomicAdd(&stats->result.rows, static_cast<unsigned long long>(TR));
    atomicAdd(&stats->result.tiles, 1ULL);
    atomicAdd(&stats->result.per_row_blocks, static_cast<unsigned long long>(memberships));
    atomicAdd(&stats->result.blocks, static_cast<unsigned long long>(n));
    atomicAdd(&stats->result.chunks, static_cast<unsigned long long>(header[0] / 4U));
    atomicAdd(&stats->result.padded, static_cast<unsigned long long>(header[0] - n));
    atomicMax(&stats->result.maximum, n);
    atomicAdd(&stats->histogram[min(n, kUnionStatsBins - 1U)], 1U);
}
[[maybe_unused]] __global__ void union_stats_finish_kernel(UnionStats* stats, UnionStatsResult* out) {
    if (threadIdx.x) return;
    auto result = stats->result;
    const auto half = (result.tiles + 1ULL) / 2ULL, nine = (9ULL * result.tiles + 9ULL) / 10ULL;
    unsigned long long sum = 0ULL;
    for (unsigned i = 0; i < kUnionStatsBins; ++i) {
        const auto before = sum;
        sum += stats->histogram[i];
        if (before < half && sum >= half) result.p50 = i;
        if (before < nine && sum >= nine) result.p90 = i;
    }
    *out = result;
}

// NaNs rank as -inf, signed zeros tie. This key is only an indexer ranking, never attention arithmetic.
__device__ __forceinline__ unsigned union_score_key(float score) {
    if (isnan(score)) score = -INFINITY;
    unsigned bits = __float_as_uint(score);
    if (!(bits & 0x7fffffffU)) bits = 0U;
    return bits & 0x80000000U ? ~bits : bits | 0x80000000U;
}
template <unsigned Rows, unsigned TR>
__device__ unsigned union_block_score(const Views<Rows>& v, unsigned r0, uint2 pair) {
    unsigned best = union_score_key(-INFINITY);
    for (unsigned r = 0; r < TR; ++r)
        if ((pair.y & (1U << r)) && pair.x < v.score_blocks[r0 + r])
            best = max(best, union_score_key(v.block_scores[r0 + r][pair.x]));
    return best;
}
__device__ __forceinline__ unsigned union_rank_byte(uint2 pair, unsigned score, unsigned pass, unsigned rows) {
    if (!pass) return rows - static_cast<unsigned>(__popc(pair.y & 0xffffU));
    if (pass < 5U) return (~score >> (8U * (4U - pass))) & 255U;
    return (pair.x >> (8U * (8U - pass))) & 255U;
}
// Four-row production tiles only. Reuse the union builder's LDS; no additional launch or allocation.
// Mandatory entries can exceed the requested cap. Nine radix bytes choose a total order in O(9*N).
// For fixed finite Q/K/V in real arithmetic, dropped softmax mass p gives ||delta O|| <= 2*p*max||V||.
// Rechunking also changes rounding; this is neither a floating-point error bound nor model qualification.
template <unsigned Rows, unsigned TR>
__device__ void union_cap_body(const Views<Rows>& v, unsigned r0, std::uint32_t* header,
    unsigned cap, unsigned* scores, unsigned* waves) {
    static_assert(TR == 4U);
    constexpr unsigned Keep = 1U << 31U, Candidate = 1U << 30U;
    __shared__ unsigned n, remaining, bucket, mandatory;
    __shared__ unsigned row_keep[TR], histogram[256];
    const unsigned t = threadIdx.x;
    auto* pairs = reinterpret_cast<uint2*>(header + 4U + TR);
    __syncthreads();
    if (!t) { n = header[0]; while (n && !pairs[n - 1U].y) --n; }
    __syncthreads();
    if (!cap || n <= cap) return; // preserve every byte, including the old padding and header
    if (t < TR) {
        unsigned best = 0U, at = n;
        for (unsigned i = 0; i < n; ++i) {
            const auto pair = pairs[i];
            if (!(pair.y & (1U << t)) || pair.x * 4U >= v.visible[r0 + t]) continue;
            const unsigned key = pair.x < v.score_blocks[r0 + t] ?
                union_score_key(v.block_scores[r0 + t][pair.x]) : union_score_key(-INFINITY);
            if (at == n || key > best) { best = key; at = i; } // ascending IDs break ties
        }
        row_keep[t] = at;
    }
    if (!t) mandatory = 0U;
    __syncthreads();
    for (unsigned i = t; i < n; i += 256U) {
        auto pair = pairs[i];
        scores[i] = union_block_score<Rows, TR>(v, r0, pair);
        bool keep = false;
        for (unsigned r = 0; r < TR; ++r) {
            keep = keep || row_keep[r] == i;
            keep = keep || ((pair.y & (1U << r)) && v.visible[r0 + r] && pair.x == (v.visible[r0 + r] - 1U) / 4U);
            keep = keep || (!pair.x && header[4U + r]); // packed token-0..3 multiplicities
        }
        pair.y |= keep ? Keep : Candidate;
        pairs[i] = pair;
        if (keep) atomicAdd(&mandatory, 1U);
    }
    __syncthreads();
    if (!t) remaining = cap > mandatory ? cap - mandatory : 0U;
    __syncthreads();
    for (unsigned pass = 0; pass < 9U && remaining; ++pass) {
        histogram[t] = 0U;
        __syncthreads();
        for (unsigned i = t; i < n; i += 256U) {
            const auto pair = pairs[i];
            if (pair.y & Candidate) atomicAdd(&histogram[union_rank_byte(pair, scores[i], pass, TR)], 1U);
        }
        __syncthreads();
        if (!t) {
            bucket = 0U;
            while (bucket < 255U && remaining > histogram[bucket]) remaining -= histogram[bucket++];
        }
        __syncthreads();
        for (unsigned i = t; i < n; i += 256U) {
            auto pair = pairs[i];
            if (!(pair.y & Candidate)) continue;
            const auto byte = union_rank_byte(pair, scores[i], pass, TR);
            if (byte < bucket || (pass == 8U && byte == bucket)) pair.y = (pair.y & ~Candidate) | Keep;
            else if (byte > bucket) pair.y &= ~Candidate;
            pairs[i] = pair;
        }
        __syncthreads();
    }
    unsigned written = 0U;
    for (unsigned base = 0; base < n; base += 256U) {
        const unsigned i = base + t;
        const auto pair = i < n ? pairs[i] : make_uint2(0U, 0U);
        __syncthreads(); // all reads precede the in-place ascending compaction
        const unsigned keep = (pair.y & Keep) ? 1U : 0U;
        unsigned total;
        const auto at = block_exclusive(keep, waves, total);
        if (keep) pairs[written + at] = make_uint2(pair.x, pair.y & 0xffffU);
        written += total;
        __syncthreads();
    }
    const unsigned padded = (written + 3U) / 4U * 4U;
    for (unsigned i = written + t; i < header[0]; i += 256U) pairs[i] = make_uint2(0U, 0U);
    __syncthreads();
    if (!t) header[0] = padded;
}
template <unsigned Rows, unsigned TR>
__device__ __forceinline__ bool union_merge(const Views<Rows>& v, unsigned r0, unsigned blocks, std::uint32_t* lists,
    std::uint16_t* pre, std::uint32_t* weight, std::uint32_t* waves, uint2* pairs, unsigned& written) {
    __shared__ unsigned unsorted;
    const unsigned t = threadIdx.x;
    if (t == 0U) unsorted = 0U;
    __syncthreads();
    unsigned count[TR];
#pragma unroll
    for (unsigned r = 0; r < TR; ++r) {
        const auto* sel = v.selected[r0 + r];
        const unsigned n = count[r] = v.count[r0 + r];
        if (n > kMergeCap) { if (t == 0U) unsorted = 1U; continue; }
        for (unsigned i = t; i < n; i += 256U) {
            const unsigned token = sel[i];
            if (i > 0U && token < sel[i - 1U]) unsorted = 1U;
            lists[r * kMergeCap + i] = token / 4U;
        }
    }
    __syncthreads();
    if (unsorted) return false;
    for (unsigned r = 0; r < TR; ++r) {
        const auto* sel = v.selected[r0 + r];
        const unsigned n = count[r];
        const auto* blk = lists + r * kMergeCap;
        auto* row = pre + r * (kMergeCap + 1U);
        unsigned running = 0U;
        for (unsigned base = 0; base < n; base += 256U) {
            const unsigned i = base + t;
            unsigned owner = 0U;
            if (i < n) {
                const unsigned b = blk[i];
                if (sel[i] < 4U) atomicAdd(&weight[r * 4U + sel[i]], 1U);
                owner = b < blocks && (i == 0U || blk[i - 1U] != b) ? 1U : 0U;
                for (unsigned q = 0; owner && q < r; ++q) {
                    const auto* other = lists + q * kMergeCap;
                    const unsigned at = lower_bound_block(other, count[q], b);
                    if (at < count[q] && other[at] == b) owner = 0U;
                }
            }
            unsigned total;
            const unsigned before = block_exclusive(owner, waves, total);
            if (i < n) row[i] = static_cast<std::uint16_t>(running + before);
            running += total;
        }
        if (t == 0U) row[n] = static_cast<std::uint16_t>(running);
        written += running;
    }
    __syncthreads();
    for (unsigned r = 0; r < TR; ++r) {
        const unsigned n = count[r];
        const auto* blk = lists + r * kMergeCap;
        const auto* row = pre + r * (kMergeCap + 1U);
        for (unsigned i = t; i < n; i += 256U) {
            if (row[i + 1U] == row[i]) continue; // not an owner (row[n] holds the row's total)
            const unsigned b = blk[i];
            unsigned at = 0U, bits = 0U;
#pragma unroll
            for (unsigned q = 0; q < TR; ++q) {
                const auto* other = lists + q * kMergeCap;
                const unsigned lb = q == r ? i : lower_bound_block(other, count[q], b);
                at += pre[q * (kMergeCap + 1U) + lb];
                if (lb < count[q] && other[lb] == b) bits |= 1U << q;
            }
            pairs[at] = make_uint2(b, bits);
        }
    }
    return true;
}

// tile: the TR-row tile of the group (blockIdx.x of union_kernel / attend_kernel).
template <unsigned Rows, unsigned TR, bool Merge = true, bool Capped = false>
__device__ __forceinline__ void union_body(const Views<Rows>& v, std::uint32_t union_at, unsigned tile,
    unsigned cap = 0U, UnionStats* stats = nullptr) {
    __shared__ std::uint32_t mask[Capped ? TR * kMergeCap : kUnionLdsWords<TR>]; // cap reuses this for score keys
    __shared__ std::uint16_t merge_prefix[Merge && kMergeFits<TR> ? TR * (kMergeCap + 1U) : 1U];
    __shared__ std::uint32_t weight[TR * 4U];
    __shared__ std::uint32_t waves[8];
    const unsigned t = threadIdx.x, r0 = tile * TR;
    unsigned blocks = 0U;
    for (unsigned r = 0; r < TR; ++r) blocks = max(blocks, (v.visible[r0 + r] + 3U) / 4U);
    if (t < TR * 4U) weight[t] = 0U;
    auto* header = reinterpret_cast<std::uint32_t*>(v.scratch[r0] + union_at);
    auto* pairs = reinterpret_cast<uint2*>(header + 4U + TR);
    unsigned written = 0U;
    bool merged = false;
    if constexpr (Merge && kMergeFits<TR>) merged = blocks > kMergeMinBlocks && union_merge<Rows, TR>(v, r0, blocks, mask, merge_prefix, weight, waves, pairs, written);
    for (unsigned base = 0; !merged && base < blocks; base += kRangeBlocks) {
        const unsigned span = min(kRangeBlocks, blocks - base);
        for (unsigned i = t; i < span; i += 256U) mask[i] = 0U;
        __syncthreads();
        for (unsigned r = 0; r < TR; ++r) {
            const auto* sel = v.selected[r0 + r];
            const unsigned n = v.count[r0 + r];
            for (unsigned i = t; i < n; i += 256U) {
                const unsigned token = sel[i], b = token / 4U;
                if (base == 0U && token < 4U) atomicAdd(&weight[r * 4U + token], 1U);
                if (b >= base && b - base < span) atomicOr(&mask[b - base], 1U << r);
            }
        }
        __syncthreads();
        const unsigned per = (span + 255U) / 256U, first = min(t * per, span), last = min(first + per, span);
        unsigned c = 0U;
        for (unsigned j = first; j < last; ++j) c += mask[j] != 0U;
        unsigned total;
        unsigned at = written + block_exclusive(c, waves, total);
        for (unsigned j = first; j < last; ++j) if (mask[j]) pairs[at++] = make_uint2(base + j, mask[j]);
        written += total;
        __syncthreads();
    }
    const unsigned padded = (written + 3U) / 4U * 4U;
    if (t < padded - written) pairs[written + t] = make_uint2(0U, 0U);
    if (t == 0U) header[0] = padded;
    if (t < TR) {
        unsigned packed = 0U;
        for (unsigned k = 0; k < 4U; ++k) packed |= min(weight[t * 4U + k], 255U) << (8U * k);
        header[4U + t] = packed;
    }
    if (stats) {
        __syncthreads();
        if (!t) union_stats_add<TR>(header, stats); // original union, before any FAST removal
    }
    if constexpr (Capped) union_cap_body<Rows, TR>(v, r0, header, cap, mask, waves);
}

template <unsigned Rows, unsigned TR, bool Merge = true, bool Capped = false>
__global__ __launch_bounds__(256) void union_kernel(Views<Rows> v, std::uint32_t union_at,
    unsigned cap = 0U, UnionStats* stats = nullptr) {
    union_body<Rows, TR, Merge, Capped>(v, union_at, blockIdx.x, cap, stats);
}
// Several groups of one call in one launch (STRIX_QSA_ATTEND_BATCH): block x = group * (Rows / TR) + tile.
template <unsigned Rows, unsigned TR, bool Merge = true, bool Capped = false>
__global__ __launch_bounds__(256) void union_multi_kernel(const Views<Rows>* groups, std::uint32_t union_at,
    unsigned cap = 0U, UnionStats* stats = nullptr) {
    union_body<Rows, TR, Merge, Capped>(groups[blockIdx.x / (Rows / TR)], union_at, blockIdx.x % (Rows / TR), cap, stats);
}

// One workgroup: G column groups of 16 query columns, each run by a pair of compute waves that split the 256
// features (each keeps its half of Q as FP16 WMMA fragments in registers, forms the partial scores over that
// half, and accumulates that half of O), and four loader waves. The loaders keep D chunks in flight in
// registers and stage the oldest as FP16 into one of two LDS buffers (with its blocks and masks) while the compute
// waves run the other. Per chunk: partial scores, barrier, exchange (each wave adds the pair's halves in
// the same order, so both hold the same scores and run the same softmax), values, barrier. debug (fixture
// ablations): bit 0 loads token 0 for every slot, bit 1 skips the compute.
// Int8 (STRIX_QSA_FAST_INT8=1 with kv-int8-tok-v1): the loaders read the int8 codes (8 bytes per task instead of 16)
// and stage them as FP16 (exact: |code| <= 127); loader lanes 0..31 stage the chunk's 16 K and 16 V FP32 scales of the
// KV head. The compute waves apply the scale of a token to its FP32 score (S = (codes . q) * s_k) and to its FP16
// probability operand of the value product (P * s_v), so O = sum P s_v codes_v. Identity kv8-fast-qsa-v1: the same
// selection, masks, online softmax and combine; only the place where the dequantization scale enters changes.
__device__ __forceinline__ unsigned q8x2_to_f16x2(unsigned codes, unsigned shift) {
    const _Float16 a = static_cast<_Float16>(static_cast<float>(static_cast<std::int8_t>((codes >> shift) & 0xffU)));
    const _Float16 b = static_cast<_Float16>(static_cast<float>(static_cast<std::int8_t>((codes >> (shift + 8U)) & 0xffU)));
    return static_cast<unsigned>(__builtin_bit_cast(unsigned short, a)) |
        (static_cast<unsigned>(__builtin_bit_cast(unsigned short, b)) << 16U);
}
typedef unsigned u32x2 __attribute__((ext_vector_type(2)));
__device__ __forceinline__ u32x2 load8(const void* p) {
    return *reinterpret_cast<const __attribute__((address_space(1))) u32x2*>(reinterpret_cast<std::uintptr_t>(p));
}
__device__ __forceinline__ float load_scale(const void* p) {
    return *reinterpret_cast<const __attribute__((address_space(1))) float*>(reinterpret_cast<std::uintptr_t>(p));
}
// Selection and scale adapter for the shared selected-key flash chunk.
template <bool Int8>
struct SelectedFlashPolicy {
    static constexpr unsigned kDim = qsa_fast::kDim, kPitch = qsa_fast::kPitch;
    uint2 meta[4];
    unsigned bit, visible, weights, half;
    float score_scale;
    const float* scales;
    __device__ __forceinline__ SelectedFlashPolicy(const uint2 (&m)[4], unsigned b, unsigned vis, unsigned w,
        unsigned h, float ss, const float* sc) : meta{m[0], m[1], m[2], m[3]}, bit(b), visible(vis), weights(w),
        half(h), score_scale(ss), scales(sc) {}
    // Element e: token slot 2e + half of the chunk (block e / 2), column col.
    __device__ __forceinline__ float score(float x, unsigned e) const {
        const uint2 blk = meta[e / 2U];
        const unsigned token = blk.x * 4U + 2U * (e % 2U) + half;
        const bool live = (blk.y & bit) != 0U && token < visible;
        if constexpr (Int8) x = x * scales[2U * e + half];
        return live ? x * score_scale : -INFINITY;
    }
    __device__ __forceinline__ void weight(float (&p)[8]) const {
        if (meta[0].x == 0U)  // tokens 0..3: their multiplicity in the row's selection
#pragma unroll
            for (unsigned e = 0; e < 2U; ++e) p[e] *= static_cast<float>((weights >> (8U * (2U * e + half))) & 255U);
    }
    __device__ __forceinline__ void value_scale(float& lo, float& hi, unsigned e) const {
        if constexpr (Int8) { lo = lo * scales[16U + 2U * e]; hi = hi * scales[17U + 2U * e]; }
    }
    __device__ static __forceinline__ float swap(float x) { return swap16(x); }
    __device__ static __forceinline__ void barrier() { lds_barrier(); }
    __device__ static __forceinline__ unsigned value_row(unsigned d) { return v_row(d); }
};

template <unsigned Rows, unsigned TR, unsigned G, unsigned D, bool Int8, bool Capture = false>
__device__ __forceinline__ void attend_body(const Views<Rows>& v, const std::uint16_t* kv, std::uint32_t kv_heads,
    std::uint32_t heads, std::uint32_t union_at, std::uint32_t splits, std::uint32_t debug, unsigned tile, unsigned yb) {
    constexpr unsigned HPW = 16U / TR, HG = G * HPW, Loaders = 128U, Compute = 2U * G;
    constexpr unsigned KT = kChunk * 32U / Loaders; // V: one task per loader thread
    __shared__ __align__(16) _Float16 k_lds[2][kChunk][kPitch];
    __shared__ __align__(16) _Float16 v_lds[2][kDim][kChunk];
    __shared__ __align__(16) float xch[G][2][32][8];
    __shared__ uint2 meta_lds[2][4];
    __shared__ float scale_lds[Int8 ? 2 : 1][Int8 ? 32 : 1]; // Int8: [buf][K slots 0..15, V slots 0..15]
    const unsigned t = threadIdx.x, lane = t % 32U, w = t / 32U, col = lane % 16U, half = lane / 16U;
    const bool loader = w >= Compute;
    const unsigned r0 = tile * TR;
    const unsigned per_kv = heads / kv_heads, groups = per_kv / HG;
    const unsigned kvh = yb / groups, hg = yb % groups, split = blockIdx.z;
    unsigned tile_visible = 0U;
    for (unsigned i = 0; i < TR; ++i) tile_visible = max(tile_visible, v.visible[r0 + i]);
    const auto* header = reinterpret_cast<const std::uint32_t*>(v.scratch[r0] + union_at);
    const auto* pairs = reinterpret_cast<const uint2*>(header + 4U + TR);
    const unsigned chunks = header[0] / 4U;
    const unsigned c0 = static_cast<unsigned>(static_cast<std::uint64_t>(chunks) * split / splits);
    const unsigned c1 = static_cast<unsigned>(static_cast<std::uint64_t>(chunks) * (split + 1U) / splits);

    if (loader) {
        struct Regs { u32x4 k[KT]; u32x4 v[4]; unsigned valid; };
        struct Regs8 { u32x2 k[KT]; u32x2 v[4]; float scale; unsigned valid; };
        using R = std::conditional_t<Int8, Regs8, Regs>;
        // Lane i < 4 of each loader wave holds the chunk's (block, mask) i; a slot's block comes by lane
        // shuffle (not an indexed array, which would live in scratch, whose loads wait for every load in flight).
        const auto meta_at = [&](unsigned c) { return lane < 4U ? pairs[4U * c + lane] : make_uint2(0U, 0U); };
        const auto token_of = [](uint2 m, unsigned slot) {
            return static_cast<unsigned>(__shfl(static_cast<int>(m.x), static_cast<int>(slot / 4U), 32)) * 4U + slot % 4U;
        };
        const auto row_elements = static_cast<std::uint64_t>(kv_heads) * kDim * 2U;
        const auto* k_base = kv + static_cast<std::uint64_t>(kvh) * kDim;
        const auto* v_base = kv + static_cast<std::uint64_t>(kv_heads + kvh) * kDim;
        // Int8 row: [K codes kv_width | V codes kv_width | FP32 scales: K heads, then V heads] (kv_int8_row_bytes).
        const auto* bytes = reinterpret_cast<const std::uint8_t*>(kv);
        const auto kv_width = static_cast<std::uint64_t>(kv_heads) * kDim;
        const auto row_bytes = 2U * kv_width + 8ULL * kv_heads;
        const unsigned lt = t - Compute * 32U;
        // K task i = (token i / 32, 8 features 8 (i % 32)); V task = (tokens 4 (lt % 4) .. + 3, 8 features
        // 8 (lt / 4)). Every load is issued (a token past the tile's visible prefix reads the last visible one
        // and is staged as zero), so no load waits on a branch.
        const auto fetch = [&](uint2 m, R& g) {
            g.valid = 0U;
#pragma unroll
            for (unsigned i = 0; i < KT; ++i) {
                const unsigned task = lt + i * Loaders;
                const unsigned token = (debug & 1U) ? 0U : token_of(m, task / 32U);
                g.valid |= (token < tile_visible ? 1U : 0U) << i;
                const unsigned at = token < tile_visible ? token : tile_visible - 1U;
                if constexpr (Int8) g.k[i] = load8(bytes + at * row_bytes + static_cast<std::uint64_t>(kvh) * kDim + 8U * (task % 32U));
                else g.k[i] = load16(k_base + at * row_elements + 8U * (task % 32U));
            }
#pragma unroll
            for (unsigned h = 0; h < 4U; ++h) {
                const unsigned token = (debug & 1U) ? 0U : token_of(m, 4U * (lt % 4U) + h);
                g.valid |= (token < tile_visible ? 1U : 0U) << (KT + h);
                const unsigned at = token < tile_visible ? token : tile_visible - 1U;
                if constexpr (Int8) g.v[h] = load8(bytes + at * row_bytes + kv_width + static_cast<std::uint64_t>(kvh) * kDim + 8U * (lt / 4U));
                else g.v[h] = load16(v_base + at * row_elements + 8U * (lt / 4U));
            }
            if constexpr (Int8) { // lane lt % 32: K scale of slot lt % 16 (lt % 32 < 16) or V scale of slot lt % 16
                const unsigned slot = lt % 16U, token = (debug & 1U) ? 0U : token_of(m, slot);
                const bool value = (lt % 32U) >= 16U;
                g.valid |= (token < tile_visible ? 1U : 0U) << 31U;
                g.scale = load_scale(bytes + (token < tile_visible ? token : tile_visible - 1U) * row_bytes + 2U * kv_width +
                    4ULL * ((value ? kv_heads : 0U) + kvh));
            }
        };
        // Int8: the eight codes of a task as four FP16 pairs (features 2k, 2k + 1), zero when not valid.
        const auto pairs8 = [](u32x2 c, bool keep, unsigned (&f)[4]) {
            const u32x2 x = keep ? c : u32x2{0U, 0U};
            f[0] = q8x2_to_f16x2(x.x, 0U); f[1] = q8x2_to_f16x2(x.x, 16U);
            f[2] = q8x2_to_f16x2(x.y, 0U); f[3] = q8x2_to_f16x2(x.y, 16U);
        };
        const auto stage = [&](const R& g, uint2 m, unsigned buf) {
          if constexpr (Int8) {
#pragma unroll
            for (unsigned i = 0; i < KT; ++i) {
                const unsigned task = lt + i * Loaders;
                unsigned f[4];
                pairs8(g.k[i], ((g.valid >> i) & 1U) != 0U, f);
                *reinterpret_cast<uint4*>(&k_lds[buf][task / 32U][8U * (task % 32U)]) = uint4{f[0], f[1], f[2], f[3]};
            }
            unsigned f[4][4];
#pragma unroll
            for (unsigned h = 0; h < 4U; ++h) pairs8(g.v[h], ((g.valid >> (KT + h)) & 1U) != 0U, f[h]);
            const unsigned feature = 8U * (lt / 4U), slot = 4U * (lt % 4U);
#pragma unroll
            for (unsigned k = 0; k < 4U; ++k) {
                *reinterpret_cast<uint2*>(&v_lds[buf][v_row(feature + 2U * k)][slot]) =
                    make_uint2((f[0][k] & 0xffffU) | (f[1][k] << 16U), (f[2][k] & 0xffffU) | (f[3][k] << 16U));
                *reinterpret_cast<uint2*>(&v_lds[buf][v_row(feature + 2U * k + 1U)][slot]) =
                    make_uint2((f[0][k] >> 16U) | (f[1][k] & 0xffff0000U), (f[2][k] >> 16U) | (f[3][k] & 0xffff0000U));
            }
            if (lt < 4U) meta_lds[buf][lt] = m;
            if (lt < 32U) scale_lds[Int8 ? buf : 0U][Int8 ? lt : 0U] = (g.valid >> 31U) ? g.scale : 0.0F;
          } else {
            const auto keep = [&](u32x4 x, unsigned b) { return ((g.valid >> b) & 1U) ? x : u32x4{0U, 0U, 0U, 0U}; };
#pragma unroll
            for (unsigned i = 0; i < KT; ++i) {
                const unsigned task = lt + i * Loaders;
                const u32x4 x = keep(g.k[i], i);
                *reinterpret_cast<uint4*>(&k_lds[buf][task / 32U][8U * (task % 32U)]) =
                    uint4{bf16x2_to_f16x2(x.x), bf16x2_to_f16x2(x.y), bf16x2_to_f16x2(x.z), bf16x2_to_f16x2(x.w)};
            }
            unsigned f[4][4]; // token h, FP16 pair k = features 8p + 2k, + 1
#pragma unroll
            for (unsigned h = 0; h < 4U; ++h) {
                const u32x4 x = keep(g.v[h], KT + h);
                f[h][0] = bf16x2_to_f16x2(x.x); f[h][1] = bf16x2_to_f16x2(x.y);
                f[h][2] = bf16x2_to_f16x2(x.z); f[h][3] = bf16x2_to_f16x2(x.w);
            }
            const unsigned feature = 8U * (lt / 4U), slot = 4U * (lt % 4U);
#pragma unroll
            for (unsigned k = 0; k < 4U; ++k) {
                *reinterpret_cast<uint2*>(&v_lds[buf][v_row(feature + 2U * k)][slot]) =
                    make_uint2((f[0][k] & 0xffffU) | (f[1][k] << 16U), (f[2][k] & 0xffffU) | (f[3][k] << 16U));
                *reinterpret_cast<uint2*>(&v_lds[buf][v_row(feature + 2U * k + 1U)][slot]) =
                    make_uint2((f[0][k] >> 16U) | (f[1][k] & 0xffff0000U), (f[2][k] >> 16U) | (f[3][k] & 0xffff0000U));
            }
            if (lt < 4U) meta_lds[buf][lt] = m;
          }
        };
        // D chunks in flight, slot k of the loop unrolled by D: before step c, r[k] holds chunk c + 1 (fetched
        // with meta held[k]) and raw[k] the meta of chunk c + 1 + D, loaded D chunks before its fetch (so a wait
        // for it covers only loads at least as old as the chunk staged next). Fixed slots (moving a register
        // whose load is in flight would wait for it) and no branch around a load (chunk indices clamp to the
        // last chunk; what the tail stages goes to the buffer the compute waves no longer read).
        if (c0 < c1) {
            const auto chunk = [&](unsigned c) { return c < c1 ? c : c1 - 1U; };
            R r[D];
            uint2 held[D], raw[D];
            {
                R first;
                const uint2 m = meta_at(c0);
                fetch(m, first);
                stage(first, m, 0U);
            }
#pragma unroll
            for (unsigned d = 0; d < D; ++d) { held[d] = meta_at(chunk(c0 + 1U + d)); fetch(held[d], r[d]); }
#pragma unroll
            for (unsigned d = 0; d < D; ++d) raw[d] = meta_at(chunk(c0 + 1U + D + d));
            lds_barrier();
            for (unsigned c = c0; c < c1; c += D) {
#pragma unroll
                for (unsigned k = 0; k < D; ++k) {
                    const unsigned step = c + k;
                    if (step >= c1) break;
                    stage(r[k], held[k], (step - c0 + 1U) & 1U);
                    held[k] = raw[k];
                    fetch(held[k], r[k]);
                    raw[k] = meta_at(chunk(step + 1U + 2U * D));
                    lds_barrier();
                    lds_barrier();
                }
            }
        }
        return;
    }

    const unsigned grp = w / 2U, fh = w % 2U; // column group, feature half
    const unsigned row = col % TR, head = kvh * per_kv + hg * HG + grp * HPW + col / TR, r = r0 + row;
    // Q of this lane's column: a power-of-two scale from the largest |q| of the 256 features (the lane pair
    // (col, col + 16) reads half each), then this wave's 128 features as FP16 B fragments.
    float score_scale;
    f16x16 qf[8];
    {
        const float* q = v.q_gate[r] + static_cast<std::uint64_t>(head) * 2U * kDim;
        float peak = 0.0F;
        for (unsigned i = half * (kDim / 2U); i < (half + 1U) * (kDim / 2U); i += 4U) {
            const float4 x = *reinterpret_cast<const float4*>(q + i);
            peak = fmaxf(peak, fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w))));
        }
        peak = fmaxf(peak, swap16(peak));
        const int e = q_exponent(peak);
        const float scale = ldexpf(1.0F, e);
        // Scores in base 2: the softmax below uses exp2 (v_exp_f32).
        score_scale = ldexpf(1.0F, -e) / sqrtf(static_cast<float>(kDim)) * 1.4426950408889634F;
#pragma unroll
        for (unsigned k = 0; k < 8U; ++k)
#pragma unroll
            for (unsigned i = 0; i < 16U; i += 4U) {
                const float4 x = *reinterpret_cast<const float4*>(q + fh * (kDim / 2U) + 16U * k + i);
                qf[k][i] = static_cast<_Float16>(x.x * scale); qf[k][i + 1U] = static_cast<_Float16>(x.y * scale);
                qf[k][i + 2U] = static_cast<_Float16>(x.z * scale); qf[k][i + 3U] = static_cast<_Float16>(x.w * scale);
            }
    }
    const unsigned visible = v.visible[r], bit = 1U << row, weights = header[4U + row];
    f32x8 acc[8];
#pragma unroll
    for (unsigned n = 0; n < 8U; ++n) acc[n] = f32x8{};
    float m_run = -INFINITY, l_run = 0.0F;
    if (c0 < c1) lds_barrier();
    for (unsigned c = c0; c < c1; ++c) {
        const unsigned buf = (c - c0) & 1U;
        if (debug & 2U) { lds_barrier(); lds_barrier(); continue; }
        uint2 meta[4];
#pragma unroll
        for (unsigned i = 0; i < 4U; ++i) meta[i] = meta_lds[buf][i];
        strix::attn_core::selected_flash_step(qf, acc, k_lds[buf], v_lds[buf], xch[grp], col, half, fh, lane,
            m_run, l_run, SelectedFlashPolicy<Int8>{meta, bit, visible, weights, half, score_scale, scale_lds[Int8 ? buf : 0U]});
    }
    if constexpr (Capture) { // fixture-only final online state, before the lossy output rounding
        float* state = v.scratch[r] + static_cast<std::uint64_t>(head) * kPartial;
        if (fh == 0U && half == 0U) { state[0] = m_run; state[1] = l_run; }
#pragma unroll
        for (unsigned n = 0; n < 8U; ++n)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) state[2U + fh * (kDim / 2U) + 16U * n + 2U * e + half] = acc[n][e];
    }
    if (splits == 1U) {
        // One slice: combine_kernel's arithmetic for one split, in place.
        const float* gate_in = v.q_gate[r] + static_cast<std::uint64_t>(head) * 2U * kDim + kDim;
        auto out = v.image(r) + static_cast<std::uint64_t>(head) * kDim;
#pragma unroll
        for (unsigned n = 0; n < 8U; ++n)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const unsigned d = fh * (kDim / 2U) + 16U * n + 2U * e + half;
                const float attn = round_bf16(l_run > 0.0F ? acc[n][e] / l_run : 0.0F);
                const float gate = round_bf16(1.0F / (1.0F + expf(-gate_in[d])));
                out[d] = round_bf16(attn * gate);
            }
        return;
    }
    float* partial = v.scratch[r] + (static_cast<std::uint64_t>(split) * heads + head) * kPartial;
    if (fh == 0U && half == 0U) { partial[0] = m_run; partial[1] = l_run; }
#pragma unroll
    for (unsigned n = 0; n < 8U; ++n)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) partial[2U + fh * (kDim / 2U) + 16U * n + 2U * e + half] = acc[n][e];
}

template <unsigned Rows, unsigned TR, unsigned G, unsigned D = 3U, bool Int8 = false>
__global__ __launch_bounds__((2U * G + 4U) * 32U) void attend_kernel(Views<Rows> v, const std::uint16_t* kv, std::uint32_t kv_heads,
    std::uint32_t heads, std::uint32_t union_at, std::uint32_t splits, std::uint32_t debug = 0U) {
    attend_body<Rows, TR, G, D, Int8>(v, kv, kv_heads, heads, union_at, splits, debug, blockIdx.x, blockIdx.y);
}
// FP16 KV only (the launcher queues no Int8 group).
template <unsigned Rows, unsigned TR, unsigned G, unsigned D = 3U>
__global__ __launch_bounds__((2U * G + 4U) * 32U) void attend_multi_kernel(const Views<Rows>* groups, const std::uint16_t* kv,
    std::uint32_t kv_heads, std::uint32_t heads, std::uint32_t union_at) {
    attend_body<Rows, TR, G, D, false>(groups[blockIdx.x / (Rows / TR)], kv, kv_heads, heads, union_at, 1U, 0U, blockIdx.x % (Rows / TR), blockIdx.y);
}
// qpf2k/attn candidate I (STRIX_QPF2K_QSA bit 6): attend_multi_kernel's body under a one-dimensional grid ordered
// group-major, x = (group * ys + y) * (Rows / TR) + tile with ys = kv_heads * head groups: the workgroups resident at a
// time cover as few row groups (whose unions overlap) as possible, so the KV blocks they gather stay in the last-level
// cache. Scheduling only: every workgroup runs the same (group, tile, y) body, so the outputs are bitwise the same.
template <unsigned Rows, unsigned TR, unsigned G, unsigned D = 3U>
__global__ __launch_bounds__((2U * G + 4U) * 32U) void attend_order_kernel(const Views<Rows>* groups, const std::uint16_t* kv,
    std::uint32_t kv_heads, std::uint32_t heads, std::uint32_t union_at, std::uint32_t ys) {
    constexpr unsigned tiles = Rows / TR;
    const unsigned bx = blockIdx.x, tile = bx % tiles, y = (bx / tiles) % ys, g = bx / (tiles * ys);
    attend_body<Rows, TR, G, D, false>(groups[g], kv, kv_heads, heads, union_at, 1U, 0U, tile, y);
}
// Occ-only copy: keep attend_body and the non-occ kernels unchanged. Floating-point
// statements are identical; the query-fragment fence below bounds load live ranges.
template <unsigned Rows, unsigned TR, unsigned G, unsigned D, bool Int8>
__device__ __forceinline__ void attend_occ_body(const Views<Rows>& v, const std::uint16_t* kv, std::uint32_t kv_heads,
    std::uint32_t heads, std::uint32_t union_at, std::uint32_t splits, std::uint32_t debug, unsigned tile, unsigned yb) {
    constexpr unsigned HPW = 16U / TR, HG = G * HPW, Loaders = 128U, Compute = 2U * G;
    constexpr unsigned KT = kChunk * 32U / Loaders; // V: one task per loader thread
    __shared__ __align__(16) _Float16 k_lds[2][kChunk][kPitch];
    __shared__ __align__(16) _Float16 v_lds[2][kDim][kChunk];
    __shared__ __align__(16) float xch[G][2][32][8];
    __shared__ uint2 meta_lds[2][4];
    __shared__ float scale_lds[Int8 ? 2 : 1][Int8 ? 32 : 1]; // Int8: [buf][K slots 0..15, V slots 0..15]
    const unsigned t = threadIdx.x, lane = t % 32U, w = t / 32U, col = lane % 16U, half = lane / 16U;
    const bool loader = w >= Compute;
    const unsigned r0 = tile * TR;
    const unsigned per_kv = heads / kv_heads, groups = per_kv / HG;
    const unsigned kvh = yb / groups, hg = yb % groups, split = blockIdx.z;
    unsigned tile_visible = 0U;
    for (unsigned i = 0; i < TR; ++i) tile_visible = max(tile_visible, v.visible[r0 + i]);
    const auto* header = reinterpret_cast<const std::uint32_t*>(v.scratch[r0] + union_at);
    const auto* pairs = reinterpret_cast<const uint2*>(header + 4U + TR);
    const unsigned chunks = header[0] / 4U;
    const unsigned c0 = static_cast<unsigned>(static_cast<std::uint64_t>(chunks) * split / splits);
    const unsigned c1 = static_cast<unsigned>(static_cast<std::uint64_t>(chunks) * (split + 1U) / splits);

    if (loader) {
        struct Regs { u32x4 k[KT]; u32x4 v[4]; unsigned valid; };
        struct Regs8 { u32x2 k[KT]; u32x2 v[4]; float scale; unsigned valid; };
        using R = std::conditional_t<Int8, Regs8, Regs>;
        // Lane i < 4 of each loader wave holds the chunk's (block, mask) i; a slot's block comes by lane
        // shuffle (not an indexed array, which would live in scratch, whose loads wait for every load in flight).
        const auto meta_at = [&](unsigned c) { return lane < 4U ? pairs[4U * c + lane] : make_uint2(0U, 0U); };
        const auto token_of = [](uint2 m, unsigned slot) {
            return static_cast<unsigned>(__shfl(static_cast<int>(m.x), static_cast<int>(slot / 4U), 32)) * 4U + slot % 4U;
        };
        const auto row_elements = static_cast<std::uint64_t>(kv_heads) * kDim * 2U;
        const auto* k_base = kv + static_cast<std::uint64_t>(kvh) * kDim;
        const auto* v_base = kv + static_cast<std::uint64_t>(kv_heads + kvh) * kDim;
        // Int8 row: [K codes kv_width | V codes kv_width | FP32 scales: K heads, then V heads] (kv_int8_row_bytes).
        const auto* bytes = reinterpret_cast<const std::uint8_t*>(kv);
        const auto kv_width = static_cast<std::uint64_t>(kv_heads) * kDim;
        const auto row_bytes = 2U * kv_width + 8ULL * kv_heads;
        const unsigned lt = t - Compute * 32U;
        // K task i = (token i / 32, 8 features 8 (i % 32)); V task = (tokens 4 (lt % 4) .. + 3, 8 features
        // 8 (lt / 4)). Every load is issued (a token past the tile's visible prefix reads the last visible one
        // and is staged as zero), so no load waits on a branch.
        const auto fetch = [&](uint2 m, R& g) {
            g.valid = 0U;
#pragma unroll
            for (unsigned i = 0; i < KT; ++i) {
                const unsigned task = lt + i * Loaders;
                const unsigned token = (debug & 1U) ? 0U : token_of(m, task / 32U);
                g.valid |= (token < tile_visible ? 1U : 0U) << i;
                const unsigned at = token < tile_visible ? token : tile_visible - 1U;
                if constexpr (Int8) g.k[i] = load8(bytes + at * row_bytes + static_cast<std::uint64_t>(kvh) * kDim + 8U * (task % 32U));
                else g.k[i] = load16(k_base + at * row_elements + 8U * (task % 32U));
            }
#pragma unroll
            for (unsigned h = 0; h < 4U; ++h) {
                const unsigned token = (debug & 1U) ? 0U : token_of(m, 4U * (lt % 4U) + h);
                g.valid |= (token < tile_visible ? 1U : 0U) << (KT + h);
                const unsigned at = token < tile_visible ? token : tile_visible - 1U;
                if constexpr (Int8) g.v[h] = load8(bytes + at * row_bytes + kv_width + static_cast<std::uint64_t>(kvh) * kDim + 8U * (lt / 4U));
                else g.v[h] = load16(v_base + at * row_elements + 8U * (lt / 4U));
            }
            if constexpr (Int8) { // lane lt % 32: K scale of slot lt % 16 (lt % 32 < 16) or V scale of slot lt % 16
                const unsigned slot = lt % 16U, token = (debug & 1U) ? 0U : token_of(m, slot);
                const bool value = (lt % 32U) >= 16U;
                g.valid |= (token < tile_visible ? 1U : 0U) << 31U;
                g.scale = load_scale(bytes + (token < tile_visible ? token : tile_visible - 1U) * row_bytes + 2U * kv_width +
                    4ULL * ((value ? kv_heads : 0U) + kvh));
            }
        };
        // Int8: the eight codes of a task as four FP16 pairs (features 2k, 2k + 1), zero when not valid.
        const auto pairs8 = [](u32x2 c, bool keep, unsigned (&f)[4]) {
            const u32x2 x = keep ? c : u32x2{0U, 0U};
            f[0] = q8x2_to_f16x2(x.x, 0U); f[1] = q8x2_to_f16x2(x.x, 16U);
            f[2] = q8x2_to_f16x2(x.y, 0U); f[3] = q8x2_to_f16x2(x.y, 16U);
        };
        const auto stage = [&](const R& g, uint2 m, unsigned buf) {
          if constexpr (Int8) {
#pragma unroll
            for (unsigned i = 0; i < KT; ++i) {
                const unsigned task = lt + i * Loaders;
                unsigned f[4];
                pairs8(g.k[i], ((g.valid >> i) & 1U) != 0U, f);
                *reinterpret_cast<uint4*>(&k_lds[buf][task / 32U][8U * (task % 32U)]) = uint4{f[0], f[1], f[2], f[3]};
            }
            unsigned f[4][4];
#pragma unroll
            for (unsigned h = 0; h < 4U; ++h) pairs8(g.v[h], ((g.valid >> (KT + h)) & 1U) != 0U, f[h]);
            const unsigned feature = 8U * (lt / 4U), slot = 4U * (lt % 4U);
#pragma unroll
            for (unsigned k = 0; k < 4U; ++k) {
                *reinterpret_cast<uint2*>(&v_lds[buf][v_row(feature + 2U * k)][slot]) =
                    make_uint2((f[0][k] & 0xffffU) | (f[1][k] << 16U), (f[2][k] & 0xffffU) | (f[3][k] << 16U));
                *reinterpret_cast<uint2*>(&v_lds[buf][v_row(feature + 2U * k + 1U)][slot]) =
                    make_uint2((f[0][k] >> 16U) | (f[1][k] & 0xffff0000U), (f[2][k] >> 16U) | (f[3][k] & 0xffff0000U));
            }
            if (lt < 4U) meta_lds[buf][lt] = m;
            if (lt < 32U) scale_lds[Int8 ? buf : 0U][Int8 ? lt : 0U] = (g.valid >> 31U) ? g.scale : 0.0F;
          } else {
            const auto keep = [&](u32x4 x, unsigned b) { return ((g.valid >> b) & 1U) ? x : u32x4{0U, 0U, 0U, 0U}; };
#pragma unroll
            for (unsigned i = 0; i < KT; ++i) {
                const unsigned task = lt + i * Loaders;
                const u32x4 x = keep(g.k[i], i);
                *reinterpret_cast<uint4*>(&k_lds[buf][task / 32U][8U * (task % 32U)]) =
                    uint4{bf16x2_to_f16x2(x.x), bf16x2_to_f16x2(x.y), bf16x2_to_f16x2(x.z), bf16x2_to_f16x2(x.w)};
            }
            unsigned f[4][4]; // token h, FP16 pair k = features 8p + 2k, + 1
#pragma unroll
            for (unsigned h = 0; h < 4U; ++h) {
                const u32x4 x = keep(g.v[h], KT + h);
                f[h][0] = bf16x2_to_f16x2(x.x); f[h][1] = bf16x2_to_f16x2(x.y);
                f[h][2] = bf16x2_to_f16x2(x.z); f[h][3] = bf16x2_to_f16x2(x.w);
            }
            const unsigned feature = 8U * (lt / 4U), slot = 4U * (lt % 4U);
#pragma unroll
            for (unsigned k = 0; k < 4U; ++k) {
                *reinterpret_cast<uint2*>(&v_lds[buf][v_row(feature + 2U * k)][slot]) =
                    make_uint2((f[0][k] & 0xffffU) | (f[1][k] << 16U), (f[2][k] & 0xffffU) | (f[3][k] << 16U));
                *reinterpret_cast<uint2*>(&v_lds[buf][v_row(feature + 2U * k + 1U)][slot]) =
                    make_uint2((f[0][k] >> 16U) | (f[1][k] & 0xffff0000U), (f[2][k] >> 16U) | (f[3][k] & 0xffff0000U));
            }
            if (lt < 4U) meta_lds[buf][lt] = m;
          }
        };
        // D chunks in flight, slot k of the loop unrolled by D: before step c, r[k] holds chunk c + 1 (fetched
        // with meta held[k]) and raw[k] the meta of chunk c + 1 + D, loaded D chunks before its fetch (so a wait
        // for it covers only loads at least as old as the chunk staged next). Fixed slots (moving a register
        // whose load is in flight would wait for it) and no branch around a load (chunk indices clamp to the
        // last chunk; what the tail stages goes to the buffer the compute waves no longer read).
        if (c0 < c1) {
            const auto chunk = [&](unsigned c) { return c < c1 ? c : c1 - 1U; };
            R r[D];
            uint2 held[D], raw[D];
            {
                R first;
                const uint2 m = meta_at(c0);
                fetch(m, first);
                stage(first, m, 0U);
            }
#pragma unroll
            for (unsigned d = 0; d < D; ++d) { held[d] = meta_at(chunk(c0 + 1U + d)); fetch(held[d], r[d]); }
#pragma unroll
            for (unsigned d = 0; d < D; ++d) raw[d] = meta_at(chunk(c0 + 1U + D + d));
            lds_barrier();
            for (unsigned c = c0; c < c1; c += D) {
#pragma unroll
                for (unsigned k = 0; k < D; ++k) {
                    const unsigned step = c + k;
                    if (step >= c1) break;
                    stage(r[k], held[k], (step - c0 + 1U) & 1U);
                    held[k] = raw[k];
                    fetch(held[k], r[k]);
                    raw[k] = meta_at(chunk(step + 1U + 2U * D));
                    lds_barrier();
                    lds_barrier();
                }
            }
        }
        return;
    }

    const unsigned grp = w / 2U, fh = w % 2U; // column group, feature half
    const unsigned row = col % TR, head = kvh * per_kv + hg * HG + grp * HPW + col / TR, r = r0 + row;
    // Q of this lane's column: a power-of-two scale from the largest |q| of the 256 features (the lane pair
    // (col, col + 16) reads half each), then this wave's 128 features as FP16 B fragments.
    float score_scale;
    f16x16 qf[8];
    {
        const float* q = v.q_gate[r] + static_cast<std::uint64_t>(head) * 2U * kDim;
        float peak = 0.0F;
        for (unsigned i = half * (kDim / 2U); i < (half + 1U) * (kDim / 2U); i += 4U) {
            const float4 x = *reinterpret_cast<const float4*>(q + i);
            peak = fmaxf(peak, fmaxf(fmaxf(fabsf(x.x), fabsf(x.y)), fmaxf(fabsf(x.z), fabsf(x.w))));
        }
        peak = fmaxf(peak, swap16(peak));
        const int e = q_exponent(peak);
        const float scale = ldexpf(1.0F, e);
        // Scores in base 2: the softmax below uses exp2 (v_exp_f32).
        score_scale = ldexpf(1.0F, -e) / sqrtf(static_cast<float>(kDim)) * 1.4426950408889634F;
#pragma unroll
        for (unsigned k = 0; k < 8U; ++k) {
#pragma unroll
            for (unsigned i = 0; i < 16U; i += 4U) {
                const float4 x = *reinterpret_cast<const float4*>(q + fh * (kDim / 2U) + 16U * k + i);
                qf[k][i] = static_cast<_Float16>(x.x * scale); qf[k][i + 1U] = static_cast<_Float16>(x.y * scale);
                qf[k][i + 2U] = static_cast<_Float16>(x.z * scale); qf[k][i + 3U] = static_cast<_Float16>(x.w * scale);
            }
            // Consume the finished fragment before loading the next one. Without this fence,
            // unrolling can keep all 128 FP32 query values live alongside their FP16 results.
            // Empty assembly emits no instructions and changes no values; the memory clobber
            // prevents later query loads from crossing it. All FP statements retain their order.
            asm volatile("" : : "v"(qf[k]) : "memory");
        }
    }
    const unsigned visible = v.visible[r], bit = 1U << row, weights = header[4U + row];
    f32x8 acc[8];
#pragma unroll
    for (unsigned n = 0; n < 8U; ++n) acc[n] = f32x8{};
    float m_run = -INFINITY, l_run = 0.0F;
    if (c0 < c1) lds_barrier();
    for (unsigned c = c0; c < c1; ++c) {
        const unsigned buf = (c - c0) & 1U;
        if (debug & 2U) { lds_barrier(); lds_barrier(); continue; }
        uint2 meta[4];
#pragma unroll
        for (unsigned i = 0; i < 4U; ++i) meta[i] = meta_lds[buf][i];
        strix::attn_core::selected_flash_step(qf, acc, k_lds[buf], v_lds[buf], xch[grp], col, half, fh, lane,
            m_run, l_run, SelectedFlashPolicy<Int8>{meta, bit, visible, weights, half, score_scale, scale_lds[Int8 ? buf : 0U]});
    }
    if (splits == 1U) {
        // One slice: combine_kernel's arithmetic for one split, in place.
        const float* gate_in = v.q_gate[r] + static_cast<std::uint64_t>(head) * 2U * kDim + kDim;
        auto out = v.image(r) + static_cast<std::uint64_t>(head) * kDim;
#pragma unroll
        for (unsigned n = 0; n < 8U; ++n)
#pragma unroll
            for (unsigned e = 0; e < 8U; ++e) {
                const unsigned d = fh * (kDim / 2U) + 16U * n + 2U * e + half;
                const float attn = round_bf16(l_run > 0.0F ? acc[n][e] / l_run : 0.0F);
                const float gate = round_bf16(1.0F / (1.0F + expf(-gate_in[d])));
                out[d] = round_bf16(attn * gate);
            }
        return;
    }
    float* partial = v.scratch[r] + (static_cast<std::uint64_t>(split) * heads + head) * kPartial;
    if (fh == 0U && half == 0U) { partial[0] = m_run; partial[1] = l_run; }
#pragma unroll
    for (unsigned n = 0; n < 8U; ++n)
#pragma unroll
        for (unsigned e = 0; e < 8U; ++e) partial[2U + fh * (kDim / 2U) + 16U * n + 2U * e + half] = acc[n][e];
}

// STRIX_QPF2K_QSA bit 13 (0x2000, with bit 6): attend_order_kernel with a register budget of eight waves per SIMD, so a
// WGP can hold three 10-wave workgroups (3 x 39,936 B LDS fits) instead of two. Register allocation only: the same body
// and grid, with an occ-only query load fence; every workgroup runs the same floating-point statements.
template <unsigned Rows, unsigned TR, unsigned G, unsigned D = 3U>
__global__ __launch_bounds__((2U * G + 4U) * 32U) __attribute__((amdgpu_waves_per_eu(8))) void attend_order_occ_kernel(
    const Views<Rows>* groups, const std::uint16_t* kv, std::uint32_t kv_heads, std::uint32_t heads, std::uint32_t union_at,
    std::uint32_t ys) {
    constexpr unsigned tiles = Rows / TR;
    const unsigned bx = blockIdx.x, tile = bx % tiles, y = (bx / tiles) % ys, g = bx / (tiles * ys);
    attend_occ_body<Rows, TR, G, D, false>(groups[g], kv, kv_heads, heads, union_at, 1U, 0U, tile, y);
}

template <unsigned Rows>
__global__ __launch_bounds__(256) void combine_kernel(Views<Rows> v, std::uint32_t heads, std::uint32_t splits) {
    const unsigned head = blockIdx.x, r = blockIdx.y, d = threadIdx.x;
    const float* partial = v.scratch[r] + static_cast<std::uint64_t>(head) * kPartial;
    const auto stride = static_cast<std::uint64_t>(heads) * kPartial;
    float top = -INFINITY;
    for (unsigned s = 0; s < splits; ++s) top = fmaxf(top, partial[s * stride]); // base-2 maxima
    float l = 0.0F, o = 0.0F;
    if (top != -INFINITY)
        for (unsigned s = 0; s < splits; ++s) {
            const float m = partial[s * stride];
            if (m == -INFINITY) continue;
            const float f = exp2f(m - top);
            l += partial[s * stride + 1U] * f;
            o += partial[s * stride + 2U + d] * f;
        }
    const float attn = round_bf16(l > 0.0F ? o / l : 0.0F);
    const float gate = round_bf16(1.0F / (1.0F + expf(-v.q_gate[r][static_cast<std::uint64_t>(head) * 2U * kDim + kDim + d])));
    v.image(r)[static_cast<std::uint64_t>(head) * kDim + d] = round_bf16(attn * gate);
}
} // namespace qsa_fast
