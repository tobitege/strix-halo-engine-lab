// minimax-m3 MSA decode kernels (gfx1151, wave32). Declared arithmetic: strix/minimax_msa.hpp; every kernel equals the
// host reference bit for bit (tests/hip_minimax_msa_tests.hip).
#include "strix/hip_check.hpp"
#include "strix/hip_minimax_msa.hpp"

#include <stdexcept>
#include <string>

namespace strix::minimax::msa::hip {
namespace {

constexpr unsigned kScoreThreads = kBlock;   // one thread per position, one 128-position block per thread block
constexpr unsigned kSelThreads = 256;
constexpr unsigned kMaxSelBlocks = 8192;     // 1M positions (32 KB of keys in LDS)
constexpr unsigned kAttnWaves = kGroup;  // one wave per query head of the KV group
constexpr unsigned kPartial = 2 + kHeadDim;

__device__ __forceinline__ float butterfly_sum(float v) {
    for (unsigned d = 16; d >= 1; d >>= 1U) v = v + __shfl_xor(v, static_cast<int>(d), 32);
    return v;
}
__device__ __forceinline__ float max_of(float a, float b) { return a < b ? b : a; }  // std::max(a, b)

// Lane l holds dims 4l..4l+3 of one head in x[4]. Norm + RoPE in place (host head_norm_rope order).
__device__ __forceinline__ void norm_rope_lane(float* x, const float* w, const RopeRow& rope, unsigned lane) {
    float acc = 0.0F;
    for (unsigned i = 0; i < kDimsPerLane; ++i) acc = __builtin_fmaf(x[i], x[i], acc);
    const float sum = butterfly_sum(acc);
    const float r = 1.0F / __builtin_sqrtf(sum / static_cast<float>(kHeadDim) + kNormEps);
    for (unsigned i = 0; i < kDimsPerLane; ++i) x[i] = (x[i] * r) * (1.0F + w[lane * kDimsPerLane + i]);
    float partner[kDimsPerLane];
    for (unsigned i = 0; i < kDimsPerLane; ++i) partner[i] = __shfl_xor(x[i], 8, 32);
    if (lane < 8) {  // dims 0..31: a = own, b = dim + 32
        for (unsigned i = 0; i < kDimsPerLane; ++i) {
            const unsigned d = lane * kDimsPerLane + i;
            x[i] = __builtin_fmaf(-partner[i], rope.sin[d], x[i] * rope.cos[d]);
        }
    } else if (lane < 16) {  // dims 32..63: b = own, a = dim - 32
        for (unsigned i = 0; i < kDimsPerLane; ++i) {
            const unsigned d = (lane - 8) * kDimsPerLane + i;
            x[i] = __builtin_fmaf(partner[i], rope.sin[d], x[i] * rope.cos[d]);
        }
    }
}

__global__ __launch_bounds__(32) void prep_kernel(PrepView v) {
    const unsigned r = blockIdx.x, item = blockIdx.y, lane = threadIdx.x;
    const std::uint64_t p = v.pos[r];
    float x[kDimsPerLane];
    auto load = [&](const float* src) {
        for (unsigned i = 0; i < kDimsPerLane; ++i) x[i] = src[lane * kDimsPerLane + i];
    };
    auto store_f = [&](float* dst) {
        for (unsigned i = 0; i < kDimsPerLane; ++i) dst[lane * kDimsPerLane + i] = x[i];
    };
    auto store_b = [&](std::uint16_t* dst) {
        for (unsigned i = 0; i < kDimsPerLane; ++i) dst[lane * kDimsPerLane + i] = float_to_bf16(x[i]);
    };
    unsigned i = item;
    if (i < v.q_heads) {
        float* q = v.q + (static_cast<std::size_t>(r) * v.q_heads + i) * kHeadDim;
        load(q);
        norm_rope_lane(x, v.q_w, v.rope[r], lane);
        store_f(q);
        return;
    }
    i -= v.q_heads;
    if (i < v.kv_heads) {
        load(v.k + (static_cast<std::size_t>(r) * v.kv_heads + i) * kHeadDim);
        norm_rope_lane(x, v.k_w, v.rope[r], lane);
        store_b(v.k_cache + (p * v.kv_heads + i) * kHeadDim);
        return;
    }
    i -= v.kv_heads;
    if (i < v.kv_heads) {
        load(v.v + (static_cast<std::size_t>(r) * v.kv_heads + i) * kHeadDim);
        store_b(v.v_cache + (p * v.kv_heads + i) * kHeadDim);
        return;
    }
    i -= v.kv_heads;
    if (i < v.idx_heads) {
        float* q = v.iq + (static_cast<std::size_t>(r) * v.idx_heads + i) * kIdxDim;
        load(q);
        norm_rope_lane(x, v.iq_w, v.rope[r], lane);
        store_f(q);
        return;
    }
    load(v.ik + static_cast<std::size_t>(r) * kIdxDim);
    norm_rope_lane(x, v.ik_w, v.rope[r], lane);
    store_b(v.ik_cache + p * kIdxDim);
}

// dot128 of the host reference evaluated by ONE thread: 32 lane chains, then the butterfly's lane-0 tree.
__device__ __forceinline__ float dot128_thread(const float* q, const float* k) {
    float part[kLanes];
    for (unsigned l = 0; l < kLanes; ++l) {
        float acc = 0.0F;
        for (unsigned i = 0; i < kDimsPerLane; ++i) acc = __builtin_fmaf(q[l * kDimsPerLane + i], k[l * kDimsPerLane + i], acc);
        part[l] = acc;
    }
    for (unsigned d = 16; d >= 1; d >>= 1U)
        for (unsigned l = 0; l < d; ++l) part[l] = part[l] + part[l + d];
    return part[0];
}

__global__ __launch_bounds__(kScoreThreads) void scores_kernel(ScoreView v) {
    extern __shared__ float sq[];  // R x idx_heads x 128
    __shared__ float wave_max[kScoreThreads / 32];
    const unsigned n_q = v.rows * v.idx_heads * kIdxDim;
    for (unsigned i = threadIdx.x; i < n_q; i += kScoreThreads) sq[i] = v.iq[i];
    __syncthreads();
    // Context-parallel indexer: this shard scores global blocks b = j * shard_count + shard_rank (local index j).
    const std::uint64_t j = blockIdx.x, b = j * v.shard_count + v.shard_rank, t = b * kBlock + threadIdx.x;
    const std::uint64_t last_count = v.pos[v.rows - 1] + 1;
    float key[kIdxDim];
    if (t < last_count)
        for (unsigned d = 0; d < kIdxDim; ++d) key[d] = bf16_to_float(v.ik_cache[t * kIdxDim + d]);
    for (unsigned r = 0; r < v.rows; ++r) {
        const std::uint64_t count = v.pos[r] + 1;
        if (b * kBlock >= count) continue;  // uniform over the block
        for (unsigned h = 0; h < v.idx_heads; ++h) {
            float s = -__builtin_inff();
            if (t < count) s = dot128_thread(sq + (r * v.idx_heads + h) * kIdxDim, key) * kScale;
            for (unsigned d = 16; d >= 1; d >>= 1U) s = max_of(s, __shfl_xor(s, static_cast<int>(d), 32));
            if ((threadIdx.x & 31U) == 0U) wave_max[threadIdx.x >> 5U] = s;
            __syncthreads();
            if (threadIdx.x == 0) {
                float m = wave_max[0];
                for (unsigned w = 1; w < kScoreThreads / 32; ++w) m = max_of(m, wave_max[w]);
                v.scores[(static_cast<std::size_t>(r) * v.idx_heads + h) * v.max_blocks + j] = m;
            }
            __syncthreads();
        }
    }
}

// Exact selection: block b (other than the newest) is chosen when it is among the kTopK - 1 best by (score desc,
// id asc); the newest block is always chosen. Scores become order-preserving keys (-0 canonicalised to +0 so equal
// floats give equal keys); a 4-pass 8-bit radix finds the threshold key and how many ties at it are needed; ties
// and the final list are compacted in ascending id order by a block scan, so the result equals the host's.
__device__ __forceinline__ std::uint32_t order_key(float x) {
    const std::uint32_t b = f2u(x == 0.0F ? 0.0F : x);
    return (b & 0x80000000U) != 0U ? ~b : (b | 0x80000000U);
}
// Inclusive block scan of one flag per thread (256 threads); returns the prefix and the block total.
__device__ __forceinline__ unsigned block_scan(bool flag, unsigned* wave_tot, unsigned& total) {
    const unsigned lane = threadIdx.x & 31U, wave = threadIdx.x >> 5U;
    const unsigned mask = static_cast<unsigned>(__ballot(flag));
    const unsigned in_wave = static_cast<unsigned>(__popc(mask & ((2U << lane) - 1U)));
    if (lane == 31U) wave_tot[wave] = static_cast<unsigned>(__popc(mask));
    __syncthreads();
    unsigned before = 0;
    total = 0;
    for (unsigned w = 0; w < kSelThreads / 32; ++w) {
        if (w < wave) before += wave_tot[w];
        total += wave_tot[w];
    }
    __syncthreads();
    return before + in_wave;
}
// Marks chosen[i] for the k best of n keys by (key desc, id asc), n > k. ids == nullptr means id = index (ascending).
// n <= kSelThreads: exact rank by counting (one thread per candidate). Larger n (ids must ascend with the index):
// 4-pass radix threshold, then the ties at the threshold in index order.
__device__ void choose_topk(const std::uint32_t* keys, const std::uint32_t* ids, std::uint32_t n, std::uint32_t k,
                            std::uint8_t* chosen, unsigned* hist, unsigned* wave_tot, std::uint32_t* sh) {
    if (n <= kSelThreads) {
        const std::uint32_t i = threadIdx.x;
        if (i < n) {
            const std::uint32_t key = keys[i], id = ids != nullptr ? ids[i] : i;
            std::uint32_t beaten = 0;
            for (std::uint32_t o = 0; o < n; ++o) {
                const std::uint32_t ko = keys[o], io = ids != nullptr ? ids[o] : o;
                beaten += (ko > key || (ko == key && io < id)) ? 1U : 0U;
            }
            chosen[i] = beaten < k ? 1U : 0U;
        }
        __syncthreads();
        return;
    }
    std::uint32_t prefix = 0, prefix_mask = 0, need = k;
    for (int shift = 24; shift >= 0; shift -= 8) {
        hist[threadIdx.x] = 0;
        __syncthreads();
        for (unsigned i = threadIdx.x; i < n; i += kSelThreads)
            if ((keys[i] & prefix_mask) == prefix) atomicAdd(&hist[(keys[i] >> shift) & 255U], 1U);
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned above = 0, d = 255;
            for (;; --d) {
                if (above + hist[d] >= need) break;
                above += hist[d];
            }
            sh[0] = prefix | (d << shift);
            sh[1] = need - above;
        }
        __syncthreads();
        prefix = sh[0];
        need = sh[1];
        prefix_mask |= 255U << shift;
        __syncthreads();
    }
    unsigned ties_before = 0;
    for (std::uint32_t base = 0; base < n; base += kSelThreads) {
        const std::uint32_t i = base + threadIdx.x;
        const std::uint32_t key = i < n ? keys[i] : 0U;
        const bool tie = i < n && key == prefix;
        unsigned tie_total = 0;
        const unsigned tie_rank = ties_before + block_scan(tie, wave_tot, tie_total);  // inclusive
        if (i < n) chosen[i] = (key > prefix || (tie && tie_rank <= need)) ? 1U : 0U;
        ties_before += tie_total;
    }
    __syncthreads();
}
// Calls f(i, slot) for every chosen i, slots in index order (block scan compaction); returns the count.
template <class F> __device__ unsigned compact_chosen(const std::uint8_t* chosen, std::uint32_t n, unsigned* wave_tot, F&& f) {
    unsigned before = 0;
    for (std::uint32_t base = 0; base < n; base += kSelThreads) {
        const std::uint32_t i = base + threadIdx.x;
        const bool c = i < n && chosen[i] != 0U;
        unsigned total = 0;
        const unsigned rank = before + block_scan(c, wave_tot, total);
        if (c) f(i, rank - 1);
        before += total;
    }
    return before;
}

struct SelScratch {
    std::uint32_t keys[kMaxSelBlocks];
    std::uint32_t ids[2 * kTopK];
    std::uint8_t chosen[kMaxSelBlocks];
    unsigned hist[256];
    unsigned wave_tot[kSelThreads / 32];
    std::uint32_t sh[2];
};

__global__ __launch_bounds__(kSelThreads) void select_kernel(SelectView v) {
    __shared__ SelScratch S;
    const unsigned rh = blockIdx.x, r = rh / v.heads;
    const std::uint32_t nb = static_cast<std::uint32_t>(block_count(v.pos[r] + 1));
    std::uint32_t* idx = v.idx + static_cast<std::size_t>(rh) * kTopK;
    if (nb <= kTopK) {
        for (unsigned b = threadIdx.x; b < nb; b += kSelThreads) idx[b] = b;
        if (threadIdx.x == 0) v.idx_count[rh] = nb;
        return;
    }
    const float* sc = v.scores + static_cast<std::size_t>(rh) * v.max_blocks;
    const std::uint32_t rest = nb - 1;
    for (unsigned b = threadIdx.x; b < rest; b += kSelThreads) S.keys[b] = order_key(sc[b]);
    __syncthreads();
    choose_topk(S.keys, nullptr, rest, kTopK - 1, S.chosen, S.hist, S.wave_tot, S.sh);
    (void)compact_chosen(S.chosen, rest, S.wave_tot, [&](std::uint32_t i, unsigned o) { idx[o] = i; });
    if (threadIdx.x == 0) {
        idx[kTopK - 1] = rest;
        v.idx_count[rh] = kTopK;
    }
}

// Context-parallel part: the best kTopK - 1 of this shard's blocks (the newest block excluded), as (key, global id)
// in ascending id order. Every block of the global top kTopK - 1 is in its shard's list, so the merge is exact.
__global__ __launch_bounds__(kSelThreads) void candidates_kernel(CandidateView v) {
    __shared__ SelScratch S;
    const unsigned rh = blockIdx.x, r = rh / v.heads;
    const std::uint32_t nb = static_cast<std::uint32_t>(block_count(v.pos[r] + 1));
    const std::uint32_t rest = nb - 1, P = v.shard_count, R = v.shard_rank;
    const std::uint32_t n = rest > R ? (rest - R + P - 1) / P : 0U;  // local blocks with global id < rest
    std::uint32_t* ck = v.cand_key + static_cast<std::size_t>(rh) * kTopK;
    std::uint32_t* ci = v.cand_id + static_cast<std::size_t>(rh) * kTopK;
    if (nb <= kTopK) {
        if (threadIdx.x == 0) v.cand_count[rh] = 0;
        return;
    }
    const float* sc = v.scores + static_cast<std::size_t>(rh) * v.max_blocks;
    for (unsigned i = threadIdx.x; i < n; i += kSelThreads) {
        S.keys[i] = order_key(sc[i]);
        S.chosen[i] = 1U;
    }
    __syncthreads();
    if (n > kTopK - 1) choose_topk(S.keys, nullptr, n, kTopK - 1, S.chosen, S.hist, S.wave_tot, S.sh);
    const unsigned c = compact_chosen(S.chosen, n, S.wave_tot, [&](std::uint32_t i, unsigned o) {
        ck[o] = S.keys[i];
        ci[o] = i * P + R;
    });
    if (threadIdx.x == 0) v.cand_count[rh] = c;
}

// Merge of the shards' candidates: the best kTopK - 1 by (key desc, id asc), plus the newest block, ascending.
__global__ __launch_bounds__(kSelThreads) void merge_kernel(MergeView v) {
    __shared__ SelScratch S;
    __shared__ std::uint32_t out[kTopK];
    const unsigned rh = blockIdx.x, r = rh / v.heads;
    const std::uint32_t nb = static_cast<std::uint32_t>(block_count(v.pos[r] + 1));
    std::uint32_t* idx = v.idx + static_cast<std::size_t>(rh) * kTopK;
    if (nb <= kTopK) {
        for (unsigned b = threadIdx.x; b < nb; b += kSelThreads) idx[b] = b;
        if (threadIdx.x == 0) v.idx_count[rh] = nb;
        return;
    }
    if (threadIdx.x == 0) {
        std::uint32_t n = 0;
        for (unsigned s = 0; s < 2; ++s) {
            const std::uint32_t c = v.cand_count[s][rh];
            for (std::uint32_t i = 0; i < c; ++i) {
                S.keys[n] = v.cand_key[s][static_cast<std::size_t>(rh) * kTopK + i];
                S.ids[n] = v.cand_id[s][static_cast<std::size_t>(rh) * kTopK + i];
                ++n;
            }
        }
        S.sh[0] = n;
    }
    __syncthreads();
    const std::uint32_t n = S.sh[0];
    for (unsigned i = threadIdx.x; i < n; i += kSelThreads) S.chosen[i] = 1U;
    __syncthreads();
    if (n > kTopK - 1) choose_topk(S.keys, S.ids, n, kTopK - 1, S.chosen, S.hist, S.wave_tot, S.sh);
    if (threadIdx.x == 0) {
        unsigned c = 0;
        for (std::uint32_t i = 0; i < n; ++i)
            if (S.chosen[i] != 0U) {
                const std::uint32_t id = S.ids[i];
                unsigned at = c++;
                while (at > 0 && out[at - 1] > id) {  // insertion sort of at most 15 ids
                    out[at] = out[at - 1];
                    --at;
                }
                out[at] = id;
            }
        for (unsigned i = 0; i < c; ++i) idx[i] = out[i];
        idx[c] = nb - 1;
        v.idx_count[rh] = c + 1;
    }
}

// One thread block per (row, KV head, selected block): wave w runs query head w of the group. Scores: lane j owns key
// j of a group of 32 and evaluates dot128 in the declared order alone (no shuffles); the online softmax then walks the
// tiles of 16 keys in key order with the scores and probabilities broadcast from their lanes.
__global__ __launch_bounds__(kAttnWaves * 32) void attention_chunk_kernel(AttentionView v) {
    __shared__ float qs[kGroup * kHeadDim];
    const unsigned rg = blockIdx.x, r = rg / v.kv_heads, g = rg % v.kv_heads, c = blockIdx.y;
    const unsigned wave = threadIdx.x >> 5U, lane = threadIdx.x & 31U;
    const std::uint32_t cnt = v.idx_count[rg];
    if (c >= cnt) return;
    const unsigned q_heads = v.kv_heads * kGroup;
    const float* qg = v.q + (static_cast<std::size_t>(r) * q_heads + g * kGroup) * kHeadDim;
    for (unsigned i = threadIdx.x; i < kGroup * kHeadDim; i += kAttnWaves * 32) qs[i] = qg[i];
    __syncthreads();
    const std::uint64_t count = v.pos[r] + 1;
    const std::uint64_t lo = static_cast<std::uint64_t>(v.idx[static_cast<std::size_t>(rg) * kTopK + c]) * kBlock;
    const unsigned n_keys = static_cast<unsigned>((count < lo + kBlock ? count : lo + kBlock) - lo);
    const unsigned qh = g * kGroup + wave;
    const float* q = qs + wave * kHeadDim;
    float acc[kDimsPerLane] = {0.0F, 0.0F, 0.0F, 0.0F};
    float m = -__builtin_inff(), l = 0.0F;
    bool first = true;
    for (unsigned k0 = 0; k0 < n_keys; k0 += 32) {
        float s = -__builtin_inff();
        if (k0 + lane < n_keys) {
            // Keys are read per lane from the cache (L2 hits across the 16 waves). A shared LDS key tile was
            // measured slower (measurement: R1 8K 48 -> 55 us, R8 189 -> 209 us): the extra barrier costs more than it saves.
            const std::uint16_t* kr = v.k_cache + ((lo + k0 + lane) * v.kv_heads + g) * kHeadDim;
            float part[kLanes];
            for (unsigned ll = 0; ll < kLanes; ++ll) {
                const uint2 raw = *reinterpret_cast<const uint2*>(kr + ll * kDimsPerLane);
                const float kv[4] = {u2f(raw.x << 16U), u2f(raw.x & 0xFFFF0000U), u2f(raw.y << 16U), u2f(raw.y & 0xFFFF0000U)};
                float a = 0.0F;
                for (unsigned i = 0; i < kDimsPerLane; ++i) a = __builtin_fmaf(q[ll * kDimsPerLane + i], kv[i], a);
                part[ll] = a;
            }
            for (unsigned d = 16; d >= 1; d >>= 1U)
                for (unsigned ll = 0; ll < d; ++ll) part[ll] = part[ll] + part[ll + d];
            s = part[0] * kScale;
        }
        for (unsigned t0 = k0; t0 < k0 + 32 && t0 < n_keys; t0 += kTile) {
            const unsigned nt = n_keys - t0 < kTile ? n_keys - t0 : kTile;
            float tm = -__builtin_inff();
            for (unsigned j = 0; j < nt; ++j) tm = max_of(tm, __shfl(s, static_cast<int>(t0 - k0 + j), 32));
            const float mn = max_of(m, tm);
            const float corr = first ? 0.0F : exp_f(m - mn);
            l = l * corr;
            for (unsigned i = 0; i < kDimsPerLane; ++i) acc[i] = acc[i] * corr;
            const float p_own = exp_f(s - mn);
            for (unsigned j = 0; j < nt; ++j) {
                const float pj = __shfl(p_own, static_cast<int>(t0 - k0 + j), 32);
                l = l + pj;
                const std::uint16_t* vr = v.v_cache + ((lo + t0 + j) * v.kv_heads + g) * kHeadDim + lane * kDimsPerLane;
                const uint2 raw = *reinterpret_cast<const uint2*>(vr);
                const float vv[4] = {u2f(raw.x << 16U), u2f(raw.x & 0xFFFF0000U), u2f(raw.y << 16U), u2f(raw.y & 0xFFFF0000U)};
                for (unsigned i = 0; i < kDimsPerLane; ++i) acc[i] = __builtin_fmaf(pj, vv[i], acc[i]);
            }
            m = mn;
            first = false;
        }
    }
    float* out = v.partial + ((static_cast<std::size_t>(r) * q_heads + qh) * kMaxChunks + c) * kPartial;
    if (lane == 0) {
        out[0] = m;
        out[1] = l;
    }
    for (unsigned i = 0; i < kDimsPerLane; ++i) out[2 + lane * kDimsPerLane + i] = acc[i];
}

__global__ __launch_bounds__(32) void attention_combine_kernel(AttentionView v) {
    const unsigned q_heads = v.kv_heads * kGroup;
    const unsigned rq = blockIdx.x, r = rq / q_heads, qh = rq % q_heads, g = qh / kGroup, lane = threadIdx.x;
    const std::uint32_t cnt = v.idx_count[r * v.kv_heads + g];
    const unsigned chunks = (cnt + kChunkBlocks - 1) / kChunkBlocks;
    const float* part = v.partial + static_cast<std::size_t>(rq) * kMaxChunks * kPartial;
    float big = -__builtin_inff();
    for (unsigned c = 0; c < chunks; ++c) big = max_of(big, part[c * kPartial]);
    float L = 0.0F, e[kMaxChunks];
    for (unsigned c = 0; c < chunks; ++c) {
        e[c] = exp_f(part[c * kPartial] - big);
        L = __builtin_fmaf(part[c * kPartial + 1], e[c], L);
    }
    float* o = v.o + static_cast<std::size_t>(rq) * kHeadDim;
    for (unsigned i = 0; i < kDimsPerLane; ++i) {
        const unsigned d = lane * kDimsPerLane + i;
        float acc = 0.0F;
        for (unsigned c = 0; c < chunks; ++c) acc = __builtin_fmaf(part[c * kPartial + 2 + d], e[c], acc);
        o[d] = acc / L;
    }
}

using strix::hip_check;

}  // namespace

void enqueue_prep(const PrepView& v, hipStream_t stream) {
    const unsigned items = v.q_heads + 2 * v.kv_heads + v.idx_heads + 1;
    hipLaunchKernelGGL(prep_kernel, dim3(v.rows, items), dim3(32), 0, stream, v);
    hip_check(hipGetLastError(), "minimax prep");
}

void enqueue_scores(const ScoreView& v, hipStream_t stream) {
    // One thread block per 128 positions of this shard; rows whose count ends earlier skip uniformly.
    if (v.shard_count == 0 || v.shard_rank >= v.shard_count) throw std::invalid_argument("minimax scores: bad shard");
    const unsigned local = (v.max_blocks + v.shard_count - 1 - v.shard_rank) / v.shard_count;
    const std::size_t lds = static_cast<std::size_t>(v.rows) * v.idx_heads * kIdxDim * sizeof(float);
    if (local != 0) hipLaunchKernelGGL(scores_kernel, dim3(local), dim3(kScoreThreads), lds, stream, v);
    hip_check(hipGetLastError(), "minimax scores");
}

void enqueue_select(const SelectView& v, hipStream_t stream) {
    if (v.max_blocks > kMaxSelBlocks) throw std::invalid_argument("minimax select: more than 8192 blocks");
    hipLaunchKernelGGL(select_kernel, dim3(v.rows * v.heads), dim3(kSelThreads), 0, stream, v);
    hip_check(hipGetLastError(), "minimax select");
}

void enqueue_candidates(const CandidateView& v, hipStream_t stream) {
    if (v.max_blocks > kMaxSelBlocks || v.shard_count == 0 || v.shard_rank >= v.shard_count)
        throw std::invalid_argument("minimax candidates: bad view");
    hipLaunchKernelGGL(candidates_kernel, dim3(v.rows * v.heads), dim3(kSelThreads), 0, stream, v);
    hip_check(hipGetLastError(), "minimax candidates");
}

void enqueue_merge(const MergeView& v, hipStream_t stream) {
    hipLaunchKernelGGL(merge_kernel, dim3(v.rows * v.heads), dim3(kSelThreads), 0, stream, v);
    hip_check(hipGetLastError(), "minimax merge");
}

void enqueue_attention(const AttentionView& v, hipStream_t stream) {
    hipLaunchKernelGGL(attention_chunk_kernel, dim3(v.rows * v.kv_heads, kMaxChunks), dim3(kAttnWaves * 32), 0, stream, v);
    hip_check(hipGetLastError(), "minimax attention chunk");
    hipLaunchKernelGGL(attention_combine_kernel, dim3(v.rows * v.kv_heads * kGroup), dim3(32), 0, stream, v);
    hip_check(hipGetLastError(), "minimax attention combine");
}

}  // namespace strix::minimax::msa::hip
