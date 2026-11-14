#include "strix/hip_check.hpp"
#include "strix/hip_token_sampling_device.hpp"
#include "strix/hip_token_sampling.hpp"

#include <hip/hip_runtime.h>
#include "strix/env_flags.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>

namespace strix {
namespace {

// Rows are split into kBlocks workgroups of kThreads threads; thread t of block b owns the contiguous index
// range [(b * kThreads + t) * per, +per), so every scan below runs in token id order.
constexpr unsigned kThreads = 256, kBlocks = 32, kMaxRows = 8, kCand = device_top_p_candidates, kSmallThreads = 1024;
// Top-k up to kTopkOnePass: each workgroup keeps its own top-k, one workgroup merges the kBlocks lists.
constexpr unsigned kTopkOnePass = kCand / kBlocks;
constexpr std::uint32_t kNone = 0xffffffffU;
enum : std::uint32_t {
    flag_nonfinite = 1U, flag_host = 2U, flag_drafted = 4U, flag_accepted = 8U, flag_residual = 16U,
    flag_top_p = 32U, flag_need_residual = 64U, flag_done = 128U, flag_small = 256U, flag_bad_draft = 512U,
};
using u64 = unsigned long long;
using u128 = unsigned __int128;

// Index i of a row is its i-th entry in token id order: entry order[i] of the logits, token ids[i]. Target rows
// (the full vocabulary) have neither map; draft rows (the draft head's entries) have both.
struct RowArgs {
    const float* logits = nullptr;
    const std::uint32_t* order = nullptr;
    const std::uint32_t* ids = nullptr;
    const std::uint32_t* q_tokens = nullptr; // draft distribution: token ids ascending, zero entries allowed
    const double* q_probs = nullptr;
    double* q_out = nullptr;                 // draft rows: the distribution per index, 0 outside it
    std::uint32_t* token_out = nullptr;      // draft rows: device copy of the sampled token
    std::uint32_t draft = kNone, q_count = 0;
    double u_plain = 0.0, u_accept = 0.0, u_residual = 0.0;
};
struct Rows { RowArgs row[kMaxRows]; };
struct Policy {
    std::uint32_t vocab = 0, top_k = 0, top_p_active = 0, rows = 0;
    double temperature = 1.0, cutoff = -40.0, top_p = 1.0;
};
// Radix selection of the m-th largest order key: after four 8-bit rounds `key` is the m-th key and `need` the
// number of elements equal to it that belong to the m largest (lowest token ids first).
struct Select { std::uint32_t active, prefix, mask, remaining; };
// Mass quantile in (key desc, token id asc) order over large rows: the first entry whose cumulative value reaches
// (kind 0, top-p boundary) or exceeds (kinds 1 and 2, inverse CDF) a target. Values are summed in 80-bit fixed
// point (units of 2^-80, two 64-bit accumulators), so every sum is exact and independent of the summation order.
struct Quant {
    u128 target, above, total, cum;
    double scale;           // kind 0: top_p; kinds 1 and 2: the uniform
    std::uint32_t active, kind, prefix, mask, index, zero, pad0, pad1;
};
struct RowState {
    float max;
    std::uint32_t flags, cut_count, kept_count, last_kept, last_residual, prefix_key, prefix_index;
    double total, mass;     // large rows: kept weight total; top-p prefix mass
    Select topk;
    Quant quant;
};
struct Candidate { std::uint32_t key, id; double w; };
struct Scratch {
    float* block_max;
    std::uint32_t *block_count, *block_eq, *hist;
    double *weights, *block_sum, *block_residual, *rvals;
    u64 *qhist, *qtie;
    Candidate* candidates;
    RowState* state;
};
struct Result { std::uint32_t token, flags; };

__device__ __forceinline__ std::uint32_t order_key(float x) {
    const auto u = __float_as_uint(x);
    return (u & 0x80000000U) ? ~u : (u | 0x80000000U);
}
__device__ __forceinline__ float logit_at(const RowArgs& a, std::uint32_t i) { return a.logits[a.order ? a.order[i] : i]; }
__device__ __forceinline__ std::uint32_t token_at(const RowArgs& a, std::uint32_t i) { return a.ids ? a.ids[i] : i; }
__device__ __forceinline__ std::uint32_t per_thread(std::uint32_t vocab) {
    return (vocab + kBlocks * kThreads - 1U) / (kBlocks * kThreads);
}
__device__ __forceinline__ void thread_range(std::uint32_t vocab, unsigned block, unsigned thread, std::uint32_t& begin,
    std::uint32_t& end) {
    const auto per = per_thread(vocab);
    const auto first = (block * kThreads + thread) * per;
    begin = first < vocab ? first : vocab;
    end = begin + per < vocab ? begin + per : vocab;
}
// sampling-v1 step 1a/2: z = (logit - max logit) / temperature in double.
__device__ __forceinline__ double z_of(float logit, float max, double temperature) {
    return (static_cast<double>(logit) - static_cast<double>(max)) / temperature;
}
// Fixed point: x >= 0 (below 2^23) as hi * 2^-40 + lo * 2^-80, truncated below 2^-80.
__device__ __forceinline__ void fp_split(double x, u64& hi, u64& lo) {
    const double s = x * 0x1p40;
    const double h = floor(s);
    hi = static_cast<u64>(h);
    lo = static_cast<u64>(floor((s - h) * 0x1p40));
}
__device__ __forceinline__ u128 fp_join(u64 hi, u64 lo) { return (static_cast<u128>(hi) << 40U) + lo; }
__device__ __forceinline__ u128 fp_of(double x) { u64 hi, lo; fp_split(x, hi, lo); return fp_join(hi, lo); }
__device__ __forceinline__ double fp_double(u128 v) {
    return static_cast<double>(static_cast<u64>(v >> 64U)) * 0x1p-16 + static_cast<double>(static_cast<u64>(v)) * 0x1p-80;
}
// Deterministic block reductions: a fixed LDS tree.
template <typename T, typename Op> __device__ T block_reduce(T value, T* lds, Op op) {
    lds[threadIdx.x] = value;
    __syncthreads();
    for (unsigned s = kThreads / 2U; s > 0U; s >>= 1U) {
        if (threadIdx.x < s) lds[threadIdx.x] = op(lds[threadIdx.x], lds[threadIdx.x + s]);
        __syncthreads();
    }
    const T out = lds[0];
    __syncthreads();
    return out;
}
// Sum of per-thread values in thread order (thread 0 adds them sequentially): the order the searches use.
__device__ double block_sequential_sum(double value, double* lds) {
    lds[threadIdx.x] = value;
    __syncthreads();
    if (threadIdx.x == 0U) { double sum = 0.0; for (unsigned t = 0; t < kThreads; ++t) sum += lds[t]; lds[0] = sum; }
    __syncthreads();
    const double out = lds[0];
    __syncthreads();
    return out;
}
// Exclusive prefix of per-thread counts in thread order (thread 0 sums sequentially).
__device__ std::uint32_t block_exclusive(std::uint32_t value, std::uint32_t* lds) {
    lds[threadIdx.x] = value;
    __syncthreads();
    if (threadIdx.x == 0U) {
        std::uint32_t sum = 0;
        for (unsigned t = 0; t < kThreads; ++t) { const auto v = lds[t]; lds[t] = sum; sum += v; }
    }
    __syncthreads();
    const auto out = lds[threadIdx.x];
    __syncthreads();
    return out;
}

__global__ __launch_bounds__(kThreads) void sample_max_kernel(Rows rows, Policy policy, Scratch s) {
    const auto row = blockIdx.y;
    const RowArgs& a = rows.row[row];
    std::uint32_t begin, end;
    thread_range(policy.vocab, blockIdx.x, threadIdx.x, begin, end);
    float m = -INFINITY;
    bool bad = false;
    for (auto i = begin; i < end; ++i) { const float v = logit_at(a, i); bad = bad || !isfinite(v); m = fmaxf(m, v); }
    __shared__ float lds[kThreads];
    const float block = block_reduce(m, lds, [](float x, float y) { return fmaxf(x, y); });
    if (bad) atomicOr(&s.state[row].flags, flag_nonfinite);
    if (threadIdx.x == 0U) s.block_max[row * kBlocks + blockIdx.x] = block;
}
__device__ float row_max(const Scratch& s, unsigned row) {
    float m = -INFINITY;
    for (unsigned b = 0; b < kBlocks; ++b) m = fmaxf(m, s.block_max[row * kBlocks + b]);
    return m;
}
// Elements passing the cutoff, per block.
__global__ __launch_bounds__(kThreads) void sample_count_kernel(Rows rows, Policy policy, Scratch s) {
    const auto row = blockIdx.y;
    const RowArgs& a = rows.row[row];
    const float max = row_max(s, row);
    std::uint32_t begin, end;
    thread_range(policy.vocab, blockIdx.x, threadIdx.x, begin, end);
    std::uint32_t n = 0;
    for (auto i = begin; i < end; ++i) n += z_of(logit_at(a, i), max, policy.temperature) >= policy.cutoff ? 1U : 0U;
    __shared__ std::uint32_t lds[kThreads];
    const auto total = block_reduce(n, lds, [](std::uint32_t x, std::uint32_t y) { return x + y; });
    if (threadIdx.x == 0U) s.block_count[row * kBlocks + blockIdx.x] = total;
    if (blockIdx.x == 0U && threadIdx.x == 0U) s.state[row].max = max;
}
// After the counts: top-k selection state, and whether the kept set is small enough for one workgroup.
__global__ void sample_after_count_kernel(Policy policy, Scratch s) {
    const auto row = blockIdx.x;
    auto& st = s.state[row];
    std::uint32_t cut = 0;
    for (unsigned b = 0; b < kBlocks; ++b) cut += s.block_count[row * kBlocks + b];
    st.cut_count = cut;
    const bool topk = policy.top_k && cut > policy.top_k;
    st.topk = {topk ? 1U : 0U, 0U, 0U, policy.top_k};
    if ((topk ? policy.top_k : cut) <= kCand) st.flags |= flag_small;
}
// One radix round of the top-k selection over every finite element.
__global__ __launch_bounds__(kThreads) void sample_hist_kernel(Rows rows, Policy policy, Scratch s, unsigned shift) {
    const auto row = blockIdx.y;
    const auto sel = s.state[row].topk;
    if (!sel.active) return;
    __shared__ std::uint32_t hist[256];
    hist[threadIdx.x] = 0U;
    __syncthreads();
    const RowArgs& a = rows.row[row];
    std::uint32_t begin, end;
    thread_range(policy.vocab, blockIdx.x, threadIdx.x, begin, end);
    for (auto i = begin; i < end; ++i) {
        const auto key = order_key(logit_at(a, i));
        if ((key & sel.mask) == sel.prefix) atomicAdd(&hist[(key >> shift) & 255U], 1U);
    }
    __syncthreads();
    s.hist[(static_cast<std::uint64_t>(row) * kBlocks + blockIdx.x) * 256U + threadIdx.x] = hist[threadIdx.x];
}
__global__ __launch_bounds__(kThreads) void sample_pick_kernel(Scratch s, unsigned shift) {
    const auto row = blockIdx.x;
    auto& sel = s.state[row].topk;
    if (!sel.active) return;
    __shared__ std::uint32_t counts[256];
    std::uint32_t c = 0;
    for (unsigned b = 0; b < kBlocks; ++b) c += s.hist[(static_cast<std::uint64_t>(row) * kBlocks + b) * 256U + threadIdx.x];
    counts[threadIdx.x] = c;
    __syncthreads();
    if (threadIdx.x == 0U) {
        std::uint32_t above = 0;
        for (int bin = 255; bin >= 0; --bin) {
            if (above + counts[bin] >= sel.remaining) {
                sel.remaining -= above;
                sel.prefix |= static_cast<std::uint32_t>(bin) << shift;
                sel.mask |= 255U << shift;
                break;
            }
            above += counts[bin];
        }
    }
}
// Elements equal to the selected key, per block.
__global__ __launch_bounds__(kThreads) void sample_equal_kernel(Rows rows, Policy policy, Scratch s) {
    const auto row = blockIdx.y;
    const auto sel = s.state[row].topk;
    if (!sel.active) return;
    const RowArgs& a = rows.row[row];
    std::uint32_t begin, end;
    thread_range(policy.vocab, blockIdx.x, threadIdx.x, begin, end);
    std::uint32_t n = 0;
    for (auto i = begin; i < end; ++i) n += order_key(logit_at(a, i)) == sel.prefix ? 1U : 0U;
    __shared__ std::uint32_t lds[kThreads];
    const auto total = block_reduce(n, lds, [](std::uint32_t x, std::uint32_t y) { return x + y; });
    if (threadIdx.x == 0U) s.block_eq[row * kBlocks + blockIdx.x] = total;
}
// Rank (among elements equal to the selected key, token id order) of the first element of this thread's range.
__device__ std::uint32_t equal_rank_base(const Scratch& s, const Select& sel, const RowArgs& a, std::uint32_t begin,
    std::uint32_t end, unsigned row, std::uint32_t* lds) {
    std::uint32_t mine = 0;
    if (sel.active) for (auto i = begin; i < end; ++i) mine += order_key(logit_at(a, i)) == sel.prefix ? 1U : 0U;
    std::uint32_t base = block_exclusive(mine, lds);
    if (sel.active) for (unsigned b = 0; b < blockIdx.x; ++b) base += s.block_eq[row * kBlocks + b];
    return base;
}
__device__ __forceinline__ bool selected(const Select& sel, std::uint32_t key, std::uint32_t& equal_rank) {
    if (!sel.active) return true;
    if (key > sel.prefix) return true;
    if (key < sel.prefix) return false;
    return equal_rank++ < sel.remaining;
}
__device__ double q_probability(const RowArgs& a, std::uint32_t token) {
    std::uint32_t lo = 0, hi = a.q_count;
    while (lo < hi) { const auto mid = (lo + hi) / 2U; if (a.q_tokens[mid] < token) lo = mid + 1U; else hi = mid; }
    return lo < a.q_count && a.q_tokens[lo] == token ? a.q_probs[lo] : 0.0;
}
__device__ __forceinline__ bool candidate_before(const Candidate& a, const Candidate& b) {
    return a.key > b.key || (a.key == b.key && a.id < b.id);
}
// Bitonic sort of c[0, size) (size a power of two) by (key desc, id asc), all threads of the block.
__device__ void sort_candidates(Candidate* c, unsigned size) {
    for (unsigned span = 2U; span <= size; span <<= 1U)
        for (unsigned stride = span >> 1U; stride > 0U; stride >>= 1U) {
            for (unsigned j = threadIdx.x; j < size; j += blockDim.x) {
                const auto partner = j ^ stride;
                if (partner > j) {
                    const bool ascending = (j & span) == 0U;
                    const bool swap = ascending ? candidate_before(c[partner], c[j]) : candidate_before(c[j], c[partner]);
                    if (swap) { const auto t = c[j]; c[j] = c[partner]; c[partner] = t; }
                }
            }
            __syncthreads();
        }
}
// Bitonic sort of v[0, size) descending.
__device__ void sort_desc(u64* v, unsigned size) {
    for (unsigned span = 2U; span <= size; span <<= 1U)
        for (unsigned stride = span >> 1U; stride > 0U; stride >>= 1U) {
            for (unsigned j = threadIdx.x; j < size; j += blockDim.x) {
                const auto partner = j ^ stride;
                if (partner > j) {
                    const bool descending = (j & span) == 0U;
                    const bool swap = descending ? v[j] < v[partner] : v[partner] < v[j];
                    if (swap) { const auto t = v[j]; v[j] = v[partner]; v[partner] = t; }
                }
            }
            __syncthreads();
        }
}

// The kept entries c[0, n) of a row (token id order, w = exp z, room for the next power of two) decided by the
// whole workgroup exactly as the host does: the total summed in token id order, the top-p prefix of the
// (w desc, id asc) order and its mass, the cumulative probabilities, the acceptance test and the residual walk.
__device__ void decide_kept(const RowArgs& a, const Policy& policy, Candidate* c, std::uint32_t n, Result* result) {
    __shared__ double total_shared;
    unsigned size = 2U;
    while (size < n) size <<= 1U;
    for (unsigned j = n + threadIdx.x; j < size; j += blockDim.x) c[j] = Candidate{0U, kNone, 0.0};
    if (a.q_out) for (std::uint32_t j = threadIdx.x; j < policy.vocab; j += blockDim.x) a.q_out[j] = 0.0;
    __syncthreads();
    if (threadIdx.x == 0U) { double total = 0.0; for (std::uint32_t j = 0; j < n; ++j) total += c[j].w; total_shared = total; }
    __syncthreads();
    const double total = total_shared;
    if (policy.top_p_active) sort_candidates(c, size);
    if (threadIdx.x != 0U) return;
    std::uint32_t count = n;
    double norm = total;
    if (policy.top_p_active) {
        const double target = policy.top_p * total;
        double mass = 0.0;
        count = 0;
        while (count < n) { mass += c[count].w; ++count; if (mass >= target) break; }
        norm = mass;
    }
    const auto p_of = [&](std::uint32_t j) { return c[j].w / norm; };
    const auto sample = [&](double u) {
        double cumulative = 0.0;
        for (std::uint32_t j = 0; j < count; ++j) { cumulative += p_of(j); if (u < cumulative) return token_at(a, c[j].id); }
        return token_at(a, c[count - 1U].id);
    };
    std::uint32_t token = kNone, flags = flag_done;
    if (a.draft == kNone) token = sample(a.u_plain);
    else {
        flags |= flag_drafted;
        const double qd = q_probability(a, a.draft);
        double pd = 0.0;
        for (std::uint32_t j = 0; j < count; ++j) if (token_at(a, c[j].id) == a.draft) { pd = p_of(j); break; }
        if (!(qd > 0.0)) { token = a.draft; flags |= flag_bad_draft; }
        else if (a.u_accept * qd < pd) { token = a.draft; flags |= flag_accepted; }
        else {
            flags |= flag_residual;
            double mass = 0.0;
            for (std::uint32_t j = 0; j < count; ++j) mass += fmax(0.0, p_of(j) - q_probability(a, token_at(a, c[j].id)));
            if (!(mass > 0.0)) token = sample(a.u_residual);
            else {
                const double target = a.u_residual * mass;
                double cumulative = 0.0;
                std::uint32_t last = a.draft;
                for (std::uint32_t j = 0; j < count; ++j) {
                    const auto t = token_at(a, c[j].id);
                    const double r = fmax(0.0, p_of(j) - q_probability(a, t));
                    if (!(r > 0.0)) continue;
                    cumulative += r; last = t;
                    if (target < cumulative) { token = t; break; }
                }
                if (token == kNone) token = last;
            }
        }
    }
    if (a.q_out) for (std::uint32_t j = 0; j < count; ++j) a.q_out[c[j].id] = p_of(j);
    if (a.token_out) *a.token_out = token;
    *result = {token, flags};
}

// Small rows: at most kCand entries pass the cutoff and top-k. They are compacted in token order and decided by
// decide_kept.
__global__ __launch_bounds__(kThreads) void sample_small_count_kernel(Rows rows, Policy policy, Scratch s) {
    const auto row = blockIdx.y;
    const auto& st = s.state[row];
    if ((st.flags & (flag_small | flag_nonfinite)) != flag_small) return;
    const RowArgs& a = rows.row[row];
    std::uint32_t begin, end;
    thread_range(policy.vocab, blockIdx.x, threadIdx.x, begin, end);
    __shared__ std::uint32_t ranks[kThreads];
    const auto topk = st.topk;
    const float max = st.max;
    auto rank = equal_rank_base(s, topk, a, begin, end, row, ranks);
    std::uint32_t n = 0;
    for (auto i = begin; i < end; ++i) {
        const float l = logit_at(a, i);
        n += z_of(l, max, policy.temperature) >= policy.cutoff && selected(topk, order_key(l), rank) ? 1U : 0U;
    }
    __shared__ std::uint32_t counts[kThreads];
    const auto total = block_reduce(n, counts, [](std::uint32_t x, std::uint32_t y) { return x + y; });
    if (threadIdx.x == 0U) s.block_count[row * kBlocks + blockIdx.x] = total;
}
__global__ __launch_bounds__(kThreads) void sample_small_collect_kernel(Rows rows, Policy policy, Scratch s) {
    const auto row = blockIdx.y;
    const auto& st = s.state[row];
    if ((st.flags & (flag_small | flag_nonfinite)) != flag_small) return;
    const RowArgs& a = rows.row[row];
    std::uint32_t begin, end;
    thread_range(policy.vocab, blockIdx.x, threadIdx.x, begin, end);
    __shared__ std::uint32_t ranks[kThreads];
    const auto topk = st.topk;
    const float max = st.max;
    auto rank = equal_rank_base(s, topk, a, begin, end, row, ranks);
    auto rank_count = rank;
    std::uint32_t n = 0;
    for (auto i = begin; i < end; ++i) {
        const float l = logit_at(a, i);
        n += z_of(l, max, policy.temperature) >= policy.cutoff && selected(topk, order_key(l), rank_count) ? 1U : 0U;
    }
    __shared__ std::uint32_t lds[kThreads];
    auto position = block_exclusive(n, lds);
    for (unsigned b = 0; b < blockIdx.x; ++b) position += s.block_count[row * kBlocks + b];
    Candidate* out = s.candidates + static_cast<std::uint64_t>(row) * kCand;
    for (auto i = begin; i < end; ++i) {
        const float l = logit_at(a, i);
        const double z = z_of(l, max, policy.temperature);
        if (z >= policy.cutoff && selected(topk, order_key(l), rank) && position < kCand) out[position++] = {order_key(l), i, exp(z)};
    }
}
__global__ __launch_bounds__(kSmallThreads) void sample_small_decide_kernel(Rows rows, Policy policy, Scratch s, Result* results) {
    const auto row = blockIdx.x;
    const auto& st = s.state[row];
    if ((st.flags & (flag_small | flag_nonfinite)) != flag_small) return;
    __shared__ Candidate c[kCand];
    __shared__ std::uint32_t kept_shared;
    if (threadIdx.x == 0U) {
        std::uint32_t kept = 0;
        for (unsigned b = 0; b < kBlocks; ++b) kept += s.block_count[row * kBlocks + b];
        kept_shared = kept < kCand ? kept : kCand;
    }
    __syncthreads();
    const auto n = kept_shared;
    const Candidate* in = s.candidates + static_cast<std::uint64_t>(row) * kCand;
    for (unsigned j = threadIdx.x; j < n; j += blockDim.x) c[j] = in[j];
    decide_kept(rows.row[row], policy, c, n, results + row);
}

// Top-k up to kTopkOnePass in one pass over the logits. Each workgroup selects its own k largest entries
// (radix selection in LDS; ties at the k-th key take the lowest ids) and writes them in token order: the global
// top-k under (key desc, id asc) is contained in the union of these lists.
__global__ __launch_bounds__(kThreads) void topk_block_kernel(Rows rows, Policy policy, Scratch s) {
    const auto row = blockIdx.y;
    const RowArgs& a = rows.row[row];
    std::uint32_t begin, end;
    thread_range(policy.vocab, blockIdx.x, threadIdx.x, begin, end);
    __shared__ std::uint32_t hist[256];
    __shared__ std::uint32_t prefix, mask, remaining, all;
    __shared__ std::uint32_t lds[kThreads];
    if (threadIdx.x == 0U) { prefix = 0U; mask = 0U; remaining = policy.top_k; all = 0U; }
    bool bad = false;
    for (unsigned shift = 24U, round = 0U; round < 4U; ++round, shift -= 8U) {
        hist[threadIdx.x] = 0U;
        __syncthreads();
        const auto p = prefix, m = mask;
        if (all) break;
        for (auto i = begin; i < end; ++i) {
            const float l = logit_at(a, i);
            if (round == 0U) bad = bad || !isfinite(l);
            const auto key = order_key(l);
            if ((key & m) == p) atomicAdd(&hist[(key >> shift) & 255U], 1U);
        }
        __syncthreads();
        if (threadIdx.x == 0U) {
            if (round == 0U) {
                std::uint32_t n = 0;
                for (unsigned b = 0; b < 256U; ++b) n += hist[b];
                if (n <= remaining) all = 1U;
            }
            if (!all) {
                std::uint32_t above = 0;
                for (int bin = 255; bin >= 0; --bin) {
                    if (above + hist[bin] >= remaining) {
                        remaining -= above;
                        prefix |= static_cast<std::uint32_t>(bin) << shift;
                        mask |= 255U << shift;
                        break;
                    }
                    above += hist[bin];
                }
            }
        }
        __syncthreads();
    }
    if (bad) atomicOr(&s.state[row].flags, flag_nonfinite);
    const bool take_all = all != 0U;
    const auto key_k = prefix, need = remaining;
    // Ties at the k-th key: the first `need` in token order.
    std::uint32_t ties = 0;
    if (!take_all) for (auto i = begin; i < end; ++i) ties += order_key(logit_at(a, i)) == key_k ? 1U : 0U;
    auto tie_rank = block_exclusive(ties, lds);
    const auto keep = [&](std::uint32_t key, std::uint32_t& rank) {
        if (take_all || key > key_k) return true;
        if (key < key_k) return false;
        return rank++ < need;
    };
    auto rank_count = tie_rank;
    std::uint32_t mine = 0;
    for (auto i = begin; i < end; ++i) mine += keep(order_key(logit_at(a, i)), rank_count) ? 1U : 0U;
    auto position = block_exclusive(mine, lds);
    Candidate* out = s.candidates + (static_cast<std::uint64_t>(row) * kBlocks + blockIdx.x) * kTopkOnePass;
    for (auto i = begin; i < end; ++i) {
        const auto key = order_key(logit_at(a, i));
        if (keep(key, tie_rank) && position < kTopkOnePass) out[position++] = {key, i, 0.0};
    }
    __shared__ std::uint32_t counts[kThreads];
    const auto total = block_reduce(mine, counts, [](std::uint32_t x, std::uint32_t y) { return x + y; });
    if (threadIdx.x == 0U) s.block_count[row * kBlocks + blockIdx.x] = total < kTopkOnePass ? total : kTopkOnePass;
}
// Merge: the block lists in block order are in token order; the k largest of them (key desc, id asc) that pass
// the cutoff are the kept set, decided by decide_kept.
__global__ __launch_bounds__(kSmallThreads) void topk_merge_kernel(Rows rows, Policy policy, Scratch s, Result* results) {
    const auto row = blockIdx.x;
    if (s.state[row].flags & flag_nonfinite) return;
    const RowArgs& a = rows.row[row];
    __shared__ std::uint32_t offsets[kBlocks + 1U];
    __shared__ std::uint32_t index[kCand];
    __shared__ u64 order[kCand];
    __shared__ std::uint8_t chosen[kCand];
    __shared__ Candidate c[kTopkOnePass];
    __shared__ std::uint32_t kept_shared;
    if (threadIdx.x == 0U) {
        offsets[0] = 0U;
        for (unsigned b = 0; b < kBlocks; ++b) offsets[b + 1U] = offsets[b] + s.block_count[row * kBlocks + b];
    }
    __syncthreads();
    const auto n = offsets[kBlocks];
    unsigned size = 2U;
    while (size < n) size <<= 1U;
    for (unsigned j = threadIdx.x; j < size; j += blockDim.x) {
        if (j < n) {
            unsigned b = 0;
            while (offsets[b + 1U] <= j) ++b;
            const auto& e = s.candidates[(static_cast<std::uint64_t>(row) * kBlocks + b) * kTopkOnePass + (j - offsets[b])];
            index[j] = e.id;
            // (key desc, position asc): the position is in token order.
            order[j] = (static_cast<u64>(e.key) << 32U) | (0xffffffffU - j);
            chosen[j] = 0U;
        } else order[j] = 0ULL;
    }
    __syncthreads();
    sort_desc(order, size);
    const auto k = policy.top_k < n ? policy.top_k : n;
    for (unsigned j = threadIdx.x; j < k; j += blockDim.x) chosen[0xffffffffU - static_cast<std::uint32_t>(order[j])] = 1U;
    __syncthreads();
    if (threadIdx.x == 0U) {
        const float max = logit_at(a, index[0xffffffffU - static_cast<std::uint32_t>(order[0])]);
        std::uint32_t kept = 0;
        for (std::uint32_t j = 0; j < n; ++j) {
            if (!chosen[j]) continue;
            const float l = logit_at(a, index[j]);
            const double z = z_of(l, max, policy.temperature);
            if (z >= policy.cutoff) c[kept++] = {order_key(l), index[j], exp(z)};
        }
        kept_shared = kept;
    }
    __syncthreads();
    decide_kept(a, policy, c, kept_shared, results + row);
}

// Large rows: weights w = exp(z) of the kept elements (cutoff, then top-k), written per index (0 elsewhere);
// block sums.
__global__ __launch_bounds__(kThreads) void sample_weights_kernel(Rows rows, Policy policy, Scratch s) {
    const auto row = blockIdx.y;
    auto& st = s.state[row];
    if (st.flags & flag_small) return;
    const RowArgs& a = rows.row[row];
    const float max = st.max;
    std::uint32_t begin, end;
    thread_range(policy.vocab, blockIdx.x, threadIdx.x, begin, end);
    __shared__ std::uint32_t ranks[kThreads];
    const auto topk = st.topk;
    auto rank = equal_rank_base(s, topk, a, begin, end, row, ranks);
    double* weights = s.weights + static_cast<std::uint64_t>(row) * policy.vocab;
    double sum = 0.0;
    std::uint32_t kept = 0, last = 0;
    for (auto i = begin; i < end; ++i) {
        const float l = logit_at(a, i);
        const double z = z_of(l, max, policy.temperature);
        const bool keep = z >= policy.cutoff && selected(topk, order_key(l), rank);
        const double w = keep ? exp(z) : 0.0;
        weights[i] = w;
        if (keep) { sum += w; ++kept; last = i; }
    }
    __shared__ double lds[kThreads];
    __shared__ std::uint32_t counts[kThreads];
    const double block_sum = block_sequential_sum(sum, lds);
    const auto block_kept = block_reduce(kept, counts, [](std::uint32_t x, std::uint32_t y) { return x + y; });
    if (kept) atomicMax(&st.last_kept, last);
    if (threadIdx.x == 0U) { s.block_sum[row * kBlocks + blockIdx.x] = block_sum; s.block_count[row * kBlocks + blockIdx.x] = block_kept; }
}
__global__ void sample_after_weights_kernel(Policy policy, Scratch s) {
    const auto row = blockIdx.x;
    auto& st = s.state[row];
    if (st.flags & flag_small) return;
    double total = 0.0;
    std::uint32_t kept = 0;
    for (unsigned b = 0; b < kBlocks; ++b) { total += s.block_sum[row * kBlocks + b]; kept += s.block_count[row * kBlocks + b]; }
    st.total = total; st.kept_count = kept;
    if (policy.top_p_active && !(st.flags & flag_nonfinite)) {
        st.flags |= flag_top_p;
        st.quant.active = 1U; st.quant.kind = 0U; st.quant.scale = policy.top_p;
    }
}
// Large draft rows without top-p: the distribution per index.
__global__ __launch_bounds__(kThreads) void sample_q_out_kernel(Rows rows, Policy policy, Scratch s) {
    const auto row = blockIdx.y;
    const auto& st = s.state[row];
    const RowArgs& a = rows.row[row];
    if (!a.q_out || (st.flags & (flag_small | flag_nonfinite | flag_top_p))) return;
    const double* weights = s.weights + static_cast<std::uint64_t>(row) * policy.vocab;
    std::uint32_t begin, end;
    thread_range(policy.vocab, blockIdx.x, threadIdx.x, begin, end);
    for (auto i = begin; i < end; ++i) a.q_out[i] = weights[i] / st.total;
}

// Large top-p rows. The prefix is every kept entry with a key above prefix_key plus the entries at prefix_key up
// to index prefix_index (token order), found as a kind-0 quantile; its fixed-point mass normalizes p.
__device__ __forceinline__ bool in_prefix(const RowState& st, std::uint32_t key, std::uint32_t i) {
    return key > st.prefix_key || (key == st.prefix_key && i <= st.prefix_index);
}
__device__ __forceinline__ double quant_value(const Scratch& s, const RowState& st, unsigned row, std::uint32_t vocab,
    std::uint32_t i, std::uint32_t key) {
    const std::uint64_t at = static_cast<std::uint64_t>(row) * vocab + i;
    if (st.quant.kind == 0U) return s.weights[at];
    if (st.quant.kind == 1U) return in_prefix(st, key, i) ? s.weights[at] : 0.0;
    return s.rvals[at];
}
__device__ __forceinline__ bool quant_hit(const Quant& q, u128 cumulative) {
    return q.kind == 0U ? cumulative >= q.target : cumulative > q.target;
}
// One radix round: per-block fixed-point value per 8-bit digit of the entries matching the current key prefix.
__global__ __launch_bounds__(kThreads) void quant_hist_kernel(Rows rows, Policy policy, Scratch s, unsigned shift) {
    const auto row = blockIdx.y;
    const auto& st = s.state[row];
    const auto& q = st.quant;
    if (!q.active) return;
    __shared__ u64 hist[256][2];
    hist[threadIdx.x][0] = 0ULL; hist[threadIdx.x][1] = 0ULL;
    __syncthreads();
    const RowArgs& a = rows.row[row];
    std::uint32_t begin, end;
    thread_range(policy.vocab, blockIdx.x, threadIdx.x, begin, end);
    for (auto i = begin; i < end; ++i) {
        const auto key = order_key(logit_at(a, i));
        if ((key & q.mask) != q.prefix) continue;
        const double v = quant_value(s, st, row, policy.vocab, i, key);
        if (!(v > 0.0)) continue;
        u64 hi, lo;
        fp_split(v, hi, lo);
        const auto bin = (key >> shift) & 255U;
        atomicAdd(&hist[bin][0], hi);
        atomicAdd(&hist[bin][1], lo);
    }
    __syncthreads();
    u64* out = s.qhist + ((static_cast<std::uint64_t>(row) * kBlocks + blockIdx.x) * 256U + threadIdx.x) * 2U;
    out[0] = hist[threadIdx.x][0];
    out[1] = hist[threadIdx.x][1];
}
// The digit holding the target; round 0 also sets the target from the set's total.
__global__ __launch_bounds__(kThreads) void quant_pick_kernel(Scratch s, unsigned shift) {
    const auto row = blockIdx.x;
    auto& q = s.state[row].quant;
    if (!q.active) return;
    __shared__ u128 bins[256];
    u64 hi = 0, lo = 0;
    for (unsigned b = 0; b < kBlocks; ++b) {
        const u64* in = s.qhist + ((static_cast<std::uint64_t>(row) * kBlocks + b) * 256U + threadIdx.x) * 2U;
        hi += in[0]; lo += in[1];
    }
    bins[threadIdx.x] = fp_join(hi, lo);
    __syncthreads();
    if (threadIdx.x != 0U) return;
    if (shift == 24U) {
        u128 total = 0;
        for (unsigned bin = 0; bin < 256U; ++bin) total += bins[bin];
        q.total = total;
        // A zero residual mass (p == q up to rounding) samples p with the residual uniform instead.
        if (total == 0) { q.zero = 1U; q.active = 0U; return; }
        // sampling-v1: top_p * total (inclusive), or uniform * mass (exclusive); within the set.
        q.target = fp_of(q.scale * fp_double(total));
        if (q.kind != 0U && q.target >= total) q.target = total - 1U;
    }
    u128 above = 0;
    int chosen = -1, last = -1;
    for (int bin = 255; bin >= 0; --bin) {
        const u128 m = bins[bin];
        if (m == 0) continue;
        last = bin;
        if (quant_hit(q, above + m)) { chosen = bin; break; }
        above += m;
    }
    if (chosen < 0) {
        // Rounding: the target lies beyond the set; take its last entry.
        if (last < 0) { q.active = 0U; return; }
        chosen = last;
        above -= bins[last];
        q.target = q.kind == 0U ? bins[last] : bins[last] - 1U;
    } else q.target -= above;
    q.above += above;
    q.prefix |= static_cast<std::uint32_t>(chosen) << shift;
    q.mask |= 255U << shift;
}
// Entries at the final key: per-block fixed-point sums, then a walk in token order.
__global__ __launch_bounds__(kThreads) void quant_tie_sum_kernel(Rows rows, Policy policy, Scratch s) {
    const auto row = blockIdx.y;
    const auto& st = s.state[row];
    const auto& q = st.quant;
    if (!q.active) return;
    const RowArgs& a = rows.row[row];
    std::uint32_t begin, end;
    thread_range(policy.vocab, blockIdx.x, threadIdx.x, begin, end);
    u128 mine = 0;
    for (auto i = begin; i < end; ++i) {
        const auto key = order_key(logit_at(a, i));
        if (key != q.prefix) continue;
        const double v = quant_value(s, st, row, policy.vocab, i, key);
        if (v > 0.0) mine += fp_of(v);
    }
    __shared__ u128 lds[kThreads];
    lds[threadIdx.x] = mine;
    __syncthreads();
    if (threadIdx.x == 0U) {
        u128 sum = 0;
        for (unsigned t = 0; t < kThreads; ++t) sum += lds[t];
        u64* out = s.qtie + (static_cast<std::uint64_t>(row) * kBlocks + blockIdx.x) * 2U;
        out[0] = static_cast<u64>(sum >> 64U);
        out[1] = static_cast<u64>(sum);
    }
}
__global__ __launch_bounds__(kThreads) void quant_walk_kernel(Rows rows, Policy policy, Scratch s) {
    const auto row = blockIdx.x;
    auto& st = s.state[row];
    auto& q = st.quant;
    if (!q.active) return;
    const RowArgs& a = rows.row[row];
    __shared__ unsigned chosen_block, chosen_thread;
    __shared__ u128 start;
    __shared__ u128 sums[kThreads];
    if (threadIdx.x == 0U) {
        q.index = kNone; // stays kNone only if the walk fails; the row then reports no decision
        u128 cumulative = 0;
        chosen_block = kNone;
        for (unsigned b = 0; b < kBlocks; ++b) {
            const u64* in = s.qtie + (static_cast<std::uint64_t>(row) * kBlocks + b) * 2U;
            const u128 v = (static_cast<u128>(in[0]) << 64U) | in[1];
            if (v == 0) continue;
            if (quant_hit(q, cumulative + v)) { chosen_block = b; break; }
            cumulative += v;
        }
        start = cumulative;
    }
    __syncthreads();
    const auto block = chosen_block;
    if (block == kNone) { if (threadIdx.x == 0U) q.active = 0U; return; }
    std::uint32_t begin, end;
    thread_range(policy.vocab, block, threadIdx.x, begin, end);
    u128 mine = 0;
    for (auto i = begin; i < end; ++i) {
        const auto key = order_key(logit_at(a, i));
        if (key != q.prefix) continue;
        const double v = quant_value(s, st, row, policy.vocab, i, key);
        if (v > 0.0) mine += fp_of(v);
    }
    sums[threadIdx.x] = mine;
    __syncthreads();
    if (threadIdx.x == 0U) {
        u128 cumulative = start;
        chosen_thread = kNone;
        for (unsigned t = 0; t < kThreads; ++t) {
            if (sums[t] == 0) continue;
            if (quant_hit(q, cumulative + sums[t])) { chosen_thread = t; break; }
            cumulative += sums[t];
        }
        start = cumulative;
    }
    __syncthreads();
    if (threadIdx.x != chosen_thread) return;
    u128 cumulative = start;
    for (auto i = begin; i < end; ++i) {
        const auto key = order_key(logit_at(a, i));
        if (key != q.prefix) continue;
        const double v = quant_value(s, st, row, policy.vocab, i, key);
        if (!(v > 0.0)) continue;
        cumulative += fp_of(v);
        if (quant_hit(q, cumulative)) { q.index = i; q.cum = q.above + cumulative; return; }
    }
}
__device__ void quant_reset(Quant& q, std::uint32_t kind, double scale) {
    q.target = 0; q.above = 0; q.total = 0; q.cum = 0;
    q.scale = scale; q.active = 1U; q.kind = kind; q.prefix = 0U; q.mask = 0U; q.index = 0U; q.zero = 0U;
}
// After the prefix: its mass, the draft's acceptance, and the next quantile (sample or residual).
__global__ void topp_after_prefix_kernel(Rows rows, Policy policy, Scratch s, Result* results) {
    const auto row = blockIdx.x;
    auto& st = s.state[row];
    if (!(st.flags & flag_top_p)) return;
    auto& q = st.quant;
    if (!q.active || q.index == kNone) { q.active = 0U; return; }
    st.prefix_key = q.prefix; st.prefix_index = q.index; st.mass = fp_double(q.cum);
    const RowArgs& a = rows.row[row];
    if (a.draft == kNone) { quant_reset(q, 1U, a.u_plain); return; }
    const double* weights = s.weights + static_cast<std::uint64_t>(row) * policy.vocab;
    const double w = weights[a.draft];
    const double pd = w > 0.0 && in_prefix(st, order_key(a.logits[a.draft]), a.draft) ? w / st.mass : 0.0;
    const double qd = q_probability(a, a.draft);
    q.active = 0U;
    if (!(qd > 0.0)) { results[row] = {a.draft, flag_done | flag_drafted | flag_bad_draft}; return; }
    if (a.u_accept * qd < pd) { results[row] = {a.draft, flag_done | flag_drafted | flag_accepted}; return; }
    quant_reset(q, 2U, a.u_residual);
}
// Residual values r = max(0, p - q) over the prefix (drafted rows are target rows: index = token id).
__global__ __launch_bounds__(kThreads) void topp_residual_kernel(Rows rows, Policy policy, Scratch s) {
    const auto row = blockIdx.y;
    const auto& st = s.state[row];
    if (!(st.flags & flag_top_p) || !st.quant.active || st.quant.kind != 2U) return;
    const RowArgs& a = rows.row[row];
    const std::uint64_t base = static_cast<std::uint64_t>(row) * policy.vocab;
    std::uint32_t begin, end;
    thread_range(policy.vocab, blockIdx.x, threadIdx.x, begin, end);
    for (auto i = begin; i < end; ++i) {
        const double w = s.weights[base + i];
        s.rvals[base + i] = w > 0.0 && in_prefix(st, order_key(a.logits[i]), i) ? fmax(0.0, w / st.mass - q_probability(a, i)) : 0.0;
    }
}
// Large top-p draft rows: the distribution per index.
__global__ __launch_bounds__(kThreads) void topp_q_out_kernel(Rows rows, Policy policy, Scratch s) {
    const auto row = blockIdx.y;
    const auto& st = s.state[row];
    const RowArgs& a = rows.row[row];
    if (!a.q_out || !(st.flags & flag_top_p)) return;
    const double* weights = s.weights + static_cast<std::uint64_t>(row) * policy.vocab;
    std::uint32_t begin, end;
    thread_range(policy.vocab, blockIdx.x, threadIdx.x, begin, end);
    for (auto i = begin; i < end; ++i) {
        const double w = weights[i];
        a.q_out[i] = w > 0.0 && in_prefix(st, order_key(logit_at(a, i)), i) ? w / st.mass : 0.0;
    }
}
// After a sample or residual quantile: the token, or (zero residual mass) a sample of p with the residual uniform.
__global__ void topp_finish_kernel(Rows rows, Scratch s, Result* results) {
    const auto row = blockIdx.x;
    auto& st = s.state[row];
    if (!(st.flags & flag_top_p)) return;
    auto& q = st.quant;
    const RowArgs& a = rows.row[row];
    if (!q.active) {
        if (q.kind == 2U && q.zero) quant_reset(q, 1U, a.u_residual);
        return;
    }
    if (q.index == kNone) { q.active = 0U; return; }
    const auto token = token_at(a, q.index);
    if (a.token_out) *a.token_out = token;
    results[row] = {token, flag_done | (a.draft == kNone ? 0U : flag_drafted | flag_residual)};
    q.active = 0U;
}

// Large rows without top-p: find the element where the running sum of `value(i)` first exceeds `target`;
// thread 0 walks the block sums, the block's threads sum their ranges, thread 0 their prefix, one thread its range.
template <typename Value>
__device__ std::uint32_t token_order_search(const double* block_sums, double target, std::uint32_t vocab, Value&& value,
    std::uint32_t fallback, double* lds_d, std::uint32_t* lds_u) {
    __shared__ unsigned chosen_block;
    __shared__ double before_block;
    if (threadIdx.x == 0U) {
        chosen_block = kNone; double cumulative = 0.0;
        for (unsigned b = 0; b < kBlocks; ++b) {
            if (target < cumulative + block_sums[b]) { chosen_block = b; break; }
            cumulative += block_sums[b];
        }
        before_block = cumulative;
    }
    __syncthreads();
    const auto block = chosen_block;
    if (block == kNone) return fallback;
    std::uint32_t begin, end;
    thread_range(vocab, block, threadIdx.x, begin, end);
    double mine = 0.0;
    for (auto i = begin; i < end; ++i) mine += value(i);
    lds_d[threadIdx.x] = mine;
    __syncthreads();
    if (threadIdx.x == 0U) {
        double cumulative = before_block, last_start = before_block;
        std::uint32_t chosen = kNone, last = kNone;
        for (unsigned t = 0; t < kThreads; ++t) {
            const double next = cumulative + lds_d[t];
            if (lds_d[t] > 0.0) { last = t; last_start = cumulative; }
            if (target < next) { chosen = t; break; }
            cumulative = next;
        }
        // Rounding: the block's threads summed from its start can stay just below a target its total exceeded.
        if (chosen == kNone) { chosen = last; cumulative = last_start; }
        lds_u[0] = chosen;
        lds_d[0] = cumulative;
    }
    __syncthreads();
    const auto thread = lds_u[0];
    const double start = lds_d[0];
    __syncthreads();
    if (thread == kNone) return fallback;
    __shared__ std::uint32_t found;
    if (threadIdx.x == 0U) found = fallback;
    __syncthreads();
    if (threadIdx.x == thread) {
        thread_range(vocab, block, thread, begin, end);
        double cumulative = start;
        std::uint32_t last = kNone;
        for (auto i = begin; i < end; ++i) {
            const double v = value(i);
            if (!(v > 0.0)) continue;
            cumulative += v; last = i;
            if (target < cumulative) break;
        }
        // The range's own running sum may round just below the target that its total exceeded.
        if (last != kNone) found = last;
    }
    __syncthreads();
    return found;
}
// Drafted rows are target rows (index = token id).
__global__ __launch_bounds__(kThreads) void sample_decide_kernel(Rows rows, Policy policy, Scratch s, Result* results) {
    const auto row = blockIdx.x;
    auto& st = s.state[row];
    if (st.flags & (flag_nonfinite | flag_top_p | flag_small)) return;
    const RowArgs& a = rows.row[row];
    const double* weights = s.weights + static_cast<std::uint64_t>(row) * policy.vocab;
    const double total = st.total;
    __shared__ double lds_d[kThreads];
    __shared__ std::uint32_t lds_u[kThreads];
    const auto plain = [&](double u) {
        return token_order_search(s.block_sum + row * kBlocks, u * total, policy.vocab, [&](std::uint32_t i) { return weights[i]; },
            st.last_kept, lds_d, lds_u);
    };
    if (a.draft == kNone) {
        const auto token = token_at(a, plain(a.u_plain));
        if (threadIdx.x == 0U) { if (a.token_out) *a.token_out = token; results[row] = {token, flag_done}; }
        return;
    }
    const double qd = q_probability(a, a.draft);
    const double pd = weights[a.draft] / total;
    if (!(qd > 0.0)) {
        if (threadIdx.x == 0U) results[row] = {a.draft, flag_done | flag_drafted | flag_bad_draft};
        return;
    }
    if (a.u_accept * qd < pd) {
        if (threadIdx.x == 0U) results[row] = {a.draft, flag_done | flag_drafted | flag_accepted};
        return;
    }
    if (threadIdx.x == 0U) st.flags |= flag_need_residual;
}
// Residual rows: r = max(0, p - q) per kept token, block sums.
__global__ __launch_bounds__(kThreads) void sample_residual_kernel(Rows rows, Policy policy, Scratch s) {
    const auto row = blockIdx.y;
    auto& st = s.state[row];
    if (!(st.flags & flag_need_residual)) return;
    const RowArgs& a = rows.row[row];
    const double* weights = s.weights + static_cast<std::uint64_t>(row) * policy.vocab;
    std::uint32_t begin, end;
    thread_range(policy.vocab, blockIdx.x, threadIdx.x, begin, end);
    double sum = 0.0;
    std::uint32_t last = 0;
    bool any = false;
    for (auto i = begin; i < end; ++i) {
        if (!(weights[i] > 0.0)) continue;
        const double r = fmax(0.0, weights[i] / st.total - q_probability(a, i));
        if (r > 0.0) { sum += r; last = i; any = true; }
    }
    __shared__ double lds[kThreads];
    const double block = block_sequential_sum(sum, lds);
    if (any) atomicMax(&st.last_residual, last);
    if (threadIdx.x == 0U) s.block_residual[row * kBlocks + blockIdx.x] = block;
}
__global__ __launch_bounds__(kThreads) void sample_residual_decide_kernel(Rows rows, Policy policy, Scratch s, Result* results) {
    const auto row = blockIdx.x;
    auto& st = s.state[row];
    if (!(st.flags & flag_need_residual)) return;
    const RowArgs& a = rows.row[row];
    const double* weights = s.weights + static_cast<std::uint64_t>(row) * policy.vocab;
    __shared__ double lds_d[kThreads];
    __shared__ std::uint32_t lds_u[kThreads];
    double mass = 0.0;
    for (unsigned b = 0; b < kBlocks; ++b) mass += s.block_residual[row * kBlocks + b];
    std::uint32_t token;
    if (!(mass > 0.0))
        token = token_order_search(s.block_sum + row * kBlocks, a.u_residual * st.total, policy.vocab,
            [&](std::uint32_t i) { return weights[i]; }, st.last_kept, lds_d, lds_u);
    else
        token = token_order_search(s.block_residual + row * kBlocks, a.u_residual * mass, policy.vocab,
            [&](std::uint32_t i) { return weights[i] > 0.0 ? fmax(0.0, weights[i] / st.total - q_probability(a, i)) : 0.0; },
            st.last_residual, lds_d, lds_u);
    if (threadIdx.x == 0U) results[row] = {token, flag_done | flag_drafted | flag_residual};
}
__global__ void sample_nonfinite_kernel(Scratch s, Result* results, unsigned rows) {
    const auto row = threadIdx.x;
    if (row < rows && (s.state[row].flags & flag_nonfinite)) results[row] = {0U, flag_nonfinite};
}

using strix::hip_check;
void require(bool ok, const char* message) { if (!ok) throw std::invalid_argument(message); }

// Scratch for `max_rows` rows of at most `capacity` entries, and the kernel sequence.
struct Engine {
    std::uint32_t capacity, max_rows;
    Scratch scratch{};
    void* device_memory = nullptr;
    Engine(std::uint32_t c, std::uint32_t rows) : capacity(c), max_rows(rows) {
        require(c > 0U && rows > 0U && rows <= kMaxRows, "device token sampler geometry is invalid");
        std::uint64_t at = 0;
        const auto take = [&](std::uint64_t bytes) { const auto out = at; at = (at + bytes + 255U) / 256U * 256U; return out; };
        const auto o_max = take(sizeof(float) * kMaxRows * kBlocks), o_count = take(4ULL * kMaxRows * kBlocks),
            o_eq = take(4ULL * 2U * kMaxRows * kBlocks), o_hist = take(4ULL * kMaxRows * kBlocks * 256U),
            o_w = take(8ULL * rows * c), o_sum = take(8ULL * kMaxRows * kBlocks), o_res = take(8ULL * kMaxRows * kBlocks),
            o_cand = take(sizeof(Candidate) * kMaxRows * kCand), o_state = take(sizeof(RowState) * kMaxRows),
            o_rvals = take(8ULL * rows * c), o_qhist = take(16ULL * kMaxRows * kBlocks * 256U),
            o_qtie = take(16ULL * kMaxRows * kBlocks);
        hip_check(hipMalloc(&device_memory, at), "device token sampler scratch");
        auto* base = static_cast<unsigned char*>(device_memory);
        scratch.block_max = reinterpret_cast<float*>(base + o_max);
        scratch.block_count = reinterpret_cast<std::uint32_t*>(base + o_count);
        scratch.block_eq = reinterpret_cast<std::uint32_t*>(base + o_eq);
        scratch.hist = reinterpret_cast<std::uint32_t*>(base + o_hist);
        scratch.weights = reinterpret_cast<double*>(base + o_w);
        scratch.block_sum = reinterpret_cast<double*>(base + o_sum);
        scratch.block_residual = reinterpret_cast<double*>(base + o_res);
        scratch.candidates = reinterpret_cast<Candidate*>(base + o_cand);
        scratch.state = reinterpret_cast<RowState*>(base + o_state);
        scratch.rvals = reinterpret_cast<double*>(base + o_rvals);
        scratch.qhist = reinterpret_cast<u64*>(base + o_qhist);
        scratch.qtie = reinterpret_cast<u64*>(base + o_qtie);
    }
    ~Engine() { if (device_memory) (void)hipFree(device_memory); }
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    // Enqueues every kernel for `count` rows of `vocab` entries; `results` (pinned host) is filled when the stream
    // reaches the end of the sequence.
    void enqueue(const Rows& r, unsigned count, std::uint32_t vocab, const TokenSamplingPolicy& policy, Result* results,
        hipStream_t stream) {
        require(count > 0U && count <= max_rows && vocab > 0U && vocab <= capacity, "device token sampling geometry is invalid");
        Policy p;
        p.vocab = vocab; p.top_k = policy.top_k; p.top_p_active = policy.top_p < 1.0F ? 1U : 0U; p.rows = count;
        p.temperature = policy.temperature; p.top_p = policy.top_p;
        // sampling-v1 step 2: the cutoff of z.
        p.cutoff = -40.0;
        if (policy.min_p > 0.0F) p.cutoff = std::max(p.cutoff, std::log(static_cast<double>(policy.min_p)));
        hip_check(hipMemsetAsync(scratch.state, 0, sizeof(RowState) * kMaxRows, stream), "device sampler state reset");
        for (unsigned i = 0; i < count; ++i) results[i] = {0U, 0U};
        const dim3 grid(kBlocks, count);
        if (p.top_k > 0U && p.top_k <= kTopkOnePass) {
            // Small top-k: one pass over the logits and one merging workgroup per row.
            hipLaunchKernelGGL(topk_block_kernel, grid, dim3(kThreads), 0, stream, r, p, scratch);
            hipLaunchKernelGGL(topk_merge_kernel, dim3(count), dim3(kSmallThreads), 0, stream, r, p, scratch, results);
        } else {
            hipLaunchKernelGGL(sample_max_kernel, grid, dim3(kThreads), 0, stream, r, p, scratch);
            hipLaunchKernelGGL(sample_count_kernel, grid, dim3(kThreads), 0, stream, r, p, scratch);
            hipLaunchKernelGGL(sample_after_count_kernel, dim3(count), dim3(1), 0, stream, p, scratch);
            if (p.top_k) {
                for (unsigned shift = 24U, round = 0U; round < 4U; ++round, shift -= 8U) {
                    hipLaunchKernelGGL(sample_hist_kernel, grid, dim3(kThreads), 0, stream, r, p, scratch, shift);
                    hipLaunchKernelGGL(sample_pick_kernel, dim3(count), dim3(kThreads), 0, stream, scratch, shift);
                }
                hipLaunchKernelGGL(sample_equal_kernel, grid, dim3(kThreads), 0, stream, r, p, scratch);
            }
            hipLaunchKernelGGL(sample_small_count_kernel, grid, dim3(kThreads), 0, stream, r, p, scratch);
            hipLaunchKernelGGL(sample_small_collect_kernel, grid, dim3(kThreads), 0, stream, r, p, scratch);
            hipLaunchKernelGGL(sample_small_decide_kernel, dim3(count), dim3(kSmallThreads), 0, stream, r, p, scratch, results);
            hipLaunchKernelGGL(sample_weights_kernel, grid, dim3(kThreads), 0, stream, r, p, scratch);
            hipLaunchKernelGGL(sample_after_weights_kernel, dim3(count), dim3(1), 0, stream, p, scratch);
            if (p.top_p_active) {
                // Prefix boundary, then the sample or the residual quantile, then (zero residual mass) a sample of p.
                const auto quantile = [&] {
                    for (unsigned shift = 24U, round = 0U; round < 4U; ++round, shift -= 8U) {
                        hipLaunchKernelGGL(quant_hist_kernel, grid, dim3(kThreads), 0, stream, r, p, scratch, shift);
                        hipLaunchKernelGGL(quant_pick_kernel, dim3(count), dim3(kThreads), 0, stream, scratch, shift);
                    }
                    hipLaunchKernelGGL(quant_tie_sum_kernel, grid, dim3(kThreads), 0, stream, r, p, scratch);
                    hipLaunchKernelGGL(quant_walk_kernel, dim3(count), dim3(kThreads), 0, stream, r, p, scratch);
                };
                quantile();
                hipLaunchKernelGGL(topp_after_prefix_kernel, dim3(count), dim3(1), 0, stream, r, p, scratch, results);
                hipLaunchKernelGGL(topp_residual_kernel, grid, dim3(kThreads), 0, stream, r, p, scratch);
                hipLaunchKernelGGL(topp_q_out_kernel, grid, dim3(kThreads), 0, stream, r, p, scratch);
                quantile();
                hipLaunchKernelGGL(topp_finish_kernel, dim3(count), dim3(1), 0, stream, r, scratch, results);
                quantile();
                hipLaunchKernelGGL(topp_finish_kernel, dim3(count), dim3(1), 0, stream, r, scratch, results);
            } else {
                hipLaunchKernelGGL(sample_q_out_kernel, grid, dim3(kThreads), 0, stream, r, p, scratch);
                hipLaunchKernelGGL(sample_decide_kernel, dim3(count), dim3(kThreads), 0, stream, r, p, scratch, results);
                hipLaunchKernelGGL(sample_residual_kernel, grid, dim3(kThreads), 0, stream, r, p, scratch);
                hipLaunchKernelGGL(sample_residual_decide_kernel, dim3(count), dim3(kThreads), 0, stream, r, p, scratch, results);
            }
        }
        hipLaunchKernelGGL(sample_nonfinite_kernel, dim3(1), dim3(kMaxRows), 0, stream, scratch, results, count);
        hip_check(hipGetLastError(), "device token sampling launch");
    }
};

Result* pinned_results(std::uint32_t count) {
    Result* out = nullptr;
    hip_check(hipHostMalloc(reinterpret_cast<void**>(&out), sizeof(Result) * count, hipHostMallocDefault), "device token sampler results");
    return out;
}

} // namespace

TokenSamplingPath token_sampling_path(const TokenSamplingPolicy& policy) noexcept {
    // -1 auto, else the forced path.
    static const int forced = [] {
        // unknown STRIX_SAMPLE_PATH throws (std::terminate here: noexcept); 2 = auto (also unset)
        const std::size_t path = strix::env_choice("STRIX_SAMPLE_PATH", 2, {"device", "host", "auto"});
        if (strix::env_flag_default_off("STRIX_SAMPLE_HOST") || path == 1) return 1;
        if (path == 0) return 0;
        return -1;
    }();
    if (forced >= 0) return forced ? TokenSamplingPath::host : TokenSamplingPath::device;
    // auto: the device path for every policy. Single smoke runs measured it faster than the host path for every
    // kind of decision (window rows of each size, serial rows, draft steps) with temperature only, top-k, top-p,
    // min-p and their combination; the host only won with small kept sets before the one-workgroup path existed.
    static_cast<void>(policy);
    return TokenSamplingPath::device;
}
bool device_draft_sampling_enabled(const TokenSamplingPolicy& policy) noexcept {
    static const bool draft_host = strix::env_flag_default_off("STRIX_SAMPLE_DRAFT_HOST");
    return !draft_host && device_token_sampling_enabled(policy);
}

struct HipTokenSamplerDevice::Impl {
    std::uint32_t vocab, max_rows;
    Engine engine;
    Result* results = nullptr;                // pinned host
    struct QBuffer { std::uint32_t* tokens = nullptr; double* probs = nullptr; std::uint32_t capacity = 0; };
    QBuffer q[kMaxRows];                      // pinned host, read by the kernels
    std::uint64_t fallbacks = 0;
    std::vector<float> host_logits;
    hipEvent_t start = nullptr, stop = nullptr;
    Impl(std::uint32_t v, std::uint32_t rows) : vocab(v), max_rows(rows), engine(v, rows), results(pinned_results(kMaxRows)) {}
    ~Impl() {
        for (auto& b : q) { if (b.tokens) (void)hipHostFree(b.tokens); if (b.probs) (void)hipHostFree(b.probs); }
        if (results) (void)hipHostFree(results);
        if (start) (void)hipEventDestroy(start);
        if (stop) (void)hipEventDestroy(stop);
    }
    // A host draft distribution, token id ascending for the kernels' binary search.
    void stage_q(unsigned row, const TokenDistribution& d, RowArgs& a) {
        auto& b = q[row];
        const auto n = static_cast<std::uint32_t>(d.tokens().size());
        if (b.capacity < n) {
            if (b.tokens) { hip_check(hipHostFree(b.tokens), "free draft tokens"); hip_check(hipHostFree(b.probs), "free draft probabilities"); }
            const auto cap = std::max<std::uint32_t>(n, 4096U);
            hip_check(hipHostMalloc(reinterpret_cast<void**>(&b.tokens), 4ULL * cap, hipHostMallocDefault), "draft tokens");
            hip_check(hipHostMalloc(reinterpret_cast<void**>(&b.probs), 8ULL * cap, hipHostMallocDefault), "draft probabilities");
            b.capacity = cap;
        }
        std::vector<std::uint32_t> order(n);
        for (std::uint32_t i = 0; i < n; ++i) order[i] = i;
        const auto tokens = d.tokens(); const auto probs = d.probabilities();
        std::sort(order.begin(), order.end(), [&](std::uint32_t x, std::uint32_t y) { return tokens[x] < tokens[y]; });
        for (std::uint32_t i = 0; i < n; ++i) { b.tokens[i] = tokens[order[i]]; b.probs[i] = probs[order[i]]; }
        a.q_tokens = b.tokens; a.q_probs = b.probs; a.q_count = n;
    }
    void set_draft(unsigned row, const TokenDraft& draft, RowArgs& a) {
        require(draft.token < vocab, "draft token lies outside the vocabulary");
        a.draft = draft.token;
        if (draft.device.probabilities) {
            a.q_tokens = draft.device.tokens; a.q_probs = draft.device.probabilities; a.q_count = draft.device.count;
        } else stage_q(row, draft.distribution, a);
    }
    void run(const Rows& r, unsigned count, const TokenSamplingPolicy& policy, hipStream_t stream) {
        const bool timed = sample_timing_enabled();
        if (timed && !start) { hip_check(hipEventCreate(&start), "sampler timing event"); hip_check(hipEventCreate(&stop), "sampler timing event"); }
        if (timed) hip_check(hipEventRecord(start, stream), "sampler timing start");
        engine.enqueue(r, count, vocab, policy, results, stream);
        if (timed) hip_check(hipEventRecord(stop, stream), "sampler timing stop");
        hip_check(hipStreamSynchronize(stream), "device token sampling completion");
        if (timed) {
            float ms = 0.0F;
            hip_check(hipEventElapsedTime(&ms, start, stop), "sampler timing");
            record_sample_timing(SampleTimingKind::kernels, count, 0.0, static_cast<double>(ms) * 1e-3, count);
        }
    }
};

HipTokenSamplerDevice::HipTokenSamplerDevice(std::uint32_t vocabulary, std::uint32_t max_rows)
    : impl_(std::make_unique<Impl>(vocabulary, max_rows)) {}
HipTokenSamplerDevice::~HipTokenSamplerDevice() = default;
std::uint64_t HipTokenSamplerDevice::host_fallbacks() const noexcept { return impl_->fallbacks; }

TokenDraft host_token_draft(const TokenDraft& draft, void* stream) {
    if (!draft.device.probabilities) return draft;
    const auto n = draft.device.count;
    std::vector<std::uint32_t> tokens(n);
    std::vector<double> probs(n);
    const auto s = static_cast<hipStream_t>(stream);
    hip_check(hipMemcpyAsync(tokens.data(), draft.device.tokens, 4ULL * n, hipMemcpyDeviceToHost, s), "draft tokens readback");
    hip_check(hipMemcpyAsync(probs.data(), draft.device.probabilities, 8ULL * n, hipMemcpyDeviceToHost, s), "draft probabilities readback");
    hip_check(hipStreamSynchronize(s), "draft distribution readback");
    TokenDraft out;
    out.position = draft.position; out.token = draft.token;
    out.distribution = make_token_distribution_from_probabilities(tokens, probs);
    return out;
}

std::vector<TokenDecision> HipTokenSamplerDevice::decide(TokenSampler& sampler, std::span<const DeviceSampleRow> rows, void* stream) {
    auto& i = *impl_;
    require(!rows.empty() && rows.size() <= i.max_rows, "device token sampling row count is invalid");
    const auto& policy = sampler.policy();
    const bool coupled = sampler.coupled();
    Rows r{};
    for (unsigned k = 0; k < rows.size(); ++k) {
        const auto& row = rows[k];
        require(row.logits.address && row.logits.bytes >= 4ULL * i.vocab, "sampled logits row is smaller than the vocabulary");
        auto& a = r.row[k];
        a.logits = reinterpret_cast<const float*>(row.logits.address);
        const auto u = sampler.uniforms(row.position, row.draft != nullptr);
        a.u_plain = u.plain; a.u_accept = u.accept; a.u_residual = u.residual;
        if (row.draft) {
            require(row.draft->position == row.position, "draft position differs from the decided position");
            // Coupled: the device decides the row as without a draft (q is not staged); acceptance is equality below.
            if (!coupled) i.set_draft(k, *row.draft, a);
        }
    }
    const auto s = static_cast<hipStream_t>(stream);
    i.run(r, static_cast<unsigned>(rows.size()), policy, s);
    std::vector<TokenDecision> out(rows.size());
    for (unsigned k = 0; k < rows.size(); ++k) {
        const auto result = i.results[k];
        require(!(result.flags & flag_nonfinite), "token distribution requires finite logits");
        require(!(result.flags & flag_bad_draft), "draft token must have positive draft probability");
        if (result.flags & flag_host || !(result.flags & flag_done)) {
            // No device decision (not expected): the host transform over this row's logits.
            ++i.fallbacks;
            read_hip_logits_row(rows[k].logits, i.vocab, stream, i.host_logits);
            if (coupled) {
                out[k] = decide_token_coupled(make_token_distribution(i.host_logits, {}, policy), rows[k].draft, r.row[k].u_plain);
                continue;
            }
            const auto draft = rows[k].draft ? std::optional<TokenDraft>(host_token_draft(*rows[k].draft, stream)) : std::nullopt;
            out[k] = decide_token(make_token_distribution(i.host_logits, {}, policy), draft ? &*draft : nullptr,
                r.row[k].u_plain, r.row[k].u_accept, r.row[k].u_residual);
            continue;
        }
        out[k].token = result.token;
        if (coupled) {
            if (rows[k].draft) { out[k].drafted = true; out[k].accepted = result.token == rows[k].draft->token; }
            continue;
        }
        out[k].drafted = (result.flags & flag_drafted) != 0U;
        out[k].accepted = (result.flags & flag_accepted) != 0U;
        out[k].residual = (result.flags & flag_residual) != 0U;
    }
    return out;
}

struct HipTokenDraftSamplerDevice::Impl {
    std::uint32_t capacity, max_steps, rows = 0, uploaded = 0;
    Engine engine;
    Result* results = nullptr;          // pinned host, one per step
    std::uint32_t* order = nullptr;     // device: index (token id order) -> head entry
    std::uint32_t* sorted_ids = nullptr; // device: index -> token id, ascending
    double* q = nullptr;                // device: [max_steps][capacity]
    std::uint32_t* staging = nullptr;   // pinned host: order then ids
    std::vector<std::uint64_t> positions;
    std::vector<hipEvent_t> starts, stops; // STRIX_SAMPLE_TIMING: per step, read after the chain completed
    Impl(std::uint32_t c, std::uint32_t steps)
        : capacity(c), max_steps(steps), engine(c, 1U), results(pinned_results(std::max<std::uint32_t>(steps, 1U))), positions(steps) {
        require(steps > 0U, "device draft sampler needs a step");
        hip_check(hipMalloc(reinterpret_cast<void**>(&order), 4ULL * c), "device draft order");
        hip_check(hipMalloc(reinterpret_cast<void**>(&sorted_ids), 4ULL * c), "device draft ids");
        hip_check(hipMalloc(reinterpret_cast<void**>(&q), 8ULL * c * steps), "device draft distributions");
        hip_check(hipHostMalloc(reinterpret_cast<void**>(&staging), 8ULL * c, hipHostMallocDefault), "device draft staging");
    }
    ~Impl() {
        if (results) (void)hipHostFree(results);
        if (staging) (void)hipHostFree(staging);
        for (auto* e : starts) (void)hipEventDestroy(e);
        for (auto* e : stops) (void)hipEventDestroy(e);
        for (void* p : {static_cast<void*>(order), static_cast<void*>(sorted_ids), static_cast<void*>(q)}) if (p) (void)hipFree(p);
    }
    void set_vocabulary(std::span<const std::uint32_t> ids, std::uint32_t n, hipStream_t stream) {
        require(n > 0U && n <= capacity && (ids.empty() || ids.size() == n), "device draft vocabulary is invalid");
        rows = n;
        if (n == uploaded) return;
        // The staging copy of the previous upload must be complete before it is rewritten.
        hip_check(hipStreamSynchronize(stream), "device draft vocabulary staging");
        std::uint32_t* perm = staging;
        std::uint32_t* sorted = staging + capacity;
        std::iota(perm, perm + n, 0U);
        if (!ids.empty()) std::sort(perm, perm + n, [&](std::uint32_t x, std::uint32_t y) { return ids[x] < ids[y]; });
        for (std::uint32_t i = 0; i < n; ++i) sorted[i] = ids.empty() ? perm[i] : ids[perm[i]];
        for (std::uint32_t i = 1; i < n; ++i) require(sorted[i - 1U] != sorted[i], "token distribution ids must be distinct");
        hip_check(hipMemcpyAsync(order, perm, 4ULL * n, hipMemcpyHostToDevice, stream), "device draft order upload");
        hip_check(hipMemcpyAsync(sorted_ids, sorted, 4ULL * n, hipMemcpyHostToDevice, stream), "device draft ids upload");
        uploaded = n;
    }
    void step(const TokenSamplingPolicy& policy, std::uint32_t k, const float* logits, double u, std::uint32_t* slot,
        hipStream_t stream) {
        require(rows > 0U && k < max_steps && logits, "device draft step is invalid");
        Rows r{};
        auto& a = r.row[0];
        a.logits = logits; a.order = order; a.ids = sorted_ids;
        a.q_out = q + static_cast<std::uint64_t>(k) * capacity; a.token_out = slot;
        a.u_plain = u;
        const bool timed = sample_timing_enabled();
        if (timed && starts.empty()) {
            starts.resize(max_steps); stops.resize(max_steps);
            for (std::uint32_t i = 0; i < max_steps; ++i) {
                hip_check(hipEventCreate(&starts[i]), "draft timing event"); hip_check(hipEventCreate(&stops[i]), "draft timing event");
            }
        }
        if (timed) hip_check(hipEventRecord(starts[k], stream), "draft timing start");
        engine.enqueue(r, 1U, rows, policy, results + k, stream);
        if (timed) hip_check(hipEventRecord(stops[k], stream), "draft timing stop");
    }
    TokenDraft draft(std::uint32_t k) const {
        const auto result = results[k];
        require(!(result.flags & flag_nonfinite), "token distribution requires finite logits");
        if (result.flags & flag_host || !(result.flags & flag_done))
            throw std::logic_error("device draft step produced no decision");
        TokenDraft d;
        d.position = positions[k]; d.token = result.token;
        d.device = {sorted_ids, q + static_cast<std::uint64_t>(k) * capacity, rows};
        return d;
    }
};

HipTokenDraftSamplerDevice::HipTokenDraftSamplerDevice(std::uint32_t capacity, std::uint32_t max_steps)
    : impl_(std::make_unique<Impl>(capacity, max_steps)) {}
HipTokenDraftSamplerDevice::~HipTokenDraftSamplerDevice() = default;
void HipTokenDraftSamplerDevice::set_vocabulary(std::span<const std::uint32_t> host_ids, std::uint32_t rows, void* stream) {
    impl_->set_vocabulary(host_ids, rows, static_cast<hipStream_t>(stream));
}
void HipTokenDraftSamplerDevice::step(const TokenSampler& sampler, std::uint32_t step, const float* logits, std::uint64_t position,
    std::uint32_t* slot, void* stream) {
    require(step < impl_->max_steps, "device draft step is invalid");
    impl_->positions[step] = position;
    // draft_policy(): the policy, or top-k 1 for coupled argmax drafts; draft_uniform(): U(n, plain) when coupled.
    impl_->step(sampler.draft_policy(), step, logits, sampler.draft_uniform(position), slot, static_cast<hipStream_t>(stream));
}
void HipTokenDraftSamplerDevice::finish(TokenSampler& sampler, std::uint32_t steps, std::vector<TokenDraft>& out) {
    require(steps <= impl_->max_steps, "device draft steps exceed the sampler");
    out.clear();
    for (std::uint32_t k = 0; k < steps; ++k) {
        out.push_back(impl_->draft(k)); sampler.count_draft();
        if (!impl_->starts.empty()) {
            float ms = 0.0F;
            hip_check(hipEventElapsedTime(&ms, impl_->starts[k], impl_->stops[k]), "draft timing");
            record_sample_timing(SampleTimingKind::draft_kernels, 1U, 0.0, static_cast<double>(ms) * 1e-3, 1U);
        }
    }
}

TokenDecision hip_decide_token_for_testing(std::span<const float> logits, const TokenSamplingPolicy& policy,
    const TokenDraft* draft, double u_plain, double u_accept, double u_residual, bool* host_fallback, bool device_q) {
    const auto vocab = static_cast<std::uint32_t>(logits.size());
    validate_token_sampling_policy(policy);
    HipTokenSamplerDevice::Impl impl(vocab, 1U);
    std::vector<void*> allocations;
    const auto upload = [&](const void* data, std::uint64_t bytes) {
        void* p = nullptr;
        hip_check(hipMalloc(&p, bytes), "testing allocation");
        allocations.push_back(p);
        hip_check(hipMemcpy(p, data, bytes, hipMemcpyHostToDevice), "testing upload");
        return p;
    };
    struct Free { std::vector<void*>& v; ~Free() { for (void* p : v) (void)hipFree(p); } } free_all{allocations};
    Rows r{};
    auto& a = r.row[0];
    a.logits = static_cast<const float*>(upload(logits.data(), 4ULL * vocab));
    a.u_plain = u_plain; a.u_accept = u_accept; a.u_residual = u_residual;
    TokenDraft view;
    if (draft) {
        if (device_q) {
            // Token ids ascending with every third id outside the distribution present as a zero entry.
            std::vector<std::uint32_t> ids;
            std::vector<double> probs;
            const auto& d = draft->distribution;
            std::vector<std::uint32_t> members(d.tokens().begin(), d.tokens().end());
            std::sort(members.begin(), members.end());
            std::size_t m = 0;
            for (std::uint32_t t = 0; t < vocab; ++t) {
                const bool member = m < members.size() && members[m] == t;
                if (member) { ids.push_back(t); probs.push_back(d.probability(t)); ++m; }
                else if (t % 3U == 0U) { ids.push_back(t); probs.push_back(0.0); }
            }
            view.position = draft->position; view.token = draft->token;
            view.device = {static_cast<const std::uint32_t*>(upload(ids.data(), 4ULL * ids.size())),
                           static_cast<const double*>(upload(probs.data(), 8ULL * probs.size())),
                           static_cast<std::uint32_t>(ids.size())};
            impl.set_draft(0U, view, a);
        } else impl.set_draft(0U, *draft, a);
    }
    impl.run(r, 1U, policy, nullptr);
    const auto result = impl.results[0];
    require(!(result.flags & flag_nonfinite), "token distribution requires finite logits");
    require(!(result.flags & flag_bad_draft), "draft token must have positive draft probability");
    const bool none = (result.flags & flag_host) || !(result.flags & flag_done);
    if (host_fallback) *host_fallback = none;
    if (none) return decide_token(make_token_distribution(logits, {}, policy), draft, u_plain, u_accept, u_residual);
    TokenDecision out;
    out.token = result.token;
    out.drafted = (result.flags & flag_drafted) != 0U;
    out.accepted = (result.flags & flag_accepted) != 0U;
    out.residual = (result.flags & flag_residual) != 0U;
    return out;
}

TokenDraft hip_draft_for_testing(std::span<const float> logits, std::span<const std::uint32_t> ids,
    const TokenSamplingPolicy& policy, double u) {
    validate_token_sampling_policy(policy);
    const auto n = static_cast<std::uint32_t>(logits.size());
    HipTokenDraftSamplerDevice::Impl impl(n, 1U);
    float* device_logits = nullptr;
    std::uint32_t* slot = nullptr;
    hip_check(hipMalloc(reinterpret_cast<void**>(&device_logits), 4ULL * n), "testing draft logits");
    hip_check(hipMalloc(reinterpret_cast<void**>(&slot), 4ULL), "testing draft slot");
    struct Free { float* l; std::uint32_t* s; ~Free() { (void)hipFree(l); (void)hipFree(s); } } free_all{device_logits, slot};
    hip_check(hipMemcpy(device_logits, logits.data(), 4ULL * n, hipMemcpyHostToDevice), "testing draft logits upload");
    impl.set_vocabulary(ids, n, nullptr);
    impl.positions[0] = 0U;
    impl.step(policy, 0U, device_logits, u, slot, nullptr);
    hip_check(hipStreamSynchronize(nullptr), "testing draft step");
    auto d = impl.draft(0U);
    std::uint32_t written = 0;
    hip_check(hipMemcpy(&written, slot, 4ULL, hipMemcpyDeviceToHost), "testing draft slot readback");
    require(written == d.token, "device draft slot differs from the reported token");
    return host_token_draft(d, nullptr);
}

} // namespace strix
