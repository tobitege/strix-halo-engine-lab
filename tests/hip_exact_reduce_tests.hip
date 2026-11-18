// Exact-reduction fixture (no model): core/ordered_reduce.hpp and the CSR scan of moe_core/device_plan.hpp against
// verbatim copies of the LDS loops they replaced: broadcasts/CSR bit for bit; legacy += trees allow NaN payload changes.
//   trees: L = 64..1024; broadcast (twice on one s: the reuse barrier), lane 0, and the PLE key/query pair, on
//          cancellation, signed zeros, subnormals, infinities, NaN payloads, overflow and random bit patterns.
//   CSR:   count -> {old scan, new scan} -> scatter on the same scratch; scratch words, offsets, touched, batch
//          experts/begins, rows, slots, summary and both histograms equal; empties, tails, duplicates, failures, prune.
//   time:  ABAB (old, new) of the tree alone and of the scan at 8K/32K/128K rows (`time` argument).
#include "../src/core/ordered_reduce.hpp"
#include "../src/moe_core/device_plan.hpp"
#include "exact_reduce_contract.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

namespace tree = strix::core::ordered_reduce;
namespace plan = strix::moe_core::device_plan;

int failures = 0;
void expect(bool ok, const char* what) { if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what); } }
void check(hipError_t e, const char* why) { if (e != hipSuccess) { std::fprintf(stderr, "%s: %s\n", why, hipGetErrorString(e)); std::exit(2); } }
template <class T> T* dev(std::size_t n) { void* p = nullptr; check(hipMalloc(&p, n * sizeof(T)), "alloc"); return static_cast<T*>(p); }
template <class T> void up(T* d, const std::vector<T>& h) { check(hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "upload"); }
template <class T> std::vector<T> down(const T* d, std::size_t n) {
    std::vector<T> h(n); check(hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost), "download"); return h;
}
template <class T> bool same_bytes(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

// ------------------------------------------------------------------------------------------------ trees
// The replaced loops, verbatim: glm_tier_a.hip lds_tree, and the shared/PLE form (stride loop, +=, lane 0 reads [0]).
template <unsigned L> __device__ float old_broadcast(float* s, float part) {
    const unsigned l = threadIdx.x;
    s[l] = part;
    __syncthreads();
    for (unsigned h = L / 2; h > 0; h >>= 1U) {
        if (l < h) s[l] = s[l] + s[l + h];
        __syncthreads();
    }
    const float r = s[0];
    __syncthreads();
    return r;
}
template <unsigned L> __device__ float old_lane0(float* s, float part) {
    const unsigned lane = threadIdx.x;
    s[lane] = part;
    __syncthreads();
    for (std::uint32_t stride = L / 2U; stride > 0; stride >>= 1U) {
        if (lane < stride) s[lane] += s[lane + stride];
        __syncthreads();
    }
    return s[0];
}
template <unsigned L> __device__ float old_pair(float* a, float* b, float x, float y, float& out_y) {
    const unsigned lane = threadIdx.x;
    a[lane] = x;
    b[lane] = y;
    __syncthreads();
    for (std::uint32_t stride = L / 2U; stride > 0; stride >>= 1U) {
        if (lane < stride) {
            a[lane] += a[lane + stride];
            b[lane] += b[lane + stride];
        }
        __syncthreads();
    }
    out_y = b[0];
    return a[0];
}

// Per CTA: two broadcast trees on one s (lanes' results), then lane 0 tree, then the pair. Output 2L + 3 words.
template <unsigned L, bool New> __global__ __launch_bounds__(L) void tree_kernel(const float* in, float* out) {
    __shared__ float s[L], s2[L];
    const unsigned l = threadIdx.x;
    const std::size_t c = blockIdx.x;
    const float x = in[2U * c * L + l], y = in[(2U * c + 1U) * L + l];
    float* o = out + c * (2U * L + 3U);
    float b1 = 0.0F, b2 = 0.0F, r1 = 0.0F, pa = 0.0F, pb = 0.0F;
    if constexpr (New) {
        b1 = tree::tree_broadcast<L>(s, x, l);
        b2 = tree::tree_broadcast<L>(s, y, l);
        r1 = tree::tree_to_lane0<L>(s, x, l);
        __syncthreads();
        float* const ss[2] = {s, s2};
        float v[2] = {x, y};
        tree::tree_to_lane0<L, 2U>(ss, v, l);
        pa = v[0];
        pb = v[1];
    } else {
        b1 = old_broadcast<L>(s, x);
        b2 = old_broadcast<L>(s, y);
        r1 = old_lane0<L>(s, x);
        __syncthreads();
        pa = old_pair<L>(s, s2, x, y, pb);
    }
    o[l] = b1;
    o[L + l] = b2;
    if (l == 0U) {
        o[2U * L] = r1;
        o[2U * L + 1U] = pa;
        o[2U * L + 2U] = pb;
    }
}

float bits_float(std::uint32_t u) { float f; std::memcpy(&f, &u, sizeof f); return f; }

// Adversarial leaves: case c of each CTA pair.
std::vector<float> tree_inputs(unsigned L, unsigned ctas, std::mt19937& g) {
    std::vector<float> v(std::size_t{2U} * ctas * L);
    std::normal_distribution<float> nd(0.0F, 1.0F);
    std::uniform_int_distribution<std::uint32_t> bits;
    for (unsigned c = 0; c < 2U * ctas; ++c) {
        float* x = v.data() + std::size_t{c} * L;
        const unsigned kind = (c / 2U) % 11U;
        for (unsigned l = 0; l < L; ++l) {
            float a = nd(g);
            switch (kind) {
                case 0: break;
                case 1: a = ((l & 1U) ? -1.0e30F : 1.0e30F) * (1.0F + 1.0e-7F * nd(g)) + 1.0e-3F * nd(g); break;  // cancellation
                case 2: a = (bits(g) & 1U) ? -0.0F : 0.0F; break;                                                    // signed zeros
                case 3: a = (bits(g) % 7U == 0U) ? bits_float(bits(g) & 0x807fffffU) : ((bits(g) & 1U) ? -0.0F : 0.0F); break;  // subnormals
                case 4: { const auto r = bits(g) % 9U; if (r == 0U) a = std::numeric_limits<float>::infinity(); if (r == 1U) a = -std::numeric_limits<float>::infinity(); } break;
                case 5: if (bits(g) % 5U == 0U) a = bits_float(0x7f800001U | (bits(g) & 0x803fffffU) | ((bits(g) & 1U) << 22U)); break;  // NaN payloads
                case 6: a = bits_float(exact_reduce_contract::collision_leaf(l, (c & 1U) != 0U)); break;
                case 7: a = std::numeric_limits<float>::max() * ((bits(g) & 1U) ? 0.75F : -0.5F) * (1.0F + 0.25F * std::fabs(nd(g))); break;  // overflow
                case 8: a = bits_float(bits(g) & 0x807fffffU); break;                                                  // all subnormal
                case 9: a = bits_float(bits(g)); break;                                                                // any bits
                default: a = 1.0F + 1.0e-7F * static_cast<float>(l); break;                                            // rounding ties
            }
            x[l] = a;
        }
    }
    return v;
}

template <unsigned L> void tree_case(std::mt19937& g) {
    constexpr unsigned kCtas = 440U;
    const auto in = tree_inputs(L, kCtas, g);
    float* din = dev<float>(in.size());
    up(din, in);
    const std::size_t n = std::size_t{kCtas} * (2U * L + 3U);
    float* a = dev<float>(n);
    float* b = dev<float>(n);
    check(hipMemset(a, 0x5a, n * sizeof(float)), "memset");
    check(hipMemset(b, 0x5a, n * sizeof(float)), "memset");
    tree_kernel<L, false><<<kCtas, L>>>(din, a);
    tree_kernel<L, true><<<kCtas, L>>>(din, b);
    check(hipDeviceSynchronize(), "tree kernels");
    const auto ha = down(a, n), hb = down(b, n);
    // Broadcasts require raw equality. Only r1/pa/pb (legacy +=) permit NaN-to-NaN payload changes.
    unsigned bad = 0U, nan = 0U, payload = 0U, shown = 0U, by_slot[5] = {}, by_kind[11] = {};
    const std::size_t per = 2U * L + 3U;
    for (std::size_t i = 0; i < n; ++i) {
        std::uint32_t ua, ub;
        std::memcpy(&ua, &ha[i], 4);
        std::memcpy(&ub, &hb[i], 4);
        const std::size_t c = i / per;
        const unsigned slot = exact_reduce_contract::output_slot(i, L);
        const bool both_nan = std::isnan(ha[i]) && std::isnan(hb[i]);
        bad += exact_reduce_contract::mismatch(ua, ub, slot);
        nan += std::isnan(ha[i]);
        if (ua == ub || !both_nan) continue;
        ++payload;
        ++by_slot[slot];
        ++by_kind[c % 11U];
        if (shown++ < 3U) std::printf("  L%u payload cta %zu kind %zu slot %u: old %08x new %08x\n", L, c, c % 11U, slot, ua, ub);
    }
    char what[96];
    std::snprintf(what, sizeof what, "tree L%u: broadcast raw bits; legacy += non-NaN bits and NaN positions", L);
    expect(bad == 0U, what);
    std::printf("tree L%-4u %u CTAs x 11 kinds: %zu words, %u contract mismatches, %u NaN-payload-only (slots b1 b2 r1 pa pb %u %u %u %u %u;"
                " kinds 4/5/6/9 %u %u %u %u), %u NaN results\n", L, kCtas, n, bad, payload, by_slot[0], by_slot[1], by_slot[2], by_slot[3],
                by_slot[4], by_kind[4], by_kind[5], by_kind[6], by_kind[9], nan);
    check(hipFree(din), "free");
    check(hipFree(a), "free");
    check(hipFree(b), "free");
}

// Timing: T back-to-back lane-0 trees (shared/PLE form) per CTA, nothing else; 32,768 CTAs x 256 lanes (the PLE grid).
template <unsigned L, bool New, unsigned T> __global__ __launch_bounds__(L) void tree_cost_kernel(float* out) {
    __shared__ float s[L];
    const unsigned l = threadIdx.x;
    float x = static_cast<float>(l) * 0.5F + static_cast<float>(blockIdx.x);
    float acc = 0.0F;
#pragma unroll 1
    for (unsigned i = 0; i < T; ++i) {
        const float r = New ? tree::tree_to_lane0<L>(s, x, l) : old_lane0<L>(s, x);
        acc = acc + r;
        x = x + 1.0F;
        __syncthreads();
    }
    if (l == 0U) out[blockIdx.x] = acc;
}

// --------------------------------------------------------------------------------------------------- CSR
struct Tables { unsigned *summary, *offsets, *touched, *rows, *slots, *experts, *begins; };
struct View { const unsigned* routes; const float* weights; const unsigned* status; Tables csr; unsigned top_k; };
struct Policy {
    static constexpr unsigned kExperts = 512U, kTopK = 10U, kThreads = 1024U, kAssignmentsPerWave = 64U;
    static constexpr unsigned kCountLimit = 65535U, kHistogramBins = 8U, kInvalid = 1U;
    using View = ::View;
    __device__ static const Tables& tables(const View& v) { return v.csr; }
    __device__ static const unsigned* routes(const View& v, unsigned r) { return v.routes + r * v.top_k; }
    __device__ static unsigned expert(const View& v, unsigned r, unsigned s) { return v.routes[r * v.top_k + s]; }
    __device__ static unsigned status(const View& v, unsigned r) { return v.status[r]; }
    __device__ static bool kept(const View& v, unsigned r, unsigned s) { return v.weights[r * v.top_k + s] != 0.0F; }
    template <bool Certificate>
    __device__ static bool invalid(const View& v, unsigned r, unsigned s, unsigned e, unsigned experts, bool& unsafe) {
        (void)unsafe;
        return e >= experts || !isfinite(v.weights[r * v.top_k + s]);
    }
    __device__ static void certificate(const View&) {}
    __device__ static void failure(const View& v, unsigned t, unsigned bits, unsigned count, unsigned rows) {
        if (t < 32U) v.csr.summary[t] = t == 0U ? bits : t == 3U ? count : t == 6U ? rows : 0xfeedU;
    }
    // Summary words 0..6 plus both histograms (unique counts, batch fills) in words 8..23.
    __device__ static void summary(const View& v, unsigned t, unsigned status, unsigned touched, unsigned batches, unsigned count,
                                   unsigned duplicates, unsigned maximum, unsigned rows, const unsigned* hist) {
        if (t < 8U) v.csr.summary[t] = t == 0U ? status : t == 1U ? touched : t == 2U ? batches : t == 3U ? count :
            t == 4U ? duplicates : t == 5U ? maximum : t == 6U ? rows : 0U;
        else if (t < 8U + 2U * kHistogramBins) v.csr.summary[t] = hist[t - 8U];
    }
};
using Plan = plan::PackedPlan<Policy>;

// The replaced csr_scan_kernel, verbatim.
template<class P,bool Prune=false>
__global__ __launch_bounds__(P::kThreads) void old_csr_scan_kernel(typename P::View v,unsigned rows,unsigned experts,unsigned top_k,unsigned cap,unsigned blocks,unsigned* scratch) {
    const auto& csr=P::tables(v);
    const unsigned t=threadIdx.x,count=rows*top_k;
    __shared__ unsigned n_of[P::kExperts],scan[3][P::kExperts],hist[2U*P::kHistogramBins],failure,status,duplicates,maximum;
    if(t==0){failure=scratch[0];status=scratch[1];duplicates=0U;maximum=0U;}
    if(t<2U*P::kHistogramBins)hist[t]=0U;
    __syncthreads();
    if(failure){P::failure(v,t,failure|status,count,rows);return;}
    if(t<P::kExperts) {
        unsigned n=0U;
        for(unsigned k=0;k<blocks;++k){unsigned* const c=scratch+plan::PackedPlan<P>::kHead+k*P::kExperts+t;const unsigned x=*c;*c=n;n+=x;}
        if(n>P::kCountLimit)atomicOr(&failure,P::kInvalid);
        n_of[t]=n;
    }
    __syncthreads();
    if(failure){if(t==0U)atomicOr(scratch,failure);P::failure(v,t,failure|status,count,rows);return;}
    const unsigned n=t<P::kExperts?n_of[t]:0U,nb=(n+cap-1U)/cap;
    if(t<P::kExperts){scan[0][t]=n;scan[1][t]=n!=0U;scan[2][t]=nb;}
    __syncthreads();
    for(unsigned d=1U;d<P::kExperts;d<<=1U) {
        unsigned x[3]={};
        if(t>=d&&t<P::kExperts)for(unsigned k=0;k<3U;++k)x[k]=scan[k][t-d];
        __syncthreads();
        if(t>=d&&t<P::kExperts)for(unsigned k=0;k<3U;++k)scan[k][t]+=x[k];
        __syncthreads();
    }
    if(t<experts) {
        const unsigned off=scan[0][t]-n,touched=scan[1][t]-(n!=0U),batch=scan[2][t]-nb;
        csr.offsets[t]=off;
        if(t==experts-1U)csr.offsets[experts]=off+n;
        if(n) {
            csr.touched[touched]=t;atomicMax(&maximum,n);
            for(unsigned b=0;b<nb;++b){csr.experts[batch+b]=t;csr.begins[batch+b]=off+b*cap;atomicAdd(&hist[P::kHistogramBins+min(P::kHistogramBins,min(cap,n-b*cap))-1U],1U);}
            const unsigned unique=n-scratch[2U+t];
            atomicAdd(&hist[min(P::kHistogramBins,unique)-1U],1U);atomicAdd(&duplicates,n-unique);
        }
    }
    __syncthreads();
    const unsigned planned=Prune?scan[0][P::kExperts-1U]:count;
    if(t==0U)csr.begins[scan[2][P::kExperts-1U]]=planned;
    P::summary(v,t,status,scan[1][P::kExperts-1U],scan[2][P::kExperts-1U],planned,duplicates,maximum,rows,hist);
}

enum class Routes { uniform, few, same, invalid_expert, nan_weight, skewed, repeats };
struct CsrCase { const char* name; unsigned rows, experts, cap; Routes kind; bool prune; };

struct DevTables {
    Tables t{};
    std::size_t words[7]{};
    DevTables(unsigned rows, unsigned experts) {
        const std::size_t count = std::size_t{rows} * Policy::kTopK;
        const std::size_t sizes[7] = {32U, Policy::kExperts + 1U, Policy::kExperts, count + 64U, count + 64U,
                                      count + Policy::kExperts + 64U, count + Policy::kExperts + 64U};
        unsigned** p[7] = {&t.summary, &t.offsets, &t.touched, &t.rows, &t.slots, &t.experts, &t.begins};
        (void)experts;
        for (unsigned i = 0; i < 7U; ++i) {
            words[i] = sizes[i];
            *p[i] = dev<unsigned>(sizes[i]);
            check(hipMemset(*p[i], 0xa5, sizes[i] * sizeof(unsigned)), "memset tables");
        }
    }
    std::vector<unsigned> get(unsigned i) const {
        const unsigned* const p[7] = {t.summary, t.offsets, t.touched, t.rows, t.slots, t.experts, t.begins};
        return down(p[i], words[i]);
    }
    void release() {
        unsigned* p[7] = {t.summary, t.offsets, t.touched, t.rows, t.slots, t.experts, t.begins};
        for (auto* q : p) check(hipFree(q), "free");
    }
};

struct Routing { std::vector<unsigned> routes, status; std::vector<float> weights; };
Routing make_routes(const CsrCase& c, std::mt19937& g) {
    const unsigned k = Policy::kTopK;
    Routing r;
    r.routes.resize(std::size_t{c.rows} * k);
    r.weights.resize(r.routes.size());
    r.status.assign(c.rows, 0U);
    std::uniform_int_distribution<unsigned> any(0U, c.experts - 1U);
    std::uniform_real_distribution<float> w(0.01F, 1.0F);
    std::vector<unsigned> pick(c.experts);
    for (unsigned e = 0; e < c.experts; ++e) pick[e] = e;
    for (unsigned row = 0; row < c.rows; ++row) {
        unsigned* x = r.routes.data() + std::size_t{row} * k;
        if (c.kind == Routes::uniform || c.kind == Routes::invalid_expert || c.kind == Routes::nan_weight) {
            for (unsigned s = 0; s < k; ++s) std::swap(pick[s], pick[s + any(g) % (c.experts - s)]);
            for (unsigned s = 0; s < k; ++s) x[s] = pick[s];
        } else if (c.kind == Routes::few) {
            for (unsigned s = 0; s < k; ++s) x[s] = (s * 37U + row % 3U) % 13U;  // 13 live experts, many empty
        } else if (c.kind == Routes::same) {
            for (unsigned s = 0; s < k; ++s) x[s] = 100U + s;
        } else if (c.kind == Routes::skewed) {
            for (unsigned s = 0; s < k; ++s) { const unsigned u = any(g); x[s] = (u * u / c.experts) % c.experts; }  // tails, empties
        } else {
            for (unsigned s = 0; s < k; ++s) x[s] = (row + s / 3U) % c.experts;  // repeated experts inside a row
        }
        for (unsigned s = 0; s < k; ++s) r.weights[std::size_t{row} * k + s] = c.prune && (s * 7U + row) % 4U == 3U ? 0.0F : w(g);
        if (row % 97U == 5U) r.status[row] = 1U << (row % 3U);
    }
    if (c.kind == Routes::invalid_expert) r.routes[r.routes.size() / 2U] = c.experts;
    if (c.kind == Routes::nan_weight) r.weights[r.weights.size() / 3U] = std::numeric_limits<float>::quiet_NaN();
    return r;
}

template <bool Prune, bool New>
void scan_launch(View v, const CsrCase& c, unsigned blocks, unsigned* scratch, hipStream_t s) {
    if constexpr (New) plan::csr_scan_kernel<Policy, Prune><<<1, Policy::kThreads, 0, s>>>(v, c.rows, c.experts, Policy::kTopK, c.cap, blocks, scratch);
    else old_csr_scan_kernel<Policy, Prune><<<1, Policy::kThreads, 0, s>>>(v, c.rows, c.experts, Policy::kTopK, c.cap, blocks, scratch);
}

template <bool Prune> void csr_case(const CsrCase& c, std::mt19937& g, bool bench) {
    const auto r = make_routes(c, g);
    unsigned* routes = dev<unsigned>(r.routes.size());
    float* weights = dev<float>(r.weights.size());
    unsigned* status = dev<unsigned>(r.status.size());
    up(routes, r.routes);
    up(weights, r.weights);
    up(status, r.status);
    const unsigned blocks = Plan::blocks_for(c.rows, Policy::kTopK);
    const std::size_t sw = Plan::scratch_bytes(c.rows) / sizeof(unsigned);
    unsigned* pristine = dev<unsigned>(sw);
    unsigned* sa = dev<unsigned>(sw);
    unsigned* sb = dev<unsigned>(sw);
    check(hipMemset(pristine, 0, sw * sizeof(unsigned)), "memset scratch");
    DevTables ta(c.rows, c.experts), tb(c.rows, c.experts);
    View va{routes, weights, status, ta.t, Policy::kTopK}, vb{routes, weights, status, tb.t, Policy::kTopK};
    plan::csr_count_kernel<Policy, false, Prune><<<blocks, Policy::kThreads>>>(va, c.rows, c.experts, Policy::kTopK, pristine);
    check(hipMemcpy(sa, pristine, sw * sizeof(unsigned), hipMemcpyDeviceToDevice), "copy");
    check(hipMemcpy(sb, pristine, sw * sizeof(unsigned), hipMemcpyDeviceToDevice), "copy");
    scan_launch<Prune, false>(va, c, blocks, sa, nullptr);
    scan_launch<Prune, true>(vb, c, blocks, sb, nullptr);
    plan::csr_scatter_kernel<Policy, Prune><<<blocks, Policy::kThreads>>>(va, c.rows, Policy::kTopK, sa);
    plan::csr_scatter_kernel<Policy, Prune><<<blocks, Policy::kThreads>>>(vb, c.rows, Policy::kTopK, sb);
    check(hipDeviceSynchronize(), "csr kernels");
    bool same = same_bytes(down(sa, sw), down(sb, sw));
    for (unsigned i = 0; i < 7U; ++i) same = same && same_bytes(ta.get(i), tb.get(i));
    const auto sum = ta.get(0);
    char what[160];
    std::snprintf(what, sizeof what, "csr %s rows %u experts %u cap %u prune %d identical", c.name, c.rows, c.experts, c.cap, int(Prune));
    expect(same, what);
    std::printf("csr %-9s rows %6u E %3u cap %5u prune %d blocks %3u: %s (fail/status %u touched %u batches %u planned %u dup %u max %u)\n",
                c.name, c.rows, c.experts, c.cap, int(Prune), blocks, same ? "identical" : "DIFFER", sum[0], sum[1], sum[2], sum[3], sum[4], sum[5]);
    if (bench) {
        hipEvent_t e0, e1;
        check(hipEventCreate(&e0), "event");
        check(hipEventCreate(&e1), "event");
        constexpr unsigned kRounds = 4U, kReps = 40U;
        double ms[2][kRounds] = {};
        for (unsigned round = 0; round < kRounds; ++round)
            for (unsigned arm = 0; arm < 2U; ++arm)
                for (unsigned rep = 0; rep < kReps; ++rep) {
                    check(hipMemcpyAsync(sa, pristine, sw * sizeof(unsigned), hipMemcpyDeviceToDevice, nullptr), "copy");
                    check(hipEventRecord(e0, nullptr), "record");
                    if (arm == 0U) scan_launch<Prune, false>(va, c, blocks, sa, nullptr);
                    else scan_launch<Prune, true>(va, c, blocks, sa, nullptr);
                    check(hipEventRecord(e1, nullptr), "record");
                    check(hipEventSynchronize(e1), "sync");
                    float t = 0.0F;
                    check(hipEventElapsedTime(&t, e0, e1), "elapsed");
                    ms[arm][round] += static_cast<double>(t) / kReps;
                }
        std::printf("  time scan blocks %3u (us/call) old:", blocks);
        for (unsigned round = 0; round < kRounds; ++round) std::printf(" %.2f", ms[0][round] * 1e3);
        std::printf(" | new:");
        for (unsigned round = 0; round < kRounds; ++round) std::printf(" %.2f", ms[1][round] * 1e3);
        std::printf("\n");
        check(hipEventDestroy(e0), "event");
        check(hipEventDestroy(e1), "event");
    }
    ta.release();
    tb.release();
    for (void* p : {static_cast<void*>(routes), static_cast<void*>(weights), static_cast<void*>(status), static_cast<void*>(pristine),
                    static_cast<void*>(sa), static_cast<void*>(sb)})
        check(hipFree(p), "free");
}

template <unsigned L> void tree_time() {
    constexpr unsigned kCtas = 32768U, kTrees = 3U, kRounds = 4U;
    float* out = dev<float>(kCtas);
    hipEvent_t e0, e1;
    check(hipEventCreate(&e0), "event");
    check(hipEventCreate(&e1), "event");
    tree_cost_kernel<L, false, kTrees><<<kCtas, L>>>(out);
    tree_cost_kernel<L, true, kTrees><<<kCtas, L>>>(out);
    check(hipDeviceSynchronize(), "warm");
    std::printf("  time tree L%u, %u CTAs x %u trees (ms) old/new:", L, kCtas, kTrees);
    for (unsigned round = 0; round < kRounds; ++round)
        for (unsigned arm = 0; arm < 2U; ++arm) {
            check(hipEventRecord(e0, nullptr), "record");
            if (arm == 0U) tree_cost_kernel<L, false, kTrees><<<kCtas, L>>>(out);
            else tree_cost_kernel<L, true, kTrees><<<kCtas, L>>>(out);
            check(hipEventRecord(e1, nullptr), "record");
            check(hipEventSynchronize(e1), "sync");
            float t = 0.0F;
            check(hipEventElapsedTime(&t, e0, e1), "elapsed");
            std::printf("%s%.3f", arm == 0U ? " " : "/", static_cast<double>(t));
        }
    std::printf("\n");
    check(hipEventDestroy(e0), "event");
    check(hipEventDestroy(e1), "event");
    check(hipFree(out), "free");
}

}  // namespace

int main(int argc, char** argv) {
    const bool bench = argc > 1 && std::strcmp(argv[1], "time") == 0;
    std::mt19937 g(20261115U);
    tree_case<64>(g);
    tree_case<128>(g);
    tree_case<256>(g);
    tree_case<512>(g);
    tree_case<1024>(g);
    const CsrCase cases[] = {
        {"uniform", 1U, 512U, 128U, Routes::uniform, false},     {"uniform", 7U, 512U, 1U, Routes::uniform, false},
        {"uniform", 204U, 512U, 128U, Routes::uniform, false},   {"uniform", 8192U, 512U, 128U, Routes::uniform, false},
        {"uniform", 8191U, 500U, 96U, Routes::uniform, false},   {"few", 8192U, 512U, 128U, Routes::few, false},
        {"same", 6000U, 512U, 128U, Routes::same, false},        {"same", 8192U, 512U, 65535U, Routes::same, false},
        {"skewed", 32768U, 512U, 128U, Routes::skewed, false},   {"repeats", 4099U, 512U, 7U, Routes::repeats, false},
        {"overflow", 70000U, 512U, 128U, Routes::same, false},   {"invalid", 3000U, 512U, 128U, Routes::invalid_expert, false},
        {"nan", 3000U, 512U, 128U, Routes::nan_weight, false},   {"uniform", 32768U, 512U, 128U, Routes::uniform, false},
        {"uniform", 131072U, 512U, 128U, Routes::uniform, false},
    };
    for (const auto& c : cases) csr_case<false>(c, g, false);
    for (const auto& c : cases) if (c.kind != Routes::nan_weight) csr_case<true>({c.name, c.rows, c.experts, c.cap, c.kind, true}, g, false);
    if (bench) {
        const CsrCase timed[] = {{"uniform", 204U, 512U, 128U, Routes::uniform, false}, {"uniform", 8192U, 512U, 128U, Routes::uniform, false},
                                 {"uniform", 32768U, 512U, 128U, Routes::uniform, false}, {"uniform", 131072U, 512U, 128U, Routes::uniform, false}};
        for (const auto& c : timed) csr_case<false>(c, g, true);
        tree_time<256>();
        tree_time<128>();
    }
    std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
}
