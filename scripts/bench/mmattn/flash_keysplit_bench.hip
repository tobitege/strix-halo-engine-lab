// Microbench of the shared FAST key-split flash core (src/attn_core/sparse_flash_keysplit.hpp) at the MiniMax MSA
// prefill shape: 16 query heads per KV group, D 128, 128-key segments, top-16 selected blocks (newest always), BF16
// caches laid out [pos][kv_heads][128]. Synthetic data; the selection of a row is shared by groups of R rows
// (R = 1: independent rows, larger R: more cache reuse between neighbouring work-groups).
//
// Build: amdclang++ -x hip --offload-arch=gfx1151 -O3 -std=c++20 -I<repo>/src -I<repo>/include flash_keysplit_bench.hip
// Run:   flash_keysplit_bench POS0 ROWS R OPT [SPLIT]   -> one line: ms (median of 20), TFLOPS, output fnv64
//        POS0 = position of the first row, ROWS = rows in the launch, OPT = core Opt bits (0 = incumbent schedule);
//        OPT 8 + b = row-group form (sparse_flash_rowgroup.hpp, 4 rows per work-group, Opt bits b = 0..3; SPLIT 0).
//        [DRIFT]: if given (>= 0), each row's selection is its predecessor's with DRIFT of the 15 older blocks
//        replaced (R ignored). OPT != 0 also prints max |out - out(OPT 0)| (same SPLIT) and max |out(OPT 0)|.
#include "attn_core/sparse_flash_keysplit.hpp"
#include "attn_core/sparse_flash_rowgroup.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {
constexpr unsigned kH = 16, kD = 128, kBlk = 128, kTop = 16, kKvh = 2;

struct BenchPolicy {
    struct Args {
        const std::uint16_t* kc;
        const std::uint16_t* vc;
        const float* q;
        const std::uint32_t* ids;
        float* o;
        std::uint32_t rows, pos0;
    };
    struct Row {
        const std::uint16_t* kc;
        const std::uint16_t* vc;
        const float* q;
        const std::uint32_t* ids;
        float* o;
        std::uint64_t count;
        std::uint32_t cnt, g, all;
    };
    static constexpr unsigned kHeads = kH, kDim = kD, kSegKeys = kBlk;
    static constexpr strix::attn_core::tile::Operand kOp = strix::attn_core::tile::Operand::bf16;
    static constexpr float kScaleLog2 = 0.08838834764831845F * 1.44269504088896341F;
    static __device__ __forceinline__ bool begin(const Args& a, unsigned wg, Row& x) {
        const unsigned r = wg / kKvh, g = wg % kKvh;
        if (r >= a.rows) return false;
        x.count = a.pos0 + r + 1ULL;
        const std::uint32_t nb = static_cast<std::uint32_t>((x.count + kBlk - 1) / kBlk);
        x.all = nb <= kTop ? 1U : 0U;
        x.cnt = x.all != 0U ? nb : kTop;
        x.ids = a.ids + (static_cast<std::size_t>(r) * kKvh + g) * kTop;
        x.kc = a.kc;
        x.vc = a.vc;
        x.q = a.q + (static_cast<std::size_t>(r) * kKvh + g) * kH * kD;
        x.o = a.o + (static_cast<std::size_t>(r) * kKvh + g) * kH * kD;
        x.g = g;
        return true;
    }
    static __device__ __forceinline__ unsigned segments(const Args&, const Row& x) { return x.cnt; }
    static __device__ __forceinline__ std::uint64_t first(const Row& x, unsigned c) {
        return static_cast<std::uint64_t>(x.all != 0U ? c : x.ids[c]) * kBlk;
    }
    static __device__ __forceinline__ unsigned seg_keys(const Args&, const Row& x, unsigned c) {
        const std::uint64_t lo = first(x, c);
        return static_cast<unsigned>((x.count < lo + kBlk ? x.count : lo + kBlk) - lo);
    }
    static __device__ __forceinline__ const std::uint16_t* key_row(const Args&, const Row& x, unsigned c, unsigned k) {
        return x.kc + ((first(x, c) + k) * kKvh + x.g) * kD;
    }
    static __device__ __forceinline__ const std::uint16_t* value_row(const Args&, const Row& x, unsigned c, unsigned k) {
        return x.vc + ((first(x, c) + k) * kKvh + x.g) * kD;
    }
    static __device__ __forceinline__ void query(const Args&, const Row& x, unsigned head, unsigned lane, float (&q)[4]) {
        const float4 v = *reinterpret_cast<const float4*>(x.q + static_cast<std::size_t>(head) * kD + 4U * lane);
        q[0] = v.x;
        q[1] = v.y;
        q[2] = v.z;
        q[3] = v.w;
    }
    static __device__ __forceinline__ float* out(const Args&, const Row& x) { return x.o; }
    static constexpr unsigned kMaxSegs = kTop;
    static __device__ __forceinline__ unsigned member(const Args& a, unsigned group, unsigned i, unsigned n) {
        const unsigned r = n * (group / kKvh) + i;
        return r < a.rows ? r * kKvh + group % kKvh : ~0U;
    }
    static __device__ __forceinline__ std::size_t key_stride(const Args&) { return std::size_t{kKvh} * kD; }
};

std::uint16_t bf16(float f) {
    std::uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    return static_cast<std::uint16_t>((u + 0x7fffU + ((u >> 16) & 1U)) >> 16);
}

void check(hipError_t e, const char* what) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
        std::exit(2);
    }
}

template <unsigned Opt, bool Split> void launch(const BenchPolicy::Args& a, unsigned wgs) {
    namespace ks = strix::attn_core::sparse;
    hipLaunchKernelGGL((ks::flash_rows_keysplit_kernel<BenchPolicy, Split, Opt>), dim3(wgs), dim3(ks::KeySplit<kH, kD>::kThreads), 0,
                       nullptr, a);
}
template <unsigned Opt, unsigned Rows = 4> void launch_rowgroup(const BenchPolicy::Args& a, unsigned rows) {
    namespace ks = strix::attn_core::sparse;
    using G = ks::RowGroup<kH, kD, kTop, Rows>;
    hipLaunchKernelGGL((ks::flash_rowgroup_kernel<BenchPolicy, Opt, Rows>), dim3((rows + G::kRows - 1) / G::kRows * kKvh), dim3(G::kThreads), 0,
                       nullptr, a);
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: %s POS0 ROWS R OPT [SPLIT]\n", argv[0]);
        return 1;
    }
    const unsigned pos0 = std::atoi(argv[1]), rows = std::atoi(argv[2]), reuse = std::max(1, std::atoi(argv[3]));
    const unsigned opt = std::atoi(argv[4]), split = argc > 5 ? std::atoi(argv[5]) : 0;
    const int drift = argc > 6 ? std::atoi(argv[6]) : -1;
    const std::size_t npos = pos0 + rows;
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> u(-1.0F, 1.0F);
    std::vector<std::uint16_t> kc(npos * kKvh * kD), vc(kc.size());
    for (auto& x : kc) x = bf16(u(rng));
    for (auto& x : vc) x = bf16(u(rng));
    std::vector<float> q(std::size_t{rows} * kKvh * kH * kD);
    for (auto& x : q) x = 2.0F * u(rng);
    std::vector<std::uint32_t> ids(std::size_t{rows} * kKvh * kTop);
    std::uint64_t keys = 0;
    for (unsigned r = 0; r < rows; ++r) {
        const unsigned nb = (pos0 + r + 1 + kBlk - 1) / kBlk;
        keys += kKvh * std::uint64_t{nb <= kTop ? pos0 + r + 1 : (kTop - 1) * kBlk + (pos0 + r + 1 - (nb - 1) * kBlk)};
        if (nb <= kTop) continue;
        if (drift >= 0) {
            static std::vector<std::vector<std::uint32_t>> prev(kKvh);
            static std::mt19937 dr(777U);
            for (unsigned g = 0; g < kKvh; ++g) {
                std::vector<std::uint32_t>& cur = prev[g];
                const auto has = [&](std::uint32_t b) { return std::find(cur.begin(), cur.end(), b) != cur.end(); };
                if (cur.empty()) {
                    while (cur.size() < kTop - 1) {
                        const std::uint32_t b = dr() % (nb - 1);
                        if (!has(b)) cur.push_back(b);
                    }
                } else {
                    for (int i = 0; i < drift; ++i) {
                        std::uint32_t b = dr() % (nb - 1);
                        while (has(b)) b = dr() % (nb - 1);
                        cur[dr() % cur.size()] = b;
                    }
                }
                std::sort(cur.begin(), cur.end());
                std::uint32_t* d = &ids[(std::size_t{r} * kKvh + g) * kTop];
                for (unsigned i = 0; i < kTop - 1; ++i) d[i] = cur[i];
                d[kTop - 1] = nb - 1;
            }
            continue;
        }
        for (unsigned g = 0; g < kKvh; ++g) {
            std::mt19937 sr(1000003U * (r / reuse) + 7U * g + 1U);
            std::vector<std::uint32_t> pool(nb - 1);
            for (unsigned i = 0; i < nb - 1; ++i) pool[i] = i;
            for (unsigned i = 0; i < kTop - 1; ++i) std::swap(pool[i], pool[i + sr() % (nb - 1 - i)]);
            std::sort(pool.begin(), pool.begin() + (kTop - 1));
            std::uint32_t* d = &ids[(std::size_t{r} * kKvh + g) * kTop];
            for (unsigned i = 0; i < kTop - 1; ++i) d[i] = pool[i];
            d[kTop - 1] = nb - 1;
        }
    }
    std::uint16_t *dk, *dv;
    float *dq, *dout;
    std::uint32_t* dids;
    check(hipMalloc(&dk, kc.size() * 2), "malloc");
    check(hipMalloc(&dv, vc.size() * 2), "malloc");
    check(hipMalloc(&dq, q.size() * 4), "malloc");
    check(hipMalloc(&dids, ids.size() * 4), "malloc");
    check(hipMalloc(&dout, q.size() * 4), "malloc");
    check(hipMemcpy(dk, kc.data(), kc.size() * 2, hipMemcpyHostToDevice), "copy");
    check(hipMemcpy(dv, vc.data(), vc.size() * 2, hipMemcpyHostToDevice), "copy");
    check(hipMemcpy(dq, q.data(), q.size() * 4, hipMemcpyHostToDevice), "copy");
    check(hipMemcpy(dids, ids.data(), ids.size() * 4, hipMemcpyHostToDevice), "copy");
    const BenchPolicy::Args a{dk, dv, dq, dids, dout, rows, pos0};
    const unsigned wgs = rows * kKvh;
    const auto run_opt = [&](unsigned opt) {
        const unsigned key = opt * 2U + (split != 0 ? 1U : 0U);
        switch (key) {
            case 0: launch<0, false>(a, wgs); break;
            case 1: launch<0, true>(a, wgs); break;
            case 2: launch<1, false>(a, wgs); break;
            case 3: launch<1, true>(a, wgs); break;
            case 4: launch<2, false>(a, wgs); break;
            case 5: launch<2, true>(a, wgs); break;
            case 16: launch_rowgroup<0>(a, rows); break;
            case 18: launch_rowgroup<1>(a, rows); break;
            case 20: launch_rowgroup<2>(a, rows); break;
            case 22: launch_rowgroup<3>(a, rows); break;
            case 24: launch_rowgroup<1, 8>(a, rows); break;  // OPT 12: lazy rescale, 8 rows per work-group
            default: std::fprintf(stderr, "unknown OPT\n"); std::exit(1);
        }
    };
    const auto run = [&] { run_opt(opt); };
    for (int i = 0; i < 3; ++i) run();
    check(hipDeviceSynchronize(), "warmup");
    hipEvent_t e0, e1;
    check(hipEventCreate(&e0), "event");
    check(hipEventCreate(&e1), "event");
    std::vector<float> ms;
    for (int i = 0; i < 20; ++i) {
        check(hipEventRecord(e0, nullptr), "rec");
        run();
        check(hipEventRecord(e1, nullptr), "rec");
        check(hipEventSynchronize(e1), "sync");
        float t = 0;
        check(hipEventElapsedTime(&t, e0, e1), "elapsed");
        ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    std::vector<float> out(q.size());
    check(hipMemcpy(out.data(), dout, out.size() * 4, hipMemcpyDeviceToHost), "copy back");
    std::uint64_t h = 1469598103934665603ULL;
    const auto* b = reinterpret_cast<const unsigned char*>(out.data());
    for (std::size_t i = 0; i < out.size() * 4; ++i) h = (h ^ b[i]) * 1099511628211ULL;
    float diff = 0.0F, big = 0.0F;
    if (opt != 0) {
        check(hipMemset(dout, 0, out.size() * 4), "clear");
        run_opt(0);
        std::vector<float> ref(out.size());
        check(hipMemcpy(ref.data(), dout, ref.size() * 4, hipMemcpyDeviceToHost), "copy back");
        for (std::size_t i = 0; i < ref.size(); ++i) {
            diff = std::max(diff, std::fabs(out[i] - ref[i]));
            big = std::max(big, std::fabs(ref[i]));
        }
    }
    const double flop = double(keys) * kH * kD * 4.0 * (split != 0 ? 2.0 : 1.0);
    std::printf("pos0 %u rows %u R %u opt %u split %u ms %.3f min %.3f TFLOPS %.2f fnv %016llx drift %d maxdiff %.3g maxref %.3g\n", pos0, rows, reuse, opt, split, ms[10],
                ms[0], flop / (ms[10] * 1e-3) / 1e12, static_cast<unsigned long long>(h), drift, diff, big);
    return 0;
}
