// Identity fixture of the PackedIU4 128x128 tile pipeline (load-cohort schedule seam).
// Shapes: gate/up K=6144 N=1536 and down K=1536 N=6144, scale group 64, four groups of M rows,
// M = 1, 127, 128, 129. For every case three FNV-1a hashes are pinned: the I32 group sums and the
// FP32 output of the dense launch (M rows), and the FP32 output of the grouped launch (4 x M rows).
// Kernel resources of the three instantiations are pinned as well. A load schedule may only
// change when these stay identical. The gate also runs the grouped kernel under two other
// schedules of the contract trait (weights issued late, everything issued late) and requires the
// same output bits: the schedule orders loads, it never changes an operand.
//   strix-hip-iu4-load-cohort-tests            gate (pinned hashes + resources)
//   strix-hip-iu4-load-cohort-tests print      print the table instead of comparing
//   strix-hip-iu4-load-cohort-tests bench R    routed replay timing: 4,096 token rows, 288 groups, top-8
// STRIX_IU4_COHORT_FIXTURE_LIBRARY_ONLY builds the library arms alone (to re-pin against an older core).
#include "core/wmma_tiled.hpp"
#ifndef STRIX_IU4_COHORT_FIXTURE_LIBRARY_ONLY
#include "core/tiled_core.hpp"
#endif
#include <hip/hip_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {
using namespace strix::core;
void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) { std::fprintf(stderr, "FAIL %s: %s\n", what, hipGetErrorString(e)); std::exit(1); }
}
struct Rng {
    std::uint64_t s;
    std::uint32_t next() { s = s * 6364136223846793005ULL + 1442695040888963407ULL; return std::uint32_t(s >> 33); }
};
std::uint64_t fnv(const void* p, std::size_t n, std::uint64_t h = 1469598103934665603ULL) {
    const auto* b = static_cast<const unsigned char*>(p);
    for (std::size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}
template<class T> struct Dev {
    T* p = nullptr; std::size_t n = 0;
    explicit Dev(std::size_t count) : n(count) { ok(hipMalloc(reinterpret_cast<void**>(&p), std::max<std::size_t>(count, 1) * sizeof(T)), "hipMalloc"); }
    ~Dev() { (void)hipFree(p); }
    Dev(const Dev&) = delete; Dev& operator=(const Dev&) = delete;
    void put(const std::vector<T>& v) { ok(hipMemcpy(p, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice), "upload"); }
    std::uint64_t hash(std::size_t count) const {
        std::vector<T> v(count);
        ok(hipMemcpy(v.data(), p, count * sizeof(T), hipMemcpyDeviceToHost), "download");
        return fnv(v.data(), count * sizeof(T));
    }
};
struct Shape { const char* name; int K, N; };
constexpr Shape kShapes[2] = {{"gate/up", 6144, 1536}, {"down", 1536, 6144}};
constexpr int kRows[4] = {1, 127, 128, 129};
constexpr int kExperts = 4, kTile = 2, kGroup = 64;

std::vector<std::uint8_t> bytes(std::size_t n, std::uint64_t seed) {
    std::vector<std::uint8_t> v(n); Rng r{seed};
    for (auto& x : v) x = std::uint8_t(r.next());
    return v;
}
std::vector<float> a_scales(std::size_t n, std::uint64_t seed) {
    std::vector<float> v(n); Rng r{seed};
    for (auto& x : v) x = float(1U + r.next() % 4093U) * (1.0f / 8192.0f);
    return v;
}
std::vector<_Float16> w_scales(std::size_t n, std::uint64_t seed) {
    std::vector<_Float16> v(n); Rng r{seed};
    for (auto& x : v) x = _Float16(float(1U + r.next() % 1021U) * (1.0f / 2048.0f));
    return v;
}
struct Hashes { std::uint64_t sums, dense, grouped; bool schedules_differ = false; };
#ifndef STRIX_IU4_COHORT_FIXTURE_LIBRARY_ONLY
// The grouped raw kernel of the library plan under an explicit load schedule.
template<unsigned Early> struct ScheduleContract : TiledContract<PackedIU4> { using Schedule = pipeline::CtaStageAhead<Early>; };
template<unsigned Early> hipError_t launch_schedule(const IU4GroupedOperands& o, const IU4GroupTile* tiles, const std::uint32_t* ntiles,
    std::uint32_t max_tiles, float* output, int inputs, int outputs) {
    constexpr auto c = iu4_tiles[kTile];
    using G = wmma_tiled::Geometry<c.bm, c.bn, c.wm, c.wn, PackedIU4>;
    wmma_tiled::GroupedArgs gp;
    gp.tiles = tiles; gp.ntiles = ntiles; gp.w_expert_bytes = o.w_expert_bytes; gp.w_scale_expert = o.w_scale_expert;
    gp.col_scale = nullptr; gp.map = o.map; gp.a_mul = o.a_mul; gp.a_add = o.a_add; gp.o_mul = o.o_mul; gp.o_add = o.o_add;
    wmma_tiled::tiled_core<ScheduleContract<Early>, c.bm, c.bn, c.wm, c.wn, false, 0, false, false, 64, true>
        <<<dim3{max_tiles * unsigned(outputs / c.bn)}, G::threads, 0, nullptr>>>(reinterpret_cast<const std::uint32_t*>(o.w_q), nullptr, output,
        inputs, outputs, 0, 1, nullptr, nullptr, nullptr, 1.f, 0.f, 1, o.a_q, o.a_scale, o.w_scale, nullptr, gp, wmma_tiled::NoAdapter{}, wmma_tiled::NoChain{});
    return hipGetLastError();
}
constexpr int kArms = 4;
constexpr const char* kArmName[kArms] = {"library", "all-early", "weights-late", "all-late"};
#else
constexpr int kArms = 1;
constexpr const char* kArmName[kArms] = {"library"};
#endif
hipError_t launch_arm(int arm, const IU4GroupedOperands& o, const IU4GroupTile* tiles, const std::uint32_t* ntiles,
    std::uint32_t max_tiles, float* output, int inputs, int outputs) {
    switch (arm) {
#ifndef STRIX_IU4_COHORT_FIXTURE_LIBRARY_ONLY
        case 1: return launch_schedule<pipeline::cohort_all>(o, tiles, ntiles, max_tiles, output, inputs, outputs);
        case 2: return launch_schedule<pipeline::cohort_a>(o, tiles, ntiles, max_tiles, output, inputs, outputs);
        case 3: return launch_schedule<0U>(o, tiles, ntiles, max_tiles, output, inputs, outputs);
#endif
        default: return launch_wmma_tiled_iu4_grouped(o, tiles, ntiles, max_tiles, output, inputs, outputs, nullptr, kTile);
    }
}
// Pinned on gfx1151 from the schedule that preceded the seam (issue s+1 -> matrix work -> publish -> barrier).
constexpr Hashes kExpected[2][4] = {
    {{0x842cd7b8624dbbfcULL, 0x3cda5edf81935d9eULL, 0x4c49be713973723bULL},
     {0xd52616eff674bbd9ULL, 0xd11a03a89510be81ULL, 0xca17a40d63a02612ULL},
     {0xb04a29218b789ae5ULL, 0x39d3cd895e89bc71ULL, 0x7ea3de79e5032cceULL},
     {0xf8b7c202c6492b85ULL, 0x4a3daa1a60334570ULL, 0x1da43f1ff20916d2ULL}},
    {{0x6866fe2f4d3f876fULL, 0x653e06eabbb6e9e0ULL, 0x3ea2a1b283846195ULL},
     {0x46a6a887f5a71801ULL, 0x36340f640a3c370cULL, 0x6e5b20a0ae64ec46ULL},
     {0x69268aa6d7f3e79bULL, 0x3f57bd9c2f82bd6bULL, 0x079343bea0b40e2fULL},
     {0x4fac7c8a96d2a422ULL, 0x846eebfa3c1dcbabULL, 0x084e60181095f0fbULL}},
};
// {dense, grouped raw, grouped transform}: LDS bytes exact, no scratch, VGPR ceiling.
constexpr int kLdsBytes = 36864, kVgprCap[3] = {221, 209, 223};

Hashes run_case(const Shape& s, int M) {
    const std::size_t K = std::size_t(s.K), N = std::size_t(s.N), groups = K / kGroup, rows = std::size_t(M) * kExperts;
    const std::size_t wq_e = K * N / 2, ws_e = N * groups;
    Dev<std::uint8_t> aq(rows * K / 2 + 16), wq(wq_e * kExperts + 16);
    Dev<float> as(rows * groups), out(rows * N);
    Dev<_Float16> ws(ws_e * kExperts);
    aq.put(bytes(rows * K / 2, 0x51ULL + K + std::size_t(M)));
    wq.put(bytes(wq_e * kExperts, 0x77ULL + N));
    as.put(a_scales(rows * groups, 0x91ULL + std::size_t(M)));
    ws.put(w_scales(ws_e * kExperts, 0xb3ULL + K));
    Hashes h{};
    {   // Dense: the first group's M rows against expert 0, with the I32 group-sum audit.
        Dev<std::int32_t> sums(groups * std::size_t(M) * N);
        IU4Operands o; o.a_q = aq.p; o.a_scale = as.p; o.w_q = wq.p; o.w_scale = ws.p; o.group = kGroup;
        ok(hipMemset(out.p, 0xff, rows * N * sizeof(float)), "memset");
        ok(launch_wmma_tiled_iu4(o, out.p, s.K, s.N, M, nullptr, kTile, sums.p), "dense launch");
        ok(hipDeviceSynchronize(), "sync");
        h.sums = sums.hash(groups * std::size_t(M) * N);
        h.dense = out.hash(std::size_t(M) * N);
    }
    {   // Grouped: four groups of M rows, raw FP32 stores.
        std::vector<IU4Group> g(kExperts);
        for (int e = 0; e < kExperts; ++e) g[std::size_t(e)] = {std::uint32_t(e), std::uint32_t(e * M), std::uint32_t(M)};
        const std::size_t max_tiles = iu4_grouped_max_tiles(rows, kExperts, kTile);
        Dev<IU4Group> dg(kExperts); dg.put(g);
        Dev<IU4GroupTile> tiles(max_tiles); Dev<std::uint32_t> ntiles(1);
        ok(build_iu4_grouped_tiles(dg.p, kExperts, kTile, tiles.p, ntiles.p, std::uint32_t(max_tiles), nullptr), "tiles");
        IU4GroupedOperands o; o.a_q = aq.p; o.a_scale = as.p; o.w_q = wq.p; o.w_scale = ws.p;
        o.w_expert_bytes = wq_e; o.w_scale_expert = ws_e; o.group = kGroup;
        for (int arm = 0; arm < kArms; ++arm) {
            ok(hipMemset(out.p, 0xff, rows * N * sizeof(float)), "memset");
            ok(launch_arm(arm, o, tiles.p, ntiles.p, std::uint32_t(max_tiles), out.p, s.K, s.N), "grouped launch");
            ok(hipDeviceSynchronize(), "sync");
            const std::uint64_t g = out.hash(rows * N);
            if (arm == 0) h.grouped = g;
            else if (g != h.grouped) { std::printf("%s M=%d: schedule %s differs from the library kernel\n", s.name, M, kArmName[arm]); h.schedules_differ = true; }
        }
    }
    return h;
}

int gate(bool print) {
    int bad = 0;
    IU4Resources r[3];
    ok(query_wmma_iu4_resources(kTile, &r[0], false, kGroup), "query dense");
    ok(query_wmma_iu4_grouped_resources(kTile, false, &r[1], kGroup), "query grouped");
    ok(query_wmma_iu4_grouped_resources(kTile, true, &r[2], kGroup), "query grouped transform");
    for (int i = 0; i < 3; ++i) {
        std::printf("resources[%d] vgpr %d active_waves %d scratch %zu lds %zu\n", i, r[i].vgpr, r[i].active_waves, r[i].scratch_bytes, r[i].lds_bytes);
        if (!print && (r[i].lds_bytes != std::size_t(kLdsBytes) || r[i].scratch_bytes != 0 || r[i].vgpr > kVgprCap[i])) { std::printf("  RESOURCE MISMATCH\n"); ++bad; }
    }
    for (int si = 0; si < 2; ++si)
        for (int mi = 0; mi < 4; ++mi) {
            const Hashes h = run_case(kShapes[si], kRows[mi]);
            const Hashes& e = kExpected[si][mi];
            const bool same = h.sums == e.sums && h.dense == e.dense && h.grouped == e.grouped;
            if (h.schedules_differ) ++bad;
            std::printf("%s M=%d {0x%016llxULL, 0x%016llxULL, 0x%016llxULL}%s\n", kShapes[si].name, kRows[mi],
                static_cast<unsigned long long>(h.sums), static_cast<unsigned long long>(h.dense),
                static_cast<unsigned long long>(h.grouped), print ? "" : (same ? " ok" : " MISMATCH"));
            if (!print && !same) ++bad;
        }
    if (bad) { std::printf("FAIL: %d mismatches\n", bad); return 1; }
    std::printf(print ? "printed\n" : "PASS\n");
    return 0;
}

// Routed replay: 4,096 token rows x top-8 over 288 groups (224 of 114 rows, 64 of 113), resident
// weights for all groups. Reports the median grouped GEMM time per shape and its output hash.
int bench(int rounds) {
    constexpr int E = 288, pairs = 4096 * 8;
    std::vector<IU4Group> g(E);
    for (int e = 0, first = 0; e < E; ++e) { const int c = e < 224 ? 114 : 113; g[std::size_t(e)] = {std::uint32_t(e), std::uint32_t(first), std::uint32_t(c)}; first += c; }
    const std::size_t max_tiles = iu4_grouped_max_tiles(pairs, E, kTile);
    Dev<IU4Group> dg(E); dg.put(g);
    Dev<IU4GroupTile> tiles(max_tiles); Dev<std::uint32_t> ntiles(1);
    ok(build_iu4_grouped_tiles(dg.p, E, kTile, tiles.p, ntiles.p, std::uint32_t(max_tiles), nullptr), "tiles");
    struct Arm { Dev<std::uint8_t>* aq; Dev<std::uint8_t>* wq; Dev<float>* as; Dev<_Float16>* ws; Dev<float>* out; IU4GroupedOperands o; std::vector<double> ms[kArms]; };
    Arm arm[2];
    for (int si = 0; si < 2; ++si) {
        const std::size_t K = std::size_t(kShapes[si].K), N = std::size_t(kShapes[si].N), groups = K / kGroup, wq_e = K * N / 2, ws_e = N * groups;
        Arm& a = arm[si];
        a.aq = new Dev<std::uint8_t>(std::size_t(pairs) * K / 2 + 16); a.wq = new Dev<std::uint8_t>(wq_e * E + 16);
        a.as = new Dev<float>(std::size_t(pairs) * groups); a.ws = new Dev<_Float16>(ws_e * E); a.out = new Dev<float>(std::size_t(pairs) * N);
        a.aq->put(bytes(std::size_t(pairs) * K / 2, 0x1234ULL + K));
        a.as->put(a_scales(std::size_t(pairs) * groups, 0x4321ULL + K));
        // Eight distinct experts, replicated on the device: the stream is DRAM-sized, the bytes repeat.
        a.wq->put(bytes(wq_e * 8, 0x9999ULL + N)); a.ws->put(w_scales(ws_e * 8, 0x5555ULL + N));
        for (std::size_t have = 8; have < std::size_t(E);) {
            const std::size_t n = std::min(have, std::size_t(E) - have);
            ok(hipMemcpy(a.wq->p + have * wq_e, a.wq->p, n * wq_e, hipMemcpyDeviceToDevice), "replicate");
            ok(hipMemcpy(a.ws->p + have * ws_e, a.ws->p, n * ws_e * sizeof(_Float16), hipMemcpyDeviceToDevice), "replicate");
            have += n;
        }
        a.o.a_q = a.aq->p; a.o.a_scale = a.as->p; a.o.w_q = a.wq->p; a.o.w_scale = a.ws->p;
        a.o.w_expert_bytes = wq_e; a.o.w_scale_expert = ws_e; a.o.group = kGroup;
    }
    std::uint64_t hash[2][kArms] = {};
    for (int r = -1; r < rounds; ++r)
        for (int k = 0; k < kArms; ++k) {
            const int ka = (r & 1) ? kArms - 1 - k : k;   // alternate the arm order every round
            for (int si = 0; si < 2; ++si) {
                Arm& a = arm[si];
                ok(hipDeviceSynchronize(), "sync");
                const auto t0 = std::chrono::steady_clock::now();
                ok(launch_arm(ka, a.o, tiles.p, ntiles.p, std::uint32_t(max_tiles), a.out->p, kShapes[si].K, kShapes[si].N), "launch");
                ok(hipDeviceSynchronize(), "sync");
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (r >= 0) a.ms[ka].push_back(ms);
                else hash[si][ka] = a.out->hash(std::size_t(pairs) * std::size_t(kShapes[si].N));
            }
        }
    for (int k = 0; k < kArms; ++k) {
        double total = 0;
        for (int si = 0; si < 2; ++si) {
            auto& ms = arm[si].ms[k];
            std::sort(ms.begin(), ms.end());
            const double med = ms[ms.size() / 2];
            const double tops = 2.0 * double(pairs) * double(kShapes[si].K) * double(kShapes[si].N) / (med * 1e-3) * 1e-12;
            total += med;
            std::printf("bench %-12s %-7s K=%d N=%d pairs=%d median %.3f ms min %.3f max %.3f  %.1f TOPS  hash 0x%016llx%s\n", kArmName[k], kShapes[si].name,
                kShapes[si].K, kShapes[si].N, pairs, med, ms.front(), ms.back(), tops, static_cast<unsigned long long>(hash[si][k]),
                hash[si][k] == hash[si][0] ? "" : " DIFFERS");
        }
        std::printf("bench %-12s combined median %.3f ms (%d rounds, interleaved)\n", kArmName[k], total, rounds);
    }
    return 0;
}
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "bench") == 0) return bench(argc > 2 ? std::max(1, std::atoi(argv[2])) : 9);
    return gate(argc > 1 && std::strcmp(argv[1], "print") == 0);
}
