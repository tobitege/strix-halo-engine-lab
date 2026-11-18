// Sustained RAM -> device bandwidth of the transfer shapes the prefill expert provider can use (not a test).
//
//   strix-cuda-ds41-xfer-bench [--pinned-gib G] [--slot-gib S] [--rounds N] [--device D]
//
// Source: G GiB of host memory, destination: two device slots of S GiB (a layer's routed experts are 3.82 GB). Every
// case copies the whole source `rounds` times into the slots in turn and reports the sustained GB/s over all of it
// (one JSON line per case):
//   whole    the slot filled by 3 copies (gate / up / down: the provider's current whole-layer copy)
//   piece P  the same bytes as P MiB pieces on one copy stream (events per piece, as the per-group transport records)
//   pool P   the source as separate P MiB pinned allocations (persistent buffers) instead of one large one
//   stage P  pageable source, 3 host threads copying into 4 persistent P MiB pinned buffers, DMA from those
// --interleave K: instead, K passes over {hot, whole, piece 64/256, pool 64/256}, one slot's worth each (the host's
// load drifts: alternating the shapes compares them under the same conditions); median and best GB/s per shape.
#include "strix/cuda/ds41_common.hpp"

#include <algorithm>
#include <map>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace {

#define CK(x)                                                                                              \
    do {                                                                                                   \
        const cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) {                                                                           \
            std::fprintf(stderr, "CUDA %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
            std::exit(1);                                                                                  \
        }                                                                                                  \
    } while (0)

constexpr std::size_t MiB = std::size_t{1} << 20;

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

struct Ctx {
    std::size_t src_bytes = 0, slot_bytes = 0;
    int rounds = 3;
    std::uint8_t* dev[2] = {nullptr, nullptr};
    cudaStream_t copy[2] = {nullptr, nullptr};
};

void report(const char* mode, std::size_t piece, int streams, std::size_t bytes, double s, const char* extra = "") {
    std::printf("{\"mode\":\"%s\",\"piece_mib\":%zu,\"streams\":%d,\"gb\":%.2f,\"seconds\":%.3f,\"gbps\":%.2f%s}\n", mode, piece / MiB, streams,
                double(bytes) / 1e9, s, double(bytes) / s / 1e9, extra);
    std::fflush(stdout);
}

// Copies `src` (any view: a list of host ranges in order) into the slots in pieces; slot k % 2 per pass.
struct Range {
    const std::uint8_t* p;
    std::size_t n;
};
double copy_ranges(Ctx& c, const std::vector<Range>& ranges, std::size_t piece, int streams) {
    std::vector<cudaEvent_t> ev;
    CK(cudaDeviceSynchronize());
    const double t0 = now_s();
    std::size_t total = 0;
    for (int r = 0; r < c.rounds; ++r) {
        std::uint8_t* dst = c.dev[r % 2];
        std::size_t at = 0;
        int k = 0;
        for (const auto& g : ranges)
            for (std::size_t d = 0; d < g.n; d += piece) {
                const std::size_t n = std::min(piece, g.n - d);
                if (at + n > c.slot_bytes) at = 0;
                cudaStream_t s = c.copy[(k++) % streams];
                CK(cudaMemcpyAsync(dst + at, g.p + d, n, cudaMemcpyHostToDevice, s));
                cudaEvent_t e;
                CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
                CK(cudaEventRecord(e, s));
                ev.push_back(e);
                at += n;
                total += n;
            }
    }
    CK(cudaDeviceSynchronize());
    const double s = now_s() - t0;
    for (auto e : ev) cudaEventDestroy(e);
    return double(total) / s / 1e9;
}

}  // namespace

int main(int argc, char** argv) {
    double pinned_gib = 4.0, slot_gib = 3.82 * 1e9 / double(1ULL << 30);
    int rounds = 3, device = 0, interleave = 0;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--pinned-gib" && i + 1 < argc) pinned_gib = std::atof(argv[++i]);
        else if (a == "--slot-gib" && i + 1 < argc) slot_gib = std::atof(argv[++i]);
        else if (a == "--rounds" && i + 1 < argc) rounds = std::atoi(argv[++i]);
        else if (a == "--device" && i + 1 < argc) device = std::atoi(argv[++i]);
        else if (a == "--interleave" && i + 1 < argc) interleave = std::atoi(argv[++i]);
        else {
            std::fprintf(stderr, "usage: strix-cuda-ds41-xfer-bench [--pinned-gib G] [--slot-gib S] [--rounds N] [--device D]\n");
            return 2;
        }
    }
    CK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, device));
    std::printf("{\"device\":\"%s\",\"sms\":%d,\"pinned_gib\":%.2f,\"slot_gib\":%.2f,\"rounds\":%d}\n", prop.name, prop.multiProcessorCount, pinned_gib,
                slot_gib, rounds);
    Ctx c;
    c.rounds = rounds;
    c.src_bytes = static_cast<std::size_t>(pinned_gib * double(1ULL << 30)) / (256U * MiB) * (256U * MiB);
    c.slot_bytes = static_cast<std::size_t>(slot_gib * double(1ULL << 30));
    for (auto& d : c.dev) CK(cudaMalloc(&d, c.slot_bytes));
    for (auto& s : c.copy) CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    const std::size_t per_round = std::min(c.src_bytes, c.slot_bytes);
    if (interleave > 0) {
        // one large pinned source (the provider's host tier) and persistent 64 / 256 MiB pinned pieces
        std::uint8_t* big = nullptr;
        CK(cudaHostAlloc(reinterpret_cast<void**>(&big), per_round, cudaHostAllocDefault));
        std::memset(big, 1, per_round);
        std::map<std::size_t, std::vector<std::uint8_t*>> pool;
        for (const std::size_t p : {64U * MiB, 256U * MiB}) {
            for (std::size_t at = 0; at + p <= per_round / 2; at += p) {
                std::uint8_t* b = nullptr;
                CK(cudaHostAlloc(reinterpret_cast<void**>(&b), p, cudaHostAllocDefault));
                std::memset(b, 1, p);
                pool[p].push_back(b);
            }
        }
        c.rounds = 1;
        std::map<std::string, std::vector<double>> got;
        const std::size_t g = per_round * 3 / 10, d = per_round - 2 * g;
        for (int k = 0; k < interleave; ++k) {
            got["hot256"].push_back(copy_ranges(c, std::vector<Range>(per_round / (256U * MiB), Range{big, 256U * MiB}), 256U * MiB, 1));
            got["whole"].push_back(copy_ranges(c, {{big, g}, {big + g, g}, {big + 2 * g, d}}, per_round, 1));
            got["piece64"].push_back(copy_ranges(c, {{big, per_round}}, 64U * MiB, 1));
            got["piece256"].push_back(copy_ranges(c, {{big, per_round}}, 256U * MiB, 1));
            for (const std::size_t p : {64U * MiB, 256U * MiB}) {
                std::vector<Range> r;
                for (std::size_t i = 0; i * p < per_round; ++i) r.push_back({pool[p][i % pool[p].size()], p});
                got[p == 64U * MiB ? "pool64" : "pool256"].push_back(copy_ranges(c, r, p, 1));
            }
        }
        for (auto& [name, v] : got) {
            std::sort(v.begin(), v.end());
            std::printf("{\"mode\":\"%s\",\"passes\":%zu,\"gb_per_pass\":%.2f,\"median_gbps\":%.2f,\"best_gbps\":%.2f,\"worst_gbps\":%.2f}\n",
                        name.c_str(), v.size(), double(per_round) / 1e9, v[v.size() / 2], v.back(), v.front());
        }
        for (auto& [p, v] : pool)
            for (auto b : v) CK(cudaFreeHost(b));
        CK(cudaFreeHost(big));
        for (auto dv : c.dev) cudaFree(dv);
        return 0;
    }

    // one large pinned allocation
    {
        std::uint8_t* big = nullptr;
        CK(cudaHostAlloc(reinterpret_cast<void**>(&big), c.src_bytes, cudaHostAllocDefault));
        std::memset(big, 1, c.src_bytes);
        // hot: the first 256 MiB of the source again and again (the ceiling of the moment: small-buffer bandwidth)
        {
            std::vector<Range> r(per_round / (256U * MiB), Range{big, 256U * MiB});
            const double gbps = copy_ranges(c, r, 256U * MiB, 1);
            report("hot", 256U * MiB, 1, r.size() * 256U * MiB * rounds, double(r.size() * 256U * MiB * rounds) / gbps / 1e9);
        }
        // whole: 3 copies of the layer's tensors (gate, up = 2 x 0.30, down 0.40 of it), one stream
        {
            const std::size_t g = per_round * 3 / 10, d = per_round - 2 * g;
            const std::vector<Range> r{{big, g}, {big + g, g}, {big + 2 * g, d}};
            copy_ranges(c, r, per_round, 1);  // warm
            report("whole", 0, 1, per_round * rounds, double(per_round * rounds) / copy_ranges(c, r, per_round, 1) / 1e9);
        }
        for (const std::size_t p : {16U * MiB, 64U * MiB, 128U * MiB, 256U * MiB, 512U * MiB})
            for (int streams = 1; streams <= 2; ++streams) {
                const std::vector<Range> r{{big, per_round}};
                const double gbps = copy_ranges(c, r, p, streams);
                report("piece", p, streams, per_round * rounds, double(per_round * rounds) / gbps / 1e9);
            }
        // the whole source (larger than a slot), as the provider's pinned host tier is
        {
            const std::vector<Range> r{{big, c.src_bytes}};
            const double gbps = copy_ranges(c, r, 256U * MiB, 1);
            report("piece_all_src", 256U * MiB, 1, c.src_bytes * rounds, double(c.src_bytes * rounds) / gbps / 1e9);
        }
        CK(cudaFreeHost(big));
    }
    // persistent P MiB pinned buffers
    for (const std::size_t p : {64U * MiB, 256U * MiB}) {
        std::vector<std::uint8_t*> pool(c.src_bytes / p);
        std::vector<Range> r;
        for (auto& b : pool) {
            CK(cudaHostAlloc(reinterpret_cast<void**>(&b), p, cudaHostAllocDefault));
            std::memset(b, 1, p);
        }
        for (std::size_t i = 0; i < pool.size() && i * p < per_round; ++i) r.push_back({pool[i], p});
        const double gbps = copy_ranges(c, r, p, 1);
        report("pool", p, 1, per_round * rounds, double(per_round * rounds) / gbps / 1e9);
        for (auto b : pool) CK(cudaFreeHost(b));
    }
    // pageable source staged through 4 persistent pinned buffers by 3 host threads
    for (const std::size_t p : {64U * MiB, 256U * MiB}) {
        std::vector<std::uint8_t> src(per_round);
        std::memset(src.data(), 1, src.size());
        constexpr int kBufs = 4;
        std::uint8_t* buf[kBufs];
        cudaEvent_t done[kBufs];
        for (int i = 0; i < kBufs; ++i) {
            CK(cudaHostAlloc(reinterpret_cast<void**>(&buf[i]), p, cudaHostAllocDefault));
            CK(cudaEventCreateWithFlags(&done[i], cudaEventDisableTiming));
        }
        CK(cudaDeviceSynchronize());
        const double t0 = now_s();
        for (int round = 0; round < rounds; ++round) {
            std::size_t k = 0;
            for (std::size_t at = 0; at < per_round; at += p, ++k) {
                const std::size_t n = std::min(p, per_round - at);
                const int b = static_cast<int>(k % kBufs);
                if (k >= kBufs) CK(cudaEventSynchronize(done[b]));
                // 3 threads memcpy thirds of the piece
                std::thread th[3];
                for (int t = 0; t < 3; ++t)
                    th[t] = std::thread([&, t] {
                        const std::size_t a = n * t / 3, e = n * (t + 1) / 3;
                        std::memcpy(buf[b] + a, src.data() + at + a, e - a);
                    });
                for (auto& t : th) t.join();
                CK(cudaMemcpyAsync(c.dev[round % 2] + at, buf[b], n, cudaMemcpyHostToDevice, c.copy[0]));
                CK(cudaEventRecord(done[b], c.copy[0]));
            }
        }
        CK(cudaDeviceSynchronize());
        report("stage", p, 1, per_round * rounds, now_s() - t0);
        for (int i = 0; i < kBufs; ++i) {
            CK(cudaFreeHost(buf[i]));
            cudaEventDestroy(done[i]);
        }
    }
    for (auto d : c.dev) cudaFree(d);
    return 0;
}
