// Microbench of the host routed-expert executor (src/cpu/core/routed_expert.cpp): Qwen shape 2560/640, Q4_K gate/up +
// Q5_1 down = 3.072 MB per expert, experts spread over the whole buffer so DDR (not cache) is measured.
// Usage: cpu_expert_bench [--experts-gib G=2] [--threads 16] [--rows 1,2,4,8] [--misses 1,3,5,10] [--reps 20] [--tables DIR]
// misses = routed experts of each row that are computed on the host. Without --tables the exp tables are synthetic
// (same code path, cheaper table lookups than the real probe tables).
#if defined(_MSC_VER)
#define _CRT_SECURE_NO_WARNINGS 1
#endif
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <algorithm>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "cpu/core/routed_expert.hpp"

namespace ex = strix::cpu::expert;
using u8 = std::uint8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;

namespace {

std::vector<unsigned> parse_list(const char* s) {
    std::vector<unsigned> v;
    for (const char* p = s; *p;) {
        char* e = nullptr;
        v.push_back(static_cast<unsigned>(std::strtoul(p, &e, 10)));
        if (e == p) break;
        p = *e == ',' ? e + 1 : e;
    }
    return v;
}

double seconds() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void parallel_for(unsigned threads, const std::function<void(unsigned)>& fn) {
    std::vector<std::thread> ts;
    for (unsigned t = 1; t < threads; ++t) ts.emplace_back(fn, t);
    fn(0);
    for (auto& t : ts) t.join();
}

}  // namespace

int main(int argc, char** argv) {
    double gib = 2.0;
    std::vector<unsigned> threads = {16}, rows = {1, 2, 4, 8}, misses = {1, 3, 5, 10};
    unsigned reps = 20;
    std::string tables;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--experts-gib") gib = std::atof(argv[i + 1]);
        else if (a == "--threads") threads = parse_list(argv[i + 1]);
        else if (a == "--rows") rows = parse_list(argv[i + 1]);
        else if (a == "--misses") misses = parse_list(argv[i + 1]);
        else if (a == "--reps") reps = static_cast<unsigned>(std::atoi(argv[i + 1]));
        else if (a == "--tables") tables = argv[i + 1];
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    constexpr u32 hidden = 2560, ffn = 640;
    constexpr std::size_t gate_bytes = static_cast<std::size_t>(ffn) * (hidden / 256) * 144, down_bytes = static_cast<std::size_t>(hidden) * (ffn / 32) * 24;
    constexpr std::size_t expert_bytes = 2 * gate_bytes + down_bytes;
    const u32 experts = static_cast<u32>(gib * 1024.0 * 1024.0 * 1024.0 / static_cast<double>(expert_bytes));
    if (experts < 16) { std::fprintf(stderr, "experts-gib too small\n"); return 2; }

    std::string dir = tables;
    if (dir.empty()) {
        dir = (std::filesystem::temp_directory_path() / "strix_expert_bench_tables").string();
        std::filesystem::create_directories(dir);
        const u32 zero = 0;
        for (const char* n : {"/tbl_exp_pm.bin", "/hard_v_exp_f32.bin"}) {
            FILE* f = std::fopen((dir + n).c_str(), "wb");
            std::fwrite(&zero, 4, 1, f); std::fwrite(&zero, 4, 1, f);
            std::fclose(f);
        }
    }
    const ex::Gfx1151Exp exp(dir);

    std::unique_ptr<u8[]> gate(new u8[gate_bytes * experts]), up(new u8[gate_bytes * experts]), down(new u8[down_bytes * experts]);
    const unsigned fill_threads = std::max(1U, std::thread::hardware_concurrency());
    auto fill = [&](u8* base, std::size_t block, std::size_t total, bool has_min) {
        parallel_for(fill_threads, [&](unsigned t) {
            std::mt19937_64 g(1234 + t);
            const std::size_t nblocks = total / block, lo = nblocks * t / fill_threads, hi = nblocks * (t + 1) / fill_threads;
            for (std::size_t b = lo; b < hi; ++b) {
                u8* p = base + b * block;
                for (std::size_t i = 0; i + 8 <= block; i += 8) { const std::uint64_t x = g(); std::memcpy(p + i, &x, 8); }
                const u16 d = static_cast<u16>((8U + (g() % 5)) << 10 | (g() & 1023U)), m = static_cast<u16>((8U + (g() % 5)) << 10 | (g() & 1023U) | ((g() & 1) << 15));
                std::memcpy(p, &d, 2);
                if (has_min) std::memcpy(p + 2, &m, 2);
            }
        });
    };
    fill(gate.get(), 144, gate_bytes * experts, true);
    fill(up.get(), 144, gate_bytes * experts, true);
    fill(down.get(), 24, down_bytes * experts, true);

    ex::LayerWeights w;
    w.gate = gate.get(); w.up = up.get(); w.down = down.get();
    w.gate_format = ex::GateFormat::q4_k; w.down_format = ex::DownFormat::q5_1; w.shape = {hidden, ffn};

    constexpr unsigned kMaxRows = 16, kSlots = 10;
    struct Row { std::vector<std::int8_t> codes; std::vector<float> deltas; std::vector<std::int32_t> sums; };
    std::vector<Row> rowdata(kMaxRows);
    std::vector<ex::RowA8> rowviews(kMaxRows);
    std::mt19937_64 rng(99);
    for (unsigned r = 0; r < kMaxRows; ++r) {
        auto& d = rowdata[r];
        d.codes.resize(hidden); d.deltas.assign(hidden / 32, 0.004F); d.sums.assign(hidden / 32, 0);
        for (u32 i = 0; i < hidden; ++i) { const int c = static_cast<int>(rng() % 255) - 127; d.codes[i] = static_cast<std::int8_t>(c); d.sums[i / 32] += c; }
        rowviews[r] = {d.codes.data(), d.deltas.data(), d.sums.data()};
    }
    std::vector<float> raw(static_cast<std::size_t>(kMaxRows) * kSlots * hidden);
    std::vector<std::uint32_t> status(kMaxRows);

    std::printf("cpu_expert_bench: %u experts (%.2f GiB), %.3f MB/expert, hw threads %u\n", experts,
                static_cast<double>(experts) * static_cast<double>(expert_bytes) / 1073741824.0, static_cast<double>(expert_bytes) / 1e6,
                std::thread::hardware_concurrency());
    // DDR ceiling: plain multi-threaded read of the three buffers
    for (unsigned t : threads) {
        std::vector<std::uint64_t> sink(t);
        double best = 1e9;
        const std::size_t words = gate_bytes * experts / 8;
        for (int rep = 0; rep < 3; ++rep) {
            const double t0 = seconds();
            parallel_for(t, [&](unsigned k) {
                const auto* g = reinterpret_cast<const std::uint64_t*>(gate.get());
                std::uint64_t s = 0;
                for (std::size_t i = words * k / t; i < words * (k + 1) / t; ++i) s += g[i];
                sink[k] = s;
            });
            best = std::min(best, seconds() - t0);
        }
        std::printf("read-bw threads=%u: %.1f GB/s (uint64 sum over %.2f GB)\n", t, static_cast<double>(words * 8) / best / 1e9, static_cast<double>(words * 8) / 1e9);
    }
    for (unsigned t : threads) {
        ex::Executor E(t, exp);
        for (unsigned R : rows)
            for (unsigned M : misses) {
                if (R > kMaxRows || M > kSlots) continue;
                std::mt19937_64 pick(7 + R * 131 + M);
                double total = 0.0, uniq_sum = 0.0;
                for (unsigned rep = 0; rep < reps + 1; ++rep) {
                    std::vector<ex::Assignment> as;
                    std::set<u32> uniq;
                    for (unsigned r = 0; r < R; ++r) {
                        std::set<u32> mine;
                        for (unsigned s = 0; s < M; ++s) {
                            u32 e;
                            do { e = static_cast<u32>(pick() % experts); } while (mine.count(e));
                            mine.insert(e); uniq.insert(e);
                            as.push_back({r, s, e});
                        }
                    }
                    ex::Job j;
                    j.weights = w; j.rows = rowviews.data(); j.assignments = as.data(); j.count = static_cast<u32>(as.size());
                    j.rawdots = raw.data(); j.rawdot_row_stride = static_cast<std::uint64_t>(kSlots) * hidden; j.status = status.data();
                    const double t0 = seconds();
                    E.run(j);
                    const double dt = seconds() - t0;
                    if (rep > 0) { total += dt; uniq_sum += static_cast<double>(uniq.size()); }
                }
                const double job_us = total / reps * 1e6, uniq = uniq_sum / reps;
                std::printf("%s T=%u R=%u M=%u assign=%u uniq=%.1f: job %.0f us, %.0f us/expert, %.1f GB/s weights\n", E.isa(), t, R, M, R * M, uniq,
                            job_us, job_us / uniq, uniq * static_cast<double>(expert_bytes) / (job_us * 1e-6) / 1e9);
            }
    }
    return status[0] == 0xdeadbeefU ? 1 : 0;
}
