#include "strix/device/hip_gate.hpp"
#include "strix/hip_check.hpp"
#include "strix/dist/communicator.hpp"
#include "strix/dist/open_transport.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <memory>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// strix-dist-host-exchange-bench: a host-resident payload exchanged between the two ranks through the communicator,
// (a) the device path (what a device-forward exchange of host bytes does: host -> device control buffer, the
// communicator's exchange (device -> pinned -> wire -> pinned -> device), device -> host, stream sync) and (b)
// Communicator::exchange_host (pinned -> wire -> pinned, no device copy). Checks every delivery (each rank's bytes
// are a function of rank, size and iteration) and the mixed case (one rank on each path: the same wire messages).
//   strix-dist-host-exchange-bench --transport emulate | tcp-listen:PORT | tcp-connect:HOST:PORT | usb4auto [--rank R]
//       [--sizes 4096,65536,...] [--iters N] [--warmup N] [--pinned]
// Output: one line per (path, size): median / p10 / p90 microseconds per exchange, rank 0's view.
namespace {

struct Args {
    std::string transport = "emulate";
    std::uint32_t rank = 0;
    std::vector<std::size_t> sizes{4096, 16384, 65600, 131072, 172032, 262144, 344064, 524288, 1048576};
    int iters = 200, warmup = 20;
    bool pinned = false;  // host buffers pinned (a TAIL's receive buffer is) or pageable
};

using strix::hip_check;

void fill(std::byte* p, std::size_t n, std::uint32_t rank, std::size_t size, int iter) {
    std::uint64_t x = 0x9E3779B97F4A7C15ULL ^ (std::uint64_t{rank} << 56U) ^ (std::uint64_t{size} << 20U) ^ static_cast<std::uint64_t>(iter);
    for (std::size_t i = 0; i < n; ++i) {
        x ^= x << 13U;
        x ^= x >> 7U;
        x ^= x << 17U;
        p[i] = static_cast<std::byte>(x & 0xFFU);
    }
}

struct Rank {
    std::uint32_t rank;
    strix::dist::Communicator comm;
    hipStream_t stream = nullptr;
    std::uint8_t* control = nullptr;  // [send | receive] x 1 MB, as the device forward's control buffer
    std::byte *out = nullptr, *in = nullptr, *expect = nullptr;
    bool pinned;
    static constexpr std::size_t chunk = std::size_t{1} << 20U;
    Rank(std::uint32_t r, std::unique_ptr<strix::dist::Transport> t, bool pin)
        : rank(r), comm(strix::dist::Group{r, 2U, 7U, 0x68787362ULL}, std::move(t), chunk), pinned(pin) {
        hip_check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "stream");
        hip_check(hipMalloc(reinterpret_cast<void**>(&control), 2U * chunk), "control");
        if (pin) {
            hip_check(hipHostMalloc(reinterpret_cast<void**>(&out), chunk, hipHostMallocDefault), "out");
            hip_check(hipHostMalloc(reinterpret_cast<void**>(&in), chunk, hipHostMallocDefault), "in");
        } else {
            out = new std::byte[chunk];
            in = new std::byte[chunk];
        }
        expect = new std::byte[chunk];
    }
    ~Rank() {
        (void)hipFree(control);
        if (pinned) {
            (void)hipHostFree(out);
            (void)hipHostFree(in);
        } else {
            delete[] out;
            delete[] in;
        }
        delete[] expect;
        (void)hipStreamDestroy(stream);
    }
    // the device forward's exchange_host before this change, one chunk
    void device_path(std::size_t n, std::uint64_t at) {
        hip_check(hipMemcpyAsync(control, out, n, hipMemcpyHostToDevice, stream), "control upload");
        comm.exchange(stream, control, control + chunk, n, strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 0xC1U, at});
        hip_check(hipMemcpyAsync(in, control + chunk, n, hipMemcpyDeviceToHost, stream), "control download");
        hip_check(hipStreamSynchronize(stream), "control");
    }
    void host_path(std::size_t n, std::uint64_t at) {
        comm.exchange_host(std::span<const std::byte>(out, n), std::span<std::byte>(in, n),
                           strix::dist::Tag{static_cast<std::uint32_t>(strix::dist::TagKind::control), 0xC1U, at});
    }
};

struct Line {
    std::string path;
    std::size_t size;
    double med, p10, p90, stage, wire, check;
    std::size_t bad;
};

// path: 'd' device, 'h' host; the mixed arm runs rank 0 on `p0` and rank 1 on `p1`
std::vector<Line> run(Rank& r, const Args& a, const std::vector<std::pair<char, char>>& arms) {
    std::vector<Line> lines;
    for (const auto& [p0, p1] : arms) {
        const char mine = r.rank == 0U ? p0 : p1;
        for (const auto n : a.sizes) {
            std::vector<double> us;
            std::size_t bad = 0;
            const auto s0 = r.comm.stats();
            for (int it = -a.warmup; it < a.iters; ++it) {
                fill(r.out, n, r.rank, n, it);
                fill(r.expect, n, 1U - r.rank, n, it);
                std::memset(r.in, 0, n);
                const auto t0 = std::chrono::steady_clock::now();
                if (mine == 'd') r.device_path(n, static_cast<std::uint64_t>(it + a.warmup));
                else r.host_path(n, static_cast<std::uint64_t>(it + a.warmup));
                const auto t1 = std::chrono::steady_clock::now();
                if (std::memcmp(r.in, r.expect, n) != 0) ++bad;
                if (it >= 0) us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            const auto s1 = r.comm.stats();
            std::sort(us.begin(), us.end());
            const double k = 1e6 / static_cast<double>(a.iters + a.warmup);
            lines.push_back({std::string(1, p0) + std::string(1, p1), n, us[us.size() / 2U], us[us.size() / 10U], us[us.size() * 9U / 10U],
                             (s1.stage - s0.stage) * k, (s1.wire - s0.wire) * k, (s1.check - s0.check) * k, bad});
        }
    }
    return lines;
}

void report(const std::vector<Line>& lines, std::uint32_t rank) {
    for (const auto& l : lines)
        std::printf("rank %u path %s bytes %zu median_us %.1f p10_us %.1f p90_us %.1f mean_stage_us %.1f mean_wire_us %.1f mean_check_us %.1f bad %zu\n", rank,
                    l.path.c_str(), l.size, l.med, l.p10, l.p90, l.stage, l.wire, l.check, l.bad);
    std::size_t bad = 0;
    for (const auto& l : lines) bad += l.bad;
    std::printf("rank %u deliveries %s\n", rank, bad == 0U ? "all byte-identical" : "MISMATCH");
    std::fflush(stdout);
}

const std::vector<std::pair<char, char>> kArms{{'d', 'd'}, {'h', 'h'}, {'h', 'd'}, {'d', 'h'}, {'d', 'd'}, {'h', 'h'}};

int bench(const Args& a, std::uint32_t rank, std::unique_ptr<strix::dist::Transport> t) {
    Rank r(rank, std::move(t), a.pinned);
    const auto lines = run(r, a, kArms);
    report(lines, rank);
    for (const auto& l : lines)
        if (l.bad) return 1;
    return 0;
}

}  // namespace

int main(int argc, char** argv) try {
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared one and prints the "device:" receipt line. Without a declaration the single integrated
    // GPU of a node keeps working (declared=implicit).
    strix::device::bind_hip_main(argc, argv);
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        const auto next = [&] {
            if (i + 1 >= argc) throw std::invalid_argument(s + " needs a value");
            return std::string(argv[++i]);
        };
        if (s == "--transport") a.transport = next();
        else if (s == "--rank") a.rank = static_cast<std::uint32_t>(std::stoul(next()));
        else if (s == "--iters") a.iters = std::stoi(next());
        else if (s == "--warmup") a.warmup = std::stoi(next());
        else if (s == "--pinned") a.pinned = true;
        else if (s == "--sizes") {
            a.sizes.clear();
            std::stringstream ss(next());
            for (std::string t; std::getline(ss, t, ',');) a.sizes.push_back(std::stoull(t));
        } else throw std::invalid_argument("unknown option " + s);
    }
    for (const auto n : a.sizes)
        if (n == 0U || n > Rank::chunk) throw std::invalid_argument("--sizes: 1 .. 1 MB");
    if (a.iters < 10) throw std::invalid_argument("--iters >= 10");
    if (a.transport == "emulate") {
        auto [t0, t1] = strix::dist::memory_pair();
        int rc1 = 0;
        std::exception_ptr failure;
        std::thread peer([&, t = std::move(t1)]() mutable {
            try {
                rc1 = bench(a, 1U, std::move(t));
            } catch (...) {
                failure = std::current_exception();
            }
        });
        int rc0 = 0;
        try {
            rc0 = bench(a, 0U, std::move(t0));
        } catch (...) {
            peer.join();
            throw;
        }
        peer.join();
        if (failure) std::rethrow_exception(failure);
        return rc0 | rc1;
    }
    // The host exchange is what this bench measures: a GPU-direct exchange (usb4gpu) is negotiated but not used.
    auto t = strix::dist::open_pair(a.transport, a.rank, {std::chrono::seconds(60)}).host;
    std::printf("rank %u transport %s\n", a.rank, t->name().c_str());
    return bench(a, a.rank, std::move(t));
} catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 2;
}
