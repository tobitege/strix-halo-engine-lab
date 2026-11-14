// Checkpoint publication and reads draw from the process-wide byte budget: unlimited when no session decodes,
// held to the cap while one does, and available as a background host-lane job.
#include "core/host_lane.hpp"
#include "strix/session/checkpoint_io.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
namespace io = strix::session::checkpoint_io;
using Clock = std::chrono::steady_clock;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

std::vector<std::byte> pattern(std::size_t size, unsigned seed) {
    std::vector<std::byte> bytes(size);
    std::uint32_t x = seed;
    for (auto& b : bytes) {
        x = x * 1664525U + 1013904223U;
        b = static_cast<std::byte>(x >> 24U);
    }
    return bytes;
}

std::vector<std::byte> read_back(const std::filesystem::path& path) {
    std::vector<std::byte> out;
    io::read(path, 0, std::uint64_t{1} << 30U, [&](std::uint64_t size) {
        out.resize(static_cast<std::size_t>(size));
        return std::span<std::byte>(out);
    });
    return out;
}

void run(const std::filesystem::path& dir) {
    constexpr std::size_t mib = std::size_t{1} << 20U;
    auto& budget = strix::core::shared_byte_budget();
    require(budget.cap_bytes_per_second() > 31.9e6 && budget.cap_bytes_per_second() < 32.1e6, "the test needs STRIX_HOST_BYTE_CAP_GBPS=0.032");
    io::Options options;
    options.durable = false;

    // No decode: nothing is capped and nothing waits.
    const auto large = pattern(16U * mib, 1);
    auto begin = Clock::now();
    io::atomic_write(dir / "idle.bin", large, options);
    require(read_back(dir / "idle.bin") == large, "idle round trip");
    auto stats = budget.stats();
    require(stats.bytes_granted == 32U * mib && stats.bytes_capped == 0 && stats.cap_hits == 0, "idle I/O was governed");

    // Decode active: 4 MiB at 32 MB/s with a 0.64 MB burst owes about 0.11 s.
    const auto small = pattern(4U * mib, 2);
    {
        strix::core::ByteBudget::DecodeScope decode(budget);
        begin = Clock::now();
        io::atomic_write(dir / "decode.bin", small, options);
        const auto elapsed = std::chrono::duration<double>(Clock::now() - begin).count();
        require(elapsed >= 0.09, "a publication during decode exceeded the byte cap");
        stats = budget.stats();
        require(stats.bytes_capped == 4U * mib && stats.cap_hits >= 1 && stats.throttled_ns >= 90000000ULL, "cap accounting");

        // The same publication as a background job of the process-wide lane, still under the cap.
        const auto shared = std::make_shared<const std::vector<std::byte>>(pattern(2U * mib, 3));
        begin = Clock::now();
        io::atomic_write_background(dir / "background.bin", shared, options).get();
        require(std::chrono::duration<double>(Clock::now() - begin).count() >= 0.05, "background publication ignored the cap");
        require(strix::core::shared_host_lane().stats().background_jobs == 1, "publication did not run on the lane");
        require(&strix::core::shared_host_lane().budget() == &budget, "the lane does not share the process budget");
    }
    // The cap lingers between decode steps and lifts after the last one.
    require(budget.capped(), "cap lifted right after a decode step");
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    require(!budget.capped(), "cap held after decode ended");
    const auto capped_before = budget.stats().bytes_capped;
    begin = Clock::now();
    require(read_back(dir / "decode.bin") == small, "decode round trip");
    require(read_back(dir / "background.bin") == pattern(2U * mib, 3), "background round trip");
    require(budget.stats().bytes_capped == capped_before, "idle read was governed");

    bool refused = false;
    try {
        (void)io::atomic_write_background(dir / "none.bin", nullptr, options);
    } catch (const std::invalid_argument&) {
        refused = true;
    }
    require(refused, "a background publication without bytes was accepted");
}
} // namespace

int main() {
    // The process-wide budget reads its cap once, at first use.
#if defined(_WIN32)
    _putenv_s("STRIX_HOST_BYTE_CAP_GBPS", "0.032");
#else
    setenv("STRIX_HOST_BYTE_CAP_GBPS", "0.032", 1);
#endif
    std::error_code ignored;
    const auto dir = std::filesystem::temp_directory_path() /
                     ("strix-governed-io-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    int rc = 0;
    try {
        std::filesystem::create_directories(dir);
        run(dir);
        std::cout << "governed checkpoint io tests passed\n";
    } catch (const std::exception& error) {
        std::cerr << "governed checkpoint io tests failed: " << error.what() << '\n';
        rc = 1;
    }
    std::filesystem::remove_all(dir, ignored);
    return rc;
}
