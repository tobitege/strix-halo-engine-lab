#include "strix/ds41_engram.hpp"
#include <array>
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace {
void expect(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void env(const char* key, const char* value) {
#ifdef _WIN32
    expect(_putenv_s(key, value) == 0, "set environment");
#else
    expect(::setenv(key, value, 1) == 0, "set environment");
#endif
}
struct Fixture {
    std::filesystem::path path = "engram-policy-fixture.bin";
    std::array<std::uint8_t, 65536> bytes{};
    Fixture() {
        for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<std::uint8_t>(i % 251U);
        std::ofstream out(path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        expect(out.good(), "write fixture");
    }
    ~Fixture() { std::error_code ec; std::filesystem::remove(path, ec); }
};
}
int main() try {
    using namespace strix::ds41;
    const Fixture fixture;
    env("STRIX_DS41_ENGRAM_URING", "1");
    for (const char* depth : {"3junk", "-1", "4097"}) {
        env("STRIX_DS41_ENGRAM_URING_DEPTH", depth);
        bool caught = false;
        try { EngramRowReader reader(fixture.path, {{0, 1}}, 1, true, 0); }
        catch (const std::invalid_argument& e) {
            caught = std::string(e.what()).find("STRIX_DS41_ENGRAM_URING_DEPTH") != std::string::npos;
        }
        expect(caught, "invalid ring depth accepted");
    }
    env("STRIX_DS41_ENGRAM_URING_DEPTH", "0");
    for (const bool direct : {false, true}) {
        EngramRowReader reader(fixture.path, {{0, 1}}, 1, direct, 0);
        expect(!reader.ring_io(), "zero depth must select workers");
        std::array<std::uint8_t, 264> out{};
        const EngramRowRequest request{0, 0};
        reader.read(std::span(&request, 1), out);
        expect(std::equal(out.begin(), out.end(), fixture.bytes.begin()), "fallback row bytes");
        std::printf("PASS zero-depth workers effective=%s\n", reader.direct_io() ? "direct" : "buffered");
    }
    std::puts("PASS strict_ring_depth_and_explicit_fallback");
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
