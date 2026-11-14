#include "strix/ds41_engram.hpp"
#include "strix/ds41_formats.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#ifdef _WIN32
#include <process.h>
static int process_id() { return ::_getpid(); }
static int setenv(const char* name, const char* value, int) { return ::_putenv_s(name, value); }
static int unsetenv(const char* name) { return ::_putenv_s(name, ""); }
#else
#include <unistd.h>
static int process_id() { return ::getpid(); }
#endif

using namespace strix::ds41;
namespace {
void expect(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
struct Fixture {
    std::filesystem::path path = std::filesystem::current_path() /
        ("strix-engram-cache-" + std::to_string(process_id()) + ".bin");
    std::vector<std::uint8_t> bytes;
    Fixture() : bytes(524288) {
        std::mt19937 random(714);
        for (auto& b : bytes) b = static_cast<std::uint8_t>(random());
        std::ofstream file(path, std::ios::binary);
        file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    ~Fixture() { std::error_code ec; std::filesystem::remove(path, ec); }
};
bool run(const Fixture& fixture, bool direct, bool ring, unsigned depth, unsigned threads, std::uint64_t budget) {
    // Row 0 crosses a page, table 1 aliases table 0, table 2 has different bytes at the same row id.
    const std::vector<EngramTable> tables{{4000, 120}, {4000, 120}, {37000, 100}};
    EngramRowReader cached(fixture.path, tables, threads, direct, budget);
    EngramRowReader plain(fixture.path, tables, threads, direct, 0);
    std::fprintf(stderr, "cache case requested=%s depth=%u effective=%s/%s threads=%u budget=%llu\n",
                 ring ? "ring" : "workers", depth, cached.ring_io() ? "ring" : "workers",
                 cached.direct_io() ? "direct" : "buffered", threads, static_cast<unsigned long long>(budget));
    expect(cached.direct_io() == plain.direct_io() && cached.ring_io() == plain.ring_io(), "backend disagreement");
    if ((direct && !cached.direct_io()) || (ring && !cached.ring_io())) {
        std::puts("SKIP requested direct/ring backend unavailable");
        return false;
    }
    expect(cached.ring_io() == ring, "unexpected ring backend");
    expect(cached.direct_io() == direct, "unexpected direct backend");
    std::mt19937 random(561);
    for (unsigned batch = 0; batch < 60; ++batch) {
        std::vector<EngramRowRequest> requests;
        for (unsigned j = 0; j < 40; ++j) {
            const unsigned table = static_cast<unsigned>(random() % 3U);
            const auto row = batch < 5 ? 0U : random() % static_cast<unsigned>(tables[table].rows);
            requests.push_back({table, row});
            if (j % 3U == 0U) requests.push_back(requests.back());
        }
        std::vector<std::uint8_t> a(requests.size() * engram_row_bytes, 0x5a), b(a.size());
        cached.read(requests, a); plain.read(requests, b);
        expect(a == b, "cached rows differ from uncached rows");
        for (std::size_t i = 0; i < requests.size(); ++i) {
            const auto offset = tables[requests[i].table].file_offset + requests[i].row * engram_row_bytes;
            expect(std::equal(a.begin() + static_cast<std::ptrdiff_t>(i * engram_row_bytes),
                              a.begin() + static_cast<std::ptrdiff_t>((i + 1) * engram_row_bytes),
                              fixture.bytes.begin() + static_cast<std::ptrdiff_t>(offset)), "row differs from file bytes");
        }
    }
    const auto stats = cached.stats(), off = plain.stats();
    expect(stats.requests == stats.payload_hits + stats.payload_misses + stats.batch_duplicates, "cache accounting");
    expect(stats.page_reads == stats.payload_misses, "unique misses must match read count");
    expect(stats.cache_storage_bytes <= budget && stats.cache_capacity > 0, "cache budget");
    expect(stats.cache_entries <= stats.cache_capacity && stats.payload_hits > 0 && stats.batch_duplicates > 0, "cache exercised");
    if (budget < 1000) expect(stats.cache_evictions > 0, "CLOCK eviction exercised");
    expect(off.cache_capacity == 0 && off.page_reads == off.requests && off.batch_duplicates == 0, "OFF behavior");
    expect(stats.output_bytes == off.output_bytes && stats.payload_hit_bytes == stats.payload_hits * 264U, "byte accounting");
    std::vector<std::uint8_t> one(engram_row_bytes);
    for (const auto bad : {EngramRowRequest{3, 0}, EngramRowRequest{0, 120}}) {
        bool caught = false;
        try { cached.read(std::span(&bad, 1), one); } catch (const std::out_of_range&) { caught = true; }
        expect(caught, "invalid cached request must fail");
    }
    const EngramRowRequest good{0, 0};
    cached.read(std::span(&good, 1), one);
    expect(std::equal(one.begin(), one.end(), fixture.bytes.begin() + 4000), "recovery after invalid request");
    cached.read({}, {});
    return true;
}

bool recycle(const Fixture& fixture, unsigned depth) {
    EngramRowReader reader(fixture.path, {{0, fixture.bytes.size() / engram_row_bytes}}, 1, true, 0);
    if (!reader.direct_io() || !reader.ring_io()) {
        std::fprintf(stderr, "SKIP ring recycling depth=%u effective=%s/%s\n", depth,
                     reader.ring_io() ? "ring" : "workers", reader.direct_io() ? "direct" : "buffered");
        return false;
    }
    // Disjoint page runs: 20 reads exceed both tested depths, forcing slot recycling.
    std::vector<EngramRowRequest> requests;
    for (unsigned i = 0; i < 20U; ++i) requests.push_back({0, i * 80U});
    std::vector<std::uint8_t> out(requests.size() * engram_row_bytes);
    reader.read(requests, out);
    for (std::size_t i = 0; i < requests.size(); ++i)
        expect(std::equal(out.begin() + static_cast<std::ptrdiff_t>(i * engram_row_bytes),
                          out.begin() + static_cast<std::ptrdiff_t>((i + 1) * engram_row_bytes),
                          fixture.bytes.begin() + static_cast<std::ptrdiff_t>(requests[i].row * engram_row_bytes)),
               "ring recycling bytes");
    expect(reader.ring_io(), "ring recycling lost backend");
    std::fprintf(stderr, "PASS ring/direct recycling depth=%u reads=20\n", depth);
    return true;
}
} // namespace

int main(int argc, char** argv) try {
    const bool require_ring = argc == 2 && std::strcmp(argv[1], "--require-ring") == 0;
    expect(argc == 1 || require_ring, "usage: ds41-engram-cache-tests [--require-ring]");
    const Fixture fixture;
    bool ring_covered = true;
    // Explicit disabled and zero-depth fallback arms are separate from the required ring arms.
    for (const unsigned depth : {0U, 3U, 8U}) {
        const bool ring = depth != 0U;
        ::setenv("STRIX_DS41_ENGRAM_URING", ring ? "1" : "0", 1);
        ::setenv("STRIX_DS41_ENGRAM_URING_DEPTH", std::to_string(depth).c_str(), 1);
        for (const bool direct : {false, true}) {
            if (!direct && ring) continue;
            for (const unsigned threads : {1U, 8U, 64U})
                for (const std::uint64_t budget : {600ULL, 4096ULL, 1ULL << 20U}) {
                    const bool covered = run(fixture, direct, ring, depth, threads, budget);
                    if (ring) ring_covered = covered && ring_covered;
                }
        }
        if (ring) ring_covered = recycle(fixture, depth) && ring_covered;
    }
    ::setenv("STRIX_DS41_ENGRAM_URING", "1", 1);
    ::setenv("STRIX_DS41_ENGRAM_URING_DEPTH", "0", 1);
    (void)run(fixture, true, false, 0, 1, 4096);
    expect(!require_ring || ring_covered, "required ring/direct coverage unavailable");
    ::unsetenv("STRIX_DS41_ENGRAM_URING");
    ::unsetenv("STRIX_DS41_ENGRAM_URING_DEPTH");
    // Independent readers of two files must never share an offset's payload.
    EngramRowReader first(fixture.path, {{4000, 1}}, 2, false, 4096);
    std::vector<std::uint8_t> a(264), b(264);
    const EngramRowRequest row{0, 0};
    first.read(std::span(&row, 1), a);
    EngramRowReader second(fixture.path, {{4001, 1}}, 2, false, 4096);
    second.read(std::span(&row, 1), b);
    expect(a != b, "different absolute offsets must not alias");
    bool overflow = false;
    try { EngramRowReader bad(fixture.path, {{UINT64_MAX - 10, UINT64_MAX}}, 1, false, 0); }
    catch (const std::invalid_argument&) { overflow = true; }
    expect(overflow, "table overflow must be rejected");
    std::printf("Engram cache cases PASS; ring/direct coverage: %s\n", ring_covered ? "COMPLETE" : "SKIPPED (unavailable)");
    return 0;
} catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
