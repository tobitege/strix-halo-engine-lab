#include <array>
#include <barrier>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string_view>
#include <thread>

namespace dist {
enum class TagKind : std::uint32_t { control = 4 };
struct Tag { std::uint32_t kind, layer, chunk; };
}
struct Wire {
    std::barrier<> ready{2};
    std::array<std::byte, 2> offers{};
    std::array<dist::Tag, 2> tags{};
    std::array<unsigned, 2> calls{};
};
// Only transport I/O is doubled. Decisions below are extracted verbatim from
// Communicator; startup policy is extracted from the actual BatchForward ctor.
struct Comm {
    using Tag = dist::Tag;
    Wire* wire = nullptr;
    unsigned rank = 0;
    std::size_t capacity = 4U << 20U, ready_capacity = 262144U;
    bool trace_ = false, direct_ = true, tail_direct = false, full_direct = false;
    struct Route { bool direct; };
    [[nodiscard]] Route route(std::size_t bytes) const {
        return {bytes <= 160000U ? tail_direct : full_direct};
    }
    [[nodiscard]] std::size_t max_bytes() const { return capacity; }
    [[nodiscard]] std::size_t staged_ready_capacity() const { return ready_capacity < capacity ? ready_capacity : capacity; }
    void exchange_host(std::span<const std::byte> local, std::span<std::byte> peer, const dist::Tag& tag) {
        wire->offers[rank] = local[0];
        wire->tags[rank] = tag;
        ++wire->calls[rank];
        wire->ready.arrive_and_wait();
        peer[0] = wire->offers[1U - rank];
        wire->ready.arrive_and_wait();
    }
#include "admission.inc"
};
namespace strix::envcfg {
thread_local std::array<bool, 3> values{true, true, true};
bool flag(std::string_view name, bool fallback) {
    if (name == "STRIX_DS41_ENGRAM_ASYNC") return values[0];
    if (name == "STRIX_DS41_BATCH_ENGRAM_ASYNC") return values[1];
    if (name == "STRIX_DS41_ENGRAM_FROM_LEAD") return values[2];
    return fallback;
}
}
struct Admission {
    std::array<bool, 3> flags{true, true, true};
    bool reader = true;
    std::size_t scratch = 245760U, capacity = 4U << 20U;
};
bool startup(Comm& comm, const Admission& a, std::size_t W, std::uint32_t cols) {
    struct Scratch { std::size_t n; std::size_t bytes() const { return n; } };
    struct Engram { std::uint32_t cols; std::uint32_t rows_per_token() const { return cols; } };
    struct Model { Engram engram; };
    struct Impl {
        bool engram_async = false, engram_from_lead = false;
        const Admission* engram_reader = nullptr;
        Model model;
        Scratch half_other;
        bool distributed() const { return true; }
    } m{false, false, a.reader ? &a : nullptr, {{cols}}, {a.scratch}};
    struct Options { Comm* communicator; } options{&comm};
    strix::envcfg::values = a.flags;
#include "startup.inc"
    return m.engram_from_lead;
}
bool pair(const Admission& a, const Admission& b, std::size_t rows, std::uint32_t cols, bool expected) {
    Wire wire;
    Comm c0, c1;
    c0.wire = c1.wire = &wire;
    c1.rank = 1U;
    c0.capacity = a.capacity;
    c1.capacity = b.capacity;
    bool peer = false;
    std::thread worker([&] { peer = startup(c1, b, rows, cols); });
    const bool lead = startup(c0, a, rows, cols);
    worker.join();
    return lead == expected && peer == expected && wire.calls[0] == 1U && wire.calls[1] == 1U &&
           wire.tags[0].kind == 4U && wire.tags[1].kind == 4U &&
           wire.tags[0].layer == 0x6567726dU && wire.tags[1].layer == 0x6567726dU;
}
bool pairs() {
    Admission normal;
    if (!pair(normal, normal, 8U, 120U, false)) return false; // audit: 253440 > 245760
    if (!pair(normal, normal, 8U, 24U, true)) return false;
    normal.scratch = 491520U;
    if (!pair(normal, normal, 16U, 24U, true)) return false;
    Admission exact = normal;
    exact.scratch = exact.capacity = 16U * 24U * 264U;
    if (!pair(exact, exact, 16U, 24U, true)) return false;
    for (unsigned side = 0; side < 2U; ++side) {
        for (unsigned reason = 0; reason < 6U; ++reason) {
            auto refused = exact;
            if (reason < 3U) refused.flags[reason] = false;
            else if (reason == 3U) refused.reader = false;
            else if (reason == 4U) --refused.scratch;
            else --refused.capacity;
            if (!pair(side ? exact : refused, side ? refused : exact, 16U, 24U, false)) return false;
        }
    }
    return true;
}
bool chunks() {
    Comm c;
    constexpr std::size_t row_bytes = 51200U;
    // 7 rows balance to 4+3; a small direct tail must decline the whole plan.
    if (c.staged_ready_chunk_rows(7U, row_bytes) != 4U) return false;
    c.tail_direct = true;
    if (c.staged_ready_chunk_rows(7U, row_bytes) != 0U) return false;
    // Full-sized chunks can still use host views under the same size router.
    if (c.staged_ready_chunk_rows(8U, row_bytes) != 4U) return false;
    c.tail_direct = false;
    c.full_direct = true;
    if (c.staged_ready_chunk_rows(7U, row_bytes) != 0U) return false;
    c.direct_ = false;
    if (c.staged_ready_chunk_rows(7U, row_bytes) != 4U) return false;
    c.trace_ = true;
    if (c.staged_ready_chunk_rows(7U, row_bytes) != 0U) return false;
    c.trace_ = false;
    c.capacity = row_bytes - 1U;
    if (c.staged_ready_chunk_rows(7U, row_bytes) != 0U) return false;
    c.capacity = 4U << 20U;
    for (std::uint32_t rows = 1; rows <= 16U; ++rows) {
        for (std::size_t cap_rows = 1; cap_rows <= 16U; ++cap_rows) {
            c.ready_capacity = cap_rows * row_bytes;
            const auto chunk = c.staged_ready_chunk_rows(rows, row_bytes);
            if (!chunk || chunk > rows || chunk > cap_rows) return false;
        }
    }
    c.ready_capacity = 0U;
    return c.staged_ready_chunk_rows(7U, row_bytes) == 0U &&
           c.staged_ready_chunk_rows(0U, row_bytes) == 0U && c.staged_ready_chunk_rows(7U, 0U) == 0U;
}
int main(int argc, char** argv) {
    const bool ok = argc == 2 && (std::string_view(argv[1]) == "pair" ? pairs() : chunks());
    if (!ok) std::fputs("PR583 admission regression\n", stderr);
    return ok ? 0 : 1;
}
