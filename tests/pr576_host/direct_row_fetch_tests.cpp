// Exercise the production ring loop on any host. Only OS entry points and allocation are substituted;
// on Linux use the real UAPI types, on Windows use field-compatible stand-ins (no native I/O claim).
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#ifdef __linux__
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#else
#include <malloc.h>
struct io_uring_sqe {
    unsigned opcode = 0;
    int fd = 0;
    std::uint64_t addr = 0, off = 0, user_data = 0;
    std::uint32_t len = 0;
};
struct io_uring_cqe { std::uint64_t user_data = 0; std::int32_t res = 0; };
struct io_uring_params {
    unsigned sq_entries = 0, cq_entries = 0;
    struct { unsigned head = 0, tail = 0, ring_mask = 0, array = 0; } sq_off;
    struct { unsigned head = 0, tail = 0, ring_mask = 0, cqes = 0; } cq_off;
};
constexpr int SYS_io_uring_setup = 0, SYS_io_uring_enter = 1, IORING_ENTER_GETEVENTS = 1, IORING_OP_READ = 22;
constexpr std::uint64_t IORING_OFF_SQ_RING = 0, IORING_OFF_CQ_RING = 1, IORING_OFF_SQES = 2;
constexpr int PROT_READ = 1, PROT_WRITE = 2, MAP_SHARED = 1, MAP_POPULATE = 2;
inline void* const MAP_FAILED = reinterpret_cast<void*>(static_cast<std::intptr_t>(-1));
#endif

namespace fake {
enum class Mode { normal, sink, partial_sink, denied, fatal_pending, fatal_consumed, read_error };
struct SQ { std::atomic<unsigned> head{0}, tail{0}; unsigned mask = 7; unsigned array[8]{}; } sq;
struct CQ { std::atomic<unsigned> head{0}, tail{0}; unsigned mask = 7; io_uring_cqe entries[8]{}; } cq;
io_uring_sqe sqes[8]{};
std::vector<io_uring_sqe> deferred;
std::thread producer;
std::atomic<unsigned> active{0};
unsigned allocations = 0, enters = 0, submitted = 0, closes = 0;
bool unsafe_release = false, deny_setup = false;
Mode mode = Mode::normal;

void expect(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void finish(const io_uring_sqe& e, bool error = false) {
    auto* data = reinterpret_cast<std::uint8_t*>(e.addr);
    for (unsigned i = 0; i < e.len; ++i) data[i] = static_cast<std::uint8_t>((e.off + i) % 251U);
    const auto tail = cq.tail.load();
    cq.entries[tail & cq.mask].user_data = e.user_data;
    cq.entries[tail & cq.mask].res = error ? -EIO : static_cast<std::int32_t>(e.len);
    --active;
    cq.tail.store(tail + 1U, std::memory_order_release);
}
void reset(Mode next) {
    if (producer.joinable()) producer.join();
    expect(allocations == 0 && active == 0 && !unsafe_release, "prior ring did not release safely");
    sq.head = 0; sq.tail = 0; cq.head = 0; cq.tail = 0;
    enters = submitted = closes = 0;
    deferred.clear(); mode = next; deny_setup = false;
}
long syscall(long, unsigned, io_uring_params* p) {
    if (deny_setup) { errno = EPERM; return -1; }
    p->sq_entries = p->cq_entries = 8;
    p->sq_off.head = offsetof(SQ, head); p->sq_off.tail = offsetof(SQ, tail);
    p->sq_off.ring_mask = offsetof(SQ, mask); p->sq_off.array = offsetof(SQ, array);
    p->cq_off.head = offsetof(CQ, head); p->cq_off.tail = offsetof(CQ, tail);
    p->cq_off.ring_mask = offsetof(CQ, mask); p->cq_off.cqes = offsetof(CQ, entries);
    return 42;
}
long syscall(long, int, unsigned pending, unsigned, unsigned, std::nullptr_t, int) {
    ++enters;
    if (mode == Mode::denied) { errno = EPERM; return -1; }
    if (mode == Mode::fatal_pending && enters == 2U) {
        producer = std::thread([] { std::this_thread::sleep_for(std::chrono::milliseconds(5)); finish(deferred.front()); });
        errno = EPERM; return -1;
    }
    for (const auto& e : deferred) finish(e);
    deferred.clear();
    const unsigned take = (mode == Mode::partial_sink || mode == Mode::fatal_pending || mode == Mode::fatal_consumed)
        ? std::min(1U, pending) : pending;
    for (unsigned i = 0; i < take; ++i) {
        const auto head = sq.head.load();
        const auto e = sqes[sq.array[head & sq.mask]];
        sq.head.store(head + 1U, std::memory_order_release);
        ++active; ++submitted;
        if (mode == Mode::fatal_pending || mode == Mode::fatal_consumed ||
            ((mode == Mode::sink || mode == Mode::read_error) && i != 0U && enters == 1U)) deferred.push_back(e);
        else finish(e, mode == Mode::read_error);
    }
    if (mode == Mode::fatal_consumed) {
        producer = std::thread([] { std::this_thread::sleep_for(std::chrono::milliseconds(5)); finish(deferred.front()); });
        errno = EPERM; return -1;
    }
    return static_cast<long>(take);
}
void* mmap(void*, std::size_t, int, int, int, off_t offset) {
    if (static_cast<std::uint64_t>(offset) == IORING_OFF_SQ_RING) return &sq;
    if (static_cast<std::uint64_t>(offset) == IORING_OFF_CQ_RING) return &cq;
    return sqes;
}
int munmap(void*, std::size_t) { return 0; }
int close(int) { unsafe_release |= active != 0U; ++closes; return 0; }
std::int64_t pread(int, void*, std::size_t, off_t) { errno = EIO; return -1; }
void* allocate(std::size_t alignment, std::size_t bytes) {
#ifdef _WIN32
    void* p = _aligned_malloc(bytes, alignment);
#else
    void* p = std::aligned_alloc(alignment, bytes);
#endif
    if (p) ++allocations;
    return p;
}
void release(void* p) {
    if (!p) return;
    unsafe_release |= active != 0U;
    --allocations;
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}
}  // namespace fake

// All standard/platform headers above are already guarded. Redirect only the production header's calls.
namespace test_std {
using namespace std;
inline void* aligned_alloc(std::size_t alignment, std::size_t bytes) { return fake::allocate(alignment, bytes); }
inline void free(void* p) { fake::release(p); }
}
#define STRIX_DIRECT_ROW_FETCH_URING 1
#define std test_std
#define syscall fake::syscall
#define mmap fake::mmap
#define munmap fake::munmap
#define close fake::close
#define pread fake::pread
#include "core/direct_row_fetch.hpp"
#undef pread
#undef close
#undef munmap
#undef mmap
#undef syscall
#undef std

using strix::core::DirectRowFetcher;
using strix::core::RowFetch;
using fake::expect;
const std::array<RowFetch, 5> rows{{{0, 32}, {32768, 32}, {65536, 32}, {98304, 32}, {131072, 32}}};
void check_row(std::uint32_t i, const std::uint8_t* p) {
    expect(i < rows.size(), "stale completion destination");
    for (unsigned j = 0; j < rows[i].bytes; ++j)
        expect(p[j] == static_cast<std::uint8_t>((rows[i].offset + j) % 251U), "wrong row bytes");
}
void throwing_sink(bool reuse, bool partial) {
    fake::reset(partial ? fake::Mode::partial_sink : fake::Mode::sink);
    auto ring = DirectRowFetcher::make(7, 2);
    expect(ring != nullptr, "ring setup");
    unsigned calls = 0;
    bool caught = false;
    try {
        ring->fetch(rows, [&](std::uint32_t i, const std::uint8_t* p) {
            check_row(i, p); ++calls;
            if (!partial) expect(fake::active == 1U, "throw must have outstanding I/O");
            if (calls == 1U) throw 73;
        });
    } catch (int value) { caught = value == 73; }
    expect(caught && calls == 1U, "preserve sink exception and suppress delivery");
    expect(fake::active == 0U && fake::cq.head == fake::cq.tail, "sink exception must drain completions");
    expect(fake::submitted == (partial ? 1U : 2U), "failure must stop submission");
    if (reuse) {
        fake::mode = fake::Mode::normal;
        ring->fetch(std::span(rows).first(1), check_row);
        ring->fetch(rows, check_row);
    }
    ring.reset();
    expect(fake::allocations == 0 && !fake::unsafe_release, "sink cleanup ownership");
}
void entry_failure(fake::Mode mode) {
    fake::reset(mode);
    auto ring = DirectRowFetcher::make(7, 2);
    expect(ring && fake::allocations == 1U, "entry allocation accounting");
    bool caught = false;
    try { ring->fetch(rows, check_row); } catch (const std::runtime_error&) { caught = true; }
    if (fake::producer.joinable()) fake::producer.join();
    expect(caught && !ring->usable(), "fatal entry disables ring");
    expect(fake::allocations == 0 && fake::active == 0 && !fake::unsafe_release, "entry error leaked or freed active storage");
    caught = false;
    try { ring->fetch(std::span(rows).first(1), check_row); } catch (const std::runtime_error&) { caught = true; }
    expect(caught, "disabled ring reuse must fail cleanly");
    ring.reset();
    expect(fake::closes == 1U && fake::allocations == 0, "entry destruction ownership");
}
int main(int argc, char** argv) try {
    expect(argc == 2, "choose sink, entry or lifecycle");
    const std::string test = argv[1];
    if (test == "sink") {
    throwing_sink(true, false);
    throwing_sink(false, false);
    throwing_sink(true, true);
    std::puts("PASS throwing_sink_outstanding_reuse_destroy_and_partial_submission");
    return 0;
    }
    if (test == "entry") {
    for (const auto mode : {fake::Mode::denied, fake::Mode::fatal_pending, fake::Mode::fatal_consumed}) entry_failure(mode);
    std::puts("PASS entry_failure_allocation_accounting_and_delayed_completion");
    return 0;
    }
    expect(test == "lifecycle", "unknown test case");
    fake::reset(fake::Mode::read_error);
    {
        auto ring = DirectRowFetcher::make(7, 2);
        bool caught = false;
        unsigned deliveries = 0;
        try { ring->fetch(rows, [&](auto, auto) { ++deliveries; }); }
        catch (const std::runtime_error&) { caught = true; }
        expect(caught && fake::active == 0 && deliveries == 0, "read failure must drain without delivery");
        fake::mode = fake::Mode::normal;
        ring->fetch(rows, check_row);
    }
    fake::reset(fake::Mode::normal);
    fake::deny_setup = true;
    expect(!DirectRowFetcher::make(7, 3) && fake::allocations == 0, "setup denial must permit fallback");
    fake::deny_setup = false;
    {
        auto ring = DirectRowFetcher::make(7, 3);
        for (unsigned n = 0; n < 4; ++n) ring->fetch(rows, check_row);
    }
    expect(fake::allocations == 0 && !fake::unsafe_release, "recycled ring ownership");
    std::puts("PASS read_error_reuse_setup_denial_and_depth3_recycling");
    return 0;
} catch (const std::exception& e) {
    if (fake::producer.joinable()) fake::producer.join();
    std::fprintf(stderr, "%s\n", e.what()); return 1;
}
