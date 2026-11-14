#pragma once
// Rows of an on-disk table fetched with one deep queue of direct reads (shared: any engine whose row tables stay on
// the disk). The caller names byte ranges of an O_DIRECT descriptor; the fetcher sorts them by offset, reads each
// distinct page run once (rows that share or touch pages are coalesced into one read of at most slot_pages pages),
// keeps `depth` reads in flight on one io_uring ring driven by the calling thread (no worker threads, no context
// switch per row), and hands every row's bytes to the caller's sink. The bytes are the file's: a consumer gets the
// same rows as with one positional read per row. Linux only; make() returns null where the ring cannot be set up
// (other systems, io_uring disabled by policy or seccomp), and the caller keeps its own read path.
#include <algorithm>
#include <cstdint>
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

#if defined(__linux__) && __has_include(<linux/io_uring.h>)
#define STRIX_DIRECT_ROW_FETCH_URING 1
#include <atomic>
#include <cerrno>
#include <linux/io_uring.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

namespace strix::core {

struct RowFetch {
    std::uint64_t offset = 0;  // absolute byte offset of the row
    std::uint32_t bytes = 0;   // row length (at most (slot_pages - 1) pages)
};
struct RowFetchStats {
    std::uint64_t reads = 0, read_bytes = 0, rows = 0, shared = 0;  // shared: rows served by another row's read
};

class DirectRowFetcher {
public:
    static constexpr std::uint64_t page = 4096;
    static constexpr std::uint32_t slot_pages = 4;

    // null: no ring on this system. depth: reads in flight (a power of two is not required).
    static std::unique_ptr<DirectRowFetcher> make(int fd, unsigned depth) {
#ifdef STRIX_DIRECT_ROW_FETCH_URING
        if (fd < 0 || depth == 0U || depth > 4096U) return nullptr;
        std::unique_ptr<DirectRowFetcher> f(new DirectRowFetcher());
        if (!f->open(fd, depth)) return nullptr;
        return f;
#else
        (void)fd;
        (void)depth;
        return nullptr;
#endif
    }
    DirectRowFetcher(const DirectRowFetcher&) = delete;
    DirectRowFetcher& operator=(const DirectRowFetcher&) = delete;
    ~DirectRowFetcher() { close_ring(); }
    [[nodiscard]] bool usable() const noexcept { return ring_ready(); }

    // sink(i, p): row i's bytes at p (valid during the call only). One caller at a time. Throws on a read error, after
    // every read in flight has completed (the ring is reusable, including after a sink exception).
    // A fatal ring-entry error drains consumed submissions and disables the ring.
    template <class Sink> void fetch(std::span<const RowFetch> rows, Sink&& sink, RowFetchStats* stats = nullptr) {
#ifdef STRIX_DIRECT_ROW_FETCH_URING
        if (rows.empty()) return;
        if (!usable()) throw std::runtime_error("row fetch: ring unavailable");
        order_.resize(rows.size());
        std::iota(order_.begin(), order_.end(), std::uint32_t{0});
        std::sort(order_.begin(), order_.end(), [&](std::uint32_t a, std::uint32_t b) { return rows[a].offset < rows[b].offset; });
        const std::uint64_t slot_bytes = slot_pages * page;
        for (const auto& r : rows)
            if (r.bytes == 0U || r.bytes > slot_bytes - page) throw std::invalid_argument("row fetch: row size");
        std::size_t next = 0, in_flight = 0;
        bool failed = false;
        std::exception_ptr delivery_error;
        RowFetchStats s;
        const auto deliver = [&](const Group& g, const std::uint8_t* data) {
            for (std::size_t k = g.begin; k < g.end; ++k) {
                const RowFetch& r = rows[order_[k]];
                sink(order_[k], data + (r.offset - g.first));
            }
        };
        unsigned pending = 0;  // queued in the submission ring, not yet taken by the kernel
        while ((!failed && next < rows.size()) || in_flight != 0U) {
            while (!failed && next < rows.size() && !free_.empty()) {
                // one group: rows in offset order whose page runs overlap or touch, within one slot
                Group g;
                g.begin = next;
                g.first = rows[order_[next]].offset & ~(page - 1U);
                std::uint64_t last = (rows[order_[next]].offset + rows[order_[next]].bytes + page - 1U) & ~(page - 1U);
                for (++next; next < rows.size(); ++next) {
                    const RowFetch& r = rows[order_[next]];
                    const std::uint64_t r_last = (r.offset + r.bytes + page - 1U) & ~(page - 1U);
                    if ((r.offset & ~(page - 1U)) > last || r_last - g.first > slot_bytes) break;
                    last = std::max(last, r_last);
                }
                g.end = next;
                g.length = static_cast<std::uint32_t>(last - g.first);
                const unsigned slot = free_.back();
                free_.pop_back();
                groups_[slot] = g;
                const unsigned idx = tail_ & sq_mask_;
                io_uring_sqe& e = sqes_[idx];
                std::memset(&e, 0, sizeof e);
                e.opcode = IORING_OP_READ;
                e.fd = fd_;
                e.addr = reinterpret_cast<std::uint64_t>(buffers_ + std::size_t{slot} * slot_bytes);
                e.len = g.length;
                e.off = g.first;
                e.user_data = slot;
                sq_array_[idx] = idx;
                ++tail_;
                ++pending;
                ++in_flight;
                ++s.reads;
                s.read_bytes += g.length;
                s.shared += g.end - g.begin - 1U;
            }
            sq_tail_->store(tail_, std::memory_order_release);
            const long rc = ::syscall(SYS_io_uring_enter, ring_, pending, 1U, IORING_ENTER_GETEVENTS, nullptr, 0);
            if (rc < 0 && errno != EINTR && errno != EAGAIN && errno != EBUSY) {
                // No SQPOLL: only enter consumes submissions. The SQ head also accounts for partial consumption
                // on an error. CQ publication needs no further enter; retain slots until every consumed read ends.
                const int error = errno;
                in_flight -= tail_ - sq_head_->load(std::memory_order_acquire);
                while (in_flight != 0U) {
                    const unsigned head = cq_head_->load(std::memory_order_relaxed);
                    const unsigned end = cq_tail_->load(std::memory_order_acquire);
                    in_flight -= end - head;
                    cq_head_->store(end, std::memory_order_release);
                    if (in_flight != 0U) std::this_thread::yield();
                }
                close_ring();
                if (delivery_error) std::rethrow_exception(delivery_error);
                throw std::runtime_error("row fetch: io_uring_enter: " + std::string(std::strerror(error)));
            }
            pending = tail_ - sq_head_->load(std::memory_order_acquire);
            unsigned head = cq_head_->load(std::memory_order_relaxed);
            const unsigned ctail = cq_tail_->load(std::memory_order_acquire);
            for (; head != ctail; ++head) {
                const io_uring_cqe& c = cqes_[head & cq_mask_];
                const auto slot = static_cast<unsigned>(c.user_data);
                const Group g = groups_[slot];
                std::uint8_t* data = buffers_ + std::size_t{slot} * slot_bytes;
                --in_flight;
                bool ok = c.res == static_cast<std::int32_t>(g.length);
                if (!ok && !failed) {  // short read or error: the plain positional read decides
                    std::size_t got = 0;
                    while (got < g.length) {
                        const auto n = ::pread(fd_, data + got, g.length - got, static_cast<off_t>(g.first + got));
                        if (n < 0 && errno == EINTR) continue;
                        if (n <= 0) break;
                        got += static_cast<std::size_t>(n);
                    }
                    ok = got == g.length;
                }
                if (ok && !failed) {
                    try { deliver(g, data); }
                    catch (...) { delivery_error = std::current_exception(); failed = true; }
                }
                else failed = true;
                free_.push_back(slot);
            }
            cq_head_->store(head, std::memory_order_release);
            if (failed) {
                // Withdraw queued but unconsumed entries; no new I/O after a delivery/read failure.
                while (pending != 0U) {
                    --tail_;
                    free_.push_back(static_cast<unsigned>(sqes_[tail_ & sq_mask_].user_data));
                    --pending;
                    --in_flight;
                }
                sq_tail_->store(tail_, std::memory_order_release);
            }
        }
        s.rows = rows.size();
        if (delivery_error) std::rethrow_exception(delivery_error);
        if (failed) throw std::runtime_error("row fetch: direct read failed");
        if (stats) *stats = s;
#else
        (void)rows;
        (void)sink;
        (void)stats;
        throw std::logic_error("row fetch: no ring on this system");
#endif
    }

private:
    DirectRowFetcher() = default;
#ifdef STRIX_DIRECT_ROW_FETCH_URING
    bool ring_ready() const noexcept { return ring_ >= 0; }
    struct Group {
        std::size_t begin = 0, end = 0;  // order_ range
        std::uint64_t first = 0;         // page-aligned file offset
        std::uint32_t length = 0;        // whole pages
    };
    bool open(int fd, unsigned depth) {
        io_uring_params p{};
        ring_ = static_cast<int>(::syscall(SYS_io_uring_setup, depth, &p));
        if (ring_ < 0) return false;
        sq_bytes_ = p.sq_off.array + p.sq_entries * sizeof(unsigned);
        cq_bytes_ = p.cq_off.cqes + p.cq_entries * sizeof(io_uring_cqe);
        sqe_bytes_ = p.sq_entries * sizeof(io_uring_sqe);
        sq_ = map(sq_bytes_, IORING_OFF_SQ_RING);
        cq_ = map(cq_bytes_, IORING_OFF_CQ_RING);
        sqes_ = static_cast<io_uring_sqe*>(static_cast<void*>(map(sqe_bytes_, IORING_OFF_SQES)));
        // every read in flight owns a slot, and the completion ring holds at least as many entries as the submission
        // ring, so completions never overflow
        const unsigned slots = std::min(depth, p.sq_entries);
        buffers_ = static_cast<std::uint8_t*>(std::aligned_alloc(page, std::size_t{slots} * slot_pages * page));
        if (!sq_ || !cq_ || !sqes_ || !buffers_) {
            close_ring();
            return false;
        }
        sq_head_ = reinterpret_cast<std::atomic<unsigned>*>(sq_ + p.sq_off.head);
        sq_tail_ = reinterpret_cast<std::atomic<unsigned>*>(sq_ + p.sq_off.tail);
        sq_mask_ = *reinterpret_cast<unsigned*>(sq_ + p.sq_off.ring_mask);
        sq_array_ = reinterpret_cast<unsigned*>(sq_ + p.sq_off.array);
        cq_head_ = reinterpret_cast<std::atomic<unsigned>*>(cq_ + p.cq_off.head);
        cq_tail_ = reinterpret_cast<std::atomic<unsigned>*>(cq_ + p.cq_off.tail);
        cq_mask_ = *reinterpret_cast<unsigned*>(cq_ + p.cq_off.ring_mask);
        cqes_ = reinterpret_cast<io_uring_cqe*>(cq_ + p.cq_off.cqes);
        tail_ = sq_tail_->load(std::memory_order_relaxed);
        fd_ = fd;
        groups_.resize(slots);
        for (unsigned i = 0; i < slots; ++i) free_.push_back(i);
        return true;
    }
    std::uint8_t* map(std::size_t bytes, std::uint64_t offset) {
        void* p = ::mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, ring_, static_cast<off_t>(offset));
        return p == MAP_FAILED ? nullptr : static_cast<std::uint8_t*>(p);
    }
    void close_ring() noexcept {
        if (sq_) ::munmap(sq_, sq_bytes_);
        if (cq_) ::munmap(cq_, cq_bytes_);
        if (sqes_) ::munmap(sqes_, sqe_bytes_);
        if (ring_ >= 0) ::close(ring_);
        std::free(buffers_);
        sq_ = cq_ = nullptr;
        sqes_ = nullptr;
        buffers_ = nullptr;
        ring_ = -1;
    }
    int ring_ = -1, fd_ = -1;
    std::uint8_t *sq_ = nullptr, *cq_ = nullptr, *buffers_ = nullptr;
    io_uring_sqe* sqes_ = nullptr;
    io_uring_cqe* cqes_ = nullptr;
    std::size_t sq_bytes_ = 0, cq_bytes_ = 0, sqe_bytes_ = 0;
    std::atomic<unsigned>*sq_head_ = nullptr, *sq_tail_ = nullptr, *cq_head_ = nullptr, *cq_tail_ = nullptr;
    unsigned* sq_array_ = nullptr;
    unsigned sq_mask_ = 0, cq_mask_ = 0, tail_ = 0;
    std::vector<Group> groups_;
    std::vector<unsigned> free_;
    std::vector<std::uint32_t> order_;
#else
    bool ring_ready() const noexcept { return false; }
    void close_ring() noexcept {}
#endif
};

}  // namespace strix::core
