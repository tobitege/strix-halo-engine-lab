#include "core/weight_load_queue.hpp"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>
#ifdef _MSC_VER
#include <crtdbg.h>
#include <malloc.h>
#endif

#if defined(__GNUC__) && !defined(__clang__)
// The replaced operator new / delete below are a malloc / free pair; GCC reports the inlined pair as mismatched.
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
// Fault injection: while armed, allocation number `at` (and, sticky, every later one) fails, on any thread.
static std::atomic<long long> g_alloc_at{-1}, g_alloc_seen{0};
static std::atomic<bool> g_alloc_sticky{false}, g_alloc_hit{false};
void* operator new(std::size_t bytes) {
    const long long at = g_alloc_at.load();
    if (at >= 0) {
        const long long i = g_alloc_seen.fetch_add(1);
        if (i == at || (i > at && g_alloc_sticky.load())) {
            g_alloc_hit = true;
            throw std::bad_alloc();
        }
    }
    if (void* p = std::malloc(bytes != 0 ? bytes : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

namespace {
using namespace strix::core;

struct FailAllocation {
    FailAllocation(long long at, bool sticky) {
        g_alloc_seen = 0;
        g_alloc_hit = false;
        g_alloc_sticky = sticky;
        g_alloc_at = at;
    }
    ~FailAllocation() { g_alloc_at = -1; }
    [[nodiscard]] bool hit() const { return g_alloc_hit.load(); }
};

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template<class F> bool throws(F f) {
    try {
        f();
    } catch (const std::exception&) {
        return true;
    }
    return false;
}

// A file in memory; `align` > 1 rejects what an unbuffered descriptor would reject.
class MemorySource final : public WeightSource {
public:
    MemorySource(std::size_t bytes, std::uint64_t align, unsigned seed) : data_(bytes), align_(align) {
        std::mt19937_64 rng(seed);
        for (auto& b : data_) b = static_cast<std::uint8_t>(rng());
    }
    std::uint64_t alignment() const override { return align_; }
    std::uint64_t read(std::uint64_t offset, std::uint64_t bytes, std::uint8_t* dst) const override {
        if (fail_at != 0 && offset >= fail_at) throw std::runtime_error("injected read failure");
        if (align_ > 1 && ((offset | bytes | reinterpret_cast<std::uintptr_t>(dst)) & (align_ - 1)) != 0)
            throw std::runtime_error("unaligned unbuffered read");
        if (offset >= data_.size()) return 0;
        const std::uint64_t n = std::min<std::uint64_t>(bytes, data_.size() - offset);
        const std::uint64_t part = short_reads && n > 8192 ? (n / 2) & ~std::uint64_t{4095} : n;  // short, still aligned
        std::memcpy(dst, data_.data() + offset, static_cast<std::size_t>(part));
        return part;
    }
    const std::uint8_t* bytes() const { return data_.data(); }
    std::size_t size() const { return data_.size(); }
    std::uint64_t fail_at = 0;
    bool short_reads = false;

private:
    std::vector<std::uint8_t> data_;
    std::uint64_t align_;
};

// "Device" memory is host memory. A copy reads its staging only when it completes (`delay` after the submit), so a
// staging buffer recycled before its event corrupts the result.
class HostUploader final : public WeightUploader {
public:
    explicit HostUploader(std::chrono::milliseconds delay = {}) : delay_(delay) {}
    std::uint8_t* allocate(std::uint64_t bytes) override {
        void* p = nullptr;
#ifdef _MSC_VER
        p = _aligned_malloc(static_cast<std::size_t>(bytes), 4096);
#else
        if (posix_memalign(&p, 4096, static_cast<std::size_t>(bytes)) != 0) p = nullptr;
#endif
        if (p == nullptr) throw std::bad_alloc();
        const std::uint64_t now = live_bytes += bytes;
        std::uint64_t peak = peak_bytes.load();
        while (now > peak && !peak_bytes.compare_exchange_weak(peak, now)) {
        }
        return static_cast<std::uint8_t*>(p);
    }
    void release(std::uint8_t* staging, std::uint64_t bytes) noexcept override {
        live_bytes -= bytes;
        {
            std::lock_guard<std::mutex> lock(m_);
            for (const auto& [token, other] : pending_)
                if (other.staging == staging) recycled = true;  // freed while a copy still reads it
        }
#ifdef _MSC_VER
        _aligned_free(staging);
#else
        std::free(staging);
#endif
    }
    Token submit(const std::uint8_t* staging, const Copy* copies, std::size_t count) override {
        std::lock_guard<std::mutex> lock(m_);
        for (const auto& [token, other] : pending_)
            if (other.staging == staging) recycled = true;  // staging reused while a copy still reads it
        Pending p;
        p.staging = staging;
        p.copies.assign(copies, copies + count);
        p.due = std::chrono::steady_clock::now() + delay_;
        pending_.emplace(next_, std::move(p));
        ++open;
        return next_++;
    }
    bool done(Token token) override {
        std::lock_guard<std::mutex> lock(m_);
        if (fail_wait && pending_.count(token) != 0) return false;
        return finish(token, false);
    }
    void wait(Token token) override {
        std::lock_guard<std::mutex> lock(m_);
        if (fail_wait && pending_.count(token) != 0) throw std::runtime_error("injected upload wait failure");
        (void)finish(token, true);
    }
    void retire(Token token) noexcept override {
        std::lock_guard<std::mutex> lock(m_);
        if (pending_.count(token) != 0) late_retire = true;  // retired before its copy completed: a queue bug
        --open;
    }
    void drain() override {
        std::lock_guard<std::mutex> lock(m_);
        ++drains;
        while (!pending_.empty()) (void)finish(pending_.begin()->first, true);
    }
    std::atomic<std::uint64_t> live_bytes{0}, peak_bytes{0};
    std::uint64_t drains = 0;
    long long open = 0;  // tokens submitted and not retired: 0 once the queue is gone
    bool late_retire = false, recycled = false;
    std::atomic<bool> fail_wait{false};  // a copy never reports done and its wait throws (drain still completes it)
    [[nodiscard]] bool clean() const { return open == 0 && !late_retire && !recycled && live_bytes == 0; }

private:
    struct Pending {
        const std::uint8_t* staging = nullptr;
        std::vector<Copy> copies;
        std::chrono::steady_clock::time_point due;
    };
    bool finish(Token token, bool block) {
        const auto it = pending_.find(token);
        if (it == pending_.end()) return true;
        if (std::chrono::steady_clock::now() < it->second.due) {
            if (!block) return false;
            std::this_thread::sleep_until(it->second.due);
        }
        for (const auto& c : it->second.copies)
            std::memcpy(c.device, it->second.staging + c.staging_offset, static_cast<std::size_t>(c.bytes));
        pending_.erase(it);
        return true;
    }
    std::mutex m_;
    std::map<Token, Pending> pending_;
    Token next_ = 1;
    std::chrono::milliseconds delay_;
};

void identical_bytes(std::uint64_t align, bool short_reads, unsigned workers, std::uint64_t budget) {
    MemorySource a(std::size_t{3} << 20, align, 1), b((std::size_t{2} << 20) + 777, align, 2);
    a.short_reads = b.short_reads = short_reads;
    HostUploader up(std::chrono::milliseconds(1));
    std::mt19937_64 rng(align * 31 + workers);
    struct Want {
        std::vector<std::uint8_t> device, expect;
        std::uint64_t n1 = 0;
        WeightLoadQueue::Ticket ticket = 0;
    };
    std::vector<Want> wants(200);
    {
        WeightLoadQueue q(up, {workers, budget, nullptr, true});
        for (auto& w : wants) {
            WeightLoadManifest m;
            const MemorySource& s = rng() % 2 ? a : b;
            const std::uint64_t n1 = 1 + rng() % 200000, o1 = rng() % (s.size() - n1);
            const std::uint64_t n2 = 1 + rng() % 5000, o2 = s.size() - n2;  // the tail of the file: a short last window
            const std::uint64_t at1 = m.read(s, o1, n1), at2 = m.read(s, o2, n2);
            w.n1 = n1;
            w.expect.assign(s.bytes() + o1, s.bytes() + o1 + n1);
            w.expect.insert(w.expect.end(), s.bytes() + o2, s.bytes() + o2 + n2);
            w.device.resize(n1 + n2);
            if (rng() % 3 == 0) {  // a conversion: the first range reversed into scratch
                const std::uint64_t out = m.scratch(n1);
                m.convert([=](std::uint8_t* st) {
                    for (std::uint64_t i = 0; i < n1; ++i) st[out + i] = st[at1 + n1 - 1 - i];
                });
                std::reverse(w.expect.begin(), w.expect.begin() + static_cast<std::ptrdiff_t>(n1));
                m.upload(out, n1, w.device.data());
            } else {
                m.upload(at1, n1, w.device.data());
            }
            m.upload(at2, n2, w.device.data() + n1);
            w.ticket = q.submit(std::move(m));
        }
        q.wait(wants[7].ticket);  // the first consumer of one tensor: ready before publication
        require(wants[7].device == wants[7].expect, "a waited ticket is in device memory");
        q.publish();
        const auto st = q.stats();
        require(st.tickets == wants.size() && st.upload_events == wants.size(), "one ready event per ticket");
        require(st.peak_staging_bytes <= budget && up.peak_bytes <= budget, "resident staging stays inside the budget");
        for (auto& w : wants) {
            require(w.device == w.expect, "loaded bytes are identical");
            const std::uint64_t h = weight_hash64(w.expect.data() + w.n1, w.expect.size() - w.n1, weight_hash64(w.expect.data(), w.n1));
            require(q.hash(w.ticket) == h, "the ticket hash chains its upload ranges");
        }
    }
    require(up.drains >= 1 && !up.late_retire, "staging is retired only after its upload event");
    require(up.live_bytes == 0, "all staging returned");
}

void held_in_order() {
    MemorySource s(std::size_t{4} << 20, 4096, 3);
    HostUploader up;
    const std::uint64_t budget = std::uint64_t{2} << 20;  // two 1 MiB tickets resident under eight workers
    {
        WeightLoadQueue q(up, {8, budget, nullptr, true});
        std::vector<WeightLoadQueue::Ticket> tickets;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> where;
        for (unsigned i = 0; i < 64; ++i) {
            WeightLoadManifest m;
            const std::uint64_t off = (i * 60001U) % (3U << 20), n = 500000;
            const std::uint64_t at = m.read(s, off, n);
            m.hold();
            m.output(at, n);
            where.emplace_back(off, at);
            tickets.push_back(q.submit(std::move(m)));
        }
        for (unsigned i = 0; i < 64; ++i) {
            const std::uint8_t* st = q.take(tickets[i]);
            require(std::memcmp(st + where[i].second, s.bytes() + where[i].first, 500000) == 0, "held bytes are identical");
            require(q.hash(tickets[i]) == weight_hash64(s.bytes() + where[i].first, 500000), "held ticket hash");
            if (i != 63) q.release(tickets[i]);  // the last one is still held when the queue goes away
        }
        q.publish();
        require(q.stats().peak_staging_bytes <= budget, "held tickets stay inside the budget");
        require(throws([&] { q.release(tickets[0]); }), "a double release is rejected");
    }
    require(up.live_bytes == 0, "held staging returned at destruction");
}

void failure_cancels() {
    MemorySource s(std::size_t{1} << 20, 1, 4);
    s.fail_at = 600000;
    HostUploader up;
    std::vector<std::uint8_t> dev(1000);
    WeightLoadQueue q(up, {3, std::uint64_t{8} << 20, nullptr, false});
    std::vector<WeightLoadQueue::Ticket> t;
    for (unsigned i = 0; i < 40; ++i) {
        WeightLoadManifest m;
        m.upload(m.read(s, i == 5 ? 700000 : 1000 * i, 1000), 1000, dev.data());
        t.push_back(q.submit(std::move(m)));
    }
    require(throws([&] { q.wait(t[5]); }), "the failed ticket rethrows at its consumer");
    require(throws([&] { q.publish(); }), "publication rethrows the first failure");
    WeightLoadManifest big;
    (void)big.scratch(std::uint64_t{9} << 20);
    require(throws([&] { (void)q.submit(std::move(big)); }), "a ticket larger than the budget is rejected");
    WeightLoadManifest bad;
    require(throws([&] { bad.upload(0, 1, nullptr); }), "a null destination is rejected");
}

std::vector<WeightLoadQueue::Ticket> submit_rows(WeightLoadQueue& q, const MemorySource& s, std::uint8_t* dev, unsigned count,
                                                 std::uint64_t stride, std::uint64_t bytes) {
    std::vector<WeightLoadQueue::Ticket> t;
    for (unsigned i = 0; i < count; ++i) {
        WeightLoadManifest m;
        m.upload(m.read(s, i * stride, bytes), bytes, dev + i * stride);
        t.push_back(q.submit(std::move(m)));
    }
    return t;
}

// Audit P1: an allocation failure anywhere in a load with delayed copies (queue bookkeeping included) may fail the
// load, never recycle or leak staging that a copy still reads.
void allocation_failures() {
    const MemorySource s(128 * 1024, 1, 7);
    constexpr unsigned rows = 8;
    constexpr std::uint64_t row = 10000;
    for (int sticky = 0; sticky < 2; ++sticky) {
        for (long long at = 0;; ++at) {
            std::vector<std::uint8_t> dev(s.size());
            HostUploader up(std::chrono::milliseconds(1));
            bool hit = false, published = false;
            {
                FailAllocation arm(at, sticky != 0);
                try {
                    WeightLoadQueue q(up, {3, std::uint64_t{2} << 20, nullptr, false});
                    (void)submit_rows(q, s, dev.data(), rows, row, row);
                    q.publish();
                    published = true;
                } catch (const std::exception&) {
                }
                hit = arm.hit();
            }
            require(up.clean(), "an allocation failure never recycles, leaks or frees staging in flight");
            require(hit || published, "a load without an injected failure publishes");
            if (published) require(std::memcmp(dev.data(), s.bytes(), rows * row) == 0, "published bytes are the file's");
            if (!hit) break;
        }
    }
}

// Audit P2: a worker thread that does not start is an exception, not std::terminate.
void worker_start_failure() {
    HostUploader up;
    for (long long at = 0; at <= 4; ++at) {  // the worker list, then one allocation per worker thread
        FailAllocation arm(at, false);
        const bool threw = throws([&] { WeightLoadQueue q(up, {4, std::uint64_t{2} << 20, nullptr, false}); });
        require(threw && arm.hit(), "a worker that does not start is an exception, with the started ones joined");
    }
}

// Audit P1: a throwing upload wait while a worker waits for staging (one buffer, several workers) is a propagated
// failure: the worker handles it with the queue lock held.
void failed_upload_wait() {
    const MemorySource s(std::size_t{1} << 20, 1, 9);
    std::vector<std::uint8_t> dev(s.size());
    HostUploader up(std::chrono::milliseconds(1));
    up.fail_wait = true;
    {
        WeightLoadQueue q(up, {4, std::uint64_t{1} << 20, nullptr, false});
        const auto t = submit_rows(q, s, dev.data(), 16, 4096, 4096);
        require(throws([&] { q.publish(); }), "publication rethrows the failed upload wait");
        unsigned failed = 0;
        for (const auto x : t) failed += throws([&] { q.wait(x); }) ? 1U : 0U;
        require(failed == 15, "the ticket that waited and every later one fail; the one in flight completed");
        require(std::memcmp(dev.data(), s.bytes(), 4096) == 0, "the ticket in flight was drained");
    }
    require(up.clean(), "staging and tokens are returned after a failed wait");
}

// Audit P1: cleanup after a load error does not throw (it runs in destructors) and does not wait for consumers.
void cancel_in_cleanup() {
    MemorySource s(std::size_t{1} << 20, 1, 11);
    s.fail_at = 300000;
    std::vector<std::uint8_t> dev(s.size());
    HostUploader up(std::chrono::milliseconds(1));
    {
        WeightLoadQueue q(up, {3, std::uint64_t{2} << 20, nullptr, false});
        static_assert(noexcept(q.cancel()), "cancel is the destructor-safe exit");
        struct Cleanup {
            WeightLoadQueue& q;
            ~Cleanup() { q.cancel(); }
        };
        bool caught = false;
        try {
            const Cleanup cleanup{q};
            WeightLoadManifest held;
            held.output(held.read(s, 0, 1000), 1000);
            held.hold();
            const auto h = q.submit(std::move(held));
            (void)submit_rows(q, s, dev.data(), 20, 20000, 1000);  // rows 15.. are past fail_at
            (void)q.take(h);                                       // never released: the consumer is unwinding
            q.publish();
        } catch (const std::runtime_error&) {
            caught = true;
        }
        require(caught, "the read failure reaches the caller through the cleanup");
        require(throws([&] { q.publish(); }), "a cancelled queue does not publish");
        const auto late = submit_rows(q, s, dev.data(), 1, 0, 1000);
        require(throws([&] { q.wait(late[0]); }), "a ticket submitted after the cancel fails");
    }
    require(up.clean(), "a cancelled load returns its staging");
    HostUploader up2;
    {
        WeightLoadQueue q(up2, {2, std::uint64_t{1} << 20, nullptr, false});
        WeightLoadManifest held;
        held.output(held.read(s, 0, 1000), 1000);
        held.hold();
        const auto h = q.submit(std::move(held));
        (void)q.take(h);  // holds the whole budget: the next ticket waits for staging
        const auto t = submit_rows(q, s, dev.data(), 3, 1000, 1000);
        q.cancel();
        require(throws([&] { q.wait(t[0]); }) && throws([&] { q.wait(t[2]); }), "tickets waiting for staging are cancelled");
    }
    require(up2.clean(), "held staging goes back with the queue");
}

// Audit P2: ranges become tickets by their aligned staging size, so a small budget and tiny ranges both fit.
void small_budget_ranges() {
    const MemorySource s(std::size_t{3} << 20, 4096, 13);
    constexpr std::uint64_t mib = std::uint64_t{1} << 20;
    {
        HostUploader up;
        std::vector<std::uint8_t> dev(mib);
        {
            WeightLoadQueue q(up, {2, mib, nullptr, false});
            WeightRangeTickets r(q);
            r.load(s, 32, mib, dev.data());  // window 1 MiB + 4096: more than the only size class of this budget
            r.flush();
            q.publish();
            require(q.stats().tickets == 2, "the range is cut to fit the aligned staging");
        }
        require(std::memcmp(dev.data(), s.bytes() + 32, mib) == 0 && up.peak_bytes <= mib && up.clean(), "1 MiB budget load");
    }
    {
        HostUploader up;
        constexpr unsigned count = 4000;
        std::vector<std::uint8_t> dev(count * 16);
        {
            WeightLoadQueue q(up, {3, 2 * mib, nullptr, false});
            WeightRangeTickets r(q);
            for (unsigned i = 0; i < count; ++i) r.load(s, 100 + i * 600, 16, dev.data() + i * 16);  // a page each
            r.flush();
            q.publish();
            require(q.stats().tickets > 1, "tiny ranges are grouped by staging, not by payload");
        }
        bool same = true;
        for (unsigned i = 0; i < count; ++i) same = same && std::memcmp(dev.data() + i * 16, s.bytes() + 100 + i * 600, 16) == 0;
        require(same && up.peak_bytes <= 2 * mib && up.clean(), "tiny ranges inside a 2 MiB budget");
    }
    HostUploader up;
    WeightLoadQueue small(up, {1, mib / 2, nullptr, false});
    require(throws([&] { WeightRangeTickets r(small); }), "a budget below the smallest size class is refused before loading");
    WeightLoadQueue usual(up, {1, 512 * mib, nullptr, false});
    require(WeightRangeTickets(usual).piece_bytes() == ((512 * mib / 12) & ~std::uint64_t{4095}), "usual budgets keep their piece");
    require(WeightLoadQueue::max_ticket_staging_bytes(100 * mib) == 96 * mib &&
                WeightLoadQueue::staging_class_bytes(WeightLoadQueue::max_ticket_staging_bytes(100 * mib)) <= 100 * mib,
            "the largest ticket of a budget is a size class inside it");
}
}  // namespace

int main() {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    try {
        require(weight_hash64("abc", 3) != weight_hash64("abd", 3) && weight_hash64("abc", 3) == weight_hash64("abc", 3), "hash");
        identical_bytes(1, false, 1, std::uint64_t{4} << 20);
        identical_bytes(1, true, 4, std::uint64_t{2} << 20);
        identical_bytes(512, false, 4, std::uint64_t{3} << 20);
        identical_bytes(4096, true, 6, std::uint64_t{1} << 20);
        held_in_order();
        failure_cancels();
        allocation_failures();
        worker_start_failure();
        failed_upload_wait();
        cancel_in_cleanup();
        small_budget_ranges();
    } catch (const std::exception& e) {
        std::cerr << "weight load queue tests: " << e.what() << "\n";
        return 1;
    }
    std::cout << "weight load queue tests ok\n";
    return 0;
}
