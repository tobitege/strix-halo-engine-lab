#pragma once
// One weight-load queue for every engine: file ranges -> per-ticket staging -> device, overlapped.
//
//   manifest (immutable)   file ranges to read, an optional CPU conversion, the staging ranges that leave the ticket
//   read workers (bounded) read + convert tickets in parallel; a ticket is admitted (given its staging buffer) in
//                          submit order, so a consumer that takes tickets in order can never be starved by later ones
//   staging                one buffer per ticket from a pool, bounded by a resident byte budget (an input); a buffer
//                          returns to the pool when the ticket's upload event completed. No shared bounce buffer: an
//                          unbuffered read lands directly in the ticket's staging at the file's alignment
//   upload                 through a WeightUploader (the HIP one owns an explicit non-blocking copy stream); every
//                          ticket gets its own ready event
//   waits                  none per tensor. wait(ticket) at the first consumer, or publish() once at end of load
//
// Host only (no GPU header): the HIP part is hip_weight_uploader.hpp, tests use a host uploader.
#include "byte_budget.hpp"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace strix::core {

// 64-bit content hash of loaded bytes (identity gates: old loader vs queue). Not cryptographic.
inline std::uint64_t weight_hash64(const void* data, std::uint64_t bytes, std::uint64_t seed = 0) {
    const auto* p = static_cast<const std::uint8_t*>(data);
    std::uint64_t h[4] = {seed ^ 0x9E3779B97F4A7C15ULL, seed ^ 0xC2B2AE3D27D4EB4FULL, seed ^ 0x165667B19E3779F9ULL,
                          seed ^ 0x27D4EB2F165667C5ULL};
    const auto mix = [](std::uint64_t a, std::uint64_t w) {
        a = (a ^ w) * 0xFF51AFD7ED558CCDULL;
        return (a << 29U) | (a >> 35U);
    };
    std::uint64_t i = 0;
    for (; i + 32 <= bytes; i += 32) {
        std::uint64_t w[4];
        std::memcpy(w, p + i, 32);
        for (int k = 0; k < 4; ++k) h[k] = mix(h[k], w[k]);
    }
    std::uint64_t tail[4] = {0, 0, 0, 0};
    std::memcpy(tail, p + i, static_cast<std::size_t>(bytes - i));
    for (int k = 0; k < 4; ++k) h[k] = mix(h[k], tail[k]);
    std::uint64_t r = bytes;
    for (int k = 0; k < 4; ++k) {
        r = (r ^ h[k]) * 0xC4CEB9FE1A85EC53ULL;
        r ^= r >> 33U;
    }
    return r;
}

// A file (or part of one) readable from several threads at once.
class WeightSource {
public:
    virtual ~WeightSource() = default;
    // 1 for buffered reads. A power of two > 1 for unbuffered reads: the queue then passes offsets, lengths and
    // destination addresses that are multiples of it.
    [[nodiscard]] virtual std::uint64_t alignment() const = 0;
    // Reads up to `bytes` at `offset` into `dst`; returns the bytes read, short only at the end of the file.
    virtual std::uint64_t read(std::uint64_t offset, std::uint64_t bytes, std::uint8_t* dst) const = 0;
};

// Staging memory and host-to-device copies. Every method may be called from any queue thread.
class WeightUploader {
public:
    using Token = std::uint64_t;
    struct Copy {
        void* device = nullptr;
        std::uint64_t staging_offset = 0, bytes = 0;
    };
    virtual ~WeightUploader() = default;
    // Staging for one ticket; the address is a multiple of 4096.
    virtual std::uint8_t* allocate(std::uint64_t bytes) = 0;
    virtual void release(std::uint8_t* staging, std::uint64_t bytes) noexcept = 0;
    // Starts the copies and returns the ticket's ready token. Does not wait.
    virtual Token submit(const std::uint8_t* staging, const Copy* copies, std::size_t count) = 0;
    [[nodiscard]] virtual bool done(Token token) = 0;  // non-blocking
    virtual void wait(Token token) = 0;
    virtual void retire(Token token) noexcept = 0;  // after done / wait; the token is not used again
    virtual void drain() = 0;                       // every copy submitted so far completed
};

// What one ticket reads, converts and hands over. Build it, then submit it: the queue never changes it.
class WeightLoadManifest {
public:
    static constexpr std::uint64_t page = 4096;
    // Bytes [offset, offset + bytes) of `source`; returns where they start in the ticket's staging.
    std::uint64_t read(const WeightSource& source, std::uint64_t offset, std::uint64_t bytes) {
        const std::uint64_t a = source.alignment();
        if (a == 0 || (a & (a - 1)) != 0 || a > page) throw std::invalid_argument("weight load: unsupported source alignment");
        const std::uint64_t first = offset & ~(a - 1), lead = offset - first;
        const std::uint64_t window = (lead + bytes + a - 1) & ~(a - 1);
        reads_.push_back({&source, first, window, size_, lead + bytes});
        const std::uint64_t at = size_ + lead;
        size_ = (size_ + window + page - 1) & ~(page - 1);
        return at;
    }
    // Staging the conversion writes into; returns its offset.
    std::uint64_t scratch(std::uint64_t bytes) {
        const std::uint64_t at = size_;
        size_ = (size_ + bytes + page - 1) & ~(page - 1);
        return at;
    }
    // CPU conversion on the read worker, after all reads of this ticket (order inside one tensor is the caller's).
    void convert(std::function<void(std::uint8_t* staging)> f) { convert_ = std::move(f); }
    // Staging range copied to device memory. The destination stays valid until the ticket is ready.
    void upload(std::uint64_t staging_offset, std::uint64_t bytes, void* device) {
        if (device == nullptr) throw std::invalid_argument("weight load: null upload destination");
        if (bytes != 0) copies_.push_back({device, staging_offset, bytes});
    }
    // The staging stays with the ticket until WeightLoadQueue::release (a consumer that needs the host bytes).
    void hold() { held_ = true; }
    // Staging range that is part of the ticket's content hash without being uploaded (held tickets).
    void output(std::uint64_t staging_offset, std::uint64_t bytes) { outputs_.push_back({staging_offset, bytes}); }
    [[nodiscard]] std::uint64_t staging_bytes() const { return std::max<std::uint64_t>(size_, page); }

private:
    friend class WeightLoadQueue;
    struct Read {
        const WeightSource* source;
        std::uint64_t file_offset, window, staging_offset, need;
    };
    std::vector<Read> reads_;
    std::vector<WeightUploader::Copy> copies_;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> outputs_;
    std::function<void(std::uint8_t*)> convert_;
    std::uint64_t size_ = 0;
    bool held_ = false;
};

struct WeightLoadOptions {
    unsigned workers = 4;                                         // read + convert threads
    std::uint64_t staging_budget_bytes = std::uint64_t{1} << 30;  // resident staging, free pool included
    ByteBudget* rate = nullptr;                                   // host bytes/s governor for the reads (may be null)
    bool hash = false;                                            // content hash per ticket (identity gates)
};

struct WeightLoadStats {
    std::uint64_t tickets = 0, read_bytes = 0, upload_bytes = 0, held_bytes = 0;
    std::uint64_t peak_staging_bytes = 0, staging_allocations = 0, upload_events = 0;
    double read_seconds = 0;     // summed over workers (reads overlap: can exceed the wall time)
    double convert_seconds = 0;  // summed over workers
    double admit_wait_seconds = 0;  // workers waiting for staging inside the budget
    double wall_seconds = 0;        // first submit to the last publish()
};

class WeightLoadQueue {
public:
    using Ticket = std::uint64_t;
    using Clock = std::chrono::steady_clock;

    WeightLoadQueue(WeightUploader& uploader, WeightLoadOptions options) : up_(uploader), opt_(options) {
        if (opt_.workers == 0) opt_.workers = 1;
        try {
            workers_.reserve(opt_.workers);
            for (unsigned i = 0; i < opt_.workers; ++i) workers_.emplace_back([this] { work(); });
        } catch (...) {  // a thread did not start: the destructor will not run, so the started ones are joined here
            {
                std::lock_guard<std::mutex> lock(m_);
                stop_ = true;
            }
            cv_.notify_all();
            for (auto& t : workers_) t.join();
            throw;
        }
    }
    ~WeightLoadQueue() {
        {
            std::lock_guard<std::mutex> lock(m_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
        try {
            up_.drain();
        } catch (...) {
        }
        for (auto& f : flying_) {
            up_.retire(f.token);
            up_.release(f.buffer.p, f.buffer.cap);
        }
        for (auto& r : recs_)
            if (r.state == State::held && r.buffer.p != nullptr) up_.release(r.buffer.p, r.buffer.cap);
        for (auto& b : free_) up_.release(b.p, b.cap);
    }
    WeightLoadQueue(const WeightLoadQueue&) = delete;
    WeightLoadQueue& operator=(const WeightLoadQueue&) = delete;

    Ticket submit(WeightLoadManifest manifest) {
        if (class_bytes(manifest.staging_bytes()) > opt_.staging_budget_bytes)
            throw std::invalid_argument("weight load: one ticket needs more staging than the budget; split it");
        if (manifest.held_ && !manifest.copies_.empty()) throw std::invalid_argument("weight load: a held ticket has no uploads");
        std::lock_guard<std::mutex> lock(m_);
        // Every ticket owns at most one staging buffer, which is in exactly one of free_ / flying_ / its record. The
        // room for it is taken here, where a failed allocation only fails this submit: the workers then move buffers
        // between the lists without allocating, so no failure can return staging that a copy still reads.
        room(free_, recs_.size() + 1);
        room(flying_, recs_.size() + 1);
        recs_.emplace_back();
        recs_.back().manifest = std::move(manifest);
        if (stats_.tickets == 0) t0_ = Clock::now();
        ++stats_.tickets;
        cv_.notify_all();
        return recs_.size() - 1;
    }
    // Abandons the load without throwing (destructors, failure paths): tickets not yet read fail, the workers go
    // idle and every submitted copy completed before this returns, so the destinations may be freed afterwards.
    // The queue stays cancelled: wait / take / publish of an unfinished ticket throw, held staging is still released
    // by release() or the destructor.
    void cancel() noexcept {
        std::unique_lock<std::mutex> lock(m_);
        cancel_ = true;
        cv_.notify_all();
        cv_.wait(lock, [&] { return next_ == recs_.size() && active_ == 0; });
        lock.unlock();
        try {
            up_.drain();
        } catch (...) {  // the uploader quiesces its stream before it reports a failure
        }
        lock.lock();
        reap(true);
    }
    // The staging the pool keeps for a ticket that needs `need` bytes (its size class).
    [[nodiscard]] static std::uint64_t staging_class_bytes(std::uint64_t need) { return class_bytes(need); }
    // The largest ticket staging a resident budget admits; 0 when the budget is below the smallest size class.
    [[nodiscard]] static std::uint64_t max_ticket_staging_bytes(std::uint64_t budget) {
        std::uint64_t step = std::uint64_t{1} << 20;
        while (step * 8 <= budget) step *= 2;
        return budget / step * step;
    }
    // Tensor-ready: the ticket's bytes are in device memory (a held ticket: read and converted).
    void wait(Ticket t) {
        std::unique_lock<std::mutex> lock(m_);
        Rec& r = rec(t);
        cv_.wait(lock, [&] { return r.state >= State::flying; });
        if (r.state == State::failed) failed(r);
        if (r.state != State::flying) return;
        const auto token = r.token;
        lock.unlock();
        up_.wait(token);
        lock.lock();
        reap(false);
    }
    // Non-blocking: the workers are finished with the ticket (read, converted, and its copies submitted or its
    // staging held), or it failed.
    [[nodiscard]] bool read_done(Ticket t) {
        std::lock_guard<std::mutex> lock(m_);
        return rec(t).state >= State::flying;
    }
    // Host bytes of a held ticket; valid until release(t). Take held tickets in submit order.
    const std::uint8_t* take(Ticket t) {
        std::unique_lock<std::mutex> lock(m_);
        Rec& r = rec(t);
        cv_.wait(lock, [&] { return r.state >= State::flying; });
        if (r.state == State::failed) failed(r);
        if (r.state != State::held) throw std::logic_error("weight load: take() of a ticket that is not held");
        return r.buffer.p;
    }
    void release(Ticket t) {
        std::lock_guard<std::mutex> lock(m_);
        Rec& r = rec(t);
        if (r.state != State::held) throw std::logic_error("weight load: release() of a ticket that is not held");
        free_.push_back(r.buffer);
        r.buffer = {};
        r.state = State::done;
        cv_.notify_all();
    }
    // End-of-load publication: every submitted ticket is read and its copies completed, with one upload wait.
    void publish() {
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [&] { return next_ == recs_.size() && active_ == 0; });
        lock.unlock();
        up_.drain();
        lock.lock();
        reap(true);
        if (stats_.tickets != 0) stats_.wall_seconds = std::chrono::duration<double>(Clock::now() - t0_).count();
        if (error_) std::rethrow_exception(error_);
        if (cancel_) throw std::runtime_error("weight load: the load was cancelled");
    }
    // With options.hash: the hash of the ticket's uploaded (or output) ranges, in manifest order.
    [[nodiscard]] std::uint64_t hash(Ticket t) {
        std::unique_lock<std::mutex> lock(m_);
        Rec& r = rec(t);
        cv_.wait(lock, [&] { return r.state >= State::flying; });
        if (r.state == State::failed) failed(r);
        return r.hash;
    }
    [[nodiscard]] WeightLoadStats stats() const {
        std::lock_guard<std::mutex> lock(m_);
        return stats_;
    }
    [[nodiscard]] const WeightLoadOptions& options() const { return opt_; }

private:
    enum class State { queued, reading, flying, held, done, failed };  // >= flying: the worker is finished with it
    struct Buffer {
        std::uint8_t* p = nullptr;
        std::uint64_t cap = 0;
    };
    struct Rec {
        WeightLoadManifest manifest;
        State state = State::queued;
        Buffer buffer;
        WeightUploader::Token token = 0;
        std::uint64_t hash = 0;
        std::exception_ptr error;
    };
    struct Flying {
        WeightUploader::Token token;
        Buffer buffer;
        Ticket ticket;
    };
    // Pool size classes: at most 25 % above the request, so differently sized tickets share buffers.
    static std::uint64_t class_bytes(std::uint64_t need) {
        std::uint64_t step = std::uint64_t{1} << 20;
        while (step * 8 <= need) step *= 2;
        return (need + step - 1) / step * step;
    }
    template<class V> static void room(V& v, std::size_t n) {
        if (v.capacity() < n) v.reserve(std::max(n, v.capacity() * 2));
    }
    [[noreturn]] static void failed(const Rec& r) {
        if (r.error) std::rethrow_exception(r.error);
        throw std::runtime_error("weight load: the load was cancelled");
    }
    Rec& rec(Ticket t) {
        if (t >= recs_.size()) throw std::out_of_range("weight load: unknown ticket");
        return recs_[t];
    }
    // Completed uploads return their staging to the pool (lock held).
    void reap(bool all) {
        for (std::size_t i = 0; i < flying_.size();) {
            if (all || up_.done(flying_[i].token)) {
                up_.retire(flying_[i].token);
                free_.push_back(flying_[i].buffer);
                recs_[flying_[i].ticket].state = State::done;
                flying_.erase(flying_.begin() + static_cast<std::ptrdiff_t>(i));
            } else {
                ++i;
            }
        }
        cv_.notify_all();
    }
    // Staging for the next ticket in order (lock held on entry and exit). Only one worker admits at a time.
    Buffer admit(std::unique_lock<std::mutex>& lock, std::uint64_t need) {
        const std::uint64_t cap = class_bytes(need);
        const auto t0 = Clock::now();
        bool waited = false;
        for (;;) {
            if (cancel_) throw std::runtime_error("weight load: the load was cancelled");
            std::size_t best = free_.size();
            for (std::size_t i = 0; i < free_.size(); ++i)
                if (free_[i].cap >= cap && free_[i].cap <= 2 * cap && (best == free_.size() || free_[i].cap < free_[best].cap)) best = i;
            if (best != free_.size()) {
                const Buffer b = free_[best];
                free_.erase(free_.begin() + static_cast<std::ptrdiff_t>(best));
                if (waited) stats_.admit_wait_seconds += std::chrono::duration<double>(Clock::now() - t0).count();
                return b;
            }
            if (resident_ + cap <= opt_.staging_budget_bytes) {
                resident_ += cap;
                stats_.peak_staging_bytes = std::max(stats_.peak_staging_bytes, resident_);
                ++stats_.staging_allocations;
                lock.unlock();
                Buffer b;
                try {
                    b = {up_.allocate(cap), cap};
                } catch (...) {
                    lock.lock();
                    resident_ -= cap;
                    throw;
                }
                lock.lock();
                if (waited) stats_.admit_wait_seconds += std::chrono::duration<double>(Clock::now() - t0).count();
                return b;
            }
            if (!free_.empty()) {  // a pooled buffer of another size class makes room
                const Buffer b = free_.back();
                free_.pop_back();
                resident_ -= b.cap;
                lock.unlock();
                up_.release(b.p, b.cap);
                lock.lock();
                continue;
            }
            waited = true;
            reap(false);
            if (!free_.empty()) continue;
            if (!flying_.empty()) {  // the oldest copy in flight frees staging next
                const auto token = flying_.front().token;
                lock.unlock();
                try {
                    up_.wait(token);
                } catch (...) {  // the caller handles the failure with the lock held, like every other exit
                    lock.lock();
                    throw;
                }
                lock.lock();
                reap(false);
                continue;
            }
            if (stop_) throw std::runtime_error("weight load: queue closed while a ticket waited for staging");
            cv_.wait(lock);  // other tickets are being read, or a held ticket waits for its consumer
        }
    }
    void run(Ticket ticket, Rec& r, Buffer b) {
        const WeightLoadManifest& m = r.manifest;
        double read_s = 0, convert_s = 0;
        std::uint64_t read_bytes = 0;
        auto t0 = Clock::now();
        for (const auto& rd : m.reads_) {
            std::uint8_t* dst = b.p + rd.staging_offset;
            const std::uint64_t a = rd.source->alignment();
            const auto piece = [&](std::uint64_t at, std::uint64_t n) {
                std::uint64_t got = 0;
                while (got < n) {
                    const std::uint64_t k = rd.source->read(rd.file_offset + at + got, n - got, dst + at + got);
                    if (k == 0) break;
                    if (a > 1 && (k & (a - 1)) != 0 && got + k < n) {  // an unaligned short read: only at the end of the file
                        got += k;
                        break;
                    }
                    got += k;
                }
                if (at + got < std::min(at + n, rd.need)) throw std::runtime_error("weight load: short read of the weights");
            };
            constexpr std::uint64_t slice = std::uint64_t{32} << 20;
            if (opt_.rate != nullptr)
                governed_slices(*opt_.rate, rd.window, slice, piece);
            else
                for (std::uint64_t at = 0; at < rd.window; at += slice) piece(at, std::min(slice, rd.window - at));
            read_bytes += rd.need;
        }
        auto t1 = Clock::now();
        read_s = std::chrono::duration<double>(t1 - t0).count();
        if (m.convert_) {
            m.convert_(b.p);
            convert_s = std::chrono::duration<double>(Clock::now() - t1).count();
        }
        std::uint64_t h = 0, up_bytes = 0, held_bytes = 0;
        for (const auto& c : m.copies_) {
            if (opt_.hash) h = weight_hash64(b.p + c.staging_offset, c.bytes, h);
            up_bytes += c.bytes;
        }
        for (const auto& [at, n] : m.outputs_) {
            if (opt_.hash) h = weight_hash64(b.p + at, n, h);
            held_bytes += n;
        }
        WeightUploader::Token token = 0;
        const bool fly = !m.held_ && !m.copies_.empty();
        if (fly) token = up_.submit(b.p, m.copies_.data(), m.copies_.size());
        // From here the copies may read the staging: nothing below can fail (submit() took the room in flying_ and
        // free_), so the buffer reaches the pool only through its ready event.
        std::lock_guard<std::mutex> lock(m_);
        stats_.read_seconds += read_s;
        stats_.convert_seconds += convert_s;
        stats_.read_bytes += read_bytes;
        stats_.upload_bytes += up_bytes;
        stats_.held_bytes += held_bytes;
        r.hash = h;
        if (fly) {
            ++stats_.upload_events;
            r.token = token;
            r.state = State::flying;
            flying_.push_back({token, b, ticket});
        } else if (m.held_) {
            r.buffer = b;
            r.state = State::held;
        } else {
            free_.push_back(b);
            r.state = State::done;
        }
    }
    void work() {
        std::unique_lock<std::mutex> lock(m_);
        for (;;) {
            cv_.wait(lock, [&] { return stop_ || (next_ < recs_.size() && !admitting_); });
            if (stop_) return;
            const Ticket t = next_++;
            Rec& r = recs_[t];  // std::deque: stable across later submits
            r.state = State::reading;
            ++active_;
            Buffer b;
            std::exception_ptr error = error_;  // after a failure the rest of the load is cancelled, not read
            bool failed_ticket = cancel_ || error != nullptr;
            if (!failed_ticket) {
                admitting_ = true;
                try {
                    b = admit(lock, r.manifest.staging_bytes());
                } catch (...) {
                    error = std::current_exception();
                    failed_ticket = true;
                }
                admitting_ = false;
                cv_.notify_all();
            }
            if (!failed_ticket) {
                lock.unlock();
                try {
                    run(t, r, b);
                } catch (...) {  // before the copies were submitted (see run): the staging is not referenced
                    error = std::current_exception();
                    failed_ticket = true;
                }
                lock.lock();
            }
            if (failed_ticket) {
                if (b.p != nullptr) free_.push_back(b);  // room taken by submit(): does not allocate
                r.error = error;
                r.state = State::failed;
                if (!error_ && error) error_ = error;
            }
            r.manifest = WeightLoadManifest{};  // the record keeps only its state
            --active_;
            cv_.notify_all();
        }
    }

    WeightUploader& up_;
    WeightLoadOptions opt_;
    mutable std::mutex m_;
    std::condition_variable cv_;
    std::deque<Rec> recs_;
    std::size_t next_ = 0, active_ = 0;
    bool admitting_ = false, stop_ = false, cancel_ = false;
    std::vector<Buffer> free_;
    std::vector<Flying> flying_;
    std::uint64_t resident_ = 0;
    std::exception_ptr error_;
    WeightLoadStats stats_;
    Clock::time_point t0_{};
    std::vector<std::thread> workers_;
};

// File byte ranges -> tickets: a long range is cut into pieces and small consecutive ranges share one ticket.
// Pieces and ticket boundaries follow the staging a ticket really takes (alignment slack and at least one page per
// read included), so every ticket fits the largest size class of the budget, whatever the budget and the offsets.
class WeightRangeTickets {
public:
    explicit WeightRangeTickets(WeightLoadQueue& queue) : queue_(queue) {
        const std::uint64_t budget = queue.options().staging_budget_bytes;
        const std::uint64_t most = WeightLoadQueue::max_ticket_staging_bytes(budget);
        if (most <= slack) throw std::invalid_argument("weight load: the staging budget is below 1 MiB");
        // One ticket's staging is its piece plus alignment slack and the pool's size class (25 %): eight tickets fit.
        const std::uint64_t wanted = std::clamp<std::uint64_t>(budget / 12, std::uint64_t{1} << 20, std::uint64_t{64} << 20);
        piece_ = std::min(wanted & ~(WeightLoadManifest::page - 1), most - slack);
        limit_ = std::min(most, 2 * piece_);
    }
    // Queues the copy of [offset, offset + bytes) of `source` to `device`; nothing waits.
    void load(const WeightSource& source, std::uint64_t offset, std::uint64_t bytes, void* device) {
        for (std::uint64_t done = 0; done < bytes;) {
            const std::uint64_t n = std::min(piece_, bytes - done);
            if (pending_ != 0 && (pending_ + n > piece_ || current_.staging_bytes() + n + slack > limit_)) flush();
            current_.upload(current_.read(source, offset + done, n), n, static_cast<std::uint8_t*>(device) + done);
            pending_ += n;
            done += n;
            if (pending_ >= piece_) flush();
        }
    }
    // Submits the ranges gathered so far as one ticket.
    void flush() {
        if (pending_ == 0) return;
        WeightLoadManifest m = std::move(current_);
        current_ = WeightLoadManifest{};
        pending_ = 0;
        (void)queue_.submit(std::move(m));
    }
    [[nodiscard]] std::uint64_t piece_bytes() const { return piece_; }

private:
    // One read takes at most its bytes + the lead and the tail of an aligned window + the round-up to a page.
    static constexpr std::uint64_t slack = 2 * WeightLoadManifest::page;
    WeightLoadQueue& queue_;
    WeightLoadManifest current_;
    std::uint64_t pending_ = 0, piece_ = 0, limit_ = 0;
};

}  // namespace strix::core
