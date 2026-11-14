#include "strix/env_flags.hpp"
#include "strix/dist/gpu_direct.hpp"
#include "strix/hip_check.hpp"
#include "strix/dist/gpu_direct_copy.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <optional>
#include <sstream>
#include <vector>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <utility>

#include <pthread.h>
#include <sched.h>

namespace strix::dist {
namespace {

using transport::GpuQueueOperation;
using transport::LinuxGpuQueue;

void check(hipError_t status, const char* what) {
    ::strix::hip_check_prefixed("gpu-direct: ", status, what);
}
inline void relax() noexcept {
#if defined(__x86_64__)
    __builtin_ia32_pause();
#endif
}
// Spins until `event` is complete (the CPU witness of a copy).
void spin_event(hipEvent_t event, const char* what, std::chrono::steady_clock::time_point deadline) {
    while (true) {
        const auto ready = hipEventQuery(event);
        if (ready == hipSuccess) return;
        if (ready != hipErrorNotReady) check(ready, what);
        if (std::chrono::steady_clock::now() > deadline) throw std::runtime_error(std::string("gpu-direct: ") + what + " did not complete in time");
        relax();
    }
}

// How long an idle progress thread spins before it sleeps (STRIX_DIST_GPU_SPIN_US, default 300).
std::chrono::microseconds spin_window() {
    static const std::chrono::microseconds us{[] {
        const char* v = std::getenv("STRIX_DIST_GPU_SPIN_US");
        return v && *v ? std::stoll(v) : 300LL;
    }()};
    return us;
}
// TX completions off the critical path (STRIX_DIST_GPU_LAZY_TX, default 1): an exchange's job ends when every RX has
// landed; its TX completions are consumed by a later job, by start() before that TX slot is reused, or at drain.
// STRIX_DIST_GPU_BLOCK_WAIT=N bytes (default 65536; 0 off; 1 every size): before the next exchange, the caller's
// thread waits (hipEventSynchronize) for the previous exchange's gather instead of spinning on the progress thread,
// when the previous exchange was at most N bytes. On the hardware this made small-message decode faster (16.9 ->
// 20.2 tok/s with polled completion) but halved prefill when applied to its large chunks.
std::size_t block_wait_bytes() {
    static const std::size_t n = [] {
        const char* v = std::getenv("STRIX_DIST_GPU_BLOCK_WAIT");
        if (!v || !*v) return std::size_t{65536};
        if (v[0] == '0') return std::size_t{0};
        const auto x = std::stoull(v);
        return x == 1U ? SIZE_MAX : static_cast<std::size_t>(x);
    }();
    return n;
}
// Diagnostic (STRIX_DIST_GPU_RX_LEAD_US, default 0): after posting an exchange's RX, wait this long before its TX, so
// that the peer's RX is posted before this side's frames reach it.
long long rx_lead_us() {
    static const long long v = [] {
        const char* e = std::getenv("STRIX_DIST_GPU_RX_LEAD_US");
        return e && *e ? std::stoll(e) : 0LL;
    }();
    return v;
}

// The RX of the next exchange is submitted before this exchange's TX (STRIX_DIST_GPU_PREPOST, default 1; 0: off).
// The peer can only send the next exchange after receiving this one, so it always finds its receive descriptors
// posted. A TX that starts while the peer's GPU receive ring is empty can wait ~100-200 ms for E2E credits on these
// controllers (measured: 9-15 per 3000 back-to-back exchanges without, none with). The next exchange's geometry comes
// from the communicator's hint (expect_next).
bool prepost_rx() {
    static const bool on = strix::envcfg::flag("STRIX_DIST_GPU_PREPOST", true);
    return on;
}
bool lazy_tx() {
    static const bool on = strix::envcfg::flag("STRIX_DIST_GPU_LAZY_TX", true);
    return on;
}

// Experiment (STRIX_DIST_GPU_AWAKE_CPUS=1,4,5): one thread pinned to each listed CPU spins while exchanges are
// active (within 20 ms of the last start), so the cores that service the controller's interrupts stay out of deep
// idle states (wake-up latency up to hundreds of microseconds). Idle otherwise (1 ms sleeps).
class KeepAwake {
public:
    KeepAwake() {
        const char* v = std::getenv("STRIX_DIST_GPU_AWAKE_CPUS");
        if (!v || !*v) return;
        std::stringstream in(v);
        for (std::string cpu; std::getline(in, cpu, ',');) {
            const int c = std::stoi(cpu);
            threads_.emplace_back([this, c] {
                cpu_set_t set;
                CPU_ZERO(&set);
                CPU_SET(c, &set);
                (void)pthread_setaffinity_np(pthread_self(), sizeof set, &set);
                while (!stop_.load(std::memory_order_relaxed)) {
                    const auto last = last_.load(std::memory_order_relaxed);
                    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
                    if (now - last < 20'000'000) relax();
                    else std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            });
        }
    }
    ~KeepAwake() {
        stop_ = true;
        for (auto& t : threads_) t.join();
    }
    void touch() noexcept { last_.store(std::chrono::steady_clock::now().time_since_epoch().count(), std::memory_order_relaxed); }

private:
    std::atomic<bool> stop_{false};
    std::atomic<std::int64_t> last_{0};
    std::vector<std::thread> threads_;
};

// One job at a time on a persistent thread; the caller waits for completion by spinning.
class Progress {
public:
    Progress() : thread_([this] { run(); }) {}
    ~Progress() {
        { std::lock_guard lock(mutex_); stop_ = true; }
        wake_.notify_all();
        thread_.join();
    }
    void post(std::function<void()> job) {
        wait_quiet();
        { std::lock_guard lock(mutex_); job_ = std::move(job); ++posted_; }
        wake_.notify_one();
    }
    // Waits for the posted job; rethrows its exception once.
    void wait() {
        while (finished_.load(std::memory_order_acquire) != posted_) relax();
        if (error_) { auto e = error_; error_ = nullptr; std::rethrow_exception(e); }
    }

private:
    void wait_quiet() { while (finished_.load(std::memory_order_acquire) != posted_) relax(); }
    void run() {
        std::uint64_t seen = 0;
        while (true) {
            const auto spin_end = std::chrono::steady_clock::now() + spin_window();
            std::function<void()> job;
            while (true) {
                {
                    std::unique_lock lock(mutex_, std::try_to_lock);
                    if (lock.owns_lock()) {
                        if (stop_) return;
                        if (posted_ != seen) { job = std::move(job_); break; }
                    }
                }
                if (std::chrono::steady_clock::now() > spin_end) {
                    std::unique_lock lock(mutex_);
                    wake_.wait(lock, [&] { return stop_ || posted_ != seen; });
                    if (stop_) return;
                    job = std::move(job_);
                    break;
                }
                relax();
            }
            ++seen;
            try { job(); } catch (...) { error_ = std::current_exception(); }
            finished_.store(seen, std::memory_order_release);
        }
    }
    std::mutex mutex_;
    std::condition_variable wake_;
    std::function<void()> job_;
    std::uint64_t posted_ = 0;
    std::atomic<std::uint64_t> finished_{0};
    std::exception_ptr error_;
    bool stop_ = false;
    std::thread thread_;
};

// The completion flag: the last published sequence (64 bits: it never wraps into the failure value), or the
// failure value, which is latched (no later publication overwrites it).
constexpr std::uint64_t failed_flag = ~std::uint64_t{0};

// STRIX_DIST_GPU_DEBUG=1: one line per step of every exchange on stderr.
bool debug() {
    static const bool on = strix::envcfg::flag("STRIX_DIST_GPU_DEBUG", false);
    return on;
}
void trace(const char* what, std::uint64_t sequence, std::size_t extra = 0) {
    if (debug()) std::fprintf(stderr, "gpu-direct seq %llu: %s %zu\n", static_cast<unsigned long long>(sequence), what, extra);
}

class GpuDirectExchange final : public GpuDirectLinks {
public:
    GpuDirectExchange(std::vector<GpuDirectLink> links, GpuDirectOptions options) : options_(options) {
        if (links.empty() || links.size() > 8U) throw std::invalid_argument("gpu-direct: one to eight streams");
        if (!options_.epoch || !options_.pool_identity) throw std::invalid_argument("gpu-direct: the epoch and pool identity must be peer-agreed and nonzero");
        for (auto& l : links) {
            Link link;
            link.name = l.name;
            link.tx = static_cast<std::byte*>(l.arena.tx);
            link.rx = static_cast<std::byte*>(l.arena.rx);
            link.queue = std::make_unique<LinuxGpuQueue>(std::move(l.io), l.expected);
            link.queue->import_pool(std::move(l.arena.pool), options_.epoch, options_.pool_identity);
            link.queue->arm();
            links_.push_back(std::move(link));
        }
        depth_ = links_[0].queue->depth();
        capacity_ = links_[0].queue->slot_capacity();
        for (const auto& link : links_)
            if (link.queue->depth() != depth_ || link.queue->slot_capacity() != capacity_)
                throw std::invalid_argument("gpu-direct: every stream needs the same depth and slot geometry");
        terminal_ = links_[0].queue->required_terminal_bytes();
        if (!terminal_) terminal_ = 64U; // application terminal record size; unused by the kernel-header mode
        for (auto& link : links_) {
            link.tx_ops.resize(depth_);
            link.rx_ops.resize(depth_);
            link.tx_free.reset(new std::atomic<bool>[depth_]);
            for (std::uint32_t s = 0; s < depth_; ++s) link.tx_free[s].store(true);
        }
        send_ready_.resize(depth_);
        recv_done_.resize(depth_);
        recv_pending_.assign(depth_, false);
        for (std::uint32_t s = 0; s < depth_; ++s) {
            check(hipEventCreateWithFlags(&send_ready_[s], hipEventDisableTiming), "event");
            check(hipEventCreateWithFlags(&recv_done_[s], hipEventDisableTiming), "event");
        }
        check(hipHostMalloc(reinterpret_cast<void**>(&error_), sizeof(GpuDirectError), hipHostMallocMapped | hipHostMallocCoherent), "error record");
        std::memset(error_, 0, sizeof(GpuDirectError));
        check(hipHostGetDevicePointer(reinterpret_cast<void**>(&error_device_), error_, 0), "error record");
        check(hipHostMalloc(reinterpret_cast<void**>(&produced_), 64, hipHostMallocMapped | hipHostMallocCoherent), "producer record");
        __atomic_store_n(produced_, std::uint64_t{0}, __ATOMIC_RELEASE);
        check(hipHostGetDevicePointer(reinterpret_cast<void**>(&produced_device_), produced_, 0), "producer record");
        check(hipMalloc(reinterpret_cast<void**>(&scratch_), 2 * sizeof(GpuDirectScratch)), "checksum scratch");
        check(hipMemset(scratch_, 0, 2 * sizeof(GpuDirectScratch)), "checksum scratch");
        if (const char* t = std::getenv("STRIX_DIST_GPU_TIMING"); t && t[0] == '1') {
            timing_ = true;
            for (auto* e : {&t_scatter0_, &t_scatter1_, &t_gather0_, &t_gather1_}) check(hipEventCreate(e), "timing event");
        }
        // Signal memory (fine-grained, written by the CPU, polled by the stream's wait packet).
        check(hipExtMallocWithFlags(reinterpret_cast<void**>(&flag_), 8, hipMallocSignalMemory), "flag");
        __atomic_store_n(flag_, std::uint64_t{0}, __ATOMIC_RELEASE);
        flag_device_ = flag_;
    }
    ~GpuDirectExchange() override {
        // Drain unconditionally: an exchange whose finish() was called may still have its TX in flight.
        try { progress_.wait(); } catch (...) {}
        if (trace_ && trace_pending_) { trace_pending_ = false; write_trace(); }
        // TX operations still in flight (lazy TX completion) finish before the arenas go away.
        if (!poisoned_) try { drain_tx(std::nullopt, std::chrono::steady_clock::now() + options_.timeout); } catch (...) {}
        // Every stream that copied into or out of the arenas has finished when its events have.
        for (std::uint32_t s = 0; s < depth_; ++s) {
            (void)hipEventSynchronize(send_ready_[s]);
            (void)hipEventSynchronize(recv_done_[s]);
        }
        for (auto& link : links_) (void)link.queue->release();
        for (std::uint32_t s = 0; s < depth_; ++s) {
            (void)hipEventDestroy(send_ready_[s]);
            (void)hipEventDestroy(recv_done_[s]);
        }
        if (boot_stream_) (void)hipStreamDestroy(boot_stream_);
        (void)hipFree(flag_);
        (void)hipFree(scratch_);
        (void)hipHostFree(error_);
        (void)hipHostFree(produced_);
        if (dummy_local_) (void)hipFree(dummy_local_);
        if (dummy_remote_) (void)hipFree(dummy_remote_);
        if (timing_) for (auto e : {t_scatter0_, t_scatter1_, t_gather0_, t_gather1_}) (void)hipEventDestroy(e);
    }
    [[nodiscard]] std::size_t capacity() const noexcept override { return capacity_ * links_.size() - gpu_direct_header_bytes; }
    [[nodiscard]] std::string name() const override {
        std::string out = "usb4-gpu-direct:";
        for (std::size_t k = 0; k < links_.size(); ++k) out += (k ? "," : "") + links_[k].name;
        return out + " (depth " + std::to_string(depth_) + ", " + std::to_string(capacity_ >> 10U) + " KiB per slot)";
    }
    [[nodiscard]] DeviceExchangeStats stats() const override { return stats_; }

    [[nodiscard]] std::string failure() const override {
        const auto code = __atomic_load_n(&error_->code, __ATOMIC_ACQUIRE);
        if (code) {
            return "exchange " + std::to_string(error_->sequence) + " (tag " + std::to_string(error_->tag) + ", layer " + std::to_string(error_->layer) +
                   ") was refused by the receiving GPU: " + gpu_direct_error_text(code) +
                   (code >= 6U && code <= 7U ? " (peer: tag " + std::to_string(error_->peer_tag) + ", layer " + std::to_string(error_->peer_layer) + ")" : "") +
                   host_side();
        }
        if (auto h = host_side(); !h.empty()) return h.substr(2);
        return poisoned_ ? poison_ : std::string();
    }
    [[nodiscard]] std::string host_side() const {
        std::lock_guard lock(failure_mutex_);
        return host_failure_.empty() ? std::string() : "; host side: " + host_failure_;
    }
    void check_failure() {
        if (const auto f = failure(); !f.empty()) {
            poisoned_ = true;
            if (poison_.empty()) poison_ = f;
            throw std::runtime_error("gpu-direct: " + f);
        }
    }
    void set_trace(std::FILE* trace) override {
        // The trace derives the consumer's time from the GPU's own timeline: the producer and consumer events
        // become timing events (no exchange has run yet).
        for (std::uint32_t s = 0; s < depth_; ++s) {
            check(hipEventSynchronize(send_ready_[s]), "event");
            check(hipEventSynchronize(recv_done_[s]), "event");
            check(hipEventDestroy(send_ready_[s]), "event");
            check(hipEventDestroy(recv_done_[s]), "event");
            check(hipEventCreate(&send_ready_[s]), "trace event");
            check(hipEventCreate(&recv_done_[s]), "trace event");
        }
        trace_ = trace;
    }
    // Rings, wire and payload parts of an exchange of `bytes` (the rule of start()).
    struct Geometry {
        std::size_t bytes = 0, used = 0;
        std::array<std::pair<std::size_t, std::size_t>, 8> parts{}, wire{};
        [[nodiscard]] bool same(const Geometry& o) const {
            return bytes == o.bytes && used == o.used && std::equal(wire.begin(), wire.begin() + used, o.wire.begin());
        }
    };
    Geometry geometry(std::size_t bytes, unsigned rings) const {
        Geometry g;
        g.bytes = bytes;
        // The wire message is the payload (zero-padded to whole 4-byte words) followed by the 64-byte header; it is
        // split over the streams at 64-byte boundaries, and the header travels at the end of the last part.
        const std::size_t padded = (bytes + 3U) / 4U * 4U, wire = padded + gpu_direct_header_bytes;
        // A requested ring count (the router's) is raised to the fewest rings whose slots hold the message.
        const std::size_t fit = (wire + capacity_ - 1U) / capacity_;
        std::size_t used = rings ? std::clamp<std::size_t>(std::max<std::size_t>(rings, fit), 1U, links_.size())
                                 : std::clamp<std::size_t>(wire / options_.min_part, 1U, links_.size());
        // The header (and at least one payload byte) must lie wholly in the last part.
        while (used > 1U && part_stride(wire, used) * (used - 1U) + gpu_direct_header_bytes + 4U > wire) --used;
        g.used = used;
        for (std::size_t j = 0; j < used; ++j) {
            g.wire[j] = part(wire, j, used);
            g.parts[j] = g.wire[j];
        }
        for (std::size_t j = 0; j < used; ++j)
            if (g.wire[j].second == 0U || g.wire[j].second > capacity_)
                throw std::logic_error("gpu-direct: part " + std::to_string(j) + " of " + std::to_string(wire) + " wire bytes does not fit a slot");
        g.parts[used - 1U].second -= gpu_direct_header_bytes + (padded - bytes); // payload bytes of the last part
        return g;
    }
    void expect_next(std::size_t bytes, unsigned rings) override { hint_bytes_ = bytes; hint_rings_ = rings; }
    bool use_direction(ExchangeDirection direction) override { next_direction_ = direction; return true; }
    void quiesce(hipStream_t stream) override {
        check_failure();
        if (started_) throw std::logic_error("gpu-direct: quiesce() while an exchange is in flight");
        collect();
        if (!prepost_.valid || prepost_.index != next_) return;
        dummy(stream, nullptr);
    }
    // An internal exchange with the early receive's geometry (scratch buffers, a reserved control identity); `next`:
    // the geometry whose receive it posts before its own TX (the real exchange after a wrong guess), or none.
    void dummy(hipStream_t stream, const Geometry* next) {
        if (!dummy_local_) {
            check(hipMalloc(&dummy_local_, capacity()), "gpu-direct scratch");
            check(hipMalloc(&dummy_remote_, capacity()), "gpu-direct scratch");
            check(hipMemsetAsync(dummy_local_, 0, capacity(), stream), "gpu-direct scratch");
        }
        const Geometry guessed = prepost_.geometry;
        begin(stream, dummy_local_, guessed, ExchangeIdentity{0U, 0U, 4U, 0xFFFFFFFFU}, next, trace_ ? trace_wall_ns() : 0U);
        finish(stream, dummy_remote_);
        collect();
    }
    void start(hipStream_t stream, const void* local, std::size_t bytes, const ExchangeIdentity& identity) override {
        const std::uint64_t entry = trace_ ? trace_wall_ns() : 0U;
        check_failure();
        awake_.touch();
        if (started_) throw std::logic_error("gpu-direct: start() while an exchange is in flight");
        if (bytes == 0U || bytes > capacity())
            throw std::invalid_argument("gpu-direct: exchange of " + std::to_string(bytes) + " bytes (capacity " + std::to_string(capacity()) + ")");
        const unsigned rings = next_rings_;
        next_rings_ = 0;
        const ExchangeDirection direction = next_direction_;
        next_direction_ = ExchangeDirection::both;
        const bool one_way = direction != ExchangeDirection::both;
        if (direction != ExchangeDirection::receive && !local) throw std::invalid_argument("gpu-direct: start() needs the send buffer");
        // A one-way transfer posts no early receive: the receiver submits its RX when it starts and the sender's TX
        // waits for it on the rings' flow control, so a wrong guess can never cost a scratch exchange here.
        const std::size_t hint_bytes = one_way ? 0U : hint_bytes_;
        const unsigned hint_rings = hint_rings_;
        hint_bytes_ = 0;
        // The previous exchange's job (its RX and TX completions) has finished before this one is posted.
        collect();
        const Geometry g = geometry(bytes, rings);
        if (prepost_.valid && prepost_.index == next_ && (one_way || !prepost_.geometry.same(g))) {
            // The guess was wrong: both ranks made the same guess and see the same exchange, so both first complete an
            // internal exchange with the guessed geometry (scratch buffers), which posts this exchange's RX early
            // (a symmetric one's; a one-way transfer never has an early receive).
            ++stats_.prepost_misses;
            dummy(stream, one_way ? nullptr : &g);
        }
        std::optional<Geometry> next;
        if (hint_bytes && hint_bytes <= capacity()) next = geometry(hint_bytes, hint_rings);
        begin(stream, local, g, identity, next ? &*next : nullptr, entry, direction);
    }
    // Posts the job of exchange next_ with geometry `g`; `next`: the following exchange's (its RX goes before this TX).
    void begin(hipStream_t stream, const void* local, const Geometry& g, const ExchangeIdentity& identity, const Geometry* next,
               std::uint64_t entry, ExchangeDirection direction = ExchangeDirection::both) {
        const std::uint64_t k = next_++;
        const std::uint32_t slot = static_cast<std::uint32_t>(k % depth_);
        const std::size_t used = g.used, bytes = g.bytes;
        current_ = Exchange{k, k + 1U, slot, used, bytes, g.parts, g.wire, {}};
        current_.identity = GpuDirectIdentity{k + 1U, identity.plan, identity.request, bytes, identity.tag, identity.layer};
        current_.tx = direction != ExchangeDirection::receive;
        current_.rx = direction != ExchangeDirection::send;
        if (prepost_rx() && depth_ >= 2U && next) {
            current_.has_next = true;
            current_.next = *next;
        }
        if (trace_) { times_ = TraceTimes{}; times_.entry = entry; }
        if (prepost_.valid && prepost_.index == k) {
            if (!prepost_.geometry.same(g)) throw std::logic_error("gpu-direct: a preposted RX does not match its exchange");
            current_.ring_sequence = prepost_.ring_sequence;
            current_.preposted = true;
            prepost_.valid = false;
            ++stats_.preposted;
        } else {
            for (std::size_t j = 0; j < used; ++j) current_.ring_sequence[j] = links_[j].next_sequence++;
        }
        const auto deadline = std::chrono::steady_clock::now() + options_.timeout;
        // The previous job has finished, so the queues are idle and this thread may use them. The TX slot was
        // released at its TX completion (the controller read it). The RX slot's previous operation is retired
        // once the CPU has seen its consumer copy complete (it ran `depth` exchanges ago).
        try {
            for (auto& link : links_) {
                if (!link.tx_free[slot].load(std::memory_order_acquire)) drain_tx(slot, deadline); // its TX left `depth` exchanges ago
                if (current_.preposted) continue; // the slot's previous RX was retired when this one was preposted
                if (auto& old = link.rx_ops[slot]) {
                    spin_event(recv_done_[slot], "the consumer copy of an RX slot", deadline);
                    link.queue->gpu_consumed(*old);
                    link.queue->retire(*old);
                    old.reset();
                }
            }
        } catch (const std::exception& e) {
            if (!poisoned_) { poisoned_ = true; poison_ = e.what(); }
            throw;
        }
        if (current_.tx) {
            GpuDirectParts parts;
            parts.count = static_cast<unsigned>(used);
            for (std::size_t j = 0; j < used; ++j) {
                parts.slot[j] = links_[j].tx + std::size_t{slot} * capacity_;
                parts.offset[j] = current_.parts[j].first;
                parts.bytes[j] = current_.parts[j].second;
            }
            if (timing_) check(hipEventRecord(t_scatter0_, stream), "timing event");
            enqueue_scatter(parts, local, current_.identity, scratch_, produced_device_, stream);
            if (timing_) check(hipEventRecord(t_scatter1_, stream), "timing event");
            check(hipEventRecord(send_ready_[slot], stream), "producer event");
            trace("start: scatter enqueued, streams", current_.sequence, used);
        }
        started_ = true;
        const Exchange x = current_;
        progress_.post([this, x] {
            try { transfer(x); } catch (const std::exception& e) {
                abort_links(e.what());
                throw;
            } catch (...) {
                abort_links("unknown error");
                throw;
            }
        });
    }
    void finish(hipStream_t stream, void* remote) override {
        check_failure();
        if (!started_) throw std::logic_error("gpu-direct: finish() without start()");
        started_ = false;
        const auto& x = current_;
        ++stats_.exchanges;
        stats_.bytes += x.bytes;
        if (x.tx) stats_.bytes_tx += x.bytes;
        if (x.rx) stats_.bytes_rx += x.bytes;
        if (!x.tx || !x.rx) ++stats_.one_way;
        pending_job_ = true;
        // One-way send: nothing lands here. The receiver's GPU checks the header; this rank's TX completion is
        // consumed by the progress job or by the next use of the slot.
        if (!x.rx) return;
        if (!remote) throw std::invalid_argument("gpu-direct: finish() needs the receive buffer");
        trace("finish: wait + gather enqueued", x.sequence);
        // The stream waits until the progress thread has seen every RX part land (the flag is monotonic).
        check(hipStreamWaitValue64(stream, flag_device_, x.sequence, hipStreamWaitValueGte, ~std::uint64_t{0}),
              "wait for the received sequence");
        GpuDirectParts parts;
        parts.count = static_cast<unsigned>(x.used);
        for (std::size_t j = 0; j < x.used; ++j) {
            parts.slot[j] = links_[j].rx + std::size_t{x.slot} * capacity_;
            parts.offset[j] = x.parts[j].first;
            parts.bytes[j] = x.parts[j].second;
        }
        if (timing_) check(hipEventRecord(t_gather0_, stream), "timing event");
        enqueue_gather(parts, remote, x.identity, scratch_ + 1, flag_device_, error_device_, stream);
        if (timing_) { check(hipEventRecord(t_gather1_, stream), "timing event"); timed_pending_ = x.tx; }
        check(hipEventRecord(recv_done_[x.slot], stream), "consumer event");
        recv_pending_[x.slot] = true;
        if (trace_ && x.tx) trace_pending_ = true; // the trace line needs both the scatter and the gather events
    }

    // ---- bootstrap steps (before the first exchange; the caller's thread, the progress thread is idle)
    [[nodiscard]] std::size_t link_count() const noexcept override { return links_.size(); }
    [[nodiscard]] unsigned rings() const noexcept override { return static_cast<unsigned>(links_.size()); }
    void use_rings(unsigned n) override { next_rings_ = n; }
    std::string diagnose(std::size_t bytes) override {
        if (started_) throw std::logic_error("gpu-direct diagnose: an exchange is in flight");
        bytes = std::min<std::size_t>(bytes & ~std::size_t{15}, capacity_);
        if (!bytes) return {};
        hipStream_t s{};
        void *control = nullptr, *dst = nullptr, *sink = nullptr;
        check(hipStreamCreateWithFlags(&s, hipStreamNonBlocking), "diagnose stream");
        check(hipMalloc(&control, bytes), "diagnose buffer");
        check(hipMalloc(&dst, bytes), "diagnose buffer");
        check(hipMalloc(&sink, 64), "diagnose buffer");
        check(hipMemsetAsync(control, 0x3C, bytes, s), "diagnose buffer");
        hipEvent_t e0{}, e1{};
        check(hipEventCreate(&e0), "diagnose event");
        check(hipEventCreate(&e1), "diagnose event");
        std::ostringstream out;
        out << "read path, " << bytes << " B, us (p50 of 20):";
        const char* modes[] = {"copy", "sum", "copy+sum"};
        for (const auto& [where, src] : {std::pair<const char*, const void*>{"hipMalloc", control}, {"TX arena", links_[0].tx},
                                         {"RX arena", links_[0].rx}})
            for (int mode = 0; mode < 3; ++mode)
                for (unsigned blocks : {16U, 64U, 256U, 1024U}) {
                    std::vector<float> t;
                    for (int k = 0; k < 22; ++k) {
                        check(hipEventRecord(e0, s), "diagnose event");
                        enqueue_read_test(src, dst, bytes, mode, blocks, sink, s);
                        check(hipEventRecord(e1, s), "diagnose event");
                        check(hipEventSynchronize(e1), "diagnose");
                        float ms = 0;
                        check(hipEventElapsedTime(&ms, e0, e1), "diagnose event");
                        if (k >= 2) t.push_back(ms * 1000.0F);
                    }
                    std::sort(t.begin(), t.end());
                    char line[96];
                    std::snprintf(line, sizeof line, " %s/%s/%u=%.1f", where, modes[mode], blocks, t[t.size() / 2U]);
                    out << line;
                }
        (void)hipEventDestroy(e0);
        (void)hipEventDestroy(e1);
        (void)hipFree(control);
        (void)hipFree(dst);
        (void)hipFree(sink);
        (void)hipStreamDestroy(s);
        return out.str();
    }
    [[nodiscard]] std::string parameters() const override {
        return "wire 1 (one 64-byte header after the payload, position-weighted word checksum) links " + std::to_string(links_.size()) +
               " depth " + std::to_string(depth_) + " slot " + std::to_string(capacity_) + " min_part " + std::to_string(options_.min_part) +
               " split equal-64B-ceil terminal " + std::to_string(terminal_);
    }
    [[nodiscard]] std::string link_name(std::size_t j) const override { return links_.at(j).name; }
    std::vector<std::string> hello(const std::vector<std::string>& mine, std::chrono::milliseconds timeout) override {
        if (mine.size() != links_.size()) throw std::invalid_argument("gpu-direct hello: one message per link");
        std::vector<std::vector<std::byte>> out(links_.size(), std::vector<std::byte>(hello_bytes)), in(links_.size(), std::vector<std::byte>(hello_bytes));
        for (std::size_t j = 0; j < links_.size(); ++j) {
            if (mine[j].size() >= hello_bytes) throw std::invalid_argument("gpu-direct hello: message too long");
            std::memcpy(out[j].data(), mine[j].data(), mine[j].size());
        }
        std::vector<std::size_t> all(links_.size());
        for (std::size_t j = 0; j < all.size(); ++j) all[j] = j;
        round(all, out, in, timeout, "HELLO");
        std::vector<std::string> got;
        for (const auto& m : in) {
            const auto* c = reinterpret_cast<const char*>(m.data());
            got.emplace_back(c, strnlen(c, hello_bytes));
        }
        return got;
    }
    void reorder(const std::vector<std::size_t>& order) override {
        if (next_ != 0U) throw std::logic_error("gpu-direct: links are reordered only before the first exchange");
        std::vector<bool> seen(links_.size(), false);
        if (order.size() != links_.size()) throw std::invalid_argument("gpu-direct reorder: not a permutation");
        for (auto j : order) {
            if (j >= links_.size() || seen[j]) throw std::invalid_argument("gpu-direct reorder: not a permutation");
            seen[j] = true;
        }
        std::vector<Link> next;
        for (auto j : order) next.push_back(std::move(links_[j]));
        links_ = std::move(next);
    }
    void keep(std::size_t count) override {
        if (next_ != 0U) throw std::logic_error("gpu-direct: links are dropped only before the first exchange");
        if (count == 0U || count > links_.size()) throw std::invalid_argument("gpu-direct keep: 1 .. link count");
        for (std::size_t j = count; j < links_.size(); ++j) (void)links_[j].queue->release();
        links_.resize(count);
    }
    std::vector<double> round_trip_us(std::size_t rounds, std::size_t bytes) override {
        if (!bytes || bytes > capacity_ || !rounds) throw std::invalid_argument("gpu-direct round trip size");
        std::vector<double> p50;
        std::vector<std::vector<std::byte>> out(1, std::vector<std::byte>(bytes)), in(1, std::vector<std::byte>(bytes));
        for (std::size_t j = 0; j < links_.size(); ++j) {
            std::vector<double> t;
            for (std::size_t r = 0; r < rounds; ++r) {
                const auto t0 = std::chrono::steady_clock::now();
                round({j}, out, in, options_.timeout, "round trip");
                t.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
            }
            std::sort(t.begin(), t.end());
            p50.push_back(t[t.size() / 2U]);
        }
        return p50;
    }

private:
    // One synchronous operation each way on each of `which` links, in slot 0 (free before the first exchange,
    // and free again on return): host bytes -> TX slot, RX slot -> host bytes. Each ring advances its own sequence.
    void round(const std::vector<std::size_t>& which, const std::vector<std::vector<std::byte>>& out, std::vector<std::vector<std::byte>>& in,
               std::chrono::milliseconds timeout, const char* what) {
        if (next_ != 0U) throw std::logic_error("gpu-direct: link rounds run only before the first exchange");
        struct EndStream { GpuDirectExchange* self; ~EndStream() { self->end_boot_stream(); } } end_stream{this};
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        std::vector<GpuQueueOperation> rx, tx;
        for (std::size_t i = 0; i < which.size(); ++i) {
            auto& link = links_[which[i]];
            const auto bytes = out[i].size();
            check(hipMemcpyAsync(link.tx, out[i].data(), bytes, hipMemcpyHostToDevice, boot_stream()), "bootstrap TX copy");
            check(hipStreamSynchronize(boot_stream()), "bootstrap TX copy");
            const auto seq = link.next_sequence++;
            rx.push_back(link.queue->prepare(SGU_DIRECTION_RX, 0, seq, bytes, terminal_));
            retry([&] { return link.queue->submit(rx.back()); }, link, "RX SUBMIT", deadline);
            tx.push_back(link.queue->prepare(SGU_DIRECTION_TX, 0, seq, bytes, terminal_));
            link.queue->gpu_produced(tx.back());
            retry([&] { return link.queue->submit(tx.back()); }, link, "TX SUBMIT", deadline);
        }
        std::vector<bool> rx_done(which.size(), false), tx_done(which.size(), false);
        for (std::size_t left = 2U * which.size(); left;) {
            for (std::size_t i = 0; i < which.size(); ++i) {
                auto& link = links_[which[i]];
                while (auto done = link.queue->poll()) {
                    const auto& op = done->operation;
                    if (op.direction == SGU_DIRECTION_TX) {
                        link.queue->peer_consumed(op);
                        link.queue->retire(op);
                        if (!tx_done[i]) { tx_done[i] = true; --left; }
                    } else {
                        check(hipMemcpyAsync(in[i].data(), link.rx, in[i].size(), hipMemcpyDeviceToHost, boot_stream()), "bootstrap RX copy");
                        check(hipStreamSynchronize(boot_stream()), "bootstrap RX copy");
                        link.queue->gpu_consumed(op);
                        link.queue->retire(op);
                        if (!rx_done[i]) { rx_done[i] = true; --left; }
                    }
                }
            }
            if (!left) break;
            if (std::chrono::steady_clock::now() > deadline) {
                std::string pending;
                for (std::size_t i = 0; i < which.size(); ++i)
                    if (!rx_done[i] || !tx_done[i]) pending += (pending.empty() ? "" : ", ") + links_[which[i]].name + (rx_done[i] ? " (TX)" : " (RX)");
                throw std::runtime_error(std::string("gpu-direct: ") + what + " not complete within " + std::to_string(timeout.count()) +
                                         " ms on " + pending + ": the peer's end of that cable is not running this bootstrap (not in gpu_direct mode, "
                                         "not started, or a different link set)");
            }
            relax();
        }
    }
    // A private non-blocking stream for the bootstrap's copies, only while a round runs (the null stream would
    // order them after every blocking stream of the process; a lasting stream would occupy a hardware queue).
    hipStream_t boot_stream() {
        if (!boot_stream_) check(hipStreamCreateWithFlags(&boot_stream_, hipStreamNonBlocking), "bootstrap stream");
        return boot_stream_;
    }
    void end_boot_stream() noexcept {
        if (boot_stream_) (void)hipStreamDestroy(boot_stream_);
        boot_stream_ = nullptr;
    }
    struct Link {
        std::string name;
        std::byte* tx = nullptr;
        std::byte* rx = nullptr;
        std::unique_ptr<LinuxGpuQueue> queue;
        std::vector<std::optional<GpuQueueOperation>> tx_ops, rx_ops;
        std::unique_ptr<std::atomic<bool>[]> tx_free;
        std::uint64_t next_sequence = 1; // this ring's operations (both directions), increasing with no gaps
    };
    struct Exchange {
        std::uint64_t index = 0, sequence = 0;
        std::uint32_t slot = 0;
        std::size_t used = 0, bytes = 0;
        std::array<std::pair<std::size_t, std::size_t>, 8> parts{}; // payload (offset, bytes) per stream
        std::array<std::pair<std::size_t, std::size_t>, 8> wire{};  // bytes on the wire per stream (last + header)
        std::array<std::uint64_t, 8> ring_sequence{};                 // the operation sequence on each used ring
        GpuDirectIdentity identity{};
        bool preposted = false; // its RX were submitted by the previous exchange's job (STRIX_DIST_GPU_PREPOST)
        bool has_next = false;  // the next exchange's geometry is known: its RX is posted before this TX
        Geometry next{};
        bool tx = true, rx = true; // a one-way transfer has only its own direction's operation, scatter or gather
    };
    struct Prepost {
        bool valid = false;
        std::uint64_t index = 0;
        Geometry geometry;
        std::array<std::uint64_t, 8> ring_sequence{};
    };
    // Part j of `used` equal parts at 64-byte boundaries: every part but the last is ceil(bytes / used) rounded up
    // to 64, so no part exceeds the slot when bytes <= used x slot (the slot is whole frames).
    static std::size_t part_stride(std::size_t bytes, std::size_t used) { return ((bytes + used - 1U) / used + 63U) & ~std::size_t{63}; }
    static std::pair<std::size_t, std::size_t> part(std::size_t bytes, std::size_t j, std::size_t used) {
        const auto stride = part_stride(bytes, used);
        const auto begin = std::min(bytes, stride * j), end = j + 1U == used ? bytes : std::min(bytes, stride * (j + 1U));
        return {begin, end - begin};
    }
    // A failed exchange: say why, stop every queue (the peer's pending operations fail instead of waiting for
    // their deadline) and release the stream waiting on the flag with the failure value, which makes the gather
    // kernel trap before any later work can consume the data.
    void abort_links(const std::string& why) noexcept {
        std::fprintf(stderr, "gpu-direct: exchange failed: %s\n", why.c_str());
        {
            std::lock_guard lock(failure_mutex_);
            if (host_failure_.empty()) host_failure_ = why;
        }
        for (auto& link : links_) {
            link.queue->invalidate();
            (void)link.queue->stop();
        }
        __atomic_store_n(flag_, failed_flag, __ATOMIC_RELEASE);
    }
    // Publishes `sequence` unless the flag already holds the failure value.
    void publish(std::uint64_t sequence) noexcept {
        auto seen = __atomic_load_n(flag_, __ATOMIC_ACQUIRE);
        while (seen != failed_flag && !__atomic_compare_exchange_n(flag_, &seen, sequence, false, __ATOMIC_RELEASE, __ATOMIC_ACQUIRE)) {}
    }
    [[noreturn]] void poison(const std::string& what) {
        poisoned_ = true;
        poison_ = what;
        abort_links(what);
        throw std::runtime_error("gpu-direct: " + what);
    }
    // Waits for the previous exchange's job and surfaces its error.
    void collect() {
        if (timed_pending_) {
            timed_pending_ = false;
            float a = 0, b = 0, c = 0;
            if (hipEventSynchronize(t_gather1_) == hipSuccess && hipEventElapsedTime(&a, t_scatter0_, t_scatter1_) == hipSuccess &&
                hipEventElapsedTime(&b, t_scatter1_, t_gather0_) == hipSuccess && hipEventElapsedTime(&c, t_gather0_, t_gather1_) == hipSuccess) {
                stats_.scatter_gpu += a * 1e-3;
                stats_.wait_gpu += b * 1e-3;
                stats_.gather_gpu += c * 1e-3;
                ++stats_.timed;
            }
        }
        if (!pending_job_) return;
        pending_job_ = false;
        if (current_.bytes <= block_wait_bytes() && recv_pending_[current_.slot])
            check(hipEventSynchronize(recv_done_[current_.slot]), "previous gather");
        try { progress_.wait(); } catch (const std::exception& e) {
            poisoned_ = true;
            poison_ = e.what();
            throw;
        }
        check_failure();
        if (trace_ && trace_pending_) { trace_pending_ = false; write_trace(); }
    }
    // One line for the exchange in current_ (its job has finished; its gather may still run: wait for it).
    // consumer = the gather's end on the GPU timeline, placed on the wall clock from the scatter's end, which the
    // progress thread saw as `produced` (it polls the scatter's host-mapped record).
    void write_trace() {
        const auto& x = current_;
        float ms = 0.0F;
        std::uint64_t consumer = 0;
        if (hipEventSynchronize(recv_done_[x.slot]) == hipSuccess &&
            hipEventElapsedTime(&ms, send_ready_[x.slot], recv_done_[x.slot]) == hipSuccess && times_.produced)
            consumer = times_.produced + static_cast<std::uint64_t>(static_cast<double>(ms) * 1e6);
        std::fprintf(trace_, "G %llu %zu %u %u %zu %llu %llu %llu %llu %llu %llu\n", static_cast<unsigned long long>(x.sequence),
                     x.bytes, x.identity.tag, x.identity.layer, x.used, static_cast<unsigned long long>(times_.entry),
                     static_cast<unsigned long long>(times_.produced), static_cast<unsigned long long>(times_.submitted),
                     static_cast<unsigned long long>(times_.rx_done), static_cast<unsigned long long>(times_.tx_done),
                     static_cast<unsigned long long>(consumer));
    }
    static std::uint64_t trace_wall_ns() {
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
    }
    template <class F> void retry(F&& submit, Link& link, const char* what, std::chrono::steady_clock::time_point deadline) {
        while (!submit()) {
            stats_.eagain_retries += 1.0;
            if (std::chrono::steady_clock::now() > deadline) throw std::runtime_error("gpu-direct: " + link.name + ": " + what + " kept returning EAGAIN");
            relax();
        }
    }
    void handle(Link& link, const GpuQueueOperation& op) {
        auto& ops = op.direction == SGU_DIRECTION_TX ? link.tx_ops : link.rx_ops;
        if (op.direction == SGU_DIRECTION_TX) {
            // The bytes left this slot: the peer had posted the matching RX (end-to-end flow control), so the
            // slot may be overwritten by the next producer copy.
            link.queue->peer_consumed(op);
            link.queue->retire(op);
            ops[op.slot].reset();
            link.tx_free[op.slot].store(true, std::memory_order_release);
        } else {
            rx_landed_[op.slot][static_cast<std::size_t>(&link - links_.data())] = true;
        }
    }
    void transfer(const Exchange& x) {
        const auto deadline = std::chrono::steady_clock::now() + options_.timeout;
        trace("transfer: begin", x.sequence);
        for (auto& link : links_) while (auto c = link.queue->poll()) handle(link, c->operation); // earlier TX completions
        // RX first (same sequence and size as the peer's TX), then TX once the producer copy is complete.
        if (x.rx && !x.preposted)
            for (std::size_t j = 0; j < x.used; ++j) {
                auto& link = links_[j];
                const auto op = link.queue->prepare(SGU_DIRECTION_RX, x.slot, x.ring_sequence[j], x.wire[j].second, terminal_);
                link.rx_ops[x.slot] = op;
                retry([&] { return link.queue->submit(op); }, link, "RX SUBMIT", deadline);
            }
        trace("transfer: RX submitted", x.sequence);
        if (const auto lead = rx_lead_us(); lead > 0) {
            const auto until = std::chrono::steady_clock::now() + std::chrono::microseconds(lead);
            while (std::chrono::steady_clock::now() < until) relax();
        }
        // TX: the producer copy is complete as seen by the CPU before the controller may read the slot.
        const auto t0 = std::chrono::steady_clock::now();
        // Poll the scatter's host-mapped record (no HIP call), then confirm with the producer event (C-1 witness).
        if (x.tx) {
            while (__atomic_load_n(produced_, __ATOMIC_ACQUIRE) < x.sequence) {
                if (std::chrono::steady_clock::now() > deadline) throw std::runtime_error("gpu-direct: the producer copy did not complete in time");
                relax();
            }
            if (trace_) times_.produced = trace_wall_ns();
            spin_event(send_ready_[x.slot], "the producer copy", deadline);
            trace("transfer: producer copy complete", x.sequence);
        }
        if (x.has_next) {
            // The next exchange's RX before this TX. Its slot's previous RX (depth exchanges ago) is retired once its
            // gather is done (enqueued before this exchange's scatter on the same stream, so normally complete here).
            const std::uint32_t next_slot = static_cast<std::uint32_t>((x.index + 1U) % depth_);
            Prepost p;
            p.valid = true;
            p.index = x.index + 1U;
            p.geometry = x.next;
            for (std::size_t j = 0; j < x.next.used; ++j) {
                auto& link = links_[j];
                if (auto& old = link.rx_ops[next_slot]) {
                    spin_event(recv_done_[next_slot], "the consumer copy of an RX slot", deadline);
                    link.queue->gpu_consumed(*old);
                    link.queue->retire(*old);
                    old.reset();
                }
                p.ring_sequence[j] = link.next_sequence++;
                const auto op = link.queue->prepare(SGU_DIRECTION_RX, next_slot, p.ring_sequence[j], x.next.wire[j].second, terminal_);
                link.rx_ops[next_slot] = op;
                retry([&] { return link.queue->submit(op); }, link, "RX SUBMIT (next)", deadline);
            }
            prepost_ = p;
        }
        const auto t1 = std::chrono::steady_clock::now();
        stats_.producer_wait += std::chrono::duration<double>(t1 - t0).count();
        const auto t_witness = std::chrono::steady_clock::now();
        for (std::size_t j = 0; x.tx && j < x.used; ++j) {
            auto& link = links_[j];
            link.tx_free[x.slot].store(false, std::memory_order_release);
            const auto op = link.queue->prepare(SGU_DIRECTION_TX, x.slot, x.ring_sequence[j], x.wire[j].second, terminal_);
            link.tx_ops[x.slot] = op;
            link.queue->gpu_produced(op);
            retry([&] { return link.queue->submit(op); }, link, "TX SUBMIT", deadline);
        }
        trace("transfer: TX submitted", x.sequence);
        if (trace_) times_.submitted = trace_wall_ns();
        // Completions of every used stream: RX lands, TX leaves.
        const auto t_submitted = std::chrono::steady_clock::now();
        stats_.submit += std::chrono::duration<double>(t_submitted - t_witness).count();
        std::chrono::steady_clock::time_point t_flag{};
        bool flagged = false;
        while (true) {
            bool all_rx = true, all_tx = true;
            for (std::size_t j = 0; j < x.used; ++j) {
                auto& link = links_[j];
                while (auto done = link.queue->poll()) handle(link, done->operation);
                all_rx = all_rx && (!x.rx || rx_landed_[x.slot][j]);
                all_tx = all_tx && (!x.tx || !link.tx_ops[x.slot]);
            }
            if (all_rx && !flagged) {
                // Every part is in device memory: release the stream waiting in finish().
                publish(x.sequence);
                if (trace_) times_.rx_done = trace_wall_ns();
                flagged = true;
                t_flag = std::chrono::steady_clock::now();
                stats_.rx_wait += std::chrono::duration<double>(t_flag - t_submitted).count();
                trace("transfer: RX complete, flag written", x.sequence);
            }
            if (trace_ && all_tx && !times_.tx_done) times_.tx_done = trace_wall_ns();
            if (all_rx && (all_tx || lazy_tx())) {
                for (std::size_t j = 0; j < x.used; ++j) rx_landed_[x.slot][j] = false;
                break;
            }
            if (std::chrono::steady_clock::now() > deadline)
                throw std::runtime_error("gpu-direct: exchange " + std::to_string(x.sequence) + " not complete within " +
                                         std::to_string(options_.timeout.count()) + " ms (" + (all_rx ? "TX" : "RX") +
                                         " pending: peer stalled, gone, or ranks disagree on the exchanges)");
            relax();
        }
        const auto t_done = std::chrono::steady_clock::now();
        stats_.completion_wait += std::chrono::duration<double>(t_done - t1).count();
        if (!lazy_tx()) stats_.tx_tail += std::chrono::duration<double>(t_done - t_flag).count();
    }
    // Consumes TX completions (on the calling thread; the progress thread is idle or is the caller) until TX slot
    // `slot` is free on every link, or every TX is complete when `slot` is empty.
    void drain_tx(std::optional<std::uint32_t> slot, std::chrono::steady_clock::time_point deadline) {
        while (true) {
            bool done = true;
            for (auto& link : links_) {
                while (auto c = link.queue->poll()) handle(link, c->operation);
                if (slot) done = done && link.tx_free[*slot].load(std::memory_order_acquire);
                else for (std::uint32_t s = 0; s < depth_; ++s) done = done && link.tx_free[s].load(std::memory_order_acquire);
            }
            if (done) return;
            if (std::chrono::steady_clock::now() > deadline)
                poison("a TX operation did not complete in time (the peer stopped receiving)");
            relax();
        }
    }

    GpuDirectOptions options_;
    std::vector<Link> links_;
    std::uint32_t depth_ = 0, terminal_ = 0;
    std::uint64_t capacity_ = 0, next_ = 0;
    std::vector<hipEvent_t> send_ready_, recv_done_;
    std::vector<bool> recv_pending_;
    std::array<std::array<bool, 8>, 8> rx_landed_{}; // [slot][link]: its RX completed, not yet consumed by transfer()
    Prepost prepost_;
    std::uint64_t* flag_ = nullptr;
    GpuDirectScratch* scratch_ = nullptr; // [0] scatter, [1] gather
    GpuDirectError* error_ = nullptr;
    mutable std::mutex failure_mutex_;
    std::string host_failure_;
    GpuDirectError* error_device_ = nullptr;
    std::uint64_t* produced_ = nullptr;        // host view: the last sequence whose scatter completed
    std::uint64_t* produced_device_ = nullptr; // the same, for the scatter kernel
    bool timing_ = false, timed_pending_ = false;
    hipEvent_t t_scatter0_{}, t_scatter1_{}, t_gather0_{}, t_gather1_{};
    std::uint64_t* flag_device_ = nullptr;
    Exchange current_;
    hipStream_t boot_stream_ = nullptr;
    unsigned next_rings_ = 0;
    ExchangeDirection next_direction_ = ExchangeDirection::both; // of the next start() (use_direction), one shot
    std::size_t hint_bytes_ = 0;
    unsigned hint_rings_ = 0;
    void* dummy_local_ = nullptr;  // internal exchange after a wrong guess (allocated at the first one)
    void* dummy_remote_ = nullptr;
    bool started_ = false, pending_job_ = false, poisoned_ = false;
    std::string poison_;
    DeviceExchangeStats stats_;
    // STRIX_DIST_TRACE: wall-clock ns of the exchange in flight (written by its job, read after progress_.wait()).
    struct TraceTimes { std::uint64_t entry = 0, produced = 0, submitted = 0, rx_done = 0, tx_done = 0; };
    std::FILE* trace_ = nullptr;
    TraceTimes times_;
    bool trace_pending_ = false;
    KeepAwake awake_;
    Progress progress_;
};

} // namespace

std::unique_ptr<GpuDirectLinks> gpu_direct_exchange(std::vector<GpuDirectLink> links, GpuDirectOptions options) {
    return std::make_unique<GpuDirectExchange>(std::move(links), options);
}

} // namespace strix::dist
