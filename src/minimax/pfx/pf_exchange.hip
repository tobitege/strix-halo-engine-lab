// MiniMax M3.1 TP2 prefill exchange module: the engine (see pf_exchange.hpp).
#include "pf_exchange.hpp"
#include "minimax/xfer/minimax_xfer_device.hpp"
#include "strix/env_flags.hpp"

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <span>
#include <sstream>
#include <stdexcept>

namespace strix::minimax::pfx {
namespace {

void hip_check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("mmpfx: ") + what + ": " + hipGetErrorString(e));
}

using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double>(b - a).count(); }

constexpr std::size_t kTxFlag = 0, kRxFlag = 8;  // u64 index: separate 64-byte lines

__global__ void publish_kernel(std::uint64_t* flag, std::uint64_t sequence) {
    if (threadIdx.x == 0) {
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "");
        __hip_atomic_store(flag, sequence, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    }
}

constexpr unsigned kCopyThreads = 256, kCopyMaxBlocks = 512;
__global__ __launch_bounds__(kCopyThreads) void copy_publish_kernel(std::byte* dst, const std::byte* src, std::size_t bytes,
                                                                    xfer::ProducerTarget target) {
    const std::size_t vec = bytes / 16, stride = static_cast<std::size_t>(gridDim.x) * kCopyThreads;
    auto* d = reinterpret_cast<uint4*>(dst);
    const auto* s = reinterpret_cast<const uint4*>(src);
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * kCopyThreads + threadIdx.x; i < vec; i += stride) d[i] = s[i];
    if (blockIdx.x == 0)
        for (std::size_t i = vec * 16 + threadIdx.x; i < bytes; i += kCopyThreads) dst[i] = src[i];
    xfer::mmx_publish(target, gridDim.x);
}

void pin(std::thread& t, int cpu) {
    if (cpu < 0) return;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    pthread_setaffinity_np(t.native_handle(), sizeof(set), &set);
}

bool same_identity(const Header& a, const Header& b) {
    return a.epoch == b.epoch && a.plan == b.plan && a.sequence == b.sequence && a.request == b.request && a.bytes == b.bytes &&
           a.kind == b.kind && a.layer == b.layer && a.chunk == b.chunk && a.rank + b.rank == 1U;
}

std::string describe(const Header& h) {
    return "seq " + std::to_string(h.sequence) + " request " + std::to_string(h.request) + " kind " + std::to_string(h.kind) + " layer " +
           std::to_string(h.layer) + " bytes " + std::to_string(h.bytes) + " chunk " + std::to_string(h.chunk) + " rank " + std::to_string(h.rank);
}

}  // namespace

void Options::from_env() {
    depth = static_cast<unsigned>(strix::env_number<std::uint64_t>("STRIX_MM_PFX_DEPTH", depth));
    chunk = static_cast<std::size_t>(strix::env_number<std::uint64_t>("STRIX_MM_PFX_CHUNK", chunk));
    if (const char* v = std::getenv("STRIX_MM_PFX_CRC"); v != nullptr && v[0] == '0') crc = false;
    joined = strix::envcfg::flag("STRIX_MM_PFX_JOINED", joined);
    if (const char* v = std::getenv("STRIX_MM_PFX_PROGRESS_CPU"); v != nullptr && *v != '\0') progress_cpu = std::atoi(v);
    if (const char* v = std::getenv("STRIX_MM_PFX_CRC_CPUS"); v != nullptr && *v != '\0') {
        std::stringstream in(v);
        std::string a, b;
        std::getline(in, a, ',');
        std::getline(in, b, ',');
        if (a.empty() || b.empty()) throw std::invalid_argument("STRIX_MM_PFX_CRC_CPUS must be TX_CPU,RX_CPU");
        crc_tx_cpu = std::stoi(a);
        crc_rx_cpu = std::stoi(b);
    }
    timeout = std::chrono::milliseconds(strix::env_number<std::uint64_t>("STRIX_MM_PFX_TIMEOUT_MS", static_cast<std::uint64_t>(timeout.count())));
}

Exchanger::Exchanger(dist::Transport& borrowed, Options options)
    : transport_(&borrowed), options_(options),
      frame_(options.chunk ? (options.capacity + options.chunk - 1) / options.chunk : 1) {
    if (options_.rank > 1) throw std::invalid_argument("mmpfx: rank must be 0 or 1");
    if (options_.capacity == 0) throw std::invalid_argument("mmpfx: capacity");
    if (options_.depth < 2 || options_.depth > 16) throw std::invalid_argument("mmpfx: depth must be 2..16");
    if (options_.chunk < (std::size_t{64} << 10) || options_.chunk > (std::size_t{64} << 20) || options_.chunk % 64 != 0)
        throw std::invalid_argument("mmpfx: chunk must be a multiple of 64 in [64 KiB, 64 MiB]");
    max_chunks_ = (options_.capacity + options_.chunk - 1) / options_.chunk;
    // The joined form writes its record (identity, CRC list) behind the payload of the slot.
    const std::size_t tail = options_.joined ? xport::FramedExchange<Header>::tail_bytes(max_chunks_) : 0;
    stride_ = (options_.capacity + tail + 4095) & ~std::size_t{4095};
    const std::size_t total = stride_ * options_.depth;
    const unsigned flags = hipHostMallocMapped | hipHostMallocCoherent;
    if (options_.tx_uncached) {
        hip_check(hipExtMallocWithFlags(reinterpret_cast<void**>(&tx_dev_), total, hipDeviceMallocUncached), "uncached TX slots");
        tx_host_ = tx_dev_;
    } else {
        hip_check(hipHostMalloc(reinterpret_cast<void**>(&tx_host_), total, flags), "TX slots");
        hip_check(hipHostGetDevicePointer(reinterpret_cast<void**>(&tx_dev_), tx_host_, 0), "TX slots");
    }
    hip_check(hipHostMalloc(reinterpret_cast<void**>(&rx_host_), total, flags), "RX slots");
    hip_check(hipHostMalloc(reinterpret_cast<void**>(&flags_host_), 4096, flags), "flags");
    hip_check(hipHostGetDevicePointer(reinterpret_cast<void**>(&rx_dev_), rx_host_, 0), "RX slots");
    hip_check(hipHostGetDevicePointer(reinterpret_cast<void**>(&flags_dev_), flags_host_, 0), "flags");
    std::memset(flags_host_, 0, 4096);
    hip_check(hipMalloc(reinterpret_cast<void**>(&counter_dev_), 64), "counter");
    hip_check(hipMemset(counter_dev_, 0, 64), "counter");
    hip_check(hipDeviceSynchronize(), "init");
    tx_lane_.crcs.resize(max_chunks_);
    rx_lane_.crcs.resize(max_chunks_);
    if (options_.crc) {
        tx_lane_.thread = std::thread([this] { lane_run(tx_lane_, options_.crc_tx_cpu); });
        rx_lane_.thread = std::thread([this] { lane_run(rx_lane_, options_.crc_rx_cpu); });
    }
    thread_ = std::thread([this] { run(); });
    pin(thread_, options_.progress_cpu);
}

Exchanger::~Exchanger() {
    if (done_.load() < posted_.load() && !failed_.load()) transport_->abort("mmpfx: closing with exchanges in flight");
    stop_ = true;
    {
        std::lock_guard<std::mutex> g(mutex_);
        wake_.notify_all();
    }
    if (thread_.joinable()) thread_.join();
    for (Lane* l : {&tx_lane_, &rx_lane_}) {
        {
            std::lock_guard<std::mutex> g(l->mutex);
            l->stop = true;
            l->generation.fetch_add(1);
        }
        l->wake.notify_all();
        if (l->thread.joinable()) l->thread.join();
    }
    if (flags_host_ != nullptr) {
        __atomic_store_n(&flags_host_[kRxFlag], ~std::uint64_t{0}, __ATOMIC_RELEASE);  // release any waiting stream
        (void)hipStreamSynchronize(options_.stream);
    }
    (void)hipFree(counter_dev_);
    (void)hipHostFree(flags_host_);
    (void)hipHostFree(rx_host_);
    if (options_.tx_uncached) (void)hipFree(tx_dev_);
    else (void)hipHostFree(tx_host_);
}

std::byte* Exchanger::tx_slot(std::uint64_t sequence) const { return tx_host_ + ((sequence - 1) % options_.depth) * stride_; }
std::byte* Exchanger::rx_slot(std::uint64_t sequence) const { return rx_host_ + ((sequence - 1) % options_.depth) * stride_; }

bool Exchanger::set_tx_uncached(bool on) {
    if (on == options_.tx_uncached) return true;
    check();
    if (retired_ != next_) throw std::logic_error("mmpfx: changing TX memory with unretired exchanges");
    drain();
    hip_check(hipStreamSynchronize(options_.stream), "TX memory change");
    std::byte *host = nullptr, *dev = nullptr;
    const std::size_t total = stride_ * options_.depth;
    if (on) {
        const hipError_t status = hipExtMallocWithFlags(reinterpret_cast<void**>(&dev), total, hipDeviceMallocUncached);
        if (status != hipSuccess) {
            (void)hipGetLastError();
            std::printf("mmpfx: uncached TX slots unavailable (%s, %zu B); TX stays host-coherent\n", hipGetErrorString(status), total);
            return false;
        }
        host = dev;
    } else {
        hip_check(hipHostMalloc(reinterpret_cast<void**>(&host), total, hipHostMallocMapped | hipHostMallocCoherent), "TX slots");
        const hipError_t status = hipHostGetDevicePointer(reinterpret_cast<void**>(&dev), host, 0);
        if (status != hipSuccess) {
            (void)hipHostFree(host);
            hip_check(status, "TX slots");
        }
    }
    if (options_.tx_uncached) (void)hipFree(tx_dev_);
    else (void)hipHostFree(tx_host_);
    tx_host_ = host;
    tx_dev_ = dev;
    options_.tx_uncached = on;
    return true;
}

void* Exchanger::tx() {
    const std::uint64_t s = next_ + 1;
    if (s > retired_ + options_.depth) throw std::logic_error("mmpfx: tx() of a sequence whose slot is still in use (retire() of s - depth missing)");
    return tx_dev_ + (tx_slot(s) - tx_host_);
}

void* Exchanger::tx(unsigned ahead) {
    const std::uint64_t s = next_ + 1 + ahead;
    if (s > retired_ + options_.depth) throw std::logic_error("mmpfx: tx() of a sequence whose slot is still in use (retire() of s - depth missing)");
    return tx_dev_ + (tx_slot(s) - tx_host_);
}

void Exchanger::send(std::size_t bytes, const Tag& tag) {
    check();
    if (bytes == 0 || bytes > options_.capacity) throw std::invalid_argument("mmpfx: message size outside 1..capacity");
    const std::uint64_t s = next_ + 1;
    if (s > retired_ + options_.depth) throw std::logic_error("mmpfx: send() of a sequence whose slot is still in use");
    hipLaunchKernelGGL(publish_kernel, dim3(1), dim3(32), 0, options_.stream, flags_dev_ + kTxFlag, s);
    hip_check(hipGetLastError(), "publish kernel");
    slot_wait_.block_while(static_cast<std::uint32_t>(tag.kind), [&] {
        return posted_.load(std::memory_order_relaxed) - done_.load(std::memory_order_acquire) >= kQueue;
    }, [&] {
        check();
        _mm_pause();
    });
    jobs_[(s - 1) % kQueue] = Job{s, bytes, tag};
    next_ = s;
    posted_.store(s, std::memory_order_seq_cst);
    if (sleeping_.load(std::memory_order_seq_cst)) {
        std::lock_guard<std::mutex> g(mutex_);
        wake_.notify_one();
    }
}

void Exchanger::send_copy(const void* local, std::size_t bytes, const Tag& tag) {
    check();
    if (bytes == 0 || bytes > options_.capacity) throw std::invalid_argument("mmpfx: message size outside 1..capacity");
    auto* dst = static_cast<std::byte*>(tx());
    const std::uint64_t s = next_ + 1;
    const unsigned blocks =
        static_cast<unsigned>(std::clamp<std::size_t>((bytes + 16 * kCopyThreads - 1) / (16 * kCopyThreads), 1, kCopyMaxBlocks));
    hipLaunchKernelGGL(copy_publish_kernel, dim3(blocks), dim3(kCopyThreads), 0, options_.stream, dst, static_cast<const std::byte*>(local),
                       bytes, xfer::ProducerTarget{flags_dev_ + kTxFlag, counter_dev_, s});
    hip_check(hipGetLastError(), "copy kernel");
    slot_wait_.block_while(static_cast<std::uint32_t>(tag.kind), [&] {
        return posted_.load(std::memory_order_relaxed) - done_.load(std::memory_order_acquire) >= kQueue;
    }, [&] {
        check();
        _mm_pause();
    });
    jobs_[(s - 1) % kQueue] = Job{s, bytes, tag};
    next_ = s;
    posted_.store(s, std::memory_order_seq_cst);
    if (sleeping_.load(std::memory_order_seq_cst)) {
        std::lock_guard<std::mutex> g(mutex_);
        wake_.notify_one();
    }
}

Exchanger::View Exchanger::recv() {
    const std::uint64_t s = received_ + 1;
    if (s > next_) throw std::logic_error("mmpfx: recv() without a matching send()");
    hip_check(hipStreamWaitValue64(options_.stream, flags_dev_ + kRxFlag, s, hipStreamWaitValueGte, ~std::uint64_t{0}),
              "wait for the peer's message");
    received_ = s;
    return View{tx_dev_ + (tx_slot(s) - tx_host_), rx_dev_ + (rx_slot(s) - rx_host_)};
}

void Exchanger::retire() {
    if (retired_ >= received_) throw std::logic_error("mmpfx: retire() without a received exchange");
    ++retired_;
}

void Exchanger::drain() {
    const auto deadline = Clock::now() + options_.timeout;
    while (done_.load(std::memory_order_acquire) < posted_.load(std::memory_order_acquire) && !failed_.load()) {
        if (Clock::now() > deadline) {
            const std::string why = "mmpfx: drain: exchanges still in flight after the deadline";
            {
                std::lock_guard<std::mutex> g(failure_mutex_);
                if (failure_.empty()) failure_ = why;
            }
            failed_.store(true, std::memory_order_release);
            __atomic_store_n(&flags_host_[kRxFlag], ~std::uint64_t{0}, __ATOMIC_RELEASE);
            transport_->abort(why);
            throw std::runtime_error(why);
        }
        _mm_pause();
    }
    check();
}

void Exchanger::check() const {
    if (!failed_.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> g(failure_mutex_);
    throw std::runtime_error("mmpfx: exchange failed: " + failure_);
}

Stats Exchanger::stats() const {
    std::lock_guard<std::mutex> g(stats_mutex_);
    return stats_;
}

std::string Exchanger::name() const {
    return "mmpfx(" + transport_->name() + ", depth " + std::to_string(options_.depth) + ", chunk " + std::to_string(options_.chunk) +
           " B, crc " + (options_.crc ? "on" : "off") + ", TX " + (options_.tx_uncached ? "uncached" : "host-coherent") + ")";
}

std::string Exchanger::debug_state() const {
    return "rank " + std::to_string(options_.rank) + " posted " + std::to_string(posted_.load()) + " done " + std::to_string(done_.load()) +
           " sent " + std::to_string(next_) + " received " + std::to_string(received_) + " retired " + std::to_string(retired_) + " txflag " +
           std::to_string(__atomic_load_n(&flags_host_[kTxFlag], __ATOMIC_ACQUIRE)) + " rxflag " +
           std::to_string(__atomic_load_n(&flags_host_[kRxFlag], __ATOMIC_ACQUIRE)) + " txlane " + std::to_string(tx_lane_.done.load()) + "/" +
           std::to_string(tx_lane_.count) + " rxlane " + std::to_string(rx_lane_.done.load()) + "/" + std::to_string(rx_lane_.ready.load()) + "/" +
           std::to_string(rx_lane_.count) + " failed " + (failed_.load() ? "yes" : "no") + " sleeping " + (sleeping_.load() ? "yes" : "no");
}

void Exchanger::fail(const std::string& why) {
    {
        std::lock_guard<std::mutex> g(failure_mutex_);
        if (failure_.empty()) failure_ = why;
    }
    failed_.store(true, std::memory_order_release);
    __atomic_store_n(&flags_host_[kRxFlag], ~std::uint64_t{0}, __ATOMIC_RELEASE);
    transport_->abort(why);
    throw std::runtime_error(why);
}

// ---- CRC lanes: a job is (data, bytes, chunk, count); chunk k is computed once `ready` > k.
void Exchanger::lane_run(Lane& lane, int cpu) {
    if (cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
    std::uint64_t seen = 0;
    const auto idle = std::chrono::microseconds(options_.idle_spin_us);
    while (true) {
        auto last = Clock::now();
        std::uint64_t g;
        while ((g = lane.generation.load(std::memory_order_acquire)) == seen) {
            if (Clock::now() - last < idle) {
                _mm_pause();
                continue;
            }
            std::unique_lock<std::mutex> lock(lane.mutex);
            lane.wake.wait_for(lock, std::chrono::milliseconds(50),
                               [&] { return lane.generation.load(std::memory_order_acquire) != seen; });
        }
        seen = g;
        if (lane.stop.load()) return;
        // The job's metadata, read once after the generation acquire: post_lane() rewrites it for the next job as soon
        // as `done` reaches `count`, possibly while this loop is still finishing.
        const std::byte* const data = lane.data;
        const std::size_t bytes = lane.bytes, chunk = lane.chunk, count = lane.count;
        std::uint32_t* const crcs = lane.crcs.data();
        for (std::size_t k = 0; k < count; ++k) {
            while (lane.ready.load(std::memory_order_acquire) <= k) {
                if (failed_.load(std::memory_order_relaxed) || lane.stop.load(std::memory_order_relaxed)) break;
                _mm_pause();
            }
            if (lane.ready.load(std::memory_order_acquire) <= k) break;  // failed / stopped: abandon the job
            const std::size_t off = k * chunk, n = std::min(chunk, bytes - off);
            crcs[k] = crc32c(data + off, n);
            lane.done.store(k + 1, std::memory_order_release);
        }
    }
}

void Exchanger::post_lane(Lane& lane, const std::byte* data, std::size_t bytes, std::size_t ready) {
    lane.data = data;
    lane.bytes = bytes;
    lane.chunk = options_.chunk;
    lane.count = (bytes + options_.chunk - 1) / options_.chunk;
    lane.done.store(0, std::memory_order_relaxed);
    lane.ready.store(ready, std::memory_order_relaxed);
    {
        std::lock_guard<std::mutex> g(lane.mutex);
        lane.generation.fetch_add(1, std::memory_order_release);
    }
    lane.wake.notify_one();
}

void Exchanger::wait_lane(Lane& lane, Clock::time_point deadline) {
    unsigned n = 0;
    while (lane.done.load(std::memory_order_acquire) < lane.count) {
        if ((++n & 1023U) == 0 && Clock::now() > deadline) fail("mmpfx: CRC lane did not finish within the timeout");
        _mm_pause();
    }
}

void Exchanger::process(const Job& job) {
    const auto t0 = Clock::now();
    const auto deadline = t0 + options_.timeout;
    unsigned spins = 0;
    while (__atomic_load_n(&flags_host_[kTxFlag], __ATOMIC_ACQUIRE) < job.sequence) {
        if (stop_.load(std::memory_order_relaxed)) fail("mmpfx: stopped while waiting for a producer");
        if ((++spins & 1023U) == 0 && Clock::now() > deadline)
            fail("mmpfx: producer of sequence " + std::to_string(job.sequence) + " not published within the timeout");
        _mm_pause();
    }
    const auto t1 = Clock::now();
    std::byte* tx = tx_slot(job.sequence);
    std::byte* rx = rx_slot(job.sequence);
    const std::size_t count = (job.bytes + options_.chunk - 1) / options_.chunk;
    if (options_.crc) post_lane(tx_lane_, tx, job.bytes, count);  // the TX slot is complete: every chunk is ready
    const Header mine = PrefillTraits::header(options_, job.sequence, job.bytes, job.tag);
    const auto kind = static_cast<std::uint32_t>(job.tag.kind);
    const auto wire = [&](std::span<const std::byte> out, std::span<std::byte> in) {
        peer_wait_.measure(kind, [&] { transport_->exchange(out, in); });
    };
    const auto landed = [&](std::size_t chunks) {
        if (options_.crc) rx_lane_.ready.store(chunks, std::memory_order_release);
    };
    const auto tx_crcs = [&] { wait_lane(tx_lane_, deadline); return static_cast<const std::uint32_t*>(tx_lane_.crcs.data()); };
    const auto differs = [&](const Header& peer) {
        fail("mmpfx: peer header differs: mine " + describe(mine) + " / peer " + describe(peer));
    };
    xport::FrameResult<Header> peer;
    try {
        if (!negotiated_) {
            // Once per connection, before the first exchange: both ranks offer their forms and use the common ones.
            // mine.chunk = the wire chunk size | the CRC bit: the two values that fix every later part size.
            if (options_.joined)
                forms_ = xport::negotiate_frame_forms(wire, xport::frame_rendezvous | xport::frame_joined, mine.chunk);
            negotiated_ = true;
        }
        if ((forms_ & xport::frame_joined) != 0) {
            if (options_.crc) post_lane(rx_lane_, rx, job.bytes, 0);
            peer = frame_.joined(wire, mine, kTrailerMagic, tx, rx, job.bytes, options_.chunk, options_.crc, landed, tx_crcs);
        } else {
            // The ranks agree on the identity and the size before any payload moves (a mismatch cannot desynchronise
            // the streams' part sizes).
            peer = frame_.rendezvous(wire, mine, kTrailerMagic, tx, rx, job.bytes, options_.chunk, options_.crc,
                [&](const Header& theirs) {
                    if (theirs.magic != kHeaderMagic || !same_identity(mine, theirs)) differs(theirs);
                    if (options_.crc) post_lane(rx_lane_, rx, job.bytes, 0);
                }, landed, tx_crcs);
        }
    } catch (const std::exception& e) {
        if (failed_.load()) throw;
        fail(std::string("mmpfx: transport: ") + e.what());
    }
    if (peer.first.magic != peer.first_magic || !same_identity(mine, peer.first)) differs(peer.first);
    const auto t2 = peer.landed;
    if (options_.crc) {
        // The identity again + the peer's per-chunk TX CRCs: the list must equal our RX CRCs.
        if (peer.trailer.magic != kTrailerMagic || !same_identity(mine, peer.trailer))
            fail("mmpfx: peer trailer differs: " + describe(peer.trailer));
        wait_lane(rx_lane_, deadline);
        if (std::memcmp(peer.crcs, rx_lane_.crcs.data(), count * 4) != 0) {
            std::size_t k = 0;
            while (k < count && std::memcmp(peer.crcs + k * 4, &rx_lane_.crcs[k], 4) == 0) ++k;
            fail("mmpfx: payload CRC mismatch at sequence " + std::to_string(job.sequence) + " chunk " + std::to_string(k) + " of " +
                 std::to_string(count));
        }
    }
    __atomic_store_n(&flags_host_[kRxFlag], job.sequence, __ATOMIC_RELEASE);
    const auto t3 = Clock::now();
    std::lock_guard<std::mutex> g(stats_mutex_);
    stats_.exchanges += 1;
    stats_.bytes += job.bytes;
    stats_.producer_wait += seconds(t0, t1);
    stats_.wire += seconds(t1, t2);
    stats_.tail += seconds(t2, t3);
}

void Exchanger::run() {
    auto last = Clock::now();
    const auto idle = std::chrono::microseconds(options_.idle_spin_us);
    try {
        while (!stop_.load(std::memory_order_relaxed)) {
            const std::uint64_t d = done_.load(std::memory_order_relaxed);
            if (d < posted_.load(std::memory_order_acquire)) {
                process(jobs_[d % kQueue]);
                done_.store(d + 1, std::memory_order_release);
                last = Clock::now();
                continue;
            }
            if (Clock::now() - last < idle) {
                _mm_pause();
                continue;
            }
            std::unique_lock<std::mutex> lock(mutex_);
            sleeping_.store(true, std::memory_order_seq_cst);
            wake_.wait(lock, [&] { return stop_.load() || done_.load() < posted_.load(std::memory_order_seq_cst); });
            sleeping_.store(false, std::memory_order_seq_cst);
            last = Clock::now();
        }
    } catch (const std::exception& e) {
        if (!failed_.load()) {
            std::lock_guard<std::mutex> g(failure_mutex_);
            if (failure_.empty()) failure_ = e.what();
            failed_.store(true);
        }
        __atomic_store_n(&flags_host_[kRxFlag], ~std::uint64_t{0}, __ATOMIC_RELEASE);
    }
}

}  // namespace strix::minimax::pfx
