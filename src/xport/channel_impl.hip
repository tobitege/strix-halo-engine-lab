// Template implementation; included by each engine instantiation TU.
#include "channel.hpp"
#include "channel_device.hpp"
#include "wire.hpp"
#include "small_progress.hpp"
#include "strix/hip_argmax_core.hpp"

#include <immintrin.h>
#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <span>
#include <stdexcept>

namespace strix::xport {
namespace {

void hip_check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("glmxfer: ") + what + ": " + hipGetErrorString(e));
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

constexpr unsigned kCopyThreads = 256;
__global__ __launch_bounds__(kCopyThreads) void copy_publish_kernel(std::byte* dst, const std::byte* src, std::size_t bytes,
                                                                    ProducerTarget target) {
    const std::size_t vec = bytes / 16, stride = static_cast<std::size_t>(gridDim.x) * kCopyThreads;
    auto* d = reinterpret_cast<uint4*>(dst);
    const auto* s = reinterpret_cast<const uint4*>(src);
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * kCopyThreads + threadIdx.x; i < vec; i += stride) d[i] = s[i];
    if (blockIdx.x == 0)
        for (std::size_t i = vec * 16 + threadIdx.x; i < bytes; i += kCopyThreads) dst[i] = src[i];
    publish(target, gridDim.x);
}

constexpr unsigned kArgThreads = 1024;
__device__ __forceinline__ bool better(float v, std::uint32_t i, float bv, std::uint32_t bi) { return v > bv || (v == bv && i < bi); }

[[maybe_unused]] __global__ __launch_bounds__(kArgThreads) void argmax_candidate_kernel(const float* logits, unsigned vn, unsigned ld, std::uint32_t id0,
                                                                       std::uint32_t* tx, ProducerTarget target) {
    struct Best {
        float value;
        std::uint32_t id;
    };
    __shared__ Best sb[kArgThreads];
    __shared__ unsigned bad;
    const unsigned t = threadIdx.x;
    if (t == 0) bad = 0;
    __syncthreads();
    const float* l = logits + static_cast<std::size_t>(blockIdx.x) * ld;
    float bv = -__builtin_inff();
    std::uint32_t bi = 0xFFFFFFFFU;
    bool finite = true;
    for (unsigned j = t; j < vn; j += kArgThreads) {
        const float v = l[j];
        finite &= __builtin_isfinite(v);
        if (better(v, j, bv, bi)) {
            bv = v;
            bi = j;
        }
    }
    if (!finite) atomicOr(&bad, 1U);
    // Shared tree (R2-07); ordering stays float-compare so the transmitted
    // value bits are unchanged for non-finite rows.
    const Best best = strix::argmax::block_reduce<kArgThreads>(
        sb, t, Best{bv, bi}, [](const Best& a, const Best& b) { return better(b.value, b.id, a.value, a.id) ? b : a; });
    if (t == 0) {
        tx[2 * blockIdx.x] = __float_as_uint(best.value);
        tx[2 * blockIdx.x + 1] = bad != 0 || best.id == 0xFFFFFFFFU ? 0xFFFFFFFFU : id0 + best.id;
    }
    publish(target, gridDim.x);
}

[[maybe_unused]] __global__ void argmax_pick_kernel(const std::uint32_t* c0, const std::uint32_t* c1, unsigned rows, std::uint32_t* out) {
    const unsigned r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const std::uint32_t i0 = c0[2 * r + 1], i1 = c1[2 * r + 1];
    const float v0 = __uint_as_float(c0[2 * r]), v1 = __uint_as_float(c1[2 * r]);
    out[r] = i0 == 0xFFFFFFFFU || i1 == 0xFFFFFFFFU ? 0xFFFFFFFFU : v1 > v0 ? i1 : i0;  // tie: rank 0 (lower ids)
}

[[maybe_unused]] std::uint64_t env_u64(const char* name, std::uint64_t fallback) {
    const char* v = std::getenv(name);
    return v != nullptr && *v != '\0' ? std::stoull(v) : fallback;
}

}  // namespace

// One thread computing CRC-32C of payloads in submission order (deferred CRCs). It spins while work may come and naps
// after an idle period; submit() and wait() are lock-free (one producer: the progress thread).
template<class Traits>
class Channel<Traits>::CrcWorker {
public:
    explicit CrcWorker(int cpu) : thread_([this] { run(); }) {
        if (cpu >= 0) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpu, &set);
            pthread_setaffinity_np(thread_.native_handle(), sizeof(set), &set);
        }
    }
    ~CrcWorker() {
        stop_ = true;
        thread_.join();
    }
    std::uint64_t submit(const std::byte* p, std::size_t n) {
        const std::uint64_t t = submitted_.load(std::memory_order_relaxed) + 1;
        while (t - completed_.load(std::memory_order_acquire) > kRing) _mm_pause();
        Task& k = ring_[t % kRing];
        k.p = p;
        k.n = n;
        submitted_.store(t, std::memory_order_release);
        return t;
    }
    std::uint32_t wait(std::uint64_t ticket) {
        while (completed_.load(std::memory_order_acquire) < ticket) _mm_pause();
        return ring_[ticket % kRing].crc;
    }

private:
    static constexpr std::uint64_t kRing = 8;
    struct Task {
        const std::byte* p = nullptr;
        std::size_t n = 0;
        std::uint32_t crc = 0;
    };
    void run() {
        auto last = Clock::now();
        while (!stop_.load(std::memory_order_relaxed)) {
            const std::uint64_t c = completed_.load(std::memory_order_relaxed);
            if (c < submitted_.load(std::memory_order_acquire)) {
                Task& k = ring_[(c + 1) % kRing];
                k.crc = dist::crc32c(std::span<const std::byte>(k.p, k.n));
                completed_.store(c + 1, std::memory_order_release);
                last = Clock::now();
            } else if (Clock::now() - last < std::chrono::milliseconds(3)) {
                _mm_pause();
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
            }
        }
    }
    Task ring_[kRing];
    std::atomic<std::uint64_t> submitted_{0}, completed_{0};
    std::atomic<bool> stop_{false};
    std::thread thread_;
};

template<class Traits>
void ChannelOptions<Traits>::from_env() {
    if (const char* v = std::getenv("STRIX_XPORT_SMALL_SPIN"); v != nullptr && *v != '\0') {
        small_spin = SmallProgress::parse(v);
    }
    depth = static_cast<unsigned>(env_u64((std::string(Traits::kEnvironment) + "DEPTH").c_str(), depth));
    {   // Exchange row fold: messages of at most <ENV>FOLD rows may be folded by the progress thread (0 = off). The
        // default is the engine's measured crossover (Traits::kFoldRows), absent = off.
        std::uint64_t rows = 0;
        if constexpr (requires { Traits::kFoldRows; }) rows = Traits::kFoldRows;
        rows = env_u64((std::string(Traits::kEnvironment) + "FOLD").c_str(), rows);
        fold_capacity = std::min<std::size_t>(static_cast<std::size_t>(rows) * Traits::kWidth * sizeof(typename Traits::Payload), capacity);
    }
    if (const char* v = std::getenv((std::string(Traits::kEnvironment) + "PUBLISH").c_str()); v != nullptr && *v != '\0') {
        const std::string s(v);
        if (s == "kernel") publish = Publish::kernel;
        else if (s == "write_value") publish = Publish::write_value;
        else throw std::invalid_argument("STRIX_GLMX_PUBLISH must be kernel or write_value");
    }
    if (const char* v = std::getenv((std::string(Traits::kEnvironment) + "CRC").c_str()); v != nullptr && *v != '\0') {
        const std::string c(v);
        if (c == "0") crc = false;
        else if (c == "1") crc = true, crc_deferred = false;
        else if (c == "deferred") crc = true, crc_deferred = true;
        else throw std::invalid_argument("STRIX_GLMX_CRC must be 0, 1 or deferred");
    }
    auto memory = [](const char* name, Memory fallback) {
        const char* v = std::getenv(name);
        if (v == nullptr || *v == '\0') return fallback;
        const std::string m(v);
        const int kind = Traits::memory_kind(m);
        if (kind >= 0) return static_cast<Memory>(kind);
        throw std::invalid_argument(std::string(name) + " must be " + Traits::kMemoryChoices);
    };
    tx_memory = memory((std::string(Traits::kEnvironment) + "TX_MEM").c_str(), tx_memory);
    rx_memory = memory((std::string(Traits::kEnvironment) + "RX_MEM").c_str(), rx_memory);
    if (const char* v = std::getenv((std::string(Traits::kEnvironment) + "PROGRESS_CPU").c_str()); v != nullptr && *v != '\0') progress_cpu = std::atoi(v);
    if (const char* v = std::getenv((std::string(Traits::kEnvironment) + "CRC_CPU").c_str()); v != nullptr && *v != '\0') crc_cpu = std::atoi(v);
    test_corrupt_sequence = env_u64((std::string(Traits::kEnvironment) + "TEST_CORRUPT").c_str(), test_corrupt_sequence);
    idle_spin_us = static_cast<unsigned>(env_u64((std::string(Traits::kEnvironment) + "IDLE_SPIN_US").c_str(), idle_spin_us));
    timeout = std::chrono::milliseconds(env_u64((std::string(Traits::kEnvironment) + "TIMEOUT_MS").c_str(), static_cast<std::uint64_t>(timeout.count())));
}

template<class Traits>
Channel<Traits>::Channel(std::unique_ptr<dist::Transport> owned, Options options)
    : owned_(std::move(owned)), transport_(owned_.get()), options_(options) {
    init();
}

template<class Traits>
Channel<Traits>::Channel(dist::Transport& borrowed, Options options) : transport_(&borrowed), options_(options) { init(); }

template<class Traits>
void Channel<Traits>::init() {
    if (transport_ == nullptr) throw std::invalid_argument("glmxfer: no transport");
    if (options_.rank > 1) throw std::invalid_argument("glmxfer: rank must be 0 or 1");
    if (options_.capacity == 0 || options_.capacity > 0xFFFFFFFFULL) throw std::invalid_argument("glmxfer: capacity");
    if (options_.depth < 2 || options_.depth > 64) throw std::invalid_argument("glmxfer: depth must be 2..64");
    stride_ = (options_.capacity + sizeof(Trailer) + 4095) & ~std::size_t{4095};
    const std::size_t total = stride_ * options_.depth;
    const unsigned flags = hipHostMallocMapped | hipHostMallocCoherent;
    auto slots = [&](Memory m, std::byte** host, std::byte** dev, void** alloc, const char* what) {
        if (m != Memory::host_coherent) {
            hip_check(hipExtMallocWithFlags(alloc, total, m == Memory::device_fine ? hipDeviceMallocFinegrained : hipDeviceMallocUncached), what);
            *host = *dev = static_cast<std::byte*>(*alloc);
        } else {
            hip_check(hipHostMalloc(alloc, total, flags), what);
            *host = static_cast<std::byte*>(*alloc);
            hip_check(hipHostGetDevicePointer(reinterpret_cast<void**>(dev), *host, 0), what);
        }
    };
    slots(options_.tx_memory, &tx_host_, &tx_dev_, &tx_alloc_, "TX slots");
    slots(options_.rx_memory, &rx_host_, &rx_dev_, &rx_alloc_, "RX slots");
    hip_check(hipHostMalloc(reinterpret_cast<void**>(&flags_host_), 4096, flags), "flags");
    hip_check(hipHostGetDevicePointer(reinterpret_cast<void**>(&flags_dev_), flags_host_, 0), "flags");
    std::memset(tx_host_, 0, total);
    std::memset(rx_host_, 0, total);
    std::memset(flags_host_, 0, 4096);
    hip_check(hipMalloc(reinterpret_cast<void**>(&counter_dev_), 64), "counter");
    hip_check(hipMemset(counter_dev_, 0, 64), "counter");
    hip_check(hipMalloc(reinterpret_cast<void**>(&gate_dev_), 64), "gate");
    hip_check(hipMemset(gate_dev_, 0, 64), "gate");
    if (options_.fold_capacity != 0) {
        fx_stride_ = (options_.fold_capacity + 4095) & ~std::size_t{4095};
        hip_check(hipHostMalloc(reinterpret_cast<void**>(&fx_host_), fx_stride_ * options_.depth, flags), "fold slots");
        hip_check(hipHostGetDevicePointer(reinterpret_cast<void**>(&fx_dev_), fx_host_, 0), "fold slots");
        std::memset(fx_host_, 0, fx_stride_ * options_.depth);
        folded_.assign(options_.depth, 0);
    }
    if (const char* v = std::getenv((std::string(Traits::kEnvironment) + "RX_TRACE").c_str()); v != nullptr && *v != '\0') {
        rx_trace_path_ = v;
        rx_trace_.reserve(std::size_t{1} << 20);
    }
    if (options_.crc && options_.crc_deferred) crc_worker_ = std::make_unique<CrcWorker>(options_.crc_cpu);
    thread_ = std::thread([this] { run(); });
    if (options_.progress_cpu >= 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(options_.progress_cpu, &set);
        pthread_setaffinity_np(thread_.native_handle(), sizeof(set), &set);
    }
}

template<class Traits>
Channel<Traits>::~Channel() {
    if (done_.load() < posted_.load() && transport_ != nullptr) transport_->abort("glmxfer: closing with exchanges in flight");
    stop_ = true;
    {
        std::lock_guard<std::mutex> g(mutex_);
        wake_.notify_all();
    }
    if (thread_.joinable()) thread_.join();
    if (flags_host_ != nullptr) {
        __atomic_store_n(&flags_host_[kRxFlag], ~std::uint64_t{0}, __ATOMIC_RELEASE);  // release any waiting stream
        (void)hipStreamSynchronize(options_.stream);
    }
    crc_worker_.reset();
    (void)hipFree(counter_dev_);
    if (!rx_trace_path_.empty()) {
        if (std::FILE* f = std::fopen((rx_trace_path_ + "." + std::to_string(options_.rank)).c_str(), "w")) {
            for (const auto& r : rx_trace_)
                std::fprintf(f, "%llu %llu %llu %llu %llu %llu\n", static_cast<unsigned long long>(r.sequence),
                             static_cast<unsigned long long>(r.kind), static_cast<unsigned long long>(r.bytes),
                             static_cast<unsigned long long>(r.tx_seen_ns), static_cast<unsigned long long>(r.wire_done_ns),
                             static_cast<unsigned long long>(r.rx_published_ns));
            std::fclose(f);
        }
    }
    if (folds_ != 0) std::fprintf(stderr, "xport row_fold rank=%u folded=%llu of %llu exchanges\n", options_.rank,
                                  static_cast<unsigned long long>(folds_), static_cast<unsigned long long>(stats_.exchanges));
    if (fx_host_ != nullptr) (void)hipHostFree(fx_host_);
    (void)hipFree(gate_dev_);
    (void)hipHostFree(flags_host_);
    if (options_.rx_memory != Memory::host_coherent) (void)hipFree(rx_alloc_);
    else (void)hipHostFree(rx_alloc_);
    if (options_.tx_memory != Memory::host_coherent) (void)hipFree(tx_alloc_);
    else (void)hipHostFree(tx_alloc_);
}

template<class Traits>
std::byte* Channel<Traits>::tx_slot(std::uint64_t sequence) const { return tx_host_ + ((sequence - 1) % options_.depth) * stride_; }
template<class Traits>
std::byte* Channel<Traits>::rx_slot(std::uint64_t sequence) const { return rx_host_ + ((sequence - 1) % options_.depth) * stride_; }

template<class Traits>
void* Channel<Traits>::tx() {
    const std::uint64_t s = next_ + 1;
    if (s > received_ + options_.depth) throw std::logic_error("glmxfer: tx() of a sequence whose slot is still in use (recv() of s - depth missing)");
    tx_taken_ = s;
    return tx_dev_ + (tx_slot(s) - tx_host_);
}

template<class Traits>
ProducerTarget Channel<Traits>::producer_target() {
    return ProducerTarget{flags_dev_ + kTxFlag, counter_dev_, next_ + 1};
}

template<class Traits>
void Channel<Traits>::send(std::size_t bytes, const Tag& tag, Publish how) {
    check();
    if (bytes == 0 || bytes > options_.capacity) throw std::invalid_argument("glmxfer: message size outside 1..capacity");
    const std::uint64_t s = next_ + 1;
    if (s > received_ + options_.depth) throw std::logic_error("glmxfer: send() of a sequence whose slot is still in use");
    if (how == Publish::kernel) {
        hipLaunchKernelGGL(publish_kernel, dim3(1), dim3(32), 0, options_.stream, flags_dev_ + kTxFlag, s);
        hip_check(hipGetLastError(), "publish kernel");
    } else if (how == Publish::write_value) {
        hip_check(hipStreamWriteValue64(options_.stream, flags_dev_ + kTxFlag, s, 0), "publish write");
    }
    next_ = s;
    tail_deferred_ = options_.crc && options_.crc_deferred && bytes > kInlineCrcMax;
    Job job{s, bytes, tag, false};
    if (fold_next_ != RowFold::none) {
        if (!can_fold(bytes)) throw std::invalid_argument("glmxfer: fold of a message outside the fold capacity");
        job.fold = fold_next_;
        fold_next_ = RowFold::none;
    }
    if (!folded_.empty()) folded_[(s - 1) % options_.depth] = job.fold != RowFold::none ? 1 : 0;
    post(job);
}

template<class Traits>
void Channel<Traits>::post(const Job& job) {
    slot_wait_.block_while(static_cast<std::uint32_t>(job.tag.kind), [&] {
        return jobs_posted_ - done_.load(std::memory_order_acquire) >= kQueue;
    }, [&] {
        check();
        _mm_pause();
    });
    jobs_[jobs_posted_ % kQueue] = job;
    ++jobs_posted_;
    posted_.store(jobs_posted_, std::memory_order_seq_cst);
    if (sleeping_.load(std::memory_order_seq_cst)) {
        std::lock_guard<std::mutex> g(mutex_);
        wake_.notify_one();
    }
}

template<class Traits>
void Channel<Traits>::send_copy(const void* local, std::size_t bytes, const Tag& tag) {
    auto* dst = static_cast<std::byte*>(tx());
    const unsigned blocks = static_cast<unsigned>(std::clamp<std::size_t>((bytes + 16 * kCopyThreads - 1) / (16 * kCopyThreads), 1, 64));
    hipLaunchKernelGGL(copy_publish_kernel, dim3(blocks), dim3(kCopyThreads), 0, options_.stream, dst,
                       static_cast<const std::byte*>(local), bytes, producer_target());
    hip_check(hipGetLastError(), "copy kernel");
    send(bytes, tag, Publish::producer);
}

template<class Traits>
void Channel<Traits>::send_host(const void* local, void* remote, std::size_t bytes, const Tag& tag) {
    check();
    if (!local || !remote || bytes == 0 || bytes > options_.capacity)
        throw std::invalid_argument("glmxfer: direct host payload outside 1..capacity");
    const auto s = next_ + 1;
    if (s > received_ + options_.depth) throw std::logic_error("glmxfer: direct host slot still in use");
    hip_check(hipStreamWriteValue64(options_.stream, flags_dev_ + kTxFlag, s, 0), "direct host producer ready");
    next_ = s;
    tail_deferred_ = options_.crc && options_.crc_deferred && bytes > kInlineCrcMax;
    if (fold_next_ != RowFold::none) throw std::invalid_argument("glmxfer: direct host messages are not folded");
    if (!folded_.empty()) folded_[(s - 1) % options_.depth] = 0;
    post(Job{s, bytes, tag, false, static_cast<const std::byte*>(local), static_cast<std::byte*>(remote)});
}

template<class Traits>
void argmax_exchange(Channel<Traits>& x, const float* logits, unsigned rows, unsigned vn, unsigned ld, std::uint32_t id0, std::uint64_t step,
                     std::uint32_t* out_ids) {
    if (rows == 0 || vn == 0 || ld < vn) throw std::invalid_argument("glmxfer argmax: bad shape");
    auto* tx = static_cast<std::uint32_t*>(x.tx());
    const hipStream_t s = x.options().stream;
    hipLaunchKernelGGL(argmax_candidate_kernel, dim3(rows), dim3(kArgThreads), 0, s, logits, vn, ld, id0, tx, x.producer_target());
    hip_check(hipGetLastError(), "argmax candidate kernel");
    x.send(std::size_t{rows} * 8, typename Traits::Tag{Traits::Kind::argmax, Traits::kHeadLayer, step}, Publish::producer);
    const auto* peer = static_cast<const std::uint32_t*>(x.recv());
    const auto* local = static_cast<const std::uint32_t*>(x.last_local());
    const bool r0 = x.options().rank == 0;
    hipLaunchKernelGGL(argmax_pick_kernel, dim3((rows + 63) / 64), dim3(64), 0, s, r0 ? local : peer, r0 ? peer : local, rows, out_ids);
    hip_check(hipGetLastError(), "argmax pick kernel");
}

template<class Traits>
const void* Channel<Traits>::recv() {
    const std::uint64_t s = received_ + 1;
    if (s > next_) throw std::logic_error("glmxfer: recv() without a matching send()");
    hip_check(hipStreamWaitValue64(options_.stream, flags_dev_ + kRxFlag, s, hipStreamWaitValueGte, ~std::uint64_t{0}),
              "wait for the peer's message");
    received_ = s;
    last_local_ = tx_dev_ + (tx_slot(s) - tx_host_);
    last_folded_ = !folded_.empty() && folded_[(s - 1) % options_.depth] != 0 ? fx_dev_ + ((s - 1) % options_.depth) * fx_stride_ : nullptr;
    return rx_dev_ + (rx_slot(s) - rx_host_);
}

template<class Traits>
RxView Channel<Traits>::recv_view() {
    const std::uint64_t s = received_ + 1;
    if (s > next_) throw std::logic_error("glmxfer: recv_view() without a matching send()");
    received_ = s;
    last_local_ = tx_dev_ + (tx_slot(s) - tx_host_);
    last_folded_ = !folded_.empty() && folded_[(s - 1) % options_.depth] != 0 ? fx_dev_ + ((s - 1) % options_.depth) * fx_stride_ : nullptr;
    return RxView{rx_dev_ + (rx_slot(s) - rx_host_), flags_dev_ + kRxFlag, s, gate_dev_};
}

template<class Traits>
void Channel<Traits>::drain() {
    if (tail_deferred_) {  // the last payload's CRC has not travelled yet: one host-only seal message (both ranks)
        tail_deferred_ = false;
        post(Job{next_, 0, Tag{Kind::control, 0xFFFFFFFFU, 0}, true});
    }
    const auto deadline = Clock::now() + options_.timeout;
    while (done_.load(std::memory_order_acquire) < posted_.load(std::memory_order_acquire) && !failed_.load()) {
        if (Clock::now() > deadline) {
            transport_->abort("glmxfer: drain deadline");
            throw std::runtime_error("glmxfer: drain: exchanges still in flight after the deadline");
        }
        _mm_pause();
    }
    check();
}

template<class Traits>
void Channel<Traits>::check() const {
    if (!failed_.load(std::memory_order_acquire)) return;
    std::lock_guard<std::mutex> g(failure_mutex_);
    throw std::runtime_error("glmxfer: exchange failed: " + failure_);
}

template<class Traits>
Stats Channel<Traits>::stats() const { return stats_; }

template<class Traits>
std::string Channel<Traits>::name() const {
    const char* pub = options_.publish == Publish::kernel ? "kernel" : options_.publish == Publish::write_value ? "write_value" : "producer";
    return "glmxfer(" + transport_->name() + ", depth " + std::to_string(options_.depth) + ", publish " + pub +
           ", crc " + Traits::crc_description(options_.crc, options_.crc_deferred) + ")";
}

template<class Traits>
void Channel<Traits>::fail(const std::string& why) {
    {
        std::lock_guard<std::mutex> g(failure_mutex_);
        if (failure_.empty()) failure_ = why;
    }
    failed_.store(true, std::memory_order_release);
    __atomic_store_n(&flags_host_[kRxFlag], ~std::uint64_t{0}, __ATOMIC_RELEASE);
    transport_->abort(why);
    throw std::runtime_error(why);
}

template<class Traits>
void Channel<Traits>::process(const Job& job) {
    const auto t0 = Clock::now();
    const auto deadline = t0 + options_.timeout;
    unsigned n = 0;
    while (!job.seal && __atomic_load_n(&flags_host_[kTxFlag], __ATOMIC_ACQUIRE) < job.sequence) {
        if (stop_.load(std::memory_order_relaxed)) fail("glmxfer: stopped while waiting for a producer");
        if ((++n & 1023U) == 0 && Clock::now() > deadline)
            fail("glmxfer: producer of sequence " + std::to_string(job.sequence) + " not published within the timeout");
        _mm_pause();
    }
    const auto t1 = Clock::now();
    const std::byte* tx = job.direct_tx ? job.direct_tx : (job.seal ? seal_tx_ : tx_slot(job.sequence));
    std::byte* rx = job.direct_rx ? job.direct_rx : (job.seal ? seal_rx_ : rx_slot(job.sequence));
    const bool deferred = options_.crc && options_.crc_deferred && job.bytes > kInlineCrcMax;
    std::uint64_t tx_ticket = 0;
    if (deferred) tx_ticket = crc_worker_->submit(tx, job.bytes);  // overlaps the wire below
    const auto t = make_trailer<Traits>(job.tag, options_.epoch, options_.plan, job.sequence,
        job.bytes, options_.rank, deferred,
        !deferred && options_.crc ? dist::crc32c(std::span<const std::byte>(tx, job.bytes)) : 0U,
        prev_deferred_, prev_tx_crc_);
    Trailer p;
    try {
        peer_wait_.measure(static_cast<std::uint32_t>(job.tag.kind), [&] {
            if (job.direct_rx) exchange_frame_direct(*transport_, tx, rx, job.bytes, t, p);
            else exchange_frame(*transport_, job.seal ? seal_tx_ : tx_slot(job.sequence), rx, job.bytes, t);
        });
    } catch (const std::exception& e) {
        fail(std::string("glmxfer: transport: ") + e.what());
    }
    const auto t2 = Clock::now();
    if (!job.seal && job.sequence == options_.test_corrupt_sequence && job.bytes > 0) rx[0] ^= std::byte{1};
    if (!job.direct_rx) std::memcpy(&p, rx + job.bytes, sizeof(p));
    if (p.magic != kTrailerMagic || p.epoch != t.epoch || p.plan != t.plan || p.sequence != t.sequence || p.step != t.step ||
        p.bytes != t.bytes || p.kind != t.kind || p.layer != t.layer || p.rank != 1U - options_.rank || p.flags != t.flags)
        fail("glmxfer: peer message identity differs at sequence " + std::to_string(job.sequence) + " (peer seq " +
             std::to_string(p.sequence) + " kind " + std::to_string(p.kind) + " layer " + std::to_string(p.layer) + " bytes " +
             std::to_string(p.bytes) + ")");
    if (prev_deferred_ && p.prev_crc32c != crc_worker_->wait(prev_rx_ticket_))
        fail("glmxfer: payload CRC mismatch at sequence " + std::to_string(prev_sequence_) + " (deferred check)");
    if (options_.crc && !deferred && p.crc32c != dist::crc32c(std::span<const std::byte>(rx, job.bytes)))
        fail("glmxfer: payload CRC mismatch at sequence " + std::to_string(job.sequence));
    if (deferred) prev_tx_crc_ = crc_worker_->wait(tx_ticket);  // TX slot fully read before the stream may reuse it
    prev_deferred_ = deferred;
    if (job.seal) return;
    if (job.fold != RowFold::none) {  // rank 0's partial first on both ranks; the consumer reads it only after the flag
        ++folds_;
        fold_rows(job.fold, fx_host_ + ((job.sequence - 1) % options_.depth) * fx_stride_, options_.rank == 0 ? tx : rx,
                  options_.rank == 0 ? rx : tx, job.bytes);
    }
    __atomic_store_n(&flags_host_[kRxFlag], job.sequence, __ATOMIC_RELEASE);
    if (deferred) {
        prev_rx_ticket_ = crc_worker_->submit(rx, job.bytes);  // after the release: off the critical path
        prev_sequence_ = job.sequence;
    }
    const auto t3 = Clock::now();
    stats_.exchanges += 1;
    stats_.bytes += job.bytes;
    stats_.producer_wait += seconds(t0, t1);
    stats_.wire += seconds(t1, t2);
    stats_.check += seconds(t2, t3);
    if (!rx_trace_path_.empty() && rx_trace_.size() < rx_trace_.capacity()) {
        auto ns = [](Clock::time_point t) {
            return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count());
        };
        rx_trace_.push_back(RxTrace{job.sequence, static_cast<std::uint64_t>(job.tag.kind), job.bytes, ns(t1), ns(t2), ns(t3)});
    }
}

template<class Traits>
void Channel<Traits>::run() {
    auto last = Clock::now();
    const auto idle = std::chrono::microseconds(options_.idle_spin_us);
    try {
        while (!stop_.load(std::memory_order_relaxed)) {
            const std::uint64_t d = done_.load(std::memory_order_relaxed);
            if (d < posted_.load(std::memory_order_acquire)) {
                const auto& job = jobs_[d % kQueue];
                // Capture before publishing done_: the producer may then reuse this slot.
                const bool small = options_.small_spin && !job.seal &&
                    SmallProgress::eligible(job.bytes + sizeof(typename Traits::Trailer));
                process(job);
                done_.store(d + 1, std::memory_order_release);
                last = Clock::now();
                if (small) {
                    spin_small_mailbox([&] {
                        return stop_.load(std::memory_order_relaxed) ||
                            done_.load(std::memory_order_relaxed) < posted_.load(std::memory_order_acquire);
                    }, [] { _mm_pause(); }, std::min(idle, SmallProgress::mailbox_spin));
                    last = Clock::now() - idle; // budget exhausted, or a job is already in the mailbox
                }
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

}  // namespace strix::xport
