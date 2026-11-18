// Tiered expert residency, CUDA side. See include/strix/cuda/tiered_experts.hpp for the protocol.
#include "strix/cuda/tiered_experts.hpp"
#include "strix/residency/expert_residency.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

namespace strix::cuda::tiered {
namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("tiered experts: ") + what + ": " + cudaGetErrorString(e));
}
constexpr std::uint64_t align_up(std::uint64_t v, std::uint64_t a) { return (v + a - 1U) / a * a; }

__device__ __forceinline__ std::uint64_t global_timer() {
    std::uint64_t t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
__device__ __forceinline__ std::uint64_t load_u64_cv(const volatile std::uint64_t* p) { return *p; }
__device__ __forceinline__ std::uint32_t load_u32_cv(const std::uint32_t* p) {
    return __ldcv(p);
}

// One block per row (live and padding rows).
__global__ __launch_bounds__(256) void publish_kernel(TierDev d, std::uint32_t layer, std::uint32_t rows,
                                                      const std::uint32_t* routes, std::uint32_t* slot_routes,
                                                      const std::int32_t* table, std::uint32_t zero_slot,
                                                      const std::int8_t* codes, const float* deltas, const std::int32_t* sums,
                                                      std::uint32_t* counts) {
    // The preceding wait can time out while the CPU still owns these inputs.
    // dead is stable for the whole launch (all exchange kernels use one stream).
    if (*reinterpret_cast<volatile std::uint32_t*>(&d.state->dead)) return;
    __shared__ std::uint32_t any_miss, last_block;
    const std::uint32_t row = blockIdx.x, t = threadIdx.x;
    const bool live = row < rows;
    bool miss = false;
    std::uint32_t id = 0;
    if (t < d.top_k) {
        id = routes[std::uint64_t{row} * d.top_k + t];
        std::uint32_t out = zero_slot;
        if (live) {
            if (id >= d.experts) {
                out = id;  // invalid ids stay invalid: the routed kernels fail closed exactly as without tiering
            } else {
                if (counts) atomicAdd(&counts[id], 1U);
                const std::int32_t s = table[id];
                if (s >= 0) out = static_cast<std::uint32_t>(s); else miss = true;
            }
        }
        slot_routes[std::uint64_t{row} * d.top_k + t] = out;
    }
    if (t < 32U) {
        const unsigned bal = __ballot_sync(0xffffffffU, miss);
        const unsigned n = __popc(bal);
        unsigned base = 0;
        if (t == 0U && n) base = atomicAdd(&d.state->n_miss, n);
        base = __shfl_sync(0xffffffffU, base, 0);
        if (miss) {
            Miss m;
            m.row = row; m.slot = t; m.expert = id; m.pad = 0;
            d.miss[base + __popc(bal & ((1U << t) - 1U))] = m;
            __threadfence_system();
        }
        if (t == 0U) any_miss = n != 0U;
    }
    __syncthreads();
    if (any_miss) {
        const std::uint32_t words = d.hidden / 4U, groups = d.hidden / 32U;
        const auto* src = reinterpret_cast<const std::uint32_t*>(codes + std::uint64_t{row} * d.hidden);
        auto* dst = reinterpret_cast<std::uint32_t*>(d.codes + std::uint64_t{row} * d.hidden);
        for (std::uint32_t i = t; i < words; i += blockDim.x) dst[i] = src[i];
        for (std::uint32_t i = t; i < groups; i += blockDim.x) {
            d.deltas[std::uint64_t{row} * groups + i] = deltas[std::uint64_t{row} * groups + i];
            d.sums[std::uint64_t{row} * groups + i] = sums[std::uint64_t{row} * groups + i];
        }
        __threadfence_system();
    }
    __syncthreads();
    if (t == 0U) {
        __threadfence();
        last_block = atomicAdd(&d.state->done, 1U) + 1U == gridDim.x;
    }
    __syncthreads();
    if (!last_block || t != 0U) return;
    __threadfence();
    const std::uint32_t n = *reinterpret_cast<volatile std::uint32_t*>(&d.state->n_miss);
    d.state->last_n = n;
    if (n && !d.state->dead) {
        d.header[0] = layer;
        d.header[1] = rows;
        d.header[2] = n;
        __threadfence_system();
        const std::uint64_t seq = d.state->seq + 1U;
        d.state->seq = seq;
        *d.seq_published = seq;
        __threadfence_system();
    }
    d.state->n_miss = 0U;
    d.state->done = 0U;
    __threadfence();
}

__global__ __launch_bounds__(256) void wait_kernel(TierDev d, std::uint32_t rows, std::uint32_t* status,
                                                   std::uint32_t status_stride_words, std::uint32_t* summary) {
    __shared__ std::uint32_t fail;
    const std::uint32_t n = *reinterpret_cast<volatile std::uint32_t*>(&d.state->last_n);
    if (n == 0U) return;  // block-uniform: nothing was signalled
    if (threadIdx.x == 0U) {
        fail = 0U;
        if (*reinterpret_cast<volatile std::uint32_t*>(&d.state->dead)) {
            fail = 1U;
        } else {
            const std::uint64_t target = d.state->seq, t0 = global_timer();
            while (load_u64_cv(d.seq_done) != target) {
                if (global_timer() - t0 > d.timeout_ns) { fail = 1U; *reinterpret_cast<volatile std::uint32_t*>(&d.state->dead) = 1U; break; }
                __nanosleep(100);
            }
            if (!fail && load_u32_cv(d.header + 3) != 0U) fail = 1U;
        }
        if (fail) atomicOr(summary, 8U);
    }
    __syncthreads();
    for (std::uint32_t r = threadIdx.x; r < rows; r += blockDim.x) {
        // host bit kNonfinite (1) = the card's projection_nonfinite (8, qwen_moe.cu:404)
        const std::uint32_t bits = (fail || load_u32_cv(d.status + r)) ? 8U : 0U;
        if (bits) {
            atomicOr(status + std::uint64_t{r} * status_stride_words, bits);
            atomicOr(summary, 8U);  // the card's projection_nonfinite rule: row status and job summary carry the bit
        }
    }
}

// Block b answers miss b (blocks beyond last_n exit).
__global__ __launch_bounds__(256) void gather_kernel(TierDev d, float* rawdots, std::uint64_t stride_floats) {
    const std::uint32_t n = *reinterpret_cast<volatile std::uint32_t*>(&d.state->last_n);
    if (blockIdx.x >= n || *reinterpret_cast<volatile std::uint32_t*>(&d.state->dead)) return;
    const std::uint32_t* e = reinterpret_cast<const std::uint32_t*>(d.miss) + std::uint64_t{blockIdx.x} * 4U;
    const std::uint32_t row = load_u32_cv(e), slot = load_u32_cv(e + 1);
    const std::uint64_t at = std::uint64_t{row} * d.top_k * d.hidden + std::uint64_t{slot} * d.hidden;
    const float4* src = reinterpret_cast<const float4*>(d.rawdots + at);
    float4* dst = reinterpret_cast<float4*>(rawdots + std::uint64_t{row} * stride_floats + std::uint64_t{slot} * d.hidden);
    for (std::uint32_t i = threadIdx.x; i < d.hidden / 4U; i += blockDim.x) dst[i] = __ldcv(src + i);
}

}  // namespace

namespace {
thread_local std::uint64_t allocation_initial_free = 0, allocation_budget = 0;
}
AllocationBudget::AllocationBudget(std::uint64_t initial_free, std::uint64_t budget)
    : previous_free_(allocation_initial_free), previous_budget_(allocation_budget) {
    allocation_initial_free = initial_free;
    allocation_budget = budget;
}
AllocationBudget::~AllocationBudget() {
    allocation_initial_free = previous_free_;
    allocation_budget = previous_budget_;
}
void require_budget(std::uint64_t bytes, const char* what) {
    if (!allocation_budget) return;
    std::size_t available = 0, total = 0;
    ck(cudaMemGetInfo(&available, &total), "allocation memory info");
    if (!residency::allocation_fits(allocation_budget, allocation_initial_free, available, bytes))
        throw std::runtime_error(std::string("residency allocation exceeds VRAM budget: ") + what);
}

Exchange::Exchange(const TierLayout& layout) : layout_(layout) {
    if (layout.hidden % 128U || !layout.top_k || layout.top_k > 32U || !layout.max_rows || !layout.experts)
        throw std::invalid_argument("tiered exchange: bad layout");
    const std::uint64_t groups = layout.hidden / 32U, rows = layout.max_rows;
    std::uint64_t at = 0;
    const auto place = [&](std::uint64_t bytes) { const auto o = at; at = align_up(at + bytes, 256); return o; };
    const auto o_pub = place(8), o_done = place(8) , o_header = place(16);
    const auto o_miss = place(rows * layout.top_k * sizeof(Miss));
    const auto o_codes = place(rows * layout.hidden), o_deltas = place(rows * groups * 4), o_sums = place(rows * groups * 4);
    const auto o_status = place(rows * 4);
    const auto o_raw = place(rows * layout.top_k * layout.hidden * 4);
    host_bytes_ = at;
    try {
        ck(cudaHostAlloc(reinterpret_cast<void**>(&host_base_), host_bytes_, cudaHostAllocMapped | cudaHostAllocPortable), "exchange alloc");
        std::memset(host_base_, 0, host_bytes_);
        std::uint8_t* dbase = nullptr;
        ck(cudaHostGetDevicePointer(reinterpret_cast<void**>(&dbase), host_base_, 0), "exchange device pointer");
        host_published_ = reinterpret_cast<std::uint64_t*>(host_base_ + o_pub);
        host_done_ = reinterpret_cast<std::uint64_t*>(host_base_ + o_done);
        host_header_ = reinterpret_cast<std::uint32_t*>(host_base_ + o_header);
        host_miss_ = reinterpret_cast<Miss*>(host_base_ + o_miss);
        host_codes_ = reinterpret_cast<std::int8_t*>(host_base_ + o_codes);
        host_deltas_ = reinterpret_cast<float*>(host_base_ + o_deltas);
        host_sums_ = reinterpret_cast<std::int32_t*>(host_base_ + o_sums);
        host_status_ = reinterpret_cast<std::uint32_t*>(host_base_ + o_status);
        host_rawdots_ = reinterpret_cast<float*>(host_base_ + o_raw);
        require_budget(sizeof(DevState), "exchange state");
        ck(cudaMalloc(reinterpret_cast<void**>(&state_), sizeof(DevState)), "state alloc");
        ck(cudaMemset(state_, 0, sizeof(DevState)), "state clear");
        dev_.seq_published = reinterpret_cast<volatile std::uint64_t*>(dbase + o_pub);
        dev_.seq_done = reinterpret_cast<volatile std::uint64_t*>(dbase + o_done);
        dev_.header = reinterpret_cast<std::uint32_t*>(dbase + o_header);
        dev_.miss = reinterpret_cast<Miss*>(dbase + o_miss);
        dev_.codes = reinterpret_cast<std::int8_t*>(dbase + o_codes);
        dev_.deltas = reinterpret_cast<float*>(dbase + o_deltas);
        dev_.sums = reinterpret_cast<std::int32_t*>(dbase + o_sums);
        dev_.status = reinterpret_cast<std::uint32_t*>(dbase + o_status);
        dev_.rawdots = reinterpret_cast<float*>(dbase + o_raw);
        dev_.state = state_;
        dev_.hidden = layout.hidden;
        dev_.top_k = layout.top_k;
        dev_.experts = layout.experts;
        dev_.max_rows = layout.max_rows;
    } catch (...) {
        if (state_) cudaFree(state_);
        if (host_base_) cudaFreeHost(host_base_);
        throw;
    }
}

Exchange::~Exchange() {
    if (state_) cudaFree(state_);
    if (host_base_) cudaFreeHost(host_base_);
}

void enqueue_publish(const TierDev& d, std::uint32_t layer, std::uint32_t rows, std::uint32_t pad, const std::uint32_t* routes,
                     std::uint32_t* slot_routes, const std::int32_t* table, std::uint32_t zero_slot, const std::int8_t* codes,
                     const float* deltas, const std::int32_t* sums, std::uint32_t* counts, cudaStream_t s) {
    if (rows > d.max_rows || pad < rows) throw std::invalid_argument("tiered publish: rows exceed the exchange");
    publish_kernel<<<pad, 256, 0, s>>>(d, layer, rows, routes, slot_routes, table, zero_slot, codes, deltas, sums, counts);
    ck(cudaGetLastError(), "publish");
}

void enqueue_wait_gather(const TierDev& d, std::uint32_t rows, float* rawdots, std::uint64_t rawdot_stride_bytes,
                         std::uint32_t* status, std::uint64_t status_stride_bytes, std::uint32_t* summary, cudaStream_t s) {
    if (rows > d.max_rows) throw std::invalid_argument("tiered gather: rows exceed the exchange");
    wait_kernel<<<1, 256, 0, s>>>(d, rows, status, static_cast<std::uint32_t>(status_stride_bytes / 4U), summary);
    ck(cudaGetLastError(), "wait");
    gather_kernel<<<rows * d.top_k, 256, 0, s>>>(d, rawdots, rawdot_stride_bytes / 4U);
    ck(cudaGetLastError(), "gather");
}

TierWorker::TierWorker(const Exchange& ex, std::vector<strix::cpu::expert::LayerWeights> layers,
                       strix::cpu::expert::Executor& executor)
    : ex_(ex), layers_(std::move(layers)), executor_(executor) {
    thread_ = std::thread([this] { loop(); });
}

TierWorker::~TierWorker() {
    stop_.store(true, std::memory_order_release);
    if (thread_.joinable()) thread_.join();
}

WorkerStats TierWorker::stats() const {
    WorkerStats s;
    s.jobs = jobs_.load(std::memory_order_relaxed);
    s.misses = misses_.load(std::memory_order_relaxed);
    s.cpu_ms = static_cast<double>(cpu_ns_.load(std::memory_order_relaxed)) * 1e-6;
    return s;
}

void TierWorker::loop() {
    using clock = std::chrono::steady_clock;
    const auto& lay = ex_.layout();
    std::vector<strix::cpu::expert::RowA8> a8(lay.max_rows);
    std::vector<strix::cpu::expert::Assignment> work(static_cast<std::size_t>(lay.max_rows) * lay.top_k);
    const std::uint32_t groups = lay.hidden / 32U;
    auto idle_since = clock::now();
    std::uint64_t answered = 0;
    for (;;) {
        // Observe stop before published: after the producer is quiescent, shutdown
        // must still answer its final publication before exchange storage can go.
        const bool stopping = stop_.load(std::memory_order_acquire);
        const std::uint64_t seq = ex_.published().load(std::memory_order_acquire);
        if (seq == answered) {
            if (stopping) break;
            if (clock::now() - idle_since > std::chrono::milliseconds(2)) std::this_thread::sleep_for(std::chrono::microseconds(50));
            else std::this_thread::yield();
            continue;
        }
        const auto t0 = clock::now();
        const std::uint32_t* h = ex_.header();
        const std::uint32_t layer = h[0], rows = h[1], n = h[2];
        bool bad = layer >= layers_.size() || rows > lay.max_rows || n > work.size();
        if (!bad) {
            try {
                const Miss* m = ex_.miss();
                for (std::uint32_t i = 0; i < n; ++i) work[i] = {m[i].row, m[i].slot, m[i].expert};
                for (std::uint32_t r = 0; r < rows; ++r) {
                    a8[r].codes = ex_.codes() + std::uint64_t{r} * lay.hidden;
                    a8[r].deltas = ex_.deltas() + std::uint64_t{r} * groups;
                    a8[r].sums = ex_.sums() + std::uint64_t{r} * groups;
                }
                std::memset(ex_.status(), 0, std::size_t{rows} * 4U);
                strix::cpu::expert::Job job;
                job.weights = layers_[layer];
                job.rows = a8.data();
                job.assignments = work.data();
                job.count = n;
                job.rawdots = ex_.rawdots();
                job.rawdot_row_stride = std::uint64_t{lay.top_k} * lay.hidden;
                job.status = ex_.status();
                executor_.run(job);
            } catch (...) {
                bad = true;
            }
        }
        if (bad) {
            failed_.store(true, std::memory_order_relaxed);
            std::atomic_ref<std::uint32_t>(ex_.header()[3]).store(1U, std::memory_order_release);
        }
        cpu_ns_.fetch_add(static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(clock::now() - t0).count()),
                          std::memory_order_relaxed);
        jobs_.fetch_add(1, std::memory_order_relaxed);
        misses_.fetch_add(n, std::memory_order_relaxed);
        answered = seq;
        ex_.done().store(seq, std::memory_order_release);
        idle_since = clock::now();
    }
}

}  // namespace strix::cuda::tiered
