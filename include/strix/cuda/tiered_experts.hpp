// Tiered expert residency, CUDA side (model-agnostic: shapes come from TierLayout, a model adapter fills the rest).
//
// An expert that is not on the card is computed on the host (strix::cpu::expert::Executor, bitwise = the card's A8
// chains) and its per-(row, slot) rawdots go back into the card's own merge. Per layer, in stream order:
//   prologue -> publish -> [routed kernels on SLOT routes] -> wait -> gather -> combine
// publish  maps expert ids to slots through the layer's device table (expert -> slot or -1); a miss is routed to the
//          layer's ZERO slot (a zero-filled expert: finite 0 outputs) and listed, with its row's A8 codes/deltas/sums,
//          in the host-mapped exchange; a flag word (seq_published) is written last. No miss: nothing is signalled.
// wait     spins (volatile, watchdog) on seq_done == seq; on timeout the step fails closed (row status bit 8).
// gather   copies each miss's rawdots into the card's rawdots[row][slot] and ORs the host row status.
// All kernels take device pointers and per-launch constants only: capturable in a CUDA graph and replayable (the table
// is device memory that is written between replays). One exchange region serves every layer (stream order serialises
// the layers); the host worker answers one job at a time. After timeout, dead is
// sticky: later publishes leave all exchange inputs untouched and waits fail.
// Quiesce the producing stream, then destroy/join TierWorker before destroying,
// resetting or reusing exchange storage. A GPU timeout is not a CPU cancellation.
#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "cpu/core/routed_expert.hpp"

namespace strix::cuda::tiered {

// Scope a resolved cap around allocation entry points, independent of expert mode.
// Nested scopes restore the caller's budget. Only setup/allocation paths query the card.
class AllocationBudget {
public:
    AllocationBudget(std::uint64_t initial_free, std::uint64_t budget);
    ~AllocationBudget();
    AllocationBudget(const AllocationBudget&) = delete;
    AllocationBudget& operator=(const AllocationBudget&) = delete;
private:
    std::uint64_t previous_free_, previous_budget_;
};
void require_budget(std::uint64_t bytes, const char* what);

struct TierLayout {
    std::uint32_t hidden = 0, ffn = 0, top_k = 0, experts = 0, max_rows = 0;
};

struct Miss {
    std::uint32_t row, slot, expert, pad;
};

// Device-memory control block (counters of the publish kernel, the seq the wait kernel expects).
struct DevState {
    std::uint32_t n_miss, done, last_n, dead;
    std::uint64_t seq;
};

// Kernel-side view of the exchange (device addresses of the mapped host memory + the device control block).
struct TierDev {
    volatile std::uint64_t* seq_published = nullptr;
    volatile std::uint64_t* seq_done = nullptr;
    std::uint32_t* header = nullptr;   // [layer, rows, n_miss, failed]; failed is written by the host only
    Miss* miss = nullptr;
    std::int8_t* codes = nullptr;      // [max_rows][hidden]
    float* deltas = nullptr;           // [max_rows][hidden / 32]
    std::int32_t* sums = nullptr;      // [max_rows][hidden / 32]
    std::uint32_t* status = nullptr;   // [max_rows]
    float* rawdots = nullptr;          // [max_rows][top_k * hidden] (the Rows::rawdots layout)
    DevState* state = nullptr;
    std::uint32_t hidden = 0, top_k = 0, experts = 0, max_rows = 0;
    std::uint64_t timeout_ns = 10'000'000'000ULL;
};

// Owns the mapped exchange (cudaHostAlloc Mapped|Portable, the PLE pattern) and the device control block.
class Exchange {
public:
    explicit Exchange(const TierLayout& layout);
    ~Exchange();
    Exchange(const Exchange&) = delete;
    Exchange& operator=(const Exchange&) = delete;
    [[nodiscard]] const TierLayout& layout() const noexcept { return layout_; }
    [[nodiscard]] const TierDev& dev() const noexcept { return dev_; }
    [[nodiscard]] std::uint64_t host_bytes() const noexcept { return host_bytes_; }
    void set_timeout_ns(std::uint64_t ns) { dev_.timeout_ns = ns; }  // before any capture
    // Host views (the same memory the device addresses).
    std::atomic_ref<std::uint64_t> published() const { return std::atomic_ref<std::uint64_t>(*host_published_); }
    std::atomic_ref<std::uint64_t> done() const { return std::atomic_ref<std::uint64_t>(*host_done_); }
    std::uint32_t* header() const noexcept { return host_header_; }
    Miss* miss() const noexcept { return host_miss_; }
    std::int8_t* codes() const noexcept { return host_codes_; }
    float* deltas() const noexcept { return host_deltas_; }
    std::int32_t* sums() const noexcept { return host_sums_; }
    std::uint32_t* status() const noexcept { return host_status_; }
    float* rawdots() const noexcept { return host_rawdots_; }
private:
    TierLayout layout_;
    TierDev dev_;
    std::uint8_t* host_base_ = nullptr;
    std::uint64_t host_bytes_ = 0;
    std::uint64_t* host_published_ = nullptr;
    std::uint64_t* host_done_ = nullptr;
    std::uint32_t* host_header_ = nullptr;
    Miss* host_miss_ = nullptr;
    std::int8_t* host_codes_ = nullptr;
    float* host_deltas_ = nullptr;
    std::int32_t* host_sums_ = nullptr;
    std::uint32_t* host_status_ = nullptr;
    float* host_rawdots_ = nullptr;
    DevState* state_ = nullptr;
};

// Rows 0..rows-1 are live; rows..pad-1 are padding (routed to the zero slot, never signalled). routes/slot_routes:
// [pad][top_k] uint32; table: int32[experts] of this layer; counts: optional uint32[experts] of this layer (one
// increment per live (row, slot) route: the planner's step usage). codes/deltas/sums: the prologue's A8 rows, contiguous.
void enqueue_publish(const TierDev&, std::uint32_t layer, std::uint32_t rows, std::uint32_t pad, const std::uint32_t* routes,
                     std::uint32_t* slot_routes, const std::int32_t* table, std::uint32_t zero_slot, const std::int8_t* codes,
                     const float* deltas, const std::int32_t* sums, std::uint32_t* counts, cudaStream_t);
// wait + gather. rawdots/status are the Rows views (strides in bytes); summary = the CSR summary word block.
void enqueue_wait_gather(const TierDev&, std::uint32_t rows, float* rawdots, std::uint64_t rawdot_stride_bytes,
                         std::uint32_t* status, std::uint64_t status_stride_bytes, std::uint32_t* summary, cudaStream_t);

struct WorkerStats {
    std::uint64_t jobs = 0, misses = 0;
    double cpu_ms = 0;  // executor time summed over jobs
};

// Host thread: polls seq_published != seq_done (spin, then yield/sleep backoff when idle > ~2 ms), runs the layer's
// job on the Executor and answers with seq_done. layers[l] = the stored expert tensors of layer l (file-mapped).
class TierWorker {
public:
    TierWorker(const Exchange& ex, std::vector<strix::cpu::expert::LayerWeights> layers, strix::cpu::expert::Executor& executor);
    ~TierWorker();  // drains/acknowledges the outstanding job; producer must be quiescent
    TierWorker(const TierWorker&) = delete;
    TierWorker& operator=(const TierWorker&) = delete;
    [[nodiscard]] WorkerStats stats() const;
    [[nodiscard]] bool failed() const noexcept { return failed_.load(std::memory_order_relaxed); }
private:
    void loop();
    const Exchange& ex_;
    std::vector<strix::cpu::expert::LayerWeights> layers_;
    strix::cpu::expert::Executor& executor_;
    std::atomic<bool> stop_{false}, failed_{false};
    std::atomic<std::uint64_t> jobs_{0}, misses_{0}, cpu_ns_{0};
    std::thread thread_;
};

}  // namespace strix::cuda::tiered
