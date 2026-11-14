#include "strix/hip_check.hpp"
#include "strix/hip_generation_output.hpp"
#include "strix/hip_argmax_core.hpp"
#include "strix/mtp_payload_geometry.hpp"
#include "strix/hip_setup_quiescence.hpp"
#if defined(STRIX_BUILD_GREEDY_OUTPUT_DIAGNOSTIC)
#include "strix/hip_generation_output_diagnostic.hpp"
#endif
#include <hip/hip_runtime.h>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

namespace strix {
#if defined(STRIX_BUILD_GREEDY_OUTPUT_DIAGNOSTIC)
namespace {
thread_local HipGreedyOutputDiagnosticFault diagnostic_fault = HipGreedyOutputDiagnosticFault::none;
bool consume_fault(HipGreedyOutputDiagnosticFault expected) noexcept {
    if (diagnostic_fault != expected) return false;
    diagnostic_fault = HipGreedyOutputDiagnosticFault::none;
    return true;
}
void diagnostic_boundary(HipGreedyOutputDiagnosticFault expected) {
    if (consume_fault(expected)) throw std::runtime_error("injected GPU greedy host boundary failure");
}
}
void arm_hip_greedy_output_diagnostic_fault(HipGreedyOutputDiagnosticFault fault) {
    if (diagnostic_fault != HipGreedyOutputDiagnosticFault::none)
        throw std::logic_error("GPU greedy diagnostic fault already armed");
    diagnostic_fault = fault;
}
void clear_hip_greedy_output_diagnostic_fault() noexcept {
    diagnostic_fault = HipGreedyOutputDiagnosticFault::none;
}
#endif
namespace {
constexpr unsigned threads = 256;
static_assert(sizeof(float) == 4);
using argmax::Partial;
struct alignas(8) Reply {
    std::uint64_t epoch, position;
    std::uint32_t token, status;
    std::uint64_t execution_identity;
};
static_assert(sizeof(Reply) == 32);
static_assert(offsetof(Reply, execution_identity) == 24);

__global__ __launch_bounds__(threads) void greedy_output_partials(
    const float* logits, std::uint32_t vocabulary, Partial* partials) {
    __shared__ Partial values[threads];
    const auto lane = static_cast<unsigned>(threadIdx.x);
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * threads + lane;
    const Partial value = index < vocabulary
        ? argmax::load(logits[index], static_cast<std::uint32_t>(index)) : argmax::empty();
    argmax::block_argmax<threads>(values, lane, value);
    if (!lane) partials[blockIdx.x] = values[0];
}
__global__ __launch_bounds__(threads) void greedy_output_finish(
    const Partial* partials, std::uint32_t count, Reply* reply,
    std::uint64_t epoch, std::uint64_t position, std::uint64_t execution_identity) {
    __shared__ Partial values[threads];
    const auto lane = static_cast<unsigned>(threadIdx.x);
    const auto best = argmax::block_argmax_partials<threads>(values, lane, partials, count, argmax::empty());
    if (!lane) {
        *reply = Reply{epoch, position, best.token,
                       argmax::status(best), execution_identity};
    }
}
struct GreedyRowsViews { const float* logits[hip_greedy_rows_max]; };
__global__ __launch_bounds__(threads) void greedy_rows_partials(
    GreedyRowsViews views, std::uint32_t vocabulary, std::uint32_t count, Partial* partials) {
    __shared__ Partial values[threads];
    const auto lane = static_cast<unsigned>(threadIdx.x);
    const auto index = static_cast<std::uint64_t>(blockIdx.x) * threads + lane;
    const auto* logits = views.logits[blockIdx.y];
    const Partial value = index < vocabulary
        ? argmax::load(logits[index], static_cast<std::uint32_t>(index)) : argmax::empty();
    argmax::block_argmax<threads>(values, lane, value);
    if (!lane) partials[static_cast<std::uint64_t>(blockIdx.y) * count + blockIdx.x] = values[0];
}
__global__ __launch_bounds__(threads) void greedy_rows_finish(
    const Partial* partials, std::uint32_t count, HipGreedyRowsReply* replies) {
    __shared__ Partial values[threads];
    const auto lane = static_cast<unsigned>(threadIdx.x);
    const auto* row = partials + static_cast<std::uint64_t>(blockIdx.x) * count;
    const auto best = argmax::block_argmax_partials<threads>(values, lane, row, count, argmax::empty());
    if (!lane) {
        replies[blockIdx.x] = HipGreedyRowsReply{best.token, argmax::status(best)};
    }
}
using strix::hip_check;
void require(bool value, const char* message) {
    if (!value) throw std::invalid_argument(message);
}
bool overlaps(std::uintptr_t first, std::uint64_t bytes, const void* other, std::uint64_t count) {
    const auto second = reinterpret_cast<std::uintptr_t>(other);
    return first < second + count && second < first + bytes;
}
} // namespace

struct HipGreedyOutput::Impl {
    const std::uint32_t vocabulary;
    const ExecutionMemoryBinding scratch;
    const std::uint64_t execution_identity;
    const std::uint32_t partial_count;
    const std::uint64_t partial_bytes;
    std::shared_ptr<const void> source_owner;
    std::shared_ptr<const HipSetupQuiescenceGuard> guard;
    hipStream_t stream = nullptr;
    Partial* partials = nullptr;
    Reply* device_reply = nullptr;
    Reply* host_reply = nullptr;
    ExecutionBufferAddress binding{};
    HipGreedyOutputStats counters{};
    std::uint64_t last_epoch = 0;
    std::uint64_t pending_position = 0;
    bool bound = false, pending = false, broken = false;

    Impl(std::uint32_t vocab, ExecutionMemoryBinding arena, std::shared_ptr<const void> owner,
         std::shared_ptr<const HipSetupQuiescenceGuard> setup)
        : vocabulary(vocab), scratch(arena), execution_identity(arena.execution_identity),
          partial_count(vocab ? (vocab - 1U) / threads + 1U : 0),
          partial_bytes(HipGreedyOutput::required_scratch_bytes(vocab) - 2U * sizeof(Reply)),
          source_owner(std::move(owner)), guard(std::move(setup)) {
        require(vocabulary && execution_identity && source_owner && guard,
                "GPU greedy output requires fixed vocabulary, identity and actual retained owners");
        require(scratch.arena_base && scratch.arena_bytes && scratch.allocation_generation &&
                scratch.arena_bytes <= std::numeric_limits<std::uintptr_t>::max() - scratch.arena_base,
                "GPU greedy output requires a bounded retained scratch arena");
        require(partial_bytes <= std::numeric_limits<std::size_t>::max(), "GPU greedy scratch exceeds host size");
        guard->require_original();
        try {
            hip_check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking), "create greedy output stream");
#if defined(STRIX_BUILD_GREEDY_OUTPUT_DIAGNOSTIC)
            diagnostic_boundary(HipGreedyOutputDiagnosticFault::partial_allocation);
#endif
            hip_check(hipMalloc(reinterpret_cast<void**>(&partials), static_cast<std::size_t>(partial_bytes)), "allocate greedy partials");
            ++counters.pool_allocations;
#if defined(STRIX_BUILD_GREEDY_OUTPUT_DIAGNOSTIC)
            diagnostic_boundary(HipGreedyOutputDiagnosticFault::reply_allocation);
#endif
            hip_check(hipMalloc(reinterpret_cast<void**>(&device_reply), sizeof(Reply)), "allocate greedy result");
            ++counters.pool_allocations;
#if defined(STRIX_BUILD_GREEDY_OUTPUT_DIAGNOSTIC)
            diagnostic_boundary(HipGreedyOutputDiagnosticFault::pinned_allocation);
#endif
            hip_check(hipHostMalloc(reinterpret_cast<void**>(&host_reply), sizeof(Reply)), "allocate greedy pinned result");
            ++counters.pool_allocations;
        } catch (...) { cleanup(); throw; }
    }
    void cleanup() noexcept {
        if (!guard) return;
        // Restores the original device and proves quiescence or fail-stops without
        // unwinding pinned host staging/source owners. Also handles partial setup.
        guard->recover_or_fail_stop(); pending = false;
        const auto release = [](hipError_t e) { if (e != hipSuccess) std::_Exit(3); };
        if (host_reply) { release(hipHostFree(host_reply)); host_reply = nullptr; }
        if (device_reply) { release(hipFree(device_reply)); device_reply = nullptr; }
        if (partials) { release(hipFree(partials)); partials = nullptr; }
        if (stream) { release(hipStreamDestroy(stream)); stream = nullptr; }
    }
    ~Impl() { cleanup(); }
    void admit(const ExecutionBufferAddress& input, std::uint32_t vocab, std::uint64_t epoch) {
        require(!broken && !pending, "GPU greedy owner is failed or already pending");
        require(vocab == vocabulary && epoch && epoch > last_epoch, "GPU greedy vocabulary/epoch mismatch");
        const auto required = static_cast<std::uint64_t>(vocabulary) * sizeof(float);
        require(input.address && input.address % alignof(float) == 0 && input.bytes == required &&
                input.allocation_generation == scratch.allocation_generation &&
                input.address >= scratch.arena_base && input.address - scratch.arena_base <= scratch.arena_bytes &&
                input.bytes <= scratch.arena_bytes - (input.address - scratch.arena_base),
                "GPU greedy logits span is invalid");
        require(!overlaps(input.address, input.bytes, partials, partial_bytes) &&
                !overlaps(input.address, input.bytes, device_reply, sizeof(Reply)), "GPU greedy logits alias output pool");
        if (bound) require(input.address == binding.address && input.bytes == binding.bytes &&
                           input.allocation_generation == binding.allocation_generation,
                           "GPU greedy logits allocation identity changed");
        guard->require_original();
        if (!bound) { binding = input; bound = true; }
        // A failed submission may have consumed this epoch. Never reuse it.
        last_epoch = epoch;
    }
    void submit(const ExecutionBufferAddress& input, std::uint32_t vocab,
                std::uint64_t position, std::uint64_t epoch) {
        admit(input, vocab, epoch);
        require(counters.calls < std::numeric_limits<std::uint64_t>::max() / 64U,
                "GPU greedy counters exhausted");
        ++counters.calls;
        pending_position = position;
        pending = true;
        try {
            hip_check(hipPeekAtLastError(), "GPU greedy preexisting HIP error");
            // Poison the complete device reply before EACH launch. If a stage is
            // accidentally omitted, stale metadata cannot pass current-call checks.
            *host_reply = Reply{~epoch, ~position, 0xffffffffU, 0xffffffffU, ~execution_identity};
            hip_check(hipMemcpyAsync(device_reply, host_reply, sizeof(Reply), hipMemcpyHostToDevice, stream), "poison greedy result");
            counters.poison_h2d_bytes += sizeof(Reply);
#if defined(STRIX_BUILD_GREEDY_OUTPUT_DIAGNOSTIC)
            diagnostic_boundary(HipGreedyOutputDiagnosticFault::after_poison);
#endif
            hipLaunchKernelGGL(greedy_output_partials, dim3(partial_count), dim3(threads), 0, stream,
                               reinterpret_cast<const float*>(input.address), vocabulary, partials);
            hip_check(hipGetLastError(), "submit greedy partials"); ++counters.reduction_kernel_submissions;
#if defined(STRIX_BUILD_GREEDY_OUTPUT_DIAGNOSTIC)
            diagnostic_boundary(HipGreedyOutputDiagnosticFault::after_partials);
#endif
            hipLaunchKernelGGL(greedy_output_finish, dim3(1), dim3(threads), 0, stream,
                               partials, partial_count, device_reply, epoch, position, execution_identity);
            hip_check(hipGetLastError(), "submit greedy finish"); ++counters.reduction_kernel_submissions;
#if defined(STRIX_BUILD_GREEDY_OUTPUT_DIAGNOSTIC)
            diagnostic_boundary(HipGreedyOutputDiagnosticFault::after_finish);
#endif
            hip_check(hipMemcpyAsync(host_reply, device_reply, sizeof(Reply), hipMemcpyDeviceToHost, stream), "read greedy result");
            counters.result_d2h_bytes += sizeof(Reply);
#if defined(STRIX_BUILD_GREEDY_OUTPUT_DIAGNOSTIC)
            diagnostic_boundary(HipGreedyOutputDiagnosticFault::after_readback);
#endif
        } catch (...) {
            broken = true; ++counters.failures;
            guard->recover_or_fail_stop(); pending = false;
            throw;
        }
    }
    std::uint32_t complete() {
        require(!broken && pending, "GPU greedy owner has no completable pending request");
        const auto epoch = last_epoch;
        const auto position = pending_position;
        try {
            guard->require_original();
            hip_check(hipStreamSynchronize(stream), "complete greedy result");
            ++counters.stream_completions;
#if defined(STRIX_BUILD_GREEDY_OUTPUT_DIAGNOSTIC)
            diagnostic_boundary(HipGreedyOutputDiagnosticFault::after_completion);
            if (consume_fault(HipGreedyOutputDiagnosticFault::corrupt_epoch)) host_reply->epoch ^= 1U;
            if (consume_fault(HipGreedyOutputDiagnosticFault::corrupt_position)) host_reply->position ^= 1U;
            if (consume_fault(HipGreedyOutputDiagnosticFault::corrupt_identity)) host_reply->execution_identity ^= 1U;
            if (consume_fault(HipGreedyOutputDiagnosticFault::corrupt_status)) host_reply->status = 1U;
            if (consume_fault(HipGreedyOutputDiagnosticFault::corrupt_token)) host_reply->token = vocabulary;
#endif
            guard->require_original();
            pending = false;
            require(host_reply->epoch == epoch && host_reply->position == position &&
                    host_reply->execution_identity == execution_identity, "stale GPU greedy result");
            require(host_reply->status == 0 && host_reply->token < vocabulary,
                    "GPU greedy logits must all be finite and selection valid");
            ++counters.completed;
            return host_reply->token;
        } catch (...) {
            broken = true; ++counters.failures;
            // D1 may already have committed this token. Its abort() alone is NOT
            // sufficient: this owner's newly enqueued selection must be drained.
            guard->recover_or_fail_stop(); pending = false;
            throw;
        }
    }
};
std::uint64_t HipGreedyOutput::required_scratch_bytes(std::uint32_t vocabulary) {
    return mtp_payload::selector(vocabulary);
}
HipGreedyOutput::HipGreedyOutput(std::uint32_t vocab, ExecutionMemoryBinding scratch,
    std::shared_ptr<const void> owner, std::shared_ptr<const HipSetupQuiescenceGuard> guard)
    : impl_(std::make_unique<Impl>(vocab, scratch, std::move(owner), std::move(guard))) {}
HipGreedyOutput::~HipGreedyOutput() = default;
std::uint32_t HipGreedyOutput::select(const ExecutionBufferAddress& input,
    std::uint32_t vocab, std::uint64_t position, std::uint64_t epoch) {
    impl_->submit(input, vocab, position, epoch);
    return impl_->complete();
}
void HipGreedyOutput::submit(const ExecutionBufferAddress& input,
    std::uint32_t vocab, std::uint64_t position, std::uint64_t epoch) {
    impl_->submit(input, vocab, position, epoch);
}
std::uint32_t HipGreedyOutput::complete() { return impl_->complete(); }
HipGreedyOutputStats HipGreedyOutput::stats() const noexcept { return impl_->counters; }
bool HipGreedyOutput::failed() const noexcept { return impl_->broken; }
std::uint64_t HipGreedyOutput::scratch_bytes() const noexcept { return required_scratch_bytes(impl_->vocabulary); }
std::uint64_t hip_greedy_rows_scratch_bytes(std::uint32_t rows, std::uint32_t vocabulary) {
    require(rows && rows <= hip_greedy_rows_max && vocabulary, "GPU greedy rows require 1..8 rows and a nonzero vocabulary");
    return static_cast<std::uint64_t>(rows) * ((static_cast<std::uint64_t>(vocabulary) - 1U) / threads + 1U) * sizeof(Partial);
}
static_assert(sizeof(HipGreedyRowsPartial) == sizeof(Partial) && threads == hip_greedy_rows_partial_logits);
void enqueue_hip_greedy_rows_finish(ExecutionBufferAddress partials, std::uint32_t rows, std::uint32_t vocabulary,
    HipGreedyRowsReply* replies, void* stream) {
    const auto bytes = hip_greedy_rows_scratch_bytes(rows, vocabulary);
    require(replies && partials.address && partials.address % alignof(Partial) == 0 && partials.bytes >= bytes,
        "GPU greedy rows scratch/reply views are invalid");
    void* device_replies = nullptr;
    hip_check(hipHostGetDevicePointer(&device_replies, replies, 0), "GPU greedy rows replies must be host-mapped");
    const auto count = static_cast<std::uint32_t>((static_cast<std::uint64_t>(vocabulary) - 1U) / threads + 1U);
    hipLaunchKernelGGL(greedy_rows_finish, dim3(rows), dim3(threads), 0, reinterpret_cast<hipStream_t>(stream),
        reinterpret_cast<const Partial*>(partials.address), count, static_cast<HipGreedyRowsReply*>(device_replies));
    hip_check(hipGetLastError(), "submit greedy rows finish");
}
void enqueue_hip_greedy_rows(std::span<const ExecutionBufferAddress> logits, std::uint32_t vocabulary,
    ExecutionBufferAddress partials, HipGreedyRowsReply* replies, void* stream) {
    const auto rows = static_cast<std::uint32_t>(logits.size());
    const auto bytes = hip_greedy_rows_scratch_bytes(rows, vocabulary);
    require(replies && partials.address && partials.address % alignof(Partial) == 0 && partials.bytes >= bytes,
        "GPU greedy rows scratch/reply views are invalid");
    GreedyRowsViews views{};
    for (std::uint32_t r = 0; r < rows; ++r) {
        const auto& view = logits[r];
        require(view.address && view.address % alignof(float) == 0 &&
            view.bytes >= static_cast<std::uint64_t>(vocabulary) * sizeof(float) &&
            !overlaps(view.address, view.bytes, reinterpret_cast<const void*>(partials.address), bytes),
            "GPU greedy rows logits span is invalid or aliases its scratch");
        views.logits[r] = reinterpret_cast<const float*>(view.address);
    }
    void* device_replies = nullptr;
    hip_check(hipHostGetDevicePointer(&device_replies, replies, 0), "GPU greedy rows replies must be host-mapped");
    const auto count = static_cast<std::uint32_t>((static_cast<std::uint64_t>(vocabulary) - 1U) / threads + 1U);
    const auto queue = reinterpret_cast<hipStream_t>(stream);
    auto* scratch = reinterpret_cast<Partial*>(partials.address);
    hip_check(hipPeekAtLastError(), "GPU greedy rows preexisting HIP error");
    hipLaunchKernelGGL(greedy_rows_partials, dim3(count, rows), dim3(threads), 0, queue, views, vocabulary, count, scratch);
    hip_check(hipGetLastError(), "submit greedy rows partials");
    hipLaunchKernelGGL(greedy_rows_finish, dim3(rows), dim3(threads), 0, queue, scratch, count,
        static_cast<HipGreedyRowsReply*>(device_replies));
    hip_check(hipGetLastError(), "submit greedy rows finish");
}
} // namespace strix
