#include "strix/hip_check.hpp"
#include "strix/hip_qsa_selection_diagnostic.hpp"
#include <hip/hip_runtime.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strix::qsa_selection_diagnostic {
namespace {
constexpr unsigned threads = 256, tile_items = 1024;
void require(bool ok, const char* text) { if (!ok) throw std::invalid_argument(text); }
using strix::hip_check;
void validate_shape(Shape shape) {
    require(shape.blocks <= 32768 && shape.tail < compression &&
            static_cast<std::uint64_t>(shape.blocks) * compression + shape.tail <= 131072,
            "QSA diagnostic requires B<=32768, tail<4 and visible<=131072");
}
struct Range { std::uintptr_t begin; std::uint64_t bytes; };
template<class T> Range range(View<T> view, std::uint64_t expected) {
    require(view.count == expected, "QSA diagnostic view extent mismatch");
    if (!expected) { require(view.data == nullptr, "QSA diagnostic empty view must be null"); return {}; }
    const auto pointer = reinterpret_cast<std::uintptr_t>(view.data);
    require(pointer && pointer % alignof(T) == 0 &&
            expected <= std::numeric_limits<std::uint64_t>::max() / sizeof(T), "QSA diagnostic pointer/alignment invalid");
    const auto bytes = expected * sizeof(T);
    require(bytes <= std::numeric_limits<std::uintptr_t>::max() - pointer, "QSA diagnostic range overflow");
    return {pointer, bytes};
}
__device__ std::uint32_t key(float score) {
    auto bits = __float_as_uint(score);
    if ((bits & 0x7fffffffU) == 0) bits = 0;
    return bits & 0x80000000U ? ~bits : bits ^ 0x80000000U;
}
__device__ bool before(Item a, Item b, std::uint32_t blocks) {
    const bool av = a.index < blocks, bv = b.index < blocks;
    if (av != bv) return av;
    if (av && key(a.score) != key(b.score)) return key(a.score) > key(b.score);
    return a.index < b.index;
}
__device__ bool rejected(const std::uint32_t* status) {
    // Only validation writes status, in a prior same-stream kernel. Caller
    // forbids concurrent external writers; all subsequent kernels are readers.
    return *status != 0;
}
__global__ void validate_scores(const float* scores, std::uint32_t count, std::uint32_t* status) {
    const auto i = static_cast<std::uint32_t>(blockIdx.x) * threads + threadIdx.x;
    if (i < count && (__float_as_uint(scores[i]) & 0x7f800000U) == 0x7f800000U)
        atomicOr(status, nonfinite_score);
}
__global__ __launch_bounds__(threads) void direct_tokens(std::uint32_t* selected,
    std::uint32_t visible, const std::uint32_t* status) {
    if (rejected(status)) return;
    const auto i = static_cast<std::uint32_t>(blockIdx.x) * threads + threadIdx.x;
    if (i < visible) selected[i] = i;
}
__global__ __launch_bounds__(threads) void tile_top512(const float* scores, std::uint32_t blocks,
    Item* output, const std::uint32_t* status) {
    if (rejected(status)) return;
    __shared__ Item tile[tile_items]; // 8192 bytes, four items/thread.
    const auto lane = static_cast<unsigned>(threadIdx.x);
    const auto base = static_cast<std::uint32_t>(blockIdx.x) * tile_items;
    for (unsigned i = lane; i < tile_items; i += threads) {
        const auto id = base + i;
        tile[i] = Item{id < blocks ? scores[id] : 0.0F, id};
    }
    __syncthreads();
    for (unsigned span = 2; span <= tile_items; span <<= 1U) {
        for (unsigned distance = span / 2; distance; distance >>= 1U) {
            for (unsigned i = lane; i < tile_items; i += threads) {
                const auto other = i ^ distance;
                if (i < other) {
                    const auto a = tile[i], b = tile[other];
                    const bool forward = (i & span) == 0;
                    if (forward ? before(b, a, blocks) : before(a, b, blocks)) {
                        tile[i] = b; tile[other] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (unsigned i = lane; i < block_topk; i += threads)
        output[static_cast<std::uint32_t>(blockIdx.x) * block_topk + i] = tile[i];
}
__global__ __launch_bounds__(threads) void merge_top512(const Item* source, Item* destination,
    std::uint32_t lists, std::uint32_t blocks, const std::uint32_t* status) {
    if (rejected(status)) return;
    const auto pair = static_cast<std::uint32_t>(blockIdx.x);
    const auto first = pair * 2U;
    const auto base = first * block_topk;
    if (first + 1U == lists) {
        for (unsigned i = threadIdx.x; i < block_topk; i += threads)
            destination[pair * block_topk + i] = source[base + i];
        return;
    }
    for (unsigned i = threadIdx.x; i < 2U * block_topk; i += threads) {
        const bool right = i >= block_topk;
        const auto local = i % block_topk;
        const auto value = source[base + (right ? block_topk : 0U) + local];
        const auto other_base = base + (right ? 0U : block_topk);
        unsigned lo = 0, hi = block_topk;
        while (lo < hi) {
            const auto mid = lo + (hi - lo) / 2U;
            if (before(source[other_base + mid], value, blocks)) lo = mid + 1U;
            else hi = mid;
        }
        const auto rank = local + lo;
        // Every original/padding index is unique across the two sorted lists;
        // therefore the combined binary ranks are distinct and writes disjoint.
        if (rank < block_topk) destination[pair * block_topk + rank] = value;
    }
}
__global__ __launch_bounds__(threads) void chronological_expand(const Item* source,
    std::uint32_t blocks, std::uint32_t tail, std::uint32_t* selected,
    const std::uint32_t* status) {
    if (rejected(status)) return;
    __shared__ Item order[block_topk]; // 4096 bytes; scores unused after rank selection.
    const auto lane = static_cast<unsigned>(threadIdx.x);
    for (unsigned i = lane; i < block_topk; i += threads) order[i] = source[i];
    __syncthreads();
    for (unsigned span = 2; span <= block_topk; span <<= 1U) {
        for (unsigned distance = span / 2; distance; distance >>= 1U) {
            for (unsigned i = lane; i < block_topk; i += threads) {
                const auto other = i ^ distance;
                if (i < other) {
                    const auto a = order[i], b = order[other];
                    const bool forward = (i & span) == 0;
                    if (forward ? b.index < a.index : a.index < b.index) { order[i] = b; order[other] = a; }
                }
            }
            __syncthreads();
        }
    }
    for (unsigned i = lane; i < block_topk * compression; i += threads)
        selected[i] = order[i / compression].index * compression + i % compression;
    if (lane < tail) selected[block_topk * compression + lane] = blocks * compression + lane;
}
} // namespace
std::uint32_t selected_count(Shape shape) {
    validate_shape(shape);
    return (shape.blocks < block_topk ? shape.blocks : block_topk) * compression + shape.tail;
}
std::uint64_t scratch_items(Shape shape) {
    validate_shape(shape);
    return shape.blocks <= block_topk ? 0 :
        static_cast<std::uint64_t>((shape.blocks - 1U) / tile_items + 1U) * block_topk;
}
void enqueue(Shape shape, View<const float> scores, View<std::uint32_t> selected,
    View<Item> ping, View<Item> pong, View<std::uint32_t> status, hipStream_t stream) {
    validate_shape(shape);
    require(stream && stream != hipStreamLegacy && stream != hipStreamPerThread,
            "QSA diagnostic requires an explicitly owned non-default stream");
    const bool direct = shape.blocks <= block_topk;
    const auto count = selected_count(shape);
    const auto scratch = scratch_items(shape);
    const std::array<Range,5> ranges{
        range(scores, direct && scores.count == 0 ? 0 : shape.blocks), range(selected, count),
        range(ping, scratch), range(pong, scratch), range(status, 1)};
    for (std::size_t i = 0; i < ranges.size(); ++i)
        for (std::size_t j = i + 1; j < ranges.size(); ++j)
            if (ranges[i].bytes && ranges[j].bytes)
                require(ranges[i].begin >= ranges[j].begin + ranges[j].bytes ||
                        ranges[j].begin >= ranges[i].begin + ranges[i].bytes,
                        "QSA diagnostic input/output/status/scratch overlap");
    if (!count) return;
    hip_check(hipPeekAtLastError(), "QSA diagnostic preexisting HIP error");
    if (direct) {
        hipLaunchKernelGGL(direct_tokens, dim3((count - 1U) / threads + 1U), dim3(threads), 0, stream,
                           selected.data, count, status.data);
        hip_check(hipGetLastError(), "QSA diagnostic direct selection");
        return;
    }
    hipLaunchKernelGGL(validate_scores, dim3((shape.blocks - 1U) / threads + 1U), dim3(threads), 0, stream,
                       scores.data, shape.blocks, status.data);
    hip_check(hipGetLastError(), "QSA diagnostic finite validation");
    std::uint32_t lists = (shape.blocks - 1U) / tile_items + 1U;
    hipLaunchKernelGGL(tile_top512, dim3(lists), dim3(threads), 0, stream,
                       scores.data, shape.blocks, ping.data, status.data);
    hip_check(hipGetLastError(), "QSA diagnostic local top512");
    Item* source = ping.data;
    Item* destination = pong.data;
    while (lists > 1) {
        const auto next = (lists + 1U) / 2U;
        hipLaunchKernelGGL(merge_top512, dim3(next), dim3(threads), 0, stream,
                           source, destination, lists, shape.blocks, status.data);
        hip_check(hipGetLastError(), "QSA diagnostic top512 merge");
        auto* previous = source; source = destination; destination = previous;
        lists = next;
    }
    hipLaunchKernelGGL(chronological_expand, dim3(1), dim3(threads), 0, stream,
                       source, shape.blocks, shape.tail, selected.data, status.data);
    hip_check(hipGetLastError(), "QSA diagnostic chronological expansion");
}
} // namespace strix::qsa_selection_diagnostic
