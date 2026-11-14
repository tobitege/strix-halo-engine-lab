// Device CRC-32C: one thread per 256-byte lane, one block per 64 KiB span, lanes folded in a tree inside the block.
#include "xport/device_crc32c.hpp"

#include <stdexcept>
#include <string>

namespace strix::xport {
namespace {
struct FoldFactors {
    unsigned level[crc32c_fold_levels];
};
void crc_check(hipError_t status, const char* what) {
    if (status != hipSuccess) throw std::runtime_error(std::string("device crc32c: ") + what + ": " + hipGetErrorString(status));
}

__global__ __launch_bounds__(crc32c_fold_lanes) void crc32c_span_kernel(const uint4* __restrict__ data,
    const unsigned* __restrict__ table, FoldFactors factors, unsigned* __restrict__ digests) {
    __shared__ unsigned lane[crc32c_fold_lanes];
    constexpr unsigned groups = static_cast<unsigned>(crc32c_fold_lane_bytes / 16U);
    const uint4* source = data + static_cast<unsigned long long>(blockIdx.x) * (crc32c_fold_span_bytes / 16U) +
        threadIdx.x * groups;
    unsigned state = 0xffffffffU;
    for (unsigned g = 0; g < groups; ++g) {
        const uint4 v = source[g];
        state = crc32c_word(table, state, v.x);
        state = crc32c_word(table, state, v.y);
        state = crc32c_word(table, state, v.z);
        state = crc32c_word(table, state, v.w);
    }
    lane[threadIdx.x] = ~state;
    __syncthreads();
    for (unsigned level = 0; level < crc32c_fold_levels; ++level) {
        const unsigned half = 1U << level;
        if ((threadIdx.x & (2U * half - 1U)) == 0U)
            lane[threadIdx.x] = crc32c_append(lane[threadIdx.x], lane[threadIdx.x + half], factors.level[level]);
        __syncthreads();
    }
    if (threadIdx.x == 0U) digests[blockIdx.x] = lane[0];
}
} // namespace

DeviceCrc32c::~DeviceCrc32c() {
    if (tables_) (void)hipFree(tables_);
    if (digests_) (void)hipFree(digests_);
    if (host_digests_) (void)hipHostFree(host_digests_);
}

void DeviceCrc32c::allocate(std::size_t capacity) {
    if (tables_) throw std::logic_error("device crc32c: allocated twice");
    spans_ = capacity / crc32c_fold_span_bytes;
    const auto tables = crc32c_slice_tables();
    crc_check(hipMalloc(reinterpret_cast<void**>(&tables_), sizeof(tables)), "tables");
    crc_check(hipMemcpy(tables_, tables.data(), sizeof(tables), hipMemcpyHostToDevice), "tables");
    const std::size_t bytes = (spans_ ? spans_ : 1U) * sizeof(std::uint32_t);
    crc_check(hipMalloc(reinterpret_cast<void**>(&digests_), bytes), "digests");
    crc_check(hipHostMalloc(reinterpret_cast<void**>(&host_digests_), bytes, hipHostMallocDefault), "pinned digests");
}

void DeviceCrc32c::enqueue(hipStream_t stream, const void* base, std::size_t offset, std::size_t bytes) {
    const std::size_t first = offset / crc32c_fold_span_bytes, count = bytes / crc32c_fold_span_bytes;
    if (!tables_ || offset % crc32c_fold_span_bytes || first + count > spans_ || reinterpret_cast<std::uintptr_t>(base) % 16U)
        throw std::invalid_argument("device crc32c: run outside the allocated buffer or misaligned");
    if (!count) return;
    static const auto host_factors = crc32c_fold_factors();
    FoldFactors factors{};
    for (unsigned level = 0; level < crc32c_fold_levels; ++level) factors.level[level] = host_factors[level];
    hipLaunchKernelGGL(crc32c_span_kernel, dim3(static_cast<unsigned>(count)), dim3(crc32c_fold_lanes), 0, stream,
        reinterpret_cast<const uint4*>(static_cast<const std::byte*>(base) + offset), tables_, factors, digests_ + first);
    crc_check(hipGetLastError(), "launch");
    crc_check(hipMemcpyAsync(host_digests_ + first, digests_ + first, count * sizeof(std::uint32_t), hipMemcpyDeviceToHost, stream),
        "digest copy");
}

std::uint32_t DeviceCrc32c::value(std::size_t offset, std::size_t bytes, std::span<const std::byte> tail) const {
    const std::size_t first = offset / crc32c_fold_span_bytes, count = bytes / crc32c_fold_span_bytes;
    if (!tables_ || offset % crc32c_fold_span_bytes || first + count > spans_ || tail.size() != tail_bytes(bytes))
        throw std::invalid_argument("device crc32c: value of a run that was not enqueued");
    return crc32c_fold(std::span<const std::uint32_t>(host_digests_ + first, count), tail);
}
} // namespace strix::xport
