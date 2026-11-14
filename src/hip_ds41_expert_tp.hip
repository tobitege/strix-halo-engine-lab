#include "strix/hip_ds41_expert_tp.hpp"

#include "strix/ds41_expert_tp.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <string>

namespace strix::ds41::expert_tp {
namespace {

constexpr unsigned full_row = 756U, part_row = 420U, last = 336U;  // 9 and 5 Q2_K blocks; rank 1's block 4 at byte 336

// Source byte of byte j of a rank's packed row (pack_down_row): rank 0 the first 420 bytes; rank 1 blocks 5..8, then
// block 4 with scales 8..15 over 0..7 and the codes of groups 4..7 over those of groups 0..3.
__device__ __forceinline__ unsigned source_byte(unsigned rank, unsigned j) {
    if (rank == 0U) return j;
    if (j < last) return part_row + j;
    const unsigned k = j - last;
    if (k < 8U) return last + 8U + k;
    if (k >= 16U && k < 48U) return last + 32U + k;
    return last + k;
}

// The packed form (pack_down_row_packed, 380 bytes): 4 whole blocks, then the rank's half of block 4 (scales, codes, d/dmin).
constexpr unsigned packed_row = 380U;
__device__ __forceinline__ unsigned source_byte_packed(unsigned rank, unsigned j) {
    if (j < last) return rank == 0U ? j : part_row + j;
    const unsigned k = j - last;
    if (k < 8U) return last + 8U * rank + k;
    if (k < 40U) return last + 16U + 32U * rank + (k - 8U);
    return last + 80U + (k - 40U);
}

__global__ __launch_bounds__(256) void pack_down_kernel(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst,
                                                        unsigned long long rows, unsigned rank, bool packed) {
    const unsigned width = packed ? packed_row : part_row;
    const unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * 256U + threadIdx.x;
    if (i >= rows * width) return;
    const unsigned long long row = i / width;
    const unsigned j = static_cast<unsigned>(i % width);
    dst[i] = src[row * full_row + (packed ? source_byte_packed(rank, j) : source_byte(rank, j))];
}

}  // namespace

void enqueue_pack_down_rows(const std::uint8_t* src, std::uint8_t* dst, std::uint64_t rows, unsigned rank, bool packed, void* stream) {
    static_assert(down_row_bytes == part_row && down_row_bytes_packed == packed_row);
    if (!src || !dst || rows == 0U || rank >= ranks) throw std::invalid_argument("ds41 expert tp: pack down rows view");
    const unsigned long long n = rows * (packed ? packed_row : part_row);
    hipLaunchKernelGGL(pack_down_kernel, dim3(static_cast<unsigned>((n + 255U) / 256U)), dim3(256), 0, static_cast<hipStream_t>(stream),
                       src, dst, static_cast<unsigned long long>(rows), rank, packed);
    if (const auto e = hipGetLastError(); e != hipSuccess)
        throw std::runtime_error(std::string("ds41 expert tp: pack launch: ") + hipGetErrorString(e));
}

std::uint64_t part_bytes(std::uint64_t bytes, std::uint32_t experts, bool down, std::uint32_t hidden) {
    return down ? std::uint64_t{experts} * hidden * device_down_row_bytes() : bytes / 2U;
}

void load_part(const LoadFn& load, std::uint64_t offset, std::uint64_t bytes, std::uint32_t experts, bool down, std::uint32_t hidden,
               unsigned rank, std::uint8_t* out, void* stream, const std::function<void()>& loaded) {
    const auto check = [](hipError_t e, const char* what) {
        if (e != hipSuccess) throw std::runtime_error(std::string("ds41 expert tp: ") + what + ": " + hipGetErrorString(e));
    };
    const std::uint64_t E = experts, per = bytes / E;
    if (per * E != bytes || rank >= ranks) throw std::invalid_argument("ds41 expert tp: routed tensor geometry");
    if (down) {
        const std::uint64_t rows = hidden;
        if (per != rows * full_row) throw std::invalid_argument("ds41 expert tp: the down tensor is not Q2_K x 2304");
        constexpr std::uint64_t chunk = 16;
        std::uint8_t* staging = nullptr;
        check(hipMalloc(reinterpret_cast<void**>(&staging), chunk * per), "staging");
        try {
            for (std::uint64_t e = 0; e < E; e += chunk) {
                const auto n = std::min(chunk, E - e);
                load(offset + e * per, n * per, staging);
                if (loaded) loaded();  // the pack kernel reads the staging
                enqueue_pack_down_rows(staging, out + e * rows * device_down_row_bytes(), n * rows, rank, packed_from_env(), stream);
                check(hipStreamSynchronize(static_cast<hipStream_t>(stream)), "pack");  // before the staging is refilled
            }
        } catch (...) {
            (void)hipFree(staging);
            throw;
        }
        check(hipFree(staging), "staging free");
        return;
    }
    const std::uint64_t row = per / inter;
    if (row * inter != per) throw std::invalid_argument("ds41 expert tp: gate/up rows");
    for (std::uint64_t e = 0; e < E; ++e) {
        std::uint8_t* dst = out + e * row * half;
        for (std::uint32_t j = 0; j < half;) {  // runs of consecutive rows in the local order
            const auto g = inter_index(rank, j);
            std::uint32_t n = 1;
            while (j + n < half && inter_index(rank, j + n) == g + n) ++n;
            load(offset + e * per + std::uint64_t{g} * row, std::uint64_t{n} * row, dst + std::uint64_t{j} * row);
            j += n;
        }
    }
}

}  // namespace strix::ds41::expert_tp
