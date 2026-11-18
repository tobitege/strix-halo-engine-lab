// RTX MTP draft kernels (see include/strix/cuda/qwen_mtp_draft.hpp). Free arithmetic: drafts never reach output.
#include "../../weight/arch/sm120_sources.cuh"
#include "strix/cuda/qwen_mtp_draft.hpp"
#include "strix/env_flags.hpp"

#include <cuda_fp16.h>

#include <stdexcept>
#include <string>

namespace strix::cuda::qwen_mtp_draft {
namespace {

constexpr unsigned kBlock = 34, kFull = 0xffffffffU;

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("qwen_mtp_draft ") + what + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float bf16f(std::uint16_t v) { return __uint_as_float(static_cast<unsigned>(v) << 16); }
__device__ __forceinline__ std::uint16_t f2bf(float f) {  // RNE
    const unsigned u = __float_as_uint(f);
    if ((u & 0x7fffffffU) > 0x7f800000U) return static_cast<std::uint16_t>((u >> 16) | 64U);
    return static_cast<std::uint16_t>((u + 0x7fffU + ((u >> 16) & 1U)) >> 16);
}
__device__ __forceinline__ float scale_of(const unsigned char* block) {
    return weight::Q8Block::scale(block);
}

__global__ void q8_to_f32_kernel(const unsigned char* q8, std::uint64_t elements, float* out) {
    const std::uint64_t i = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= elements) return;
    const unsigned char* b = q8 + i / 32U * kBlock;
    out[i] = scale_of(b) * static_cast<float>(weight::Q8Block::code(b, i % 32U));
}

__global__ void gather_kernel(const unsigned char* src, std::uint64_t row_bytes, const std::uint32_t* ids, unsigned char* dst) {
    const unsigned char* s = src + static_cast<std::uint64_t>(ids[blockIdx.x]) * row_bytes;
    unsigned char* d = dst + static_cast<std::uint64_t>(blockIdx.x) * row_bytes;
    for (std::uint64_t i = threadIdx.x; i < row_bytes; i += blockDim.x) d[i] = s[i];
}

// warp dot of a Q8_0 row (n features) with FP32 x in shared memory
template <unsigned N>
__device__ __forceinline__ float dot_q8(const unsigned char* row, const float* x, unsigned lane) {
    float acc = 0.0F;
    for (unsigned b = lane; b < N / 32U; b += 32U) {
        const unsigned char* blk = row + b * kBlock;
        float s = 0.0F;
#pragma unroll
        for (unsigned j = 0; j < 32U; ++j) s += x[b * 32U + j] * static_cast<float>(weight::Q8Block::code(blk, j));
        acc += scale_of(blk) * s;
    }
#pragma unroll
    for (unsigned d = 16U; d; d /= 2U) acc += __shfl_xor_sync(kFull, acc, d);
    return acc;
}

// grid (kFfn / 8, kTopK), 256 threads: warp per ffn row of one slot
__global__ __launch_bounds__(256) void gate_up_kernel(const std::uint16_t* hidden, const std::uint32_t* routes, const unsigned char* gate,
    const unsigned char* up, float* scratch) {
    __shared__ float x[kHidden];
    for (unsigned i = threadIdx.x; i < kHidden; i += blockDim.x) x[i] = bf16f(hidden[i]);
    __syncthreads();
    const unsigned slot = blockIdx.y, r = blockIdx.x * 8U + threadIdx.x / 32U, lane = threadIdx.x % 32U;
    const std::uint64_t expert_bytes = std::uint64_t{kFfn} * (kHidden / 32U) * kBlock, row_bytes = (kHidden / 32U) * kBlock;
    const std::uint64_t off = routes[slot] * expert_bytes + r * row_bytes;
    const float g = dot_q8<kHidden>(gate + off, x, lane), u = dot_q8<kHidden>(up + off, x, lane);
    if (lane == 0U) scratch[slot * kFfn + r] = g / (1.0F + __expf(-g)) * u;
}

// grid kHidden / 8, 256 threads: warp per hidden output over all slots
__global__ __launch_bounds__(256) void down_kernel(const std::uint32_t* routes, const float* weights, const unsigned char* down,
    const float* scratch, std::uint16_t* out) {
    __shared__ float h[kTopK * kFfn];
    for (unsigned i = threadIdx.x; i < kTopK * kFfn; i += blockDim.x) h[i] = scratch[i];
    __syncthreads();
    const unsigned o = blockIdx.x * 8U + threadIdx.x / 32U, lane = threadIdx.x % 32U;
    const std::uint64_t expert_bytes = std::uint64_t{kHidden} * (kFfn / 32U) * kBlock, row_bytes = (kFfn / 32U) * kBlock;
    float sum = 0.0F;
    for (unsigned s = 0; s < kTopK; ++s)
        sum += weights[s] * dot_q8<kFfn>(down + routes[s] * expert_bytes + o * row_bytes, h + s * kFfn, lane);
    if (lane == 0U) out[o] = f2bf(sum);
}

__global__ __launch_bounds__(1024) void select_kernel(const float* logits, std::uint32_t count, const std::uint32_t* ids,
    std::uint32_t* out, std::uint32_t* next_token) {
    __shared__ float bv[1024];
    __shared__ unsigned bi[1024];
    float best = __int_as_float(0xff800000);
    unsigned index = 0xffffffffU;
    for (unsigned i = threadIdx.x; i < count; i += blockDim.x)
        if (logits[i] > best) { best = logits[i]; index = i; }
    bv[threadIdx.x] = best;
    bi[threadIdx.x] = index;
    __syncthreads();
    for (unsigned d = 512U; d; d /= 2U) {
        if (threadIdx.x < d) {
            const float v = bv[threadIdx.x + d];
            const unsigned j = bi[threadIdx.x + d];
            if (v > bv[threadIdx.x] || (v == bv[threadIdx.x] && j < bi[threadIdx.x])) { bv[threadIdx.x] = v; bi[threadIdx.x] = j; }
        }
        __syncthreads();
    }
    const float m = bv[0];
    const unsigned arg = bi[0];
    __syncthreads();
    float s = 0.0F;
    for (unsigned i = threadIdx.x; i < count; i += blockDim.x) s += __expf(logits[i] - m);
    bv[threadIdx.x] = s;
    __syncthreads();
    for (unsigned d = 512U; d; d /= 2U) {
        if (threadIdx.x < d) bv[threadIdx.x] += bv[threadIdx.x + d];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const unsigned token = arg < count ? ids[arg] : 0U;
        out[0] = token;
        out[1] = __float_as_uint(arg < count ? 1.0F / bv[0] : 0.0F);
        if (next_token) *next_token = token;
    }
}

// ---- Decode forms (STRIX_QRTX_MTP_DRAFT, default on): identical per-output FP32 chains as above; x/h staged in smem with a
// 36-float block stride (conflict-free float4 reads), weight rows staged by coalesced wide loads or read as u16, wider batches
// of loads in flight in select. Rows/experts/blocks are visited in the same order, so results are byte-identical.
constexpr unsigned kXs = 36;   // smem floats per 32-feature block
__device__ __forceinline__ unsigned short ld16(const unsigned char* p) { return *reinterpret_cast<const unsigned short*>(p); }

// warp dot of one Q8_0 row held in smem/global (2-byte aligned blocks) with x[b * kXs + j]
template <unsigned N>
__device__ __forceinline__ float dot_q8_v2(const unsigned char* row, const float* x, unsigned lane) {
    float acc = 0.0F;
    for (unsigned b = lane; b < N / 32U; b += 32U) {
        const unsigned char* blk = row + b * kBlock;
        const float scale = __half2float(__ushort_as_half(ld16(blk)));
        float s = 0.0F;
#pragma unroll
        for (unsigned j = 0; j < 32U; j += 4U) {
            const float4 xv = *reinterpret_cast<const float4*>(x + b * kXs + j);
            const unsigned a = ld16(blk + 2U + j), c = ld16(blk + 4U + j);
            s += xv.x * static_cast<float>(static_cast<signed char>(a & 255U));
            s += xv.y * static_cast<float>(static_cast<signed char>(a >> 8U));
            s += xv.z * static_cast<float>(static_cast<signed char>(c & 255U));
            s += xv.w * static_cast<float>(static_cast<signed char>(c >> 8U));
        }
        acc += scale * s;
    }
#pragma unroll
    for (unsigned d = 16U; d; d /= 2U) acc += __shfl_xor_sync(kFull, acc, d);
    return acc;
}

// grid (kFfn / 4, kTopK), 128 threads: warp per ffn row of one slot; gate/up rows staged (2720 B each, 170 uint4)
__global__ __launch_bounds__(128) void gate_up_v2_kernel(const std::uint16_t* hidden, const std::uint32_t* routes, const unsigned char* gate,
    const unsigned char* up, float* scratch) {
    constexpr unsigned kRow = (kHidden / 32U) * kBlock, kVec = kRow / 16U;   // 2720, 170
    __shared__ __align__(16) float x[(kHidden / 32U) * kXs];
    __shared__ __align__(16) unsigned char wst[4][2][kRow];
    const unsigned slot = blockIdx.y, w = threadIdx.x / 32U, r = blockIdx.x * 4U + w, lane = threadIdx.x % 32U;
    const std::uint64_t expert_bytes = std::uint64_t{kFfn} * kRow;
    const std::uint64_t off = routes[slot] * expert_bytes + r * std::uint64_t{kRow};
    // issue weight loads first (independent of x)
    uint4 tg[(kVec + 31U) / 32U], tu[(kVec + 31U) / 32U];
    const uint4* sg = reinterpret_cast<const uint4*>(gate + off);
    const uint4* su = reinterpret_cast<const uint4*>(up + off);
#pragma unroll
    for (unsigned i = 0; i < (kVec + 31U) / 32U; ++i) if (lane + 32U * i < kVec) { tg[i] = sg[lane + 32U * i]; tu[i] = su[lane + 32U * i]; }
    for (unsigned i = threadIdx.x; i < kHidden; i += blockDim.x) x[(i / 32U) * kXs + i % 32U] = bf16f(hidden[i]);
#pragma unroll
    for (unsigned i = 0; i < (kVec + 31U) / 32U; ++i)
        if (lane + 32U * i < kVec) {
            reinterpret_cast<uint4*>(wst[w][0])[lane + 32U * i] = tg[i];
            reinterpret_cast<uint4*>(wst[w][1])[lane + 32U * i] = tu[i];
        }
    __syncthreads();
    const float g = dot_q8_v2<kHidden>(wst[w][0], x, lane), u = dot_q8_v2<kHidden>(wst[w][1], x, lane);
    if (lane == 0U) scratch[slot * kFfn + r] = g / (1.0F + __expf(-g)) * u;
}

// grid kHidden / 4, 128 threads: warp per hidden output over all slots; weight rows (680 B) staged per half of the slots
__global__ __launch_bounds__(128) void down_v2_kernel(const std::uint32_t* routes, const float* weights, const unsigned char* down,
    const float* scratch, std::uint16_t* out) {
    constexpr unsigned kRow = (kFfn / 32U) * kBlock, kVec = kRow / 8U, kBlk = kFfn / 32U, kHalf = kTopK / 2U;   // 680, 85, 20, 5
    __shared__ __align__(16) float h[kTopK * kBlk * kXs];
    __shared__ __align__(16) unsigned char wst[4][kHalf][kRow];
    const unsigned w = threadIdx.x / 32U, o = blockIdx.x * 4U + w, lane = threadIdx.x % 32U;
    const std::uint64_t expert_bytes = std::uint64_t{kHidden} * kRow;
    float sum = 0.0F;
    for (unsigned half = 0; half < 2U; ++half) {
        uint2 t[kHalf][(kVec + 31U) / 32U];
#pragma unroll
        for (unsigned s = 0; s < kHalf; ++s) {
            const uint2* src = reinterpret_cast<const uint2*>(down + routes[half * kHalf + s] * expert_bytes + o * std::uint64_t{kRow});
#pragma unroll
            for (unsigned i = 0; i < (kVec + 31U) / 32U; ++i) if (lane + 32U * i < kVec) t[s][i] = src[lane + 32U * i];
        }
        if (half == 0U)
            for (unsigned i = threadIdx.x; i < kTopK * kFfn; i += blockDim.x) h[(i / kFfn) * (kBlk * kXs) + ((i % kFfn) / 32U) * kXs + i % 32U] = scratch[i];
#pragma unroll
        for (unsigned s = 0; s < kHalf; ++s)
#pragma unroll
            for (unsigned i = 0; i < (kVec + 31U) / 32U; ++i) if (lane + 32U * i < kVec) reinterpret_cast<uint2*>(wst[w][s])[lane + 32U * i] = t[s][i];
        __syncthreads();
#pragma unroll
        for (unsigned s = 0; s < kHalf; ++s) {
            const unsigned slot = half * kHalf + s;
            sum += weights[slot] * dot_q8_v2<kFfn>(wst[w][s], h + slot * (kBlk * kXs), lane);
        }
        __syncwarp();
    }
    if (lane == 0U) out[o] = f2bf(sum);
}

constexpr unsigned kSelU = 16;   // loads in flight per thread
__global__ __launch_bounds__(1024) void select_v2_kernel(const float* logits, std::uint32_t count, const std::uint32_t* ids,
    std::uint32_t* out, std::uint32_t* next_token) {
    __shared__ float bv[1024];
    __shared__ unsigned bi[1024];
    float best = __int_as_float(0xff800000);
    unsigned index = 0xffffffffU;
    for (unsigned i0 = threadIdx.x; i0 < count; i0 += 1024U * kSelU) {
        float l[kSelU];
#pragma unroll
        for (unsigned u = 0; u < kSelU; ++u) { const unsigned i = i0 + 1024U * u; l[u] = i < count ? logits[i] : 0.0F; }
#pragma unroll
        for (unsigned u = 0; u < kSelU; ++u) { const unsigned i = i0 + 1024U * u; if (i < count && l[u] > best) { best = l[u]; index = i; } }
    }
    bv[threadIdx.x] = best;
    bi[threadIdx.x] = index;
    __syncthreads();
    for (unsigned d = 512U; d; d /= 2U) {
        if (threadIdx.x < d) {
            const float v = bv[threadIdx.x + d];
            const unsigned j = bi[threadIdx.x + d];
            if (v > bv[threadIdx.x] || (v == bv[threadIdx.x] && j < bi[threadIdx.x])) { bv[threadIdx.x] = v; bi[threadIdx.x] = j; }
        }
        __syncthreads();
    }
    const float m = bv[0];
    const unsigned arg = bi[0];
    __syncthreads();
    float s = 0.0F;
    for (unsigned i0 = threadIdx.x; i0 < count; i0 += 1024U * kSelU) {
        float l[kSelU];
#pragma unroll
        for (unsigned u = 0; u < kSelU; ++u) { const unsigned i = i0 + 1024U * u; l[u] = i < count ? logits[i] : 0.0F; }
#pragma unroll
        for (unsigned u = 0; u < kSelU; ++u) { const unsigned i = i0 + 1024U * u; if (i < count) s += __expf(l[u] - m); }
    }
    bv[threadIdx.x] = s;
    __syncthreads();
    for (unsigned d = 512U; d; d /= 2U) {
        if (threadIdx.x < d) bv[threadIdx.x] += bv[threadIdx.x + d];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const unsigned token = arg < count ? ids[arg] : 0U;
        out[0] = token;
        out[1] = __float_as_uint(arg < count ? 1.0F / bv[0] : 0.0F);
        if (next_token) *next_token = token;
    }
}

bool draft_v2_enabled() {
    static const bool on = strix::envcfg::flag("STRIX_QRTX_MTP_DRAFT", true);
    return on;
}

}  // namespace

void q8_to_f32(const void* q8, std::uint64_t elements, float* out, cudaStream_t stream) {
    q8_to_f32_kernel<<<static_cast<unsigned>((elements + 255U) / 256U), 256, 0, stream>>>(static_cast<const unsigned char*>(q8), elements, out);
    check(cudaGetLastError(), "q8_to_f32");
}
void gather_rows(const void* src, std::uint64_t row_bytes, const std::uint32_t* ids, std::uint32_t count, void* dst,
    cudaStream_t stream) {
    if (!count) return;
    gather_kernel<<<count, 256, 0, stream>>>(static_cast<const unsigned char*>(src), row_bytes, ids, static_cast<unsigned char*>(dst));
    check(cudaGetLastError(), "gather");
}
void routed_q8_one_row(const std::uint16_t* hidden, const std::uint32_t* routes, const float* route_weights, const void* gate,
    const void* up, const void* down, float* scratch, std::uint16_t* out, cudaStream_t stream) {
    const auto al = [](const void* p, std::uintptr_t a) { return (reinterpret_cast<std::uintptr_t>(p) & (a - 1U)) == 0U; };
    const bool v2 = draft_v2_enabled() && al(gate, 16) && al(up, 16) && al(down, 8);
    if (v2) gate_up_v2_kernel<<<dim3(kFfn / 4U, kTopK), 128, 0, stream>>>(hidden, routes, static_cast<const unsigned char*>(gate),
        static_cast<const unsigned char*>(up), scratch);
    else gate_up_kernel<<<dim3(kFfn / 8U, kTopK), 256, 0, stream>>>(hidden, routes, static_cast<const unsigned char*>(gate),
        static_cast<const unsigned char*>(up), scratch);
    check(cudaGetLastError(), "gate/up");
    if (v2) down_v2_kernel<<<kHidden / 4U, 128, 0, stream>>>(routes, route_weights, static_cast<const unsigned char*>(down), scratch, out);
    else down_kernel<<<kHidden / 8U, 256, 0, stream>>>(routes, route_weights, static_cast<const unsigned char*>(down), scratch, out);
    check(cudaGetLastError(), "down");
}
void select(const float* logits, std::uint32_t count, const std::uint32_t* ids, std::uint32_t* out, std::uint32_t* next_token,
    cudaStream_t stream) {
    if (draft_v2_enabled()) select_v2_kernel<<<1, 1024, 0, stream>>>(logits, count, ids, out, next_token);
    else select_kernel<<<1, 1024, 0, stream>>>(logits, count, ids, out, next_token);
    check(cudaGetLastError(), "select");
}

}  // namespace strix::cuda::qwen_mtp_draft
