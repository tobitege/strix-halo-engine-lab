// Project implementation of the circular 16-bit-state compressed linear contract.
#include "strix/cuda/trellis.hpp"

#include <cuda_fp16.h>
#include <mma.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace strix::ds41::cuda::trellis {
namespace {
constexpr float hn = 0x1.6a09e6p-4F;
void need(bool ok, const char* msg) {
    if (!ok) throw std::invalid_argument(std::string("cuda trellis: ") + msg);
}
std::size_t mul(std::size_t a, std::size_t b) {
    need(b == 0 || a <= std::numeric_limits<std::size_t>::max() / b, "size overflow");
    return a * b;
}
std::size_t add(std::size_t a, std::size_t b) {
    need(a <= std::numeric_limits<std::size_t>::max() - b, "size overflow");
    return a + b;
}
std::size_t aligned(std::size_t n) { return add(n, 255) & ~std::size_t{255}; }
struct Span { std::uintptr_t begin, end; };
Span span(const void* p, std::size_t bytes, std::size_t align) {
    const auto a = reinterpret_cast<std::uintptr_t>(p);
    need(p != nullptr && a % align == 0, "null or misaligned device pointer");
    need(bytes <= std::numeric_limits<std::uintptr_t>::max() - a, "pointer span overflow");
    return {a, a + bytes};
}
bool overlaps(Span a, Span b) { return a.begin < b.end && b.begin < a.end; }
void validate_weights(const Weights& w) {
    need(w.packed_bytes == packed_bytes(w.input, w.output, w.bits), "packed byte count");
    need(w.suh_bytes == mul(2, w.input) && w.svh_bytes == mul(2, w.output), "scale byte count");
    need(w.bias_bytes == 0 || w.bias_bytes == mul(4, w.output), "bias byte count");
    need((w.bias == nullptr) == (w.bias_bytes == 0), "bias presence");
    need(w.marker == 0x83dcd12dU, "unsupported codebook marker");
}

__device__ __forceinline__ float hf(std::uint16_t v) { return __half2float(__ushort_as_half(v)); }
__device__ __forceinline__ std::uint16_t fh(float v) { return __half_as_ushort(__float2half_rn(v)); }
__device__ __forceinline__ std::uint16_t book(unsigned state) {
    const unsigned p = state * 0x83dcd12dU;
    const unsigned sum = (p & 255U) + ((p >> 8) & 255U) + ((p >> 16) & 255U) + (p >> 24);
    // The dyadic numerator is exactly representable in FP32; one RN16 only.
    const int numerator = static_cast<int>((1024U + sum) * 1774U) - 1329 * 2048;
    return fh(static_cast<float>(numerator) * 0x1p-18F);
}
__device__ __forceinline__ unsigned word(const std::uint16_t* p, unsigned i) {
    return unsigned(p[2 * i]) | (unsigned(p[2 * i + 1]) << 16);
}
template <int K>
__device__ __forceinline__ unsigned state(const std::uint16_t* p, unsigned pos) {
    const unsigned start = ((pos + 1) * K + 256 * K - 16) % (256 * K);
    const unsigned i = start / 32, shift = start % 32;
    const unsigned a = word(p, i), b = word(p, (i + 1) % (8 * K));
    return ((a << shift) | (shift ? b >> (32 - shift) : 0U)) >> 16;
}
__global__ void codebook_kernel(std::uint16_t* q) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 65536) q[i] = book(i);
}
template <int K>
__global__ void decode_kernel(Weights w, std::uint16_t* q) {
    const std::size_t tile = blockIdx.x;
    const unsigned pos = threadIdx.x, lane = pos / 8, f = pos % 8;
    const unsigned i = 2 * (lane % 4) + (f % 2) + 8 * ((f / 2) % 2);
    const unsigned o = lane / 4 + 8 * (f / 4);
    q[(tile / (w.output / 16) * 16 + i) * w.output + tile % (w.output / 16) * 16 + o] =
        book(state<K>(w.packed + tile * 16 * K, pos));
}
__global__ void input_kernel(Linear v, std::uint16_t* a) {
    __shared__ float h[128];
    const unsigned t = threadIdx.x;
    const std::size_t row = blockIdx.x / (v.weights.input / 128);
    const std::size_t c = (blockIdx.x % (v.weights.input / 128)) * 128 + t;
    h[t] = hf(fh(__fmul_rn(v.x[row * v.x_stride + c], hf(v.weights.suh[c]))));
    __syncthreads();
    for (unsigned s = 1; s < 128; s *= 2) {
        const float lo = h[t & ~s], hi = h[t | s];
        const float next = (t & s) ? __fsub_rn(lo, hi) : __fadd_rn(lo, hi);
        __syncthreads();
        h[t] = next;
        __syncthreads();
    }
    a[row * v.weights.input + c] = fh(__fmul_rn(h[t], hn));
}

// Each warp decodes one Q tile into registers, reuses it across all real rows,
// and keeps two accumulators per row. Four lanes own one output column.
// The explicit pair sum is a CUDA quality identity, not an emulation of fdot2.
template <int K, int R>
__global__ void gemv_kernel(Linear v, const std::uint16_t* a, float* partial) {
    const unsigned lane = threadIdx.x % 32;
    const unsigned n = blockIdx.x * 4 + threadIdx.x / 32;
    if (n >= v.weights.output / 16) return;  // whole warps only
    const unsigned split = blockIdx.y, tiles = v.weights.input / 16;
    const unsigned begin = tiles * split / v.splits, end = tiles * (split + 1) / v.splits;
    float acc[R][2] = {};
    for (unsigned k = begin; k < end; ++k) {
        const auto* p = v.weights.packed + (std::size_t{k} * (v.weights.output / 16) + n) * 16 * K;
        #pragma unroll
        for (unsigned f = 0; f < 4; ++f) {
            const float q0 = hf(book(state<K>(p, 8 * lane + 2 * f)));
            const float q1 = hf(book(state<K>(p, 8 * lane + 2 * f + 1)));
            const unsigned i = 16 * k + 2 * (lane & 3) + 8 * (f & 1);
            #pragma unroll
            for (int r = 0; r < R; ++r) {
                const auto* ar = a + static_cast<std::size_t>(r) * v.weights.input;
                const float pair = __fadd_rn(__fmul_rn(q0, hf(ar[i])), __fmul_rn(q1, hf(ar[i + 1])));
                acc[r][f / 2] = __fadd_rn(acc[r][f / 2], pair);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; ++r) {
        #pragma unroll
        for (unsigned h = 0; h < 2; ++h) {
            float z = acc[r][h];
            z = __fadd_rn(z, __shfl_down_sync(0xffffffffU, z, 2, 4));
            z = __fadd_rn(z, __shfl_down_sync(0xffffffffU, z, 1, 4));
            if ((lane & 3) == 0)
                partial[(std::size_t{split} * R + r) * v.weights.output + n * 16 + lane / 4 + 8 * h] = z;
        }
    }
}

template <int K>
__global__ void mma_kernel(Linear v, const std::uint16_t* a, float* partial) {
    using namespace nvcuda;
    __shared__ __align__(32) __half am[256], bm[16 * 64];
    __shared__ __align__(32) float cm[16 * 64];
    const unsigned t = threadIdx.x, warp = t / 32;
    const unsigned n0 = blockIdx.x * 64, r0 = blockIdx.y * 16, split = blockIdx.z;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> bf;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> cf;
    wmma::fill_fragment(cf, 0.0F);
    const unsigned tiles = v.weights.input / 16;
    for (unsigned k = tiles * split / v.splits; k < tiles * (split + 1) / v.splits; ++k) {
        for (unsigned j = t; j < 256; j += 128)
            am[j] = r0 + j / 16 < v.rows ? __ushort_as_half(a[std::size_t{r0 + j / 16} * v.weights.input + 16 * k + j % 16]) : __float2half_rn(0);
        for (unsigned j = t; j < 1024; j += 128) {
            const unsigned tile_n = j / 256, pos = j % 256, lane = pos / 8, f = pos % 8;
            const unsigned i = 2 * (lane % 4) + (f % 2) + 8 * ((f / 2) % 2);
            const unsigned o = tile_n * 16 + lane / 4 + 8 * (f / 4);
            const auto* p = v.weights.packed + (std::size_t{k} * (v.weights.output / 16) + n0 / 16 + tile_n) * 16 * K;
            bm[i * 64 + o] = __ushort_as_half(book(state<K>(p, pos)));
        }
        __syncthreads();
        wmma::load_matrix_sync(af, am, 16);
        wmma::load_matrix_sync(bf, bm + warp * 16, 64);
        wmma::mma_sync(cf, af, bf, cf);
        __syncthreads();
    }
    wmma::store_matrix_sync(cm + warp * 16, cf, 64, wmma::mem_row_major);
    __syncthreads();
    for (unsigned j = t; j < 1024; j += 128)
        if (r0 + j / 64 < v.rows)
            partial[(std::size_t{split} * v.rows + r0 + j / 64) * v.weights.output + n0 + j % 64] = cm[j];
}

__global__ void output_kernel(Linear v, const float* p) {
    __shared__ float h[128];
    const unsigned t = threadIdx.x;
    const std::size_t row = blockIdx.x / (v.weights.output / 128);
    const std::size_t c = blockIdx.x % (v.weights.output / 128) * 128 + t;
    float z = 0;
    for (unsigned s = 0; s < v.splits; ++s)
        z = __fadd_rn(z, p[(std::size_t{s} * v.rows + row) * v.weights.output + c]);
    h[t] = z;
    __syncthreads();
    for (unsigned s = 1; s < 128; s *= 2) {
        const float lo = h[t & ~s], hi = h[t | s];
        const float next = (t & s) ? __fsub_rn(lo, hi) : __fadd_rn(lo, hi);
        __syncthreads();
        h[t] = next;
        __syncthreads();
    }
    z = __fmul_rn(__fmul_rn(h[t], hn), hf(v.weights.svh[c]));
    if (v.weights.bias) z = __fadd_rn(z, v.weights.bias[c]);
    v.y[row * v.y_stride + c] = z;
}
template <int K>
void core(const Linear& v, const std::uint16_t* a, float* p, cudaStream_t st) {
    if (v.mode == Mode::mma_prefill_quality) {
        mma_kernel<K><<<dim3(v.weights.output / 64, (v.rows + 15) / 16, v.splits), 128, 0, st>>>(v, a, p);
    } else {
        const dim3 grid((v.weights.output / 16 + 3) / 4, v.splits);
        #define TRL_ROW(R) case R: gemv_kernel<K, R><<<grid, 128, 0, st>>>(v, a, p); break
        switch (v.rows) {
            TRL_ROW(1); TRL_ROW(2); TRL_ROW(3); TRL_ROW(4);
            TRL_ROW(5); TRL_ROW(6); TRL_ROW(7); TRL_ROW(8);
        }
        #undef TRL_ROW
    }
}
}  // namespace

std::size_t packed_bytes(std::uint32_t i, std::uint32_t o, std::uint32_t bits) {
    need(i && o && i % 128 == 0 && o % 128 == 0 && i <= 1048576 && o <= 1048576, "dimensions must be H128 aligned and <= 2^20");
    need(bits >= 1 && bits <= 8, "integer bits 1..8 required");
    need(mul(i / 16, o / 16) <= 2147483647U, "tile grid domain");
    return mul(mul(i, o) / 8, bits);
}
std::size_t workspace_bytes(const Linear& v) {
    validate_weights(v.weights);
    need(v.mode == Mode::gemv_quality || v.mode == Mode::mma_prefill_quality, "mode disabled or HIP bitwise identity unqualified");
    need(v.rows <= 65535 && (v.mode != Mode::gemv_quality || v.rows <= 8), "row domain");
    need(v.splits >= 1 && v.splits <= 32 && v.splits <= v.weights.input / 16, "split domain");
    need(v.x_stride >= v.weights.input && v.y_stride >= v.weights.output, "row stride");
    if (!v.rows) return 0;
    return add(aligned(mul(mul(2, v.rows), v.weights.input)), mul(mul(mul(4, v.rows), v.weights.output), v.splits));
}
void enqueue_linear(const Linear& v, cudaStream_t st) {
    const auto bytes = workspace_bytes(v);
    if (!v.rows) return;
    const auto xb = mul(add(mul(v.rows - 1, v.x_stride), v.weights.input), 4);
    const auto yb = mul(add(mul(v.rows - 1, v.y_stride), v.weights.output), 4);
    need(v.x_bytes >= xb && v.y_bytes >= yb && v.workspace_bytes >= bytes, "buffer capacity");
    const Span spans[] = {span(v.x, v.x_bytes, 4), span(v.y, v.y_bytes, 4), span(v.workspace, v.workspace_bytes, 256),
        span(v.weights.packed, v.weights.packed_bytes, 4), span(v.weights.suh, v.weights.suh_bytes, 2),
        span(v.weights.svh, v.weights.svh_bytes, 2)};
    // X/weights may share readonly storage; Y and workspace must be disjoint
    // from every other declared span, including padding.
    for (unsigned a : {1U, 2U})
        for (unsigned b = 0; b < 6; ++b)
            if (a != b) need(!overlaps(spans[a], spans[b]), "writable buffer overlap");
    if (v.weights.bias) {
        const auto b = span(v.weights.bias, v.weights.bias_bytes, 4);
        need(!overlaps(b, spans[1]) && !overlaps(b, spans[2]), "bias overlap");
    }
    auto* a = static_cast<std::uint16_t*>(v.workspace);
    auto* p = reinterpret_cast<float*>(static_cast<std::uint8_t*>(v.workspace) + aligned(mul(mul(2, v.rows), v.weights.input)));
    input_kernel<<<v.rows * (v.weights.input / 128), 128, 0, st>>>(v, a);
    STRIX_CUDA_CHECK(cudaGetLastError());
    #define TRL_BITS(K) case K: core<K>(v, a, p, st); break
    switch (v.weights.bits) { TRL_BITS(1); TRL_BITS(2); TRL_BITS(3); TRL_BITS(4); TRL_BITS(5); TRL_BITS(6); TRL_BITS(7); TRL_BITS(8); }
    #undef TRL_BITS
    STRIX_CUDA_CHECK(cudaGetLastError());
    output_kernel<<<v.rows * (v.weights.output / 128), 128, 0, st>>>(v, p);
    STRIX_CUDA_CHECK(cudaGetLastError());
}
void enqueue_decode(const Weights& w, std::uint16_t* q, std::size_t bytes, cudaStream_t st) {
    validate_weights(w);
    need(bytes >= mul(mul(w.input, w.output), 2), "decode capacity");
    const auto out = span(q, bytes, 2), packed = span(w.packed, w.packed_bytes, 4);
    need(!overlaps(out, packed), "decode overlap");
    const auto tiles = w.input / 16 * (w.output / 16);
    #define TRL_DECODE(K) case K: decode_kernel<K><<<tiles, 256, 0, st>>>(w, q); break
    switch (w.bits) { TRL_DECODE(1); TRL_DECODE(2); TRL_DECODE(3); TRL_DECODE(4); TRL_DECODE(5); TRL_DECODE(6); TRL_DECODE(7); TRL_DECODE(8); }
    #undef TRL_DECODE
    STRIX_CUDA_CHECK(cudaGetLastError());
}
void enqueue_codebook(std::uint16_t* values, std::size_t bytes, cudaStream_t st) {
    need(bytes >= 131072, "codebook capacity");
    (void)span(values, bytes, 2);
    codebook_kernel<<<256, 256, 0, st>>>(values);
    STRIX_CUDA_CHECK(cudaGetLastError());
}
}  // namespace strix::ds41::cuda::trellis
