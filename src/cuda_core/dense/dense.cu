// Dense rows 1..16: BF16 / Q8_0 / F32 weights, FP32 accumulation in the fixed order documented in dense.hpp.
#include "strix/cuda_core/dense.hpp"
#include "strix/cuda_core/nvfp4.hpp"

#include "../common.cuh"
#include "dense_mma.hpp"

namespace strix::cuda_core {
namespace {

// acc <- acc + w * x in the documented order. With CC_DENSE_FMA the product is not rounded separately (one fused rounding).
__device__ __forceinline__ float mac(float acc, float w, float x) {
#ifdef CC_DENSE_FMA
    return __fmaf_rn(w, x, acc);
#else
    return __fadd_rn(acc, __fmul_rn(w, x));
#endif
}

// ORO consecutive output rows per warp: the x values of a chunk are loaded once and applied to every weight row (identical
// per-(output, row) arithmetic and order as ORO = 1).
template <int R, int ORO>
__device__ __forceinline__ void dot_bf16(const std::uint16_t* wrow, std::size_t row_stride, const float* x, std::uint32_t in,
                                         std::uint32_t rows, int lane, float (&acc)[ORO][R]) {
    const std::uint32_t nch = in >> 3;
#pragma unroll 2
    for (std::uint32_t c = lane; c < nch; c += 32) {
        float w[ORO][8];
#pragma unroll
        for (int oi = 0; oi < ORO; ++oi) {
            const uint4 u = __ldg(reinterpret_cast<const uint4*>(wrow + oi * row_stride) + c);
            w[oi][0] = bf16_lo(u.x); w[oi][1] = bf16_hi(u.x); w[oi][2] = bf16_lo(u.y); w[oi][3] = bf16_hi(u.y);
            w[oi][4] = bf16_lo(u.z); w[oi][5] = bf16_hi(u.z); w[oi][6] = bf16_lo(u.w); w[oi][7] = bf16_hi(u.w);
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (static_cast<std::uint32_t>(r) < rows) {
                const float4* xp = reinterpret_cast<const float4*>(x + static_cast<std::size_t>(r) * in + c * 8);
                const float4 a = xp[0], b2 = xp[1];
                const float xv[8] = {a.x, a.y, a.z, a.w, b2.x, b2.y, b2.z, b2.w};
#pragma unroll
                for (int oi = 0; oi < ORO; ++oi) {
                    float sacc = acc[oi][r];
#pragma unroll
                    for (int j = 0; j < 8; ++j) sacc = mac(sacc, w[oi][j], xv[j]);
                    acc[oi][r] = sacc;
                }
            }
        }
    }
}

template <int R>
__device__ __forceinline__ void dot_f32(const float* wrow, const float* x, std::uint32_t in, std::uint32_t rows, int lane,
                                        float (&acc)[R]) {
    const std::uint32_t nch = in >> 3;
    for (std::uint32_t c = lane; c < nch; c += 32) {
        const float4 wa = reinterpret_cast<const float4*>(wrow)[c * 2], wb = reinterpret_cast<const float4*>(wrow)[c * 2 + 1];
        const float w[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (static_cast<std::uint32_t>(r) < rows) {
                const float4* xp = reinterpret_cast<const float4*>(x + static_cast<std::size_t>(r) * in + c * 8);
                const float4 a = xp[0], b = xp[1];
                const float xv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
                float s = acc[r];
#pragma unroll
                for (int j = 0; j < 8; ++j) s = __fadd_rn(s, __fmul_rn(w[j], xv[j]));
                acc[r] = s;
            }
        }
    }
}

template <int R>
__device__ __forceinline__ void dot_q8(const unsigned char* wrow, const float* x, std::uint32_t in, std::uint32_t rows,
                                       int lane, float (&acc)[R]) {
    const std::uint32_t nb = in >> 5;
    for (std::uint32_t b = lane; b < nb; b += 32) {
        const std::uint16_t* p = reinterpret_cast<const std::uint16_t*>(wrow + static_cast<std::size_t>(b) * 34);
        const float d = __half2float(__ushort_as_half(__ldg(p)));
        float q[32];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const std::uint32_t v = __ldg(p + 1 + j);
            q[2 * j] = static_cast<float>(static_cast<std::int8_t>(v & 0xFF));
            q[2 * j + 1] = static_cast<float>(static_cast<std::int8_t>(v >> 8));
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            if (static_cast<std::uint32_t>(r) < rows) {
                const float4* xp = reinterpret_cast<const float4*>(x + static_cast<std::size_t>(r) * in + b * 32);
                float s = 0.0f;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const float4 a = xp[k];
                    s = __fadd_rn(s, __fmul_rn(q[4 * k], a.x));
                    s = __fadd_rn(s, __fmul_rn(q[4 * k + 1], a.y));
                    s = __fadd_rn(s, __fmul_rn(q[4 * k + 2], a.z));
                    s = __fadd_rn(s, __fmul_rn(q[4 * k + 3], a.w));
                }
                acc[r] = __fadd_rn(acc[r], __fmul_rn(d, s));
            }
        }
    }
}

template <WeightFormat F, int R, int ORO>
__device__ __forceinline__ void dot_any(const void* w, std::uint32_t o, const float* x, std::uint32_t in, std::uint32_t rows,
                                        int lane, float (&acc)[ORO][R]) {
    if constexpr (F == WeightFormat::bf16) {
        dot_bf16<R, ORO>(static_cast<const std::uint16_t*>(w) + static_cast<std::size_t>(o) * in, in, x, in, rows, lane, acc);
    } else {
        static_assert(ORO == 1, "only bf16 has multi-row warps");
        float(&a1)[R] = acc[0];
        if constexpr (F == WeightFormat::f32)
            dot_f32<R>(static_cast<const float*>(w) + static_cast<std::size_t>(o) * in, x, in, rows, lane, a1);
        else
            dot_q8<R>(static_cast<const unsigned char*>(w) + static_cast<std::size_t>(o) * (in >> 5) * 34, x, in, rows, lane, a1);
    }
}

// Up to four weights sharing x (and format/in) are served by ONE launch: warp g of the grid owns global output row g, mapped to
// (weight j, row o). The per-row arithmetic does not depend on the mapping.
constexpr int kMaxMulti = 4;
constexpr int kWarpsPerBlock = 8;  // 128-thread blocks: finer grid -> less wave-quantisation tail on 188 SMs
struct MultiArgs {
    const void* w[kMaxMulti];
    float* y[kMaxMulti];
    std::uint32_t out[kMaxMulti];
    std::uint32_t row_end[kMaxMulti];  // cumulative rows
    int n;
};

// L2 prefetch of the first 4 KiB of this warp's weight row while the predecessor kernel is still running (weights do not depend on it).
template <WeightFormat F>
__device__ __forceinline__ void prefetch_row(const void* w, std::uint32_t o, std::uint32_t in, int lane) {
    const std::size_t row_bytes = F == WeightFormat::bf16 ? static_cast<std::size_t>(in) * 2
                                  : F == WeightFormat::f32 ? static_cast<std::size_t>(in) * 4
                                                           : static_cast<std::size_t>(in >> 5) * 34;
    const char* p = static_cast<const char*>(w) + static_cast<std::size_t>(o) * row_bytes + static_cast<std::size_t>(lane) * 128;
    if (static_cast<std::size_t>(lane) * 128 < row_bytes) asm volatile("prefetch.global.L2 [%0];" ::"l"(p));
}

template <WeightFormat F, int R, bool FUSED, int ORO>
__global__ void __launch_bounds__(kWarpsPerBlock * 32)
    dense_kernel(MultiArgs ma, const void* w1, const float* x, const float* resid, std::uint32_t in, std::uint32_t rows) {
    pdl_trigger();
    const std::uint32_t g = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);  // warp group: ORO consecutive output rows
    if (g >= ma.row_end[kMaxMulti - 1]) {  // row_end counts groups and is padded with the total
        pdl_wait();
        return;
    }
    // compile-time unrolled selection: no dynamic indexing of the kernel parameter array (would force a local-memory copy)
    const void* wsel = ma.w[0];
    float* y = ma.y[0];
    std::uint32_t start = 0, out = ma.out[0];
#pragma unroll
    for (int jj = 0; jj < kMaxMulti - 1; ++jj) {
        if (g >= ma.row_end[jj]) {
            start = ma.row_end[jj];
            wsel = ma.w[jj + 1];
            y = ma.y[jj + 1];
            out = ma.out[jj + 1];
        }
    }
    const std::uint32_t o0 = (g - start) * ORO;
    const int lane = threadIdx.x & 31;
    prefetch_row<F>(wsel, o0, in, lane);
    if constexpr (FUSED) prefetch_row<F>(w1, o0, in, lane);
    pdl_wait();
    float a0[ORO][R], a1[ORO][R];
#pragma unroll
    for (int oi = 0; oi < ORO; ++oi)
#pragma unroll
        for (int r = 0; r < R; ++r) a0[oi][r] = a1[oi][r] = 0.0f;
    dot_any<F, R, ORO>(wsel, o0, x, in, rows, lane, a0);
    if constexpr (FUSED) dot_any<F, R, ORO>(w1, o0, x, in, rows, lane, a1);
#pragma unroll
    for (int oi = 0; oi < ORO; ++oi) {
        const std::uint32_t o = o0 + oi;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float gsum = warp_sum_xor(a0[oi][r]);
            float v = gsum;
            if constexpr (FUSED) {
                const float u = warp_sum_xor(a1[oi][r]);
                v = __fmul_rn(cc_silu(gsum), u);
            } else if (resid != nullptr) {
                const std::size_t idx = static_cast<std::size_t>(r) * out + o;
                if (static_cast<std::uint32_t>(r) < rows) v = __fadd_rn(resid[idx], gsum);
            }
            if (lane == 0 && static_cast<std::uint32_t>(r) < rows) y[static_cast<std::size_t>(r) * out + o] = v;
        }
    }
}

template <WeightFormat F, int R, bool FUSED, int ORO>
cudaError_t launch(const MultiArgs& ma, const void* w1, const float* x, const float* resid, std::uint32_t in, std::uint32_t rows,
                   cudaStream_t s) {
    const std::uint32_t total = ma.row_end[kMaxMulti - 1];
    return launch_pdl(dense_kernel<F, R, FUSED, ORO>, dim3((total + kWarpsPerBlock - 1) / kWarpsPerBlock), dim3(kWarpsPerBlock * 32), s,
                      ma, w1, x, resid, in, rows);
}

// Output rows per warp by row count (bf16 only): more rows share each x load as the row count grows.
template <WeightFormat F, int R, bool FUSED>
cudaError_t launch_oro(const MultiArgs& ma, std::uint32_t oro, const void* w1, const float* x, const float* resid, std::uint32_t in,
                       std::uint32_t rows, cudaStream_t s) {
    if constexpr (F == WeightFormat::bf16) {
        if constexpr (R >= 4 && R <= 8) {
            if (oro == 4) return launch<F, R, FUSED, 4>(ma, w1, x, resid, in, rows, s);
        }
        if constexpr (R >= 16 && !(FUSED && R > 16)) {
            if (oro == 2) return launch<F, R, FUSED, 2>(ma, w1, x, resid, in, rows, s);
        }
    }
    return launch<F, R, FUSED, 1>(ma, w1, x, resid, in, rows, s);
}

template <WeightFormat F, bool FUSED>
cudaError_t by_rows(const MultiArgs& ma, std::uint32_t oro, const void* w1, const float* x, const float* resid, std::uint32_t in,
                    std::uint32_t rows, cudaStream_t s) {
    if (rows <= 1) return launch_oro<F, 1, FUSED>(ma, oro, w1, x, resid, in, rows, s);
    if (rows <= 2) return launch_oro<F, 2, FUSED>(ma, oro, w1, x, resid, in, rows, s);
    if (rows <= 4) return launch_oro<F, 4, FUSED>(ma, oro, w1, x, resid, in, rows, s);
    if (rows <= 8) return launch_oro<F, 8, FUSED>(ma, oro, w1, x, resid, in, rows, s);
    if (rows <= 16) return launch_oro<F, 16, FUSED>(ma, oro, w1, x, resid, in, rows, s);
    return launch_oro<F, 32, FUSED>(ma, oro, w1, x, resid, in, rows, s);
}

bool valid(const DenseWeight& w, std::uint32_t rows) {
    if (w.data == nullptr || rows == 0 || rows > kMaxRows || w.out == 0) return false;
    return w.format == WeightFormat::q8_0 ? (w.in % 32 == 0) : (w.in % 8 == 0);
}

// ma.row_end is given in ROWS here; it is converted to groups of `oro` rows when every weight has out % oro == 0.
template <bool FUSED>
cudaError_t dispatch(MultiArgs ma, const DenseWeight& a, const void* w1, const float* x, const float* resid,
                     std::uint32_t rows, cudaStream_t s) {
    std::uint32_t oro = 1;
    if (a.format == WeightFormat::bf16 && rows >= 4) {
        const bool small = rows <= 8;  // row bucket 4 / 8 -> 4 outputs per warp; bucket 16 / 32 -> 2 (32 only unfused)
        oro = small ? 4 : ((FUSED && rows > 16) ? 1 : 2);
        for (int j = 0; j < ma.n; ++j) {
            if (ma.out[j] % oro != 0) oro = 1;
        }
        if (oro > 1) {
            for (int j = 0; j < kMaxMulti; ++j) ma.row_end[j] /= oro;
        }
    }
    switch (a.format) {
        case WeightFormat::bf16: return by_rows<WeightFormat::bf16, FUSED>(ma, oro, w1, x, resid, a.in, rows, s);
        case WeightFormat::q8_0: return by_rows<WeightFormat::q8_0, FUSED>(ma, 1, w1, x, resid, a.in, rows, s);
        case WeightFormat::f32: return by_rows<WeightFormat::f32, FUSED>(ma, 1, w1, x, resid, a.in, rows, s);
    }
    return cudaErrorInvalidValue;
}

__global__ void embedding_kernel(const unsigned char* table, WeightFormat fmt, std::uint32_t in, const std::uint32_t* ids,
                                 float* out) {
    pdl_trigger();
    pdl_wait();
    const std::size_t id = ids[blockIdx.x];
    float* o = out + static_cast<std::size_t>(blockIdx.x) * in;
    for (std::uint32_t i = threadIdx.x; i < in; i += blockDim.x) {
        float v;
        if (fmt == WeightFormat::bf16) {
            v = __uint_as_float(static_cast<std::uint32_t>(reinterpret_cast<const std::uint16_t*>(table)[id * in + i]) << 16);
        } else if (fmt == WeightFormat::f32) {
            v = reinterpret_cast<const float*>(table)[id * in + i];
        } else {
            const unsigned char* blk = table + (id * (in >> 5) + (i >> 5)) * 34;
            const std::uint16_t hb = static_cast<std::uint16_t>(blk[0] | (blk[1] << 8));
            v = __fmul_rn(__half2float(__ushort_as_half(hb)), static_cast<float>(static_cast<std::int8_t>(blk[2 + (i & 31)])));
        }
        o[i] = v;
    }
}

// Two-stage argmax, lowest index wins ties. Stage 1: block (c, r) scans a contiguous chunk of row r; stage 2: one block per row.
constexpr std::uint32_t kArgmaxChunks = 128;
__device__ __forceinline__ bool better(float v, std::uint32_t i, float bv, std::uint32_t bi) {
    return bi == 0xFFFFFFFFu || v > bv || (v == bv && i < bi);
}
__global__ void argmax_stage1(const float* logits, std::uint32_t vocab, float* pv, std::uint32_t* pi) {
    pdl_trigger();
    pdl_wait();
    const float* l = logits + static_cast<std::size_t>(blockIdx.y) * vocab;
    const std::uint32_t per = (vocab + kArgmaxChunks - 1) / kArgmaxChunks;
    const std::uint32_t lo = blockIdx.x * per, hi = min(vocab, lo + per);
    float best = 0.0f;
    std::uint32_t bi = 0xFFFFFFFFu;
    for (std::uint32_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
        const float v = l[i];
        if (better(v, i, best, bi)) {
            best = v;
            bi = i;
        }
    }
    __shared__ float sv[256];
    __shared__ std::uint32_t si[256];
    sv[threadIdx.x] = best;
    si[threadIdx.x] = bi;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && si[threadIdx.x + s] != 0xFFFFFFFFu && better(sv[threadIdx.x + s], si[threadIdx.x + s], sv[threadIdx.x], si[threadIdx.x])) {
            sv[threadIdx.x] = sv[threadIdx.x + s];
            si[threadIdx.x] = si[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        pv[blockIdx.y * kArgmaxChunks + blockIdx.x] = sv[0];
        pi[blockIdx.y * kArgmaxChunks + blockIdx.x] = si[0];
    }
}
__global__ void argmax_stage2(const float* pv, const std::uint32_t* pi, std::uint32_t* out) {
    pdl_wait();
    const std::size_t base = static_cast<std::size_t>(blockIdx.x) * kArgmaxChunks;
    __shared__ float sv[kArgmaxChunks];
    __shared__ std::uint32_t si[kArgmaxChunks];
    sv[threadIdx.x] = pv[base + threadIdx.x];
    si[threadIdx.x] = pi[base + threadIdx.x];
    __syncthreads();
    for (unsigned s = kArgmaxChunks / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s && si[threadIdx.x + s] != 0xFFFFFFFFu && better(sv[threadIdx.x + s], si[threadIdx.x + s], sv[threadIdx.x], si[threadIdx.x])) {
            sv[threadIdx.x] = sv[threadIdx.x + s];
            si[threadIdx.x] = si[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = si[0];
}

}  // namespace

void set_programmatic_launch(bool on) { pdl_enabled() = on; }

namespace {
DenseArith g_arith = DenseArith::mma3;
}
void set_dense_arith(DenseArith a) { g_arith = a; }
DenseArith dense_arith() { return g_arith; }

cudaError_t dense_rows(const DenseWeight& w, const float* x, float* y, std::uint32_t rows, const float* residual,
                       cudaStream_t stream, DenseArith arithmetic) {
    if (w.format == WeightFormat::nvfp4) return nvfp4_dense_rows(w, x, y, rows, residual, stream);
    return dense_rows_multi(&w, &y, 1, x, rows, residual, stream, arithmetic);
}

cudaError_t dense_rows_multi(const DenseWeight* ws, float* const* ys, int n, const float* x, std::uint32_t rows,
                             const float* residual, cudaStream_t stream, DenseArith arithmetic) {
    if (n < 1 || n > kMaxMulti || (residual != nullptr && n != 1)) return cudaErrorInvalidValue;
    if (ws[0].format == WeightFormat::nvfp4) {  // W4A16 GEMV: one launch for jobs sharing x, bitwise equal to separate calls
        Nvfp4Job jobs[kMaxMulti];
        for (int j = 0; j < n; ++j) {
            if (ws[j].format != WeightFormat::nvfp4 || ws[j].in != ws[0].in) return cudaErrorInvalidValue;
            jobs[j] = Nvfp4Job{ws[j], ys[j], residual};
        }
        return nvfp4_dense_rows_multi(jobs, static_cast<std::uint32_t>(n), x, rows, stream);
    }
    MultiArgs ma{};
    ma.n = n;
    std::uint32_t total = 0;
    for (int j = 0; j < n; ++j) {
        if (!valid(ws[j], rows) || ws[j].format != ws[0].format || ws[j].in != ws[0].in) return cudaErrorInvalidValue;
        ma.w[j] = ws[j].data;
        ma.y[j] = ys[j];
        ma.out[j] = ws[j].out;
        total += ws[j].out;
        ma.row_end[j] = total;
    }
    for (int j = n; j < kMaxMulti; ++j) ma.row_end[j] = total;
    if (arithmetic != DenseArith::fp32) {
        MmaCall mc{};
        mc.n = n; mc.format = ws[0].format; mc.in = ws[0].in; mc.rows = rows; mc.x = x; mc.residual = residual;
        mc.terms = arithmetic == DenseArith::mma3 ? 3 : arithmetic == DenseArith::mma2 ? 2 : 1;
        for (int j = 0; j < n; ++j) { mc.w[j] = ws[j].data; mc.y[j] = ys[j]; mc.out[j] = ws[j].out; }
        if (dense_mma_supported(mc)) return dense_mma_launch(mc, stream);
        // A sibling's fallback must not change another matrix's arithmetic.
        if (n > 1) {
            DenseWeight grouped[2][kMaxMulti];
            float* outputs[2][kMaxMulti];
            int counts[2]{};
            for (int j = 0; j < n; ++j) {
                MmaCall one = mc;
                one.n = 1; one.out[0] = ws[j].out;
                const int route = dense_mma_supported(one) ? 1 : 0;
                outputs[route][counts[route]] = ys[j];
                grouped[route][counts[route]++] = ws[j];
            }
            if (counts[1]) {
                for (int route = 0; route < 2; ++route) {
                    if (!counts[route]) continue;
                    const auto error = dense_rows_multi(grouped[route], outputs[route], counts[route], x, rows,
                                                        nullptr, stream, arithmetic);
                    if (error != cudaSuccess) return error;
                }
                return cudaSuccess;
            }
        }
    }
    return dispatch<false>(ma, ws[0], nullptr, x, residual, rows, stream);
}

cudaError_t dense_gate_up_silu_rows(const DenseWeight& gate, const DenseWeight& up, const float* x, float* y,
                                    std::uint32_t rows, cudaStream_t stream, DenseArith arithmetic) {
    if (gate.format == WeightFormat::nvfp4) return nvfp4_gate_up_silu_rows(gate, up, x, y, rows, stream);
    if (!valid(gate, rows) || !valid(up, rows) || gate.format != up.format || gate.in != up.in || gate.out != up.out)
        return cudaErrorInvalidValue;
    MultiArgs ma{};
    ma.n = 1;
    ma.w[0] = gate.data;
    ma.y[0] = y;
    ma.out[0] = gate.out;
    for (int j = 0; j < kMaxMulti; ++j) ma.row_end[j] = gate.out;
    if (arithmetic != DenseArith::fp32) {
        MmaCall mc{};
        mc.n = 1; mc.format = gate.format; mc.in = gate.in; mc.rows = rows; mc.x = x; mc.w1 = up.data;
        mc.terms = arithmetic == DenseArith::mma3 ? 3 : arithmetic == DenseArith::mma2 ? 2 : 1;
        mc.w[0] = gate.data; mc.y[0] = y; mc.out[0] = gate.out;
        if (dense_mma_supported(mc)) return dense_mma_launch(mc, stream);
    }
    return dispatch<true>(ma, gate, up.data, x, nullptr, rows, stream);
}

cudaError_t embedding_rows(const DenseWeight& table, const std::uint32_t* ids, float* out, std::uint32_t rows,
                           cudaStream_t stream) {
    if (table.data == nullptr || rows == 0) return cudaErrorInvalidValue;
    return launch_pdl(embedding_kernel, dim3(rows), dim3(256), stream, static_cast<const unsigned char*>(table.data), table.format, table.in, ids, out);
}

cudaError_t argmax_rows(const float* logits, std::uint32_t vocab, std::uint32_t rows, std::uint32_t* out, void* workspace,
                        cudaStream_t stream) {
    if (workspace == nullptr || rows == 0 || rows > kMaxRows) return cudaErrorInvalidValue;
    float* pv = static_cast<float*>(workspace);
    std::uint32_t* pi = reinterpret_cast<std::uint32_t*>(pv + static_cast<std::size_t>(kMaxRows) * kArgmaxChunks);
    if (auto e = launch_pdl(argmax_stage1, dim3(kArgmaxChunks, rows), dim3(256), stream, logits, vocab, pv, pi); e != cudaSuccess) return e;
    return launch_pdl(argmax_stage2, dim3(rows), dim3(kArgmaxChunks), stream, static_cast<const float*>(pv), static_cast<const std::uint32_t*>(pi), out);
}

}  // namespace strix::cuda_core
