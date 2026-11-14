// GLM-5.3-Flash attention decode path, optimized schedule (see glm_attn_fast.hpp and
// design notes). Operation orders = src/glm/host/glm_host.cpp; for each kernel the comment names the
// host loop it reproduces and how the lanes map onto the declared reduction (R1 chains, R2 halving trees).
// Build with -ffp-contract=off -fno-fast-math -fno-gpu-flush-denormals-to-zero.
#include "glm/attn/glm_attn_fast.hpp"
#include "attn_core/select.hpp"
#include "glm/attn/glm_dsa_policy.hpp"
#include "glm/common/glm_dsa_pool.hpp"
#include "glm/kernels/glm_tree.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strix::glm::fast {

namespace {

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

// Inverse of ordered_key for the max: the key of a finite float back to a float (-0 comes back as +0; the sign of a
// zero maximum never changes s - m, see design notes section 3.3).
__device__ __forceinline__ float key_to_float(std::uint32_t key) {
    const std::uint32_t u = (key & 0x80000000U) ? (key & 0x7fffffffU) : ~key;
    return __builtin_bit_cast(float, u);
}
typedef unsigned u32x4 __attribute__((ext_vector_type(4)));  // native vectors: arrays of them stay in VGPRs
typedef float f32x4 __attribute__((ext_vector_type(4)));
__device__ __forceinline__ void unpack_f16x8(const uint4 v, float* w) {
    w[0] = f16_bits_to_float(v.x & 0xffffU);
    w[1] = f16_bits_to_float(v.x >> 16U);
    w[2] = f16_bits_to_float(v.y & 0xffffU);
    w[3] = f16_bits_to_float(v.y >> 16U);
    w[4] = f16_bits_to_float(v.z & 0xffffU);
    w[5] = f16_bits_to_float(v.z >> 16U);
    w[6] = f16_bits_to_float(v.w & 0xffffU);
    w[7] = f16_bits_to_float(v.w >> 16U);
}
__device__ __forceinline__ void unpack_bf16x8(const uint4 v, float* w) {
    const unsigned a[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (unsigned e = 0; e < 4; ++e) {
        w[2 * e] = __builtin_bit_cast(float, a[e] << 16U);
        w[2 * e + 1] = __builtin_bit_cast(float, a[e] & 0xffff0000U);
    }
}

// R2 halving tree in LDS over L lanes (same code as the batch-1/2 kernels): s[l] = s[l] + s[l + h], h = L/2 .. 1.
template <unsigned L> __device__ __forceinline__ float lds_tree(float* s, float part) {
    const unsigned l = threadIdx.x;
    s[l] = part;
    __syncthreads();
    for (unsigned h = L / 2; h > 0; h >>= 1U) {
        if (l < h) s[l] = s[l] + s[l + h];
        __syncthreads();
    }
    const float r = s[0];
    __syncthreads();
    return r;
}


// ---------------------------------------------------------------- R1 dense chains (F16)
// Host: dense() = chain_dot over i ascending, acc = fmaf(x[i], W[o][i], acc). One lane per output o; the R rows' chains
// run side by side on the same weight value (each row's chain is still i-ascending). x is staged through LDS.
struct DenseJobs {
    DenseJob j[4];
    unsigned block0[5];
    unsigned njobs;
};
// Field-by-field select of job jj (a whole-struct select with a runtime index goes through scratch).
#define STRIX_GLM_PICK(f) (jj == 0 ? J.j[0].f : jj == 1 ? J.j[1].f : jj == 2 ? J.j[2].f : J.j[3].f)
__device__ __forceinline__ DenseJob pick_job(const DenseJobs& J, unsigned jj) {
    DenseJob d;
    d.w = STRIX_GLM_PICK(w);
    d.x = STRIX_GLM_PICK(x);
    d.y = STRIX_GLM_PICK(y);
    d.n = STRIX_GLM_PICK(n);
    d.in = STRIX_GLM_PICK(in);
    d.x_stride = STRIX_GLM_PICK(x_stride);
    d.y_stride = STRIX_GLM_PICK(y_stride);
    d.groups = STRIX_GLM_PICK(groups);
    d.w_gstride = STRIX_GLM_PICK(w_gstride);
    d.x_gstride = STRIX_GLM_PICK(x_gstride);
    d.y_gstride = STRIX_GLM_PICK(y_gstride);
    return d;
}
#undef STRIX_GLM_PICK
constexpr unsigned kDenseThreads = 64, kDenseChunk = 512;
template <unsigned R>
__global__ __launch_bounds__(kDenseThreads) void dense_f16_kernel(DenseJobs J) {
    __shared__ float xs[R][kDenseChunk];
    const unsigned b = blockIdx.x;
    unsigned jj = 0;
#pragma unroll
    for (unsigned q = 1; q < 4; ++q)
        if (q < J.njobs && b >= J.block0[q]) jj = q;
    const DenseJob job = pick_job(J, jj);
    const unsigned o = (b - J.block0[jj]) * kDenseThreads + threadIdx.x;
    const bool active = o < job.n;
    const uint4* wr = reinterpret_cast<const uint4*>(job.w + static_cast<std::size_t>(active ? o : 0) * job.in);
    float acc[R];
#pragma unroll
    for (unsigned r = 0; r < R; ++r) acc[r] = 0.0F;
    for (unsigned c0 = 0; c0 < job.in; c0 += kDenseChunk) {
        const unsigned cn = job.in - c0 < kDenseChunk ? job.in - c0 : kDenseChunk;
        __syncthreads();
        for (unsigned e = threadIdx.x; e < R * cn; e += kDenseThreads) {
            const unsigned r = e / cn, i = e - r * cn;
            xs[r][i] = job.x[static_cast<std::size_t>(r) * job.x_stride + c0 + i];
        }
        __syncthreads();
        if (active) {
            for (unsigned i = 0; i < cn; i += 32) {  // 4 x 16 B of the row in flight, then the 32 chain steps
                uint4 v[4];
#pragma unroll
                for (unsigned u = 0; u < 4; ++u) v[u] = wr[(c0 + i) / 8 + u];
#pragma unroll
                for (unsigned u = 0; u < 4; ++u) {
                    float w[8];
                    unpack_f16x8(v[u], w);
#pragma unroll
                    for (unsigned e = 0; e < 8; ++e)
#pragma unroll
                        for (unsigned r = 0; r < R; ++r) acc[r] = __builtin_fmaf(xs[r][i + u * 8 + e], w[e], acc[r]);
                }
            }
        }
    }
    if (active)
#pragma unroll
        for (unsigned r = 0; r < R; ++r) job.y[static_cast<std::size_t>(r) * job.y_stride + o] = acc[r];
}

// Staged schedule (in % 256 == 0): block = 256 threads and a tile of OB (16 or 8) consecutive outputs of one group.
// Waves 4..7 only load: per 256-column chunk they read the tile's OB weight rows (OB / 4 x 16 B per thread), decode
// F16 -> fp32 once (f16_bits_to_float) and stage the weights and the R rows' x chunk in double-buffered LDS (weight rows
// padded to 260 floats: conflict-free b128 reads). Waves 0..3 only compute: lane c < OB R runs the chain of output
// (c % OB) for row (c / OB) from LDS (float4 w, float4 x), the same i-ascending R1 fmaf chain as host dense().
// Barriers: both roles execute 2 nch of them (loader: publish chunk 0; then per chunk c >= 1 "computers finished
// chunk c - 1" + publish c; then the final one. Computers: B2 for chunk 0; per chunk c >= 1 B1 + B2; then the final one).
constexpr unsigned kStCols = 256, kStPadF = 4, kStThreads = 256, kStLoaders = 128;
template <unsigned R, unsigned OB>
__global__ __launch_bounds__(kStThreads) void dense_f16_staged_kernel(DenseJobs J) {
    constexpr unsigned kStOut = OB, kPer = OB / 4;  // weight rows per loader thread
    __shared__ __attribute__((aligned(16))) float wt[2][kStOut][kStCols + kStPadF];
    __shared__ __attribute__((aligned(16))) float xs[2][R][kStCols];
    const unsigned b = blockIdx.x, tau = threadIdx.x;
    unsigned jj = 0;
#pragma unroll
    for (unsigned q = 1; q < 4; ++q)
        if (q < J.njobs && b >= J.block0[q]) jj = q;
    const DenseJob job = pick_job(J, jj);
    const unsigned tiles = (job.n + kStOut - 1) / kStOut, local = b - J.block0[jj];
    const unsigned g = local / tiles, o0 = (local - g * tiles) * kStOut;
    const std::uint16_t* wg = job.w + g * job.w_gstride;
    const float* xg = job.x + g * job.x_gstride;
    const unsigned nch = job.in / kStCols;
    const bool loader = tau >= kStThreads - kStLoaders;
    if (loader) {
        // loader lane lt: weight rows lt / 32 + 4 k (k < OB / 4), columns (lt % 32) * 8 .. + 7; rows past n load a valid
        // row, never used. x: float4 e = lt, lt + 128, ... of the R x 64 float4 chunk.
        const unsigned lt = tau - (kStThreads - kStLoaders), lrow = lt / 32, lcol = (lt % 32) * 8;
        const std::uint16_t* rowp[kPer];
#pragma unroll
        for (unsigned k = 0; k < kPer; ++k) {
            const unsigned o = o0 + lrow + 4 * k < job.n ? o0 + lrow + 4 * k : job.n - 1;
            rowp[k] = wg + static_cast<std::size_t>(o) * job.in + lcol;
        }
        constexpr unsigned kXPer = (R * kStCols / 4 + kStLoaders - 1) / kStLoaders;
        for (unsigned c = 0; c < nch; ++c) {
            u32x4 v[kPer];
#pragma unroll
            for (unsigned k = 0; k < kPer; ++k) v[k] = *reinterpret_cast<const u32x4*>(rowp[k] + c * kStCols);
            f32x4 xv[kXPer];
#pragma unroll
            for (unsigned k = 0; k < kXPer; ++k) {
                const unsigned e = lt + kStLoaders * k, row = e / (kStCols / 4), c4 = e % (kStCols / 4);
                if (row < R) xv[k] = *reinterpret_cast<const f32x4*>(xg + static_cast<std::size_t>(row) * job.x_stride + c * kStCols + c4 * 4);
            }
            if (c > 0) __syncthreads();  // computers finished chunk c - 1 (buffer c & 1, last read by chunk c - 2, is free)
#pragma unroll
            for (unsigned k = 0; k < kPer; ++k) {
                float w[8];
                w[0] = f16_bits_to_float(v[k].x & 0xffffU);
                w[1] = f16_bits_to_float(v[k].x >> 16U);
                w[2] = f16_bits_to_float(v[k].y & 0xffffU);
                w[3] = f16_bits_to_float(v[k].y >> 16U);
                w[4] = f16_bits_to_float(v[k].z & 0xffffU);
                w[5] = f16_bits_to_float(v[k].z >> 16U);
                w[6] = f16_bits_to_float(v[k].w & 0xffffU);
                w[7] = f16_bits_to_float(v[k].w >> 16U);
                f32x4* dst = reinterpret_cast<f32x4*>(&wt[c & 1U][lrow + 4 * k][lcol]);
                dst[0] = f32x4{w[0], w[1], w[2], w[3]};
                dst[1] = f32x4{w[4], w[5], w[6], w[7]};
            }
#pragma unroll
            for (unsigned k = 0; k < kXPer; ++k) {
                const unsigned e = lt + kStLoaders * k, row = e / (kStCols / 4), c4 = e % (kStCols / 4);
                if (row < R) *reinterpret_cast<f32x4*>(&xs[c & 1U][row][c4 * 4]) = xv[k];
            }
            __syncthreads();             // chunk c published
        }
        __syncthreads();                 // pairs with the compute waves' last barrier
        return;
    }
    const bool chain = tau < kStOut * R;
    const unsigned oo = tau % kStOut, r = chain ? tau / kStOut : 0;
    float acc = 0.0F;
    for (unsigned c = 0; c < nch; ++c) {
        if (c > 0) __syncthreads();  // B1: this wave finished chunk c - 1
        __syncthreads();             // B2: chunk c published
        if (chain) {
            const f32x4* w4 = reinterpret_cast<const f32x4*>(&wt[c & 1U][oo][0]);
            const f32x4* x4 = reinterpret_cast<const f32x4*>(&xs[c & 1U][r][0]);
#pragma unroll 8
            for (unsigned i4 = 0; i4 < kStCols / 4; ++i4) {
                const f32x4 w = w4[i4], x = x4[i4];
                acc = __builtin_fmaf(x.x, w.x, acc);
                acc = __builtin_fmaf(x.y, w.y, acc);
                acc = __builtin_fmaf(x.z, w.z, acc);
                acc = __builtin_fmaf(x.w, w.w, acc);
            }
        }
    }
    __syncthreads();
    if (chain && o0 + oo < job.n) job.y[g * job.y_gstride + static_cast<std::size_t>(r) * job.y_stride + o0 + oo] = acc;
}

// ---------------------------------------------------------------- KDA (R8, R9, R10)
// R2 tree over L participating lanes indexed by idx (the other threads only pass the barriers).
template <unsigned L> __device__ __forceinline__ float lds_tree_p(float* s, unsigned idx, bool on, float part) {
    if (on) s[idx] = part;
    __syncthreads();
    for (unsigned h = L / 2; h > 0; h >>= 1U) {
        if (on && idx < h) s[idx] = s[idx] + s[idx + h];
        __syncthreads();
    }
    const float r = s[0];
    __syncthreads();
    return r;
}
// One block per (head, slot), 128 Q threads (Q = lanes per state column: 2, 4 or 8). Prep (threads 0..127, thread =
// channel i of the head's q, k and v): R8 conv + state shift, SiLU, the l2norm trees, g, beta, exactly as
// kda_prep_kernel. Update: column v of S is split over Q lanes of one wave (lanes c + j * 32 / Q, j = qid < Q): lane qid
// holds k = a + 32m for a in [qid A, (qid + 1) A), A = 32 / Q, m = 0..3 (4 A registers). The declared 128-lane tree for
// u_v / o_v (lane k = 0 + term_k; pairs (k, k + h), h = 64 .. 1): levels 64 and 32 pair registers of one lane, levels
// 16 .. A pair lanes qid and qid + h / A (one shuffle each; the lower k is always the left operand), levels A / 2 .. 1
// in-lane. Q = 2 is the original schedule instruction for instruction. Out: R10 as kda_out_kernel over the 128
// columns. kSeq: npos positions of one slot (state stays in registers).
// kIl (interleaved ownership, review): lane qid holds k = qid + Q j, j = 0 .. 128/Q - 1 (Sc[j]). Every level h >= Q of the
// same declared tree pairs two k of one lane (k = qid + Q j and k + h = qid + Q (j + h/Q); k < h <=> j < h/Q): in-lane
// t[j] = t[j] + t[j + h/Q]; levels h = Q/2 .. 1 pair lanes qid and qid + h (one shuffle each, lower k left), then the
// broadcast: log2(Q) + 1 shuffles per tree instead of A log2(Q) + 1 (17/17/13 -> 2/3/4 for Q = 2/4/8). kIl = false is
// the previous code unchanged (every kIl branch is `if constexpr`).
using detail::tree128_w0;
template <bool kTr> __device__ __forceinline__ float tree128_sel(float* s, unsigned idx, bool on, float part) {
    if constexpr (kTr) return tree128_w0(s, s + 128, idx, on, part);
    else return lds_tree_p<128>(s, idx, on, part);
}
// Shared R8 preparation: one 128-thread CTA per head/slot; thread i owns channel i.
// This is the sole conv_state / conv_snap owner for shared-prep split modes.
template <bool kSeq, bool kTr>
__global__ __launch_bounds__(128) void kda_prep_kernel(dev::KdaDev k, const float* qkv, const float* f,
                                                     const float* b, float* conv_state, float* conv_snap,
                                                     float* W, unsigned npos) {
    __shared__ float s[kTr ? 129 : 128];
    const unsigned i = threadIdx.x, h = blockIdx.x, H = k.heads, C = H * kKdaDim;
    const bool prep = true;
    const unsigned slot = kSeq ? 0U : blockIdx.y;
    float* cs = conv_state + static_cast<std::size_t>(slot) * 3 * C * 3;
    float st[3][3];
#pragma unroll
    for (unsigned part = 0; part < 3; ++part) {
        const unsigned ch = part * C + h * kKdaDim + i;
#pragma unroll
        for (unsigned j = 0; j < 3; ++j) st[part][j] = prep ? cs[static_cast<std::size_t>(ch) * 3 + j] : 0.0F;
    }
    const float expA = declared_exp(k.A_log[h]);
    const float dtb = k.dt_bias[h * kKdaDim + i];
    // W[row][head] = {qs[128], ks[128], ds[128], vs[128], beta}.
    for (unsigned p = 0; p < npos; ++p) {
        const unsigned row = kSeq ? p : blockIdx.y;
        float* work = W + (static_cast<std::size_t>(row) * H + h) * (4 * kKdaDim + 1);
        float* qs = work;
        float* ks = work + kKdaDim;
        float* ds = work + 2 * kKdaDim;
        float* vs = work + 3 * kKdaDim;
        const float* in = qkv + static_cast<std::size_t>(row) * 3 * C;
        float out[3] = {0.0F, 0.0F, 0.0F};
        if (prep) {
#pragma unroll
            for (unsigned part = 0; part < 3; ++part) {  // R8 (oldest first), then SiLU; shift
                const unsigned ch = part * C + h * kKdaDim + i;
                const std::uint16_t* cw = k.conv_w + static_cast<std::size_t>(ch) * kConvK;  // re-read: cache hit, frees 12 VGPRs
                const float nw = in[ch];
                float acc = __builtin_fmaf(bf16_to_float(cw[0]), st[part][0], 0.0F);
                acc = __builtin_fmaf(bf16_to_float(cw[1]), st[part][1], acc);
                acc = __builtin_fmaf(bf16_to_float(cw[2]), st[part][2], acc);
                acc = __builtin_fmaf(bf16_to_float(cw[3]), nw, acc);
                out[part] = silu(acc);
                st[part][0] = st[part][1];
                st[part][1] = st[part][2];
                st[part][2] = nw;
            }
        }
        const float sq = tree128_sel<kTr>(s, i, prep, __builtin_fmaf(out[0], out[0], 0.0F));
        const float sk = tree128_sel<kTr>(s, i, prep, __builtin_fmaf(out[1], out[1], 0.0F));
        if (prep) {
            const float nq = declared_sqrt(sq + kL2Eps), nk = declared_sqrt(sk + kL2Eps);
            qs[i] = (out[0] / nq) * kInvSqrt128;
            ks[i] = out[1] / nk;
            ds[i] = declared_exp(kGateLow * sigmoid(expA * (f[static_cast<std::size_t>(row) * C + h * kKdaDim + i] + dtb)));
            vs[i] = out[2];
        }
        const float bt = sigmoid(b[static_cast<std::size_t>(row) * H + h]);
        if (i == 0) work[4 * kKdaDim] = bt;
        if (kSeq && conv_snap) {
            float* cp = conv_snap + static_cast<std::size_t>(p) * 3 * C * 3;
#pragma unroll
            for (unsigned part = 0; part < 3; ++part)
#pragma unroll
                for (unsigned j = 0; j < 3; ++j) cp[static_cast<std::size_t>(part * C + h * kKdaDim + i) * 3 + j] = st[part][j];
        }
    }
    if (prep)
#pragma unroll
        for (unsigned part = 0; part < 3; ++part)
#pragma unroll
            for (unsigned j = 0; j < 3; ++j) cs[static_cast<std::size_t>(part * C + h * kKdaDim + i) * 3 + j] = st[part][j];
}

// kCq: 256 threads (128 when split), wave w owns columns base + 16w..+15. Lane cq + 4j owns
// four columns at v = base + 16w + 4cq and rows k = j + 8 ii; the interleaved Q8 tree uses shuffle strides 4h.
// Split column-quad CTAs own 64 columns each, base = (blockIdx.x % 2) * 64.
// Split CTAs own disjoint columns; R10 is deferred to the finalizer.
// kSp consumes shared prep from W; otherwise conv_state writes are also deferred to the finalizer.
template <bool kSeq, unsigned Q, bool kIl, bool kTr, bool kSplit, bool kCq = false, bool kSp = false>
__global__ __launch_bounds__(kSplit ? 128 : (kCq ? 256 : 128 * Q)) void kda_fused_kernel(dev::KdaDev k, const float* qkv, const float* f, const float* gate,
                                                            const float* b, float* S, float* conv_state, float* y,
                                                            float* snap, float* conv_snap, unsigned* flag, unsigned npos, const float* W = nullptr) {
    static_assert(!kSp || kSplit, "shared prep requires split producers");
    static_assert(Q == 2 || Q == 4 || Q == 8, "lanes per column");
    static_assert(!kCq || (!kIl && Q == 2 && (!kSplit || kSp)), "column quad requires Q2, non-interleaved, and shared prep when split");
    constexpr unsigned A = 32 / Q, CW = 32 / Q;  // k-values (a) per lane and per 32-block; columns per wave
    __shared__ float s[kTr ? 129 : 128];  // kTr: s[128] = the tree result slot
    __shared__ float qs[kKdaDim], ks[kKdaDim], ds[kKdaDim], vs[kKdaDim];
    const unsigned tau = threadIdx.x, H = k.heads, C = H * kKdaDim;
    unsigned h = blockIdx.x;
    const unsigned lane = tau & 31U, qid = kCq ? (lane >> 2U) : lane / CW;
    const unsigned cq = lane & 3U;
    unsigned v = kCq ? (tau >> 5U) * 16U + 4U * cq : (tau >> 5U) * CW + (lane % CW);
    if constexpr (kSplit) {
        constexpr unsigned NB = kCq ? 2U : Q;
        h = blockIdx.x / NB;
        v += (blockIdx.x % NB) * (kKdaDim / NB);
    }
    const bool prep = tau < kKdaDim;
    const unsigned i = prep ? tau : 0U;  // prep channel
    const unsigned slot = kSeq ? 0U : blockIdx.y;
    float* Sh = S + (static_cast<std::size_t>(slot) * H + h) * kKdaDim * kKdaDim;
    float* cs = kSp ? nullptr : conv_state + static_cast<std::size_t>(slot) * 3 * C * 3;
    float st[3][3];
    if constexpr (!kSp) {
#pragma unroll
        for (unsigned part = 0; part < 3; ++part) {
            const unsigned ch = part * C + h * kKdaDim + i;
#pragma unroll
            for (unsigned j = 0; j < 3; ++j) st[part][j] = prep ? cs[static_cast<std::size_t>(ch) * 3 + j] : 0.0F;
        }
    }
    // Sc[m * A + aa] = S[k = qid * A + aa + 32 m][v]
    float Sc[kCq ? 1 : 4 * A];
    float4 Sc4[kCq ? 16 : 1];  // kCq requires 16-byte aligned S and optional snapshot bases
    if constexpr (kCq) {
#pragma unroll
        for (unsigned ii = 0; ii < 16; ++ii)
            Sc4[ii] = *reinterpret_cast<const float4*>(Sh + (qid + 8U * ii) * kKdaDim + v);
    } else if constexpr (kIl) {
#pragma unroll
        for (unsigned j = 0; j < 4 * A; ++j) Sc[j] = Sh[(qid + Q * j) * kKdaDim + v];
    } else {
#pragma unroll
        for (unsigned m = 0; m < 4; ++m)
#pragma unroll
            for (unsigned aa = 0; aa < A; ++aa) Sc[m * A + aa] = Sh[(qid * A + aa + 32 * m) * kKdaDim + v];
    }
    const float expA = kSp ? 0.0F : declared_exp(k.A_log[h]);
    const float dtb = kSp ? 0.0F : k.dt_bias[h * kKdaDim + i], onw = bf16_to_float(k.o_norm[v]);
    for (unsigned p = 0; p < npos; ++p) {
        const unsigned row = kSeq ? p : blockIdx.y;
        float bt;
        if constexpr (kSp) {
            const float* work = W + (static_cast<std::size_t>(row) * H + h) * (4 * kKdaDim + 1);
            qs[i] = work[i];
            ks[i] = work[kKdaDim + i];
            ds[i] = work[2 * kKdaDim + i];
            vs[i] = work[3 * kKdaDim + i];
            bt = work[4 * kKdaDim];
        } else {
            const float* in = qkv + static_cast<std::size_t>(row) * 3 * C;
            float out[3] = {0.0F, 0.0F, 0.0F};
            if (prep) {
#pragma unroll
                for (unsigned part = 0; part < 3; ++part) {  // R8 (oldest first), then SiLU; shift
                    const unsigned ch = part * C + h * kKdaDim + i;
                    const std::uint16_t* cw = k.conv_w + static_cast<std::size_t>(ch) * kConvK;  // re-read: cache hit, frees 12 VGPRs
                    const float nw = in[ch];
                    float acc = __builtin_fmaf(bf16_to_float(cw[0]), st[part][0], 0.0F);
                    acc = __builtin_fmaf(bf16_to_float(cw[1]), st[part][1], acc);
                    acc = __builtin_fmaf(bf16_to_float(cw[2]), st[part][2], acc);
                    acc = __builtin_fmaf(bf16_to_float(cw[3]), nw, acc);
                    out[part] = silu(acc);
                    st[part][0] = st[part][1];
                    st[part][1] = st[part][2];
                    st[part][2] = nw;
                }
            }
            const float sq = tree128_sel<kTr>(s, i, prep, __builtin_fmaf(out[0], out[0], 0.0F));
            const float sk = tree128_sel<kTr>(s, i, prep, __builtin_fmaf(out[1], out[1], 0.0F));
            if (prep) {
                const float nq = declared_sqrt(sq + kL2Eps), nk = declared_sqrt(sk + kL2Eps);
                qs[i] = (out[0] / nq) * kInvSqrt128;
                ks[i] = out[1] / nk;
                ds[i] = declared_exp(kGateLow * sigmoid(expA * (f[static_cast<std::size_t>(row) * C + h * kKdaDim + i] + dtb)));
                vs[i] = out[2];
            }
            bt = sigmoid(b[static_cast<std::size_t>(row) * H + h]);
        }
        __syncthreads();
        if constexpr (kCq) {
            // Each lane owns four adjacent columns, k = qid + 8 ii. The lower k is always left.
            auto tree4 = [&](const float* vec) {
                float4 t[16];
#pragma unroll
                for (unsigned ii = 0; ii < 16; ++ii) {
                    const float x = vec[qid + 8U * ii];
                    t[ii] = float4{0.0F + Sc4[ii].x * x, 0.0F + Sc4[ii].y * x,
                                   0.0F + Sc4[ii].z * x, 0.0F + Sc4[ii].w * x};
                }
#pragma unroll
                for (unsigned hh = 64; hh >= 8; hh >>= 1U)
#pragma unroll
                    for (unsigned ii = 0; ii < hh / 8; ++ii) {
                        t[ii].x = t[ii].x + t[ii + hh / 8].x;
                        t[ii].y = t[ii].y + t[ii + hh / 8].y;
                        t[ii].z = t[ii].z + t[ii + hh / 8].z;
                        t[ii].w = t[ii].w + t[ii + hh / 8].w;
                    }
                float4 r = t[0];
#pragma unroll
                for (unsigned hh = 4; hh > 0; hh >>= 1U) {
                    r.x = r.x + __shfl_xor(r.x, static_cast<int>(4U * hh), 32);
                    r.y = r.y + __shfl_xor(r.y, static_cast<int>(4U * hh), 32);
                    r.z = r.z + __shfl_xor(r.z, static_cast<int>(4U * hh), 32);
                    r.w = r.w + __shfl_xor(r.w, static_cast<int>(4U * hh), 32);
                }
                return float4{__shfl(r.x, static_cast<int>(cq), 32), __shfl(r.y, static_cast<int>(cq), 32),
                              __shfl(r.z, static_cast<int>(cq), 32), __shfl(r.w, static_cast<int>(cq), 32)};
            };
#pragma unroll
            for (unsigned ii = 0; ii < 16; ++ii) {
                const float d = ds[qid + 8U * ii];
                Sc4[ii].x = Sc4[ii].x * d;
                Sc4[ii].y = Sc4[ii].y * d;
                Sc4[ii].z = Sc4[ii].z * d;
                Sc4[ii].w = Sc4[ii].w * d;
            }
            const float4 u = tree4(ks);
            const float4 dlt{(vs[v] - u.x) * bt, (vs[v + 1] - u.y) * bt,
                             (vs[v + 2] - u.z) * bt, (vs[v + 3] - u.w) * bt};
            bool fin = true;
#pragma unroll
            for (unsigned ii = 0; ii < 16; ++ii) {
                const float kval = ks[qid + 8U * ii];
                Sc4[ii].x = __builtin_fmaf(kval, dlt.x, Sc4[ii].x);
                Sc4[ii].y = __builtin_fmaf(kval, dlt.y, Sc4[ii].y);
                Sc4[ii].z = __builtin_fmaf(kval, dlt.z, Sc4[ii].z);
                Sc4[ii].w = __builtin_fmaf(kval, dlt.w, Sc4[ii].w);
                fin = fin && is_finite(Sc4[ii].x) && is_finite(Sc4[ii].y) && is_finite(Sc4[ii].z) && is_finite(Sc4[ii].w);
            }
            if (!fin && flag) atomicOr(flag, 1U);
            const float4 ov = tree4(qs);
            if constexpr (kSplit) {
                if (qid == 0) {
                    const std::size_t at = static_cast<std::size_t>(row) * C + h * kKdaDim + v;
                    y[at] = ov.x;
                    y[at + 1] = ov.y;
                    y[at + 2] = ov.z;
                    y[at + 3] = ov.w;
                }
            } else {
                if (qid == 0) {
                    s[v] = __builtin_fmaf(ov.x, ov.x, 0.0F);
                    s[v + 1] = __builtin_fmaf(ov.y, ov.y, 0.0F);
                    s[v + 2] = __builtin_fmaf(ov.z, ov.z, 0.0F);
                    s[v + 3] = __builtin_fmaf(ov.w, ov.w, 0.0F);
                }
                __syncthreads();  // publish all four leaves, as in tree128_sel
                float total;
                if constexpr (kTr) {
                    if (tau < 32) {
                        float x = (s[tau] + s[tau + 64]) + (s[tau + 32] + s[tau + 96]);
#pragma unroll
                        for (unsigned hh = 16; hh > 0; hh >>= 1U) x = x + __shfl_down(x, hh, 32);
                        if (tau == 0) s[128] = x;
                    }
                    __syncthreads();
                    total = s[128];
                } else {
                    for (unsigned hh = 64; hh > 0; hh >>= 1U) {
                        if (tau < hh) s[tau] = s[tau] + s[tau + hh];
                        __syncthreads();
                    }
                    total = s[0];
                    __syncthreads();
                }
                const float ms = total / static_cast<float>(kKdaDim);
                const float rr = inv_sqrt(ms + kEps);
                if (qid == 0) {
                    const std::size_t at = static_cast<std::size_t>(row) * C + h * kKdaDim + v;
                    y[at] = ((ov.x * rr) * bf16_to_float(k.o_norm[v])) * sigmoid(gate[at]);
                    y[at + 1] = ((ov.y * rr) * bf16_to_float(k.o_norm[v + 1])) * sigmoid(gate[at + 1]);
                    y[at + 2] = ((ov.z * rr) * bf16_to_float(k.o_norm[v + 2])) * sigmoid(gate[at + 2]);
                    y[at + 3] = ((ov.w * rr) * bf16_to_float(k.o_norm[v + 3])) * sigmoid(gate[at + 3]);
                }
            }
        } else {
            // the 128-lane tree of 0 + S_kv * vec_k for this column's Q lanes; every lane of the column returns qid 0's sum
            auto tree = [&](const float* vec) {
                if constexpr (kIl) {
                    float t[4 * A];
#pragma unroll
                    for (unsigned j = 0; j < 4 * A; ++j) t[j] = 0.0F + Sc[j] * vec[qid + Q * j];  // leaf k = qid + Q j
#pragma unroll
                    for (unsigned hh = 64; hh >= Q; hh >>= 1U)  // s[k] + s[k + hh], both in this lane
#pragma unroll
                        for (unsigned j = 0; j < hh / Q; ++j) t[j] = t[j] + t[j + hh / Q];
                    float r = t[0];  // s[qid] after level Q
#pragma unroll
                    for (unsigned hh = Q / 2; hh > 0; hh >>= 1U) {  // s[qid] + s[qid + hh] (valid where qid < hh)
                        const float other = __shfl_xor(r, static_cast<int>(hh * CW), 32);
                        r = r + other;
                    }
                    return __shfl(r, static_cast<int>(lane % CW), 32);
                }
                float t[A];
#pragma unroll
                for (unsigned aa = 0; aa < A; ++aa) {
                    const unsigned a = qid * A + aa;
                    const float l64a = (0.0F + Sc[aa] * vec[a]) + (0.0F + Sc[2 * A + aa] * vec[a + 64]);      // level 64: k = a
                    const float l64b = (0.0F + Sc[A + aa] * vec[a + 32]) + (0.0F + Sc[3 * A + aa] * vec[a + 96]);  // level 64: k = a + 32
                    t[aa] = l64a + l64b;                                                                       // level 32: k = a
                }
#pragma unroll
                for (unsigned hh = 16; hh >= A; hh >>= 1U)  // s[a] + s[a + hh] (lane qid + hh / A; valid where qid < hh / A)
#pragma unroll
                    for (unsigned aa = 0; aa < A; ++aa) {
                        const float other = __shfl_xor(t[aa], static_cast<int>((hh / A) * CW), 32);
                        t[aa] = t[aa] + other;
                    }
#pragma unroll
                for (unsigned hh = A / 2; hh > 0; hh >>= 1U)
#pragma unroll
                    for (unsigned aa = 0; aa < hh; ++aa) t[aa] = t[aa] + t[aa + hh];
                return __shfl(t[0], static_cast<int>(lane % CW), 32);
            };
            // R9 for column v
            if constexpr (kIl) {
#pragma unroll
                for (unsigned j = 0; j < 4 * A; ++j) Sc[j] = Sc[j] * ds[qid + Q * j];
            } else {
#pragma unroll
                for (unsigned m = 0; m < 4; ++m)
#pragma unroll
                    for (unsigned aa = 0; aa < A; ++aa) Sc[m * A + aa] = Sc[m * A + aa] * ds[qid * A + aa + 32 * m];
            }
            const float u = tree(ks);
            const float dlt = (vs[v] - u) * bt;
            bool fin = true;
            if constexpr (kIl) {
#pragma unroll
                for (unsigned j = 0; j < 4 * A; ++j) {
                    Sc[j] = __builtin_fmaf(ks[qid + Q * j], dlt, Sc[j]);
                    fin = fin && is_finite(Sc[j]);
                }
            } else {
#pragma unroll
                for (unsigned m = 0; m < 4; ++m)
#pragma unroll
                    for (unsigned aa = 0; aa < A; ++aa) {
                        Sc[m * A + aa] = __builtin_fmaf(ks[qid * A + aa + 32 * m], dlt, Sc[m * A + aa]);
                        fin = fin && is_finite(Sc[m * A + aa]);
                    }
            }
            if (!fin && flag) atomicOr(flag, 1U);
            const float ov = tree(qs);
            if constexpr (kSplit) {
                if (qid == 0) {
                    const std::size_t at = static_cast<std::size_t>(row) * C + h * kKdaDim + v;
                    y[at] = ov;
                }
            } else {
                // R10 over the 128 columns (the qid-0 lanes participate, index v)
                const float ms = tree128_sel<kTr>(s, v, qid == 0, __builtin_fmaf(ov, ov, 0.0F)) / static_cast<float>(kKdaDim);
                const float rr = inv_sqrt(ms + kEps);
                if (qid == 0) {
                    const std::size_t at = static_cast<std::size_t>(row) * C + h * kKdaDim + v;
                    y[at] = ((ov * rr) * onw) * sigmoid(gate[at]);
                }
            }
        }
        if (kSeq && snap) {
            float* sp = snap + (static_cast<std::size_t>(p) * H + h) * kKdaDim * kKdaDim;
            if constexpr (kCq) {
#pragma unroll
                for (unsigned ii = 0; ii < 16; ++ii)
                    *reinterpret_cast<float4*>(sp + (qid + 8U * ii) * kKdaDim + v) = Sc4[ii];
            } else if constexpr (kIl) {
#pragma unroll
                for (unsigned j = 0; j < 4 * A; ++j) sp[(qid + Q * j) * kKdaDim + v] = Sc[j];
            } else {
#pragma unroll
                for (unsigned m = 0; m < 4; ++m)
#pragma unroll
                    for (unsigned aa = 0; aa < A; ++aa) sp[(qid * A + aa + 32 * m) * kKdaDim + v] = Sc[m * A + aa];
            }
        }
        if constexpr (!kSp) {
            bool conv_writer = prep;
            if constexpr (kSplit) conv_writer = prep && blockIdx.x % Q == 0;
            if (kSeq && conv_snap && conv_writer) {
                float* cp = conv_snap + static_cast<std::size_t>(p) * 3 * C * 3;
#pragma unroll
                for (unsigned part = 0; part < 3; ++part)
#pragma unroll
                    for (unsigned j = 0; j < 3; ++j) cp[static_cast<std::size_t>(part * C + h * kKdaDim + i) * 3 + j] = st[part][j];
            }
        }
        __syncthreads();  // qs / ks / ds / vs are rewritten by the next position
    }
    if constexpr (kCq) {
#pragma unroll
        for (unsigned ii = 0; ii < 16; ++ii)
            *reinterpret_cast<float4*>(Sh + (qid + 8U * ii) * kKdaDim + v) = Sc4[ii];
    } else if constexpr (kIl) {
#pragma unroll
        for (unsigned j = 0; j < 4 * A; ++j) Sh[(qid + Q * j) * kKdaDim + v] = Sc[j];
    } else {
#pragma unroll
        for (unsigned m = 0; m < 4; ++m)
#pragma unroll
            for (unsigned aa = 0; aa < A; ++aa) Sh[(qid * A + aa + 32 * m) * kKdaDim + v] = Sc[m * A + aa];
    }
    if constexpr (!kSplit && !kSp) {
        if (prep)
#pragma unroll
            for (unsigned part = 0; part < 3; ++part)
#pragma unroll
                for (unsigned j = 0; j < 3; ++j) cs[static_cast<std::size_t>(part * C + h * kKdaDim + i) * 3 + j] = st[part][j];
    }
}

// R10 consumes split outputs; shared-prep mode leaves all convolution work to prep.
template <bool kSeq, bool kSp = false>
__global__ __launch_bounds__(128) void kda_r10_finalize_kernel(dev::KdaDev k, const float* qkv, const float* gate,
                                                             float* conv_state, float* y, unsigned npos, bool kTr) {
    __shared__ float s[129];
    const unsigned h = blockIdx.x, row = blockIdx.y, v = threadIdx.x, C = k.heads * kKdaDim;
    const std::size_t at = static_cast<std::size_t>(row) * C + h * kKdaDim + v;
    const float ov = y[at];
    const float leaf = __builtin_fmaf(ov, ov, 0.0F);
    const float ms = (kTr ? tree128_sel<true>(s, v, true, leaf) : tree128_sel<false>(s, v, true, leaf)) /
                     static_cast<float>(kKdaDim);
    const float rr = inv_sqrt(ms + kEps);
    const float onw = bf16_to_float(k.o_norm[v]);
    y[at] = ((ov * rr) * onw) * sigmoid(gate[at]);
    if constexpr (!kSp) {
        bool conv_writer = true;
        if constexpr (kSeq) conv_writer = row == 0;
        if (conv_writer) {
            const unsigned slot = kSeq ? 0U : row;
            float* cs = conv_state + static_cast<std::size_t>(slot) * 3 * C * 3;
#pragma unroll
            for (unsigned part = 0; part < 3; ++part) {
                const unsigned ch = part * C + h * kKdaDim + v;
                float st[3];
#pragma unroll
                for (unsigned j = 0; j < 3; ++j) st[j] = cs[static_cast<std::size_t>(ch) * 3 + j];
                for (unsigned p = 0; p < npos; ++p) {
                    const unsigned pos = kSeq ? p : row;
                    const float nw = qkv[static_cast<std::size_t>(pos) * 3 * C + ch];
                    st[0] = st[1];
                    st[1] = st[2];
                    st[2] = nw;
                }
#pragma unroll
                for (unsigned j = 0; j < 3; ++j) cs[static_cast<std::size_t>(ch) * 3 + j] = st[j];
            }
        }
    }
}

// Copy-through with the default Q2 state ownership and register footprint.
__global__ __launch_bounds__(256) void kda_state_copy_floor_kernel(float* S, unsigned heads) {
    const unsigned tau = threadIdx.x, lane = tau & 31U, qid = lane / 16;
    const unsigned v = (tau >> 5U) * 16 + lane % 16;
    float* Sh = S + (static_cast<std::size_t>(blockIdx.y) * heads + blockIdx.x) * kKdaDim * kKdaDim;
    float Sc[64];
#pragma unroll
    for (unsigned m = 0; m < 4; ++m)
#pragma unroll
        for (unsigned aa = 0; aa < 16; ++aa) Sc[m * 16 + aa] = Sh[(qid * 16 + aa + 32 * m) * kKdaDim + v];
#pragma unroll
    for (unsigned j = 0; j < 64; ++j) __asm__ __volatile__("" : "+v"(Sc[j]) : : "memory");
#pragma unroll
    for (unsigned m = 0; m < 4; ++m)
#pragma unroll
        for (unsigned aa = 0; aa < 16; ++aa) Sh[(qid * 16 + aa + 32 * m) * kKdaDim + v] = Sc[m * 16 + aa];
}

// ---------------------------------------------------------------- DSA cache writes (R4, C3, R12, C4)
// Same arithmetic as dsa_store_kernel (glm_dsa.hip) with the slot / position per row; also clears hist0[row].
__global__ __launch_bounds__(128) void dsa_store_kernel(DsaRows d, const float* latent, dev::IndexerDev w, const float* key,
                                                        const float* gs, std::uint32_t* hist0, unsigned* flag,
                                                        unsigned flag_stride) {
    __shared__ float s[128];
    const unsigned r = blockIdx.x, i = threadIdx.x;
    flag += static_cast<std::size_t>(r) * flag_stride;  // 0: one flag; 1: per-row flags (engine spec verify, C3)
    const dev::DsaSlot sl = d.slots[r];
    const unsigned t = d.tpos[r];
    hist0[r * 256 + i] = 0;
    hist0[r * 256 + 128 + i] = 0;
    const float kv = key[static_cast<std::size_t>(r) * kIdxDim + i];
    const float mu = lds_tree<128>(s, 0.0F + kv) / static_cast<float>(kIdxDim);
    const float dd = kv - mu;
    const float var = lds_tree<128>(s, __builtin_fmaf(dd, dd, 0.0F)) / static_cast<float>(kIdxDim);
    const float rr = inv_sqrt(var + kLnEps);
    const std::uint16_t kb = float_to_bf16((dd * rr) * f16_to_float(w.k_norm_w[i]) + f16_to_float(w.k_norm_b[i]));
    const std::uint16_t gb = float_to_bf16(gs[static_cast<std::size_t>(r) * kIdxDim + i]);
    if (!bf16_is_finite(kb) || !bf16_is_finite(gb)) atomicOr(flag, 1U);
    sl.ikey[static_cast<std::size_t>(t) * kIdxDim + i] = kb;
    sl.igate[static_cast<std::size_t>(t) * kIdxDim + i] = gb;
    for (unsigned l = i; l < kKvLora; l += 128) {
        const std::uint16_t lb = float_to_bf16(latent[static_cast<std::size_t>(r) * kKvLora + l]);
        if (!bf16_is_finite(lb)) atomicOr(flag, 1U);
        sl.latent[static_cast<std::size_t>(t) * kKvLora + l] = lb;
    }
}
// Pool completion (R12, C4) in its own launch: when several rows append consecutive positions of ONE slot, a pool's
// four keys / gates may come from different blocks of dsa_store_kernel, which a block barrier cannot order.
__global__ __launch_bounds__(128) void dsa_pool_kernel(DsaRows d, dev::IndexerDev w, unsigned* flag, unsigned flag_stride) {
    const unsigned r = blockIdx.x, i = threadIdx.x;
    flag += static_cast<std::size_t>(r) * flag_stride;  // C4 -> the flag of the row that closes the pool
    const dev::DsaSlot sl = d.slots[r];
    const unsigned t = d.tpos[r];
    if ((t + 1) % kKpool == 0) {
        const unsigned t0 = t + 1 - kKpool;
        const auto offset = static_cast<std::size_t>(t0) * kIdxDim;
        const float K = dsa_pool_channel(sl.ikey + offset, sl.igate + offset, w.ape, i);
        if (!is_finite(K)) atomicOr(flag, 1U);
        sl.pooled[static_cast<std::size_t>(t / kKpool) * kIdxDim + i] = K;
    }
}

// ---------------------------------------------------------------- indexer scores (R11) + top-byte histogram
// Host: for pool p: acc = +0; for h ascending: dot = chain_i fmaf(qI[h][i], K[i]); acc = fmaf(w_h, relu(dot * c), acc).
// One lane per pool; the 32 head chains run side by side (each i-ascending), qI / w_h from LDS.
constexpr unsigned kScoreThreads = 64;
__global__ __launch_bounds__(kScoreThreads) void idx_score_kernel(DsaRows d, const float* qI, const float* wraw,
                                                                  std::uint32_t* keys, unsigned key_stride,
                                                                  std::uint32_t* hist0) {
    __shared__ float4 qs[kIdxHeads * kIdxDim / 4];
    __shared__ float wh[kIdxHeads];
    __shared__ unsigned lh[256];
    const unsigned r = blockIdx.y, l = threadIdx.x;
    const unsigned P = (d.tpos[r] + 1) / kKpool, p0 = blockIdx.x * kScoreThreads;
    if (p0 >= P) return;
    const float4* q4 = reinterpret_cast<const float4*>(qI + static_cast<std::size_t>(r) * kIdxHeads * kIdxDim);
    for (unsigned e = l; e < kIdxHeads * kIdxDim / 4; e += kScoreThreads) qs[e] = q4[e];
    if (l < kIdxHeads) wh[l] = wraw[static_cast<std::size_t>(r) * kIdxHeads + l] * kInvSqrt32;
    for (unsigned e = l; e < 256; e += kScoreThreads) lh[e] = 0;
    __syncthreads();
    const unsigned p = p0 + l;
    if (p < P) {
        const float4* K = reinterpret_cast<const float4*>(d.slots[r].pooled + static_cast<std::size_t>(p) * kIdxDim);
        float dot[kIdxHeads];
#pragma unroll
        for (unsigned h = 0; h < kIdxHeads; ++h) dot[h] = 0.0F;
        for (unsigned i4 = 0; i4 < kIdxDim / 4; ++i4) {
            const float4 kv = K[i4];
#pragma unroll
            for (unsigned h = 0; h < kIdxHeads; ++h) {
                const float4 q = qs[h * (kIdxDim / 4) + i4];
                dot[h] = __builtin_fmaf(q.x, kv.x, dot[h]);
                dot[h] = __builtin_fmaf(q.y, kv.y, dot[h]);
                dot[h] = __builtin_fmaf(q.z, kv.z, dot[h]);
                dot[h] = __builtin_fmaf(q.w, kv.w, dot[h]);
            }
        }
        float acc = 0.0F;
#pragma unroll
        for (unsigned h = 0; h < kIdxHeads; ++h) {
            const float sc = dot[h] * kInvSqrt128;
            acc = __builtin_fmaf(wh[h], sc > 0.0F ? sc : 0.0F, acc);
        }
        const std::uint32_t key = ordered_key(acc);
        keys[static_cast<std::size_t>(r) * key_stride + p] = key;
        atomicAdd(&lh[key >> 24U], 1U);
    }
    __syncthreads();
    for (unsigned e = l; e < 256; e += kScoreThreads)
        if (lh[e]) atomicAdd(&hist0[r * 256 + e], lh[e]);
}

// RG=2 uses 35,072 bytes of LDS; RG=4 would exceed 64 KiB.
template <unsigned RG>
__global__ __launch_bounds__(kScoreThreads) void idx_score_rg_kernel(DsaRows d, const float* qI, const float* wraw,
                                                                     std::uint32_t* keys, unsigned key_stride,
                                                                     std::uint32_t* hist0) {
    static_assert(RG == 2, "indexer row grouping must fit in 64 KiB LDS");
    __shared__ float4 qs[RG][kIdxHeads * kIdxDim / 4];
    __shared__ float wh[RG][kIdxHeads];
    __shared__ unsigned lh[RG][256];
    static_assert(sizeof(qs) + sizeof(wh) + sizeof(lh) <= 65536, "indexer LDS budget");
    const unsigned g0 = blockIdx.y * RG, l = threadIdx.x;
    const unsigned p = blockIdx.x * kScoreThreads + l;
    const float* pooled[RG];
    unsigned pools[RG];
#pragma unroll
    for (unsigned u = 0; u < RG; ++u) {
        const unsigned r = g0 + u;
        pooled[u] = r < d.rows ? d.slots[r].pooled : nullptr;
        pools[u] = r < d.rows ? (d.tpos[r] + 1) / kKpool : 0U;
        if (r < d.rows) {
            const float4* q4 = reinterpret_cast<const float4*>(qI + static_cast<std::size_t>(r) * kIdxHeads * kIdxDim);
            for (unsigned e = l; e < kIdxHeads * kIdxDim / 4; e += kScoreThreads) qs[u][e] = q4[e];
            if (l < kIdxHeads) wh[u][l] = wraw[static_cast<std::size_t>(r) * kIdxHeads + l] * kInvSqrt32;
        }
        for (unsigned e = l; e < 256; e += kScoreThreads) lh[u][e] = 0;
    }
    __syncthreads();
    // Pass 0 shares the first slot's keys. Other slots retain their own key loads.
#pragma unroll
    for (unsigned pass = 0; pass < RG; ++pass) {
        bool active[RG];
        bool any = false;
#pragma unroll
        for (unsigned u = 0; u < RG; ++u) {
            active[u] = p < pools[u] &&
                        (pass == 0 ? pooled[u] == pooled[0] : u == pass && pooled[u] != pooled[0]);
            any = any || active[u];
        }
        if (any) {
            const float4* K = reinterpret_cast<const float4*>(pooled[pass] + static_cast<std::size_t>(p) * kIdxDim);
            float dot[RG][kIdxHeads];
#pragma unroll
            for (unsigned u = 0; u < RG; ++u)
#pragma unroll
                for (unsigned h = 0; h < kIdxHeads; ++h) dot[u][h] = 0.0F;
            for (unsigned i4 = 0; i4 < kIdxDim / 4; ++i4) {
                const float4 kv = K[i4];
#pragma unroll
                for (unsigned u = 0; u < RG; ++u) {
                    if (active[u]) {
#pragma unroll
                        for (unsigned h = 0; h < kIdxHeads; ++h) {
                            const float4 q = qs[u][h * (kIdxDim / 4) + i4];
                            dot[u][h] = __builtin_fmaf(q.x, kv.x, dot[u][h]);
                            dot[u][h] = __builtin_fmaf(q.y, kv.y, dot[u][h]);
                            dot[u][h] = __builtin_fmaf(q.z, kv.z, dot[u][h]);
                            dot[u][h] = __builtin_fmaf(q.w, kv.w, dot[u][h]);
                        }
                    }
                }
            }
#pragma unroll
            for (unsigned u = 0; u < RG; ++u) {
                if (active[u]) {
                    float acc = 0.0F;
#pragma unroll
                    for (unsigned h = 0; h < kIdxHeads; ++h) {
                        const float sc = dot[u][h] * kInvSqrt128;
                        acc = __builtin_fmaf(wh[u][h], sc > 0.0F ? sc : 0.0F, acc);
                    }
                    const std::uint32_t key = ordered_key(acc);
                    keys[static_cast<std::size_t>(g0 + u) * key_stride + p] = key;
                    atomicAdd(&lh[u][key >> 24U], 1U);
                }
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (unsigned u = 0; u < RG; ++u)
        if (g0 + u < d.rows)
            for (unsigned e = l; e < 256; e += kScoreThreads)
                if (lh[u][e]) atomicAdd(&hist0[(g0 + u) * 256 + e], lh[u][e]);
}

// Register-blocked score variants: RK_ROWS=2 (selector 3), RK_ROWS=1 (selector 5).
// 64 lanes cover 128 pools: lane l owns p0+l and p0+l+64. Only row histograms use LDS.
// The two-row form has 128 dot accumulators/lane (one-row: 64). Review generated ISA
// for scalar query loads, VGPR allocation and spills; selector 5 is the lower-register option.
template <unsigned RK_ROWS>
__global__ __launch_bounds__(kScoreThreads) void idx_score_rk_kernel(DsaRows d, const float* qI, const float* wraw,
                                                                     std::uint32_t* keys, unsigned key_stride,
                                                                     std::uint32_t* hist0) {
    static_assert(RK_ROWS == 1 || RK_ROWS == 2, "indexer register-blocked rows");
    __shared__ unsigned lh[RK_ROWS][256];
    const unsigned g0 = blockIdx.y * RK_ROWS, l = threadIdx.x;
    const unsigned p0 = blockIdx.x * (2 * kScoreThreads);
    const float* pooled[RK_ROWS];
    unsigned pools[RK_ROWS];
    unsigned max_pools = 0;
#pragma unroll
    for (unsigned u = 0; u < RK_ROWS; ++u) {
        const unsigned r = g0 + u;
        pooled[u] = r < d.rows ? d.slots[r].pooled : nullptr;
        pools[u] = r < d.rows ? (d.tpos[r] + 1) / kKpool : 0U;
        max_pools = pools[u] > max_pools ? pools[u] : max_pools;
    }
    if (p0 >= max_pools) return;  // CTA-uniform, including zero-pool rows
#pragma unroll
    for (unsigned u = 0; u < RK_ROWS; ++u)
        for (unsigned e = l; e < 256; e += kScoreThreads) lh[u][e] = 0;
    __syncthreads();
    float dot[RK_ROWS][2][kIdxHeads];
#pragma unroll
    for (unsigned u = 0; u < RK_ROWS; ++u)
#pragma unroll
        for (unsigned j = 0; j < 2; ++j)
#pragma unroll
            for (unsigned h = 0; h < kIdxHeads; ++h) dot[u][j][h] = 0.0F;
    // For a shared slot, the first row's key loads cover the union of both P bounds.
    unsigned first_pools = pools[0];
    if constexpr (RK_ROWS == 2)
        if (pooled[1] == pooled[0]) first_pools = max_pools;
    for (unsigned i4 = 0; i4 < kIdxDim / 4; ++i4) {
        float4 kv[2];
#pragma unroll
        for (unsigned j = 0; j < 2; ++j) {
            const unsigned p = p0 + l + j * kScoreThreads;
            kv[j] = float4{};
            if (p < first_pools)
                kv[j] = reinterpret_cast<const float4*>(pooled[0] + static_cast<std::size_t>(p) * kIdxDim)[i4];
        }
#pragma unroll
        for (unsigned u = 0; u < RK_ROWS; ++u) {
            if (p0 < pools[u]) {  // uniform: never predicate query loads on a lane's key validity
                float4 row_kv[2];
#pragma unroll
                for (unsigned j = 0; j < 2; ++j) {
                    row_kv[j] = kv[j];
                    if (pooled[u] != pooled[0]) {
                        const unsigned p = p0 + l + j * kScoreThreads;
                        row_kv[j] = float4{};
                        if (p < pools[u])
                            row_kv[j] = reinterpret_cast<const float4*>(pooled[u] + static_cast<std::size_t>(p) * kIdxDim)[i4];
                    }
                }
                // Uniform addresses permit s_load: no lane index or lane-dependent branch.
                // Two-head microtiles, double-buffered: prefetch the next pair before using
                // the current pair, then rotate. Each query float4 serves both keys.
                const float4* q4 = reinterpret_cast<const float4*>(
                    qI + static_cast<std::size_t>(g0 + u) * kIdxHeads * kIdxDim);
                float4 qa = q4[i4], qb = q4[kIdxDim / 4 + i4];
#pragma unroll
                for (unsigned h = 0; h < kIdxHeads; h += 2) {
                    float4 next_a{}, next_b{};
                    if (h + 2 < kIdxHeads) {
                        next_a = q4[(h + 2) * (kIdxDim / 4) + i4];
                        next_b = q4[(h + 3) * (kIdxDim / 4) + i4];
                    }
#pragma unroll
                    for (unsigned j = 0; j < 2; ++j) {
                        dot[u][j][h] = __builtin_fmaf(qa.x, row_kv[j].x, dot[u][j][h]);
                        dot[u][j][h] = __builtin_fmaf(qa.y, row_kv[j].y, dot[u][j][h]);
                        dot[u][j][h] = __builtin_fmaf(qa.z, row_kv[j].z, dot[u][j][h]);
                        dot[u][j][h] = __builtin_fmaf(qa.w, row_kv[j].w, dot[u][j][h]);
                        dot[u][j][h + 1] = __builtin_fmaf(qb.x, row_kv[j].x, dot[u][j][h + 1]);
                        dot[u][j][h + 1] = __builtin_fmaf(qb.y, row_kv[j].y, dot[u][j][h + 1]);
                        dot[u][j][h + 1] = __builtin_fmaf(qb.z, row_kv[j].z, dot[u][j][h + 1]);
                        dot[u][j][h + 1] = __builtin_fmaf(qb.w, row_kv[j].w, dot[u][j][h + 1]);
                    }
                    qa = next_a;
                    qb = next_b;
                }
            }
        }
    }
#pragma unroll
    for (unsigned u = 0; u < RK_ROWS; ++u) {
        if (p0 < pools[u]) {
            float acc[2] = {0.0F, 0.0F};
#pragma unroll
            for (unsigned h = 0; h < kIdxHeads; ++h) {
                const float wh = wraw[static_cast<std::size_t>(g0 + u) * kIdxHeads + h] * kInvSqrt32;
#pragma unroll
                for (unsigned j = 0; j < 2; ++j) {
                    const float sc = dot[u][j][h] * kInvSqrt128;
                    acc[j] = __builtin_fmaf(wh, sc > 0.0F ? sc : 0.0F, acc[j]);
                }
            }
#pragma unroll
            for (unsigned j = 0; j < 2; ++j) {
                const unsigned p = p0 + l + j * kScoreThreads;
                if (p < pools[u]) {
                    const std::uint32_t key = ordered_key(acc[j]);
                    keys[static_cast<std::size_t>(g0 + u) * key_stride + p] = key;
                    atomicAdd(&lh[u][key >> 24U], 1U);
                }
            }
        }
    }
    __syncthreads();
#pragma unroll
    for (unsigned u = 0; u < RK_ROWS; ++u)
        if (g0 + u < d.rows)
            for (unsigned e = l; e < 256; e += kScoreThreads)
                if (lh[u][e]) atomicAdd(&hist0[(g0 + u) * 256 + e], lh[u][e]);
}

// ---------------------------------------------------------------- R5 exact top-k + canonical token list
// Radix select of the k-th largest key K* (top byte from hist0, then 3 byte passes); selected = key > K* plus the
// lowest-index pools with key == K* (the host stable order's first k). Ordered compaction: thread l owns the contiguous
// pool segment [l * seg, (l + 1) * seg); exclusive block scans give each segment's equal-rank base and output base.
__global__ __launch_bounds__(1024) void idx_select_kernel(DsaRows d, const std::uint32_t* keys, unsigned key_stride,
                                                          const std::uint32_t* hist0, std::uint32_t* tokens,
                                                          std::uint32_t* n_out) {
    __shared__ unsigned hist[256], wsum[32];
    __shared__ unsigned sh_prefix, sh_rem;
    const unsigned r = blockIdx.x, l = threadIdx.x;
    const unsigned t = d.tpos[r], P = (t + 1) / kKpool, k = P < kIdxPools ? P : kIdxPools;
    const std::uint32_t* kr = keys + static_cast<std::size_t>(r) * key_stride;
    std::uint32_t* tok = tokens + static_cast<std::size_t>(r) * kTokStride;
    strix::attn_core::select::select_row<strix::attn_core::select::SegmentPolicy<kKpool>>(
        kr, P, t, k, l, tok, n_out + r,
        strix::attn_core::select::SegmentStorage{hist, wsum, sh_prefix, sh_rem}, hist0, r);
}

// ---------------------------------------------------------------- MLA (R13)
// absorb: host qt[l] = chain over d = 0..255 of fmaf(q[h][d], kv_b[h*512 + d][l]). Lane = 2 latent dims (one 32-bit
// load of two F16 per row d: the wave reads 128 contiguous bytes per step); R rows share the weight.
template <unsigned R>
__global__ __launch_bounds__(128) void mla_absorb_kernel(const float* q, const std::uint16_t* kvb, unsigned H, float* qt,
                                                         std::uint32_t* mkey) {
    __shared__ float qsh[R][kMlaDim];
    const unsigned h = blockIdx.y, l0 = blockIdx.x * 256 + 2 * threadIdx.x;
    for (unsigned e = threadIdx.x; e < R * kMlaDim; e += 128) {
        const unsigned r = e / kMlaDim, dd = e - r * kMlaDim;
        qsh[r][dd] = q[(static_cast<std::size_t>(r) * H + h) * kMlaDim + dd];
    }
    if (blockIdx.x == 0 && threadIdx.x < R) mkey[threadIdx.x * H + h] = 0;
    __syncthreads();
    const std::uint32_t* w = reinterpret_cast<const std::uint32_t*>(kvb + static_cast<std::size_t>(h) * 2 * kMlaDim * kKvLora + l0);
    float a0[R], a1[R];
#pragma unroll
    for (unsigned r = 0; r < R; ++r) a0[r] = a1[r] = 0.0F;
    for (unsigned d0 = 0; d0 < kMlaDim; d0 += 8) {
        std::uint32_t v[8];
#pragma unroll
        for (unsigned u = 0; u < 8; ++u) v[u] = w[static_cast<std::size_t>(d0 + u) * (kKvLora / 2)];
#pragma unroll
        for (unsigned u = 0; u < 8; ++u) {
            const float w0 = f16_bits_to_float(v[u] & 0xffffU);
            const float w1 = f16_bits_to_float(v[u] >> 16U);
#pragma unroll
            for (unsigned r = 0; r < R; ++r) {
                a0[r] = __builtin_fmaf(qsh[r][d0 + u], w0, a0[r]);
                a1[r] = __builtin_fmaf(qsh[r][d0 + u], w1, a1[r]);
            }
        }
    }
#pragma unroll
    for (unsigned r = 0; r < R; ++r) {
        float* o = qt + (static_cast<std::size_t>(r) * H + h) * kKvLora + l0;
        o[0] = a0[r];
        o[1] = a1[r];
    }
}

// scores: host s_t = (chain over l of fmaf(qt[l], c_t[l])) * 1/16. Lane = token, 16 heads side by side (each chain
// l-ascending); qt of the 16 heads in LDS (broadcast reads). Per-head max over the block -> atomicMax of the ordered key
// (a max is exact in any order).
constexpr unsigned kScHeads = 16, kScThreads = 64;
template <class Policy = DsaAttentionPolicy<false>, class... Context>
__global__ __launch_bounds__(kScThreads) void mla_scores_lane_kernel(DsaRows d, const std::uint32_t* tokens, const std::uint32_t* n,
                                                                const float* qt, unsigned H, float* sc, std::uint32_t* mkey,
                                                                Context... context) {
    __shared__ float4 qs[kScHeads * kKvLora / 4];
    const unsigned r = blockIdx.z, hg = blockIdx.y * kScHeads, l = threadIdx.x;
    const unsigned nr = n[r], t = blockIdx.x * kScThreads + l;
    if (blockIdx.x * kScThreads >= nr) return;
    const float4* q4 = reinterpret_cast<const float4*>(qt + (static_cast<std::size_t>(r) * H + hg) * kKvLora);
    for (unsigned e = l; e < kScHeads * kKvLora / 4; e += kScThreads) qs[e] = q4[e];
    __syncthreads();
    const bool valid = t < nr;
    const std::uint32_t pos = tokens[static_cast<std::size_t>(r) * kTokStride + (valid ? t : 0)];
    const uint4* lat = reinterpret_cast<const uint4*>(d.slots[r].latent + static_cast<std::size_t>(Policy::address(pos, context...)) * kKvLora);
    float acc[kScHeads];
#pragma unroll
    for (unsigned hh = 0; hh < kScHeads; ++hh) acc[hh] = 0.0F;
    for (unsigned l8 = 0; l8 < kKvLora / 8; ++l8) {
        float c[8];
        unpack_bf16x8(lat[l8], c);
#pragma unroll
        for (unsigned hh = 0; hh < kScHeads; ++hh) {
            const float4 qa = qs[hh * (kKvLora / 4) + 2 * l8], qb = qs[hh * (kKvLora / 4) + 2 * l8 + 1];
            acc[hh] = __builtin_fmaf(qa.x, c[0], acc[hh]);
            acc[hh] = __builtin_fmaf(qa.y, c[1], acc[hh]);
            acc[hh] = __builtin_fmaf(qa.z, c[2], acc[hh]);
            acc[hh] = __builtin_fmaf(qa.w, c[3], acc[hh]);
            acc[hh] = __builtin_fmaf(qb.x, c[4], acc[hh]);
            acc[hh] = __builtin_fmaf(qb.y, c[5], acc[hh]);
            acc[hh] = __builtin_fmaf(qb.z, c[6], acc[hh]);
            acc[hh] = __builtin_fmaf(qb.w, c[7], acc[hh]);
        }
    }
#pragma unroll
    for (unsigned hh = 0; hh < kScHeads; ++hh) {
        float v = acc[hh] * kMlaScale;
        if constexpr (Policy::partial) {
            if (valid && Policy::owns(pos, context...) && !is_finite(v)) (atomicOr(context.flag, 1U), ...);
            if (!Policy::owns(pos, context...)) v = -__builtin_huge_valf();
        }
        if (valid) sc[(static_cast<std::size_t>(r) * H + hg + hh) * kTokStride + t] = v;
        float m = valid ? v : -__builtin_huge_valf();
#pragma unroll
        for (unsigned off = 16; off > 0; off >>= 1U) {
            const float o = __shfl_xor(m, static_cast<int>(off), 32);
            m = o > m ? o : m;
        }
        if ((l & 31U) == 0) atomicMax(&mkey[r * H + hg + hh], ordered_key(m));
    }
}

// Staged scores: block = 64 tokens x 16 heads, 256 threads; thread = token (tau % 64) x 4 heads (4 (tau / 64) ..),
// each chain l-ascending as above. The 64 latent rows are streamed through LDS in 128-dim chunks by the whole block
// (4 x 16 B per thread, rows padded to 17 x 16 B: conflict-free b128 reads); qt of the 16 heads in LDS.
constexpr unsigned kSsTok = 64, kSsL = 128, kSsPad = kSsL / 8 + 1;
template <class Policy = DsaAttentionPolicy<false>, class... Context>
__global__ __launch_bounds__(256) void mla_scores_staged_kernel(DsaRows d, const std::uint32_t* tokens, const std::uint32_t* n,
                                                                const float* qt, unsigned H, float* sc, std::uint32_t* mkey,
                                                                Context... context) {
    __shared__ float4 qs[kScHeads * kKvLora / 4];
    __shared__ u32x4 lt[kSsTok][kSsPad];
    const unsigned r = blockIdx.z, hg = blockIdx.y * kScHeads, tau = threadIdx.x, t0 = blockIdx.x * kSsTok;
    const unsigned nr = n[r];
    if (t0 >= nr) return;
    const float4* q4 = reinterpret_cast<const float4*>(qt + (static_cast<std::size_t>(r) * H + hg) * kKvLora);
    for (unsigned e = tau; e < kScHeads * kKvLora / 4; e += 256) qs[e] = q4[e];
    const std::uint32_t* tok = tokens + static_cast<std::size_t>(r) * kTokStride;
    const std::uint16_t* latent = d.slots[r].latent;
    // loader: element e = tau + 256 k (k = 0..3): token e / 16, column (e % 16) * 8 of the chunk
    const std::uint16_t* lrow[4];
#pragma unroll
    for (unsigned k = 0; k < 4; ++k) {
        const unsigned tt = t0 + (tau + 256 * k) / 16;
        lrow[k] = latent + static_cast<std::size_t>(Policy::address(tok[tt < nr ? tt : 0], context...)) * kKvLora + ((tau + 256 * k) % 16) * 8;
    }
    const unsigned t = tau % kSsTok, hq = (tau / kSsTok) * 4;
    float acc[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    for (unsigned c = 0; c < kKvLora / kSsL; ++c) {
        u32x4 v[4];
#pragma unroll
        for (unsigned k = 0; k < 4; ++k) v[k] = *reinterpret_cast<const u32x4*>(lrow[k] + c * kSsL);
        __syncthreads();  // previous chunk consumed (and, first time, qs visible after the next barrier)
#pragma unroll
        for (unsigned k = 0; k < 4; ++k) lt[(tau + 256 * k) / 16][(tau + 256 * k) % 16] = v[k];
        __syncthreads();
        for (unsigned l8 = 0; l8 < kSsL / 8; ++l8) {
            const u32x4 lv = lt[t][l8];
            float cv[8];
            unpack_bf16x8(uint4{lv.x, lv.y, lv.z, lv.w}, cv);
#pragma unroll
            for (unsigned hh = 0; hh < 4; ++hh) {
                const unsigned qi = (hq + hh) * (kKvLora / 4) + (c * kSsL + l8 * 8) / 4;
                const float4 qa = qs[qi], qb = qs[qi + 1];
                acc[hh] = __builtin_fmaf(qa.x, cv[0], acc[hh]);
                acc[hh] = __builtin_fmaf(qa.y, cv[1], acc[hh]);
                acc[hh] = __builtin_fmaf(qa.z, cv[2], acc[hh]);
                acc[hh] = __builtin_fmaf(qa.w, cv[3], acc[hh]);
                acc[hh] = __builtin_fmaf(qb.x, cv[4], acc[hh]);
                acc[hh] = __builtin_fmaf(qb.y, cv[5], acc[hh]);
                acc[hh] = __builtin_fmaf(qb.z, cv[6], acc[hh]);
                acc[hh] = __builtin_fmaf(qb.w, cv[7], acc[hh]);
            }
        }
    }
    const bool valid = t0 + t < nr;
#pragma unroll
    for (unsigned hh = 0; hh < 4; ++hh) {
        float v = acc[hh] * kMlaScale;
        if constexpr (Policy::partial) {
            const bool owned = Policy::owns(tok[valid ? t0 + t : 0], context...);
            if (valid && owned && !is_finite(v)) (atomicOr(context.flag, 1U), ...);
            if (!owned) v = -__builtin_huge_valf();
        }
        if (valid) sc[(static_cast<std::size_t>(r) * H + hg + hq + hh) * kTokStride + t0 + t] = v;
        float m = valid ? v : -__builtin_huge_valf();
#pragma unroll
        for (unsigned off = 16; off > 0; off >>= 1U) {
            const float o = __shfl_xor(m, static_cast<int>(off), 32);
            m = o > m ? o : m;
        }
        if ((tau & 31U) == 0) atomicMax(&mkey[r * H + hg + hq + hh], ordered_key(m));
    }
}

// softmax: host e_t = declared_exp(s_t - m); sum = tree_sum<256> (lane j: a = +0; a = a + e_t, t = j, j + 256, ...);
// p_t = e_t / sum. One block of 256 lanes per (head, row).
template <class Policy = DsaAttentionPolicy<false>, class... Context>
__global__ __launch_bounds__(256) void mla_softmax_kernel(const std::uint32_t* n, const float* sc, const std::uint32_t* mkey,
                                                          unsigned H, float* p, Context... context) {
    __shared__ float s[256];
    const unsigned h = blockIdx.x, r = blockIdx.y, j = threadIdx.x, nr = n[r];
    float m = key_to_float(mkey[r * H + h]);
    if constexpr (Policy::partial) {
        if (nr == 0) m = -__builtin_huge_valf();
    }
    const float* sr = sc + (static_cast<std::size_t>(r) * H + h) * kTokStride;
    float* pr = p + (static_cast<std::size_t>(r) * H + h) * kTokStride;
    constexpr unsigned kPer = (kMaxTok + 255) / 256;
    float e[kPer];
    float a = 0.0F;
#pragma unroll
    for (unsigned mm = 0; mm < kPer; ++mm) {
        const unsigned t = j + 256 * mm;
        e[mm] = t < nr ? Policy::weight(sr[t], m) : 0.0F;
        if (t < nr) a = a + e[mm];
    }
    const float sum = lds_tree<256>(s, a);
    if constexpr (Policy::partial) {
        if (j == 0) {
            const std::size_t offset = (static_cast<std::size_t>(r) * H + h) * (kKvLora + 2);
            ((context.output[offset] = m), ...);
            ((context.output[offset + 1] = sum), ...);
            if (!is_finite(sum)) (atomicOr(context.flag, 1U), ...);
        }
    }
#pragma unroll
    for (unsigned mm = 0; mm < kPer; ++mm) {
        const unsigned t = j + 256 * mm;
        if (t < nr) pr[t] = Policy::probability(e[mm], sum);
    }
}

// pv: host ob[l] = tree_fma<256> (lane j: b = +0; b = fmaf(p_t, c_t[l], b), t = j, j + 256, ...; then the halving tree).
// Block = 256 lanes j = threadIdx x HG heads x D latent dims (64 accumulators per lane; default 4 x 16). Tree:
// levels 128, 64, 32 across waves through LDS, levels 16 .. 1 inside wave 0 with shuffles (lane j + off -> lane j).
constexpr unsigned kPvHeads = 4, kPvDims = 16, kPvAcc = kPvHeads * kPvDims;
template <unsigned HG, unsigned D, class Policy = DsaAttentionPolicy<false>, class... Context>
__global__ __launch_bounds__(256) void mla_pv_kernel_t(DsaRows d, const std::uint32_t* tokens, const std::uint32_t* n,
                                                     const float* p, unsigned H, float* ob, Context... context) {
    static_assert(HG * D == kPvAcc && (D == 16 || D == 8 || D == 4));
    __shared__ float buf[kPvAcc][128];
    const unsigned l0 = blockIdx.x * D, hg = blockIdx.y * HG, r = blockIdx.z, j = threadIdx.x;
    const unsigned nr = n[r];
    const std::uint32_t* tok = tokens + static_cast<std::size_t>(r) * kTokStride;
    const std::uint16_t* lat = d.slots[r].latent;
    float acc[kPvAcc];
#pragma unroll
    for (unsigned a = 0; a < kPvAcc; ++a) acc[a] = 0.0F;
    constexpr unsigned kPer = (kMaxTok + 255) / 256;
#pragma unroll
    for (unsigned mm = 0; mm < kPer; ++mm) {
        const unsigned t = j + 256 * mm;
        if (t < nr && Policy::owns(tok[t], context...)) {
            float c[D];
            if constexpr (D == 4) {
                const uint2 v = *reinterpret_cast<const uint2*>(lat + static_cast<std::size_t>(Policy::address(tok[t], context...)) * kKvLora + l0);
                c[0] = __builtin_bit_cast(float, v.x << 16U);
                c[1] = __builtin_bit_cast(float, v.x & 0xffff0000U);
                c[2] = __builtin_bit_cast(float, v.y << 16U);
                c[3] = __builtin_bit_cast(float, v.y & 0xffff0000U);
            } else {
                const uint4* lv = reinterpret_cast<const uint4*>(lat + static_cast<std::size_t>(Policy::address(tok[t], context...)) * kKvLora + l0);
                unpack_bf16x8(lv[0], c);
                if constexpr (D == 16) unpack_bf16x8(lv[1], c + 8);
            }
#pragma unroll
            for (unsigned hh = 0; hh < HG; ++hh) {
                const float pv = p[(static_cast<std::size_t>(r) * H + hg + hh) * kTokStride + t];
#pragma unroll
                for (unsigned li = 0; li < D; ++li) acc[hh * D + li] = __builtin_fmaf(pv, c[li], acc[hh * D + li]);
            }
        }
    }
#pragma unroll
    for (unsigned lvl = 128; lvl >= 32; lvl >>= 1U) {
        if (j >= lvl && j < 2 * lvl)
#pragma unroll
            for (unsigned a = 0; a < kPvAcc; ++a) buf[a][j - lvl] = acc[a];
        __syncthreads();
        if (j < lvl)
#pragma unroll
            for (unsigned a = 0; a < kPvAcc; ++a) acc[a] = acc[a] + buf[a][j];
        __syncthreads();
    }
    if (j < 32) {
#pragma unroll
        for (unsigned off = 16; off > 0; off >>= 1U)
#pragma unroll
            for (unsigned a = 0; a < kPvAcc; ++a) acc[a] = acc[a] + __shfl_down(acc[a], off, 32);
        if (j == 0)
#pragma unroll
            for (unsigned hh = 0; hh < HG; ++hh)
#pragma unroll
                for (unsigned li = 0; li < D; ++li) {
                    if constexpr (Policy::partial) {
                        if (!is_finite(acc[hh * D + li])) (atomicOr(context.flag, 1U), ...);
                    }
                    ob[Policy::output_index(static_cast<std::size_t>(r) * H + hg + hh, l0 + li)] = acc[hh * D + li];
                }
    }
}

// out: host o[d] = chain over l of fmaf(ob[l], kv_b[h*512 + 256 + d][l]). Lane = output dim d (its V row read in 16-byte
// pieces, l ascending); R rows share the weight; ob of the head in LDS.
template <unsigned R>
__global__ __launch_bounds__(64) void mla_out_kernel(const float* ob, const std::uint16_t* kvb, unsigned H, float* o) {
    __shared__ float obs[R][kKvLora];
    const unsigned h = blockIdx.y, dd = blockIdx.x * 64 + threadIdx.x;
    for (unsigned e = threadIdx.x; e < R * kKvLora; e += 64) {
        const unsigned r = e / kKvLora, li = e - r * kKvLora;
        obs[r][li] = ob[(static_cast<std::size_t>(r) * H + h) * kKvLora + li];
    }
    __syncthreads();
    const uint4* w = reinterpret_cast<const uint4*>(kvb + (static_cast<std::size_t>(h) * 2 * kMlaDim + kMlaDim + dd) * kKvLora);
    float acc[R];
#pragma unroll
    for (unsigned r = 0; r < R; ++r) acc[r] = 0.0F;
    for (unsigned i8 = 0; i8 < kKvLora / 8; i8 += 2) {
        uint4 v[2];
#pragma unroll
        for (unsigned u = 0; u < 2; ++u) v[u] = w[i8 + u];
#pragma unroll
        for (unsigned u = 0; u < 2; ++u) {
            float wv[8];
            unpack_f16x8(v[u], wv);
#pragma unroll
            for (unsigned e = 0; e < 8; ++e)
#pragma unroll
                for (unsigned r = 0; r < R; ++r) acc[r] = __builtin_fmaf(obs[r][(i8 + u) * 8 + e], wv[e], acc[r]);
        }
    }
#pragma unroll
    for (unsigned r = 0; r < R; ++r) o[(static_cast<std::size_t>(r) * H + h) * kMlaDim + dd] = acc[r];
}

template <template <unsigned> class F, class... A> void by_rows(unsigned rows, A&&... a) {
    switch (rows) {
        case 1: F<1>::go(a...); break;
        case 2: F<2>::go(a...); break;
        case 3: F<3>::go(a...); break;
        case 4: F<4>::go(a...); break;
        case 5: F<5>::go(a...); break;
        case 6: F<6>::go(a...); break;
        case 7: F<7>::go(a...); break;
        case 8: F<8>::go(a...); break;
        default: throw std::invalid_argument("glm attn: rows must be 1..8");
    }
}
template <unsigned R> struct DenseGo {
    static void go(const DenseJobs& J, unsigned blocks, hipStream_t s) { dense_f16_kernel<R><<<blocks, kDenseThreads, 0, s>>>(J); }
};
template <unsigned R> struct DenseStagedGo {
    static void go(const DenseJobs& J, unsigned blocks, unsigned ob, hipStream_t s) {
        if (ob == 8)
            dense_f16_staged_kernel<R, 8><<<blocks, kStThreads, 0, s>>>(J);
        else
            dense_f16_staged_kernel<R, 16><<<blocks, kStThreads, 0, s>>>(J);
    }
};
std::atomic<int> g_dense_schedule{-1};  // -1: not read yet; 0 staged; 1 lane
std::atomic<unsigned> g_dense_tile{16};
bool dense_lane_forced() {
    int v = g_dense_schedule.load(std::memory_order_relaxed);
    if (v < 0) {
        const char* e = std::getenv("STRIX_GLM_ATTN_SCHED");
        int want = (e && std::string(e) == "lane") ? 1 : 0;
        g_dense_schedule.compare_exchange_strong(v, want);
        v = g_dense_schedule.load(std::memory_order_relaxed);
    }
    return v == 1;
}
// MLA scores schedule: lane by default. measurement (node 1): staged 51.9 us vs lane 46.1 us at 8K R1 (slower), 230.7 vs 242.9 us
// at R8 (faster). Measured cause: 50 KB LDS per block = 1 block per CU, 66 blocks at R1 = 2 rounds, no double
// buffering of the 4 latent chunks. OFF until fixed; STRIX_GLM_ATTN_SCORES=staged or set_scores_schedule turns it on.
// Auto (attention, default when nothing is forced): staged when the caller bounds n <= 1024 and rows <= 4 - measured
// three measurements on the pair, ctx 1024: R1 26.3/25.7/25.7 vs lane 35.0/34.5/34.5 us, R2 -1.5..-2.7, R4 -3.4..-4.5 us; lane
// stays faster at R8 and at n = 2048. Unknown bound (0) = lane.
std::atomic<int> g_scores_schedule{-1};  // -1: not read yet; 0 staged; 1 lane; 2 auto
bool scores_staged(unsigned max_tokens, unsigned rows) {
    int v = g_scores_schedule.load(std::memory_order_relaxed);
    if (v < 0) {
        const char* e = std::getenv("STRIX_GLM_ATTN_SCORES");
        int want = 2;
        if (e && std::string(e) == "staged") want = 0;
        if (e && std::string(e) == "lane") want = 1;
        g_scores_schedule.compare_exchange_strong(v, want);
        v = g_scores_schedule.load(std::memory_order_relaxed);
    }
    if (dense_lane_forced()) return false;
    return v == 0 || (v == 2 && max_tokens != 0 && max_tokens <= 1024 && rows <= 4);
}
template <unsigned R> struct AbsorbGo {
    static void go(const float* q, const std::uint16_t* kvb, unsigned H, float* qt, std::uint32_t* mkey, hipStream_t s) {
        mla_absorb_kernel<R><<<dim3(kKvLora / 256, H), 128, 0, s>>>(q, kvb, H, qt, mkey);
    }
};
template <unsigned R> struct OutGo {
    static void go(const float* ob, const std::uint16_t* kvb, unsigned H, float* o, hipStream_t s) {
        mla_out_kernel<R><<<dim3(kMlaDim / 64, H), 64, 0, s>>>(ob, kvb, H, o);
    }
};

void check_rows(unsigned rows) {
    if (rows == 0 || rows > kMaxRows) throw std::invalid_argument("glm attn: rows must be 1..8");
}
void check_heads(unsigned heads) {
    if (heads == 0 || heads % kScHeads != 0) throw std::invalid_argument("glm attn: heads must be a positive multiple of 16");
}

__global__ void f16_table_kernel(float* out) {
    const unsigned i = blockIdx.x * 256 + threadIdx.x;
    out[i] = f16_bits_to_float(i);
}

}  // namespace

void f16_table(float* out, hipStream_t s) {
    f16_table_kernel<<<256, 256, 0, s>>>(out);
    ok(hipGetLastError(), "glm fast f16_table");
}
void set_dense_schedule(DenseSchedule d) { g_dense_schedule.store(d == DenseSchedule::kLane ? 1 : 0); }
void set_scores_schedule(DenseSchedule d) { g_scores_schedule.store(d == DenseSchedule::kLane ? 1 : 0); }
void set_scores_auto() { g_scores_schedule.store(2); }
void set_dense_tile(unsigned outputs) {
    if (outputs != 8 && outputs != 16) throw std::invalid_argument("set_dense_tile: 8 or 16");
    g_dense_tile.store(outputs);
}

void dense_f16(const DenseJob* jobs, unsigned njobs, unsigned rows, hipStream_t s) {
    if (njobs == 0 || njobs > 4) throw std::invalid_argument("dense_f16: 1..4 jobs");
    check_rows(rows);
    // the staged schedule needs every job with in % 256 == 0 and 16 B aligned x; jobs of one launch share the schedule
    bool staged = !dense_lane_forced();
    for (unsigned q = 0; q < njobs; ++q) {
        const DenseJob& j = jobs[q];
        if (j.in == 0 || j.in % 32 != 0 || j.n == 0 || j.groups == 0 || (reinterpret_cast<std::uintptr_t>(j.w) & 15U) != 0 ||
            (j.w_gstride * 2) % 16 != 0)
            throw std::invalid_argument("dense_f16: in % 32 == 0, n > 0, groups > 0 and 16-byte aligned rows required");
        staged = staged && j.in % kStCols == 0 && (reinterpret_cast<std::uintptr_t>(j.x) & 15U) == 0 && j.x_stride % 4 == 0 &&
                 j.x_gstride % 4 == 0;
    }
    const unsigned ob = g_dense_tile.load();
    DenseJobs J{};
    unsigned blocks = 0;
    for (unsigned q = 0; q < njobs; ++q) {
        J.j[q] = jobs[q];
        J.block0[q] = blocks;
        const unsigned per = staged ? (jobs[q].n + ob - 1) / ob : (jobs[q].n + kDenseThreads - 1) / kDenseThreads;
        blocks += per * (staged ? jobs[q].groups : 1U);
        if (!staged && jobs[q].groups != 1) {  // the lane schedule runs groups as separate launches
            for (unsigned g = 0; g < jobs[q].groups; ++g) {
                DenseJob one = jobs[q];
                one.groups = 1;
                one.w += g * one.w_gstride;
                one.x += g * one.x_gstride;
                one.y += g * one.y_gstride;
                dense_f16(&one, 1, rows, s);
            }
            blocks -= per;
            J.j[q].n = 0;  // nothing left for this job in the shared launch
        }
    }
    J.block0[njobs] = blocks;
    J.njobs = njobs;
    if (blocks == 0) return;
    if (staged)
        by_rows<DenseStagedGo>(rows, J, blocks, ob, s);
    else
        by_rows<DenseGo>(rows, J, blocks, s);
    ok(hipGetLastError(), "glm fast dense_f16");
}

// PV CTA ownership; -1 means the environment has not been read.
std::atomic<int> g_mla_pv_variant{-1};
unsigned mla_pv_variant() {
    int v = g_mla_pv_variant.load(std::memory_order_relaxed);
    if (v < 0) {
        const char* e = std::getenv("STRIX_GLM_MLA_PV");
        int want = 0;
        if (e != nullptr) {
            const std::string es(e);
            if (es == "1") want = 1;
            if (es == "2") want = 2;
        }
        g_mla_pv_variant.compare_exchange_strong(v, want);
        v = g_mla_pv_variant.load(std::memory_order_relaxed);
    }
    return static_cast<unsigned>(v);
}

// Lanes per state column of the fused KDA step: STRIX_GLM_KDA_LANES = 2 (original), 4 or 8 (block ownership) or
// 2i / 4i / 8i (interleaved ownership k = qid + Q j, review) at first use; internal value Q | (interleaved ? 16 : 0).
// +32 (suffix t): 2-barrier trees. +64 (suffix s): column-split producers and a finalizer.
// 128 (2c): column-quad ownership, non-split; explicit opt-in only.
// 194 (2sp) / 196 (4sp): split with shared prep (opt-in); 200 (cqs): column-quad split with shared prep
// (auto at 2..8 rows); 128 (2c): column quad (auto at one row).
// set_kda_lanes overrides (gate / bench). Schedules preserve the declared tree;
// shared-prep modes await compiler, native bitwise, model and complete-workload qualification.
std::atomic<unsigned> g_kda_lanes{0};
bool kda_lanes_valid(unsigned q) {
    static constexpr unsigned ok_vals[] = {1U, 2U, 4U, 8U, 18U, 20U, 24U, 34U, 36U, 40U, 50U, 52U, 56U, 66U, 68U, 72U, 84U, 88U, 100U, 116U, 128U, 160U, 194U, 196U, 200U, 232U};
    for (unsigned w : ok_vals)
        if (q == w) return true;
    return false;
}
bool kda_tree() {
    static const bool on = [] { const char* e = std::getenv("STRIX_GLM_KDA_TREE"); return !(e != nullptr && e[0] == '0'); }();
    return on;
}
unsigned kda_lanes() {
    unsigned v = g_kda_lanes.load(std::memory_order_relaxed);
    if (v == 0) {
        const char* e = std::getenv("STRIX_GLM_KDA_LANES");
        unsigned want = 1U;  // auto (attention measurement): 2 at one row / kSeq, 4s (68) at 2..8 independent rows
        if (e != nullptr && *e != '\0') {
            const std::string es(e);  // Unrecognized names keep the default (2).
            static constexpr const char* names[] = {"2", "4", "8", "2i", "4i", "8i", "2t", "4t", "8t", "2it", "4it", "8it", "2s", "4s", "8s", "4is", "8is", "4ts", "4its", "2c", "2sp", "4sp", "cqs", "2ct", "cqst"};
            static constexpr unsigned vals[] = {2U, 4U, 8U, 18U, 20U, 24U, 34U, 36U, 40U, 50U, 52U, 56U, 66U, 68U, 72U, 84U, 88U, 100U, 116U, 128U, 194U, 196U, 200U, 160U, 232U};
            for (unsigned n = 0; n < sizeof(vals) / sizeof(vals[0]); ++n)
                if (es == names[n]) want = vals[n];
        }
        g_kda_lanes.compare_exchange_strong(v, want);
        v = g_kda_lanes.load(std::memory_order_relaxed);
    }
    return v;
}
// Per-process workspace on the current device, retained until process exit. Like the
// existing callers, shared-prep launches require serialized host calls on one stream
// (and one device); concurrent streams / devices must not share this buffer.
// hipMalloc on growth synchronizes; reuse at the same or smaller size does not allocate.
static float* g_kda_prep_work = nullptr;
static std::size_t g_kda_prep_capacity = 0;  // floats

template <bool kSeq>
void kda_launch(dim3 grid, hipStream_t s, const dev::KdaDev& k, const float* qkv, const float* f, const float* gate, const float* b,
                float* S, float* conv_state, float* y, float* snap, float* conv_snap, unsigned* flag, unsigned npos) {
    unsigned lanes = kda_lanes();
    // Auto (1): measurement (node 1, 23:21, gate 235/235 bitwise) kda.step us q2 vs 4s: R1 36.2-37.1 vs 39.4-39.9 (4s slower),
    // R2 74.1-74.7 vs 66.8-67.2 (-10 %), R4 124.4-124.9 vs 115.1-115.2 (-7.6 %), R8 218.7-220.2 vs 209.5-209.6 (-4.4 %);
    // kSeq (verify positions of one slot) not measured with the split -> stays 2. Same bits either way.
    // Auto update (attention measurement, node 1 00:45, gate 114/114 bitwise): kda.step us q2 / 2c / cqs: R1 36.1-37.1 / 34.4 / 37.1,
    // R2 73.9-74.8 / 64.9 / 62.8-63.0, R4 123.9-125.2 / 132.1 / 109.3-109.5, R8 218.8-219.8 / 237.0 / 207.8-207.9.
    // -> 2c at one row, cqs (column-quad split, shared prep) at 2..8 rows; kSeq unmeasured for these -> 2.
    // auto: the 2-barrier-tree column-quad forms (2ct at one row, cqst at 2..8 rows; same bits as 2c / cqs, kda.step
    // -0.5..-1.8 %); STRIX_GLM_KDA_TREE=0 keeps 2c / cqs
    if (lanes == 1U) lanes = kSeq ? 2U : (grid.y < 2 ? (kda_tree() ? 160U : 128U) : (kda_tree() ? 232U : 200U));
    if (lanes == 194U || lanes == 196U || lanes == 200U || lanes == 232U) {
        const std::size_t needed = static_cast<std::size_t>(std::max(grid.y, npos)) * k.heads * (4 * kKdaDim + 1);
        if (needed > g_kda_prep_capacity) {
            float* work = nullptr;
            ok(hipMalloc(reinterpret_cast<void**>(&work), needed * sizeof(float)), "glm fast kda prep workspace");
            if (g_kda_prep_work != nullptr) ok(hipFree(g_kda_prep_work), "glm fast kda prep workspace free");
            g_kda_prep_work = work;
            g_kda_prep_capacity = needed;
        }
        if (lanes == 232U)
            kda_prep_kernel<kSeq, true><<<grid, 128, 0, s>>>(k, qkv, f, b, conv_state, conv_snap, g_kda_prep_work, npos);
        else
            kda_prep_kernel<kSeq, false><<<grid, 128, 0, s>>>(k, qkv, f, b, conv_state, conv_snap, g_kda_prep_work, npos);
        ok(hipGetLastError(), "glm fast kda shared prep");
    }
    switch (lanes) {
    case 200:
        kda_fused_kernel<kSeq, 2, false, false, true, true, true><<<dim3(k.heads * 2U, grid.y), 128, 0, s>>>(k, qkv, f, gate, b, S, conv_state, y, snap, conv_snap, flag, npos, g_kda_prep_work);
        ok(hipGetLastError(), "glm fast kda shared-prep column-quad split producer");
        kda_r10_finalize_kernel<kSeq, true><<<dim3(k.heads, kSeq ? npos : grid.y), 128, 0, s>>>(k, qkv, gate, conv_state, y, npos, false);
        break;
    case 194:
        kda_fused_kernel<kSeq, 2, false, false, true, false, true><<<dim3(k.heads * 2U, grid.y), 128, 0, s>>>(k, qkv, f, gate, b, S, conv_state, y, snap, conv_snap, flag, npos, g_kda_prep_work);
        ok(hipGetLastError(), "glm fast kda shared-prep split producer");
        kda_r10_finalize_kernel<kSeq, true><<<dim3(k.heads, kSeq ? npos : grid.y), 128, 0, s>>>(k, qkv, gate, conv_state, y, npos, false);
        break;
    case 196:
        kda_fused_kernel<kSeq, 4, false, false, true, false, true><<<dim3(k.heads * 4U, grid.y), 128, 0, s>>>(k, qkv, f, gate, b, S, conv_state, y, snap, conv_snap, flag, npos, g_kda_prep_work);
        ok(hipGetLastError(), "glm fast kda shared-prep split producer");
        kda_r10_finalize_kernel<kSeq, true><<<dim3(k.heads, kSeq ? npos : grid.y), 128, 0, s>>>(k, qkv, gate, conv_state, y, npos, false);
        break;
    case 128: kda_fused_kernel<kSeq, 2, false, false, false, true><<<grid, 256, 0, s>>>(k, qkv, f, gate, b, S, conv_state, y, snap, conv_snap, flag, npos); break;
    case 160: kda_fused_kernel<kSeq, 2, false, true, false, true><<<grid, 256, 0, s>>>(k, qkv, f, gate, b, S, conv_state, y, snap, conv_snap, flag, npos); break;
    case 232:  // cqs with the 2-barrier trees (prep norms + R10 finalizer)
        kda_fused_kernel<kSeq, 2, false, true, true, true, true><<<dim3(k.heads * 2U, grid.y), 128, 0, s>>>(k, qkv, f, gate, b, S, conv_state, y, snap, conv_snap, flag, npos, g_kda_prep_work);
        ok(hipGetLastError(), "glm fast kda shared-prep column-quad split producer (t)");
        kda_r10_finalize_kernel<kSeq, true><<<dim3(k.heads, kSeq ? npos : grid.y), 128, 0, s>>>(k, qkv, gate, conv_state, y, npos, true);
        break;
    case 66:
        kda_fused_kernel<kSeq, 2, false, false, true><<<dim3(k.heads * 2U, grid.y), 128, 0, s>>>(k, qkv, f, gate, b, S, conv_state, y, snap, conv_snap, flag, npos);
        ok(hipGetLastError(), "glm fast kda split producer");
        kda_r10_finalize_kernel<kSeq><<<dim3(k.heads, kSeq ? npos : grid.y), 128, 0, s>>>(k, qkv, gate, conv_state, y, npos, false);
        break;
    case 68:
        kda_fused_kernel<kSeq, 4, false, false, true><<<dim3(k.heads * 4U, grid.y), 128, 0, s>>>(k, qkv, f, gate, b, S, conv_state, y, snap, conv_snap, flag, npos);
        ok(hipGetLastError(), "glm fast kda split producer");
        kda_r10_finalize_kernel<kSeq><<<dim3(k.heads, kSeq ? npos : grid.y), 128, 0, s>>>(k, qkv, gate, conv_state, y, npos, false);
        break;
    case 72:
        kda_fused_kernel<kSeq, 8, false, false, true><<<dim3(k.heads * 8U, grid.y), 128, 0, s>>>(k, qkv, f, gate, b, S, conv_state, y, snap, conv_snap, flag, npos);
        ok(hipGetLastError(), "glm fast kda split producer");
        kda_r10_finalize_kernel<kSeq><<<dim3(k.heads, kSeq ? npos : grid.y), 128, 0, s>>>(k, qkv, gate, conv_state, y, npos, false);
        break;
    case 84:
        kda_fused_kernel<kSeq, 4, true, false, true><<<dim3(k.heads * 4U, grid.y), 128, 0, s>>>(k, qkv, f, gate, b, S, conv_state, y, snap, conv_snap, flag, npos);
        ok(hipGetLastError(), "glm fast kda split producer");
        kda_r10_finalize_kernel<kSeq><<<dim3(k.heads, kSeq ? npos : grid.y), 128, 0, s>>>(k, qkv, gate, conv_state, y, npos, false);
        break;
    case 88:
        kda_fused_kernel<kSeq, 8, true, false, true><<<dim3(k.heads * 8U, grid.y), 128, 0, s>>>(k, qkv, f, gate, b, S, conv_state, y, snap, conv_snap, flag, npos);
        ok(hipGetLastError(), "glm fast kda split producer");
        kda_r10_finalize_kernel<kSeq><<<dim3(k.heads, kSeq ? npos : grid.y), 128, 0, s>>>(k, qkv, gate, conv_state, y, npos, false);
        break;
    case 100:
        kda_fused_kernel<kSeq, 4, false, true, true><<<dim3(k.heads * 4U, grid.y), 128, 0, s>>>(k, qkv, f, gate, b, S, conv_state, y, snap, conv_snap, flag, npos);
        ok(hipGetLastError(), "glm fast kda split producer");
        kda_r10_finalize_kernel<kSeq><<<dim3(k.heads, kSeq ? npos : grid.y), 128, 0, s>>>(k, qkv, gate, conv_state, y, npos, true);
        break;
    case 116:
        kda_fused_kernel<kSeq, 4, true, true, true><<<dim3(k.heads * 4U, grid.y), 128, 0, s>>>(k, qkv, f, gate, b, S, conv_state, y, snap, conv_snap, flag, npos);
        ok(hipGetLastError(), "glm fast kda split producer");
        kda_r10_finalize_kernel<kSeq><<<dim3(k.heads, kSeq ? npos : grid.y), 128, 0, s>>>(k, qkv, gate, conv_state, y, npos, true);
        break;
    case 34: kda_fused_kernel<kSeq, 2, false, true, false><<<grid, 256, 0, s>>>(k, qkv, f, gate, b, S, conv_state, y, snap, conv_snap, flag, npos); break;
    case 18: kda_fused_kernel<kSeq, 2, true, false, false><<<grid, 256, 0, s>>>(k, qkv, f, gate, b, S, conv_state, y, snap, conv_snap, flag, npos); break;
    case 50: kda_fused_kernel<kSeq, 2, true, true, false><<<grid, 256, 0, s>>>(k, qkv, f, gate, b, S, conv_state, y, snap, conv_snap, flag, npos); break;
    case 4: kda_fused_kernel<kSeq, 4, false, false, false><<<grid, 512, 0, s>>>(k, qkv, f, gate, b, S, conv_state, y, snap, conv_snap, flag, npos); break;
    case 36: kda_fused_kernel<kSeq, 4, false, true, false><<<grid, 512, 0, s>>>(k, qkv, f, gate, b, S, conv_state, y, snap, conv_snap, flag, npos); break;
    case 20: kda_fused_kernel<kSeq, 4, true, false, false><<<grid, 512, 0, s>>>(k, qkv, f, gate, b, S, conv_state, y, snap, conv_snap, flag, npos); break;
    case 52: kda_fused_kernel<kSeq, 4, true, true, false><<<grid, 512, 0, s>>>(k, qkv, f, gate, b, S, conv_state, y, snap, conv_snap, flag, npos); break;
    case 8: kda_fused_kernel<kSeq, 8, false, false, false><<<grid, 1024, 0, s>>>(k, qkv, f, gate, b, S, conv_state, y, snap, conv_snap, flag, npos); break;
    case 40: kda_fused_kernel<kSeq, 8, false, true, false><<<grid, 1024, 0, s>>>(k, qkv, f, gate, b, S, conv_state, y, snap, conv_snap, flag, npos); break;
    case 24: kda_fused_kernel<kSeq, 8, true, false, false><<<grid, 1024, 0, s>>>(k, qkv, f, gate, b, S, conv_state, y, snap, conv_snap, flag, npos); break;
    case 56: kda_fused_kernel<kSeq, 8, true, true, false><<<grid, 1024, 0, s>>>(k, qkv, f, gate, b, S, conv_state, y, snap, conv_snap, flag, npos); break;
    default: kda_fused_kernel<kSeq, 2, false, false, false><<<grid, 256, 0, s>>>(k, qkv, f, gate, b, S, conv_state, y, snap, conv_snap, flag, npos); break;
    }
}
void kda_step(const dev::KdaDev& k, const float* qkv, const float* f, const float* gate, const float* b, float* S,
              float* conv_state, float* y, unsigned* flag, unsigned rows, hipStream_t s) {
    check_rows(rows);
    kda_launch<false>(dim3(k.heads, rows), s, k, qkv, f, gate, b, S, conv_state, y, nullptr, nullptr, flag, 1);
    ok(hipGetLastError(), "glm fast kda_step");
}
void kda_steps_seq(const dev::KdaDev& k, const float* qkv, const float* f, const float* gate, const float* b, float* S,
                   float* conv_state, float* y, float* snap, float* conv_snap, unsigned* flag, unsigned npos, hipStream_t s) {
    if (npos == 0 || npos > kMaxRows) throw std::invalid_argument("glm attn: npos must be 1..8");
    kda_launch<true>(dim3(k.heads, 1), s, k, qkv, f, gate, b, S, conv_state, y, snap, conv_snap, flag, npos);
    ok(hipGetLastError(), "glm fast kda_steps_seq");
}
void kda_state_copy_floor(float* S, unsigned heads, unsigned rows, hipStream_t s) {
    check_rows(rows);
    kda_state_copy_floor_kernel<<<dim3(heads, rows), 256, 0, s>>>(S, heads);
    ok(hipGetLastError(), "glm fast kda_state_copy_floor");
}
void set_kda_lanes(unsigned q) {
    if (!kda_lanes_valid(q)) throw std::invalid_argument("glm attn: KDA lanes per column must be 2, 4 or 8 (+16: interleaved, +32: 2-barrier trees, +64: column split), or 128: column quad (2c), 194: shared prep (2sp), 196: shared prep (4sp), 200: column-quad split shared prep (cqs)");
    g_kda_lanes.store(q);
}

std::atomic<unsigned> g_indexer_score_rows{0};
unsigned indexer_score_rows() {
    unsigned v = g_indexer_score_rows.load(std::memory_order_relaxed);
    if (v == 0) {
        const char* e = std::getenv("STRIX_GLM_IDX_RG");
        unsigned want = 6;  // auto (see indexer_score)
        if (e != nullptr) {
            const std::string es(e);
            if (es == "2") want = 2;
            if (es == "3") want = 3;
            if (es == "4") want = 4;
            if (es == "5") want = 5;
            if (es == "1") want = 1;
        }
        g_indexer_score_rows.compare_exchange_strong(v, want);
        v = g_indexer_score_rows.load(std::memory_order_relaxed);
    }
    return v;
}
void set_indexer_score_rows(unsigned rg) {
    if (rg < 1 || rg > 6) throw std::invalid_argument("set_indexer_score_rows: 1..5, or 6 = auto");
    g_indexer_score_rows.store(rg);
}

void dsa_store(const DsaRows& d, const float* latent, const float* key, const float* gate, const dev::IndexerDev& w,
               std::uint32_t* hist0, unsigned* flag, hipStream_t s, unsigned flag_stride) {
    check_rows(d.rows);
    dsa_store_kernel<<<d.rows, 128, 0, s>>>(d, latent, w, key, gate, hist0, flag, flag_stride);
    ok(hipGetLastError(), "glm fast dsa_store");
    dsa_pool_kernel<<<d.rows, 128, 0, s>>>(d, w, flag, flag_stride);
    ok(hipGetLastError(), "glm fast dsa_store");
}
void indexer_score(const DsaRows& d, const float* qI, const float* wraw, std::uint32_t* keys, unsigned key_stride,
                   std::uint32_t* hist0, unsigned p_bound, hipStream_t s, bool same_slot) {
    check_rows(d.rows);
    if (p_bound == 0) return;
    const unsigned tiles = (p_bound + kScoreThreads - 1) / kScoreThreads;
    unsigned rg = indexer_score_rows();
    // Auto (6, default; attention measurement node 1 00:54, gate 17/17 bitwise; same-slot rows = a spec verify window, which the
    // caller states): us rg1 / rk1 (5) / rk2 (3): 8K (2,048 pools) R2 16.8 / 15.5 / 24.0, R4 26.3 / 16.0 / 23.9,
    // R8 49.3 / 22.9 / 33.6; 32K (8,192 pools) R2 62.6 / 93.5 / 49.4, R4 91.7 / 168.9 / 81.3, R8 163.9 / 307.9 / 182.6;
    // R1 and separate slots: rg1. Between 2,048 and 8,192 pools unmeasured -> rg1. Same bits either way.
    if (rg == 6) rg = (same_slot && d.rows >= 2) ? (p_bound <= 2048U ? 5U : (p_bound >= 8192U && d.rows <= 4 ? 3U : 1U)) : 1U;
    const unsigned rk_tiles = (p_bound + 2 * kScoreThreads - 1) / (2 * kScoreThreads);
    if (rg == 3)
        idx_score_rk_kernel<2><<<dim3(rk_tiles, (d.rows + 1) / 2), kScoreThreads, 0, s>>>(d, qI, wraw, keys, key_stride, hist0);
    else if (rg == 5)
        idx_score_rk_kernel<1><<<dim3(rk_tiles, d.rows), kScoreThreads, 0, s>>>(d, qI, wraw, keys, key_stride, hist0);
    else if (rg == 1)
        idx_score_kernel<<<dim3(tiles, d.rows), kScoreThreads, 0, s>>>(d, qI, wraw, keys, key_stride, hist0);
    else  // RG=4 requests use RG=2 to stay within the LDS budget.
        idx_score_rg_kernel<2><<<dim3(tiles, (d.rows + 1) / 2), kScoreThreads, 0, s>>>(d, qI, wraw, keys, key_stride, hist0);
    ok(hipGetLastError(), "glm fast indexer_score");
}
void indexer_select(const DsaRows& d, const std::uint32_t* keys, unsigned key_stride, const std::uint32_t* hist0,
                    std::uint32_t* tokens, std::uint32_t* n, hipStream_t s) {
    check_rows(d.rows);
    idx_select_kernel<<<d.rows, 1024, 0, s>>>(d, keys, key_stride, hist0, tokens, n);
    ok(hipGetLastError(), "glm fast indexer_select");
}

void mla_absorb(const float* q, const std::uint16_t* kv_b, unsigned heads, float* qt, std::uint32_t* mkey, unsigned rows,
                hipStream_t s) {
    check_heads(heads);
    by_rows<AbsorbGo>(rows, q, kv_b, heads, qt, mkey, s);
    ok(hipGetLastError(), "glm fast mla_absorb");
}
template <class Policy, class... Context>
void mla_scores_policy(const DsaRows& d, const std::uint32_t* tokens, const std::uint32_t* n, const float* qt, unsigned heads,
                       float* sc, std::uint32_t* mkey, hipStream_t s, unsigned max_tokens, Context... context) {
    check_rows(d.rows);
    check_heads(heads);
    if (!scores_staged(max_tokens, d.rows))
        mla_scores_lane_kernel<Policy><<<dim3((kMaxTok + kScThreads - 1) / kScThreads, heads / kScHeads, d.rows), kScThreads, 0, s>>>(
            d, tokens, n, qt, heads, sc, mkey, context...);
    else
        mla_scores_staged_kernel<Policy><<<dim3((kMaxTok + kSsTok - 1) / kSsTok, heads / kScHeads, d.rows), 256, 0, s>>>(d, tokens, n, qt, heads,
                                                                                                            sc, mkey, context...);
    ok(hipGetLastError(), "glm fast mla_scores");
}
void mla_scores(const DsaRows& d, const std::uint32_t* tokens, const std::uint32_t* n, const float* qt, unsigned heads,
                float* sc, std::uint32_t* mkey, hipStream_t s, unsigned max_tokens) {
    mla_scores_policy<DsaAttentionPolicy<false>>(d, tokens, n, qt, heads, sc, mkey, s, max_tokens);
}
void mla_softmax(const std::uint32_t* n, const float* sc, const std::uint32_t* mkey, unsigned heads, float* p,
                 unsigned rows, hipStream_t s) {
    check_rows(rows);
    check_heads(heads);
    mla_softmax_kernel<><<<dim3(heads, rows), 256, 0, s>>>(n, sc, mkey, heads, p);
    ok(hipGetLastError(), "glm fast mla_softmax");
}
void set_mla_pv_variant(unsigned v) {
    if (v > 2) throw std::invalid_argument("set_mla_pv_variant: 0, 1 or 2");
    g_mla_pv_variant.store(static_cast<int>(v));
}
template <class Policy, class... Context>
void mla_pv_policy(const DsaRows& d, const std::uint32_t* tokens, const std::uint32_t* n, const float* p, unsigned heads,
                   float* ob, hipStream_t s, Context... context) {
    check_rows(d.rows);
    check_heads(heads);
    switch (mla_pv_variant()) {
    case 1:
        mla_pv_kernel_t<8, 8, Policy><<<dim3(kKvLora / 8, heads / 8, d.rows), 256, 0, s>>>(d, tokens, n, p, heads, ob, context...);
        break;
    case 2:
        mla_pv_kernel_t<16, 4, Policy><<<dim3(kKvLora / 4, heads / 16, d.rows), 256, 0, s>>>(d, tokens, n, p, heads, ob, context...);
        break;
    default:
        mla_pv_kernel_t<kPvHeads, kPvDims, Policy><<<dim3(kKvLora / kPvDims, heads / kPvHeads, d.rows), 256, 0, s>>>(d, tokens, n, p, heads, ob, context...);
        break;
    }
    ok(hipGetLastError(), "glm fast mla_pv");
}
void mla_pv(const DsaRows& d, const std::uint32_t* tokens, const std::uint32_t* n, const float* p, unsigned heads,
            float* ob, hipStream_t s) {
    mla_pv_policy<DsaAttentionPolicy<false>>(d, tokens, n, p, heads, ob, s);
}
void mla_partial(const DsaRows& d, const std::uint32_t* tokens, const std::uint32_t* n, const float* qt,
                 unsigned heads, const MlaWork& w, float* partial, unsigned rank, unsigned* flag, hipStream_t s,
                 unsigned max_tokens) {
    check_rows(d.rows);
    check_heads(heads);
    if (rank > 1) throw std::invalid_argument("mla_partial: rank must be 0 or 1");
    using Policy = DsaAttentionPolicy<true>;
    const DsaPartialContext context{rank, partial, flag};
    // Each query pass needs its own maxima, including the received query pass.
    ok(hipMemsetAsync(w.mkey, 0, static_cast<std::size_t>(d.rows) * heads * sizeof(std::uint32_t), s),
       "glm partial maxima clear");
    mla_scores_policy<Policy>(d, tokens, n, qt, heads, w.sc, w.mkey, s, max_tokens, context);
    mla_softmax_kernel<Policy><<<dim3(heads, d.rows), 256, 0, s>>>(n, w.sc, w.mkey, heads, w.p, context);
    ok(hipGetLastError(), "glm partial softmax");
    mla_pv_policy<Policy>(d, tokens, n, w.p, heads, partial, s, context);
}
void mla_out(const float* ob, const std::uint16_t* kv_b, unsigned heads, float* o, unsigned rows, hipStream_t s) {
    check_heads(heads);
    check_rows(rows);
    if (dense_lane_forced()) {
        by_rows<OutGo>(rows, ob, kv_b, heads, o, s);
        ok(hipGetLastError(), "glm fast mla_out");
        return;
    }
    // o[r][h][d] = chain over l of fmaf(ob[r][h][l], kv_b[h*512 + 256 + d][l]): one grouped dense job, group = head
    DenseJob j;
    j.w = kv_b + static_cast<std::size_t>(kMlaDim) * kKvLora;
    j.x = ob;
    j.y = o;
    j.n = kMlaDim;
    j.in = kKvLora;
    j.x_stride = heads * kKvLora;
    j.y_stride = heads * kMlaDim;
    j.groups = heads;
    j.w_gstride = static_cast<std::size_t>(2) * kMlaDim * kKvLora;
    j.x_gstride = kKvLora;
    j.y_gstride = kMlaDim;
    dense_f16(&j, 1, rows, s);
}
void mla_attention(const DsaRows& d, const std::uint32_t* tokens, const std::uint32_t* n, const float* q,
                   const std::uint16_t* kv_b, unsigned heads, const MlaWork& w, float* o, hipStream_t s, unsigned max_tokens) {
    check_rows(d.rows);
    check_heads(heads);
    mla_absorb(q, kv_b, heads, w.qt, w.mkey, d.rows, s);
    mla_scores(d, tokens, n, w.qt, heads, w.sc, w.mkey, s, max_tokens);
    mla_softmax(n, w.sc, w.mkey, heads, w.p, d.rows, s);
    mla_pv(d, tokens, n, w.p, heads, w.ob, s);
    mla_out(w.ob, kv_b, heads, o, d.rows, s);
}

}  // namespace strix::glm::fast
