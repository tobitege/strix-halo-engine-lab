// HIP runner of the shared vision encoder, arithmetic identity "vision-f16-attn2-v1" (default: two-term attention)
// or "vision-f16-v1" (EncoderOptions::attention_terms 1, one-term attention), class FAST.
// The device projector (default on) adds its identity suffix: see Encoder::effective_arithmetic().
//
// Every linear (patch embedding over flattened patches, qkv, out, MLP, projector) is the shared dense FP16 tall GEMM
// (core/f16_dense_tall.hpp): weights as FP16 panels with a power-of-two tensor scale (undone exactly in the FP32
// epilogue), activations rounded to FP16 by the core, FP32 accumulation. Attention is the shared dense flash core
// (attn_core/dense_flash_prefill.hpp) under a non-causal policy: mask::bidir_* gives every row a window of n logical
// keys and the policy backs logical key p with physical key p % n, so the causal core computes full bidirectional
// attention unchanged. The attention operands Q, K, V, P are hi + lo FP16 terms by default (the core's kTerms 2:
// hi.hi + hi.lo + lo.hi per product, ~3 x 2^-22 |ab| instead of ~2^-10 |ab| for normal f16 remainders, 2^-25
// absolute below; bounds in attn_core/two_term_model.hpp; 3x the attention WMMA); attention_terms
// 1 is the one-term form. Norms, activations, rotary rotation, residual and the softmax stay FP32.
// Buffers: encoder_layout.hpp (the same numbers plan() reports). Input pads of the GEMM (in % 128) are zero columns.
#include "strix/vision/encoder.hpp"

#include "encoder_layout.hpp"

#include <hip/hip_runtime.h>

#include "../attn_core/dense_flash_prefill.hpp"
#include "../attn_core/masks.hpp"
#include "../core/f16_dense_tall.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace strix::vision {
namespace {

namespace mask = attn_core::mask;
namespace tall = core::f16_dense_tall;

// Every row of a tile of n keys sees logical keys r .. n - 1 + r (n keys, all backed) under the causal window rule.
constexpr bool bidir_covers(unsigned n) {
    for (unsigned r = 0; r < n; ++r) {
        const unsigned p = mask::bidir_position(n) + r, w = mask::bidir_window(n);
        const unsigned lo = w && p >= w ? p - w + 1U : 0U;
        if (lo != r || p - lo + 1U != n || p >= mask::bidir_capacity(n)) return false;
        if (mask::bidir_key(lo, n) != r % n || mask::bidir_key(p, n) != (r + n - 1U) % n) return false;
    }
    return mask::bidir_capacity(n) % 16U == 0U && mask::bidir_capacity(n) == layout::key_slots(n);
}
static_assert(bidir_covers(1) && bidir_covers(2) && bidir_covers(15) && bidir_covers(16) && bidir_covers(17) && bidir_covers(169) &&
              bidir_covers(1521));

void ck(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string("vision encoder: ") + what + ": " + hipGetErrorString(e));
}

struct Dev {
    void* p = nullptr;
    std::size_t n = 0;
    Dev() = default;
    explicit Dev(std::size_t bytes) {
        if (bytes) {
            ck(hipMalloc(&p, bytes), "hipMalloc");
            n = bytes;
        }
    }
    Dev(Dev&& o) noexcept : p(std::exchange(o.p, nullptr)), n(std::exchange(o.n, 0U)) {}
    Dev& operator=(Dev&& o) noexcept {
        if (this != &o) {
            reset();
            p = std::exchange(o.p, nullptr);
            n = std::exchange(o.n, 0U);
        }
        return *this;
    }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    ~Dev() { reset(); }
    void reset() {
        if (p) (void)hipFree(p);
        p = nullptr;
        n = 0;
    }
    template <class T> T* as() const { return static_cast<T*>(p); }
};

inline constexpr unsigned kNonfinite = 1U, kOperandRange = 8U;  // failure bits of this file (the cores use 1, 2, 4)

// ---- kernels ---------------------------------------------------------------------------------------------------

// Store modes of the GEMM (GemmArgs::flags). Without kAffine the sums go to y; with it the store is the epilogue of
// the linear and the separate affine kernel is not launched.
inline constexpr unsigned kSum = 1U;        // sum = y[r][c] + sum first (y holds the high-part pass of a split operand)
inline constexpr unsigned kAffine = 2U;     // dst[r][c] = sum * scale + bias[c] for c < n
inline constexpr unsigned kResidual = 4U;   // ... + dst[r][c]
inline constexpr unsigned kPadRaw = 8U;     // dst is y: padding columns c >= n keep the raw sum, as the two-kernel form left them
inline constexpr unsigned kSkipIdle = 16U;  // row waves without a live row skip their WMMAs

struct GemmArgs {
    const float* x;      // [rows][in]
    const _Float16* w;   // [tiles * 16][in]
    float* y;            // [rows][stride]
    unsigned in, rows, stride, tiles, tile0;
    float* dst;          // kAffine: [rows][dstride]
    const float* bias;   // kAffine: [tiles * 16]
    unsigned* failure;   // kAffine: kNonfinite is reported here
    unsigned dstride, n, flags;
    float scale;
};
struct GemmCols {
    const GemmArgs& a;
    __device__ __forceinline__ tall::ColumnTile tile(unsigned i) const {
        const unsigned t = a.tile0 + i;
        const bool live = t < a.tiles;
        const unsigned u = live ? t : a.tiles - 1U;
        return {a.w + static_cast<std::size_t>(u) * 16U * a.in, a.y + u * 16U, a.stride, live};
    }
};
// Same arithmetic, in the same order, as sum_kernel followed by affine_kernel: (high + low) * scale + bias, + residual.
template <unsigned M> struct GemmModeStore {  // M = the store-mode bits of GemmArgs::flags, fixed at compile time
    const GemmArgs& a;
    unsigned bad = 0U;
    struct Tile {
        float* y;
        float* dst;
        float bias;
        unsigned c;
        bool inside;
    };
    __device__ __forceinline__ Tile open(unsigned i, const tall::ColumnTile& ct, unsigned c) const {
        const unsigned col = (a.tile0 + i) * 16U + c;
        Tile t{tall::detail::uniform(ct.y), nullptr, 0.0F, c, false};
        if (M & kAffine) {
            t.dst = a.dst + col;
            t.bias = a.bias[col];
            t.inside = col < a.n;
        }
        return t;
    }
    __device__ __forceinline__ void put(const Tile& t, unsigned r, float v) {
        float* const raw = t.y + static_cast<std::size_t>(r) * a.stride + t.c;
        if (M & kSum) v = *raw + v;
        if (!(M & kAffine)) {
            *raw = v;
        } else if (t.inside) {
            float* const d = t.dst + static_cast<std::size_t>(r) * a.dstride;
            v = v * a.scale + t.bias;
            if (M & kResidual) v += *d;
            if (!__builtin_isfinite(v)) bad = 1U;
            *d = v;
        } else if (M & kPadRaw) {
            *raw = v;
        }
    }
    __device__ __forceinline__ void close() const {
        if (bad) atomicOr(a.failure, kNonfinite);
    }
};
struct GemmStore {
    const GemmArgs& a;
    template <unsigned M, class F> __device__ __forceinline__ void mode(F& f) const {
        GemmModeStore<M> s{a};
        f(s);
    }
    template <class F> __device__ __forceinline__ void run(F&& f) const {
        switch (tall::detail::uni(a.flags) & (kSum | kAffine | kResidual | kPadRaw)) {
        case 0U: mode<0U>(f); break;
        case kSum: mode<kSum>(f); break;
        case kAffine: mode<kAffine>(f); break;
        case kAffine | kSum: mode<kAffine | kSum>(f); break;
        case kAffine | kResidual: mode<kAffine | kResidual>(f); break;
        case kAffine | kResidual | kSum: mode<kAffine | kResidual | kSum>(f); break;
        case kAffine | kPadRaw: mode<kAffine | kPadRaw>(f); break;
        case kAffine | kPadRaw | kSum: mode<kAffine | kPadRaw | kSum>(f); break;
        case kAffine | kPadRaw | kResidual: mode<kAffine | kPadRaw | kResidual>(f); break;
        default: mode<kAffine | kPadRaw | kResidual | kSum>(f); break;
        }
    }
};
template <unsigned TC> __global__ __launch_bounds__(256) void gemm_kernel(GemmArgs a) {
    __shared__ tall::Storage<TC> lds;
    tall::core<TC>(a.x, a.in, a.rows, GemmCols{a}, lds, GemmStore{a}, (a.flags & kSkipIdle) != 0U);
}

// dst[r][c] = (add ? dst[r][c] : 0) + src[r][c] * scale + bias[c]
__global__ __launch_bounds__(256) void affine_kernel(float* dst, unsigned dstride, const float* src, unsigned sstride, unsigned n,
                                                     unsigned rows, const float* bias, float scale, unsigned add, unsigned* failure) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * 256U + threadIdx.x;
    if (i >= static_cast<std::size_t>(rows) * n) return;
    const std::size_t r = i / n;
    const unsigned c = static_cast<unsigned>(i % n);
    float v = src[r * sstride + c] * scale + bias[c];
    if (add) v += dst[r * dstride + c];
    if (!__builtin_isfinite(v)) atomicOr(failure, kNonfinite);
    dst[r * dstride + c] = v;
}

// dst[i] += src[i] over two buffers of the same pitch.
__global__ __launch_bounds__(256) void sum_kernel(float* dst, const float* src, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * 256U + threadIdx.x;
    if (i < count) dst[i] += src[i];
}

// x[r][c] += pos[r][c], x rows of xstride floats, pos rows of n.
__global__ __launch_bounds__(256) void add_rows_kernel(float* x, unsigned xstride, const float* pos, unsigned n, unsigned rows) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * 256U + threadIdx.x;
    if (i >= static_cast<std::size_t>(rows) * n) return;
    x[i / n * xstride + i % n] += pos[i];
}

// lo[i] = x[i] - fp32(fp16(x[i])): the part of the operand the GEMM's FP16 rounding drops (0 when it overflows).
__global__ __launch_bounds__(256) void split_lo_kernel(const float* x, float* lo, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * 256U + threadIdx.x;
    if (i >= count) return;
    const float v = x[i], hi = static_cast<float>(static_cast<_Float16>(v));
    lo[i] = __builtin_isfinite(hi) ? v - hi : 0.0F;
}

// One work-group per row. layer != 0: LayerNorm (mean removed); else RMSNorm. y = (x - mean) * rsqrt(var + eps) * w + b.
__global__ __launch_bounds__(256) void norm_kernel(const float* x, unsigned xstride, float* y, unsigned ystride, unsigned n,
                                                   const float* w, const float* b, float eps, unsigned layer) {
    __shared__ float part[256];
    const unsigned tid = threadIdx.x;
    const float* xr = x + static_cast<std::size_t>(blockIdx.x) * xstride;
    float mean = 0.0F;
    if (layer) {
        float s = 0.0F;
        for (unsigned i = tid; i < n; i += 256U) s += xr[i];
        part[tid] = s;
        __syncthreads();
        for (unsigned step = 128U; step; step >>= 1U) {
            if (tid < step) part[tid] += part[tid + step];
            __syncthreads();
        }
        mean = part[0] / static_cast<float>(n);
        __syncthreads();
    }
    float s = 0.0F;
    for (unsigned i = tid; i < n; i += 256U) {
        const float d = xr[i] - mean;
        s += d * d;
    }
    part[tid] = s;
    __syncthreads();
    for (unsigned step = 128U; step; step >>= 1U) {
        if (tid < step) part[tid] += part[tid + step];
        __syncthreads();
    }
    const float inv = 1.0F / sqrtf(part[0] / static_cast<float>(n) + eps);
    float* yr = y + static_cast<std::size_t>(blockIdx.x) * ystride;
    for (unsigned i = tid; i < n; i += 256U) yr[i] = (xr[i] - mean) * inv * w[i] + b[i];
}

struct ActArgs {
    const float* mid;   // [rows][mstride] raw GEMM sums
    float* out;         // [rows][ostride]
    const float* bias;  // [m or 2 m]
    unsigned* failure;
    float scale, limit;
    unsigned mstride, ostride, m, rows, mode, gate_first;  // mode 0 gelu, 1 gelu-tanh, 2 swiglu
};
__global__ __launch_bounds__(256) void act_kernel(ActArgs a) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * 256U + threadIdx.x;
    if (i >= static_cast<std::size_t>(a.rows) * a.m) return;
    const std::size_t r = i / a.m;
    const unsigned j = static_cast<unsigned>(i % a.m);
    const float* m = a.mid + r * a.mstride;
    float y;
    if (a.mode == 2U) {
        const unsigned gi = a.gate_first ? j : a.m + j, ui = a.gate_first ? a.m + j : j;
        float g = m[gi] * a.scale + a.bias[gi], u = m[ui] * a.scale + a.bias[ui];
        if (a.limit > 0.0F) {
            g = fminf(g, a.limit);
            u = fminf(fmaxf(u, -a.limit), a.limit);
        }
        y = g / (1.0F + expf(-g)) * u;
    } else {
        const float v = m[j] * a.scale + a.bias[j];
        y = a.mode == 0U ? 0.5F * v * (1.0F + erff(v * 0.70710678118654752F))
                         : 0.5F * v * (1.0F + tanhf(0.79788456080286536F * (v + 0.044715F * v * v * v)));
    }
    if (!__builtin_isfinite(y)) atomicOr(a.failure, kNonfinite);
    a.out[r * a.ostride + j] = y;
}

// qkv sums -> rotated FP32 queries [row][head][dp], FP16 keys [slot][heads * dp] and FP16 V^T [heads * dp][slots],
// keys and values repeated cyclically over the tile's logical positions. One work-group per logical position.
struct PackArgs {
    const float* qkv;   // [rows][qstride]: q | k | v, each [heads][d], scale and bias applied
    const float* cosv;  // [rows][d / 2] or null
    const float* sinv;
    float* q;
    _Float16* k;
    _Float16* vt;
    _Float16* klo;   // two-term attention: lo terms of k / vt (tile::to_operand2), null for one term
    _Float16* vtlo;
    unsigned* failure;
    unsigned qstride, row0, n, slot0, heads, d, dp, w, vstride, half;
};
__global__ __launch_bounds__(256) void pack_kernel(PackArgs a) {
    const unsigned slot = blockIdx.x, src = a.row0 + mask::bidir_key(slot, a.n), hw = a.heads * a.dp, hd = a.d / 2U;
    const float* row = a.qkv + static_cast<std::size_t>(src) * a.qstride;
    bool bad = false;
    for (unsigned e = threadIdx.x; e < hw; e += 256U) {
        const unsigned h = e / a.dp, d = e % a.dp;
        float qv = 0.0F, kv = 0.0F, vv = 0.0F;
        if (d < a.d) {
            const unsigned at = h * a.d + d;
            qv = row[at];
            kv = row[a.w + at];
            vv = row[2U * a.w + at];
            if (a.cosv) {
                const bool first = a.half ? d < hd : (d & 1U) == 0U;
                const unsigned pair = a.half ? (first ? d : d - hd) : d >> 1U;
                const unsigned other = h * a.d + (a.half ? (first ? d + hd : d - hd) : d ^ 1U);
                const float c = a.cosv[static_cast<std::size_t>(src) * hd + pair], s = a.sinv[static_cast<std::size_t>(src) * hd + pair];
                const float qo = row[other], ko = row[a.w + other];
                qv = first ? qv * c - qo * s : qv * c + qo * s;
                kv = first ? kv * c - ko * s : kv * c + ko * s;
            }
        }
        if (slot < a.n) a.q[(static_cast<std::size_t>(src) * a.heads + h) * a.dp + d] = qv;
        const _Float16 kh = static_cast<_Float16>(kv), vh = static_cast<_Float16>(vv);
        bad |= !__builtin_isfinite(static_cast<float>(kh)) || !__builtin_isfinite(static_cast<float>(vh));
        a.k[static_cast<std::size_t>(a.slot0 + slot) * hw + e] = kh;
        a.vt[static_cast<std::size_t>(e) * a.vstride + a.slot0 + slot] = vh;
        if (a.klo) {
            using attn_core::tile::Operand;
            a.klo[static_cast<std::size_t>(a.slot0 + slot) * hw + e] =
                __builtin_bit_cast(_Float16, attn_core::tile::to_operand2<Operand::f16>(kv).lo);
            a.vtlo[static_cast<std::size_t>(e) * a.vstride + a.slot0 + slot] =
                __builtin_bit_cast(_Float16, attn_core::tile::to_operand2<Operand::f16>(vv).lo);
        }
    }
    if (bad) atomicOr(a.failure, kOperandRange);
}

// Attention output [row][head][dp] -> GEMM input [row][heads * d] (pitch ostride).
__global__ __launch_bounds__(256) void compact_kernel(const float* ao, float* out, unsigned ostride, unsigned heads, unsigned d,
                                                      unsigned dp, unsigned rows) {
    const unsigned w = heads * d;
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * 256U + threadIdx.x;
    if (i >= static_cast<std::size_t>(rows) * w) return;
    const std::size_t r = i / w;
    const unsigned c = static_cast<unsigned>(i % w);
    out[r * ostride + c] = ao[(r * heads + c / d) * dp + c % d];
}

__global__ __launch_bounds__(256) void splice_kernel(float* dst, std::size_t dstride, const std::uint16_t* src, unsigned width,
                                                     const std::uint32_t* dst_row, const std::uint32_t* src_row) {
    const std::uint16_t* s = src + static_cast<std::size_t>(src_row[blockIdx.x]) * width;
    float* d = dst + static_cast<std::size_t>(dst_row[blockIdx.x]) * dstride;
    for (unsigned c = threadIdx.x; c < width; c += 256U) d[c] = __builtin_bit_cast(float, static_cast<std::uint32_t>(s[c]) << 16U);
}

// ---- non-causal policy on the dense flash core -------------------------------------------------------------------

struct AttnArgs {
    const float* q;      // [row][head][dp]
    float* out;          // [row][head][dp]
    const _Float16* k;   // [slot][heads * dp]
    const _Float16* vt;  // [heads * dp][vstride]
    const _Float16* klo;   // two-term: lo terms of k, vt (same layouts)
    const _Float16* vtlo;
    unsigned* failure;
    unsigned n, row0, slot0, heads, vstride;
};
inline constexpr unsigned kPositions = 64;  // query rows per work-group: 4 waves share every staged K/V tile
// Terms: FP16 terms per attention operand (EncoderOptions::attention_terms): 2 = hi + lo (default), 1 = one term.
template <unsigned D, unsigned DP, unsigned Terms>
struct BidirPolicy {
    using Args = AttnArgs;
    static constexpr unsigned kTerms = Terms;
    using Shape = attn_core::dense::Shape<DP, 1, 1, kPositions>;
    static constexpr float kInvSqrt = D == 64U ? 0.125F : D == 72U ? 0.11785113019775792F : 0.11180339887498948F;
    static constexpr float kScaleLog2 = kInvSqrt * 1.44269504088896341F;
    static constexpr attn_core::tile::Operand kOp = attn_core::tile::Operand::f16;
    __device__ static bool valid(const Args& a) { return a.n != 0U && a.n <= 0x3FFFFFFFU; }
    __host__ __device__ static unsigned rows(const Args& a) { return a.n; }
    __device__ static unsigned position(const Args& a) { return mask::bidir_position(a.n); }
    __device__ static unsigned window(const Args& a) { return mask::bidir_window(a.n); }
    __device__ static const float* query(const Args& a, unsigned row, unsigned h) {
        return a.q + (static_cast<std::size_t>(a.row0 + row) * a.heads + h) * DP;
    }
    __device__ static float* out(const Args& a, unsigned row, unsigned h) {
        return a.out + (static_cast<std::size_t>(a.row0 + row) * a.heads + h) * DP;
    }
    __device__ static const std::uint16_t* key(const Args& a, unsigned p, unsigned kvh) {
        return reinterpret_cast<const std::uint16_t*>(a.k + static_cast<std::size_t>(a.slot0 + p) * (a.heads * DP) + kvh * DP);
    }
    __device__ static const std::uint16_t* value_t(const Args& a, unsigned kvh, unsigned d) {
        return reinterpret_cast<const std::uint16_t*>(a.vt + static_cast<std::size_t>(kvh * DP + d) * a.vstride + a.slot0);
    }
    __device__ static const std::uint16_t* key_lo(const Args& a, unsigned p, unsigned kvh) {
        return reinterpret_cast<const std::uint16_t*>(a.klo + static_cast<std::size_t>(a.slot0 + p) * (a.heads * DP) + kvh * DP);
    }
    __device__ static const std::uint16_t* value_t_lo(const Args& a, unsigned kvh, unsigned d) {
        return reinterpret_cast<const std::uint16_t*>(a.vtlo + static_cast<std::size_t>(kvh * DP + d) * a.vstride + a.slot0);
    }
    __device__ static unsigned* failure(const Args& a) { return a.failure; }
};

unsigned blocks_of(std::size_t count) { return static_cast<unsigned>((count + 255U) / 256U); }

// ---- host side ---------------------------------------------------------------------------------------------------

struct Panel {
    Dev w, b;  // FP16 [n16][inp] scaled by 2^k; FP32 bias [n16] (zeros when the linear has none)
    unsigned in = 0, inp = 0, n = 0, n16 = 0;
    float inv = 1.0F;  // 2^-k
    std::size_t bytes() const { return w.n + b.n; }
};

using LoadClock = std::chrono::steady_clock;
double ms_from(LoadClock::time_point& t) {  // ms since t, and t moves to now
    const auto now = LoadClock::now();
    const double ms = std::chrono::duration<double, std::milli>(now - t).count();
    t = now;
    return ms;
}

Panel make_panel(const Tensor& w, const Tensor& bias, EncoderLoadStats* st = nullptr) {
    EncoderLoadStats none;
    EncoderLoadStats& ls = st ? *st : none;
    auto t = LoadClock::now();
    if (!w || !w.rows || !w.cols || w.rows > 0x00FFFFFFU || w.cols > 0x00FFFFFFU) throw std::invalid_argument("vision encoder: linear shape");
    if (bias && bias.size() != w.rows) throw std::invalid_argument("vision encoder: linear bias");
    Panel p;
    p.in = static_cast<unsigned>(w.cols);
    p.n = static_cast<unsigned>(w.rows);
    p.inp = layout::round_up(p.in, layout::kGemmIn);
    p.n16 = layout::round_up(p.n, layout::kGemmOut);
    // Per-row magnitude in parallel (max is exact, so the split cannot change it), then the scale.
    std::vector<float> row_top(p.n, 0.0F);
    std::atomic<bool> nonfinite{false};
    layout::parallel_ranges(p.n, p.in, [&](std::size_t o0, std::size_t o1) {
        for (std::size_t o = o0; o < o1; ++o) {
            float t = 0.0F;
            for (std::size_t c = 0; c < p.in; ++c) {
                const float v = w.at(o * p.in + c);
                if (!std::isfinite(v)) {
                    nonfinite = true;
                    return;
                }
                t = std::max(t, std::fabs(v));
            }
            row_top[o] = t;
        }
    });
    if (nonfinite) throw std::invalid_argument("vision encoder: nonfinite weight");
    ls.scan_ms += ms_from(t);
    float top = 0.0F;
    for (const float t : row_top) top = std::max(top, t);
    // Largest magnitude lands in [2^13, 2^14): far from the FP16 overflow, small entries far from the subnormals.
    const int k = top > 0.0F ? std::clamp(13 - std::ilogb(top), -60, 60) : 0;
    p.inv = std::ldexp(1.0F, -k);
    // v * 2^k == ldexp(v, k) bit for bit: 2^k (|k| <= 60) is a normal float and both round the exact product once.
    const float scale = std::ldexp(1.0F, k);
    std::vector<_Float16> h(std::size_t{p.n16} * p.inp, static_cast<_Float16>(0.0F));
    ls.host_alloc_ms += ms_from(t);
    layout::parallel_ranges(p.n, p.in, [&](std::size_t o0, std::size_t o1) {
        for (std::size_t o = o0; o < o1; ++o)
            for (std::size_t c = 0; c < p.in; ++c) h[o * p.inp + c] = static_cast<_Float16>(w.at(o * p.in + c) * scale);
    });
    std::vector<float> b(p.n16, 0.0F);
    if (bias)
        for (std::size_t o = 0; o < p.n; ++o) b[o] = bias.at(o);
    ls.convert_ms += ms_from(t);
    p.w = Dev(layout::aligned(h.size() * 2U));
    p.b = Dev(layout::aligned(b.size() * 4U));
    ls.device_alloc_ms += ms_from(t);
    ck(hipMemcpy(p.w.p, h.data(), h.size() * 2U, hipMemcpyHostToDevice), "panel upload");
    ck(hipMemcpy(p.b.p, b.data(), b.size() * 4U, hipMemcpyHostToDevice), "bias upload");
    ls.upload_ms += ms_from(t);
    return p;
}

Dev make_vector(const Tensor& t, unsigned n) {  // FP32 [n], zeros when the tensor is absent
    std::vector<float> v(n, 0.0F);
    if (t)
        for (unsigned i = 0; i < n; ++i) v[i] = t.at(i);
    Dev d(layout::aligned(std::size_t{n} * 4U));
    ck(hipMemcpy(d.p, v.data(), std::size_t{n} * 4U, hipMemcpyHostToDevice), "vector upload");
    return d;
}

// Page-locked host buffer: the source of one asynchronous upload per encode (grows only).
struct Pinned {
    void* p = nullptr;
    std::size_t n = 0;
    Pinned() = default;
    Pinned(const Pinned&) = delete;
    Pinned& operator=(const Pinned&) = delete;
    ~Pinned() { reset(); }
    void reset() {
        if (p) (void)hipHostFree(p);
        p = nullptr;
        n = 0;
    }
    void grow(std::size_t bytes) {
        if (bytes <= n) return;
        reset();
        ck(hipHostMalloc(&p, bytes, hipHostMallocDefault), "hipHostMalloc");
        n = bytes;
    }
};
// Device scratch that grows only: a call with a shape seen before allocates nothing.
void grow(Dev& d, std::size_t bytes) {
    if (bytes > d.n) {
        d.reset();
        d = Dev(layout::aligned(bytes));
    }
}

// Projector steps on the device. x[t][inp] = unshuffled hidden rows (zeros past the grid edge and in the pads).
__global__ __launch_bounds__(256) void unshuffle_kernel(const float* hid, unsigned pitch, float* x, unsigned inp, unsigned in,
                                                        unsigned ratio, unsigned row_major, unsigned grid_h, unsigned grid_w,
                                                        unsigned mw, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * 256U + threadIdx.x;
    if (i >= count) return;
    const unsigned t = static_cast<unsigned>(i / inp), j = static_cast<unsigned>(i % inp);
    float v = 0.0F;
    if (j < in) {
        const unsigned rr = ratio * ratio, c = j / rr, slot = j % rr;
        const unsigned dy = row_major ? slot / ratio : slot % ratio, dx = row_major ? slot % ratio : slot / ratio;
        const unsigned gy = (t / mw) * ratio + dy, gx = (t % mw) * ratio + dx;
        if (gy < grid_h && gx < grid_w) v = hid[(static_cast<std::size_t>(gy) * grid_w + gx) * pitch + c];
    }
    x[i] = v;
}
// The host activate() formula (src/vision/projector.cpp) in FP64, same operation order; this file is built without
// contraction, so no step is fused.
__device__ float activate_f64(unsigned act, float v) {
    const double x = static_cast<double>(v);
    if (act == 0U) return static_cast<float>(0.5 * x * (1.0 + erf(x * 0.70710678118654752440)));
    if (act == 1U) return static_cast<float>(0.5 * x * (1.0 + tanh(0.79788456080286535588 * (x + 0.044715 * x * x * x))));
    return static_cast<float>(x / (1.0 + exp(-x)));
}
// x[r][xstride] = act(y[r][ystride] columns < n), zero pads.
__global__ __launch_bounds__(256) void project_act_kernel(const float* y, unsigned ystride, float* x, unsigned xstride, unsigned n,
                                                          unsigned act, std::size_t count) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * 256U + threadIdx.x;
    if (i >= count) return;
    const std::size_t r = i / xstride;
    const unsigned c = static_cast<unsigned>(i % xstride);
    x[i] = c < n ? activate_f64(act, y[r * ystride + c]) : 0.0F;
}

struct Block {
    Panel qkv, out, mlp1, mlp2;
    Dev n1w, n1b, n2w, n2b;
};

}  // namespace

struct Encoder::Impl {
    EncoderConfig cfg;
    EncoderOptions opt;
    hipStream_t stream = nullptr;
    bool loaded = false;
    EncoderLoadStats load_stats;
    Panel patch;
    std::vector<Block> blocks;
    Dev fnw, fnb, failure, ws;
    std::unordered_map<const void*, Panel> projector;
    // Workspace zeroing: the pads are never written, so a buffer already cleared for the same geometry (rows + tile
    // ends) still holds zero pads; any other geometry moves the sub-buffers and is cleared again.
    bool ws_clear = false;
    unsigned ws_rows = 0;
    std::vector<unsigned> ws_tiles;
    Pinned stage;                                  // inputs of one encode (patches, position addend, cos, sin)
    Dev px, py, pact, py2, plo, pylo, phid;        // projector / linear() scratch, grow only
    std::uint64_t epoch = 0;                       // encodes run; EncoderOutput::device_epoch of a resident result
    const std::shared_ptr<const void> owner = std::make_shared<const char>(char{0});
    const float* hid = nullptr;                    // resident hidden rows of encode `epoch` (null: none)
    unsigned hid_pitch = 0;

    const Panel& panel_for(const Tensor& w, const Tensor& bias) {
        auto it = projector.find(w.data);
        if (it == projector.end()) it = projector.emplace(w.data, make_panel(w, bias)).first;
        const Panel& p = it->second;
        if (p.in != w.cols || p.n != w.rows) throw std::invalid_argument("vision encoder: linear panel reuse with another shape");
        return p;
    }
    void drop_workspace() {
        ws.reset();
        ws_clear = false;
        hid = nullptr;
    }

    // y[rows][p.n16] = raw sums of x[rows][p.inp] (scale and bias are the caller's epilogue). With `lo`/`ylo`
    // (split activations) the FP16 rounding residue of x goes through a second GEMM and is added.
    static bool fused() {
        static const bool on = envcfg::flag("STRIX_VISION_GEMM_EPILOGUE", true);
        return on;
    }
    void gemm(const Panel& p, const float* x, unsigned rows, float* y, float* lo, float* ylo) const {
        const unsigned cap = std::max(128U, static_cast<unsigned>(0xFFFFFFFFULL / (4ULL * p.inp)) / 128U * 128U);
        for (unsigned r0 = 0; r0 < rows; r0 += cap) {
            const unsigned nr = std::min(cap, rows - r0);
            const float* xs = x + static_cast<std::size_t>(r0) * p.inp;
            float* ys = y + static_cast<std::size_t>(r0) * p.n16;
            launch(p, xs, nr, ys);
            if (lo) {
                const std::size_t count = static_cast<std::size_t>(nr) * p.inp, sums = static_cast<std::size_t>(nr) * p.n16;
                hipLaunchKernelGGL(split_lo_kernel, dim3(blocks_of(count)), dim3(256), 0, stream, xs, lo, count);
                if (fused()) {  // the low pass adds the high sums in its store
                    launch(p, lo, nr, ys, kSum);
                    continue;
                }
                launch(p, lo, nr, ylo);
                hipLaunchKernelGGL(sum_kernel, dim3(blocks_of(sums)), dim3(256), 0, stream, ys, static_cast<const float*>(ylo), sums);
            }
        }
    }
    void launch(const Panel& p, const float* x, unsigned rows, float* y, unsigned flags = 0U, float* dst = nullptr,
                unsigned dstride = 0U) const {
        const bool tail = tall::tail_policy();
        GemmArgs a{x, p.w.as<const _Float16>(), y, p.inp, rows, p.n16, p.n16 / 16U, 0U, dst, p.b.as<const float>(), failure.as<unsigned>(),
                   dstride, p.n, flags | (tail ? kSkipIdle : 0U), p.inv};
        const unsigned gx = (rows + tall::kRows - 1U) / tall::kRows;
        const tall::Plan pl = tall::plan(a.tiles, rows, tail);
        if (pl.full) hipLaunchKernelGGL(gemm_kernel<16>, dim3(gx, pl.full), dim3(256), 0, stream, a);
        a.tile0 = pl.full * 16U;
        const dim3 grid(gx, pl.gy);
        switch (pl.tc) {
        case 4: hipLaunchKernelGGL(gemm_kernel<4>, grid, dim3(256), 0, stream, a); break;
        case 2: hipLaunchKernelGGL(gemm_kernel<2>, grid, dim3(256), 0, stream, a); break;
        case 1: hipLaunchKernelGGL(gemm_kernel<1>, grid, dim3(256), 0, stream, a); break;
        default: break;
        }
    }
    // dst[rows][dstride] (columns < p.n) = (add ? dst : 0) + sums * scale + bias: the whole linear. `raw` is the
    // [rows][p.n16] scratch of the two-kernel form (it may be dst when dstride == p.n16). Fused form (default): the GEMM
    // stores the final value; with split activations the low pass reads the high sums from `raw` and finishes there.
    // STRIX_VISION_GEMM_EPILOGUE=0 restores GEMM + affine kernel (bit-identical results).
    void linear(const Panel& p, const float* x, unsigned rows, float* raw, float* dst, unsigned dstride, bool add, float* lo,
                float* ylo) const {
        if (!fused() || (dst == raw && dstride != p.n16)) {
            gemm(p, x, rows, raw, lo, ylo);
            affine(dst, dstride, raw, p, rows, add);
            return;
        }
        const unsigned mode = kAffine | (add ? kResidual : 0U) | (dst == raw ? kPadRaw : 0U);
        const unsigned cap = std::max(128U, static_cast<unsigned>(0xFFFFFFFFULL / (4ULL * p.inp)) / 128U * 128U);
        for (unsigned r0 = 0; r0 < rows; r0 += cap) {
            const unsigned nr = std::min(cap, rows - r0);
            const float* xs = x + static_cast<std::size_t>(r0) * p.inp;
            float* ys = raw + static_cast<std::size_t>(r0) * p.n16;
            float* ds = dst + static_cast<std::size_t>(r0) * dstride;
            if (!lo) {
                launch(p, xs, nr, ys, mode, ds, dstride);
                continue;
            }
            const std::size_t count = static_cast<std::size_t>(nr) * p.inp;
            launch(p, xs, nr, ys);
            hipLaunchKernelGGL(split_lo_kernel, dim3(blocks_of(count)), dim3(256), 0, stream, xs, lo, count);
            launch(p, lo, nr, ys, mode | kSum, ds, dstride);
        }
    }
    void affine(float* dst, unsigned dstride, const float* src, const Panel& p, unsigned rows, bool add) const {
        hipLaunchKernelGGL(affine_kernel, dim3(blocks_of(static_cast<std::size_t>(rows) * p.n)), dim3(256), 0, stream, dst, dstride, src,
                           p.n16, p.n, rows, p.b.as<const float>(), p.inv, add ? 1U : 0U, failure.as<unsigned>());
    }
    void norm(const float* x, float* y, unsigned pitch, unsigned rows, const Dev& w, const Dev& b) const {
        hipLaunchKernelGGL(norm_kernel, dim3(rows), dim3(256), 0, stream, x, pitch, y, pitch, cfg.width, w.as<const float>(),
                           b.as<const float>(), cfg.eps, cfg.norm == EncoderConfig::Norm::layernorm ? 1U : 0U);
    }
    template <unsigned Terms> void attention_terms(const AttnArgs& a, unsigned d) const {
        switch (d) {
        case 64: attn_core::dense::launch<BidirPolicy<64, 64, Terms>>(a, a.heads, stream); break;
        case 72: attn_core::dense::launch<BidirPolicy<72, 80, Terms>>(a, a.heads, stream); break;
        case 80: attn_core::dense::launch<BidirPolicy<80, 80, Terms>>(a, a.heads, stream); break;
        default: throw std::invalid_argument("vision encoder: head dimension");
        }
    }
    void attention(const AttnArgs& a, unsigned d) const {
        if (a.klo) attention_terms<2>(a, d);
        else attention_terms<1>(a, d);
    }
    void sync(const char* what) const {
        ck(hipStreamSynchronize(stream), what);
        ck(hipGetLastError(), what);
    }
    // Rows of `pitch` floats on the device -> [rows][width] on the host.
    std::vector<float> download(const float* dev, unsigned pitch, unsigned rows, unsigned width) const {
        sync("encode");
        std::vector<float> raw(std::size_t{rows} * pitch);
        ck(hipMemcpy(raw.data(), dev, raw.size() * 4U, hipMemcpyDeviceToHost), "download");
        if (pitch == width) return raw;
        std::vector<float> out(std::size_t{rows} * width);
        for (unsigned r = 0; r < rows; ++r) std::memcpy(&out[std::size_t{r} * width], &raw[std::size_t{r} * pitch], std::size_t{width} * 4U);
        return out;
    }
    void check_failure() const {
        unsigned bits = 0;
        sync("encode");
        ck(hipMemcpy(&bits, failure.p, 4U, hipMemcpyDeviceToHost), "failure flag");
        if (bits) throw std::runtime_error("vision encoder: numeric failure bits " + std::to_string(bits));
    }
};

Encoder::Encoder(EncoderConfig cfg, EncoderOptions opt) : impl_(std::make_unique<Impl>()) {
    std::string why;
    if (!hip_supported(cfg, &why)) throw std::invalid_argument("vision encoder: " + why);
    if (opt.attention_terms != 1U && opt.attention_terms != 2U) throw std::invalid_argument("vision encoder: attention_terms is 1 or 2");
    impl_->cfg = std::move(cfg);
    impl_->opt = opt;
    impl_->stream = static_cast<hipStream_t>(opt.stream);
    impl_->failure = Dev(layout::kAlign);
}
Encoder::~Encoder() = default;
const char* Encoder::arithmetic() const { return effective_arithmetic(impl_->opt.attention_terms); }

void Encoder::load(const EncoderWeights& w) {
    auto& im = *impl_;
    validate(im.cfg, w);
    unload();
    EncoderLoadStats& ls = im.load_stats;
    ls = {};
    const auto t0 = LoadClock::now();
    im.patch = make_panel(w.patch_w, w.patch_b, &ls);
    im.blocks.reserve(w.blocks.size());
    for (const auto& b : w.blocks) {
        Block d;
        d.qkv = make_panel(b.qkv_w, b.qkv_b, &ls);
        d.out = make_panel(b.out_w, b.out_b, &ls);
        d.mlp1 = make_panel(b.mlp1_w, b.mlp1_b, &ls);
        d.mlp2 = make_panel(b.mlp2_w, b.mlp2_b, &ls);
        auto tv = LoadClock::now();
        d.n1w = make_vector(b.norm1_w, im.cfg.width);
        d.n1b = make_vector(b.norm1_b, im.cfg.width);
        d.n2w = make_vector(b.norm2_w, im.cfg.width);
        d.n2b = make_vector(b.norm2_b, im.cfg.width);
        ls.vectors_ms += ms_from(tv);
        im.blocks.push_back(std::move(d));
    }
    if (im.cfg.final_norm) {
        auto tv = LoadClock::now();
        im.fnw = make_vector(w.final_norm_w, im.cfg.width);
        im.fnb = make_vector(w.final_norm_b, im.cfg.width);
        ls.vectors_ms += ms_from(tv);
    }
    im.loaded = true;
    ls.total_ms = std::chrono::duration<double, std::milli>(LoadClock::now() - t0).count();
}

const EncoderLoadStats& Encoder::load_stats() const { return impl_->load_stats; }

void Encoder::unload() {
    auto& im = *impl_;
    im.loaded = false;
    im.patch = Panel{};
    im.blocks.clear();
    im.fnw.reset();
    im.fnb.reset();
    im.drop_workspace();
    im.projector.clear();
    im.px.reset();
    im.py.reset();
    im.pact.reset();
    im.py2.reset();
    im.plo.reset();
    im.pylo.reset();
    im.phid.reset();
    im.stage.reset();
}

bool Encoder::loaded() const { return impl_->loaded; }

std::size_t Encoder::device_bytes() const {
    const auto& im = *impl_;
    std::size_t b = im.patch.bytes() + im.fnw.n + im.fnb.n + im.ws.n + im.failure.n;
    for (const auto& d : im.blocks) b += d.qkv.bytes() + d.out.bytes() + d.mlp1.bytes() + d.mlp2.bytes() + d.n1w.n + d.n1b.n + d.n2w.n + d.n2b.n;
    for (const auto& [key, p] : im.projector) b += p.bytes();
    return b + im.px.n + im.py.n + im.pact.n + im.py2.n + im.plo.n + im.pylo.n + im.phid.n;
}

std::size_t Encoder::pinned_host_bytes() const { return impl_->stage.n; }

EncoderOutput Encoder::encode(const PatchGrid& grid, const PositionInputs& pos) {
    EncodeStats stats;
    return encode(grid, pos, stats);
}

EncoderOutput Encoder::encode(const PatchGrid& grid, const PositionInputs& pos, EncodeStats& stats) {
    return run(grid, pos, stats, false);
}

EncoderOutput Encoder::encode_for_projection(const PatchGrid& grid, const PositionInputs& pos) {
    EncodeStats stats;
    return run(grid, pos, stats, project_device());
}

void Encoder::reserve(unsigned rows, unsigned tiles) {
    auto& im = *impl_;
    const EncoderPlan pl = plan(im.cfg, rows, tiles, im.opt);
    if (im.ws.n < pl.workspace_bytes()) {
        im.drop_workspace();
        im.ws = Dev(pl.workspace_bytes());
    }
}

void Encoder::prepare_linear(const Tensor& w, const Tensor& bias) {
    if (!w) throw std::invalid_argument("vision encoder: prepare_linear operand");
    (void)impl_->panel_for(w, bias);
}

EncoderOutput Encoder::run(const PatchGrid& grid, const PositionInputs& pos, EncodeStats& stats, bool keep_hidden) {
    auto& im = *impl_;
    const auto& c = im.cfg;
    if (!im.loaded) throw std::logic_error("vision encoder: encode before load");
    validate(c, grid, pos);
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<unsigned> tiles = grid.tile_end;
    if (tiles.empty()) tiles.push_back(grid.rows);
    std::uint64_t slots64 = 0;
    for (std::size_t t = 0; t < tiles.size(); ++t) slots64 += layout::key_slots(tiles[t] - (t ? tiles[t - 1U] : 0U));
    if (slots64 > 0x7FFFFFF0ULL) throw std::invalid_argument("vision encoder: too many rows");
    const unsigned N = grid.rows, slots = static_cast<unsigned>(slots64);
    const auto g = layout::geometry(c, N, slots, im.opt);
    const std::size_t need = g.fixed() + g.mlp();
    if (im.ws.n < need) {
        im.drop_workspace();
        im.ws = Dev(need);
    }
    im.hid = nullptr;  // this encode overwrites the workspace
    ++im.epoch;
    const auto stream = im.stream;
    // Zero workspace: the GEMM input pads (in % 128) and the padded head components are never written afterwards, so
    // a workspace cleared for this geometry before still has them zero; another geometry moves them.
    if (!im.ws_clear || im.ws_rows != N || im.ws_tiles != grid.tile_end) {
        ck(hipMemsetAsync(im.ws.p, 0, need, stream), "workspace clear");
        im.ws_clear = true;
        im.ws_rows = N;
        im.ws_tiles = grid.tile_end;
    }
    ck(hipMemsetAsync(im.failure.p, 0, 4U, stream), "failure clear");
    std::size_t used = 0;
    const auto take = [&](std::size_t bytes) {
        auto* p = static_cast<unsigned char*>(im.ws.p) + used;
        used += bytes;
        return p;
    };
    const auto f32 = [&](std::size_t bytes) { return bytes ? reinterpret_cast<float*>(take(bytes)) : nullptr; };
    float* const dpatch = f32(g.patches);
    float* const dpos = f32(g.pos);
    float* const dcos = f32(g.trig);
    float* const dsin = f32(g.trig);
    float* const x = f32(g.x);
    float* const nrm = f32(g.nrm);
    float* const qkv = f32(g.qkv);
    float* const q = f32(g.q);
    float* const ao = f32(g.ao);
    float* const cmp = f32(g.cmp);
    float* const o = f32(g.o);
    auto* const k = reinterpret_cast<_Float16*>(take(g.k));
    auto* const vt = reinterpret_cast<_Float16*>(take(g.vt));
    auto* const klo = g.klo ? reinterpret_cast<_Float16*>(take(g.klo)) : nullptr;
    auto* const vtlo = g.vtlo ? reinterpret_cast<_Float16*>(take(g.vtlo)) : nullptr;
    float* const lo = f32(g.lo);
    float* const ylo = f32(g.ylo);
    float* const mid = f32(g.mid);
    float* const act = f32(g.act);
    float* const o2 = f32(g.o2);
    float* const mlo = f32(g.mlo);
    float* const mylo = f32(g.mylo);
    if (used != need) throw std::logic_error("vision encoder: workspace layout");

    // Host -> device inputs: the four input buffers lead the workspace, so they are staged in page-locked memory
    // (zeros in the pads, as the clear leaves them) and go up in one asynchronous copy behind the clear. The staging
    // buffer is free again when this encode returns (it ends with a stream synchronisation).
    {
        const std::size_t prefix = static_cast<std::size_t>(reinterpret_cast<unsigned char*>(x) - static_cast<unsigned char*>(im.ws.p));
        im.stage.grow(prefix);
        auto* const base = static_cast<unsigned char*>(im.stage.p);
        std::memset(base, 0, prefix);
        const auto at = [&](const float* dev) {
            return reinterpret_cast<float*>(base + (reinterpret_cast<const unsigned char*>(dev) - static_cast<unsigned char*>(im.ws.p)));
        };
        if (g.pp == g.p) {
            std::memcpy(at(dpatch), grid.data.data(), grid.data.size() * 4U);
        } else {
            float* const hp = at(dpatch);
            for (unsigned r = 0; r < N; ++r) std::memcpy(&hp[std::size_t{r} * g.pp], &grid.data[std::size_t{r} * g.p], std::size_t{g.p} * 4U);
        }
        if (!pos.addend.empty()) std::memcpy(at(dpos), pos.addend.data(), pos.addend.size() * 4U);
        if (g.rope) {
            float* const cs = at(dcos);
            float* const sn = at(dsin);
            for (std::size_t i = 0; i < pos.angles.size(); ++i) {
                cs[i] = static_cast<float>(std::cos(static_cast<double>(pos.angles[i])));
                sn[i] = static_cast<float>(std::sin(static_cast<double>(pos.angles[i])));
            }
        }
        ck(hipMemcpyAsync(im.ws.p, base, prefix, hipMemcpyHostToDevice, stream), "input upload");
    }

    im.linear(im.patch, dpatch, N, o, x, g.wp, false, lo, ylo);
    if (!pos.addend.empty())
        hipLaunchKernelGGL(add_rows_kernel, dim3(blocks_of(static_cast<std::size_t>(N) * g.w)), dim3(256), 0, stream, x, g.wp,
                           static_cast<const float*>(dpos), g.w, N);

    EncoderOutput out;
    out.rows = N;
    out.width = c.width;
    stats.layer_ms.clear();
    std::size_t tap = 0;
    const unsigned mode = c.mlp == EncoderConfig::Mlp::gelu ? 0U : c.mlp == EncoderConfig::Mlp::gelu_tanh ? 1U : 2U;
    for (unsigned l = 0; l < c.layers; ++l) {
        const auto tl = std::chrono::steady_clock::now();
        const auto& B = im.blocks[l];
        // attention
        im.norm(x, nrm, g.wp, N, B.n1w, B.n1b);
        im.linear(B.qkv, nrm, N, qkv, qkv, g.q16, false, lo, ylo);
        unsigned row0 = 0, slot0 = 0;
        for (const unsigned end : tiles) {
            const unsigned n = end - row0, cap = layout::key_slots(n);
            const PackArgs pa{qkv, dcos, dsin, q, k, vt, klo, vtlo, im.failure.as<unsigned>(), g.q16, row0, n, slot0, g.heads, g.d, g.dp, g.w, slots,
                              c.rope_pairing == EncoderConfig::RopePairing::half ? 1U : 0U};
            hipLaunchKernelGGL(pack_kernel, dim3(cap), dim3(256), 0, stream, pa);
            const AttnArgs aa{q, ao, k, vt, klo, vtlo, im.failure.as<unsigned>(), n, row0, slot0, g.heads, slots};
            im.attention(aa, g.d);
            row0 = end;
            slot0 += cap;
        }
        const float* attn = ao;
        if (g.compact) {
            hipLaunchKernelGGL(compact_kernel, dim3(blocks_of(static_cast<std::size_t>(N) * g.w)), dim3(256), 0, stream,
                               static_cast<const float*>(ao), cmp, g.wp, g.heads, g.d, g.dp, N);
            attn = cmp;
        }
        im.linear(B.out, attn, N, o, x, g.wp, true, lo, ylo);
        // MLP, in row chunks so the wide middle stays under the workspace cap
        im.norm(x, nrm, g.wp, N, B.n2w, B.n2b);
        for (unsigned r0 = 0; r0 < N; r0 += g.chunk) {
            const unsigned nr = std::min(g.chunk, N - r0);
            im.gemm(B.mlp1, nrm + static_cast<std::size_t>(r0) * g.wp, nr, mid, mlo, mylo);
            const ActArgs a{mid, act, B.mlp1.b.as<const float>(), im.failure.as<unsigned>(), B.mlp1.inv, c.swiglu_limit, g.m116, g.mp, g.m, nr,
                            mode, c.swiglu_gate_first ? 1U : 0U};
            hipLaunchKernelGGL(act_kernel, dim3(blocks_of(static_cast<std::size_t>(nr) * g.m)), dim3(256), 0, stream, a);
            im.linear(B.mlp2, act, nr, o2, x + static_cast<std::size_t>(r0) * g.wp, g.wp, true, mlo, mylo);
        }
        if (tap < c.deepstack_taps.size() && c.deepstack_taps[tap] == l) {
            out.taps.push_back(im.download(x, g.wp, N, g.w));
            ++tap;
        }
        if (stats.per_layer) {
            im.sync("encode");
            stats.layer_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tl).count());
        }
    }
    const float* const hidden = c.final_norm ? nrm : x;
    if (c.final_norm) im.norm(x, nrm, g.wp, N, im.fnw, im.fnb);
    if (!keep_hidden) out.hidden = im.download(hidden, g.wp, N, g.w);
    im.check_failure();
    if (keep_hidden) {  // valid until the next encode, reserve() or unload()
        im.hid = hidden;
        im.hid_pitch = g.wp;
        out.device_epoch = im.epoch;
        out.device_owner = im.owner;
    }
    stats.total_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

void Encoder::linear(const Tensor& w, const Tensor& bias, const float* x, std::size_t rows, float* y) {
    auto& im = *impl_;
    if (!w || !x || !y || !rows || rows > 0x00FFFFFFU) throw std::invalid_argument("vision encoder: linear operands");
    const Panel& p = im.panel_for(w, bias);
    const unsigned nr = static_cast<unsigned>(rows);
    std::vector<float> padded(rows * p.inp, 0.0F);
    for (std::size_t r = 0; r < rows; ++r) std::memcpy(&padded[r * p.inp], x + r * p.in, std::size_t{p.in} * 4U);
    grow(im.px, padded.size() * 4U);
    grow(im.py, rows * p.n16 * 4U);
    if (im.opt.split_activations) {
        grow(im.plo, padded.size() * 4U);
        grow(im.pylo, rows * p.n16 * 4U);
    }
    ck(hipMemsetAsync(im.failure.p, 0, 4U, im.stream), "failure clear");
    ck(hipMemcpyAsync(im.px.p, padded.data(), padded.size() * 4U, hipMemcpyHostToDevice, im.stream), "linear upload");
    float* const lo = im.opt.split_activations ? im.plo.as<float>() : nullptr;
    float* const ylo = im.opt.split_activations ? im.pylo.as<float>() : nullptr;
    im.linear(p, im.px.as<const float>(), nr, im.py.as<float>(), im.py.as<float>(), p.n16, false, lo, ylo);
    const std::vector<float> got = im.download(im.py.as<const float>(), p.n16, nr, p.n);  // synchronises before `padded` dies
    im.check_failure();
    std::memcpy(y, got.data(), got.size() * 4U);
}

void Encoder::merge_project(const EncoderOutput& hidden, unsigned grid_h, unsigned grid_w, const MergeProjector& m, float* rows) {
    auto& im = *impl_;
    const bool resident = hidden.device_epoch != 0;
    if (resident && (!hidden.resident_for(im.owner, im.epoch) || !im.hid))
        throw std::logic_error("vision encoder: the device hidden rows of this output were overwritten");
    if (!project_device()) {  // host graph through linear(); encode_for_projection() downloaded the rows
        DenseOps::merge_project(hidden, grid_h, grid_w, m, rows);
        return;
    }
    const unsigned R = m.ratio, W = hidden.width;
    const auto geometry = layout::projector_geometry(hidden, grid_h, grid_w, m, rows, resident);
    const unsigned mw = geometry.mw, tokens = geometry.tokens;
    if (tokens > 0x00FFFFFFU) throw std::invalid_argument("vision encoder: merge projector rows");
    const Panel& p1 = im.panel_for(m.w1, m.b1);
    const Panel& p2 = im.panel_for(m.w2, m.b2);
    const auto stream = im.stream;
    const float* src = im.hid;
    unsigned pitch = im.hid_pitch;
    if (!resident) {  // host rows (a caller of encode()): one upload
        grow(im.phid, hidden.hidden.size() * 4U);
        ck(hipMemcpyAsync(im.phid.p, hidden.hidden.data(), hidden.hidden.size() * 4U, hipMemcpyHostToDevice, stream), "hidden upload");
        src = im.phid.as<const float>();
        pitch = W;
    }
    const std::size_t x1 = std::size_t{tokens} * p1.inp, y1 = std::size_t{tokens} * p1.n16, x2 = std::size_t{tokens} * p2.inp,
                      y2 = std::size_t{tokens} * p2.n16;
    grow(im.px, x1 * 4U);
    grow(im.py, y1 * 4U);
    grow(im.pact, x2 * 4U);
    grow(im.py2, y2 * 4U);
    float* lo = nullptr;
    float* ylo = nullptr;
    if (im.opt.split_activations) {
        grow(im.plo, std::max(x1, x2) * 4U);
        grow(im.pylo, std::max(y1, y2) * 4U);
        lo = im.plo.as<float>();
        ylo = im.pylo.as<float>();
    }
    ck(hipMemsetAsync(im.failure.p, 0, 4U, stream), "failure clear");
    hipLaunchKernelGGL(unshuffle_kernel, dim3(blocks_of(x1)), dim3(256), 0, stream, src, pitch, im.px.as<float>(), p1.inp, p1.in, R,
                       m.row_major ? 1U : 0U, grid_h, grid_w, mw, x1);
    im.linear(p1, im.px.as<const float>(), tokens, im.py.as<float>(), im.py.as<float>(), p1.n16, false, lo, ylo);
    const unsigned act = m.act == Activation::gelu ? 0U : m.act == Activation::gelu_tanh ? 1U : 2U;
    hipLaunchKernelGGL(project_act_kernel, dim3(blocks_of(x2)), dim3(256), 0, stream, im.py.as<const float>(), p1.n16, im.pact.as<float>(),
                       p2.inp, p1.n, act, x2);
    im.linear(p2, im.pact.as<const float>(), tokens, im.py2.as<float>(), im.py2.as<float>(), p2.n16, false, lo, ylo);
    const std::vector<float> got = im.download(im.py2.as<const float>(), p2.n16, tokens, p2.n);  // the one synchronisation
    im.check_failure();
    std::memcpy(rows, got.data(), got.size() * 4U);
}

void splice_rows(float* dst, std::size_t dst_stride, const std::uint16_t* src, unsigned width, const std::uint32_t* dst_row,
                 const std::uint32_t* src_row, unsigned count, void* stream) {
    if (!count) return;
    if (!dst || !src || !dst_row || !src_row || !width || dst_stride < width) throw std::invalid_argument("vision splice_rows");
    hipLaunchKernelGGL(splice_kernel, dim3(count), dim3(256), 0, static_cast<hipStream_t>(stream), dst, dst_stride, src, width, dst_row,
                       src_row);
    ck(hipGetLastError(), "splice_rows");
}

}  // namespace strix::vision
