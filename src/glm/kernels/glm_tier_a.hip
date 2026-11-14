// GLM-5.3-Flash Tier A device kernels, batch 1 (see glm_tier_a.hpp). Every kernel follows the operation order of the
// host emulation (src/glm/host/glm_host.cpp): R1 ascending fmaf chains, R2 lane-strided partials + LDS halving tree,
// separate multiply/add elsewhere. Build with -ffp-contract=off -fno-fast-math -fno-gpu-flush-denormals-to-zero.
#include "glm_norm.hpp"
#include "strix/hip_argmax_core.hpp"
#include "../../ops/norm/norm_device.hpp"

#include <stdexcept>
#include <string>

namespace strix::glm::dev {

namespace {

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

__device__ __forceinline__ float wload(const void* w, WType t, std::size_t k) {
    switch (t) {
        case WType::F16: return f16_to_float(static_cast<const std::uint16_t*>(w)[k]);
        case WType::BF16: return bf16_to_float(static_cast<const std::uint16_t*>(w)[k]);
        default: return static_cast<const float*>(w)[k];
    }
}
__device__ __forceinline__ void flag_if(unsigned* flag, float v) {
    if (flag && !is_finite(v)) atomicOr(flag, 1U);
}

// R2 tree in LDS over L lanes (s has L entries; lane l holds its partial). Returns s[0] to every lane.
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

// ---------------------------------------------------------------- R1 dense chain
__global__ void dense_kernel(const float* x, const void* w, WType wt, unsigned in, unsigned o0, unsigned n, float* y) {
    const unsigned o = blockIdx.x * 128 + threadIdx.x, r = blockIdx.y;
    if (o >= n) return;
    const float* xr = x + static_cast<std::size_t>(r) * in;
    const std::size_t base = static_cast<std::size_t>(o0 + o) * in;
    float acc = 0.0F;
    for (unsigned i = 0; i < in; ++i) acc = __builtin_fmaf(xr[i], wload(w, wt, base + i), acc);
    y[static_cast<std::size_t>(r) * n + o] = acc;
}

// ---------------------------------------------------------------- R2/R3 rmsnorm
struct GlmNormDeviceTraits : GlmNormTraits {
    template <unsigned L>
    __device__ __forceinline__ static float reduce(float* s, float part) {
        return lds_tree<L>(s, part);
    }
    __device__ __forceinline__ static float inv_sqrt(float x) { return strix::glm::inv_sqrt(x); }
    __device__ __forceinline__ static float load(const void* w, WType wt, std::size_t i) {
        return wload(w, wt, i);
    }
};

// ---------------------------------------------------------------- mHC
__global__ __launch_bounds__(256) void hc_pre_kernel(const float* X, const float* fn, const float* base,
                                                     const float* scale, float* F, float* mix, float* xc) {
    constexpr unsigned n = kHc * kHidden;
    __shared__ float s[256];
    __shared__ float z[kHcMix];
    __shared__ float m[kHcMix];
    const unsigned r = blockIdx.x, l = threadIdx.x;
    const float* Xr = X + static_cast<std::size_t>(r) * n;
    float* Fr = F + static_cast<std::size_t>(r) * n;
    float a = 0.0F;
    for (unsigned i = l; i < n; i += 256) a = __builtin_fmaf(Xr[i], Xr[i], a);
    const float ss = lds_tree<256>(s, a);
    const float rr = inv_sqrt(ss / static_cast<float>(n) + kEps);
    for (unsigned i = l; i < n; i += 256) Fr[i] = Xr[i] * rr;
    __syncthreads();
    if (l < kHcMix) {  // 24 chains over the normalized streams (each lane reads the F values it did not write: the
                       // __syncthreads above orders global writes within the block)
        float acc = 0.0F;
        const float* row = fn + static_cast<std::size_t>(l) * n;
        for (unsigned i = 0; i < n; ++i) acc = __builtin_fmaf(Fr[i], row[i], acc);
        z[l] = acc;
    }
    __syncthreads();
    if (l == 0) {  // R6 on lane 0 (host hc_mix order)
        for (unsigned j = 0; j < kHc; ++j) m[j] = sigmoid(z[j] * scale[0] + base[j]) + kHcEps;
        for (unsigned j = 0; j < kHc; ++j) m[kHc + j] = 2.0F * sigmoid(z[kHc + j] * scale[1] + base[kHc + j]);
        float* c = m + 2 * kHc;
        for (unsigned k = 0; k < kHc * kHc; ++k) c[k] = z[2 * kHc + k] * scale[2] + base[2 * kHc + k];
        for (unsigned j = 0; j < kHc; ++j) {
            float mx = c[j * kHc];
            for (unsigned k = 1; k < kHc; ++k) mx = c[j * kHc + k] > mx ? c[j * kHc + k] : mx;
            float e[kHc];
            for (unsigned k = 0; k < kHc; ++k) e[k] = declared_exp(c[j * kHc + k] - mx);
            const float sum = ((e[0] + e[1]) + e[2]) + e[3];
            for (unsigned k = 0; k < kHc; ++k) c[j * kHc + k] = e[k] / sum + kHcEps;
        }
        for (unsigned it = 0; it < kHcIters; ++it) {
            if (it > 0) {  // rows
                float sr[kHc];
                for (unsigned j = 0; j < kHc; ++j)
                    sr[j] = (((c[j * kHc + 0] + c[j * kHc + 1]) + c[j * kHc + 2]) + c[j * kHc + 3]) + kHcEps;
                for (unsigned j = 0; j < kHc; ++j)
                    for (unsigned k = 0; k < kHc; ++k) c[j * kHc + k] = c[j * kHc + k] / sr[j];
            }
            float sc[kHc];  // columns
            for (unsigned k = 0; k < kHc; ++k)
                sc[k] = (((c[0 * kHc + k] + c[1 * kHc + k]) + c[2 * kHc + k]) + c[3 * kHc + k]) + kHcEps;
            for (unsigned j = 0; j < kHc; ++j)
                for (unsigned k = 0; k < kHc; ++k) c[j * kHc + k] = c[j * kHc + k] / sc[k];
        }
        for (unsigned k = 0; k < kHcMix; ++k) mix[static_cast<std::size_t>(r) * kHcMix + k] = m[k];
    }
    __syncthreads();
    float* xr = xc + static_cast<std::size_t>(r) * kHidden;
    for (unsigned d = l; d < kHidden; d += 256)
        xr[d] = ((m[0] * Xr[d] + m[1] * Xr[kHidden + d]) + m[2] * Xr[2 * kHidden + d]) + m[3] * Xr[3 * kHidden + d];
}

__global__ void hc_expand_kernel(const float* X, const float* mix, const float* y, float* Xo, unsigned* flag, unsigned flag_stride) {
    const unsigned r = blockIdx.y, d = blockIdx.x * 256 + threadIdx.x;
    if (d >= kHidden) return;
    if (flag) flag += static_cast<std::size_t>(r) * flag_stride;  // stride 0: one flag for the call; 1: per-row flags (spec)
    const float* m = mix + static_cast<std::size_t>(r) * kHcMix;
    const float* c = m + 2 * kHc;
    const float* Xr = X + static_cast<std::size_t>(r) * kHc * kHidden;
    const float yd = y[static_cast<std::size_t>(r) * kHidden + d];
    for (unsigned i = 0; i < kHc; ++i) {
        const float mx = ((c[0 * kHc + i] * Xr[d] + c[1 * kHc + i] * Xr[kHidden + d]) + c[2 * kHc + i] * Xr[2 * kHidden + d]) +
                         c[3 * kHc + i] * Xr[3 * kHidden + d];
        const float v = m[kHc + i] * yd + mx;
        flag_if(flag, v);
        Xo[static_cast<std::size_t>(r) * kHc * kHidden + i * kHidden + d] = v;
    }
}

__global__ void hc_final_kernel(const float* X, float* h) {
    const unsigned r = blockIdx.y, d = blockIdx.x * 256 + threadIdx.x;
    if (d >= kHidden) return;
    const float* Xr = X + static_cast<std::size_t>(r) * kHc * kHidden;
    h[static_cast<std::size_t>(r) * kHidden + d] = (((Xr[d] + Xr[kHidden + d]) + Xr[2 * kHidden + d]) + Xr[3 * kHidden + d]) * 0.25F;
}

// ---------------------------------------------------------------- KDA
// 4a: one block per (head, row); lane = channel within the head (128). Conv for its q, k and v channel (R8), then
// l2norm q / k, the forget gate g and beta. scratch[row][head] = q | k | v | g (4 x 128).
__global__ __launch_bounds__(128) void kda_prep_kernel(KdaDev k, const float* qkv, const float* f, const float* b,
                                                       float* conv_state, float* scratch, float* beta) {
    __shared__ float s[128];
    const unsigned h = blockIdx.x, r = blockIdx.y, i = threadIdx.x, H = k.heads, C = H * kKdaDim;
    const float* in = qkv + static_cast<std::size_t>(r) * 3 * C;
    float* cs = conv_state + static_cast<std::size_t>(r) * 3 * C * 3;
    float out[3];
    for (unsigned part = 0; part < 3; ++part) {
        const unsigned ch = part * C + h * kKdaDim + i;
        float* st = cs + static_cast<std::size_t>(ch) * 3;
        const std::uint16_t* cw = k.conv_w + static_cast<std::size_t>(ch) * kConvK;
        const float nw = in[ch];
        float acc = __builtin_fmaf(bf16_to_float(cw[0]), st[0], 0.0F);
        acc = __builtin_fmaf(bf16_to_float(cw[1]), st[1], acc);
        acc = __builtin_fmaf(bf16_to_float(cw[2]), st[2], acc);
        acc = __builtin_fmaf(bf16_to_float(cw[3]), nw, acc);
        out[part] = silu(acc);
        st[0] = st[1];
        st[1] = st[2];
        st[2] = nw;
    }
    const float sq = lds_tree<128>(s, __builtin_fmaf(out[0], out[0], 0.0F));
    const float sk = lds_tree<128>(s, __builtin_fmaf(out[1], out[1], 0.0F));
    const float nq = declared_sqrt(sq + kL2Eps), nk = declared_sqrt(sk + kL2Eps);
    float* sc = scratch + (static_cast<std::size_t>(r) * H + h) * 4 * kKdaDim;
    sc[i] = (out[0] / nq) * kInvSqrt128;
    sc[kKdaDim + i] = out[1] / nk;
    sc[2 * kKdaDim + i] = out[2];
    const float expA = declared_exp(k.A_log[h]);
    sc[3 * kKdaDim + i] = kGateLow * sigmoid(expA * (f[static_cast<std::size_t>(r) * C + h * kKdaDim + i] + k.dt_bias[h * kKdaDim + i]));
    if (i == 0) beta[static_cast<std::size_t>(r) * H + h] = sigmoid(b[static_cast<std::size_t>(r) * H + h]);
}

// 4b: grid (head, value tile of 32, row); lane = key index k (128). For each value column v of the tile: decay,
// u_v = tree of S'_kv * k_k, delta, fmaf update, o_v = tree of S''_kv * q_k (host order per column).
__global__ __launch_bounds__(128) void kda_update_kernel(unsigned H, float* S, const float* scratch, const float* beta,
                                                         float* o, unsigned* flag) {
    __shared__ float s[128];
    const unsigned h = blockIdx.x, tile = blockIdx.y, r = blockIdx.z, kk = threadIdx.x;
    const float* sc = scratch + (static_cast<std::size_t>(r) * H + h) * 4 * kKdaDim;
    const float q = sc[kk], key = sc[kKdaDim + kk], dec = declared_exp(sc[3 * kKdaDim + kk]);
    const float bt = beta[static_cast<std::size_t>(r) * H + h];
    float* Sh = S + (static_cast<std::size_t>(r) * H + h) * kKdaDim * kKdaDim;
    for (unsigned c = 0; c < 32; ++c) {
        const unsigned vv = tile * 32 + c;
        const float sp = Sh[kk * kKdaDim + vv] * dec;
        const float u = lds_tree<128>(s, 0.0F + sp * key);  // host tree_sum: lane = +0 + term
        const float dlt = (sc[2 * kKdaDim + vv] - u) * bt;
        const float sn = __builtin_fmaf(key, dlt, sp);
        flag_if(flag, sn);
        Sh[kk * kKdaDim + vv] = sn;
        const float ov = lds_tree<128>(s, 0.0F + sn * q);
        if (kk == 0) o[(static_cast<std::size_t>(r) * H + h) * kKdaDim + vv] = ov;
    }
}

// 4c: one block per (head, row): R10.
__global__ __launch_bounds__(128) void kda_out_kernel(KdaDev k, const float* o, const float* gate, float* y) {
    __shared__ float s[128];
    const unsigned h = blockIdx.x, r = blockIdx.y, i = threadIdx.x, H = k.heads;
    const std::size_t at = (static_cast<std::size_t>(r) * H + h) * kKdaDim + i;
    const float v = o[at];
    const float ms = lds_tree<128>(s, __builtin_fmaf(v, v, 0.0F)) / static_cast<float>(kKdaDim);
    const float rr = inv_sqrt(ms + kEps);
    y[at] = ((v * rr) * bf16_to_float(k.o_norm[i])) * sigmoid(gate[at]);
}

// ---------------------------------------------------------------- router, combine, swiglu, argmax
__global__ void route_kernel(const float* logits, const float* bias, std::uint32_t* ids, float* w) {
    __shared__ float sg[kExperts];  // LDS, not private scratch (one lane uses it)
    __shared__ std::uint32_t key[kExperts];
    __shared__ bool taken[kExperts];
    if (threadIdx.x != 0) return;
    const unsigned r = blockIdx.x;
    const float* lg = logits + static_cast<std::size_t>(r) * kExperts;
    for (unsigned e = 0; e < kExperts; ++e) {
        sg[e] = sigmoid(lg[e]);
        key[e] = ordered_key(sg[e] + bias[e]);
    }
    for (unsigned e = 0; e < kExperts; ++e) taken[e] = false;
    std::uint32_t pick[kTopE];
    for (unsigned k = 0; k < kTopE; ++k) {  // largest key, ties -> lower index (== the host stable sort's first 8)
        unsigned best = kExperts;
        for (unsigned e = 0; e < kExperts; ++e)
            if (!taken[e] && (best == kExperts || key[e] > key[best])) best = e;
        taken[best] = true;
        pick[k] = best;
    }
    for (unsigned a = 1; a < kTopE; ++a)  // ascending id
        for (unsigned b = a; b > 0 && pick[b - 1] > pick[b]; --b) {
            const std::uint32_t t = pick[b];
            pick[b] = pick[b - 1];
            pick[b - 1] = t;
        }
    float den = 0.0F;
    for (unsigned k = 0; k < kTopE; ++k) den = den + sg[pick[k]];
    den = den + kRouteDenEps;
    for (unsigned k = 0; k < kTopE; ++k) {
        ids[static_cast<std::size_t>(r) * kTopE + k] = pick[k];
        w[static_cast<std::size_t>(r) * kTopE + k] = (sg[pick[k]] / den) * kRouteScale;
    }
}

__global__ void moe_combine_kernel(const float* w, const float* eo, const float* shared, float* y) {
    const unsigned r = blockIdx.y, d = blockIdx.x * 256 + threadIdx.x;
    if (d >= kHidden) return;
    float acc = 0.0F;
    for (unsigned k = 0; k < kTopE; ++k)
        acc = __builtin_fmaf(w[static_cast<std::size_t>(r) * kTopE + k], eo[(static_cast<std::size_t>(r) * kTopE + k) * kHidden + d], acc);
    y[static_cast<std::size_t>(r) * kHidden + d] = acc + shared[static_cast<std::size_t>(r) * kHidden + d];
}

__global__ void swiglu_kernel(const float* g, const float* u, float* a, std::size_t n) {
    const std::size_t i = blockIdx.x * std::size_t{256} + threadIdx.x;
    if (i >= n) return;
    const float gg = g[i] > kClamp ? kClamp : g[i];
    const float uu = u[i] > kClamp ? kClamp : (u[i] < -kClamp ? -kClamp : u[i]);
    a[i] = silu(gg) * uu;
}

__global__ __launch_bounds__(256) void argmax_kernel(const float* logits, unsigned n, std::uint32_t id0, float* value,
                                                     std::uint32_t* id, unsigned* flag, unsigned flag_stride) {
    struct Best { float value; std::uint32_t id; };
    __shared__ Best s[256];
    const unsigned r = blockIdx.x, l = threadIdx.x;
    if (flag) flag += static_cast<std::size_t>(r) * flag_stride;
    const float* lg = logits + static_cast<std::size_t>(r) * n;
    float bv = 0.0F;
    std::uint32_t bi = 0xffffffffU;
    for (unsigned i = l; i < n; i += 256) {
        const float v = lg[i];
        flag_if(flag, v);
        if (bi == 0xffffffffU || v > bv) {  // strided ascending: ties keep the lower id
            bv = v;
            bi = i;
        }
    }
    // Shared argmax tree (R2-07); this kernel keeps its own ordering (the first strided element is taken
    // even when non-finite, flagged separately) so the combine below is unchanged.
    const Best best = strix::argmax::block_reduce<256>(s, l, Best{bv, bi}, [](const Best& x, const Best& y) {
        if (y.id == 0xffffffffU) return x;
        const bool take = x.id == 0xffffffffU || y.value > x.value || (y.value == x.value && y.id < x.id);
        return take ? y : x;
    });
    if (l == 0) {
        value[r] = best.value;
        id[r] = id0 + best.id;
    }
}

}  // namespace

void dense(const float* x, const void* w, WType wt, unsigned in, unsigned o0, unsigned n, float* y, unsigned rows, hipStream_t s) {
    dense_kernel<<<dim3((n + 127) / 128, rows), 128, 0, s>>>(x, w, wt, in, o0, n, y);
    ok(hipGetLastError(), "glm dense");
}
void rmsnorm(const float* x, const void* w, WType wt, unsigned n, float eps, float* y, unsigned rows, hipStream_t s) {
    if (GlmNorm::lanes(n) == 256)
        ops::norm::rmsnorm_kernel<GlmNormDeviceTraits, 256><<<rows, 256, 0, s>>>(x, w, wt, n, eps, y);
    else
        ops::norm::rmsnorm_kernel<GlmNormDeviceTraits, 128><<<rows, 128, 0, s>>>(x, w, wt, n, eps, y);
    ok(hipGetLastError(), "glm rmsnorm");
}
void hc_pre(const float* X, const float* fn, const float* base, const float* scale, float* F, float* mix, float* xc,
            unsigned rows, hipStream_t s) {
    hc_pre_kernel<<<rows, 256, 0, s>>>(X, fn, base, scale, F, mix, xc);
    ok(hipGetLastError(), "glm hc_pre");
}
void hc_expand(const float* X, const float* mix, const float* y, float* Xo, unsigned* flag, unsigned rows, hipStream_t s,
               unsigned flag_stride) {
    hc_expand_kernel<<<dim3(kHidden / 256, rows), 256, 0, s>>>(X, mix, y, Xo, flag, flag_stride);
    ok(hipGetLastError(), "glm hc_expand");
}
void hc_final(const float* X, float* h, unsigned rows, hipStream_t s) {
    hc_final_kernel<<<dim3(kHidden / 256, rows), 256, 0, s>>>(X, h);
    ok(hipGetLastError(), "glm hc_final");
}
void kda_step(const KdaDev& k, const float* qkv, const float* f, const float* gate, const float* b, float* S, float* conv_state,
              float* scratch, float* beta, float* o, float* y, unsigned* flag, unsigned rows, hipStream_t s) {
    kda_prep_kernel<<<dim3(k.heads, rows), 128, 0, s>>>(k, qkv, f, b, conv_state, scratch, beta);
    ok(hipGetLastError(), "glm kda_prep");
    kda_update_kernel<<<dim3(k.heads, kKdaDim / 32, rows), 128, 0, s>>>(k.heads, S, scratch, beta, o, flag);
    ok(hipGetLastError(), "glm kda_update");
    kda_out_kernel<<<dim3(k.heads, rows), 128, 0, s>>>(k, o, gate, y);
    ok(hipGetLastError(), "glm kda_out");
}
void route(const float* logits, const float* bias, std::uint32_t* ids, float* w, unsigned rows, hipStream_t s) {
    route_kernel<<<rows, 32, 0, s>>>(logits, bias, ids, w);
    ok(hipGetLastError(), "glm route");
}
void moe_combine(const float* w, const float* eo, const float* shared, float* y, unsigned rows, hipStream_t s) {
    moe_combine_kernel<<<dim3(kHidden / 256, rows), 256, 0, s>>>(w, eo, shared, y);
    ok(hipGetLastError(), "glm moe_combine");
}
void swiglu(const float* g, const float* u, float* a, std::size_t n, hipStream_t s) {
    swiglu_kernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, s>>>(g, u, a, n);
    ok(hipGetLastError(), "glm swiglu");
}
void argmax(const float* logits, unsigned n, std::uint32_t id0, float* value, std::uint32_t* id, unsigned* flag, unsigned rows,
            hipStream_t s, unsigned flag_stride) {
    argmax_kernel<<<rows, 256, 0, s>>>(logits, n, id0, value, id, flag, flag_stride);
    ok(hipGetLastError(), "glm argmax");
}

}  // namespace strix::glm::dev
