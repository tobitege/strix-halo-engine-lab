// GLM post-exchange kernels (see glm_post_exchange.hpp). Build with -ffp-contract=off -fno-fast-math
// -fno-gpu-flush-denormals-to-zero, as the Tier A kernels (src/glm/kernels/glm_tier_a.hip).
#include "glm_post_exchange.hpp"
#include "glm_xfer_device.hpp"

#include "glm/common/glm_math.hpp"

#include <stdexcept>
#include <string>

namespace strix::glm::xfer {

namespace {

void ok(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

__device__ __forceinline__ void flag_if(unsigned* flag, float v) {
    if (flag && !is_finite(v)) atomicOr(flag, 1U);
}

// R17 for element d of row r: rank 0's partial first on both ranks.
__device__ __forceinline__ float reduce(const float* local, const float* peer, unsigned rank, std::size_t k) {
    if (local == nullptr) return peer[k];  // exchange row fold: `peer` is the folded slot (rank 0's + rank 1's partial)
    const float a = local[k], b = peer[k];
    return rank == 0 ? a + b : b + a;
}

__global__ __launch_bounds__(256) void post_expand_kernel(const float* X, const float* mix, const float* local, const float* peer,
                                                          RxView v, unsigned rank, float* Xo, unsigned* flag, unsigned flag_stride) {
    if (peer == nullptr) {
        glmx_wait(v);
        peer = static_cast<const float*>(v.payload);
    }
    const unsigned r = blockIdx.y, d = blockIdx.x * 256 + threadIdx.x;
    if (d >= kHidden) return;
    const float* m = mix + static_cast<std::size_t>(r) * kHcMix;
    const float* c = m + 2 * kHc;
    const float* Xr = X + static_cast<std::size_t>(r) * kHc * kHidden;
    const float yd = reduce(local, peer, rank, static_cast<std::size_t>(r) * kHidden + d);
    for (unsigned i = 0; i < kHc; ++i) {
        const float mx = ((c[0 * kHc + i] * Xr[d] + c[1 * kHc + i] * Xr[kHidden + d]) + c[2 * kHc + i] * Xr[2 * kHidden + d]) +
                         c[3 * kHc + i] * Xr[3 * kHidden + d];
        const float o = m[kHc + i] * yd + mx;
        flag_if(flag ? flag + blockIdx.y * flag_stride : nullptr, o);
        Xo[static_cast<std::size_t>(r) * kHc * kHidden + i * kHidden + d] = o;
    }
}

__global__ __launch_bounds__(256) void hc_bracket_kernel(const float* X, const float* mix, float* B) {
    const unsigned r = blockIdx.y, d = blockIdx.x * 256 + threadIdx.x;
    if (d >= kHidden) return;
    const float* c = mix + static_cast<std::size_t>(r) * kHcMix + 2 * kHc;
    const float* Xr = X + static_cast<std::size_t>(r) * kHc * kHidden;
    for (unsigned i = 0; i < kHc; ++i)
        B[static_cast<std::size_t>(r) * kHc * kHidden + i * kHidden + d] =
            ((c[0 * kHc + i] * Xr[d] + c[1 * kHc + i] * Xr[kHidden + d]) + c[2 * kHc + i] * Xr[2 * kHidden + d]) + c[3 * kHc + i] * Xr[3 * kHidden + d];
}

__global__ __launch_bounds__(256) void post_expand_bracket_kernel(const float* B, const float* mix, const float* local, const float* peer,
                                                                  RxView v, unsigned rank, float* Xo, unsigned* flag, unsigned flag_stride) {
    if (peer == nullptr) {
        glmx_wait(v);
        peer = static_cast<const float*>(v.payload);
    }
    const unsigned r = blockIdx.y, d = blockIdx.x * 256 + threadIdx.x;
    if (d >= kHidden) return;
    const float* m = mix + static_cast<std::size_t>(r) * kHcMix;
    const float yd = reduce(local, peer, rank, static_cast<std::size_t>(r) * kHidden + d);
    for (unsigned i = 0; i < kHc; ++i) {
        const std::size_t k = static_cast<std::size_t>(r) * kHc * kHidden + i * kHidden + d;
        const float o = m[kHc + i] * yd + B[k];
        flag_if(flag ? flag + blockIdx.y * flag_stride : nullptr, o);
        Xo[k] = o;
    }
}

// R2 tree in LDS over 256 lanes (as Tier A lds_tree<256>).
__device__ __forceinline__ float lds_tree256(float* s, float part) {
    const unsigned l = threadIdx.x;
    s[l] = part;
    __syncthreads();
    for (unsigned h = 128; h > 0; h >>= 1U) {
        if (l < h) s[l] = s[l] + s[l + h];
        __syncthreads();
    }
    const float r = s[0];
    __syncthreads();
    return r;
}

__global__ __launch_bounds__(256) void hc_pre_v2_kernel(const float* X, const float* fn, const float* base, const float* scale,
                                                        float* F, float* mix, float* xc) {
    constexpr unsigned n = kHc * kHidden;
    __shared__ float s[256];
    __shared__ float z[kHcMix];
    __shared__ float m[kHcMix];
    const unsigned r = blockIdx.x, l = threadIdx.x;
    const float* Xr = X + static_cast<std::size_t>(r) * n;
    float* Fr = F + static_cast<std::size_t>(r) * n;
    float a = 0.0F;
    for (unsigned i = l; i < n; i += 256) a = __builtin_fmaf(Xr[i], Xr[i], a);
    const float ss = lds_tree256(s, a);
    const float rr = inv_sqrt(ss / static_cast<float>(n) + kEps);
    for (unsigned i = l; i < n; i += 256) Fr[i] = Xr[i] * rr;
    __syncthreads();
    if (l < kHcMix) {  // the 24 R1 chains, unchanged
        float acc = 0.0F;
        const float* row = fn + static_cast<std::size_t>(l) * n;
        for (unsigned i = 0; i < n; ++i) acc = __builtin_fmaf(Fr[i], row[i], acc);
        z[l] = acc;
    }
    __syncthreads();
    if (l < 32) {  // wave 0: R6 spread over lanes (every lane runs the shuffles; lanes >= 24 idle in the math)
        const unsigned t = l < 16 ? l : 0, j = t / kHc, k = t % kHc;
        if (l >= 16 && l < 20) m[l - 16] = sigmoid(z[l - 16] * scale[0] + base[l - 16]) + kHcEps;
        if (l >= 20 && l < 24) m[kHc + l - 20] = 2.0F * sigmoid(z[kHc + l - 20] * scale[1] + base[kHc + l - 20]);
        float c = z[2 * kHc + t] * scale[2] + base[2 * kHc + t];
        {  // row softmax + eps (host: maxf, then e_k, ((e0 + e1) + e2) + e3, e / sum + eps)
            float e[kHc], cr[kHc];
            for (unsigned q = 0; q < kHc; ++q) cr[q] = __shfl(c, static_cast<int>(j * kHc + q));
            float mx = cr[0];
            for (unsigned q = 1; q < kHc; ++q) mx = cr[q] > mx ? cr[q] : mx;
            for (unsigned q = 0; q < kHc; ++q) e[q] = declared_exp(cr[q] - mx);
            const float sum = ((e[0] + e[1]) + e[2]) + e[3];
            c = e[k] / sum + kHcEps;
        }
        for (unsigned it = 0; it < kHcIters; ++it) {
            if (it > 0) {  // rows
                const float sr = (((__shfl(c, static_cast<int>(j * kHc + 0)) + __shfl(c, static_cast<int>(j * kHc + 1))) +
                                   __shfl(c, static_cast<int>(j * kHc + 2))) +
                                  __shfl(c, static_cast<int>(j * kHc + 3))) +
                                 kHcEps;
                c = c / sr;
            }
            const float sc = (((__shfl(c, static_cast<int>(0 * kHc + k)) + __shfl(c, static_cast<int>(1 * kHc + k))) +
                               __shfl(c, static_cast<int>(2 * kHc + k))) +
                              __shfl(c, static_cast<int>(3 * kHc + k))) +
                             kHcEps;
            c = c / sc;
        }
        if (l < 16) m[2 * kHc + l] = c;
    }
    __syncthreads();
    if (l < kHcMix) mix[static_cast<std::size_t>(r) * kHcMix + l] = m[l];
    float* xr = xc + static_cast<std::size_t>(r) * kHidden;
    for (unsigned d = l; d < kHidden; d += 256)
        xr[d] = ((m[0] * Xr[d] + m[1] * Xr[kHidden + d]) + m[2] * Xr[2 * kHidden + d]) + m[3] * Xr[3 * kHidden + d];
}

}  // namespace

void hc_pre_v2(const float* X, const float* fn, const float* base, const float* scale, float* F, float* mix, float* xc,
               unsigned rows, hipStream_t s) {
    hc_pre_v2_kernel<<<rows, 256, 0, s>>>(X, fn, base, scale, F, mix, xc);
    ok(hipGetLastError(), "glm hc_pre_v2");
}

void post_expand(const float* X, const float* mix, const float* local, const float* peer, const RxView& view, unsigned rank,
                 float* Xout, unsigned* flag, unsigned rows, hipStream_t s, unsigned flag_stride) {
    post_expand_kernel<<<dim3(kHidden / 256, rows), 256, 0, s>>>(X, mix, local, peer, view, rank, Xout, flag, flag_stride);
    ok(hipGetLastError(), "glm post_expand");
}
void hc_bracket(const float* X, const float* mix, float* B, unsigned rows, hipStream_t s) {
    hc_bracket_kernel<<<dim3(kHidden / 256, rows), 256, 0, s>>>(X, mix, B);
    ok(hipGetLastError(), "glm hc_bracket");
}
void post_expand_bracket(const float* B, const float* mix, const float* local, const float* peer, const RxView& view,
                         unsigned rank, float* Xout, unsigned* flag, unsigned rows, hipStream_t s, unsigned flag_stride) {
    post_expand_bracket_kernel<<<dim3(kHidden / 256, rows), 256, 0, s>>>(B, mix, local, peer, view, rank, Xout, flag, flag_stride);
    ok(hipGetLastError(), "glm post_expand_bracket");
}

}  // namespace strix::glm::xfer
