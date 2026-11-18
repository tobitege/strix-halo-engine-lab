#pragma once
// CUDA port of glm/moe/glm_trellis_device.hpp (mul1, fragment permutation,
// H128 rounding boundaries). Original provenance remains in that header.
#include <cuda_fp16.h>
#include <cstdint>
#include <cmath>
#include <limits>

namespace strix::glm::rtxpf {
using U16 = std::uint16_t;
using U32 = std::uint32_t;
__host__ __device__ inline float hf(U16 h) { return __half2float(__ushort_as_half(h)); }
__host__ __device__ inline U16 fh(float x) { return __half_as_ushort(__float2half_rn(x)); }
__host__ __device__ inline U16 book(U32 s) {
    U32 p = s * 0x83dcd12dU;
    U32 b = (p & 255) + ((p >> 8) & 255) + ((p >> 16) & 255) + (p >> 24);
    // Exactly the half fused operation on (0x6400 + byte-sum), 0x1eee,
    // 0xc931: evaluate its exact dyadic numerator before a single RN16.
    return fh(float(int((1024 + b) * 1774) - 1329 * 2048) * 0x1p-18f);
}
__host__ __device__ inline U32 state(const U16* p, unsigned pos, unsigned bits) {
    unsigned start = ((pos + 1) * bits + 256 * bits - 16) % (256 * bits);
    unsigned i = start / 32, s = start % 32, j = (i + 1) % (8 * bits);
    U32 a = U32(p[2*i]) | U32(p[2*i+1]) << 16;
    U32 b = U32(p[2*j]) | U32(p[2*j+1]) << 16;
    return ((a << s) | (s ? b >> (32-s) : 0)) >> 16;
}
__host__ __device__ inline unsigned qi(unsigned pos) {
    unsigned lane = pos / 8, f = pos % 8;
    return 2*(lane%4) + f%2 + 8*((f/2)%2);
}
__host__ __device__ inline unsigned qo(unsigned pos) { return pos/32 + 8*(pos%8/4); }
inline void hadamard(float* h) {
    for (unsigned s=1; s<128; s*=2)
        for (unsigned i=0; i<128; ++i) if (!(i&s)) {
            float a=h[i], b=h[i+s]; h[i]=a+b; h[i+s]=a-b;
        }
    for (unsigned i=0;i<128;++i) h[i] *= 0x1.6a09e6p-4f;
}
// Frozen CUDA-native nonlinear polynomial, no fast-math or contraction.
__host__ __device__ inline float declared_exp(float x) {
    if (!(x==x)) return x;
    if (x>88.f) {
#if defined(__CUDA_ARCH__)
        return __int_as_float(0x7f800000);
#else
        return std::numeric_limits<float>::infinity();
#endif
    }
    if (x< -103.f) return 0.f;
    float n=nearbyintf(x*1.44269504f);
    float r=(x-n*0.693359375f)-n*-2.12194440e-4f;
    float p=1.98412698e-4f;
    p=p*r+1.38888889e-3f; p=p*r+8.33333333e-3f;
    p=p*r+4.16666667e-2f; p=p*r+1.66666667e-1f;
    p=p*r+0.5f; p=p*r+1.f; p=p*r+1.f;
    int k=int(n), k1=k/2;
    return (p*ldexpf(1.f,k1))*ldexpf(1.f,k-k1);
}
__host__ __device__ inline float swiglu(float g,float u) {
    g=fminf(g,10.f); u=fminf(fmaxf(u,-10.f),10.f);
    return (g/(1.f+declared_exp(-g)))*u;
}
}
