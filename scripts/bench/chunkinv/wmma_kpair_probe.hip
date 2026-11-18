// WMMA K-pair probe (chunk_contract.hpp section 4): does row 0 of v_wmma_f32_16x16x16_f16 change with B row 15 when
// A[0][15] = 0? KPAIR_ZERO=one zeroes A[0][15] only (gfx1151: changes in ~3/4 of random cases), pair zeroes A[0][14]
// too (expected 0). Prints "<changed> of <trials>".
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef float f8 __attribute__((ext_vector_type(8)));
__global__ void wmma(const _Float16* A, const _Float16* B, const float* C, float* D) {
    const unsigned l = threadIdx.x, r = l & 15U, hw = l >> 4U;
    h16 a, b; f8 c;
    for (int i = 0; i < 16; ++i) { a[i] = A[r * 16 + i]; b[i] = B[i * 16 + r]; }
    for (int e = 0; e < 8; ++e) c[e] = C[(2 * e + hw) * 16 + r];
    const f8 d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
    for (int e = 0; e < 8; ++e) D[(2 * e + hw) * 16 + r] = d[e];
}
int main() {
    const char* m = std::getenv("KPAIR_ZERO");
    const bool pair = m && std::string_view(m) == "pair";
    _Float16 A[256], B[256]; float C[256], D[2][256];
    _Float16 *dA, *dB; float *dC, *dD;
    if (hipMalloc(&dA, sizeof A) || hipMalloc(&dB, sizeof B) || hipMalloc(&dC, sizeof C) || hipMalloc(&dD, sizeof C)) return 2;
    std::srand(7);
    const auto uni = [] { return float(std::rand()) / float(RAND_MAX); };
    unsigned changed = 0, trials = 2000;
    for (unsigned t = 0; t < trials; ++t) {
        for (int i = 0; i < 256; ++i) { A[i] = _Float16(uni()); B[i] = _Float16(uni() * 4 - 2); C[i] = (uni() * 8 - 4) * (t & 1U ? 1e-6F : 1.F); }
        A[15] = _Float16(0);
        if (pair) A[14] = _Float16(0);
        for (int run = 0; run < 2; ++run) {
            if (run) for (int i = 0; i < 16; ++i) B[15 * 16 + i] = _Float16(0);
            if (hipMemcpy(dA, A, sizeof A, hipMemcpyHostToDevice) || hipMemcpy(dB, B, sizeof B, hipMemcpyHostToDevice) ||
                hipMemcpy(dC, C, sizeof C, hipMemcpyHostToDevice)) return 2;
            hipLaunchKernelGGL(wmma, dim3(1), dim3(32), 0, nullptr, dA, dB, dC, dD);
            if (hipMemcpy(D[run], dD, sizeof C, hipMemcpyDeviceToHost)) return 2;
        }
        changed += std::memcmp(D[0], D[1], 16 * sizeof(float)) != 0;
    }
    std::printf("%u of %u\n", changed, trials);
    return 0;
}
