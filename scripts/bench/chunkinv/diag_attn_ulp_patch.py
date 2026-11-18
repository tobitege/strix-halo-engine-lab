# Gate null-control DIAGNOSTIC source patch (never merged): apply to a scratch copy of src/minimax/engine/engine.hip.
# STRIX_MM_DIAG_ATTN_ULP="L:seed": every prefill window (n > 8 rows) of layer L gets each finite non-zero value of
# the rank's attention output (before the o-projection) moved by one ulp, up or down by a hash of (index, seed).
# Unset = no launch, bytes unchanged.
import sys
path = sys.argv[1]
src = open(path, encoding="utf-8").read()
A1 = "struct Engine::Impl {"
A2 = '            if (tl && !traced_batch) tl->mark("msa", false);'
assert src.count(A1) == 1, ("anchor 1", src.count(A1))
assert src.count(A2) == 1, ("anchor 2", src.count(A2))
K = r'''// gate DIAGNOSTIC (null control, never merged): one ulp up / down on every finite non-zero value.
__global__ void diag_attn_ulp_kernel(float* x, unsigned long long n, unsigned seed) {
    const unsigned long long i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned b;
    __builtin_memcpy(&b, x + i, 4);
    if ((b & 0x7f800000U) == 0x7f800000U || (b & 0x7fffffffU) == 0U) return;
    unsigned z = static_cast<unsigned>(i) * 0x9E3779B1U ^ (seed * 0x85EBCA6BU + static_cast<unsigned>(i >> 32));
    z ^= z >> 15;
    z *= 0x2C1B3C6DU;
    z ^= z >> 12;
    z *= 0x297A2D39U;
    z ^= z >> 15;
    b += (z & 1U) != 0U ? 1U : 0xFFFFFFFFU;
    if ((b & 0x7f800000U) == 0x7f800000U) return;
    __builtin_memcpy(x + i, &b, 4);
}

'''
H = r'''            {   // gate DIAGNOSTIC (null control, never merged)
                static const int diag_layer = [] {
                    const char* e = std::getenv("STRIX_MM_DIAG_ATTN_ULP");
                    return e != nullptr && e[0] != 0 ? std::atoi(e) : -1;
                }();
                static const unsigned diag_seed = [] {
                    const char* e = std::getenv("STRIX_MM_DIAG_ATTN_ULP");
                    const char* c = e != nullptr ? std::strchr(e, ':') : nullptr;
                    return c != nullptr ? static_cast<unsigned>(std::atoi(c + 1)) : 1U;
                }();
                if (diag_layer == static_cast<int>(L) && n > 8) {
                    const unsigned long long diag_n = static_cast<unsigned long long>(n) * kQLocal;
                    hipLaunchKernelGGL(diag_attn_ulp_kernel, dim3(static_cast<unsigned>((diag_n + 255) / 256)), dim3(256), 0, nullptr, attn, diag_n, diag_seed);
                    std::fprintf(stderr, "DIAG attn ulp: layer %u seed %u rows %zu pos0 %llu values %llu\n", L, diag_seed,
                                 static_cast<std::size_t>(n), static_cast<unsigned long long>(pos0), diag_n);
                }
            }
'''
src = src.replace(A1, K + A1).replace(A2, H + A2)
open(path, "w", encoding="utf-8", newline="\n").write(src)
print("patched", path)
