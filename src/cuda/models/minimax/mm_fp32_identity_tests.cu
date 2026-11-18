// RTX PRO driver of the FP32 prefill identity tests (src/minimax/rtxpf/fp32_identity_tests.inc).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

#include "fp32_identity.cu"

static unsigned test_core_version = 0;
#define FP32_TEST_CONFIGURE_CORE(c) ((c).core_version = test_core_version)

static void cu_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
#define GPU_MALLOC(p, n) cu_check(cudaMalloc(reinterpret_cast<void**>(p), (n)), "malloc")
#define GPU_FREE(p) cu_check(cudaFree(p), "free")
#define GPU_H2D(d, h, n) cu_check(cudaMemcpy((d), (h), (n), cudaMemcpyHostToDevice), "h2d")
#define GPU_D2H(h, d, n) cu_check(cudaMemcpy((h), (d), (n), cudaMemcpyDeviceToHost), "d2h")
#define GPU_SYNC() cu_check(cudaDeviceSynchronize(), "sync")
#include "minimax/rtxpf/fp32_identity_tests.inc"

__global__ void check_mul1(float* out) {
    const unsigned state = blockIdx.x * blockDim.x + threadIdx.x;
    if (state < 65536) {
        out[state] = strix::minimax::rtxpf::dev::mul1_v31(state);
        out[65536 + state] = strix::minimax::rtxpf::dev::f32_mul1(state);
    }
}

int main() {
    namespace R = strix::minimax::rtxpf::ref;
    R::Gen g{7};
    long long bad = 0;
    for (int i = 0; i < 2000000; ++i) {
        const float f = g.value(4, 30);
        bad += R::float_to_half(f) != __half_as_ushort(__float2half_rn(f));
    }
    std::printf("host float_to_half vs __float2half_rn: %lld mismatches%s\n", bad, bad ? "  FAIL" : "");
    float* decoded = nullptr;
    GPU_MALLOC(&decoded, 2 * 65536 * sizeof(float));
    check_mul1<<<256, 256>>>(decoded);
    GPU_SYNC();
    std::vector<float> host(2 * 65536);
    GPU_D2H(host.data(), decoded, host.size() * sizeof(float));
    GPU_FREE(decoded);
    for (unsigned version = 0; version < 2; ++version) {
        long long mismatches = 0;
        for (unsigned state = 0; state < 65536; ++state)
            mismatches += R::bits_of(host[version * 65536 + state]) != R::bits_of(R::mul1(state));
        std::printf("%s mul1 all 65536 states vs host: %lld mismatches\n", version ? "v3" : "v3.1", mismatches);
        bad += mismatches;
    }
    struct Selection { const char* env; unsigned explicit_version, expected; const char* name; };
    const Selection cases[] = {
        {"", 0, 31, "default v3.1 (environment unset)"},
        {"3", 0, 3, "v3 environment"}, {"31", 0, 31, "v3.1 environment"},
        {"3", 31, 31, "explicit v3.1 overrides v3 environment"},
        {"31", 3, 3, "explicit v3 overrides v3.1 environment"},
    };
    int status = 0;
    for (const auto& c : cases) {
        if (_putenv_s("STRIX_MM_FP32_CORE", c.env) != 0) return 2;
        test_core_version = c.explicit_version;
        bad += strix::minimax::rtxpf::resolve_core_version(test_core_version, std::getenv("STRIX_MM_FP32_CORE")) != c.expected;
        std::printf("%s vs host reference\n", c.name);
        status |= fp32_identity_tests::run_all();
    }
    return status | (bad ? 1 : 0);
}
