#pragma once
// Contract-free R1 GDN decode step with explicit dims (no ModelContract): conv(k)+SiLU, beta/gate transform,
// q/k L2 norm (q scaled by 1/sqrt(head_dim)), recurrent step, gated RMSNorm. Rows run per row (bitwise
// independent of the window). Buffers are device pointers; qkv = [row][q|k|v] FP32 projection output,
// beta/gate = raw in_proj_b / in_proj_a outputs [row][value_heads], z = [row][value_heads*128].
// Gated-norm kind: sigmoid (Flash-Next) or swish (qwen35: core * rsqrt(..) * w * silu(z)).
// small_type: ggml kinds 0/1/30 = F32/F16/BF16 for conv/dt/a/norm. Home: shared (HIP core), launcher TU.
#include <cstdint>

namespace strix {
enum class HipGdnGatedNorm : std::uint32_t { sigmoid = 0, swish = 1 };
struct HipGdnDimsStep {
    float *qkv, *beta, *gate, *core, *matrix;
    const float* z;
    std::uint16_t* conv_state;  // [channels][kernel] bf16 bits
    const void *conv, *dt, *a, *norm;
    std::uint32_t small_type = 0, key_heads = 16, value_heads = 48, head_dim = 128, kernel = 4, rows = 1;
    float qk_epsilon = 1e-6F, rms_epsilon = 1e-6F;
    HipGdnGatedNorm norm_kind = HipGdnGatedNorm::swish;
};
void hip_gdn_decode_step_dims(const HipGdnDimsStep& step, void* stream);
} // namespace strix
