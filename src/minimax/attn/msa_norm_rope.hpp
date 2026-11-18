#pragma once
// MiniMax MSA per-head query / key norm + RoPE, one wave per head row (lane l = dims 4l..4l+3). Shared by the attention
// kernels (msa_fused.hip) and by the q projection's finish (mmx_prefill.hip dense_finish_kernel, the BF16 query image
// read by the flash core's query_op): one definition, so both compile the same arithmetic (same fmaf chain, same
// butterfly order, same RoPE expression) and the image has the bits the flash prologue computed.
#include "strix/minimax_msa.hpp"

#include <hip/hip_runtime.h>

namespace strix::minimax::msa {

__device__ __forceinline__ float butterfly_sum(float v) {
    for (unsigned d = 16; d >= 1; d >>= 1U) v = v + __shfl_xor(v, static_cast<int>(d), 32);
    return v;
}

// Lane l holds dims 4l..4l+3 of one head in x[4]. Norm + RoPE in place (host head_norm_rope order); identical to the
// K1 helper.
__device__ __forceinline__ void norm_rope_lane(float* x, const float* w, const RopeRow& rope, unsigned lane) {
    float acc = 0.0F;
    for (unsigned i = 0; i < kDimsPerLane; ++i) acc = __builtin_fmaf(x[i], x[i], acc);
    const float sum = butterfly_sum(acc);
    const float r = 1.0F / __builtin_sqrtf(sum / static_cast<float>(kHeadDim) + kNormEps);
    for (unsigned i = 0; i < kDimsPerLane; ++i) x[i] = (x[i] * r) * (1.0F + w[lane * kDimsPerLane + i]);
    float partner[kDimsPerLane];
    for (unsigned i = 0; i < kDimsPerLane; ++i) partner[i] = __shfl_xor(x[i], 8, 32);
    if (lane < 8) {
        for (unsigned i = 0; i < kDimsPerLane; ++i) {
            const unsigned d = lane * kDimsPerLane + i;
            x[i] = __builtin_fmaf(-partner[i], rope.sin[d], x[i] * rope.cos[d]);
        }
    } else if (lane < 16) {
        for (unsigned i = 0; i < kDimsPerLane; ++i) {
            const unsigned d = (lane - 8) * kDimsPerLane + i;
            x[i] = __builtin_fmaf(partner[i], rope.sin[d], x[i] * rope.cos[d]);
        }
    }
}

}  // namespace strix::minimax::msa
