#pragma once

// Test-only diagnostic: per-thread trace of the per-group partial contributions of the GSQ IQ bodies
// (src/hip_moe_q4k_a8_known_rows.hip project_iq_body / down_iq_body), each recorded where iq_apply returns,
// before it is added to the accumulator. Compiled only when STRIX_WFMT_CAPTURE is set (the wfmt fixture target);
// production objects never see this header. The fixture owns the buffers (hipMemcpyToSymbol); null disables.
#if defined(STRIX_WFMT_CAPTURE) && STRIX_WFMT_CAPTURE
#include <hip/hip_runtime.h>

namespace strix::wfmt_trace {
inline constexpr unsigned kSlots = 512U;
__device__ float* gsq_values;
__device__ unsigned* gsq_counts;
__device__ __forceinline__ void gsq_partial(float value) {
    if (!gsq_values) return;
    const unsigned t = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned i = gsq_counts[t]++;
    if (i < kSlots) gsq_values[static_cast<unsigned long long>(t) * kSlots + i] = value;
}
}  // namespace strix::wfmt_trace
#endif
