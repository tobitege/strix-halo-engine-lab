#pragma once

// Test-only diagnostic: per-thread trace of the EXL3/trellis stages of the shared HIP helpers
// (src/moe_core/exl3_trellis.hpp, the MiniMax and GLM prefill epilogues finish_block). Four streams, each recorded where
// the incumbent computes the value and never altering it: the transformed input values, the decoded tile words, the
// 4-lane reduced split partials and the finished (H128, svh) outputs. Compiled only when STRIX_WFMT_CAPTURE is set (the
// wfmt fixture target); production objects never see this header. The fixture owns the buffers (hipMemcpyToSymbol);
// null disables.
#if defined(STRIX_WFMT_CAPTURE) && STRIX_WFMT_CAPTURE
#include <hip/hip_runtime.h>

namespace strix::wfmt_trace {
enum TrellisStream : unsigned { kTrellisInput = 0, kTrellisDecode = 1, kTrellisPartial = 2, kTrellisFinish = 3, kTrellisStreams = 4 };
inline constexpr unsigned kTrellisSlots = 256U;
__device__ unsigned* trellis_values;
__device__ unsigned* trellis_counts;
__device__ __forceinline__ void trellis_trace(unsigned stream, unsigned bits) {
    if (!trellis_values) return;
    const unsigned t = ((blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    const unsigned c = t * kTrellisStreams + stream;
    const unsigned i = trellis_counts[c]++;
    if (i < kTrellisSlots) trellis_values[static_cast<unsigned long long>(c) * kTrellisSlots + i] = bits;
}
__device__ __forceinline__ void trellis_trace(unsigned stream, float v) { trellis_trace(stream, __float_as_uint(v)); }
__device__ __forceinline__ void trellis_trace(unsigned stream, _Float16 v) {
    trellis_trace(stream, static_cast<unsigned>(__builtin_bit_cast(unsigned short, v)));
}
}  // namespace strix::wfmt_trace
#endif
