#pragma once
// Wave-uniform operands for serial-scan and row-epilogue kernels.
//
// A value derived from threadIdx is per-lane to the compiler even when every lane of a wave holds the same number
// (the wave index of a CTA, a row the whole wave works on). Row bases built from it (row * stride + head offset) then
// become 64-bit per-lane address chains: two vector registers per address stream, kept live across the kernel's main
// loop next to its recurrent state, which is what pushes a scan kernel into register spills.
//
// Policy: route such a value through wave_uniform() once, outside the loop. The row and its address bases are then
// scalar-register arithmetic and each access is `scalar base + 32-bit lane offset`; branches on the row are scalar.
// The numbers are unchanged, so results are bitwise identical.
//
// Precondition (the caller's layout contract): all active lanes of the wave hold the same value. Kernels whose waves
// are 32 consecutive threads of the CTA satisfy it for threadIdx.x / 32.
namespace strix::core {
__device__ __forceinline__ unsigned wave_uniform(unsigned v) {
    return static_cast<unsigned>(__builtin_amdgcn_readfirstlane(static_cast<int>(v)));
}
__device__ __forceinline__ unsigned wave_index(unsigned thread) { return wave_uniform(thread / 32U); }
// Element pointer of a wave-uniform row: base + index as an integer sum, so the optimiser cannot reassociate a later
// `+ lane` onto `base` and hoist a per-lane 64-bit pointer out of the loop (one such pointer per output stream is what
// stays live across a scan kernel's chunk loop otherwise). Use as wave_row(base, scalar_index)[lane].
template <class T> __device__ __forceinline__ T* wave_row(T* base, unsigned long long index) {
    return reinterpret_cast<T*>(reinterpret_cast<unsigned long long>(base) + index * sizeof(T));
}
} // namespace strix::core
