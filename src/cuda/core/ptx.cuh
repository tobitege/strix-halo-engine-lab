// Shared CUDA core: PTX wrappers for async copies, ldmatrix and the INT8 tensor-core MMA (sm_80+; the same
// instructions on sm_86, sm_89 and sm_120). One definition each, byte-for-byte the bodies the model files carried
// before the backend move; internal linkage (unnamed namespace) as before, so every TU still compiles its own copy.
//
// Two orderings exist and are kept apart on purpose (the "memory" clobber changes what the compiler may move across
// the copy, not the copy itself): core::ptx has no clobber (ordering by the caller's __syncthreads / wait_group),
// core::ptx_mem carries the "memory" clobber. Pick the one the kernel was written and checked with.
#pragma once

#include <cuda_runtime.h>

#include <cstdint>

namespace strix::cuda::core::ptx {
namespace {

__device__ __forceinline__ unsigned smem_addr(const void* p) { return static_cast<unsigned>(__cvta_generic_to_shared(p)); }

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    const auto d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src));
}
__device__ __forceinline__ void cp_async4(void* dst, const void* src) {
    const auto d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(d), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(p)));
}

// D = A (16x32 s8, row) x B (32x8 s8, col), zero accumulator in, s32 out (exact).
__device__ __forceinline__ void mma_s8(const uint4& a, int b0, int b1, int c[4]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a.x), "r"(a.y), "r"(a.z), "r"(a.w), "r"(b0), "r"(b1), "r"(0));
}

// D += A (16x16 f16, row) x B (16x8 f16, col), FP32 accumulation in place (sm_80+; FAST forms only: the
// accumulation order inside the instruction is the hardware's, so callers own a named numerical identity).
__device__ __forceinline__ void mma_f16(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                        unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// D += A (16x16 bf16, row) x B (16x8 bf16, col), FP32 accumulation in place.
__device__ __forceinline__ void mma_bf16(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// mma_s8 with a caller-chosen accumulator seed (every d[i] starts at `seed`; exact integer arithmetic).
__device__ __forceinline__ void mma_s8_seeded(int (&d)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                              unsigned b1, int seed) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(seed));
}

// ldmatrix .trans: the 8x8 b16 tiles transposed (B operands stored k-contiguous by row).
__device__ __forceinline__ void ldsm_x4_trans(unsigned (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(smem_addr(p)));
}
__device__ __forceinline__ void ldsm_x2(unsigned (&r)[2], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(smem_addr(p)));
}

}  // namespace
}  // namespace strix::cuda::core::ptx

namespace strix::cuda::core::ptx_mem {
namespace {

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
    const auto d = static_cast<std::uint32_t>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(d), "l"(src) : "memory");
}
// src_bytes < 16 zero-fills the rest of the 16-byte destination.
__device__ __forceinline__ void cp_async16(void* dst, const void* src, unsigned src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(ptx::smem_addr(dst)), "l"(src), "r"(src_bytes) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

}  // namespace
}  // namespace strix::cuda::core::ptx_mem
