#pragma once
// GLM prefill FAST identity glm-pf-mla-wmma-v1: the two per-head MLA GEMMs of a chunk, absorb (q_nope x W_UK -> the
// 512-wide latent query) and out (o_bar x W_UV -> the 256-wide head output), on the shared short-K FP16 WMMA tile GEMM
// (core/f16_tile_gemm.hpp) instead of the exact FP32 VOPD chains of mla_gemm_rows_v3. Automatic when FAST is selected;
// STRIX_GLM_PF_MLA_WMMA overrides it (glm_pf_wmma.hpp mla_wmma::enabled). Contract: per output one WMMA chain from +0 over the
// 16-wide k units ascending, sum_k fp16(x) * w with FP32 accumulation; softmax, scores and o-bar are unchanged.
// kv_b of a head h: rows [h * 2 * Dim, h * 2 * Dim + Dim) = W_UK as [k = 0..Dim)[c = 0..Lora) (Layout::KN, the core
// transposes it in LDS, no weight copy), the next Dim rows = W_UV as [v = 0..Dim)[c = 0..Lora) (Layout::NK).
#include "core/f16_tile_gemm.hpp"
#include "glm/prefill/glm_prefill.hpp"
#include "glm/prefill/glm_pf_wmma.hpp"

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

namespace strix::glm::pf::mla_wmma {
namespace ftg = strix::core::f16_tile_gemm;

template <unsigned Lora, unsigned Dim, bool kOut>
inline ftg::Args args(const float* x, const std::uint16_t* kv_b, unsigned heads, unsigned C, float* y) {
    constexpr unsigned K = kOut ? Lora : Dim, N = kOut ? Dim : Lora;
    ftg::Args p{};
    p.a = x;  // [C][heads][K]
    p.a_row = std::size_t{heads} * K;
    p.a_batch = K;
    p.w = kv_b + (kOut ? std::size_t{Dim} * Lora : 0U);
    p.w_ld = Lora;
    p.w_batch = std::size_t{2} * Dim * Lora;
    p.y = y;  // [C][heads][N]
    p.y_row = std::size_t{heads} * N;
    p.y_batch = N;
    p.rows = C;
    p.K = K;
    p.N = N;
    p.batches = heads;
    p.order = 1;  // heads of a row tile in flight: the [C][H][K] rows sit at a 32 / 64 KiB stride (channel camping)
    return p;
}

// y = per-head GEMM of x (absorb: kOut = false, out: kOut = true) for C rows. image (absorb only): y receives the bf16
// q-tilde image [C][heads][Lora] u16 (glm_prefill.hip fused core 22 / 23) instead of FP32 sums.
template <unsigned Lora, unsigned Dim, bool kOut>
// a16: x holds FP16 bits (the o-bar image the flash core stored), same element strides.
inline void gemm(const float* x, const std::uint16_t* kv_b, unsigned heads, unsigned C, float* y, hipStream_t s, bool image = false,
                 bool a16 = false) {
    auto p = args<Lora, Dim, kOut>(x, kv_b, heads, C, y);
    if (a16) p.a16 = reinterpret_cast<const std::uint16_t*>(x);
    if constexpr (kOut) ftg::gemm<ftg::Layout::NK>(p, s);
    else if (image) ftg::gemm_bf16<ftg::Layout::KN>(p, s);
    else ftg::gemm<ftg::Layout::KN>(p, s);
}
template <unsigned Lora, unsigned Dim>
inline bool fits(const float* q, const float* ob, const std::uint16_t* kv_b, unsigned heads, unsigned C) {
    return C >= 1U && ftg::fits(args<Lora, Dim, false>(q, kv_b, heads, C, nullptr)) &&
           ftg::fits(args<Lora, Dim, true>(ob, kv_b, heads, C, nullptr));
}

}  // namespace strix::glm::pf::mla_wmma

namespace strix::glm::pf {
// Explicit arithmetic/image/placement selector for FAST gates. Existing versions APIs remain exact.
void mla_attention_rows_selected(const dev::DsaSlot& slot, const std::uint32_t* tokens, const std::uint32_t* n,
    const float* q, const std::uint16_t* kv_b, unsigned heads, float* qt, float* e, float* ob, float* o,
    unsigned C, hipStream_t s, unsigned scores_v, unsigned obar_v, unsigned gemm_v, unsigned query_window,
    mla_gemm::Dispatch dispatch);
}  // namespace strix::glm::pf
