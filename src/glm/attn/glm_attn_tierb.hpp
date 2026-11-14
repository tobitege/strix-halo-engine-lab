#pragma once
// GLM-5.3-Flash attention Tier B (EXL3 K6 mul1 linears), fused dispatches (
// design notes). Every output element has the bits of strix-trellis 5acf090 strix_gemv on the same member
// (R <= 8): input transform RN16(H32(RN16(RN32(x * suh)))), the library split count (measured_vector_splits) and
// split boundaries, per-lane v_dot2_f32_f16 chains in the library order, the 4-lane shuffle tree, the ascending split
// sum from +0, H32, x svh (alpha 1, beta 0, no bias). Only scheduling, addressing, launch fusion and output placement
// differ: one launch per group, the input transform inside the CTAs, the output transform in the last-arriving CTA of
// each 128-output block, outputs written at any row stride (the KDA q|k|v land directly in [rows][3C]).
// Every row's bits depend only on that row (batch invariance R = 1..8). Build with the engine flags (-ffp-contract=off).
#include <hip/hip_runtime.h>

#include "strix/pair/producer_publish.hpp"

#include <cstddef>
#include <cstdint>

namespace strix::glm::attn_tierb {

inline constexpr unsigned kMaxRows = 8;

// One EXL3 matrix (rank slice) in the strix-trellis device layout (= file layout): trellis [input/16][output/16][48]
// uint32 words (16-byte aligned), suh [input] FP16, svh [output] FP16. Same fields as engine Exl3Raw.
struct Mat {
    const std::uint32_t* trellis = nullptr;
    const std::uint16_t* suh = nullptr;
    const std::uint16_t* svh = nullptr;
    std::uint32_t input = 0, output = 0, bits = 0;
};

// The groups (shapes per TP2 rank; every member K6):
//   KdaQkv : q, k, v 4096 -> 4096 (S 16), one input row, outputs at y + p * 4096 with row stride y_stride
//   KdaO   : 4096 -> 4096 (S 16)
//   DsaA   : q_a 4096 -> 1536 (S 16) + kv_a 4096 -> 512 (S 32), one input row
//   DsaB   : q_b 1536 -> 8192 (S 4) + wq_b 1536 -> 4096 (S 4), one input row (qr)
//   DsaO   : 8192 -> 4096 (S 16)
enum class Group : unsigned { KdaQkv, KdaO, DsaA, DsaB, DsaO };

// Shape check only: true when the matrices have the group's exact shapes, bits 6, a 16-byte aligned trellis and 2-byte
// aligned suh / svh; else use strix_gemv. The calls below also check rows (1..8), x / y alignment (4 B), strides and the
// workspace, and throw std::invalid_argument when a check fails.
bool supported(Group g, const Mat* m, unsigned count);

// One device buffer of workspace_bytes(), 256-byte aligned (hipMalloc), zeroed once (hipMemset) before the first call;
// its arrival counters reset themselves at the end of every call. Calls sharing a workspace must be stream-ordered.
std::size_t workspace_bytes();
// Test hooks: byte offset and count of the uint32 arrival counters inside the workspace (all zero between calls).
std::size_t workspace_counters_offset();
unsigned workspace_counters();

// Schedule knobs (bench / sweeps; every value gives the same bits). 0 = the default for the group and rows.
struct Knobs {
    unsigned sg = 0;     // library splits per CTA: 1, 2 or 4
    int prime = -1;      // 1: issue the first weight rows before the in-CTA input transform; 0: after; -1 default
    unsigned tpw = 0;    // 16-column tiles per wave: 4 (256-output CTAs) or 2 (128-output CTAs)
};
// Environment: STRIX_GLM_TIERB_SG, STRIX_GLM_TIERB_PRIME, STRIX_GLM_TIERB_TPW (read once). With none set, the default
// is the measured per-(group, rows) table in the .hip (exact: it picks among gated knobs); STRIX_GLM_TIERB_TABLE=0 = legacy.
Knobs env_knobs();
// Private (scratch) bytes per lane of the kernel instantiation the call would launch for (group, rows, knobs); the
// bench and the defaults exclude any configuration with scratch.
std::size_t scratch_bytes(Group g, unsigned rows, Knobs k);

// rows 1..8; x rows have stride x_stride floats (>= input), 4-byte aligned; outputs as stated.
void kda_qkv(const Mat (&m)[3], const float* x, unsigned x_stride, unsigned rows, float* y, unsigned y_stride, void* ws,
             hipStream_t s, Knobs k = env_knobs());
// pub (o projections): y is the exchange TX slot holding the whole partial; the kernel publishes it after its last store.
void kda_o(const Mat& m, const float* x, unsigned x_stride, unsigned rows, float* y, unsigned y_stride, void* ws, hipStream_t s,
           Knobs k = env_knobs(), const strix::pair::ProducerPublish* pub = nullptr);
void dsa_a(const Mat& q_a, const Mat& kv_a, const float* x, unsigned x_stride, unsigned rows, float* qa, unsigned qa_stride,
           float* kva, unsigned kva_stride, void* ws, hipStream_t s, Knobs k = env_knobs());
void dsa_b(const Mat& q_b, const Mat& wq_b, const float* qr, unsigned qr_stride, unsigned rows, float* q, unsigned q_stride,
           float* qI, unsigned qI_stride, void* ws, hipStream_t s, Knobs k = env_knobs());
void dsa_o(const Mat& m, const float* x, unsigned x_stride, unsigned rows, float* y, unsigned y_stride, void* ws, hipStream_t s,
           Knobs k = env_knobs(), const strix::pair::ProducerPublish* pub = nullptr);

// Per-(group, rows) knob table (default on; STRIX_GLM_TIERB_TABLE=0 off) switched at runtime (A/B in one process).
void set_table_enabled(bool on);

}  // namespace strix::glm::attn_tierb
