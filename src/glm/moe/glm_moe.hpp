#pragma once
// GLM-5.3-Flash decode MoE for one TP2 rank (gfx1151), R = 1..8 rows (decode, MTP / speculative verify windows).
//
// Identity (bitwise, design notes section 1.0):
//   router (Tier A)  logits = R1 chain (acc = +0; acc = fmaf(y[i], float(W_f16[e][i]), acc), i ascending) = dev::dense;
//                    R14 / R5: s = sigmoid(logit), top-8 by ordered_key(s + bias) (ties -> lower id), ids ascending,
//                    den = ((s_0 + s_1) + ... + s_7) + 1e-20 over ascending ids, w = (s / den) * 2.5   = dev::route
//   experts (Tier B) g = strix_gemv(gate_e, y), u = strix_gemv(up_e, y)   strix-trellis 5acf090, K4 [4096 -> 1024 slice]
//                    a = R15 swiglu(g, u) (min(g, 10); clamp(u, +-10); silu(g) * u; glm_math.hpp = dev::swiglu)
//                    o_e = strix_gemv(down_e, a)                          K4 [1024 -> 4096 slice]
//                    shared: the same MLP with the shared expert (K6) -> s
//   combine (Tier A) R16: y = +0; for the row's 8 experts in ascending id: y = fmaf(w_e, o_e, y); then y = y + s
//                    (= dev::moe_combine)
// strix_gemv at these shapes (5acf090 backend.hip, gemv_policy.hpp): input transform RN16(H32(RN16(RN32(x * suh)))),
// split core (32 splits for 4096 -> 1024, 4 splits for 1024 -> 4096; per lane chains of v_dot2_f32_f16, 4-lane
// shuffle reduction), output transform acc = ((+0 + p_0) + p_1) + ...; RN32(H32(acc) * svh). The kernels below keep
// every one of those operations and orders; only scheduling, addressing and fusion differ.
// Every row's bits depend only on that row (batch invariance R = 1..8).
//
// The dispatch (distinct experts, their rows, pair indices) is derived inside every CTA from the 8 x rows ids
// (ascending and distinct per row, as R5 produces them); there is no dispatch launch.
//
// enqueue_moe (3 launches, no host synchronisation):
//   1. router + shared gate/up: 36 router CTAs (8 experts each, F16 rows staged through LDS in 512-channel chunks, one
//      chain per (expert, row) lane) and, in the same grid, the shared expert's gate/up CTAs (known before routing, so
//      its K6 stream overlaps the router); the last router CTA does R14/R5 (one wave per row) and writes ids / weights.
//   2. routed gate_up: grid (16-column group x split group, slot, matrix); slots = the distinct routed experts
//      ascending. Each CTA streams its 16 columns over SG library splits (16-byte k-row runs through a wave-private LDS
//      row) and stores the raw split partials; the last CTA per (slot, 128-output block) sums them in ascending split
//      order, H32, svh, R15, and writes the down input transform (FP16).
//   3. down: grid (16-column group, slot incl. shared); each wave runs the whole K (the 4 library splits accumulated in
//      ascending order), H32 + svh in the CTA; the last CTA per 256-output group does R16 for every row.
// enqueue_experts: launches 2 and 3 (shared included in 2) for given ids / weights (e.g. from dev::route).
#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

#include "strix/pair/producer_publish.hpp"

namespace strix::glm::moe {

inline constexpr unsigned kHidden = 4096, kInter = 1024 /* per rank */, kExperts = 288, kTopE = 8, kMaxRows = 8;
inline constexpr unsigned kMaxSlots = 1 + kMaxRows * kTopE;      // shared + distinct routed experts of a call
inline constexpr unsigned kMaxPairs = kMaxRows + kMaxRows * kTopE;  // (slot, row) pairs: shared R + routed 8R

// One EXL3 matrix (rank slice) resident on the device in the file layout (= strix-trellis device layout):
// trellis [input / 16][output / 16][bits * 8] uint32 words, suh [input] FP16, svh [output] FP16.
struct Exl3 {
    const std::uint32_t* trellis = nullptr;
    const std::uint16_t* suh = nullptr;
    const std::uint16_t* svh = nullptr;
    std::uint32_t input = 0, output = 0, bits = 0;
};
struct ExpertSlices {  // gate, up: 4096 -> 1024; down: 1024 -> 4096
    Exl3 gate, up, down;
};
// STRIX_GLM_MODEL_FORMAT=fast_i4 (lossy): the routed experts as the IU4 g64 pack (strix-trellis core/iu4.hpp layout,
// had128-rotated basis; the EXL3 suh / svh stay in ExpertSlices, the routed trellis pointers are null). Per matrix m
// (0 gate, 1 up, 2 down) over the 288 experts: wq [288][K/16][N/16][16][8] nibbles, ws FP16 [288][N][K/64].
struct I4Experts {
    const std::uint8_t* wq[3] = {};
    const std::uint16_t* ws[3] = {};  // wq[2] / ws[2] null: scope gate_up (routed down stays EXL3)
    bool on() const { return wq[0] != nullptr; }
};
bool fast_i4_built();  // the W4A16 routed core is compiled in (STRIX_GLM_PF_IU4_TRELLIS_DIR build)

// One device allocation of moe_workspace_bytes(), zeroed once before first use (its arrival counters reset themselves
// at the end of every call). Calls sharing a workspace must be ordered (one stream, or explicit events): concurrent
// calls on the same workspace are not supported.
struct MoeWorkspace {
    void* base = nullptr;
};
std::size_t moe_workspace_bytes();
// Diagnostics (tests): device pointers into the workspace after a call. Pair p < R: the shared expert of row p; routed
// pairs follow in (expert ascending, row ascending) order. pair_output = the expert's down output (strix_gemv bits),
// pair_down_input = its transformed down input (FP16, the library's input_transform of R15(g, u)).
const float* moe_pair_gate_partials(const MoeWorkspace& ws, unsigned pair);  // [2][32][1024], tests
const float* moe_pair_output(const MoeWorkspace& ws, unsigned pair);
const std::uint16_t* moe_pair_down_input(const MoeWorkspace& ws, unsigned pair);

struct MoeView {
    std::uint32_t rows = 0;                   // 1..8
    const float* y = nullptr;                 // rows x 4096: the FFN input (normed, after hc_pre)
    const std::uint16_t* router_w = nullptr;  // 288 x 4096 F16 (enqueue_moe)
    const float* bias = nullptr;              // 288 e_score_correction_bias (enqueue_moe)
    std::uint32_t* ids = nullptr;             // rows x 8 ascending: written by enqueue_moe, read by enqueue_experts
    float* weights = nullptr;                 // rows x 8 (R14): same
    float* logits = nullptr;                  // optional rows x 288 router logits (enqueue_moe; nullptr = workspace)
    const ExpertSlices* experts = nullptr;    // device array [288] (K4)
    ExpertSlices shared{};                    // K6 (K4 accepted)
    float* out = nullptr;                     // rows x 4096: R16 (this rank's partial, to the TP reduce)
    MoeWorkspace ws{};
    // Tests only: when set, the gate/up finish also writes the two linear outputs (strix_gemv bits of gate and up, after
    // svh, before R15) per pair: [pair][0 gate / 1 up][1024]. nullptr in production (no extra stores).
    float* debug_gate_up = nullptr;
    // fast_i4: routed gate/up/down on the W4A16 core (int4 weights, FP16 activations, F32 accumulation, per-group
    // scale); the shared expert, router, transforms, R15 and R16 stay as above. Not bitwise to the EXL3 identity.
    I4Experts i4{};
    // Producer publication (pair decode): `out` is the channel's TX slot and the down kernel raises the TX flag after its
    // last R16 store (strix/pair/producer_publish.hpp). Off (default): nothing is published.
    strix::pair::ProducerPublish publish{};
};
// Both throw std::invalid_argument on unsupported shapes (shared other than K4/K6 at the GLM shapes, rows outside
// 1..8, null pointers). Routed expert slices must be K4 at the GLM shapes (device array, not checked on the host).
// marks (optional, profiling): enqueue_moe records 2 events (after launch 1, after launch 2), enqueue_experts 1.
void enqueue_moe(const MoeView& v, hipStream_t stream, hipEvent_t* marks = nullptr);
void enqueue_experts(const MoeView& v, hipStream_t stream, hipEvent_t* marks = nullptr);

// Scheduling-only opt-in variants. Existing entrypoints always launch the old kernels.
// The engine selects these only for decode when STRIX_GLM_MOE_PIPE=1 (read once).
bool pipe_enabled();
void enqueue_moe_pipe(const MoeView& v, hipStream_t stream, hipEvent_t* marks = nullptr);
void enqueue_experts_pipe(const MoeView& v, hipStream_t stream, hipEvent_t* marks = nullptr);

// Tunable (bitwise at every value; environment, read once): STRIX_GLM_MOE_SG = library splits per gate_up CTA
// (1, 2, 4, 8) for every row count; unset = default_split_group(row bucket), measured: SG1 for R <= 2, SG2 above.
constexpr unsigned default_split_group(unsigned max_rows) { return max_rows <= 2 ? 1U : 2U; }

}  // namespace strix::glm::moe
