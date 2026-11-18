#pragma once
// minimax-m3 prefill MoE (gfx1151, one TP2 rank): every routed expert and the shared expert of a T-row chunk in a
// fixed number of grouped launches, instead of 3 strix_gemm (x 3 launches) + gather/scatter per expert.
//
// Identity (bitwise): per expert, the bring-up chain of tests/hip_minimax_g4.hip for prefill chunks, i.e. strix_gemm
// (strix-trellis 0.3, WMMA core at every row count) on the expert's gathered rows (ascending token order), swiglu,
// strix_gemm down, and per token acc = 0; acc = fmaf(w_e, y_e, acc) over its experts in ascending id; acc + shared.
// Each grouped CTA runs the library's wmma_core for one (expert, 4 tile columns, row tile, split) with the library's
// split and tile policy for that expert's row count (split_count / wmma_tile_rows, ported verbatim), and the
// partials are reduced in ascending split order before the output Hadamard, as output_transform does.
//
// Launches: gather+transform, gate/up (grouped WMMA), gate/up finish + SwiGLU + down transform, down (grouped WMMA,
// one launch per row-tile class), down finish + combine. Routes come from the host (as in G4; one D2H per chunk), or
// stay on the device with the device plan (enqueue_prefill_moe_device).
#include "minimax/moe/mmx_moe.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <hip/hip_runtime.h>

namespace strix::minimax::moe::opt {

// FAST_I4 (STRIX_MMX_MODEL_FORMAT=fast_i4; lossy identity minimax-fast-i4): one MoE layer's 128 routed experts and its
// shared expert (index kExperts) as the shared core's IU4 g64 pack in the had128-rotated EXL3 basis (core/
// activation_quant.hpp). Per matrix m (0 gate, 1 up, 2 down) over kI4Experts experts, one kI4ExpertBytes block per expert:
// nibbles [K/16][N/16][16][8] at wq[m] + e * kI4ExpertBytes, FP16 scales [N][K/64] right after them (ws[m] + e *
// kI4ExpertBytes / 2); svh = device table [e] of the EXL3 output-sign vectors (the epilogue's had128 * svh); the EXL3
// suh / svh themselves stay in ExpertSlices. Two reusable slots hold per-layer prefill INT4;
// decode always uses the original EXL3 slices, including every MTP row regime.
inline constexpr unsigned kI4Experts = kExperts + 1;
inline constexpr std::size_t kI4GroupTableBytes = (kI4Experts * 3U * sizeof(std::uint32_t) + 255U) & ~std::size_t{255};
inline constexpr std::size_t kI4Nibbles = std::size_t{kHidden} * kInter / 2U;
inline constexpr std::size_t kI4ExpertBytes = kI4Nibbles + std::size_t{kInter} * (kHidden / 64U) * 2U;
struct I4Layer {
    std::uint8_t* wq[3] = {};
    _Float16* ws[3] = {};
    const std::uint16_t* const* svh[3] = {};
    bool on() const { return wq[0] != nullptr; }
};
inline constexpr std::size_t kI4PackBytes = 3U * kI4Experts * kI4ExpertBytes;  // one layer's pack (gate, up, down)
// Device pointer tables of one layer's EXL3 experts (converter input): trellis[m][bits] = per-expert tiles of matrix m
// with that width (nullptr entries for the other widths; table nullptr when no expert has it), svh[m] = per-expert svh.
struct I4Source {
    const std::uint32_t* const* trellis[3][9] = {};
    const std::uint16_t** svh[3] = {};
};
bool iu4_built();                 // the grouped IU4 core is compiled in (HIP runtime build, strix_tiled_core)
std::size_t i4_pack_bytes();      // kI4PackBytes
std::size_t i4_source_bytes();    // device bytes of one I4Source (256-byte aligned base)
// Validates the layer (every matrix 6144 x 1536 / 1536 x 6144, 2..8 bits, resident) and uploads its tables (sync).
I4Source i4_source_build(const ExpertSlices* experts, const ExpertSlices& shared, void* base);
// The pack at `pack` (i4_pack_bytes(), 256-byte aligned) as the layer view of `src`.
I4Layer i4_layer_view(std::uint8_t* pack, const I4Source& src);
// Slot storage includes three reusable svh device tables after the packed weights/scales.
std::size_t i4_slot_bytes();
I4Layer i4_slot_view(std::uint8_t* base);
// The shared core converter (strix::core::convert_exl3_iu4); asynchronous on stream, EXL3 is read-only. gate_up
// scope skips down conversion. The slot's svh tables are refreshed on the same stream.
void i4_convert(const I4Source& src, const I4Layer& out, hipStream_t stream, bool all = true);
// Prefill MoE of an I4 layer: -1 = STRIX_MMX_PF_IU4 (default 1 = on whenever the view carries a layer), 0 = off (the
// exact EXL3 path, only while the EXL3 tiles are resident), 1 = on. STRIX_MMX_IU4_PLAN = library plan 0 | 1 | 2.
void set_prefill_iu4(int on);
bool prefill_iu4();
// STRIX_MMX_FAKEQ_A4=1 (quality study only, default off): the exact v4 path with the gate/up and down inputs
// fake-quantised to A4 g64 after their input transforms (iu4_fake128); EXL3 weights unchanged.
bool prefill_fakeq_a4();
// Per-call shadow controls; environment preserves each planner's existing behavior.
enum class PrefillFakeq { environment, off, on };

struct PrefillMoeView {
    std::uint32_t rows = 0;                     // T
    const float* y = nullptr;                   // T x 6144 normed hidden (device)
    const std::uint32_t* ids = nullptr;         // T x 4 routes (host), route_row order
    const float* weights = nullptr;             // T x 4 (host)
    const ExpertSlices* experts = nullptr;      // host array [128] of device slices
    ExpertSlices shared{};
    float* out = nullptr;                       // T x 6144 (device)
    const I4Layer* i4 = nullptr;                // FAST_I4 layer pack (null: EXL3 path)
    bool i4_all = true;                         // false: IU4 gate/up, original EXL3 down
    unsigned i4_a8 = 0;                         // W4A8 planes: bit 0 gate/up input, bit 1 down input (unfused path only)
    bool i4_chain = false;                      // fused candidate chain (prefill_i4_chain of the prompt)
    PrefillFakeq fakeq_a4 = PrefillFakeq::environment;
};

// Dense prefill linears on the v4 core: jobs share one input x [rows][input] (input % 128 == 0, output % 128 == 0)
// and each writes out [rows][output] fp32 with the library's gemm bits (input transform, gemm_splits split-K WMMA
// chain, ordered split sum + output transform). Supported bit widths: 3, 4, 5, 6, 8 (prefill_dense_supported). Item
// lists are staged through a small pinned ring and copied on the stream into the workspace; cleanup_prefill_dense()
// frees the ring.
inline constexpr unsigned kMaxDenseJobs = 4;
struct PrefillDenseJob {
    Exl3 w{};
    float* out = nullptr;
    // Optional query image (MSA FAST flash layers; output % 128 == 0, 128-column heads): instead of out, the BF16
    // image of each head after msa::norm_rope_lane (norm weights image_w [128], RoPE row image_rope[row]) as the flash
    // core's query_op reads it, and with image_lo its lo terms (mode 12). Same bits as the flash prologue on out.
    std::uint16_t* image = nullptr;
    std::uint16_t* image_lo = nullptr;
    const float* image_w = nullptr;
    const void* image_rope = nullptr;  // const msa::RopeRow* (strix/minimax_msa.hpp)
};
bool prefill_dense_supported(const Exl3& w);
std::size_t prefill_dense_workspace_bytes(const PrefillDenseJob* jobs, unsigned count, unsigned rows);
void enqueue_prefill_dense(const PrefillDenseJob* jobs, unsigned count, const float* x, unsigned rows, void* workspace,
                           std::size_t workspace_bytes, hipStream_t stream);
void cleanup_prefill_dense() noexcept;

// Device workspace for one call (depends on the routes: split counts follow each expert's row count).
std::size_t prefill_moe_workspace_bytes(const PrefillMoeView& v);
// Upper bound for any routes of T rows.
std::size_t prefill_moe_workspace_bound(std::uint32_t rows);
void enqueue_prefill_moe(const PrefillMoeView& v, void* workspace, std::size_t workspace_bytes, hipStream_t stream);
// Device plan (default; STRIX_MMX_PF_PLAN=host keeps the host plan above): the same tables and launches built on the
// GPU from the device routes by the shared moe_core::device_plan, so a call needs no route download, host sync, host
// table build or upload. Selected with mtfit and without gufuse (the defaults); the workspace is
// prefill_moe_workspace_bound(rows). Same bits as enqueue_prefill_moe.
struct PrefillMoeDeviceView {
    std::uint32_t rows = 0;                 // T
    const float* y = nullptr;               // T x 6144 normed hidden (device)
    const std::uint32_t* routes = nullptr;  // device: T x 4 expert ids, then T x 4 float weights (the router layout)
    const ExpertSlices* experts = nullptr;  // device array [128]
    ExpertSlices shared{};
    float* out = nullptr;                   // T x 6144 (device)
    const I4Layer* i4 = nullptr;
    bool i4_all = true;
    bool i4_chain = false;
    PrefillFakeq fakeq_a4 = PrefillFakeq::environment;
};
bool prefill_moe_device_plan(std::uint32_t rows);
// Group-wide device-planned call: the MoE of one layer over all chunks of a prefill group at once (chunk c = rows
// [c x seg_rows, ...) of the call, every chunk but the last full). An expert's weights are decoded once for the
// group's rows instead of once per chunk; the plan keeps each row's per-chunk K splits (rows merge per expert and split
// class), so every chunk's output is bitwise that of its own enqueue_prefill_moe_device call. Every chunk needs at
// least prefill_moe_group_min_rows() rows; the workspace is prefill_moe_group_workspace_bound(rows). EXL3 only.
inline constexpr unsigned kPrefillMoeSegments = 8;
struct PrefillMoeGroupView {
    std::uint32_t rows = 0, segments = 0, seg_rows = 0;
    const float* y[kPrefillMoeSegments] = {};               // per chunk: rows x 6144 normed hidden (device)
    const std::uint32_t* routes[kPrefillMoeSegments] = {};  // per chunk: the router layout of PrefillMoeDeviceView
    float* out[kPrefillMoeSegments] = {};                   // per chunk: rows x 6144 (device)
    const ExpertSlices* experts = nullptr;                  // device array [128]
    ExpertSlices shared{};
};
unsigned prefill_moe_group_min_rows();
std::size_t prefill_moe_group_workspace_bound(std::uint32_t rows);
void enqueue_prefill_moe_group(const PrefillMoeGroupView& v, void* workspace, std::size_t workspace_bytes, hipStream_t stream);
// The fused FAST_I4 candidate chain for a prompt of total_rows rows (0: capability/opt-in checks). Default on at
// every length (8K -1.7 %, 32K -0.9 % with the global gate save). STRIX_MOE_I4_CHAIN=0/1 overrides. Same bits.
bool prefill_i4_chain(std::size_t total_rows = 0);
std::size_t prefill_i4_workspace_bound(std::uint32_t rows);
void enqueue_prefill_moe_device(const PrefillMoeDeviceView& v, void* workspace, std::size_t workspace_bytes,
                                hipStream_t stream);
// After the stream completed: throws (once) when a device-planned call met invalid routes (an id >= 128 or repeated
// in a row; such a row was planned as experts 0..3 with weight 0). A failed exchange is the only known source.
void prefill_moe_device_check();
void cleanup_prefill_device_plan() noexcept;
// v4 exact-fit row tiles; -1 = STRIX_MMX_PREFILL_MTFIT (default off), 0 = off, 1 = on.
void set_prefill_mtfit(int on);
// Pinned H2D table ring; -1 = STRIX_MM_PF_TABLE_RING (default off), 0 = off, 1 = on.
// Single host thread per process; order reuse of each device workspace on the same stream.
void set_prefill_table_ring(int on);
// Wait for table copies and release pinned slots/events before HIP runtime teardown.
void cleanup_prefill_table_ring() noexcept;
// v4 single-split gate/up epilogue fusion; -1 = environment (default off), 0 = off, 1 = on.
void set_prefill_gufuse(int on);
// v4 token-major gather (one input read per token); -1 = environment (default on), 0 = off, 1 = on. Same bits.
void set_prefill_gather_tok(int on);
// Tests only: materialize fused gate/up partials. Optional device buffer [4][pairs][1536]
// captures finished gate, up, SwiGLU and FP16-rounded down input in both v4 arms.
// The caller keeps the buffer alive until the stream completes; nullptr disables capture.
// producer_only writes partials and uses the original finish kernel for GEMM timing.
void set_prefill_gufuse_debug(bool on, float* intermediates = nullptr, bool producer_only = false);
// PIPE is opt-in; -1 restores the environment. Getter preserves explicit overrides across A/B runs.
void set_prefill_pipe(int value);
int prefill_pipe_override();
// Under PIPE: paired MT8 bands. -1 = STRIX_MMX_PREFILL_PIPE_PAIR (default auto: calls >= 2048 rows), 0 = off,
// 1 = every call, 2 = auto.
void set_prefill_pipe_pair(int value);
// Under PIPE: the MT8 class on 8 waves of one output tile each (in place of pairs and paired bands). -1 =
// STRIX_MMX_PREFILL_PIPE_N16 (default off), 0 = off, 1 = on, 2 = auto (the calls without paired bands). Same bits.
void set_prefill_pipe_n16(int value);
// Dense linears under PIPE: tall bands. -1 = STRIX_MMX_PREFILL_DENSE_BAND (default 0 = off), 0 = off, 4 / 8 = bands of
// 4 / 8 waves over 512 / 1024 rows.
void set_prefill_dense_band(int value);
// v4 + mtfit tail classes beside the MT8 launch on a side stream: -1 = STRIX_MMX_PREFILL_TAIL_STREAM (default on),
// 0 = off, 1 = on. cleanup_ releases the side stream and its events (Engine teardown).
void set_prefill_tail_stream(int on);
// MoE numeric order contract: -1 = STRIX_MMX_PF_ORDER (default adaptive), 0 = adaptive (S from the expert's rows in
// the chunk, today's identity), 1 = invariant (gate/up/down S = 1: bits independent of chunking, identity
// mmx-pf-moe-s1-v1). Host and device plans and the group-wide plan all follow it. The same switch binds the dense v4
// linears (enqueue_prefill_dense) to core::fixed_linear_contract (S = kDenseFixedSplits); MoE + dense together are the
// named identity mmx-pf-linear-s1-v1.
// Test fixture only: one host thread, all device work synchronized before each
// change, restore -1 before leaving the fixture. Never change a running engine.
void set_prefill_moe_order(int invariant);
bool prefill_order_invariant();
inline constexpr int kDenseFixedSplits = 1;
// Split of one dense v4 job: the fixed contract under the invariant order, else gemm_splits (the library's).
unsigned prefill_dense_splits(unsigned input, unsigned output, unsigned bits, unsigned rows);
void cleanup_prefill_tail_stream() noexcept;
void set_prefill_core8(int value);  // -1 = environment, 0 = off (default), 1 = A LDS, 2 = shared B
// Profiling only: `marker(label, stream, user)` after "start" and each launch ("gather", "gate_up[.mtN]", "finish",
// "down[.mtN]", "combine"); nullptr = off.
using PrefillMarker = void (*)(const char* label, hipStream_t stream, void* user);
void set_prefill_marker(PrefillMarker marker, void* user);
// Tests: workspace regions of one call (transformed gate/up rows, down input, all split partials), as byte ranges.
// The core stores xt / at tile-major [2][tpairs/16][384][16][16] / [tpairs/16][96][16][16] with group g's rows at
// tbegin.. (xt_bytes / at_bytes cover the row-major extent [2][pairs][6144] / [pairs][1536]).
struct PrefillDebugRegions {
    std::size_t xt_offset = 0, xt_bytes = 0, at_offset = 0, at_bytes = 0, part_offset = 0, part_bytes = 0;
    std::size_t pairs = 0, tpairs = 0;
    std::vector<std::array<unsigned, 3>> groups;  // pair_begin, tbegin, rows
};
PrefillDebugRegions prefill_moe_debug_regions(const PrefillMoeView& v);

// strix-trellis 0.3 policy for a mul1 matrix at `rows` rows through strix_gemm (prefer_wmma): split count and WMMA
// row tile (backend.hip split_count, launch_policy.hpp wmma_tile_rows, gemv_policy.hpp measured_vector_splits).
unsigned gemm_splits(unsigned input, unsigned output, unsigned bits, unsigned rows);
unsigned gemm_tile_rows(unsigned input, unsigned output, unsigned bits, unsigned rows);

}  // namespace strix::minimax::moe::opt
