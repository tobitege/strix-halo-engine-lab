#pragma once
#include "backend.hpp"
namespace strix::core {
// Register-blocked WMMA core for prefill. Same numerical contract as
// launch_wmma_core: per output, K16 tiles of the split accumulate in ascending
// order into an FP32 WMMA accumulator that starts at +0. Only the work
// decomposition differs (block tile, LDS layout, pipelining).
struct WmmaFusedOutput {
    float* output = nullptr;   // non-null: apply the output transform in the epilogue
    float alpha = 1, beta = 0; // same meaning as output_transform
};
// Explicit opt-in: no default resolver calls this API. Dense, no K splits.
// Plan parameter group g (64, 128 or 256 K values per scale, K % g == 0):
// A scales [rows][K/g] FP32; W scales [N][K/g] FP16. Both operands
// must already be in the same rotated basis. Optional group_sums [g][row][N]
// is a diagnostic output, excluded from performance timing.
struct IU4Operands {
    const uint8_t* a_q = nullptr;
    const float* a_scale = nullptr;
    const uint8_t* w_q = nullptr;
    const _Float16* w_scale = nullptr;
    int group = 64;
};
struct IU4Resources {
    int vgpr = 0, active_waves = 0;
    size_t scratch_bytes = 0, lds_bytes = 0;
};
// Query the timed kernel; full_rows selects rows divisible by the plan's BM.
hipError_t query_wmma_iu4_resources(int tile, IU4Resources* resources, bool full_rows = false, int group = 64);
hipError_t launch_wmma_tiled_iu4(const IU4Operands& operands, float* output,
    int inputs, int outputs, int rows, hipStream_t stream, int tile = -1,
    int32_t* group_sums = nullptr);
// Grouped (MoE) IU4: one launch over a device tile list; no host sync on the histogram.
// groups[e] = {e, first, count}: expert e owns map positions [first, first+count) (the
// layout of a GPU MoE planner's per-expert groups). A row of map position p is
// map[p]*a_mul+a_add (map null: p), its output row map[p]*o_mul+o_add. Per expert e:
// packed W at w_q + e*w_expert_bytes, FP16 W scales at w_scale + e*w_scale_expert.
// col_scale null: raw FP32 stores (same per-element arithmetic as the dense launch of
// the same plan). Non-null: col_scale[e] is FP16-bit per-column scale and the epilogue
// applies the trellis output transform, out = had128(acc over 128 columns) * scale.
struct IU4Group { uint32_t expert, first, count; };
struct IU4GroupTile { uint32_t expert, first, rows; };
struct IU4GroupedOperands {
    const uint8_t* a_q = nullptr;
    const float* a_scale = nullptr;
    const uint8_t* w_q = nullptr;
    const _Float16* w_scale = nullptr;
    size_t w_expert_bytes = 0, w_scale_expert = 0;
    const uint16_t* const* col_scale = nullptr;
    const uint32_t* map = nullptr;
    uint32_t a_mul = 1, a_add = 0, o_mul = 1, o_add = 0;
    int group = 64;
};
// EXL3 -> IU4 g64 pack of one matrix per expert (the engines' iu4_convert_kernel, moved here
// with shared load-time rounding). trellis[e] = expert e's tiles (nullptr =
// skip). Both nibbles and scales are built at load time; source widths are 2..8.
// scale_rule: iu4_scale_rule_legacy (default, bits unchanged) or iu4_scale_rule_fast_full (core/iu4.hpp).
hipError_t convert_exl3_iu4(const uint32_t* const* trellis, int bits, int experts, int inputs, int outputs,
    uint8_t* wq, _Float16* ws, size_t wq_expert, size_t ws_expert, hipStream_t stream, int scale_rule = 0);
// Upper bound of the tile list length for a plan: ceil(rows_total / BM) + experts.
size_t iu4_grouped_max_tiles(size_t map_rows, int experts, int tile);
// Device: tiles = expert-ordered BM-row tiles of every nonempty group, *ntiles = count.
// One CTA, no host sync. tile < 0 is not accepted here (plan chosen by the caller).
hipError_t build_iu4_grouped_tiles(const IU4Group* groups, int experts, int tile,
    IU4GroupTile* tiles, uint32_t* ntiles, uint32_t max_tiles, hipStream_t stream);
// Grid = max_tiles * outputs/BN CTAs; CTAs past *ntiles return at once.
hipError_t launch_wmma_tiled_iu4_grouped(const IU4GroupedOperands& operands,
    const IU4GroupTile* tiles, const uint32_t* ntiles, uint32_t max_tiles, float* output,
    int inputs, int outputs, hipStream_t stream, int tile);
hipError_t query_wmma_iu4_grouped_resources(int tile, bool transform, IU4Resources* resources, int group = 64);
// Benchmark control: FP16 A row-major; W [K16][N16][column][K16].
// Exactly the IU4 plan's BM,BN,WM,WN/raster and the same tiled core.
hipError_t launch_wmma_tiled_f16_plan(const _Float16* a, const _Float16* w,
    float* output, int inputs, int outputs, int rows, hipStream_t stream, int tile = -1);
// Quantises finite FP16 [rows][K], after had128. Caller owns output buffers.
hipError_t quantise_wmma_iu4_activations(const _Float16* input, uint8_t* packed,
    float* scales, int inputs, int rows, hipStream_t stream, int group = 64);
// Returns hipErrorNotSupported when the geometry is outside the tiled path;
// the caller then uses launch_wmma_core. tile < 0 selects the policy.
hipError_t launch_wmma_tiled(const DeviceWeights& weights, const _Float16* input,
    float* partial, int rows, int splits, hipStream_t stream,
    const WmmaFusedOutput* fused = nullptr, int tile = -1, int raster = 0);
// True when launch_wmma_tiled would accept this call with a fused epilogue.
bool wmma_tiled_fuses_output(const DeviceWeights& weights, int rows, int splits, int tile = -1);
bool wmma_tiled_enabled();             // STRIX_TRELLIS_WMMA_TILED=0 disables
bool wmma_tiled_fused_output_enabled(); // false only with STRIX_TRELLIS_WMMA_TILED_FUSED=0
int wmma_tiled_fused_mode();            // 0 off (=0), 1 always (=1), 2 auto (default)
// Fused epilogue for this call (policy tile): mode 1 whenever possible; mode 2 only when the grid has
// at least STRIX_TRELLIS_WMMA_TILED_FUSED_MIN_BLOCKS blocks (default: CU count + 1). Same bits either way.
bool wmma_tiled_fuse_selected(const DeviceWeights& weights, int rows, int splits, int tile = -1);
int wmma_tiled_raster_rows(int tile);
}
