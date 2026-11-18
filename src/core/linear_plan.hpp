#pragma once
#include "numeric_partition.hpp"
#include "iu4.hpp"
#include "linear_plan_table.hpp"
#include "gemv_policy.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace strix::core {
// Numeric identity, independent of scheduling. K16 boundaries use floor(k*S^-1).
// Each partition starts at +0, visits K16 in ascending order, and the output
// transform adds partition partials in ascending order before final rounding.
struct LinearContract {
    int k16 = 0, splits = 1;
    unsigned version = 1, accumulation = 1; // 1 WMMA FP32, 2 DOT2 FP32
    int boundary(int s) const { return int(int64_t(k16) * s / splits); }
    static constexpr unsigned reduction_order = 1; // ascending partitions
};
// Rows-free contract: the K chain depends only on the matrix (input width) and a bound split count. Chunk-invariant by
// construction; the rows-dependent linear_contract below stays the frozen v1 identity for engines that bind it.
inline LinearContract fixed_linear_contract(int input, int splits) {
    return {input / 16, std::max(1, std::min(splits, input / 16)), 2U, 1U};
}
template<class W> LinearContract linear_contract(const W& w, int rows) {
    int requested = 1;
    if (rows > 8) {
        // Frozen numeric policy v1: these are reference buckets, NOT launch tiles.
        const auto bucket_m = size_t(numeric_partition_bucket_rows(w, rows));
        const size_t buckets = ((size_t(rows) + bucket_m - 1) / bucket_m) *
            ((size_t(w.output) + 63) / 64);
        if (numeric_partition_medium(w, rows)) {
            if (w.output >= 32768) requested = rows < 64 ? 4 : 1;
            else {
                const size_t target = rows > 64 && rows % 64 ? 512 : 256;
                while (buckets * size_t(requested) < target && requested < 16) requested *= 2;
            }
        } else while (buckets * size_t(requested) < 160 && requested < 32) requested *= 2;
    } else requested = measured_vector_splits(w.input, w.output, w.bits, w.codebook, rows);
    return {w.input / 16, std::min(requested, w.input / 16)};
}
struct LinearTile { int bm, bn, wm, wn, raster; };
inline constexpr LinearTile linear_tiles[] = {
    {128,128,64,32,2}, {64,128,64,32,8}, {128,64,64,32,8}, {64,64,32,32,8},
    {256,128,64,64,2}, {128,128,64,64,8}, {128,256,64,64,8}, {32,128,32,32,8}
};
// IU4 opt-in plans. Tail uses four waves (16x64, correcting the plan's
// inconsistent 16x32/four-wave entry). Large uses BN128 to bound live FP32.
inline constexpr LinearTile iu4_tiles[] = {
    {32,128,16,64,8}, {64,128,32,32,8}, {128,128,32,64,2}
};
inline constexpr bool iu4_group_ok(int group) { return group == 64 || group == 128 || group == 256; }
inline int iu4_policy_tile(int rows) { return rows <= 96 ? 0 : (rows <= 320 ? 1 : 2); }
inline constexpr bool iu4_geometry_ok(int inputs, int outputs, int rows, int tile) {
    return inputs > 0 && inputs % 64 == 0 && outputs > 0 && outputs % 128 == 0 &&
        rows > 0 && tile >= 0 && tile < 3;
}
enum class PhysicalPolicy { legacy, automatic };
inline PhysicalPolicy physical_policy(const char* value) {
    return !value || !*value || std::strcmp(value, "auto") == 0 ? PhysicalPolicy::automatic : PhysicalPolicy::legacy;
}
struct LinearPhysicalPlan {
    int tile = -1, bm = 0, bn = 0, waves = 0, raster = 0;
    uint64_t grid_x = 0, grid_y = 1, grid_z = 0, useful_rows = 0, emitted_rows = 0;
    // Valid packed N16*K16 tiles decoded across all CTAs; not bytes or FLOPs.
    uint64_t decode_visits = 0, lds_bytes = 0;
    bool fused_output = false;
    uint64_t launched_ctas() const { return grid_x * grid_y * grid_z; }
};
enum class LinearKernel { empty, tiled, wmma, packed, gemv, fused_input, fused_output, fused_both, completed };
struct LinearExecution {
    LinearContract numeric;
    LinearPhysicalPlan physical;
    LinearKernel kernel = LinearKernel::empty;
    GemvSelection gemv;
    size_t workspace_bytes = 0, partial_offset = 0;
    bool prefer_wmma = false;
    const char* fallback = "none";
    char symbol[192] = {};
};
template<class W> int legacy_linear_tile(const W& w, int rows) {
    if (rows < 128 || w.codebook != 2 || w.bits < 4 || w.bits > 6) return -1;
    if (rows == 128) return w.output % 256 == 0 && w.output < 16384 ? 6 : 0;
    return rows <= 1024 ? 4 : 0;
}
inline void account_linear_tile(LinearExecution& e, int rows, int output, int tile) {
    const auto& t = linear_tiles[tile];
    auto& p = e.physical;
    p.tile = tile; p.bm = t.bm; p.bn = t.bn;
    p.waves = (t.bm / t.wm) * (t.bn / t.wn); p.raster = t.raster;
    const uint64_t blocks_m = (uint64_t(rows) + uint64_t(t.bm) - 1) / uint64_t(t.bm);
    p.grid_x = blocks_m * (uint64_t(output) / uint64_t(t.bn));
    p.grid_z = uint64_t(e.numeric.splits);
    p.useful_rows = uint64_t(rows); p.emitted_rows = blocks_m * uint64_t(t.bm);
    p.decode_visits = blocks_m * uint64_t(output / 16) * uint64_t(e.numeric.k16);
    p.lds_bytes = uint64_t(2 * (t.bm + t.bn) * 40 * 2);
}
// Measured plan selection. Every compiled tile computes the same numeric
// contract (same S, K16 boundaries, ascending accumulation), so the plan is a
// pure scheduling choice and is taken from the measured table: one row per
// (shape, format, M bucket) that a gfx1151 bench found bitwise-equal to the
// legacy plan and faster at both ends of the bucket. Unknown shape or bucket:
// legacy. The table is generated, never hand-edited (bench `plan-table`).
template<class W> int measured_linear_tile(const W& w, int rows) {
    for (const auto& m : measured_plans)
        if (m.input == w.input && m.output == w.output && m.bits == w.bits &&
            m.codebook == w.codebook && rows >= m.m_lo && rows <= m.m_hi) return m.tile;
    return -1;
}
inline bool linear_tile_compatible(int tile, int output) {
    constexpr int count = int(sizeof(linear_tiles) / sizeof(linear_tiles[0]));
    return tile >= 0 && tile < count && output % linear_tiles[tile].bn == 0;
}
template<class W> LinearExecution make_linear_execution(const W& w, int rows,
    bool prefer_wmma, PhysicalPolicy policy, bool tiled_enabled = true) {
    LinearExecution e{}; e.prefer_wmma = prefer_wmma;
    if (!rows) return e;
    e.workspace_bytes = SIZE_MAX;
    if (rows < 0 || w.input <= 0 || w.output <= 0 || w.input % 128 || w.output % 128) return e;
    e.numeric = linear_contract(w, rows);
    e.numeric.accumulation = prefer_wmma ? 1U : 2U;
    const size_t in = size_t(rows) * size_t(w.input), out = size_t(rows) * size_t(w.output);
    const size_t limit = std::numeric_limits<size_t>::max();
    if (in > (limit - 255) / 2 || out > limit / 4 / size_t(e.numeric.splits) ||
        in / 128 > size_t(UINT32_MAX) * 4 || out / 128 > size_t(UINT32_MAX) * 4) return e;
    e.partial_offset = (in * 2 + 255) & ~size_t(255);
    const size_t partial = out * 4 * size_t(e.numeric.splits);
    if (partial > limit - e.partial_offset) return e;
    e.workspace_bytes = e.partial_offset + partial;
    e.kernel = prefer_wmma ? LinearKernel::wmma : LinearKernel::gemv;
    e.fallback = prefer_wmma ? (tiled_enabled ? "unqualified tiled geometry/format" : "tiled disabled") : "gemv operation";
    int tile = prefer_wmma && tiled_enabled ? legacy_linear_tile(w, rows) : -1;
    if (tile >= 0) {
        account_linear_tile(e, rows, w.output, tile);
        if (policy == PhysicalPolicy::automatic) {
            const int measured = measured_linear_tile(w, rows);
            if (linear_tile_compatible(measured, w.output)) account_linear_tile(e, rows, w.output, measured);
        }
        e.kernel = LinearKernel::tiled; e.fallback = "none";
    }
    return e;
}
} // namespace strix::core
