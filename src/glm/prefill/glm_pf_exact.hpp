#pragma once
// Exact rows=1 Tier B arithmetic; weights shared across a row tile.
// Calls sharing workspace must be stream ordered. No submit-time allocation.
#include "strix/glm/engine.hpp"
namespace strix::glm::pf::exact {
// Uses truthful raw metadata even for prepared-only handles; rejects non-mul1,
// bias-bearing prepared spans, widths outside 2..8 and non-H128 geometry.
bool supported(const engine::Linear& l);
unsigned vector_splits(unsigned input, unsigned output, unsigned bits);
// FP16 transformed input, rounded up to 256 bytes. Workspace must be 256-aligned.
std::size_t workspace_bytes(unsigned max_rows, unsigned input);
unsigned tile_rows();  // STRIX_GLM_PF_EXACT_TM=16|32|64, read per call
unsigned x_auto_rows();  // STRIX_GLM_PF_EXACT_X=auto threshold: STRIX_GLM_PF_EXACT_X_MIN_ROWS, default 64 (32 with FAST WMMA)
enum class Route { tm, x, x2, x3, x3r8 };
Route route(unsigned rows);  // STRIX_GLM_PF_EXACT_X and auto threshold, read per call
// X/Y/workspace must be disjoint; strides are float elements, >= matrix widths.
// No reads or writes of rows outside [0, rows).
void linear_rows(const engine::Linear& l, const float* x, std::size_t x_stride,
                 float* y, std::size_t y_stride, unsigned rows, void* ws,
                 std::size_t ws_bytes, hipStream_t stream);
// STRIX_GLM_PF_EXACT_X=1 selects X; =2 X2 with register prefetch; =3/4 X3 (scalar activations, four/eight rows per
// wave); =auto X3 with eight rows per wave for calls >= x_auto_rows().
hipFuncAttributes x_resources(const engine::Linear& l, unsigned variant = 1);
std::size_t scratch_bytes(const engine::Linear& l, unsigned tm);
}  // namespace strix::glm::pf::exact
