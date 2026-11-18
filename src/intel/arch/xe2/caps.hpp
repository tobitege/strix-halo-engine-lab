#pragma once
// Xe2 specialisation (layer 3): Arc B-series discrete (xe2-bmg) and Lunar Lake integrated (xe2-lnl). Panther Lake is
// Xe3 (xe3-ptl) and shares the Xe2 programming model (SIMD16 sub-groups, DPAS), so it selects this specialisation
// until an arch/xe3 directory carries measured differences. Selected by Capabilities::arch, never by model.
// Constants are the joint_matrix / DPAS shapes of the oneAPI documentation; they are planning inputs for the XMX
// dense tile, to be confirmed with the sycl_ext_oneapi_matrix combinations query on each device.
#include <cstdint>
#include <string_view>

namespace strix::intel::arch::xe2 {

inline constexpr std::uint32_t kSubgroup = 16;  // native sub-group width (SIMD16)
// joint_matrix FP16 x FP16 -> FP32 on Xe2: A M x K, B K x N, C M x N, one DPAS per tile.
inline constexpr std::uint32_t kMatM = 8, kMatN = 16, kMatK = 16;
// INT8 x INT8 -> INT32: K doubles (32).
inline constexpr std::uint32_t kMatKInt8 = 32;

[[nodiscard]] constexpr bool selects(std::string_view arch) noexcept {
    return arch == "xe2-bmg" || arch == "xe2-lnl" || arch == "xe3-ptl";
}

// Planning geometry of the first XMX dense tile: one work-group = 4 sub-groups x (32 rows x 64 columns) so a B tile
// staged in SLM feeds 4 row tiles (the HIP core feeds 8 waves from one LDS stage; Xe2 SLM and register file per
// sub-group differ, so this is the starting point for measurement, not a tuned value).
struct DenseTile {
    std::uint32_t subgroups = 4, rows = 32, cols = 64, k_stage = 64;
};
[[nodiscard]] constexpr DenseTile dense_tile() noexcept { return {}; }

} // namespace strix::intel::arch::xe2
