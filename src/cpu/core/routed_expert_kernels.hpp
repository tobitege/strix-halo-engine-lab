#pragma once
// Row kernels of the host routed-expert executor (core/routed_expert.cpp): one weight row against up to kMaxAssign
// assignments (A8 rows), so the row is read once for all of them. A table per ISA (src/cpu/arch/<isa>), bit-identical
// by contract (see routed_expert.hpp); the scalar table in routed_expert.cpp is the bitwise oracle.
#include <cstdint>

#include "cpu/core/routed_expert.hpp"

namespace strix::cpu::expert::detail {

inline constexpr std::uint32_t kMaxAssign = 16;  // assignments per kernel call (the executor chunks larger groups)
inline constexpr std::uint32_t kGateBlockBytes[2] = {144, 176};  // Q4_K, Q5_K (256 weights)
inline constexpr std::uint32_t kDownBlockBytes[2] = {24, 34};    // Q5_1, Q8_0 (32 weights)

// out[a] = gate/up value of this weight row (hidden / 256 super-blocks) for rows[a], a < na.
using GateRowFn = void (*)(const std::uint8_t* wrow, std::uint32_t blocks, const RowA8* const* rows, std::uint32_t na,
                           float* out);
// out[a] = rawdot of this down row (ffn / 32 blocks) against h codes/deltas/sums of assignment a, a < na.
using DownRowFn = void (*)(const std::uint8_t* wrow, std::uint32_t groups, const std::int8_t* const* codes,
                           const float* const* deltas, const std::int32_t* const* sums, std::uint32_t na, float* out);

struct Kernels {
    const char* name = "scalar";
    GateRowFn gate[2] = {nullptr, nullptr};  // [GateFormat]
    DownRowFn down[2] = {nullptr, nullptr};  // [DownFormat]
};

float half_to_float(std::uint16_t h) noexcept;  // exact, same as the CUDA tests' half_to_float
const Kernels& scalar_kernels() noexcept;

}  // namespace strix::cpu::expert::detail
