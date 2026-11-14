#pragma once
// GLM compatibility names for the shared EXL3 device primitives. Adopted primitive
// provenance and the operation-order contract are retained in the shared header.
#include "../../moe_core/exl3_trellis.hpp"

namespace strix::glm::moe::dev {
using strix::moe_core::Half2;
using strix::moe_core::materialize;
using strix::moe_core::hadamard128;
using strix::moe_core::half_bits;
using strix::moe_core::transform_block;
using strix::moe_core::transform_block_at;
using strix::moe_core::transform_block_values;
using strix::moe_core::transform_values;
using strix::moe_core::hash16;
using strix::moe_core::mul1_pair;
using strix::moe_core::LaneGeom;
using strix::moe_core::GlobalU32;
using strix::moe_core::LocalU32;
using strix::moe_core::uniform;
using strix::moe_core::load_u32;
using strix::moe_core::TileStates;
using strix::moe_core::reduce4;
using strix::moe_core::Slot;
using strix::moe_core::compute_slot;
// <= 8-row split policy: the one moe_core policy (gemv_policy.hpp measured_vector_splits); no GLM shape is special-cased.
using strix::moe_core::vector_splits;
}  // namespace strix::glm::moe::dev
