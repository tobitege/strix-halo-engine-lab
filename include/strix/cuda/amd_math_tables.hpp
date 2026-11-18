// Host loader of the gfx1151 math tables used by amd_math.cuh (see that header for their meaning).
#pragma once

#include <string>

#include "strix/cuda/amd_math.cuh"

namespace strix::cuda::amdmath {

// Reads <dir>/tbl_exp_pm.bin, tbl_rsq_1_4.bin, tbl_rcp_1_2.bin and hard_v_{exp,rsq,rcp}_f32.bin, canonicalises and
// sorts the hard lists, uploads everything to the current device. Throws std::runtime_error on a missing or
// inconsistent file. The returned pointers live until free_tables.
[[nodiscard]] Tables load_tables(const std::string& dir);
void free_tables(Tables& tables);

}  // namespace strix::cuda::amdmath
