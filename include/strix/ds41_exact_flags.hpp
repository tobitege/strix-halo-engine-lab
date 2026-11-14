#pragma once

#include "strix/env_flags.hpp"

#include <atomic>

namespace strix::ds41 {

// Exact decode options that are default ON (gain-audit, 29-oct: "every exact gain that is not slower is the default"):
// each is on unless its variable is set to 0, and set_*(0 / 1) overrides the environment for in-process A/Bs (the rank and
// batch apps' --aba-flags); -1 restores it. All of them change only launches, loads or staging, never the arithmetic.
//   STRIX_DS41_R1FUSE_ROUTES   the router's top-k writes the expert routes directly (r1fuse), no separate build-routes launch
//   STRIX_DS41_NORM_ILP        norm_kernel with the row's loads gathered before the first add / store
//   STRIX_DS41_TRELLIS_ROW_VIEWS strided rows go straight to the dense-r1 plan instead of a staging copy
#define STRIX_DS41_EXACT_FLAG(fn, var)                                                     \
    inline std::atomic<int>& fn##_override() {                                             \
        static std::atomic<int> mode{-1};                                                  \
        return mode;                                                                       \
    }                                                                                      \
    inline void set_##fn(int mode) { fn##_override().store(mode, std::memory_order_relaxed); } \
    inline bool fn() {                                                                     \
        const int mode = fn##_override().load(std::memory_order_relaxed);                  \
        if (mode >= 0) return mode != 0;                                                   \
        static const bool env = env_flag_default_on(var, EnvFlagMatch::first_character);                   \
        return env;                                                                        \
    }

STRIX_DS41_EXACT_FLAG(r1fuse_routes, "STRIX_DS41_R1FUSE_ROUTES")
STRIX_DS41_EXACT_FLAG(norm_ilp, "STRIX_DS41_NORM_ILP")
STRIX_DS41_EXACT_FLAG(trellis_row_views, "STRIX_DS41_TRELLIS_ROW_VIEWS")

#undef STRIX_DS41_EXACT_FLAG

}  // namespace strix::ds41
