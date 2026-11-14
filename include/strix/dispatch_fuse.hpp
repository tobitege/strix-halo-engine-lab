#pragma once
#include "strix/env_flags.hpp"
#include "strix/qpf2k_defaults.hpp"

#include <cstdlib>

namespace strix {
// STRIX_DISPATCH_FUSE=1: the decode cycle's dependent-dispatch reductions as one switch (default off).
// It turns on, unless the individual switch is set explicitly to 0:
// - STRIX_HC_PERSISTENT, STRIX_HC_PERSISTENT_INJECT: each HC step of a 2- or 4-row verify window (and the inject
//   before it) as one persistent launch instead of four (STRIX_HC_PERSISTENT_ROWS selects the sizes);
// - STRIX_WINDOW_INJECT_NORM: inject folded into the HC normalisation where the persistent fold declines (8 rows);
// - STRIX_WINDOW_COMMIT_COMBINE: MoE combine folded into the routed rows commit.
// Not included (opt-in only, measured neutral): STRIX_MTP_HC_PERSISTENT=1, the draft chain's three HC mixes per
// step as one-row persistent launches.
// Every fused form keeps the expressions and operation order of the separate launches (bit-identical).
inline bool strix_dispatch_fuse_enabled() noexcept {
    static const bool on = strix::envcfg::flag("STRIX_DISPATCH_FUSE", false);
    return !qpf2k::suppressed(qpf2k::Item::dispatch_fuse) && on;
}
// Cache only the raw tri-state. Owner filtering must run at every invocation,
// including calls made after a different owner initialized these lazy statics.
inline int strix_fuse_raw_switch(const char* name) noexcept {
    const char* v = std::getenv(name);
    return v && *v ? (v[0] == '1' ? 1 : 0) : -1;
}
inline bool strix_fuse_decision(int raw) noexcept {
    return raw < 0 ? strix_dispatch_fuse_enabled() : raw != 0;
}
#define STRIX_FUSE_READER(function, setting) \
inline bool function() noexcept { \
    static const int raw = strix_fuse_raw_switch(setting); \
    return strix_fuse_decision(raw); \
}
STRIX_FUSE_READER(strix_window_inject_norm_enabled, "STRIX_WINDOW_INJECT_NORM")
STRIX_FUSE_READER(strix_window_commit_combine_enabled, "STRIX_WINDOW_COMMIT_COMBINE")
STRIX_FUSE_READER(strix_hc_persistent_enabled, "STRIX_HC_PERSISTENT")
STRIX_FUSE_READER(strix_hc_persistent_inject_enabled, "STRIX_HC_PERSISTENT_INJECT")
#undef STRIX_FUSE_READER
inline bool strix_hc_inject_fusion_enabled() noexcept {
    return strix_hc_persistent_enabled() && strix_hc_persistent_inject_enabled();
}
} // namespace strix
