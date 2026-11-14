#pragma once

#include "strix/dispatch_fuse.hpp"
#include "strix/env_flags.hpp"

#include <cstdlib>

namespace strix {
// Independent experiments: none inherits STRIX_DISPATCH_FUSE (the window tails below do, by size). Default off.
inline bool qpersist_draft_score_enabled() noexcept {
    static const bool on = strix::envcfg::flag("STRIX_QPERSIST_DRAFT_SCORE", false);
    return on;
}
// STRIX_QSA_FAST_INT8=1 (opt-in): the fast prefill QSA kernels read kv-int8-tok-v1 (dequantized, rounded once to FP16).
inline bool qsa_fast_int8_enabled() noexcept {
    static const bool on = strix::envcfg::flag("STRIX_QSA_FAST_INT8", false);
    return on;
}
// STRIX_QSA_TAIL_INT8=1 (opt-in): the fused QSA window tail also with kv-int8-tok-v1.
inline bool qsa_tail_int8_enabled() noexcept {
    static const bool on = strix::envcfg::flag("STRIX_QSA_TAIL_INT8", false);
    return on;
}
inline bool qpersist_qsa_tail_enabled() noexcept {
    static const bool on = strix::envcfg::flag("STRIX_QPERSIST_QSA_TAIL", false);
    return on;
}
// STRIX_QPERSIST_DRAFT_ATTN (Q7 arm E, default ON): the MTP draft's score + tail use the exact
// draft-attention kernels whenever their admission holds. Unset or empty = on (default), 1 = on
// (explicit), 0 = incumbent kernels; variants STRIX_QPERSIST_DRAFT_ATTN_SCORE / _TAIL. The value is
// resolved (and rejected if unknown) by the launcher; this returns the raw setting, read once.
inline const char* qpersist_draft_attention_setting() noexcept {
    static const char* const value = std::getenv("STRIX_QPERSIST_DRAFT_ATTN");
    return value;
}
inline bool qpersist_mtp_cache_enabled() noexcept {
    static const bool on = strix::envcfg::flag("STRIX_QPERSIST_MTP_CACHE", false);
    return on;
}
inline bool qpersist_gdn_tail_enabled() noexcept {
    static const bool on = strix::envcfg::flag("STRIX_QPERSIST_GDN_TAIL", false);
    return on;
}
// Verify-window fused tails by window size. The fused kernels keep the expressions and operation order of the
// separate launches (bit-identical) but are not faster at every size, so the default admits only the sizes where
// the fused kernel costs less than the launches it replaces (kernel trace, 2-, 4- and 8-row windows):
// - GDN recurrence + gated norm (STRIX_QPERSIST_GDN_TAIL): 2 rows;
// - QSA softmax + value/gate (STRIX_QPERSIST_QSA_TAIL): 2 and 4 rows.
// Unset: those sizes, under the decode cycle's dispatch-fuse resolution (STRIX_DISPATCH_FUSE, so the baseline and
// an explicit STRIX_DISPATCH_FUSE=0 keep the separate launches). 1 = every admitted size, any other value = none.
inline bool qpersist_gdn_window_tail_admitted(unsigned rows) noexcept {
    static const int setting = strix_fuse_raw_switch("STRIX_QPERSIST_GDN_TAIL");
    return setting < 0 ? rows == 2U && strix_dispatch_fuse_enabled() : setting != 0;
}
inline bool qpersist_qsa_window_tail_admitted(unsigned rows) noexcept {
    static const int setting = strix_fuse_raw_switch("STRIX_QPERSIST_QSA_TAIL");
    return setting < 0 ? (rows == 2U || rows == 4U) && strix_dispatch_fuse_enabled() : setting != 0;
}
} // namespace strix
