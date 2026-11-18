#pragma once

#include "strix/lever_cache.hpp"

#include <atomic>
#include <cstdlib>
#include <cstring>

namespace strix::ds41 {

// STRIX_DS41_TOUCH_BATCH (default on, =0 off): the decode read-ahead touches of one prefetch window go out as one launch
// over up to 16 spans instead of one touch launch per span: the same bytes in the same order per span, no stores, so the
// same bits. set_touch_batch(0 / 1) overrides the environment (the batch app's --aba-flags touch); -1 restores it.
inline std::atomic<int>& touch_batch_override() {
    static std::atomic<int> mode{-1};
    return mode;
}
inline void set_touch_batch(int mode) { touch_batch_override().store(mode, std::memory_order_relaxed); }
inline bool touch_batch() {
    const int mode = touch_batch_override().load(std::memory_order_relaxed);
    if (mode >= 0) return mode != 0;
    static ::strix::lever_cache<bool> env{[] {
        const char* v = std::getenv("STRIX_DS41_TOUCH_BATCH");
        return !(v != nullptr && std::strcmp(v, "0") == 0);
    }};
    return env.get();
}

}  // namespace strix::ds41
