#pragma once
// Host-only census of the TURBO+ onfly INT4 copies (candidate tensors and reserved device bytes), shared by the
// prefill admission plan, the effective-config print and the loader (hip_qwen_onfly_i4.cpp).
#include "core/iu4_onfly.hpp"
#include "core/iu4_rows_plan.hpp"
#include "strix/model_manifest.hpp"
#include "strix/onfly_i4.hpp"

#include <cstdint>

namespace strix {
inline bool qwen_onfly_i4_candidate(const ManifestTensor& t, const onfly_i4::Resolved& r) {
    const unsigned group = onfly_i4::classify(t.name);
    if (!group || !(r.groups & group) || t.ggml_type != 8U || t.dimensions.size() != 2U) return false;
    const auto k = t.dimensions[0], n = t.dimensions[1];
    return onfly_i4::dispatch_width(k) && n && n <= UINT32_MAX && t.byte_size == n * (k / 32U) * 34U;
}
// Device bytes the copies of this request take (0 for native).
inline std::uint64_t qwen_onfly_i4_reservation(const ModelManifest& manifest, const onfly_i4::Resolved& r) {
    std::uint64_t total = 0;
    if (!r.active()) return 0;
    for (const auto& t : manifest.tensors)
        if (qwen_onfly_i4_candidate(t, r)) total += core::onfly::copy_bytes(t.dimensions[1], t.dimensions[0]);
    return total;
}
}  // namespace strix
