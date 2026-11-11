#pragma once
#include "glm_deploy_provider.hpp"
#include <map>
#include <string_view>

namespace strix::glm::deployment {
// Header-only upper bound on the existing TP2 loader (src/glm/engine/glm_lib.hpp,
// engine.hip load_*). Each linear takes the loader's own route:
//  - raw route (file layout in the 1-GiB arena, no strix_weights): MoE experts when
//    the fast MoE kernels run, DSA/KDA attention linears of the decoder layers when
//    the fused Tier B attention runs;
//  - prepared route (one strix_weights allocation, 2-MiB granule), plus a raw arena
//    copy only when --raw-linears keeps one.
// Arena rule as Lib::arena: requests share 1-GiB chunks; a request larger than the
// remainder opens a new chunk of max(request, 1 GiB) and discards the remainder,
// which is smaller than the largest request below 1 GiB.
struct Routes { bool fast_moe = true, attn_raw = false, keep_raw = false; };
template<class Tensors>
Owners header_owners(const Tensors& tensors, bool mtp, Routes routes) {
    using deploy::checked_add; using deploy::checked_mul; using deploy::aligned;
    constexpr std::uint64_t chunk = 1ULL << 30;
    std::uint64_t arena = 0, largest = 0, own_chunks = 0, big = 0, prepared = 0, whole = 0;
    const auto arena_add = [&](std::uint64_t n) {
        n = aligned(n, 256);
        if (n >= chunk) { big = checked_add(big, n); ++own_chunks; }
        else { arena = checked_add(arena, n); largest = std::max(largest, n); }
    };
    for (const auto& [name, t] : tensors) {
        const std::string_view n(name);
        const bool mtp_layer = n.find(".layers.45.") != n.npos;
        if (!mtp && mtp_layer) continue;
        if (n.ends_with(".suh") || n.ends_with(".svh") || n.ends_with(".mul1")) continue;
        if (n.ends_with(".trellis")) {
            const auto base = name.substr(0, name.size()-8);
            const auto& suh = tensors.at(base + ".suh");
            const auto& svh = tensors.at(base + ".svh");
            const bool out = n.find("qkv_proj.") != n.npos || n.find("q_b_proj.") != n.npos ||
                (n.find("mlp.") != n.npos && !n.ends_with("down_proj.trellis")) || base == "lm_head";
            const bool in = n.find("o_proj.") != n.npos || n.ends_with("down_proj.trellis");
            const auto tr = out || in ? checked_add(t.bytes, 1) / 2 : t.bytes;
            const auto sh = in ? checked_add(suh.bytes, 1) / 2 : suh.bytes;
            const auto sv = out ? checked_add(svh.bytes, 1) / 2 : svh.bytes;
            // Three KDA q/k/v slices require independent padding/prepared owners.
            const auto padded = checked_add(checked_add(tr, checked_add(sh, sv)), 3 * 768);
            const bool expert = n.find(".mlp.experts.") != n.npos;
            const bool attention = !mtp_layer && n.find(".self_attn.") != n.npos;
            const bool raw = (expert && routes.fast_moe) || (attention && routes.attn_raw);
            if (raw || routes.keep_raw) arena_add(padded);
            if (!raw) prepared = checked_add(prepared, checked_add(aligned(padded, 2ULL << 20), 6ULL << 20));
            // Input/output assembly can read the complete unsliced trellis.
            whole = std::max(whole, t.bytes);
        } else {
            const bool half = n.find("kv_b_proj.weight") != n.npos || n.find("f_b_proj.weight") != n.npos ||
                n.find("g_b_proj.weight") != n.npos || n.find("b_proj.weight") != n.npos ||
                n.find("conv1d.weight") != n.npos || n.find("A_log") != n.npos || n.find("dt_bias") != n.npos;
            arena_add(half ? checked_add(t.bytes, 1) / 2 : t.bytes);
            // R1b packed originals are retained. Include every eligible copy.
            if (n.find("f_a_proj.weight") != n.npos || n.find("g_a_proj.weight") != n.npos ||
                n.find("b_proj.weight") != n.npos || n.ends_with("mlp.gate.weight")) arena_add(t.bytes);
        }
    }
    if (!arena && !big && !prepared) throw std::invalid_argument("GLM deployment: empty tensor inventory");
    // Every chunk opening (shared or own) can discard less than the largest shared request.
    const auto arenas = arena ? checked_add(arena / (chunk-largest), 1) : 0;
    const auto padding = checked_mul(checked_add(arenas, own_chunks), largest);
    Owners result;
    result.weights = checked_add(prepared, checked_add(checked_add(arena, big), checked_add(padding, chunk)));
    // Expert view tables, HC constants and other small loader metadata.
    result.weights = checked_add(result.weights, 16ULL << 20);
    result.staging = checked_add(chunk, checked_add(whole, 256ULL << 20));
    return result;
}
} // namespace strix::glm::deployment
