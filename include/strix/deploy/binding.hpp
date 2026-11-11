#pragma once
// The one deployment binding shared by every engine adapter: checked byte
// arithmetic, pin queries, inspection inventory binding, rank-ordered live
// device-fact exchange and canonical plan-hash agreement. Host transport only;
// every call here runs before any device owner is constructed.
#include "strix/deploy/plan.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::deploy {

inline std::uint64_t checked_add(std::uint64_t a, std::uint64_t b) {
    if (b > UINT64_MAX - a) throw std::overflow_error("deployment byte sum overflow");
    return a + b;
}
inline std::uint64_t checked_mul(std::uint64_t a, std::uint64_t b) {
    if (a && b > UINT64_MAX / a) throw std::overflow_error("deployment byte product overflow");
    return a * b;
}
inline std::uint64_t aligned(std::uint64_t n, std::uint64_t a) {
    if (!a) throw std::invalid_argument("deployment alignment must be nonzero");
    return checked_mul(checked_add(n, a - 1) / a, a);
}

// True when the TOML explicitly selected this key (bare or memory.-qualified);
// adapters keep implicit CLI defaults for everything else.
inline bool pinned(const Spec& s, const std::string& name) {
    return s.pins.contains(name) || s.pins.contains("memory." + name);
}

// Inspection: the orchestrator-supplied inventory with this rank's entry replaced
// by the fresh local observation. Never invents a peer.
inline std::vector<DeviceFacts> inspection_devices(const Spec& spec, unsigned rank, const DeviceFacts& local) {
    auto devices = spec.devices;
    if (devices.empty()) devices.push_back(local);
    else {
        if (rank >= devices.size()) throw std::invalid_argument("deployment: rank outside supplied inventory");
        devices[rank] = local;
    }
    return devices;
}

inline void require_plan(const Plan& plan) {
    if (plan.refused) throw std::runtime_error("deployment refused: " + plan.refusal +
        (plan.reasons.empty() ? "" : "; " + plan.reasons.back()));
}

// Symmetric host exchange: send `out`, receive the peer's same-sized buffer into `in`.
using Exchange = std::function<void(std::span<const std::byte> out, std::span<std::byte> in)>;

// Fresh rank-ordered observations. Wire fields deliberately exclude private
// observation paths (drm_device) and optional GTT/VRAM detail.
inline std::vector<DeviceFacts> exchange_pair_facts(unsigned rank, const DeviceFacts& local, const Exchange& exchange) {
    if (rank > 1) throw std::invalid_argument("deployment: pair rank must be 0 or 1");
    struct Wire {
        std::uint64_t magic, rank, version, cu, total, free, unified;
        std::array<char, 64> gfx, source;
    } mine{}, peer{};
    mine.magic = 0x53544445504c3034ULL; mine.rank = rank;
    mine.version = local.gfx_target_version; mine.cu = local.cu_count;
    mine.total = local.mem_total_bytes; mine.free = local.mem_free_bytes; mine.unified = local.unified;
    if (local.gfx_target.size() >= mine.gfx.size() || local.source.size() >= mine.source.size())
        throw std::invalid_argument("deployment: observation string exceeds wire limit");
    std::copy(local.gfx_target.begin(), local.gfx_target.end(), mine.gfx.begin());
    std::copy(local.source.begin(), local.source.end(), mine.source.begin());
    exchange(std::as_bytes(std::span(&mine, 1)), std::as_writable_bytes(std::span(&peer, 1)));
    if (peer.magic != mine.magic || peer.rank != 1U - rank || peer.cu > UINT32_MAX || peer.unified > 1 ||
        peer.gfx.back() || peer.source.back()) throw std::runtime_error("deployment: incompatible peer facts");
    DeviceFacts remote;
    remote.gfx_target = peer.gfx.data(); remote.source = peer.source.data();
    remote.gfx_target_version = peer.version; remote.cu_count = static_cast<std::uint32_t>(peer.cu);
    remote.mem_total_bytes = peer.total; remote.mem_free_bytes = peer.free; remote.unified = peer.unified != 0;
    std::vector<DeviceFacts> devices(2);
    devices[rank] = local; devices[1U - rank] = remote;
    return devices;
}

// Both ranks must have resolved the same canonical plan. Refusal is left to require_plan.
inline void agree_plan(const Plan& plan, const Exchange& exchange) {
    std::array<char, 16> own{}, other{};
    const auto id = plan.hash();
    if (id.size() > own.size()) throw std::logic_error("deployment: plan id exceeds wire size");
    std::copy(id.begin(), id.end(), own.begin());
    exchange(std::as_bytes(std::span(own)), std::as_writable_bytes(std::span(other)));
    if (own != other) throw std::runtime_error("deployment: peer plan hash mismatch");
}

// Live pair resolution: exchange facts, resolve the same inventory on both ranks, agree.
inline Plan resolve_pair(const Spec& spec, const FootprintProvider& provider,
                         unsigned rank, const DeviceFacts& local, const Exchange& exchange) {
    auto plan = resolve(spec, exchange_pair_facts(rank, local, exchange), provider);
    agree_plan(plan, exchange);
    return plan;
}

} // namespace strix::deploy
