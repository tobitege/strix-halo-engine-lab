#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <stdexcept>

namespace strix::decode_cli {
// Pure admission policy; callers observe memory only when demand requires growth.
// Grow when every built slot is busy and work waits, or (spare) as soon as the last free slot is taken, so a request
// arriving during a long prompt finds a built slot (nested admission needs one inside the block).
inline bool http_slot_should_grow(bool queued, std::size_t active, std::size_t built,
    std::size_t capacity, bool memory_ok, bool spare = false) {
    return (queued || spare) && active == built && built < capacity && memory_ok;
}
inline std::uint32_t http_eager_slots(std::uint32_t capacity, std::uint32_t eager = 4) {
    return eager ? std::min(capacity, eager) : capacity;
}
inline bool http_slot_memory_fits(std::uint64_t retained, std::uint64_t active_peak,
    std::optional<std::uint64_t> available,
    std::optional<std::uint64_t> gtt_available, std::uint64_t headroom) {
    if (!available || !gtt_available) return false;
    const auto bytes = std::min(*available, *gtt_available);
    if (bytes < headroom || bytes - headroom < retained) return false;
    const auto remaining = bytes - headroom - retained;
    return remaining >= active_peak;
}
inline std::uint64_t http_slot_retained_bytes(std::uint64_t state_device, std::uint64_t state_host,
    std::uint64_t arena, std::uint64_t metadata, std::uint64_t external) {
    std::uint64_t total = 0;
    for (const auto n : {state_device, state_host, arena, metadata, external}) {
        if (n > UINT64_MAX-total) throw std::overflow_error("HTTP slot retained overflow");
        total += n;
    }
    return total;
}
struct HttpServingMemory {
    std::uint64_t shared_host = 0, shared_device = 0, slot_host = 0, slot_device = 0;
};
// At the serving boundary bridge shared setup is resident (or released).
// Reserve only unallocated cache/capture payload plus fixed headroom. Fresh
// observations already include the resident fraction of both shared payloads.
inline std::uint64_t http_shared_growth_headroom(std::uint64_t fixed,
    std::uint64_t cache_cap, std::uint64_t cache_resident,
    std::uint64_t staging_cap, std::uint64_t staging_resident) {
    for (const auto remaining : {cache_cap-std::min(cache_cap,cache_resident),
            staging_cap-std::min(staging_cap,staging_resident)}) {
        if (remaining > UINT64_MAX-fixed) throw std::overflow_error("HTTP shared growth overflow");
        fixed += remaining;
    }
    return fixed;
}
inline bool http_bridge_growth_fits(std::uint64_t retained, const HttpServingMemory& cost,
    std::optional<std::uint64_t> host, std::optional<std::uint64_t> device,
    std::uint64_t remaining_headroom) {
    return http_slot_memory_fits(retained,cost.slot_host,host,host,remaining_headroom) &&
        http_slot_memory_fits(retained,cost.slot_device,device,device,remaining_headroom);
}
struct HttpBridgeGrowthObservation {
    std::optional<std::uint64_t> host, device;
    std::uint64_t cache_resident=0, staging_resident=0;
};
// Capture the pre-bridge admission terms, but read residency and availability
// anew at each serving-boundary call. Shared setup peaks have no future charge.
template<class Observe>
auto make_http_bridge_growth_check(std::uint64_t retained, HttpServingMemory cost,
    std::uint64_t fixed_headroom, std::uint64_t cache_cap, std::uint64_t staging_cap, Observe observe) {
    return [=] {
        const auto pressure=observe();
        const auto remaining=http_shared_growth_headroom(fixed_headroom,cache_cap,pressure.cache_resident,
            staging_cap,pressure.staging_resident);
        return http_bridge_growth_fits(retained,cost,pressure.host,pressure.device,remaining);
    };
}
struct HttpSlotCapacity {
    std::uint32_t slots = 0;
    std::string limiting = "missing memory observations";
};
// Observations are remaining capacity after the primary stack, before the probe.
// Retained bytes charge only extra stacks; active peaks charge every runnable slot.
inline HttpSlotCapacity resolve_http_slot_capacity(std::uint64_t retained, std::uint64_t active_peak,
    std::optional<std::uint64_t> available, std::optional<std::uint64_t> gtt_available,
    std::uint64_t headroom, std::uint32_t cap = 0) {
    if (!available || !gtt_available || !retained) return {};
    if (active_peak > std::numeric_limits<std::uint64_t>::max() - retained)
        return {0, "cost overflow"};
    const auto per_slot = retained + active_peak;
    const auto count = [&](std::uint64_t bytes) {
        if (bytes < headroom || bytes - headroom < active_peak) return UINT64_C(0);
        const auto extra = (bytes - headroom - active_peak) / per_slot;
        return UINT64_C(1) + std::min(extra, static_cast<std::uint64_t>(UINT32_MAX) - 1U);
    };
    const auto uma = count(*available), gtt = count(*gtt_available);
    auto slots = std::min({uma, gtt, static_cast<std::uint64_t>(UINT32_MAX)});
    std::string limiting = uma <= gtt ? "MemAvailable budget" : "GTT budget";
    if (cap && cap <= slots) { slots = cap; limiting = "requested cap"; }
    return {static_cast<std::uint32_t>(slots), std::move(limiting)};
}
// The primary stack/shared prefill is already resident at runtime. Future
// bridge owners have distinct host/GTT costs; do not charge calibration host
// payload to GTT or copied sidecar device backing only to host.
inline HttpSlotCapacity resolve_http_bridge_capacity(std::uint64_t retained,
    const HttpServingMemory& cost, std::optional<std::uint64_t> host,
    std::optional<std::uint64_t> device, std::uint64_t host_headroom,
    std::uint64_t device_headroom, std::uint32_t cap = 0) {
    if (cost.shared_host > UINT64_MAX-host_headroom || cost.shared_device > UINT64_MAX-device_headroom)
        return {0,"shared cost overflow"};
    auto h=resolve_http_slot_capacity(retained,cost.slot_host,host,host,host_headroom+cost.shared_host,cap);
    auto d=resolve_http_slot_capacity(retained,cost.slot_device,device,device,device_headroom+cost.shared_device,cap);
    if (h.slots<=d.slots) { h.limiting="MemAvailable: "+h.limiting; return h; }
    d.limiting="GTT: "+d.limiting; return d;
}

} // namespace strix::decode_cli
