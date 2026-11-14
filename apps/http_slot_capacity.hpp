#pragma once
// Forwarding header: slot capacity accounting lives in the shared serve layer
// (include/strix/serve/slot_capacity.hpp). These are the historical names of this app.
#include "strix/serve/slot_capacity.hpp"

namespace strix::decode_cli {
using HttpServingMemory = strix::serve::ServingMemory;
using HttpBridgeGrowthObservation = strix::serve::BridgeGrowthObservation;
using HttpSlotCapacity = strix::serve::SlotCapacity;

inline bool http_slot_should_grow(bool queued, std::size_t active, std::size_t built,
    std::size_t capacity, bool memory_ok, bool spare = false) {
    return strix::serve::slot_should_grow(queued, active, built, capacity, memory_ok, spare);
}
inline std::uint32_t http_eager_slots(std::uint32_t capacity, std::uint32_t eager = 4) {
    return strix::serve::eager_slots(capacity, eager);
}
inline bool http_slot_memory_fits(std::uint64_t retained, std::uint64_t active_peak,
    std::optional<std::uint64_t> available,
    std::optional<std::uint64_t> gtt_available, std::uint64_t headroom) {
    return strix::serve::slot_memory_fits(retained, active_peak, available, gtt_available, headroom);
}
inline std::uint64_t http_slot_retained_bytes(std::uint64_t state_device, std::uint64_t state_host,
    std::uint64_t arena, std::uint64_t metadata, std::uint64_t external) {
    return strix::serve::slot_retained_bytes(state_device, state_host, arena, metadata, external);
}
inline std::uint64_t http_shared_growth_headroom(std::uint64_t fixed,
    std::uint64_t cache_cap, std::uint64_t cache_resident,
    std::uint64_t staging_cap, std::uint64_t staging_resident) {
    return strix::serve::shared_growth_headroom(fixed, cache_cap, cache_resident, staging_cap, staging_resident);
}
inline bool http_bridge_growth_fits(std::uint64_t retained, const HttpServingMemory& cost,
    std::optional<std::uint64_t> host, std::optional<std::uint64_t> device,
    std::uint64_t remaining_headroom) {
    return strix::serve::bridge_growth_fits(retained, cost, host, device, remaining_headroom);
}
template<class Observe>
auto make_http_bridge_growth_check(std::uint64_t retained, HttpServingMemory cost,
    std::uint64_t fixed_headroom, std::uint64_t cache_cap, std::uint64_t staging_cap, Observe observe) {
    return strix::serve::make_bridge_growth_check(retained, cost, fixed_headroom, cache_cap, staging_cap,
        std::move(observe));
}
inline HttpSlotCapacity resolve_http_slot_capacity(std::uint64_t retained, std::uint64_t active_peak,
    std::optional<std::uint64_t> available, std::optional<std::uint64_t> gtt_available,
    std::uint64_t headroom, std::uint32_t cap = 0) {
    return strix::serve::resolve_slot_capacity(retained, active_peak, available, gtt_available, headroom, cap);
}
inline HttpSlotCapacity resolve_http_bridge_capacity(std::uint64_t retained,
    const HttpServingMemory& cost, std::optional<std::uint64_t> host,
    std::optional<std::uint64_t> device, std::uint64_t host_headroom,
    std::uint64_t device_headroom, std::uint32_t cap = 0) {
    return strix::serve::resolve_bridge_capacity(retained, cost, host, device, host_headroom, device_headroom, cap);
}
} // namespace strix::decode_cli
