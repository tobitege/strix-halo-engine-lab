#pragma once

#include <atomic>
#include <cstdint>

namespace strix::device {

// The device budget in bytes (0 = none declared) of this process's bound device, written by the start gate
// (gate.cpp start_gate) from the resolved strix-memory-budget-v1. Header-only so the shared planners that do not
// receive the binding (strix::deploy::probe_device) read it without linking the gate.
inline std::atomic<std::uint64_t>& process_device_budget_bytes() noexcept {
    static std::atomic<std::uint64_t> bytes{0};
    return bytes;
}

} // namespace strix::device
