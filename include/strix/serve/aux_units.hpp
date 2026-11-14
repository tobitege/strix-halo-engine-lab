#pragma once

#include "strix/device/gate.hpp"
#include "strix/serve/aux_lane.hpp"

// The aux lane's unit list from the start gate's result. Header-only so the lane itself stays free of device types;
// a caller links the device library it already uses for the gate.
namespace strix::serve::aux {

// The bound GPU always (side models already run there), then every declared NPU or CPU unit whose `aux` role is
// granted. An NPU without its co-run qualification, an excluded unit and an undeclared unit are absent.
[[nodiscard]] inline std::vector<Unit> units_from(const device::Bound& bound) {
    std::vector<Unit> out{Unit{UnitKind::gpu, bound.device.id}};
    for (const auto& unit : bound.units) {
        if (!device::role_refusal(unit, "aux").empty()) continue;
        out.push_back(Unit{unit.kind == device::Kind::npu ? UnitKind::npu : UnitKind::cpu, unit.id});
    }
    return out;
}

} // namespace strix::serve::aux
