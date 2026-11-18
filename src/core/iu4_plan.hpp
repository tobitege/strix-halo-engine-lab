#pragma once

namespace strix::core::iu4 {
// Row count must not change the arithmetic partition. Each pass accumulates
// ascending groups within a wave, then partial passes and waves in ascending order.
struct RowPlan { unsigned S, GC; bool from_a8_slot; };
constexpr RowPlan row_plan(unsigned /* rows */, unsigned groups) {
    if (groups == 32U) return {4U, 32U, false};
    if (groups == 40U) return {4U, 40U, false};
    return {8U, 32U, false}; // G=64/96: identical partition for R=1..8.
}
} // namespace strix::core::iu4
