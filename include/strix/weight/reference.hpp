#pragma once
#include "registry.hpp"
namespace strix::weight {
// decode_reference is storage widening: request.arithmetic must be {}. Explicit
// arithmetic identities (including known legacy names) are unsupported; widening
// does not qualify their operand rounding or product/reduction contracts.
// Byte strides are admitted only for contiguous row-major scalar storage.
// open_source/validate_map remain future scaffolding, not implemented P02 APIs.
// Cost operation counts are unknown (zero denominators), not measured zero work.
Status signed_group_reference(const Registry&, const WeightView&, std::uint64_t,
                              SignedGroup32&);
}
