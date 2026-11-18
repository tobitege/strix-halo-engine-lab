#pragma once
#include <strix/weight/runtime.hpp>
namespace strix::intel::weight {
// Source registry remains empty until this backend's operand migration.
strix::weight::Status bind_runtime(const strix::weight::RuntimeAdapter&, void* stream,
    std::uint32_t device, strix::weight::ExecutionContext&, strix::weight::Hardware&);
}
