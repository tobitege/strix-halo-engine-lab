#pragma once
#include <strix/weight/runtime.hpp>
namespace strix::cuda::weight {
// Source registry remains empty until this backend's operand migration.
strix::weight::Status bind_runtime(const strix::weight::RuntimeAdapter&, void* stream,
    std::uint32_t device, strix::weight::ExecutionContext&, strix::weight::Hardware&);
strix::weight::Status bind_native_runtime(void* stream, std::uint32_t device,
    strix::weight::ExecutionContext&, strix::weight::Hardware&);
// Terminal shutdown: call after all contexts/caches stop submitting work.
void shutdown_native_runtime();
}
