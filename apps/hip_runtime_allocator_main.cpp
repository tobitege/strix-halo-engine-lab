#include "strix/device/hip_gate.hpp"
#include "strix/hip_check.hpp"
#include "strix/compiled_execution.hpp"
#include "strix/hip_runtime_allocator.hpp"
#include "strix/session_state_layout.hpp"

#include <hip/hip_runtime.h>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

using strix::hip_check;

}  // namespace

int main(int argc, char** argv) try {
    // Start gate: resolves the declared device (--devices FILE, --device-id ID, STRIX_DEVICES, STRIX_DEVICE), refuses an
    // excluded or undeclared one and prints the "device:" receipt line. Without a declaration the single integrated
    // GPU of a node keeps working (declared=implicit).
    strix::device::bind_hip_main(argc, argv);
    const bool pinned =
        argc == 2 &&
        std::string(argv[1]) ==
            "--pinned-host";

    hipDeviceProp_t properties{};
    hip_check(
        hipGetDeviceProperties(
            &properties, 0),
        "hipGetDeviceProperties");

    strix::CompiledSerialExecution compiled;
    compiled.execution_identity =
        0x1234ULL;
    compiled.arena.alignment = 256;
    compiled.arena.total_bytes =
        4ULL * 1024ULL * 1024ULL;

    strix::SerialSessionStateLayout state;
    state.layout_identity =
        0x5678ULL;
    state.device_alignment = 256;
    state.host_alignment = 64;
    state.device_total_bytes =
        16ULL * 1024ULL * 1024ULL;
    state.host_total_bytes = 4096;

    strix::HipRuntimeAllocator allocator(
        pinned
            ? strix::HipHostAllocationMode::pinned_mapped
            : strix::HipHostAllocationMode::pageable_aligned);
    strix::AllocationGenerationSource generations;

    auto memory =
        strix::RuntimeExecutionMemoryOwner::allocate(
            compiled,
            state,
            allocator,
            generations);

    hip_check(
        hipMemset(
            reinterpret_cast<void*>(
                memory.scratch().arena_base),
            0,
            static_cast<std::size_t>(
                memory.scratch().arena_bytes)),
        "hipMemset scratch");
    hip_check(
        hipMemset(
            reinterpret_cast<void*>(
                memory.state().device_base),
            0,
            static_cast<std::size_t>(
                memory.state().device_bytes)),
        "hipMemset state");

    std::cout
        << "{\n"
        << "  \"device\": \""
        << properties.name
        << "\",\n"
        << "  \"gfx_arch\": \""
        << properties.gcnArchName
        << "\",\n"
        << "  \"host_mode\": \""
        << (pinned
                ? "pinned_mapped"
                : "pageable_aligned")
        << "\",\n"
        << "  \"scratch_bytes\": "
        << memory.scratch().arena_bytes
        << ",\n"
        << "  \"state_device_bytes\": "
        << memory.state().device_bytes
        << ",\n"
        << "  \"state_host_bytes\": "
        << memory.state().host_bytes
        << ",\n"
        << "  \"scratch_generation\": "
        << memory.scratch().allocation_generation
        << ",\n"
        << "  \"state_device_generation\": "
        << memory.state().device_allocation_generation
        << ",\n"
        << "  \"state_host_generation\": "
        << memory.state().host_allocation_generation
        << "\n"
        << "}\n";

    return 0;
} catch (const std::exception& error) {
    std::cerr
        << "strix-hip-runtime-allocator: "
        << error.what()
        << '\n';
    return 1;
}
