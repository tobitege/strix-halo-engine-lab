#pragma once
// Intel backend layer (layer 2 of MODULAR BY DEVICE): the same shared core interfaces the HIP backend implements
// (dense, grouped MoE, attention, norms, weight decode), on SYCL / oneAPI over Level Zero. The header is plain C++:
// no SYCL type crosses it, so host code, tests and the serve layer compile without the oneAPI toolchain. The queue is
// an opaque handle around an in-order sycl::queue (the hipStream_t of this backend).
// Arch specialisations (layer 3) live in src/intel/arch/<arch> and are chosen by Capabilities, never by model.
#include "dense_q8a8_ref.hpp"

#include <cstddef>
#include <cstdint>
#include <string>

namespace strix::intel {

enum class Status { ok, not_built, unsupported, invalid, device_error };
[[nodiscard]] const char* status_name(Status status) noexcept;

// USM kinds. device = VRAM on Arc B (or carved LPDDR on an iGPU); shared = migratable, the default for weights on an
// iGPU where host and device share LPDDR5X (no copy, as on Strix Halo); host = pinned host memory for staging.
enum class MemoryKind { device, shared, host };

struct Capabilities {
    std::string arch;                // "xe2-bmg", "xe2-lnl", "xe3-ptl"; empty = unknown (reference paths only)
    std::uint32_t subgroup = 0;      // preferred sub-group width (16 on Xe2)
    std::uint32_t compute_units = 0; // Xe-cores x EUs as reported
    std::uint64_t global_bytes = 0;
    std::uint64_t local_bytes = 0;   // SLM per work-group
    bool integrated = false;         // host-unified memory
    bool xmx = false;                // joint_matrix / DPAS available
};

struct Queue {
    // Opaque backend state. The successful opener owns it and must close exactly once.
    // Copies are borrowed handles: never close a copy or use one after the owner closes.
    // Calls on a handle must finish before close; synchronize queued USM users before release.
    void* impl = nullptr;
};

// Open requires an empty handle; failure leaves it and capabilities unchanged.
// Asynchronous failures are sticky and reported by synchronize; close cannot report errors.
[[nodiscard]] bool sycl_built() noexcept;  // true when this build carries the SYCL backend
[[nodiscard]] Status open_queue(int level_zero_ordinal, Queue* queue, Capabilities* caps);
void close_queue(Queue& queue) noexcept;
[[nodiscard]] Status allocate(Queue queue, MemoryKind kind, std::size_t bytes, void** out);
[[nodiscard]] Status release(Queue queue, void* pointer);
[[nodiscard]] Status copy(Queue queue, void* dst, const void* src, std::size_t bytes);
[[nodiscard]] Status synchronize(Queue queue);

// Shared dense core, policy dense-q8a8-f16-v1. Reference kernel today; arch/xe2 adds the XMX path behind the same call.
[[nodiscard]] Status enqueue_dense_q8a8(Queue queue, const dense::Q8A8View& view);

} // namespace strix::intel
