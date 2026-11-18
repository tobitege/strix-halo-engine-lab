#pragma once
#include "cache.hpp"
namespace strix::weight {
// Backend-owned primitives; vendor SDK objects never enter the shared headers.
// A migration installs its own allocation/event/copy and capability callbacks.
struct RuntimeAdapter {
    std::shared_ptr<const RuntimeCallbacks> callbacks;
    std::function<Status(std::uint32_t, Hardware&)> query_hardware;
};
// Deployment appends the exact flags when matching an imported measured table.
std::string current_compiler_id();
Status bind_runtime(Backend, const RuntimeAdapter&, void* stream, std::uint32_t device,
                    ExecutionContext&, Hardware&);
}
