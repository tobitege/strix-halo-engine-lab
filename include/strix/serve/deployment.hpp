#pragma once

#include "strix/serve/engine.hpp"
#include "strix/serve/http_server.hpp"
#include "strix/serve/json.hpp"

#include <functional>
#include <optional>
#include <string>

namespace strix::serve {

struct ServeCapacity {
    std::uint64_t slots_total = 0, slots_busy = 0;
    // Unknown is null, never an invented amount of allocatable KV memory.
    std::optional<std::uint64_t> kv_bytes_free;
    std::uint64_t queue_depth = 0;
};

// Bind Plan::to_json() and Plan::hash() without depending on deployment resolver types.
// Install before serving; callbacks must be safe on concurrent HTTP threads.
struct DeploymentCallbacks {
    std::function<std::string()> deployment_json;
    std::function<std::string()> deployment_id;
    std::function<ServeCapacity()> capacity_provider;
};

// Keep direct backend requests in the same drain accounting without changing generation counters.
class QueueExternalWork {
public:
    explicit QueueExternalWork(GenerationQueue& queue) : queue_(queue.try_begin_external_work() ? &queue : nullptr) {}
    ~QueueExternalWork() { if (queue_) queue_->finish_external_work(); }
    explicit operator bool() const { return queue_ != nullptr; }
    QueueExternalWork(const QueueExternalWork&) = delete;
    QueueExternalWork& operator=(const QueueExternalWork&) = delete;
private:
    GenerationQueue* queue_;
};

void deployment_health(Json& health, const DeploymentCallbacks& callbacks);
void deployment_status(Json& status, const DeploymentCallbacks& callbacks,
    const GenerationQueue& queue, std::uint64_t slots_total);
// Call after the server's authorization check. Does not terminate or cancel running work.
bool handle_deployment_route(const HttpRequest& request, HttpResponseWriter& response,
    GenerationQueue& queue, const DeploymentCallbacks& callbacks,
    const std::function<void()>& on_drain = {});

} // namespace strix::serve
