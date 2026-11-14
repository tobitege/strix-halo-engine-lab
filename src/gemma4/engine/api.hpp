#pragma once

#include "chat_runner.hpp"
#include "strix/serve/http_server.hpp"
#include "strix/serve/deployment.hpp"
#include "strix/serve/session_runtime.hpp"

namespace strix::gemma4::engine {

// Family composition of production serve::ApiService and SessionRuntime.
// Chat jobs use the shared session owner; feature exports share backend access.
// Socket handling stays in src/serve.
// Owning, trusted library metadata. No request wire/session identity inference.
struct TurnCacheRequest {
    session::TurnMetadata metadata;
    strix::serve::CacheRetention retention = strix::serve::CacheRetention::retain;
};
struct TurnCacheConfig {
    session::Policy policy = [] { session::Policy p; p.enabled=false; return p; }();
    // Finite payload ceiling per entry and separate cache-work allocation budget.
    // The latter covers eager copies for all configured slots, not process RSS.
    std::uint64_t max_state_bytes = 0, max_transient_bytes = 0;
    std::function<std::optional<TurnCacheRequest>(const strix::serve::HttpRequest&, const ChatRequest&)> request;
};
class Api {
public:
    enum class Owner { standalone, external };
    Api(Backend& backend, std::shared_ptr<const gemma4::serve::Tokenizer> tokenizer,
        DraftFactory drafts = {}, DraftMode mode = DraftMode::automatic, std::string model = "nemotron-bolt-e4b",
        Request defaults = {}, strix::serve::DeploymentCallbacks deployment = {}, Owner owner = Owner::standalone, TurnCacheConfig turn_cache = {},
        std::size_t slots = 0);
    ~Api();
    void handle(const strix::serve::HttpRequest& request, strix::serve::HttpResponseWriter& writer);
    strix::serve::SessionRuntime& runtime();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Chat jobs that decode together in serve: STRIX_G4_SERVE_SLOTS (1..8, default 8), or 1 with
// STRIX_G4_SERVE_BATCH=0 (one chat job at a time). Api's `slots` argument (a deployment plan's) replaces the variable.
[[nodiscard]] std::size_t serve_slots(std::size_t requested = 0);
ChatRequest parse_chat_request(const strix::serve::Json& json, DraftMode default_mode = DraftMode::automatic,
    const Request& defaults = {});

} // namespace strix::gemma4::engine
