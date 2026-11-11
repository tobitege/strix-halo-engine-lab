#pragma once

#include "chat_runner.hpp"
#include "strix/serve/http_server.hpp"
#include "strix/serve/deployment.hpp"

namespace strix::gemma4::engine {

// Portable family handler over shared HTTP/JSON/GenerationQueue primitives.
// Chat jobs use one FIFO worker; feature exports share exclusive backend access.
// Socket handling stays in src/serve.
class Api {
public:
    Api(Backend& backend, std::shared_ptr<const gemma4::serve::Tokenizer> tokenizer,
        DraftFactory drafts = {}, DraftMode mode = DraftMode::automatic, std::string model = "nemotron-bolt-e4b",
        Request defaults = {}, strix::serve::DeploymentCallbacks deployment = {});
    ~Api();
    void handle(const strix::serve::HttpRequest& request, strix::serve::HttpResponseWriter& writer);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
// Chat jobs that decode together in serve: STRIX_G4_SERVE_SLOTS (1..8, default 8), or 1 with
// STRIX_G4_SERVE_BATCH=0 (one chat job at a time).
[[nodiscard]] std::size_t serve_slots();
ChatRequest parse_chat_request(const strix::serve::Json& json, DraftMode default_mode = DraftMode::automatic,
    const Request& defaults = {});

} // namespace strix::gemma4::engine
