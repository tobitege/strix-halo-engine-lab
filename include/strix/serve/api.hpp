#pragma once

#include "strix/serve/bpe_tokenizer.hpp"
#include "strix/serve/chat_template.hpp"
#include "strix/serve/engine.hpp"
#include "strix/serve/http_server.hpp"
#include "strix/serve/json.hpp"
#include "strix/serve/health.hpp"
#include "strix/serve/deployment.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace strix::serve {

struct ApiConfig {
    DeploymentCallbacks deployment;
    std::shared_ptr<const ServeHealthRecord> health_record;
    HealthProvider health_provider;
    std::string model_id = "qwen3.8-flash-next";
    std::uint32_t default_max_tokens = 4096;
    std::uint32_t http_slots = 0; // resolved capacity; zero leaves engine-specific headers unset
    bool default_thinking = true;            // the template's default (enable_thinking unset)
    std::string api_key;                     // empty: no authentication
    // Requests with temperature > 0 on an engine without sampling: "greedy" runs them greedy and
    // says so in the response, "reject" answers 400.
    std::string sampling_policy = "greedy";
    std::optional<double> default_temperature, default_top_p, default_min_p;
    std::optional<std::uint32_t> default_top_k;
    bool capabilities = true;
    bool responses = true;
    bool stream_receipts = true;
    // Resolved engine support: enable only after every constrained selection path executes it.
    bool engine_logprobs = false, engine_logit_transforms = false;
    bool engine_constraints = false;
    bool engine_thinking_budget = false;
    // Anthropic thinking.budget_tokens: false (default) = accepted and stated as not enforced (the request keeps speculative
    // decoding); true = enforced as thinking-budget-v1 (the request decodes on the serial host route).
    bool enforce_anthropic_thinking_budget = false;
    // From GenerationEngine::constraint_execution_routes(); the first route handles new requests.
    std::vector<std::string> constraint_execution_routes;
    double sse_keepalive_seconds = 10.0;      // 0: no keepalive comments
    double sse_progress_seconds = 1.0;       // 0: no progress comments
    bool engine_progress = false;           // engine reports completed prefill units
    std::function<void(Json&)> capabilities_extra; // optional engine members, included in the content hash
    std::vector<std::uint32_t> stop_tokens;  // generation ends at any of these (not emitted)
    std::size_t recent_requests = 50;
    std::string engine_description;          // free text for /api/status
    // Model family hooks; unset: the Qwen3.8 chat template, its output parser and its <tool_call> marker.
    std::function<std::string(const ChatTemplateInput&)> chat_renderer;
    std::function<AssistantOutput(const std::string& text, bool thinking, const Json& tools)> output_parser;
    std::string tool_call_marker = "<tool_call>";  // streaming: content from this marker on is kept for the tool parse
    // Engine members added to /api/status (memory, slots, group) and lines added to /metrics.
    std::function<void(Json& status)> status_extra;
    std::function<std::string()> metrics_extra;
    // POST /admin/drain (authenticated): stop admitting and finish what runs; /health answers 503 "draining" after it.
    // Optional engine notification; shared admission draining is always installed.
    std::function<void()> on_drain;
    // An arming request to this service (GET /metrics, GET /api/status?detail=1) keeps the instrumentation armed this
    // many seconds more (observe_arm); 0: until the process ends. Unset: STRIX_SERVE_OBSERVE_EXPIRY_S, else 60.
    std::optional<double> observe_expiry_seconds;
    // Further engine routes (authenticated), tried before the built-in POST routes: true when the hook answered.
    // Unset: nothing changes.
    bool sessions_enabled = false;
    std::function<std::string()> state_cache_stats;
    std::function<bool(const HttpRequest&, HttpResponseWriter&)> engine_routes;
};

// OpenAI-compatible (/v1/chat/completions, /v1/completions, /v1/responses, /v1/models) and Anthropic-compatible
// (/v1/messages) endpoints over one generation queue, plus /health, /metrics (Prometheus text),
// /api/status (JSON) and /tokenize, /detokenize.
class ApiService {
public:
    ApiService(ApiConfig config, std::shared_ptr<const BpeTokenizer> tokenizer, GenerationQueue& queue,
        std::uint64_t context_tokens, bool engine_samples);
    ~ApiService();
    void handle(const HttpRequest& request, HttpResponseWriter& response);
    // Prompt tokens of a chat request body exactly as the chat route builds them (for tests and tools).
    [[nodiscard]] std::vector<std::uint32_t> chat_prompt(const Json& body) const;
    void set_state(std::string state);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::serve
