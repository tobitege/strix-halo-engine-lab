#pragma once

#include "strix/serve/tokenizer.hpp"
#include "strix/serve/chat_template.hpp"
#include "strix/serve/engine.hpp"
#include "strix/serve/http_server.hpp"
#include "strix/serve/json.hpp"
#include "strix/serve/health.hpp"
#include "strix/serve/deployment.hpp"
#include "strix/serve/media.hpp"
#include "strix/session/embed_rows.hpp"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace strix::serve {

// Extension routes use the shared error schema and HTTP status handling.
struct ApiRouteError : std::runtime_error {
    ApiRouteError(int status_value, const std::string& message) : std::runtime_error(message), status(status_value) {}
    int status;
};

// ---- media input (images today; other kinds register the same way) ----
// One media item as the language model sees it.
struct ApiMediaSpan {
    session::EmbedLease lease;         // the interned rows; the request's EmbedTable holds it
    std::vector<std::uint32_t> layout; // ids in prompt order: lease.ids() plus real special token ids
    std::uint32_t grid_h = 0, grid_w = 0;
    double preprocess_ms = 0.0, encode_ms = 0.0;
    bool cache_hit = false;
};
// Produces the embedding rows of one media kind. The service loads one per kind it accepts; the API layer never
// names a model family, a tower or a file format beyond what this interface states.
//
// Contract of a set of adapters:
//  - Handles: an EmbedTable is keyed by handle, so every adapter of one service interns in the SAME
//    session::EmbedInterner (one handle space per process). A request whose items come back with one handle
//    for two digests is refused (500) instead of resolving rows of the wrong item.
//  - Threads: one ApiService calls encode() of its adapters one at a time. An adapter shared by two services
//    (or called by anything else) must synchronise itself.
class ApiMediaAdapter {
public:
    virtual ~ApiMediaAdapter() = default;
    // The canonical part type this adapter consumes: {"type":<kind>,"media":k} (serve::media). "image" today.
    [[nodiscard]] virtual std::string kind() const { return "image"; }
    // Weights digest + arithmetic id: part of /v1/models and of the capabilities hash.
    [[nodiscard]] virtual std::string identity() const = 0;
    // Text that tokenizes to exactly one id; that id marks where the rows of one item go.
    [[nodiscard]] virtual std::string placeholder() const = 0;
    // What the renderer receives for the ordinal-th item of the request (0-based, per kind); it must hold the
    // placeholder exactly once. Default: the placeholder alone.
    [[nodiscard]] virtual std::string part_text(std::size_t ordinal) const { static_cast<void>(ordinal); return placeholder(); }
    // Upper bound of the virtual rows of one item.
    [[nodiscard]] virtual std::uint32_t max_rows() const = 0;
    // Preprocess, encode (or hit a cache) and intern one item. row_budget is what the request may still spend
    // (min(max_rows, remaining context)). Throws media::MediaError for content the client must fix.
    [[nodiscard]] virtual ApiMediaSpan encode(const media::MediaItem& item, std::uint32_t row_budget) = 0;
    // The interner epoch (session::EmbedInterner::epoch). When it moves, cached state holding virtual ids of an
    // earlier epoch must be withdrawn before another media request runs (ApiVisionConfig::on_epoch).
    [[nodiscard]] virtual std::uint64_t epoch() const { return 0; }
    // Capability and receipt names of the kind: <input_capability> : true, strix.<receipt_name>.
    [[nodiscard]] virtual std::string input_capability() const { return kind() + "_input"; }
    [[nodiscard]] virtual std::string receipt_name() const { return kind() == "image" ? "vision" : kind(); }
};
struct ApiVisionConfig {
    // Limits and fetch policy of serve::media. Remote fetch is OFF unless the service sets it
    // (media::fetch_policy_from_env()); no fetcher is built in.
    media::Options media;
    // Row cap of one request over all items; 0: only the context bounds it.
    std::uint32_t max_request_rows = 0;
    // Loaded adapters, at most one per kind. Empty: media parts keep the engine's own 400.
    // Adapters belong to one service unless they synchronize shared encode/epoch state themselves.
    // All kinds must share one handle namespace (interner); independent handles cannot be combined.
    std::vector<std::shared_ptr<ApiMediaAdapter>> adapters;
    // Called once per epoch change before the next media request is admitted; the service withdraws cached
    // prefixes that hold virtual ids. Unset: media requests answer 503 after a wrap (text is unaffected).
    // The callback must synchronize with the cache owner and finish withdrawal before returning;
    // the service media mutex does not protect runtime caches.
    std::function<void(std::uint64_t epoch)> on_epoch;
};
struct ApiChatPrompt {
    std::vector<std::uint32_t> tokens;
    bool thinking = false;
    Json tools;
    std::uint32_t stable_prefix = 0;
    std::shared_ptr<GenerationContext> context = {};
    // Resolves the virtual ids of tokens; null for a prompt without media rows.
    std::shared_ptr<const session::EmbedTable> embeds = {};
    bool tools_off = false;  // tool_choice "none" (set by the shared layer): a call the model still emits is withheld
};

// Terminal domain data is frozen after the accepted ticket finishes. Hooks never
// receive a response writer and cannot own protocol delivery or queue work.
struct ApiTerminalData {
    Json timings;
    std::optional<double> max_gap_ms;
};
struct ApiRequestObservation {
    std::string request_id, completion_id, route;
    std::int64_t time = 0;
    bool stream = false, delivered = true, observed = false, refused = false;
    std::uint64_t prompt_tokens = 0;
    GenerationStats stats;
    std::string error;
    ApiTerminalData terminal;
};

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
    // tool_choice "none" after a history with tool calls: the closing instruction added to the prompt.
    // Unset: serve::no_tools_hint. Requests without tool_choice "none" never see it.
    std::optional<std::string> no_tools_hint = {};
    // tool_choice "none": the reply is cut from this marker on (a family whose call frame differs from
    // tool_call_marker, or whose parser leaves calls in the text). Empty: tool_call_marker.
    std::string tools_off_marker = {};
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
    // Hooks are host-only, before queue admission. prepare_chat renders once and
    // returns owning typed context; prepare_job retains it across continuation.
    std::function<ApiChatPrompt(const Json&)> prepare_chat = {};
    std::vector<std::string> chat_extension_fields = {}; // copied through shared protocol normalization
    std::function<void(const HttpRequest&, const Json&)> validate_request = {};
    std::function<void(const HttpRequest&, const Json&, GenerationJob&)> prepare_job = {};
    std::optional<std::uint64_t> default_seed = {};
    double maximum_temperature = 2.0;
    bool float_sampling_parameters = false;
    bool strict_context_limit = false;
    bool strict_request_json = false;
    // Existing compact chat envelope conventions, selected only by an adopter.
    bool compact_chat_envelope = false;
    std::string chat_id_prefix = "chatcmpl-";
    // Historical OpenAI convention: timings after usage, optional IDs only,
    // no role-only SSE event, content-only live chat, cancelled => stop.
    bool timed_completion_envelope = false;
    bool exact_max_tokens = false; // exact integer spelling; clamp to full context
    bool minimal_metadata = false;
    std::optional<std::uint64_t> status_slots = {}; // capacity metadata without queue response headers
    bool explicit_thinking_only = false; // normalization preserves an absent family control
    bool tokenize_text_field = false; // text alias, token array without count
    bool session_error_envelope = false;
    std::function<ApiTerminalData(const GenerationTicket&)> terminal_data = {};
    // Replaces the built-in recent-request owner; called once after delivery and
    // terminal cleanup, or on queue refusal. Validation creates no observation.
    std::function<void(const ApiRequestObservation&)> request_finished = {};
    std::function<Json(const QueueSnapshot&, bool stopping, bool detail)> status_data = {};
    std::function<std::string(const QueueSnapshot&)> metrics_data = {};
    // Authenticated extension returning domain JSON through shared framing.
    std::function<std::optional<Json>(const HttpRequest&)> route_data = {};
    // Media input: see ApiVisionConfig. Default: no adapter, nothing changes.
    ApiVisionConfig vision = {};

};

// OpenAI-compatible (/v1/chat/completions, /v1/completions, /v1/responses, /v1/models) and Anthropic-compatible
// (/v1/messages) endpoints over one generation queue, plus /health, /metrics (Prometheus text),
// /api/status (JSON) and /tokenize, /detokenize.
class ApiService {
public:
    ApiService(ApiConfig config, std::shared_ptr<const BpeTokenizer> tokenizer, GenerationQueue& queue,
        std::uint64_t context_tokens, bool engine_samples);
    ApiService(ApiConfig config, ApiTokenizerBinding tokenizer, GenerationQueue& queue,
        std::uint64_t context_tokens, bool engine_samples);
    ~ApiService();
    void handle(const HttpRequest& request, HttpResponseWriter& response);
    // Prompt tokens of a chat request body exactly as the chat route builds them (for tests and tools).
    // A body with media parts is refused here (std::invalid_argument): its virtual ids mean nothing without
    // the table that prepare_chat_prompt returns with them.
    [[nodiscard]] std::vector<std::uint32_t> chat_prompt(const Json& body) const;
    // The same prompt with what owns it: tokens, stable prefix and, for media, the EmbedTable of its virtual ids.
    [[nodiscard]] ApiChatPrompt prepare_chat_prompt(const Json& body) const;
    void set_state(std::string state);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace strix::serve
