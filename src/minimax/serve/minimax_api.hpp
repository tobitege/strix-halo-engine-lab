#pragma once

// The chat-completions HTTP API of strix-minimax-serve (rank 0), host only: it renders prompts, submits jobs to the
// GenerationQueue that the engine thread drains, streams or returns the reply and serves the observability routes.
// It never touches the engine; the engine thread only sees GenerationQueue jobs.
//
// Routes: POST /v1/chat/completions, POST /v1/completions (stream or not), GET /v1/models, GET /health,
// GET /metrics (metrics text), GET /api/status[?detail=1] (strix-serve-status-v1), POST /tokenize; with the
// scheduler's hook, GET / POST /admin/sched. Decoding is greedy; sampling parameters are accepted and ignored (said in
// the reply's timings, which also carry the scheduler's per-request measurements, RequestExtra).
//
// Request fields beyond the standard ones: "return_token_ids": true adds strix.token_ids (every generated id in order,
// the stop token included, so its length is usage.completion_tokens) to the reply, and in a stream to the final chunk
// (the one with finish_reason) and to the usage chunk; stream_options.include_usage adds a chunk with empty choices and
// the usage after the final chunk. Request ids: an inbound X-Request-Id of 8-64 characters [A-Za-z0-9._-] names the
// request in the status ring and is echoed in X-Request-Id on every response; a generation request that reaches the
// queue without one echoes its completion id (requests rejected before that, e.g. malformed JSON, echo only an
// accepted inbound id).

#include "minimax/serve/minimax_observe.hpp"
#include "strix/serve/engine.hpp"
#include "strix/serve/http_server.hpp"
#include "strix/serve/json.hpp"
#include "strix/serve/deployment.hpp"

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace strix::minimax::serve {

class MinimaxTokenizer;

// The tokenizer as the HTTP API uses it.
struct ApiTokenizer {
    std::uint32_t eos_id = 0;
    std::function<std::vector<std::uint32_t>(std::string_view)> encode;  // special tokens parsed
    std::function<std::string(std::span<const std::uint32_t>)> decode;   // special tokens skipped
    std::function<std::string(std::uint32_t)> stream_bytes;              // the bytes a stream writes for one id (none for specials)
};
[[nodiscard]] ApiTokenizer api_tokenizer(std::shared_ptr<const MinimaxTokenizer> tokenizer);

// The serve scheduler's per-request measurements beyond GenerationStats (zero when it reported none): time to the
// first emitted token and the longest gap between two emitted tokens (submit-relative / engine emission times, not
// HTTP delivery), the decode time stalled by prefill units, prefill wall time and units, draft cycles, the FNV-1a
// digests of the emitted ids and of the prompt, and whether the slot kept a drafter context.
struct RequestExtra {
    double ttft_ms = 0, max_gap_ms = 0, stall_ms = 0, prefill_wall_ms = 0;
    std::uint64_t units = 0, cycles = 0, drafted = 0, accepted = 0;
    std::uint64_t ids_fnv = 0, prompt_fnv = 0;
    bool context = false;
    std::string dspark_mode = "auto", dspark_reason = "plain:prior_unqualified";
    std::string sched = "off", fallback_reason;
    double service_ms = 0;
    unsigned max_unit_layers = 0;
    bool short_first = false;
    bool nested = false;
    std::uint64_t task_id = 0, parent_request = 0, nest_count = 0;
    double park_wall_ms = 0, nest_service_ms = 0, nest_switch_ms = 0;
};

// The reply's "timings" object (non-stream reply and a stream's final chunk), keys in this order: queue_ms,
// prefill_ms, decode_ms, decode_tok_s ((tokens - 1) / decode s), completion_tokens, ttft_ms, max_gap_ms, stall_ms,
// prefill_units, cycles, drafted, accepted, ids_fnv, prompt_fnv (16 hex), scheduler policy fields, sampling.
[[nodiscard]] strix::serve::Json reply_timings(const strix::serve::GenerationStats& stats, const RequestExtra& extra);

struct MinimaxApiOptions {
    strix::serve::DeploymentCallbacks deployment;
    std::uint64_t http_slots = 0;
    bool session_enabled = false;
    std::string model_id = "minimax-m3.1-flash";
    std::uint32_t default_max_tokens = 4096;
    std::uint64_t context_tokens = 0;
    // The serve scheduler's side of the API (each may be empty: no scheduler, as in tests).
    // take_extra: the scheduler's measurements of a finished ticket, removed from its table; nullopt when it has none
    //   (a request cancelled while queued never reached the scheduler). Called once per submitted request, after its
    //   ticket has finished.
    std::function<std::optional<RequestExtra>(const strix::serve::GenerationTicket&)> take_extra;
    // legacy_metrics: the mm_* value lines of GET /metrics for this queue snapshot (default legacy_metric_lines);
    //   their HELP/TYPE come from the observer's table.
    std::function<std::string(const strix::serve::QueueSnapshot&)> legacy_metrics;
    // admin_sched: GET / POST /admin/sched (any method) -> the JSON reply (an exception is a 400); empty = no route.
    std::function<strix::serve::Json(const strix::serve::HttpRequest&)> admin_sched;
};

class MinimaxApi {
public:
    MinimaxApi(ApiTokenizer tokenizer, strix::serve::GenerationQueue& queue, MinimaxApiOptions options, Observer& observer);
    void handle(const strix::serve::HttpRequest& request, strix::serve::HttpResponseWriter& response);

private:
    void generate(const strix::serve::HttpRequest& request, strix::serve::HttpResponseWriter& response,
                  std::vector<std::uint32_t> prompt, const strix::serve::Json& body, bool chat, const std::string& inbound_id);

    ApiTokenizer tok_;
    strix::serve::GenerationQueue& queue_;
    MinimaxApiOptions options_;
    Observer& observer_;
    std::atomic<std::uint64_t> next_id_{1};
};

}  // namespace strix::minimax::serve
