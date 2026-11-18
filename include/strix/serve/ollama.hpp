#pragma once

// Ollama-compatible API contract over the shared serve path. The adapter owns no inference path: a request is
// translated into the OpenAI-shaped body the chat/completions handlers already take, runs through them, and their
// output is re-encoded (JSON or NDJSON). Prompt IDs, committed tokens, admission and counters are therefore the ones
// of the equivalent OpenAI request. Options that would change the result and that the engine cannot honour are
// refused with a named error; options without any effect on the result are reported in a response header.

#include "strix/serve/http_server.hpp"
#include "strix/serve/json.hpp"

#include <chrono>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace strix::serve::ollama {

// A refused request: HTTP status, a stable code and the message the client sees as {"error": message}.
class RequestError : public std::runtime_error {
public:
    RequestError(int status, std::string code, const std::string& message)
        : std::runtime_error(message), status_(status), code_(std::move(code)) {}
    [[nodiscard]] int status() const noexcept { return status_; }
    [[nodiscard]] const std::string& code() const noexcept { return code_; }
private:
    int status_;
    std::string code_;
};

// What the served engine can execute; the adapter announces and accepts nothing beyond it.
struct Served {
    std::string model_id;
    std::uint64_t context_tokens = 0;
    bool images = false;    // an image media adapter is loaded
    bool thinking = false;  // the template has a thinking mode
    bool embeddings = false; // the engine executes an embedding route (none today)
};

enum class Route { chat, generate };

// The OpenAI-shaped body plus what the translation decided.
struct Translated {
    Json body;
    bool stream = true;                     // the Ollama default
    bool raw = false;                       // /api/generate raw=true -> /v1/completions
    bool load_only = false;                 // no prompt/messages: answer done_reason "load", run nothing
    std::uint64_t context_ceiling = 0;      // options.num_ctx below the served context, else 0
    std::vector<std::string> no_effect;     // accepted options that cannot change the result (header)
};

// /api/chat and /api/generate request -> OpenAI body. Throws RequestError.
[[nodiscard]] Translated translate(Route route, const Json& request, const Served& served);

// Process switch (STRIX_SERVE_OLLAMA, default on); ApiConfig::ollama_api is the per-service switch.
[[nodiscard]] bool enabled_by_default();

// Model name check shared by every route that names a model ("name" accepted as an alias for /api/show).
void require_model(const Json& request, const Served& served, bool allow_name_alias = false);

// Metadata documents.
[[nodiscard]] Json tags(const Served& served, std::int64_t modified_unix);
[[nodiscard]] Json show(const Served& served);
[[nodiscard]] Json ps(const Served& served);
[[nodiscard]] std::string error_body(const std::string& message);
[[nodiscard]] std::string rfc3339_now();

// Re-encodes the output of the OpenAI handlers. Non-streamed: send() receives the OpenAI JSON. Streamed: write()
// receives SSE frames; every "data:" frame becomes zero or more NDJSON lines; keepalive comments and progress frames
// are dropped (NDJSON has no comment form; client_gone() still reaches the generation).
class ResponseWriter final : public HttpResponseWriter {
public:
    ResponseWriter(HttpResponseWriter& inner, Route route, std::string model, std::vector<std::string> no_effect);
    void send(int status, std::string_view content_type, std::string_view body,
        const std::vector<std::pair<std::string, std::string>>& headers = {}) override;
    bool begin_stream(int status, std::string_view content_type,
        const std::vector<std::pair<std::string, std::string>>& headers = {}) override;
    bool write(std::string_view data) override;
    void end_stream() override;
    [[nodiscard]] bool client_gone() const override { return inner_.client_gone(); }
    void add_response_header(std::string name, std::string value) override { inner_.add_response_header(std::move(name), std::move(value)); }

    // Translation of one OpenAI SSE payload (exposed for the host test and the overhead budget).
    [[nodiscard]] std::string translate_frame(std::string_view payload);

private:
    struct ToolCall { std::string name, arguments; };
    [[nodiscard]] Json line(bool done) const;
    [[nodiscard]] std::string final_lines();
    void timing_from(const Json& out);
    void finish(Json& done) const;

    HttpResponseWriter& inner_;
    Route route_;
    std::string model_;
    std::vector<std::string> no_effect_;
    std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now(), first_{};
    std::string pending_;
    std::map<std::int64_t, ToolCall> tools_;
    std::string finish_ = "stop";
    std::int64_t prompt_tokens_ = -1, output_tokens_ = -1;
    double prefill_ms_ = -1, decode_ms_ = -1;
    bool failed_ = false, finished_ = false;
};

}  // namespace strix::serve::ollama
