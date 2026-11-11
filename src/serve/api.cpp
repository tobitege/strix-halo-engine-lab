#include "strix/serve/api.hpp"

#include "strix/serve/chat_template.hpp"
#include "strix/serve/json_constraint.hpp"
#include "strix/serve/request_fields.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <charconv>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <limits>
#include <mutex>
#include <optional>
#include <random>
#include <sstream>
#include <set>

namespace strix::serve {
namespace {

using Headers = std::vector<std::pair<std::string, std::string>>;

class ApiError : public std::runtime_error {
public:
    ApiError(int status, std::string type, const std::string& message, std::string param = {}, std::string code = {}, std::string keyword = {})
        : std::runtime_error(message), status_(status), type_(std::move(type)), param_(std::move(param)), code_(std::move(code)), keyword_(std::move(keyword)) {}
    Json strix;
    const std::string& code() const { return code_; }
    const std::string& keyword() const { return keyword_; }
    int status() const { return status_; }
    const std::string& type() const { return type_; }
    const std::string& param() const { return param_; }
private:
    int status_;
    std::string type_, param_, code_, keyword_;
};

std::string random_id(std::string_view prefix) {
    static std::mutex mutex;
    static std::mt19937_64 rng{std::random_device{}()};
    const std::lock_guard lock(mutex);
    char buffer[40];
    std::snprintf(buffer, sizeof(buffer), "%016llx%08llx", static_cast<unsigned long long>(rng()),
        static_cast<unsigned long long>(rng() & 0xffffffffULL));
    return std::string(prefix) + buffer;
}
std::int64_t unix_now() {
    return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}
// Valid UTF-8 for JSON output: invalid bytes become U+FFFD.
std::string sanitize_utf8(std::string_view text) {
    std::string out;
    out.reserve(text.size());
    for (std::size_t i = 0; i < text.size();) {
        const auto b0 = static_cast<unsigned char>(text[i]);
        std::size_t n = b0 < 0x80U ? 1U : (b0 >= 0xC2U && b0 <= 0xDFU) ? 2U : (b0 >= 0xE0U && b0 <= 0xEFU) ? 3U : (b0 >= 0xF0U && b0 <= 0xF4U) ? 4U : 0U;
        bool valid = n && i + n <= text.size();
        for (std::size_t k = 1; valid && k < n; ++k) valid = (static_cast<unsigned char>(text[i + k]) & 0xC0U) == 0x80U;
        if (valid && n == 3U) {
            const auto b1 = static_cast<unsigned char>(text[i + 1U]);
            if ((b0 == 0xE0U && b1 < 0xA0U) || (b0 == 0xEDU && b1 >= 0xA0U)) valid = false;
        }
        if (valid && n == 4U) {
            const auto b1 = static_cast<unsigned char>(text[i + 1U]);
            if ((b0 == 0xF0U && b1 < 0x90U) || (b0 == 0xF4U && b1 >= 0x90U)) valid = false;
        }
        if (valid) { out.append(text.substr(i, n)); i += n; }
        else { out += "\xEF\xBF\xBD"; ++i; }
    }
    return out;
}

// Holds back the tail that could still become a stop string; reports the first match.
class StopFilter {
public:
    explicit StopFilter(std::vector<std::string> stops) : stops_(std::move(stops)) {
        for (const auto& s : stops_) hold_ = std::max(hold_, s.empty() ? 0U : s.size() - 1U);
    }
    std::string push(std::string_view text) {
        if (matched_) return {};
        buffer_.append(text);
        std::size_t first = std::string::npos;
        for (const auto& s : stops_) {
            if (s.empty()) continue;
            const auto at = buffer_.find(s);
            if (at != std::string::npos && at < first) { first = at; match_ = s; }
        }
        if (first != std::string::npos) {
            matched_ = true;
            auto out = buffer_.substr(0, first);
            buffer_.clear();
            return out;
        }
        if (buffer_.size() <= hold_) return {};
        // Never cut a UTF-8 sequence: move the cut back to a character start.
        auto cut = buffer_.size() - hold_;
        while (cut > 0 && cut < buffer_.size() && (static_cast<unsigned char>(buffer_[cut]) & 0xC0U) == 0x80U) --cut;
        auto out = buffer_.substr(0, cut);
        buffer_.erase(0, cut);
        return out;
    }
    std::string finish() { auto out = matched_ ? std::string{} : buffer_; buffer_.clear(); return out; }
    bool matched() const { return matched_; }
    const std::string& match() const { return match_; }
private:
    std::vector<std::string> stops_;
    std::size_t hold_ = 0;
    std::string buffer_, match_;
    bool matched_ = false;
};

// Splits the assistant text stream into reasoning and content deltas; everything from the first
// <tool_call> on is kept for the final tool-call parse.
class ChatSplitter {
public:
    explicit ChatSplitter(bool thinking, std::string tool_marker = "<tool_call>")
        : phase_(thinking ? Phase::reasoning : Phase::content), tool_marker_(std::move(tool_marker)) {}
    struct Delta { bool reasoning; std::string text; };
    std::vector<Delta> push(std::string_view text) {
        std::vector<Delta> out;
        buffer_.append(text);
        for (bool progress = true; progress;) {
            progress = false;
            if (phase_ == Phase::reasoning) {
                if (reasoning_leading_) {
                    const auto first = buffer_.find_first_not_of(" \t\r\n");
                    if (first == std::string::npos) { buffer_.clear(); break; }
                    buffer_.erase(0, first);
                    reasoning_leading_ = false;
                }
                const auto end = buffer_.find("</think>");
                if (end != std::string::npos) {
                    auto before = buffer_.substr(0, end);
                    const auto last = before.find_last_not_of(" \t\r\n");
                    before.erase(last == std::string::npos ? 0U : last + 1U);
                    emit(out, true, before);
                    buffer_.erase(0, end + 8U);
                    phase_ = Phase::content;
                    leading_ = true;
                    progress = true;
                } else hold_tail(out, true, 7U);
            } else if (phase_ == Phase::content) {
                if (leading_) {
                    const auto first = buffer_.find_first_not_of(" \t\r\n");
                    if (first == std::string::npos) { buffer_.clear(); break; }
                    buffer_.erase(0, first);
                    leading_ = false;
                }
                const auto call = buffer_.find(tool_marker_);
                if (call != std::string::npos) {
                    auto before = buffer_.substr(0, call);
                    const auto last = before.find_last_not_of(" \t\r\n");
                    before.erase(last == std::string::npos ? 0U : last + 1U);
                    emit(out, false, before);
                    tools_ = buffer_.substr(call);
                    buffer_.clear();
                    phase_ = Phase::tools;
                } else hold_tail(out, false, tool_marker_.size() - 1U);
            } else {
                tools_.append(buffer_);
                buffer_.clear();
            }
        }
        return out;
    }
    std::vector<Delta> finish() {
        std::vector<Delta> out;
        if (phase_ == Phase::reasoning) {
            const auto last = buffer_.find_last_not_of(" \t\r\n");
            emit(out, true, last == std::string::npos ? std::string{} : buffer_.substr(0, last + 1U));
            buffer_.clear();
        }
        else if (phase_ == Phase::content) {
            const auto last = buffer_.find_last_not_of(" \t\r\n");
            emit(out, false, last == std::string::npos ? std::string{} : buffer_.substr(0, last + 1U));
            buffer_.clear();
        }
        return out;
    }
    bool saw_tools() const { return phase_ == Phase::tools; }
private:
    enum class Phase { reasoning, content, tools };
    void emit(std::vector<Delta>& out, bool reasoning, const std::string& text) {
        if (!text.empty()) out.push_back({reasoning, text});
    }
    void emit_cut(std::vector<Delta>& out, bool reasoning, std::size_t cut) {
        while (cut > 0 && cut < buffer_.size() && (static_cast<unsigned char>(buffer_[cut]) & 0xC0U) == 0x80U) --cut;
        if (!cut) return;
        emit(out, reasoning, buffer_.substr(0, cut));
        buffer_.erase(0, cut);
    }
    // Emits all but a possible partial marker (`hold` bytes) and the trailing whitespace, which is
    // stripped if the block ends there (as the final parse strips it).
    void hold_tail(std::vector<Delta>& out, bool reasoning, std::size_t hold) {
        auto keep = std::min<std::size_t>(buffer_.size(), hold);
        const auto last = buffer_.find_last_not_of(" \t\r\n");
        if (last == std::string::npos) keep = buffer_.size();
        else keep = std::max(keep, buffer_.size() - last - 1U);
        emit_cut(out, reasoning, buffer_.size() - keep);
    }
    Phase phase_;
    std::string tool_marker_;
    std::string buffer_, tools_;
    bool leading_ = false, reasoning_leading_ = true;
};

// C1: an inbound X-Request-Id is used when it has 8-64 characters of [A-Za-z0-9._-]; anything else is ignored.
bool valid_request_id(std::string_view id) {
    if (id.size() < 8U || id.size() > 64U) return false;
    for (const char c : id)
        if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-')) return false;
    return true;
}
// The accepted X-Request-Id of the request this connection thread is handling (ApiService::handle sets and clears it;
// the ring record of the request is written on the same thread).
thread_local std::string current_request_id;

// "detail=1" among the &-separated items of a query string.
bool detail_requested(std::string_view query) {
    for (std::size_t at = 0; at <= query.size();) {
        const auto end = std::min(query.find('&', at), query.size());
        if (query.substr(at, end - at) == "detail=1") return true;
        at = end + 1U;
    }
    return false;
}
// A label value of the metrics text exposition format.
std::string label_value(std::string_view text) {
    std::string out;
    for (const char c : text) {
        if (c == '\\') out += "\\\\";
        else if (c == '"') out += "\\\"";
        else if (c == '\n') out += "\\n";
        else out += c;
    }
    return out;
}
// {"buckets": [{"le": bound, "count": cumulative}, ..., {"le": "+Inf", "count": count}], "sum", "count"}
template <std::size_t N>
Json histogram_json(const ObserveHistogram<N>& h, const std::array<double, N>& bounds) {
    Json buckets = Json::array();
    std::uint64_t cumulative = 0;
    for (std::size_t i = 0; i < N; ++i) {
        cumulative += h.counts[i];
        Json b = Json::object();
        b.set("le", Json(bounds[i])); b.set("count", Json(cumulative));
        buckets.push(std::move(b));
    }
    Json inf = Json::object();
    inf.set("le", Json("+Inf")); inf.set("count", Json(h.count));
    buckets.push(std::move(inf));
    Json out = Json::object();
    out.set("buckets", std::move(buckets));
    out.set("sum", Json(h.sum));
    out.set("count", Json(h.count));
    return out;
}
template <std::size_t N>
void histogram_text(std::ostringstream& m, const char* name, const char* help, const ObserveHistogram<N>& h, const std::array<double, N>& bounds) {
    m << "# HELP " << name << ' ' << help << "\n# TYPE " << name << " histogram\n";
    std::uint64_t cumulative = 0;
    for (std::size_t i = 0; i < N; ++i) {
        cumulative += h.counts[i];
        char le[32];
        std::snprintf(le, sizeof(le), "%g", bounds[i]);  // the bound as written (the stream prints round-trip digits)
        m << name << "_bucket{le=\"" << le << "\"} " << cumulative << '\n';
    }
    m << name << "_bucket{le=\"+Inf\"} " << h.count << '\n' << name << "_sum " << h.sum << '\n' << name << "_count " << h.count << '\n';
}

struct RecentRequest {
    RecentRequest() = default;
    RecentRequest(std::string i, std::string r, std::string s, std::int64_t t)
        : id(std::move(i)), route(std::move(r)), status(std::move(s)), time(t) {}
    std::string id, route, status; // status: done | cancelled | error
    std::int64_t time = 0;
    std::uint64_t prompt = 0, cached = 0, output = 0, cycles = 0, accepted = 0, proposed = 0;
    double queue_ms = 0, prefill_ms = 0, decode_ms = 0;
    std::string stop_reason;
    // armed: an observed request (submitted after arming: its measurements are complete); its first token as the
    // engine reported it; C8 imported tokens (kept for every record: the prefill rate excludes them; a status field only
    // when observed); the response body's id when an inbound X-Request-Id names the record
    bool armed = false;
    std::optional<double> first_token_ms;
    std::uint64_t imported = 0;
    std::string response_id;
};

std::vector<std::string> stop_strings(const Json* stop) {
    std::vector<std::string> out;
    if (!stop || stop->is_null()) return out;
    if (stop->is_string()) { out.push_back(stop->as_string()); return out; }
    if (!stop->is_array()) throw ApiError(400, "invalid_request_error", "stop must be a string or an array of strings", "stop");
    for (const auto& s : stop->as_array()) {
        if (!s.is_string()) throw ApiError(400, "invalid_request_error", "stop must be a string or an array of strings", "stop");
        if (!s.as_string().empty()) out.push_back(s.as_string());
    }
    if (out.size() > 16U) throw ApiError(400, "invalid_request_error", "at most 16 stop strings", "stop");
    return out;
}

bool bool_or(const Json& body, std::string_view key, bool fallback) {
    const auto* value = body.find(key);
    if (!value || value->is_null()) return fallback;
    if (!value->is_bool()) throw ApiError(400, "invalid_request_error", std::string(key) + " must be a boolean", std::string(key));
    return value->as_bool();
}

}  // namespace

struct ApiService::Impl {
    ApiConfig config;
    std::shared_ptr<const BpeTokenizer> tokenizer;
    GenerationQueue& queue;
    std::uint64_t context_tokens;
    bool engine_samples;
    std::mutex mutex;
    std::deque<RecentRequest> recent;
    std::string state = "idle";
    std::uint64_t requests_total = 0, requests_failed = 0;
    std::chrono::steady_clock::time_point started = std::chrono::steady_clock::now();
    const std::string instance = random_id("").substr(0, 16);  // this process's engine instance, chosen once (status, metrics)
    const std::int64_t start_unix = unix_now();
    std::int64_t armed_unix = 0;  // when this service armed the instrumentation's current period (0: armed elsewhere)
    // the expiry this service's arming requests carry (observe_expiry_seconds; 0: never; observe_arm)
    double expiry_seconds = 0.0;
    Json capabilities_document;
    std::string capabilities_id;

    void build_capabilities() {
        Json out = Json::object();
        out.set("object", Json("strix.capabilities")); out.set("version", Json(1));
        out.set("model", Json(config.model_id)); out.set("context_tokens", Json(context_tokens));
        Json endpoints = Json::object();
        for (const char* name : {"chat", "completions", "messages"}) endpoints.set(name, Json(true));
        endpoints.set("responses", Json(config.responses)); out.set("endpoints", std::move(endpoints));
        Json sampling = Json::object();
        sampling.set("policy", Json(config.sampling_policy));
        sampling.set("greedy", Json(true)); sampling.set("sample", Json(engine_samples));
        sampling.set("positive_temperature", Json(engine_samples ? "sample" : config.sampling_policy == "reject" ? "reject" : "downgrade"));
        Json filters = Json::object();
        filters.set("temperature", parse_json("{\"min\":0,\"max\":2}"));
        filters.set("top_p", parse_json("{\"min\":0,\"min_exclusive\":true,\"max\":1}"));
        filters.set("min_p", parse_json("{\"min\":0,\"max\":1,\"max_exclusive\":true}"));
        filters.set("top_k", parse_json("{\"min\":0,\"max\":4294967295}"));
        sampling.set("filters", std::move(filters));
        sampling.set("defaults", resolve_filters(Json::object()).filters);
        sampling.set("seed_echo", Json(config.stream_receipts));  // the seed travels in the receipt
        sampling.set("filters_applied", Json(engine_samples ? "positive_temperature_only" : "unavailable"));
        out.set("sampling", std::move(sampling));
        Json stream = Json::object(); stream.set("sse", Json(true));
        stream.set("receipts", Json(config.stream_receipts)); stream.set("keepalive_seconds", Json(config.sse_keepalive_seconds));
        Json progress = Json::object();
        progress.set("interval_seconds", Json(config.sse_progress_seconds));
        progress.set("engine_reports", Json(config.engine_progress));
        stream.set("progress", std::move(progress));
        out.set("stream", std::move(stream));
        for (const char* name : {"json_schema", "json_object", "strict_tools", "penalties", "logit_bias", "logprobs", "retry",
                "prefill_progress", "response_storage", "required_tool_choice", "named_tool_choice",
                "grammar", "background", "conversation", "encrypted_reasoning", "image_input", "file_input"}) out.set(name, Json(false));
        out.set("prefill_progress", Json(config.engine_progress));
        out.set("thinking_budget", Json(config.engine_thinking_budget));
        out.set("thinking_budget_policy", config.engine_thinking_budget ? Json("thinking-budget-v1") : Json());
        out.set("logprobs", Json(config.engine_logprobs)); out.set("penalties", Json(config.engine_logit_transforms)); out.set("logit_bias", Json(config.engine_logit_transforms));
        Json scoring = Json::object(); scoring.set("max_top_logprobs", Json(20)); scoring.set("basis", Json("target-row-log-softmax")); scoring.set("execution", Json("serial-host"));
        if (config.engine_logprobs) out.set("scoring", std::move(scoring));
        if (config.engine_logit_transforms) { Json t = Json::object(); t.set("identity", Json("logit-transform-v1")); t.set("execution", Json("serial-host")); t.set("sampling", Json(false)); out.set("logit_transforms", std::move(t)); }
        out.set("json_object", Json(config.engine_constraints));
        out.set("json_schema", Json(config.engine_constraints));
        Json constraints = Json::object(), identities = Json::array(), routes = Json::array(), limits = Json::object();
        if (config.engine_constraints) {
            identities.push(Json("json-object")); identities.push(Json("strix-json-schema-v1"));
            for (const auto& route : config.constraint_execution_routes) routes.push(Json(route));
        }
        const ConstraintLimits bound;
        limits.set("schema_bytes", Json(static_cast<std::uint64_t>(bound.schema_bytes)));
        limits.set("nodes", Json(static_cast<std::uint64_t>(bound.nodes)));
        limits.set("depth", Json(static_cast<std::uint64_t>(bound.depth)));
        limits.set("properties", Json(static_cast<std::uint64_t>(bound.properties)));
        limits.set("enum_values", Json(static_cast<std::uint64_t>(bound.enum_values)));
        limits.set("max_string_length", Json(static_cast<std::uint64_t>(bound.max_string_length)));
        limits.set("max_output_bytes", Json(static_cast<std::uint64_t>(bound.max_output_bytes)));
        limits.set("compilation_steps", Json(static_cast<std::uint64_t>(bound.compilation_steps)));
        limits.set("literal_bytes", Json(static_cast<std::uint64_t>(bound.literal_bytes)));
        limits.set("trie_nodes", Json(static_cast<std::uint64_t>(bound.trie_nodes)));
        limits.set("alternative_expansions", Json(static_cast<std::uint64_t>(bound.alternative_expansions)));
        limits.set("mask_cache_bytes", Json(static_cast<std::uint64_t>(bound.mask_cache_bytes)));
        Json selection_routes=Json::array();
        if(config.engine_constraints || config.engine_thinking_budget || config.engine_logprobs || config.engine_logit_transforms)
            for(const auto& route:config.constraint_execution_routes)selection_routes.push(Json(route));
        out.set("selection_execution",std::move(selection_routes));
        constraints.set("identities", std::move(identities)); constraints.set("limits", std::move(limits));
        constraints.set("selection", Json("greedy")); constraints.set("sampling", Json(false));
        constraints.set("execution", std::move(routes)); out.set("constraints", std::move(constraints));
        out.set("default_max_tokens", Json(static_cast<std::uint64_t>(config.default_max_tokens)));
        out.set("default_thinking", Json(config.default_thinking));
        out.set("function_tools", Json(true)); out.set("plaintext_reasoning", Json(true));
        out.set("developer_role", Json("leading_system_message"));
        if (config.capabilities_extra) {
            config.capabilities_extra(out);
            Json canonical = Json::object();
            for (const auto& [key, value] : out.as_object()) if (key != "capabilities_id") canonical.set(key, value);
            out = std::move(canonical);
        }
        std::uint64_t hash = 14695981039346656037ULL;
        for (const char c_char : out.dump()) { const unsigned char c = static_cast<unsigned char>(c_char); hash ^= c; hash *= 1099511628211ULL; }
        char hex[17]; std::snprintf(hex, sizeof(hex), "%016llx", static_cast<unsigned long long>(hash));
        capabilities_id = hex;
        out.set("capabilities_id", Json(capabilities_id));
        capabilities_document = std::move(out);
    }

    double keepalive_seconds(const Json& body) const {
        if (const auto* s = body.find("strix"); s && s->is_object())
            if (!bool_or(*s, "keepalive", true)) return 0.0;
        return config.sse_keepalive_seconds;
    }
    double progress_seconds(const Json& body) const {
        if (const auto* s = body.find("strix"); s && s->is_object())
            if (!bool_or(*s, "progress", true)) return 0.0;
        return config.sse_progress_seconds;
    }
    struct StreamWriter {
        HttpResponseWriter& response;
        double interval;
        double progress_interval;
        GenerationProgress previous;
        std::chrono::steady_clock::time_point last_progress = std::chrono::steady_clock::now();
        StreamWriter(HttpResponseWriter& r, double keepalive, double progress)
            : response(r), interval(keepalive), progress_interval(progress) {}
        std::chrono::steady_clock::time_point last = std::chrono::steady_clock::now();
        bool write(const std::string& bytes) {
            const bool ok = response.write(bytes);
            if (ok) last = std::chrono::steady_clock::now();
            return ok;
        }
        bool idle(const GenerationTicket& ticket) {
            const auto now = std::chrono::steady_clock::now();
            if (progress_interval > 0.0) {
                const auto progress = ticket.progress();
                if (progress.phase != GenerationPhase::queued && progress != previous &&
                    std::chrono::duration<double>(now - last_progress).count() >= progress_interval) {
                    Json value = Json::object();
                    value.set("phase", Json(progress.phase == GenerationPhase::decode ? "decode" : "prefill"));
                    value.set("prefilled_tokens", Json(progress.prefilled_tokens));
                    value.set("cached_tokens", Json(progress.cached_tokens));
                    value.set("total_tokens", Json(progress.total_tokens));
                    if (!write(": progress " + value.dump() + "\n\n")) return false;
                    previous = progress; last_progress = last;
                }
            }
            if (interval > 0.0 && std::chrono::duration<double>(std::chrono::steady_clock::now() - last).count() >= interval)
                return write(": keepalive\n\n");
            return !response.client_gone();
        }
    };

    static double expiry_from(const std::optional<double>& configured) {
        constexpr double longest = 1e9;  // seconds (a deadline stays representable)
        if (configured) return *configured > 0.0 ? std::min(*configured, longest) : 0.0;
        if (const char* env = std::getenv("STRIX_SERVE_OBSERVE_EXPIRY_S"); env && *env) {
            char* end = nullptr;
            const double v = std::strtod(env, &end);
            if (end != env && *end == '\0' && v >= 0.0) return std::min(v, longest);
            std::fprintf(stderr, "STRIX_SERVE_OBSERVE_EXPIRY_S=%s is not a number of seconds >= 0: 60 s\n", env);
        }
        return 60.0;
    }
    Impl(ApiConfig c, std::shared_ptr<const BpeTokenizer> t, GenerationQueue& q, std::uint64_t context, bool samples)
        : config(std::move(c)), tokenizer(std::move(t)), queue(q), context_tokens(context), engine_samples(samples),
          expiry_seconds(expiry_from(config.observe_expiry_seconds)) { if (config.capabilities) build_capabilities(); }

    bool is_stop_token(std::uint32_t id) const {
        return std::find(config.stop_tokens.begin(), config.stop_tokens.end(), id) != config.stop_tokens.end();
    }
    void record(RecentRequest entry) {
        if (!current_request_id.empty()) {
            if (!entry.id.empty() && entry.id != current_request_id) entry.response_id = entry.id;
            entry.id = current_request_id;
        }
        const std::lock_guard lock(mutex);
        ++requests_total;
        if (entry.status == "error") ++requests_failed;
        recent.push_front(std::move(entry));
        while (recent.size() > config.recent_requests) recent.pop_back();
    }

    static void refuse_unexecuted(const Json& body, bool allow_constraints = false, bool allow_selection = true) {
        try { serve::refuse_unexecuted(body, allow_constraints, allow_selection); }
        catch (const RequestFieldError& e) { throw ApiError(400, "invalid_request_error", e.what(), e.param); }
    }

    struct ResolvedSampling : SamplingParameters {
        Json filters = Json::object();
        bool requested_sample = false;
    };
    static std::uint64_t unsigned_integer(const Json& value, const char* field, std::uint64_t maximum) {
        try { return serve::unsigned_integer(value, field, maximum); }
        catch (const RequestFieldError& e) { throw ApiError(400, "invalid_request_error", e.what(), e.param); }
    }
    ResolvedSampling resolve_filters(const Json& body) const {
        ResolvedSampling s;
        const auto resolve = [&](const char* key, const auto& server, auto builtin) {
            const auto* request = body.find(key);
            const bool supplied = request && !request->is_null();
            Json entry = Json::object();
            entry.set("value", supplied ? *request : server ? Json(static_cast<decltype(builtin)>(*server)) : Json(builtin));
            entry.set("source", Json(supplied ? "request" : server ? "server" : "builtin"));
            s.filters.set(key, entry);
            return *entry.find("value");
        };
        const auto real = [](const Json& value, const char* field, double low, double high, bool open_low, bool open_high) {
            try { return finite_number(value, field, low, high, open_low, open_high); }
            catch (const RequestFieldError& e) { throw ApiError(400, "invalid_request_error", e.what(), e.param); }
        };
        s.temperature = real(resolve("temperature", config.default_temperature, 0.0), "temperature", 0, 2, false, false);
        s.top_k = static_cast<std::uint32_t>(unsigned_integer(resolve("top_k", config.default_top_k, std::uint64_t{0}), "top_k", UINT32_MAX));
        s.top_p = real(resolve("top_p", config.default_top_p, 1.0), "top_p", 0, 1, true, false);
        s.min_p = real(resolve("min_p", config.default_min_p, 0.0), "min_p", 0, 1, false, true);
        s.requested_sample = s.temperature > 0.0;
        return s;
    }
    ResolvedSampling sampling(const Json& body, std::string& note, bool allow_constraints = false) const {
        refuse_unexecuted(body, allow_constraints);
        auto s = resolve_filters(body);
        if (const auto* seed = body.find("seed"); seed && !seed->is_null()) {
            s.seed = unsigned_integer(*seed, "seed", INT64_MAX); s.seeded = true;
        }
        if (s.temperature > 0.0 && !engine_samples) {
            if (config.sampling_policy == "reject")
                throw ApiError(400, "invalid_request_error", "this server decodes greedy only; send temperature 0", "temperature");
            note = "temperature ignored: this engine decodes greedy (exact)";
            for (const auto* key : {"presence_penalty", "frequency_penalty", "repetition_penalty", "logit_bias"}) {
                const auto* v = body.find(key);
                if (v && ((v->is_number() && v->as_double() != (std::string(key) == "repetition_penalty" ? 1 : 0)) || (v->is_object() && v->size())))
                    note = "temperature ignored: this engine executes greedy-logit-transform-v1";
            }
            static_cast<SamplingParameters&>(s) = SamplingParameters{};
        } else if (s.temperature == 0.0 && (s.top_k != 0 || s.top_p != 1.0 || s.min_p != 0.0)) {
            note = "filters ignored at temperature 0 (greedy)";
        }
        if (s.temperature > 0.0 && engine_samples) {  // the engine samples with float parameters: check those
            if (!(static_cast<float>(s.temperature) > 0.0F))
                throw ApiError(400, "invalid_request_error", "temperature is too small to sample with", "temperature");
            if (!(static_cast<float>(s.top_p) > 0.0F)) throw ApiError(400, "invalid_request_error", "top_p is too small to sample with", "top_p");
            if (!(static_cast<float>(s.min_p) < 1.0F)) throw ApiError(400, "invalid_request_error", "min_p is too close to 1 to sample with", "min_p");
        }
        if (s.temperature > 0.0 && !s.seeded) {
            std::random_device entropy;
            s.seed = ((static_cast<std::uint64_t>(entropy()) << 32U) | entropy()) & INT64_MAX;
            s.seeded = true;
        }
        return s;
    }
    std::uint32_t max_tokens(const Json& body, std::size_t prompt) const {
        double requested = config.default_max_tokens;
        for (const char* key : {"max_completion_tokens", "max_tokens"})
            if (const auto* v = body.find(key); v && !v->is_null()) {
                if (!v->is_number() || v->as_double() < 1.0) throw ApiError(400, "invalid_request_error", std::string(key) + " must be a positive integer", key);
                requested = v->as_double();
                break;
            }
        if (prompt >= context_tokens)
            throw ApiError(400, "invalid_request_error", "prompt of " + std::to_string(prompt) + " tokens does not fit the context of " +
                std::to_string(context_tokens) + " tokens", "messages");
        const auto room = static_cast<double>(context_tokens - prompt);
        return static_cast<std::uint32_t>(std::max(1.0, std::min(requested, room)));
    }
    void authorize(const HttpRequest& request) const {
        if (config.api_key.empty()) return;
        const auto bearer = request.header("authorization");
        if (bearer == "Bearer " + config.api_key || request.header("x-api-key") == config.api_key) return;
        throw ApiError(401, "authentication_error", "invalid or missing API key");
    }

    // ---- chat ----
    struct ChatPrompt { std::vector<std::uint32_t> tokens; bool thinking; Json tools; std::uint32_t stable_prefix = 0; };
    ChatPrompt chat_prompt(const Json& body) const {
        const auto* messages = body.find("messages");
        if (!messages || !messages->is_array() || messages->size() == 0U)
            throw ApiError(400, "invalid_request_error", "messages must be a non-empty array", "messages");
        ChatTemplateInput input;
        Json normalized = Json::array();
        for (const auto& message : messages->as_array()) {
            if (!message.is_object()) throw ApiError(400, "invalid_request_error", "every message must be an object", "messages");
            Json copy = message;
            if (!copy.find("reasoning_content")) if (const auto* r = copy.find("reasoning"); r && r->is_string()) copy.set("reasoning_content", *r);
            normalized.push(std::move(copy));
        }
        input.messages = normalized;
        const auto* tool_choice = body.find("tool_choice");
        const bool tools_off = tool_choice && tool_choice->is_string() && tool_choice->as_string() == "none";
        if (const auto* tools = body.find("tools"); tools && tools->is_array() && !tools_off) input.tools = *tools;
        const auto* kwargs = body.find("chat_template_kwargs");
        const auto kwarg = [&](std::string_view key) -> const Json* {
            if (kwargs && kwargs->is_object()) if (const auto* v = kwargs->find(key)) return v;
            return body.find(key);
        };
        if (const auto* v = kwarg("enable_thinking"); v && v->is_bool()) input.enable_thinking = v->as_bool();
        else if (!config.default_thinking) input.enable_thinking = false;
        if (const auto* v = kwarg("preserve_thinking"); v && v->is_bool()) input.preserve_thinking = v->as_bool();
        if (const auto* v = kwarg("reasoning_effort"); v && v->is_string()) {
            auto effort = v->as_string();
            if (effort == "minimal") effort = "low";
            if (effort == "none") input.enable_thinking = false;
            else input.reasoning_effort = effort;
        }
        std::string text;
        try { text = config.chat_renderer ? config.chat_renderer(input) : render_chat_template(input); }
        catch (const ChatTemplateError& error) { throw ApiError(400, "invalid_request_error", error.what(), "messages"); }
        ChatPrompt out;
        out.tokens = tokenizer->encode(text, true);
        out.thinking = !input.enable_thinking || *input.enable_thinking;
        out.tools = input.tools;
        // the stable prefix: the leading system messages and the tools rendered alone, when their tokens begin the prompt
        std::size_t lead = 0;
        while (lead + 1U < normalized.size()) {
            const auto* role = normalized.as_array()[lead].find("role");
            if (!role || !role->is_string() || role->as_string() != "system") break;
            ++lead;
        }
        if (lead > 0U) {
            ChatTemplateInput head = input;
            Json system = Json::array();
            for (std::size_t i = 0; i < lead; ++i) system.push(normalized.as_array()[i]);
            head.messages = system;
            head.add_generation_prompt = false;
            try {
                const auto ids = tokenizer->encode(config.chat_renderer ? config.chat_renderer(head) : render_chat_template(head), true);
                if (ids.size() < out.tokens.size() && std::equal(ids.begin(), ids.end(), out.tokens.begin()))
                    out.stable_prefix = static_cast<std::uint32_t>(ids.size());
            } catch (const ChatTemplateError&) {}
        }
        return out;
    }

    std::shared_ptr<const ConstraintSpec> json_constraint(const Json& body, const ChatPrompt& prompt,
        const SamplingParameters& sampling_params, const std::vector<std::string>& stops, bool responses = false) const {
        const Json* format = body.find("response_format");
        std::string prefix = "response_format";
        if (responses) {
            const auto* text = body.find("text");
            format = text && text->is_object() ? text->find("format") : nullptr;
            prefix = "text.format";
        }
        if (!format || format->is_null()) return {};
        const auto bad = [&](const std::string& path, const std::string& message) {
            throw ApiError(400, "invalid_request_error", message, path);
        };
        const auto* type = format->is_object() ? format->find("type") : nullptr;
        if (!type || !type->is_string()) bad(prefix, "format must be an object with a string type");
        const auto& kind = type->as_string();
        if (kind == "text") return {};
        if (kind != "json_object" && kind != "json_schema") bad(prefix, "unsupported response format");
        if (!config.engine_constraints) bad(prefix, "this engine does not execute constraints (see /v1/capabilities)");
        auto spec = std::make_shared<ConstraintSpec>();
        spec->identity = kind == "json_object" ? "json-object" : "strix-json-schema-v1";
        const Json* envelope = format;
        if (kind == "json_schema" && !responses) { envelope = format->find("json_schema"); prefix += ".json_schema"; }
        if (!envelope || !envelope->is_object()) bad(prefix, "json_schema must be an object");
        if (const auto* strict = envelope->find("strict"); strict && !strict->is_bool()) bad(prefix + ".strict", "strict must be a boolean");
        try {
            if (kind == "json_object") spec->program = std::make_shared<ConstraintProgram>(compile_json_object());
            else {
                const auto* name = envelope->find("name");
                if (!name || !name->is_string() || name->as_string().empty()) bad(prefix + ".name", "name must be a nonempty string");
                const auto* schema = envelope->find("schema");
                if (!schema) bad(prefix + ".schema", "schema is required");
                spec->program = std::make_shared<ConstraintProgram>(compile_json_schema(*schema));
            }
        } catch (const SchemaError& e) {
            throw ApiError(400, "invalid_request_error", e.what(), prefix + ".schema" + (e.path.empty() ? "" : "." + e.path),
                std::string(e.what()).find("unsupported") != std::string::npos || std::string(e.what()).find("Unsupported") != std::string::npos
                    ? "unsupported_schema_keyword" : "invalid_schema", e.keyword);
        }
        if (std::any_of(stops.begin(), stops.end(), [](const auto& stop) { return !stop.empty(); })) bad("stop", "stop strings cannot be combined with constraints");
        if (sampling_params.temperature > 0.0) bad("temperature", "constraints require effective greedy selection");
        const auto* choice = body.find("tool_choice");
        const bool tools_off = choice && choice->is_string() && choice->as_string() == "none";
        if (const auto* tools = body.find("tools"); tools && tools->is_array() && tools->size() && !tools_off) bad("tools", "tools cannot be combined with constraints");
        spec->eos_ids = config.stop_tokens;
        const auto eos = tokenizer->eos_token();
        if (eos < tokenizer->vocab_size() && std::find(spec->eos_ids.begin(), spec->eos_ids.end(), eos) == spec->eos_ids.end()) spec->eos_ids.push_back(eos);
        spec->starts_in_reasoning = prompt.thinking;
        const auto end = tokenizer->find("</think>");
        if (end < tokenizer->vocab_size() && std::find(spec->eos_ids.begin(), spec->eos_ids.end(), end) == spec->eos_ids.end()) spec->end_thinking_id = end;
        if (prompt.thinking && !spec->end_thinking_id) bad(responses ? "text.format" : "response_format", "constraints unavailable: missing end-thinking token");
        return spec;
    }
    std::shared_ptr<const ConstraintSpec> selection_policy(const Json& body,
        std::shared_ptr<const ConstraintSpec> base, const std::vector<std::uint32_t>& prompt,
        const SamplingParameters& params, bool completion = false) const {
        const auto bad = [](const std::string& field, const std::string& message) {
            throw ApiError(400, "invalid_request_error", message, field);
        };
        LogitTransform t;
        const auto number = [&](const char* key, double fallback, double lo, double hi) {
            const auto* v = body.find(key); if (!v || v->is_null()) return fallback;
            if (!v->is_number() || !std::isfinite(v->as_double()) || v->as_double() < lo || v->as_double() > hi)
                bad(key, std::string(key) + " is out of range");
            return v->as_double();
        };
        t.presence = number("presence_penalty", 0, -2, 2);
        t.frequency = number("frequency_penalty", 0, -2, 2);
        t.repetition = number("repetition_penalty", 1, .1, 10);
        if (const auto* bias = body.find("logit_bias"); bias && !bias->is_null()) {
            if (!bias->is_object() || bias->size() > 1024) bad("logit_bias", "logit_bias must be an object with at most 1024 entries");
            for (const auto& [key, value] : bias->as_object()) {
                std::uint32_t id = 0; const auto parsed = std::from_chars(key.data(), key.data() + key.size(), id);
                if (key.empty() || parsed.ec != std::errc{} || parsed.ptr != key.data() + key.size() || key != std::to_string(id) || id >= tokenizer->vocab_size() ||
                    !value.is_number() || !std::isfinite(value.as_double()) || value.as_double() < -100 || value.as_double() > 100)
                    bad("logit_bias", "invalid vocabulary ID or bias (expected [-100,100])");
                t.bias.emplace_back(id, value.as_double());
            }
        }
        bool scores = false;
        std::uint32_t top = 0;
        if (const auto* v = body.find("logprobs"); v && !v->is_null()) {
            if (completion) {
                if (!v->is_number() || v->as_double() < 0 || v->as_double() > 5 || std::floor(v->as_double()) != v->as_double())
                    bad("logprobs", "completions logprobs must be an integer in [0,5]");
                scores = true; top = static_cast<std::uint32_t>(v->as_int());
            } else {
                if (!v->is_bool()) bad("logprobs", "logprobs must be boolean");
                scores = v->as_bool();
            }
        }
        if (const auto* include = body.find("include"); include && include->is_array())
            for (const auto& v : include->as_array()) if (v.is_string() && v.as_string() == "message.output_text.logprobs") scores = true;
        if (body.find("top_logprobs")) {
            const double n = number("top_logprobs", 0, 0, 20);
            if (std::floor(n) != n) bad("top_logprobs", "top_logprobs must be an integer in [0,20]");
            if (completion && n > 0) bad("top_logprobs", "completions uses logprobs in [0,5]");
            top = completion ? top : static_cast<std::uint32_t>(n); scores = scores || top > 0;
        }
        if (scores && !config.engine_logprobs) bad(body.find("logprobs") || body.find("include") ? "logprobs" : "top_logprobs", "this engine does not execute logprobs (see /v1/capabilities)");
        if (t.active() && !config.engine_logit_transforms) bad(t.presence != 0 ? "presence_penalty" : t.frequency != 0 ? "frequency_penalty" : t.repetition != 1 ? "repetition_penalty" : "logit_bias", "this engine does not execute logit transforms (see /v1/capabilities)");
        if (!scores && !t.active()) return base;
        if (params.temperature > 0) bad("temperature", "logprobs and logit transforms require effective greedy selection; sampling is unavailable");
        auto spec = base ? std::make_shared<ConstraintSpec>(*base) : std::make_shared<ConstraintSpec>();
        spec->transform = std::move(t); spec->logprobs = scores; spec->top_logprobs = top; spec->prompt_ids = prompt;
        if (!base) spec->eos_ids = config.stop_tokens;
        return spec;
    }
    Json score_json(const TokenScore& score) const {
        const auto token = [&](std::uint32_t id, const std::optional<double>& value) {
            Json out = Json::object(), bytes = Json::array(); const auto raw = tokenizer->token_bytes(id);
            for (const char b_char : raw) { const unsigned char b = static_cast<unsigned char>(b_char); bytes.push(Json(static_cast<std::uint64_t>(b))); }
            out.set("token", Json(sanitize_utf8(raw))); out.set("bytes", std::move(bytes));
            out.set("logprob", value ? Json(*value) : Json()); return out;
        };
        auto out = token(score.id, score.logprob); Json top = Json::array();
        for (const auto& [id, value] : score.top) top.push(token(id, value));
        out.set("top_logprobs", std::move(top));
        if (score.forced_control) out.set("forced_control", Json(true));
        return out;
    }
    static Json chat_scores(Json content) { Json out = Json::object(); out.set("content", std::move(content)); return out; }
    static Json completion_scores(const Json& content, std::uint64_t& offset) {
        Json out = Json::object(), tokens = Json::array(), values = Json::array(), tops = Json::array(), offsets = Json::array();
        for (const auto& score : content.as_array()) {
            const auto& text = *score.find("token"); tokens.push(text); values.push(*score.find("logprob"));
            offsets.push(Json(offset)); offset += text.as_string().size();
            Json top = Json::object(); for (const auto& v : score.find("top_logprobs")->as_array()) top.set(v.find("token")->as_string(), *v.find("logprob"));
            tops.push(std::move(top));
        }
        out.set("tokens", std::move(tokens)); out.set("token_logprobs", std::move(values)); out.set("top_logprobs", std::move(tops)); out.set("text_offset", std::move(offsets)); return out;
    }
    std::shared_ptr<const ConstraintSpec> constraint(const Json& body, const ChatPrompt& prompt,
        const SamplingParameters& params, const std::vector<std::string>& stops, std::uint32_t budget,
        bool responses = false) const {
        auto base = selection_policy(body, json_constraint(body, prompt, params, stops, responses), prompt.tokens, params);
        const auto* options = body.find("strix");
        const auto* policy = options && options->is_object() ? options->find("thinking_policy") : nullptr;
        const bool fields = options && options->is_object() &&
            (options->find("max_thinking_tokens") || options->find("answer_room_tokens"));
        if (!policy && !fields) {
            if (base && !base->program && !base->thinking) {
                auto selection = std::make_shared<ConstraintSpec>(*base);
                selection->starts_in_reasoning = prompt.thinking;
                const auto end = tokenizer->find("</think>");
                if (end < tokenizer->vocab_size()) selection->end_thinking_id = end;
                return selection;
            }
            return base;
        }
        const auto bad = [](const std::string& field, const std::string& message) {
            throw ApiError(400, "invalid_request_error", message, "strix." + field);
        };
        if (!policy || !policy->is_string() || policy->as_string() != "thinking-budget-v1")
            bad("thinking_policy", "unknown thinking policy; expected thinking-budget-v1");
        if (!config.engine_thinking_budget) bad("thinking_policy", "this engine does not execute thinking budgets");
        if (!prompt.thinking) bad("thinking_policy", "thinking is disabled for this request");
        const auto integer = [&](const char* field, std::uint32_t fallback, bool required) {
            const auto* value = options->find(field);
            if (!value && !required) return fallback;
            if (!value || !value->is_number() || value->as_double() < (required ? 1.0 : 0.0) ||
                value->as_double() > UINT32_MAX || value->as_double() != static_cast<double>(value->as_int()))
                bad(field, "thinking budget must be an integer in range");
            return static_cast<std::uint32_t>(value->as_int());
        };
        ThinkingPolicy thinking;
        thinking.max_thinking_tokens = integer("max_thinking_tokens", 0, true);
        // An explicit answer room is kept as given; omitted, the answer gets what remains after the thinking budget and
        // its closure (resolved below, once the closure length is known).
        const auto* explicit_room = options->find("answer_room_tokens");
        const bool room_given = explicit_room && !explicit_room->is_null();
        thinking.answer_room_tokens = room_given ? integer("answer_room_tokens", 0, false) : 0U;
        auto spec = base ? std::make_shared<ConstraintSpec>(*base) : std::make_shared<ConstraintSpec>();
        spec->starts_in_reasoning = true;
        spec->eos_ids = config.stop_tokens;
        const auto eos = tokenizer->eos_token();
        if (eos < tokenizer->vocab_size() && std::find(spec->eos_ids.begin(), spec->eos_ids.end(), eos) == spec->eos_ids.end()) spec->eos_ids.push_back(eos);
        const auto end = tokenizer->find("</think>");
        if (end >= tokenizer->vocab_size() || tokenizer->token_bytes(end) != "</think>" ||
            (tokenizer->type(end) != BpeTokenizer::TokenType::control && tokenizer->type(end) != BpeTokenizer::TokenType::user_defined) ||
            std::find(spec->eos_ids.begin(), spec->eos_ids.end(), end) != spec->eos_ids.end())
            bad("thinking_policy", "missing usable end-thinking control token");
        spec->end_thinking_id = end;
        thinking.closure_ids.push_back(end);
        const auto suffix = tokenizer->encode(thinking_answer_separator, false);
        thinking.closure_ids.insert(thinking.closure_ids.end(), suffix.begin(), suffix.end());
        if (!room_given) {
            const auto used = static_cast<std::uint64_t>(thinking.max_thinking_tokens) + thinking.closure_ids.size();
            if (used >= budget) bad("max_thinking_tokens", "the thinking budget and its closure leave no answer room within max_tokens");
            thinking.answer_room_tokens = static_cast<std::uint32_t>(budget - used);
        }
        if (static_cast<std::uint64_t>(thinking.answer_room_tokens) + thinking.closure_ids.size() > budget)
            bad("answer_room_tokens", "answer room plus thinking closure exceeds effective max_tokens");
        thinking.reasoning_limit = std::min(thinking.max_thinking_tokens,
            budget - thinking.answer_room_tokens - static_cast<std::uint32_t>(thinking.closure_ids.size()));
        if (params.temperature > 0.0) bad("thinking_policy", "thinking budgets require effective greedy selection");
        if (std::any_of(stops.begin(), stops.end(), [](const auto& stop) { return !stop.empty(); })) bad("thinking_policy", "stop strings cannot be combined with thinking budgets");
        const auto* choice = body.find("tool_choice");
        const bool tools_off = choice && choice->is_string() && choice->as_string() == "none";
        if (const auto* tools = body.find("tools"); tools && tools->is_array() && tools->size() && !tools_off)
            bad("thinking_policy", "tools cannot be combined with thinking budgets");
        spec->thinking = std::move(thinking);
        return spec;
    }
    static void selection_json(Json& out, const ConstraintSpec& spec, const ConstraintVerdict& verdict) {
        out.set("selection_execution",Json(verdict.execution));
        out.set("selection_fallback",verdict.execution_fallback.empty()?Json():Json(verdict.execution_fallback));
        if (spec.transform.active()) {
            Json t = Json::object(); t.set("presence", Json(spec.transform.presence)); t.set("frequency", Json(spec.transform.frequency));
            t.set("repetition", Json(spec.transform.repetition)); t.set("bias_entries", Json(static_cast<std::uint64_t>(spec.transform.bias.size())));
            out.set("transform", std::move(t)); out.set("unconstrained_greedy_equal", Json(false)); out.set("sampling", Json("greedy-logit-transform-v1")); out.set("sampling_effective", Json("greedy-logit-transform-v1"));
        }
        if (spec.logprobs) {
            Json scoring = Json::object(); scoring.set("basis", Json("target-row-log-softmax"));
            scoring.set("transform", Json(spec.transform.active() ? "logit-transform-v1" : "none")); scoring.set("constrained", Json(bool(spec.program)));
            out.set("scoring", std::move(scoring));
        }
        if (spec.program) out.set("constraint", constraint_json(spec, verdict));
        if (spec.thinking) {
            Json t = Json::object(), ids = Json::array();
            t.set("policy", Json("thinking-budget-v1"));
            t.set("max_thinking_tokens", Json(static_cast<std::uint64_t>(spec.thinking->max_thinking_tokens)));
            t.set("answer_room_tokens", Json(static_cast<std::uint64_t>(spec.thinking->answer_room_tokens)));
            t.set("closure", Json(verdict.closure)); t.set("reasoning_tokens", Json(static_cast<std::uint64_t>(verdict.reasoning_tokens)));
            for (const auto id : verdict.forced_ids) ids.push(Json(static_cast<std::uint64_t>(id)));
            t.set("forced_ids", std::move(ids));
            t.set("status", Json(verdict.status));
            if (!verdict.reason.empty()) t.set("reason", Json(verdict.reason));
            out.set("thinking", std::move(t));
        }
    }
    static Json constraint_json(const ConstraintSpec& spec, const ConstraintVerdict& verdict) {
        Json out = Json::object();
        out.set("identity", Json(spec.identity)); out.set("selection", Json("greedy-lowest-id"));
        out.set("execution", Json(verdict.execution)); out.set("phase", Json(verdict.phase));
        out.set("status", Json(verdict.status)); out.set("accepting", Json(verdict.accepting));
        out.set("reason", verdict.reason.empty() ? Json() : Json(verdict.reason));
        Json profile = Json::object();
        profile.set("rows", Json(verdict.profile_rows)); profile.set("mask_ms", Json(verdict.profile_mask_ms));
        profile.set("select_ms", Json(verdict.profile_select_ms)); profile.set("window_rows", Json(verdict.profile_window_rows));
        profile.set("window_row_ms", Json(verdict.profile_window_row_ms)); profile.set("known_rows", Json(verdict.profile_known_rows));
        out.set("host_profile", std::move(profile)); return out;
    }
    ConstraintVerdict initial_verdict(const ConstraintSpec& spec) const {
        ConstraintVerdict v; v.phase = spec.starts_in_reasoning ? "reasoning" : "answer";
        if (!config.constraint_execution_routes.empty()) v.execution = config.constraint_execution_routes.front();
        return v;
    }

    // Drives one ticket: decodes tokens to text, applies stop strings, hands text to `on_text`.
    struct Outcome { std::string text; GenerationStats stats; std::string finish; std::string matched; std::string error; std::uint64_t output_tokens = 0;
        std::vector<std::uint32_t> token_ids; bool observed = false;
        Json scores = Json::array(), answer_scores = Json::array();
        std::shared_ptr<const ConstraintSpec> constraint; std::string reasoning; };
    Outcome drive(const std::shared_ptr<GenerationTicket>& ticket, const std::vector<std::string>& stops,
        const std::function<bool(const std::string&)>& on_text, const std::function<bool()>& client_alive,
        const std::function<bool()>& idle = {}, std::shared_ptr<const ConstraintSpec> spec = {},
        const std::function<bool(const std::vector<ChatSplitter::Delta>&)>& on_delta = {},
        const std::function<void(const Json&, bool)>& on_score = {}) {
        Outcome out;
        out.constraint = spec;
        bool reasoning = spec && spec->starts_in_reasoning;
        const auto emit = [&](const std::string& text) {
            if (reasoning) out.reasoning += text; else out.text += text;
            return !on_delta || text.empty() || on_delta({{reasoning, text}});
        };
        StopFilter filter(stops);
        Utf8StreamDecoder utf8;
        bool stopped = false;
        for (bool done = false; !done;) {
            // once cancelled, wait for the engine's finish (or tokens) rather than waking at once on the cancellation
            const auto tokens = ticket->wait(std::chrono::milliseconds(200), done, !ticket->cancelled());
            for (const auto id : tokens) {
                if (stopped) break;
                ++out.output_tokens;
                out.token_ids.push_back(id);
                if (spec && spec->logprobs) {
                    TokenScore score;
                    { std::lock_guard lock(spec->scores->mutex); score = spec->scores->tokens.at(out.token_ids.size() - 1); }
                    if (!is_stop_token(id)) { auto value = score_json(score); out.scores.push(value); if (!reasoning) out.answer_scores.push(value); if (on_score) on_score(value, reasoning); }
                }
                if (spec && (spec->program || spec->thinking)) {
                    if (std::find(spec->eos_ids.begin(), spec->eos_ids.end(), id) != spec->eos_ids.end()) { stopped = true; break; }
                    if (reasoning && spec->end_thinking_id == id) {
                        if (!emit(utf8.flush())) { ticket->cancel(); stopped = true; }
                        reasoning = false; continue;
                    }
                    if (!emit(utf8.push(tokenizer->token_bytes(id)))) { ticket->cancel(); stopped = true; }
                    continue;
                }
                if (spec && reasoning && spec->end_thinking_id == id) reasoning = false;
                if (is_stop_token(id)) { stopped = true; break; }
                const auto text = filter.push(utf8.push(tokenizer->token_bytes(id)));
                if (!text.empty()) {
                    out.text += text;
                    if (on_text && !on_text(text)) { ticket->cancel(); stopped = true; }
                }
                if (filter.matched()) { out.matched = filter.match(); ticket->cancel(); stopped = true; }
            }
            if (client_alive && !client_alive()) ticket->cancel();
            if (!done && !ticket->cancelled() && idle && !idle()) ticket->cancel();
        }
        if (spec && (spec->program || spec->thinking)) { if (!emit(utf8.flush())) ticket->cancel(); }
        else if (!out.matched.size()) {
            const auto tail = filter.push(utf8.flush()) + filter.finish();
            if (!tail.empty()) { out.text += tail; if (on_text) on_text(tail); }
        }
        out.stats = ticket->stats();
        out.observed = ticket->observed();
        out.error = ticket->error();
        if (spec && !out.stats.constraint) {
            auto v = initial_verdict(*spec); v.phase = reasoning ? "reasoning" : "answer";
            if (out.stats.finish == "cancelled" && out.error.empty()) { v.status = "incomplete"; v.reason = "cancelled"; }
            else {
                v.status = "failed";
                v.reason = out.error.empty() ? "constraint_failure" : "execution_failure";
                if (out.error.empty()) out.error = "engine omitted terminal constraint verdict";
            }
            out.stats.constraint = std::move(v);
        }
        if (spec && out.stats.constraint->status == "failed" && out.error.empty()) out.error = "constraint selection failed";
        out.finish = !out.matched.empty() ? "stop" : out.stats.finish == "stop" ? "stop" : out.stats.finish == "cancelled" ? "cancelled" : "length";
        return out;
    }

    std::shared_ptr<GenerationTicket> submit(std::vector<std::uint32_t> prompt, std::uint32_t max_tokens, const SamplingParameters& sampling,
        const Json& body, std::uint32_t stable_prefix = 0, std::shared_ptr<const ConstraintSpec> spec = {}) {
        GenerationJob job;
        job.constraint = std::move(spec);
        if (const auto* policy = body.find("state_cache")) {
            const auto* retain = policy->is_object() ? policy->find("retain") : nullptr;
            if (!retain || !retain->is_bool())
                throw ApiError(400, "invalid_request_error", "state_cache must be an object with boolean retain", "state_cache");
            job.cache_retention = retain->as_bool() ? CacheRetention::retain : CacheRetention::no_store;
        }
        job.prompt = std::move(prompt);
        job.stable_prefix = stable_prefix;
        job.max_tokens = max_tokens;
        job.stop_tokens = config.stop_tokens;
        job.sampling = sampling;
        if (const auto* session = body.find("session"); config.sessions_enabled && session && !session->is_null()) {
            if (!session->is_string() || session->as_string().empty() || session->as_string().size() > 128)
                throw ApiError(400, "invalid_request_error", "session requires a valid string id", "session");
            job.session_id = session->as_string();
            if (!std::all_of(job.session_id.begin(), job.session_id.end(), [](unsigned char c) {
                return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                    (c >= '0' && c <= '9') || c == '-' || c == '_';
            })) throw ApiError(400, "invalid_request_error", "invalid session id", "session");
            if (sampling.temperature > 0.0)
                throw ApiError(400, "invalid_request_error", "sessions require greedy sampling", "temperature");
            if (job.constraint)
                throw ApiError(400, "invalid_request_error", "sessions do not support constrained output", "session");
        }
        if (const auto* ngram = body.find("ngram_draft"); ngram && !ngram->is_null()) {
            if (!ngram->is_bool()) throw ApiError(400, "invalid_request_error", "ngram_draft must be a boolean", "ngram_draft");
            job.ngram_draft = ngram->as_bool();
        }
        if (const auto* depth = body.find("draft_depth_max"); depth && !depth->is_null()) {
            if (!depth->is_number() || depth->as_int() < 0 || depth->as_int() > 8)
                throw ApiError(400, "invalid_request_error", "draft_depth_max must be an integer in [0, 8]", "draft_depth_max");
            job.draft_depth_max = static_cast<std::uint32_t>(depth->as_int());
        }
        if (const auto* tool = body.find("tool_draft"); tool && !tool->is_null()) {
            if (!tool->is_bool()) throw ApiError(400, "invalid_request_error", "tool_draft must be a boolean", "tool_draft");
            job.tool_draft = tool->as_bool();
        }
        if (const auto* peers = body.find("ngram_peers"); peers && !peers->is_null()) {
            if (!peers->is_bool()) throw ApiError(400, "invalid_request_error", "ngram_peers must be a boolean", "ngram_peers");
            job.ngram_peers = peers->as_bool();
        }
        if (!job.session_id.empty() && (job.ngram_draft.value_or(false) || job.tool_draft.value_or(false) ||
            job.ngram_peers.value_or(false) || job.draft_depth_max))
            throw ApiError(400, "invalid_request_error", "session mode requires its fixed draft policy; request draft overrides are unsupported");
        auto ticket = queue.submit(std::move(job));
        if (!ticket) {
            if (queue.draining()) throw ApiError(503, "server_error", "the server is draining");
            throw ApiError(429, "rate_limit_error", "the request queue is full");
        }
        return ticket;
    }

    void queue_headers(HttpResponseWriter& response, const GenerationTicket& ticket) const {
        if (!config.http_slots) return;
        response.add_response_header("X-Strix-Http-Slots", std::to_string(config.http_slots));
        response.add_response_header("X-Strix-Queue-Position", std::to_string(ticket.submitted_queue_position()));
    }

    static Json usage_json(const Outcome& o) {
        Json usage = Json::object();
        usage.set("prompt_tokens", Json(o.stats.prompt_tokens));
        usage.set("completion_tokens", Json(o.output_tokens));
        usage.set("total_tokens", Json(o.stats.prompt_tokens + o.output_tokens));
        Json details = Json::object();
        details.set("cached_tokens", Json(o.stats.cached_tokens + o.stats.imported_tokens));  // not prefilled for this request
        usage.set("prompt_tokens_details", std::move(details));
        return usage;
    }
    // "return_token_ids": true adds the generated token IDs (stop token included) for exactness checks.
    static Json engine_json(const Outcome& o, const std::string& note, const Json& body) {
        Json e = Json::object();
        if (bool_or(body, "return_token_ids", false)) {
            Json ids = Json::array();
            for (const auto id : o.token_ids) ids.push(Json(static_cast<std::int64_t>(id)));
            e.set("token_ids", std::move(ids));
        }
        e.set("sampling", Json(o.stats.sampling));
        if (!note.empty()) e.set("note", Json(note));
        e.set("queue_ms", Json(o.stats.queue_ms));
        e.set("prefill_ms", Json(o.stats.prefill_ms));
        e.set("decode_ms", Json(o.stats.decode_ms));
        e.set("decode_tokens_per_second", Json(o.stats.decode_ms > 0 ? 1000.0 * static_cast<double>(o.stats.generated_tokens) / o.stats.decode_ms : 0.0));
        e.set("mtp_cycles", Json(o.stats.cycles));
        e.set("mtp_accepted", Json(o.stats.accepted));
        e.set("mtp_proposed", Json(o.stats.proposed));
        e.set("ngram_rows", Json(o.stats.ngram_rows));
        e.set("ngram_accepted", Json(o.stats.ngram_accepted));
        if (o.stats.tool_rows) { e.set("tool_rows", Json(o.stats.tool_rows)); e.set("tool_accepted", Json(o.stats.tool_accepted)); }
        if (o.stats.source_shadow || o.stats.frame_cycles) {
            Json d = Json::array();
            for (const auto v : {o.stats.source_shadow, o.stats.source_scored, o.stats.source_right, o.stats.frame_cycles, o.stats.frame_shadow})
                d.push(Json(v));
            e.set("draft_sources", std::move(d)); // [shadow, scored, right rows, frame cycles, frame shadows]
        }
        for (const auto& [name, value] : o.stats.engine_extra) e.set(name, parse_json(value));
        if (o.constraint && o.stats.constraint) selection_json(e, *o.constraint, *o.stats.constraint);
        return e;
    }
    // The applied policy: alone at the start of a stream (o == nullptr), after the engine members at its end.
    Json receipt(const Outcome* o, const std::string& note, const Json& body, std::uint32_t limit, const ResolvedSampling& resolved, std::shared_ptr<const ConstraintSpec> spec = {}) const {
        if (spec && !config.stream_receipts) {
            Json out = Json::object();
            selection_json(out, *spec, o && o->stats.constraint ? *o->stats.constraint : initial_verdict(*spec));
            if (!note.empty()) out.set("note", Json(note));
            return out;
        }
        auto out = o ? engine_json(*o, note, body) : Json::object();
        if (spec && !o) selection_json(out, *spec, initial_verdict(*spec));
        if (!note.empty()) out.set("note", Json(note));
        const bool requested_sample = resolved.requested_sample;
        out.set("sampling_requested", Json(requested_sample ? "sample" : "greedy"));
        const auto selection = spec ? spec : o ? o->constraint : nullptr;
        const std::string effective = selection && selection->transform.active() ? "greedy-logit-transform-v1" : requested_sample && engine_samples ? "sample" : "greedy";
        out.set("sampling_effective", Json(effective));
        out.set("filters", resolved.filters);
        if (resolved.temperature > 0.0) out.set("seed", Json(resolved.seed));
        out.set("max_tokens_effective", Json(static_cast<std::uint64_t>(limit)));
        const auto* requested = body.find("max_completion_tokens");
        if (!requested || requested->is_null()) requested = body.find("max_tokens");
        out.set("max_tokens_requested", requested && !requested->is_null() ? *requested : Json(static_cast<std::uint64_t>(config.default_max_tokens)));
        Json stream = Json::object(); stream.set("enabled", Json(bool_or(body, "stream", false)));
        stream.set("keepalive_seconds", Json(keepalive_seconds(body))); stream.set("receipts", Json(config.stream_receipts));
        if (const auto* options = body.find("stream_options"); options && options->is_object())
            stream.set("include_usage", Json(bool_or(*options, "include_usage", false)));
        Json progress = Json::object();
        progress.set("interval_seconds", Json(progress_seconds(body)));
        progress.set("engine_reports", Json(config.engine_progress));
        stream.set("progress", std::move(progress));
        out.set("stream", std::move(stream));
        return out;
    }
    static Json tool_calls_json(const Json& calls) {
        Json out = Json::array();
        for (const auto& call : calls.as_array()) {
            Json function = Json::object();
            function.set("name", *call.find("name"));
            function.set("arguments", Json(call.find("arguments")->dump()));
            Json entry = Json::object();
            entry.set("id", Json(random_id("call_")));
            entry.set("type", Json("function"));
            entry.set("function", std::move(function));
            out.push(std::move(entry));
        }
        return out;
    }

    AssistantOutput parsed_output(const Outcome& o, const ChatPrompt& prompt) const {
        if (o.constraint && (o.constraint->program || o.constraint->thinking)) { AssistantOutput p; p.content = o.text; p.reasoning = o.reasoning; p.tool_calls = Json::array(); return p; }
        return config.output_parser ? config.output_parser(o.text, prompt.thinking, prompt.tools) : parse_assistant_output(o.text, prompt.thinking, prompt.tools);
    }
    static std::string response_status(const Outcome& o) {
        if (!o.error.empty() || (o.stats.constraint && o.stats.constraint->status == "failed")) return "failed";
        if (o.constraint && (o.constraint->program || o.constraint->thinking) && o.stats.constraint) return o.stats.constraint->status == "complete" ? "completed" : "incomplete";
        return o.finish == "length" ? "incomplete" : "completed";
    }

    // Named sessions perform their capture admission on the engine owner thread. Keep
    // headers uncommitted until that check passes (or the request finishes in error).
    void stream_admission(const std::shared_ptr<GenerationTicket>& ticket, const Json& body,
        HttpResponseWriter& response, const char* type) {
        const auto* session = body.find("session");
        if (!config.sessions_enabled || !session || session->is_null()) return;
        while (!ticket->wait_admission(std::chrono::milliseconds(100))) {
            if (response.client_gone()) { ticket->cancel(); return; }
        }
        const auto error = ticket->error();
        if (!error.empty()) {
            const auto stats = ticket->stats();
            throw ApiError(stats.error_status ? stats.error_status : 500, type, error, "", stats.error_code);
        }
    }
    bool named_session(const Json& body) const {
        const auto* session = body.find("session");
        return config.sessions_enabled && session && session->is_string() && !session->as_string().empty();
    }
    static Json stream_error(const Outcome& outcome, bool messages = false) {
        Json detail = Json::object();
        detail.set("message", Json(outcome.error));
        detail.set("type", Json(messages ? "api_error" : "server_error"));
        detail.set("status", Json(outcome.stats.error_status ? outcome.stats.error_status : 500));
        detail.set("code", outcome.stats.error_code.empty() ? Json() : Json(outcome.stats.error_code));
        Json error = Json::object();
        if (messages) error.set("type", Json("error"));
        error.set("error", std::move(detail));
        return error;
    }

    void chat_completions(const HttpRequest&, HttpResponseWriter& response, const Json& body) {
        const auto prompt = chat_prompt(body);
        if (const auto* n = body.find("n"); n && n->is_number() && n->as_int() != 1)
            throw ApiError(400, "invalid_request_error", "only n = 1 is supported", "n");
        std::string note;
        const auto sampling_params = sampling(body, note, true);
        const auto stops = stop_strings(body.find("stop"));
        const bool stream = bool_or(body, "stream", false);
        const auto interval = keepalive_seconds(body);
        const auto progress_interval = progress_seconds(body);
        bool include_usage = false;
        if (const auto* options = body.find("stream_options"); options && options->is_object()) include_usage = bool_or(*options, "include_usage", false);
        const auto limit = max_tokens(body, prompt.tokens.size());
        const auto spec = constraint(body, prompt, sampling_params, stops, limit);
        const auto initial_receipt = config.stream_receipts || spec ? receipt(nullptr, note, body, limit, sampling_params, spec) : Json();
        auto ticket = submit(prompt.tokens, limit, sampling_params, body, prompt.stable_prefix, spec);
        queue_headers(response, *ticket);
        const auto id = random_id("chatcmpl-");
        const auto created = unix_now();
        RecentRequest entry{id, "chat.completions", "ok", created};
        if (!stream) {
            auto outcome = drive(ticket, stops, {}, [&] { return !response.client_gone(); }, {}, spec);
            if (!outcome.error.empty()) {
                ApiError error(outcome.stats.error_status ? outcome.stats.error_status : 500, "server_error", outcome.error, "", outcome.stats.error_code);
                if (spec && outcome.stats.constraint) {
                    error.strix = Json::object();
                    selection_json(error.strix, *spec, *outcome.stats.constraint);
                }
                throw error;
            }
            const auto parsed = parsed_output(outcome, prompt);
            Json message = Json::object();
            message.set("role", Json("assistant"));
            message.set("content", parsed.content.empty() && parsed.tool_calls.size() ? Json() : Json(sanitize_utf8(parsed.content)));
            if (prompt.thinking) message.set("reasoning_content", Json(sanitize_utf8(parsed.reasoning)));
            std::string finish = outcome.finish == "cancelled" ? "stop" : outcome.finish;
            if (parsed.tool_calls.size()) { message.set("tool_calls", tool_calls_json(parsed.tool_calls)); if (finish == "stop") finish = "tool_calls"; }
            Json choice = Json::object();
            choice.set("index", Json(0));
            choice.set("message", std::move(message));
            choice.set("logprobs", spec && spec->logprobs ? chat_scores(outcome.scores) : Json());
            choice.set("finish_reason", Json(finish));
            Json out = Json::object();
            out.set("id", Json(id)); out.set("object", Json("chat.completion")); out.set("created", Json(created));
            out.set("model", Json(config.model_id));
            Json choices = Json::array(); choices.push(std::move(choice));
            out.set("choices", std::move(choices));
            out.set("usage", usage_json(outcome));
            out.set("system_fingerprint", Json("strix-" + outcome.stats.sampling));
            out.set("strix", (config.stream_receipts ? receipt(&outcome, note, body, limit, sampling_params) : engine_json(outcome, note, body)));
            response.send(200, "application/json", out.dump());
            finish_entry(entry, outcome);
            return;
        }
        stream_admission(ticket, body, response, "server_error");
        if (!response.begin_stream(200, "text/event-stream")) { ticket->cancel(); return; }
        StreamWriter writer{response, interval, progress_interval};
        bool first_chunk = true;
        Json final_receipt;
        Json pending_scores = Json::array();
        const auto chunk = [&](Json delta, const Json& finish) {
            Json choice = Json::object();
            choice.set("index", Json(0));
            choice.set("delta", std::move(delta));
            choice.set("logprobs", spec && spec->logprobs ? chat_scores(pending_scores) : Json());
            pending_scores = Json::array();
            choice.set("finish_reason", finish);
            Json out = Json::object();
            out.set("id", Json(id)); out.set("object", Json("chat.completion.chunk")); out.set("created", Json(created));
            out.set("model", Json(config.model_id));
            Json choices = Json::array(); choices.push(std::move(choice));
            out.set("choices", std::move(choices));
            if ((config.stream_receipts || spec) && (first_chunk || !finish.is_null()))
                out.set("strix", first_chunk ? initial_receipt : final_receipt);
            first_chunk = false;
            return writer.write("data: " + out.dump() + "\n\n");
        };
        bool stream_started = false;
        const auto start_stream = [&] {
            if (stream_started) return true;
            stream_started = true;
            Json first = Json::object(); first.set("role", Json("assistant")); first.set("content", Json(""));
            return chunk(std::move(first), Json());
        };
        start_stream();  // the first event right after the headers, as before; keepalive comments follow it
        ChatSplitter splitter(prompt.thinking, config.tool_call_marker);
        const auto send_deltas = [&](const std::vector<ChatSplitter::Delta>& deltas) {
            if (!start_stream()) return false;
            for (const auto& d : deltas) {
                Json delta = Json::object();
                delta.set(d.reasoning ? "reasoning_content" : "content", Json(sanitize_utf8(d.text)));
                if (!chunk(std::move(delta), Json())) return false;
            }
            return true;
        };
        auto outcome = drive(ticket, stops, [&](const std::string& text) { return send_deltas(splitter.push(text)); },
            [&] { return !response.client_gone(); }, [&] { return writer.idle(*ticket); }, spec, send_deltas, [&](const Json& score, bool) { pending_scores.push(score); });
        if (named_session(body) && !outcome.error.empty()) {
            auto error = stream_error(outcome);
            if (config.stream_receipts || spec) error.set("strix", receipt(&outcome, note, body, limit, sampling_params, spec));
            writer.write("data: " + error.dump() + "\n\n");
            response.end_stream();
            finish_entry(entry, outcome);
            return;
        }
        if (!spec || (!spec->program && !spec->thinking)) send_deltas(splitter.finish());
        std::string finish = outcome.finish == "cancelled" ? "stop" : outcome.finish;
        if (!outcome.error.empty()) {
            Json error = Json::object();
            Json detail = Json::object(); detail.set("message", Json(outcome.error)); detail.set("type", Json("server_error"));
            error.set("error", std::move(detail));
            writer.write("data: " + error.dump() + "\n\n");
        } else if (splitter.saw_tools()) {
            const auto parsed = parsed_output(outcome, prompt);
            if (parsed.tool_calls.size()) {
                auto calls = tool_calls_json(parsed.tool_calls);
                std::int64_t index = 0;
                for (auto& call : calls.as_array()) call.set("index", Json(index++));
                Json delta = Json::object();
                delta.set("tool_calls", std::move(calls));
                chunk(std::move(delta), Json());
                if (finish == "stop") finish = "tool_calls";
            }
        }
        if (config.stream_receipts || spec) final_receipt = receipt(&outcome, note, body, limit, sampling_params, spec);
        chunk(Json::object(), Json(finish));
        if (include_usage) {
            Json out = Json::object();
            out.set("id", Json(id)); out.set("object", Json("chat.completion.chunk")); out.set("created", Json(created));
            out.set("model", Json(config.model_id)); out.set("choices", Json::array());
            out.set("usage", usage_json(outcome));
            out.set("strix", (config.stream_receipts ? receipt(&outcome, note, body, limit, sampling_params) : engine_json(outcome, note, body)));
            writer.write("data: " + out.dump() + "\n\n");
        }
        writer.write("data: [DONE]\n\n");
        response.end_stream();
        finish_entry(entry, outcome);
    }

    // ---- stateless Responses ----
    Json responses_to_chat(const Json& request) const {
        const auto reject = [](const std::string& param, const std::string& message) {
            throw ApiError(400, "invalid_request_error", message, param);
        };
        Json body = Json::object();  // a null member is an absent one (the format's nullable fields)
        for (const auto& [key, value] : request.as_object()) if (!value.is_null()) body.set(key, value);
        if (const auto* v = body.find("previous_response_id"); v && !v->is_null()) reject("previous_response_id", "response history is not supported");
        if (body.find("conversation")) reject("conversation", "conversations are not supported");
        if (bool_or(body, "store", false)) reject("store", "stored responses are not supported");
        if (bool_or(body, "background", false)) reject("background", "background responses are not supported");
        if (const auto* v = body.find("include"); v && !v->is_null()) {
            if (!v->is_array()) reject("include", "include must be an array");
            for (const auto& value : v->as_array()) if (!value.is_string() || value.as_string() != "message.output_text.logprobs") reject("include", "unsupported include");
        }
        if (const auto* v = body.find("truncation"); v && !v->is_null() && !(v->is_string() && v->as_string() == "disabled"))
            reject("truncation", "only truncation disabled is supported (a prompt that does not fit is refused)");
        if (const auto* v = body.find("max_tool_calls"); v && !v->is_null()) reject("max_tool_calls", "max_tool_calls is not supported");
        if (const auto* n = body.find("n"); n && (!n->is_number() || n->as_double() != 1.0)) reject("n", "only n = 1 is supported");
        if (const auto* text = body.find("text")) {
            if (!text->is_object()) reject("text.format", "text.format must be {type:text}");
            if (const auto* verbosity = text->find("verbosity"); verbosity && !verbosity->is_null() &&
                    !(verbosity->is_string() && verbosity->as_string() == "medium"))
                reject("text.verbosity", "text.verbosity other than medium is not supported");
            if (const auto* format = text->find("format"); format && format->is_object())
                if (const auto* type = format->find("type"); type && type->is_string() && type->as_string() == "text" && format->size() != 1U)
                    reject("text.format", "text format must be {type:text}");
        }
        Json chat = Json::object();
        for (const auto& [key, value] : body.as_object())
            if (key != "input" && key != "tools" && key != "tool_choice" && key != "instructions" && key != "reasoning") chat.set(key, value);
        if (const auto* max = body.find("max_output_tokens")) chat.set("max_tokens", *max);
        if (const auto* reasoning = body.find("reasoning")) {
            if (!reasoning->is_object()) reject("reasoning", "reasoning must be an object");
            for (const char* key : {"summary", "generate_summary"})
                if (const auto* v = reasoning->find(key); v && !v->is_null())
                    reject(std::string("reasoning.") + key, "reasoning summaries are not supported (the reply carries the reasoning text itself)");
            if (const auto* effort = reasoning->find("effort"); effort && !effort->is_null()) {
                if (!effort->is_string()) reject("reasoning.effort", "reasoning.effort must be a string");
                chat.set("reasoning_effort", *effort);
            }
        }
        if (const auto* choice = body.find("tool_choice")) {
            if (!choice->is_string() || (choice->as_string() != "auto" && choice->as_string() != "none"))
                reject("tool_choice", "required and named tool_choice are not yet supported");
            chat.set("tool_choice", *choice);
        }
        if (const auto* tools = body.find("tools")) {
            if (!tools->is_array()) reject("tools", "tools must be an array");
            Json converted = Json::array();
            for (std::size_t i = 0; i < tools->size(); ++i) {
                const auto& tool = tools->as_array()[i];
                const auto param = "tools[" + std::to_string(i) + "]";
                const auto* type = tool.find("type");
                if (!type || !type->is_string() || type->as_string() != "function") reject(param + ".type", "only function tools are supported");
                if (const auto* strict = tool.find("strict"); strict && !strict->is_null()) {
                    if (!strict->is_bool()) reject(param + ".strict", "strict must be a boolean");
                    if (strict->as_bool()) reject(param + ".strict", "strict tools are not yet supported");
                }
                const auto* name = tool.find("name");
                if (!name || !name->is_string()) reject(param + ".name", "function name must be a string");
                Json function = Json::object();
                for (const auto& [key, value] : tool.as_object())
                    if (key == "name" || key == "description" || key == "parameters" || key == "strict") function.set(key, value);
                Json t = Json::object(); t.set("type", Json("function")); t.set("function", std::move(function)); converted.push(std::move(t));
            }
            chat.set("tools", std::move(converted));
        }
        const auto text_parts = [&](const Json& value, const std::string& param, bool reasoning) {
            if (value.is_string()) return value.as_string();
            if (!value.is_array()) reject(param, "content must be text or an array of text parts");
            std::string text;
            for (const auto& part : value.as_array()) {
                const auto* type = part.find("type"); const auto* t = part.find("text");
                if (!type || !type->is_string() || !t || !t->is_string()) reject(param, "only plaintext parts are supported");
                const auto& kind = type->as_string();
                if (kind != "input_text" && kind != "output_text" && !(reasoning && (kind == "reasoning_text" || kind == "summary_text" || kind == "text")))
                    reject(param, "unsupported text part type " + kind);
                text += t->as_string();
            }
            return text;
        };
        Json messages = Json::array();
        const auto message = [](const std::string& role, const std::string& text) {
            Json m = Json::object(); m.set("role", Json(role)); m.set("content", Json(text)); return m;
        };
        if (const auto* instructions = body.find("instructions")) {
            if (!instructions->is_string()) reject("instructions", "instructions must be a string");
            messages.push(message("system", instructions->as_string()));
        }
        const auto* input = body.find("input");
        if (!input) reject("input", "input is required");
        std::set<std::string> calls, outputs;
        std::string pending_reasoning;
        bool has_reasoning = false;
        if (input->is_string()) messages.push(message("user", input->as_string()));
        else if (input->is_array()) for (std::size_t i = 0; i < input->size(); ++i) {
            const auto& item = input->as_array()[i];
            const auto param = "input[" + std::to_string(i) + "]";
            if (!item.is_object()) reject(param, "input items must be objects");
            const auto* type = item.find("type");
            if (type && !type->is_string()) reject(param + ".type", "item type must be a string");
            const auto kind = type ? type->as_string() : "message";
            if (kind == "reasoning") {
                const auto* content = item.find("content");
                const auto* summary = item.find("summary");
                std::string plaintext;
                if (content) plaintext = text_parts(*content, param + ".content", true);
                if (plaintext.empty() && summary) plaintext = text_parts(*summary, param + ".summary", true);
                if ((!content && !summary) || (plaintext.empty() && item.find("encrypted_content")))
                    reject(param, "encrypted-only reasoning is not supported");
                pending_reasoning += plaintext;
                has_reasoning = true;
                continue;
            }
            Json m;
            if (kind == "message") {
                const auto* role = item.find("role"); const auto* content = item.find("content");
                if (!role || !role->is_string() || (role->as_string() != "user" && role->as_string() != "system" &&
                        role->as_string() != "developer" && role->as_string() != "assistant")) reject(param + ".role", "unsupported message role");
                if (!content) reject(param + ".content", "message content is required");
                m = message(role->as_string(), text_parts(*content, param + ".content", false));
            } else if (kind == "function_call" || kind == "function_call_output") {
                const auto* call_id = item.find("call_id");
                if (!call_id || !call_id->is_string() || call_id->as_string().empty()) reject(param + ".call_id", "call_id must be a non-empty string");
                const auto& id = call_id->as_string();
                if (kind == "function_call") {
                    if (!calls.insert(id).second) reject(param + ".call_id", "duplicate call_id");
                    const auto* name = item.find("name"); const auto* args = item.find("arguments");
                    if (!name || !name->is_string()) reject(param + ".name", "function name must be a string");
                    if (!args || !args->is_string()) reject(param + ".arguments", "arguments must be a string");
                    Json function = Json::object(); function.set("name", *name); function.set("arguments", *args);
                    Json call = Json::object(); call.set("id", *call_id); call.set("type", Json("function")); call.set("function", std::move(function));
                    // calls after an assistant message (or after another call) belong to that assistant turn, as in chat
                    if (!has_reasoning && messages.size() > 0U) {
                        auto& last = messages.as_array().back();
                        if (last.find("role")->as_string() == "assistant") {
                            Json tool_calls = last.find("tool_calls") ? *last.find("tool_calls") : Json::array();
                            tool_calls.push(std::move(call)); last.set("tool_calls", std::move(tool_calls));
                            continue;
                        }
                    }
                    Json tool_calls = Json::array(); tool_calls.push(std::move(call));
                    m = message("assistant", ""); m.set("tool_calls", std::move(tool_calls));
                } else {
                    if (!calls.count(id) || !outputs.insert(id).second) reject(param + ".call_id", "unresolved or duplicate call_id");
                    const auto* output = item.find("output");
                    if (!output || !output->is_string()) reject(param + ".output", "function output must be a string");
                    m = message("tool", output->as_string()); m.set("tool_call_id", *call_id);
                }
            } else reject(param + ".type", "unsupported input item type " + kind);
            if (has_reasoning && m.find("role")->as_string() == "assistant") {
                m.set("reasoning_content", Json(pending_reasoning)); pending_reasoning.clear(); has_reasoning = false;
            }
            messages.push(std::move(m));
        } else reject("input", "input must be a string or an array");
        if (has_reasoning) reject("input", "reasoning must precede an assistant message");
        chat.set("messages", std::move(messages));
        return chat;
    }

    void responses(const HttpRequest&, HttpResponseWriter& response, const Json& body) {
        const auto chat = responses_to_chat(body);
        const auto prompt = chat_prompt(chat);
        std::string note;
        const auto params = sampling(chat, note);
        const auto stops = stop_strings(chat.find("stop"));
        const bool stream = bool_or(chat, "stream", false);
        const auto interval = keepalive_seconds(chat);
        const auto progress_interval = progress_seconds(chat);
        const auto limit = max_tokens(chat, prompt.tokens.size());
        const auto spec = constraint(body, prompt, params, stops, limit, true);
        auto ticket = submit(prompt.tokens, limit, params, chat, prompt.stable_prefix, spec);
        const auto id = random_id("resp_");
        const auto created = unix_now();
        RecentRequest entry{id, "responses", "ok", created};
        Json answer_scores = Json::array(), pending_scores = Json::array();
        const auto part = [&](bool reasoning, const std::string& text) {
            Json p = Json::object(); p.set("type", Json(reasoning ? "reasoning_text" : "output_text")); p.set("text", Json(sanitize_utf8(text)));
            if (!reasoning) { p.set("annotations", Json::array()); p.set("logprobs", answer_scores); }
            return p;
        };
        const auto item = [&](bool reasoning, const std::string& text, const std::string& item_id, const std::string& status) {
            Json out = Json::object(); out.set("type", Json(reasoning ? "reasoning" : "message")); out.set("id", Json(item_id));
            if (reasoning) out.set("summary", Json::array());
            else { out.set("role", Json("assistant")); out.set("status", Json(status)); }
            Json content = Json::array(); content.push(part(reasoning, text)); out.set("content", std::move(content)); return out;
        };
        const auto envelope = [&](const Outcome* run, Json output, const std::string& status) {
            static const Outcome none;
            const Outcome& o = run ? *run : none;
            Json out = Json::object(); out.set("id", Json(id)); out.set("object", Json("response")); out.set("created_at", Json(created));
            out.set("status", Json(status)); out.set("model", Json(config.model_id)); out.set("output", std::move(output));
            if (status == "incomplete") {  // the budget ran out, or a tool call was cut (stop string, end of sequence)
                Json d = Json::object(); d.set("reason", Json(o.stats.constraint ? o.stats.constraint->reason : o.finish == "length" ? "max_output_tokens" : "unfinished_tool_call"));
                out.set("incomplete_details", std::move(d));
            }
            if (status == "failed") { Json e = Json::object(); e.set("code", Json("server_error")); e.set("message", Json(o.error)); out.set("error", std::move(e)); }
            Json usage = Json::object(); usage.set("input_tokens", Json(o.stats.prompt_tokens));
            Json details = Json::object(); details.set("cached_tokens", Json(o.stats.cached_tokens + o.stats.imported_tokens));
            usage.set("input_tokens_details", std::move(details)); usage.set("output_tokens", Json(o.output_tokens));
            usage.set("output_tokens_details", Json::object()); usage.set("total_tokens", Json(o.stats.prompt_tokens + o.output_tokens));
            out.set("usage", std::move(usage)); out.set("strix", config.stream_receipts || spec ? receipt(run, note, chat, limit, params, spec) : engine_json(o, note, chat)); return out;
        };
        // a call whose frame did not close (cut by the budget, a stop string or EOS) is not marked completed
        const auto call_status = [](const AssistantOutput& parsed, std::size_t i) {
            return i < parsed.call_complete.size() && !parsed.call_complete[i] ? "incomplete" : "completed";
        };
        // an unclosed frame counts only when the request offered tools (else a literal marker is plain text)
        const bool offers_tools = prompt.tools.is_array() && prompt.tools.size() > 0U;
        const auto cut_call = [&](const AssistantOutput& parsed) {
            if (parsed.unfinished_call && offers_tools) return true;
            for (std::size_t i = 0; i < parsed.call_complete.size(); ++i) if (!parsed.call_complete[i]) return true;
            return false;
        };
        const auto function_item = [](const Json& call, const std::string& status) {
            Json out = Json::object(); out.set("type", Json("function_call")); out.set("id", Json(random_id("fc_")));
            out.set("call_id", Json(random_id("call_"))); out.set("name", *call.find("name"));
            out.set("arguments", Json(call.find("arguments")->dump())); out.set("status", Json(status)); return out;
        };
        if (!stream) {
            auto o = drive(ticket, stops, {}, [&] { return !response.client_gone(); }, {}, spec);
            if (!o.error.empty()) {
                ApiError error(o.stats.error_status ? o.stats.error_status : 500, "server_error", o.error, "", o.stats.error_code);
                if (spec && o.stats.constraint) {
                    error.strix = Json::object();
                    selection_json(error.strix, *spec, *o.stats.constraint);
                }
                throw error;
            }
            answer_scores = o.answer_scores;
            const auto parsed = parsed_output(o, prompt);
            const std::string status = spec ? response_status(o) : o.finish == "length" || cut_call(parsed) ? "incomplete" : "completed";
            Json output = Json::array();
            if (!parsed.reasoning.empty()) output.push(item(true, parsed.reasoning, random_id("rs_"), status));
            output.push(item(false, parsed.content, random_id("msg_"), status));
            if (parsed.tool_calls.is_array()) for (std::size_t i = 0; i < parsed.tool_calls.size(); ++i)
                output.push(function_item(parsed.tool_calls.as_array()[i], call_status(parsed, i)));
            response.send(200, "application/json", envelope(&o, std::move(output), status).dump()); finish_entry(entry, o); return;
        }
        if (!response.begin_stream(200, "text/event-stream")) { ticket->cancel(); return; }
        StreamWriter writer{response, interval, progress_interval};
        std::uint64_t sequence = 0;
        const auto event = [&](const std::string& type, Json data) {
            data.set("type", Json(type)); data.set("sequence_number", Json(sequence++));
            const bool ok = writer.write("event: " + type + "\ndata: " + data.dump() + "\n\n");
            if (!ok) ticket->cancel();
            return ok;
        };
        bool stream_started = false;
        const auto start_stream = [&] {
            if (stream_started) return true;
            stream_started = true;
            auto initial = envelope(nullptr, Json::array(), "in_progress");
            if (!config.stream_receipts && !spec) {
                Json copy = Json::object(); for (const auto& [key, value] : initial.as_object()) if (key != "strix") copy.set(key, value);
                initial = std::move(copy);
            }
            Json e = Json::object(); e.set("response", initial);
            if (!event("response.created", std::move(e))) return false;
            Json progress = Json::object(); progress.set("response", std::move(initial)); return event("response.in_progress", std::move(progress));
        };
        Json output = Json::array();
        bool open = false, open_reasoning = false, saw_message = false;
        std::string open_id, text;
        const auto fields = [&] {
            Json e = Json::object(); e.set("item_id", Json(open_id)); e.set("output_index", Json(output.size())); e.set("content_index", Json(0)); return e;
        };
        const auto close = [&](const std::string& status) {
            if (!open) return;
            const std::string kind = open_reasoning ? "response.reasoning_text" : "response.output_text";
            auto done = fields(); done.set("text", Json(sanitize_utf8(text)));
            if (!open_reasoning) done.set("logprobs", answer_scores);
            event(kind + ".done", std::move(done));
            auto p = fields(); p.set("part", part(open_reasoning, text)); event("response.content_part.done", std::move(p));
            auto complete = item(open_reasoning, text, open_id, status);
            auto e = fields(); e.set("item", complete); event("response.output_item.done", std::move(e));
            output.push(std::move(complete)); open = false;
        };
        const auto add = [&](bool reasoning) {
            open = true; open_reasoning = reasoning; saw_message = saw_message || !reasoning;
            open_id = random_id(reasoning ? "rs_" : "msg_"); text.clear();
            auto added = fields(); auto empty = item(reasoning, "", open_id, "in_progress"); empty.set("content", Json::array());
            added.set("item", std::move(empty)); event("response.output_item.added", std::move(added));
            auto p = fields(); p.set("part", part(reasoning, "")); event("response.content_part.added", std::move(p));
        };
        start_stream();  // the first event right after the headers, as before; keepalive comments follow it
        ChatSplitter splitter(prompt.thinking, config.tool_call_marker);
        const auto deltas = [&](const std::vector<ChatSplitter::Delta>& values) {
            if (!start_stream()) return false;
            for (const auto& d : values) {
                if (open && open_reasoning != d.reasoning) close("completed");
                if (!open) add(d.reasoning);
                text += d.text;
                auto e = fields(); e.set("delta", Json(sanitize_utf8(d.text)));
                if (!d.reasoning) { e.set("logprobs", pending_scores); pending_scores = Json::array(); }
                if (!event(d.reasoning ? "response.reasoning_text.delta" : "response.output_text.delta", std::move(e))) return false;
            }
            return true;
        };
        auto o = drive(ticket, stops, [&](const std::string& t) { return deltas(splitter.push(t)); },
            [&] { return !response.client_gone(); }, [&] { return writer.idle(*ticket); }, spec, deltas, [&](const Json& score, bool reasoning_score) { if (!reasoning_score) { pending_scores.push(score); answer_scores.push(score); } });
        if (!spec || (!spec->program && !spec->thinking)) deltas(splitter.finish());
        // parsed whether or not the splitter saw a full marker: a frame cut inside its opening tag still counts
        const auto parsed = parsed_output(o, prompt);
        const std::string status = spec ? response_status(o) : !o.error.empty() ? "failed" : o.finish == "length" || cut_call(parsed) ? "incomplete" : "completed";
        // items end completed or incomplete; a failure is the response's status
        const std::string item_status = status == "completed" ? "completed" : "incomplete";
        if (pending_scores.size()) deltas({{false, ""}});
        close(item_status);
        if (!saw_message && o.error.empty()) { add(false); close(item_status); }
        if (splitter.saw_tools()) {
            for (std::size_t i = 0; i < parsed.tool_calls.size(); ++i) {
                auto complete = function_item(parsed.tool_calls.as_array()[i], call_status(parsed, i)); auto empty = parse_json(complete.dump()); empty.set("arguments", Json("")); empty.set("status", Json("in_progress"));
                Json e = Json::object(); e.set("output_index", Json(output.size())); e.set("item", std::move(empty)); event("response.output_item.added", std::move(e));
                const auto args = complete.find("arguments")->as_string();
                Json delta = Json::object(); delta.set("output_index", Json(output.size())); delta.set("item_id", *complete.find("id")); delta.set("delta", Json(args));
                event("response.function_call_arguments.delta", std::move(delta));
                Json done = Json::object(); done.set("output_index", Json(output.size())); done.set("item_id", *complete.find("id")); done.set("arguments", Json(args));
                event("response.function_call_arguments.done", std::move(done));
                Json end = Json::object(); end.set("output_index", Json(output.size())); end.set("item", complete); event("response.output_item.done", std::move(end)); output.push(std::move(complete));
            }
        }
        Json terminal = Json::object(); terminal.set("response", envelope(&o, std::move(output), status)); event("response." + status, std::move(terminal));
        response.end_stream(); finish_entry(entry, o);
    }

    // ---- completions ----
    void completions(const HttpRequest&, HttpResponseWriter& response, const Json& body) {
        const auto* prompt_value = body.find("prompt");
        if (!prompt_value) throw ApiError(400, "invalid_request_error", "prompt is required", "prompt");
        // One candidate and no insertion; numeric logprobs requests scores, including zero.
        for (const char* key : {"n", "best_of"})
            if (const auto* v = body.find(key); v && !v->is_null() && (!v->is_number() || v->as_double() != 1.0))
                throw ApiError(400, "invalid_request_error", std::string("only ") + key + " = 1 is supported", key);
        if (const auto* v = body.find("suffix"); v && !v->is_null() && !(v->is_string() && v->as_string().empty()))
            throw ApiError(400, "invalid_request_error", "suffix is not supported by this server yet", "suffix");
        std::vector<std::uint32_t> prompt;
        const auto take_ids = [&](const Json& array) {
            for (const auto& id : array.as_array()) {
                if (!id.is_number() || id.as_int() < 0 || static_cast<std::uint64_t>(id.as_int()) >= tokenizer->vocab_size())
                    throw ApiError(400, "invalid_request_error", "prompt token IDs must be in the vocabulary", "prompt");
                prompt.push_back(static_cast<std::uint32_t>(id.as_int()));
            }
        };
        if (prompt_value->is_string()) prompt = tokenizer->encode(prompt_value->as_string(), true);
        else if (prompt_value->is_array() && prompt_value->size() == 1U && prompt_value->as_array()[0].is_string())
            prompt = tokenizer->encode(prompt_value->as_array()[0].as_string(), true);
        else if (prompt_value->is_array() && prompt_value->size() && prompt_value->as_array()[0].is_number()) take_ids(*prompt_value);
        else throw ApiError(400, "invalid_request_error", "prompt must be one string or one array of token IDs", "prompt");
        if (prompt.empty()) throw ApiError(400, "invalid_request_error", "prompt is empty", "prompt");
        if (bool_or(body, "echo", false)) throw ApiError(400, "invalid_request_error", "echo is not supported", "echo");
        std::string note;
        const auto sampling_params = sampling(body, note);
        const auto stops = stop_strings(body.find("stop"));
        const bool stream = bool_or(body, "stream", false);
        const auto interval = keepalive_seconds(body);
        const auto progress_interval = progress_seconds(body);
        const auto limit = max_tokens(body, prompt.size());
        const auto spec = selection_policy(body, {}, prompt, sampling_params, true);
        Json pending_scores = Json::array(); std::uint64_t text_offset = 0;
        const auto initial_receipt = config.stream_receipts || spec ? receipt(nullptr, note, body, limit, sampling_params, spec) : Json();
        auto ticket = submit(prompt, limit, sampling_params, body, 0, spec);
        queue_headers(response, *ticket);
        const auto id = random_id("cmpl-");
        const auto created = unix_now();
        RecentRequest entry{id, "completions", "ok", created};
        bool first_chunk = true;
        const auto chunk = [&](const std::string& text, const Json& finish, bool final_usage, const Outcome* outcome) {
            Json choice = Json::object();
            choice.set("index", Json(0)); choice.set("text", Json(sanitize_utf8(text)));
            choice.set("logprobs", spec && spec->logprobs ? completion_scores(outcome && !stream ? outcome->scores : pending_scores, text_offset) : Json()); pending_scores = Json::array(); choice.set("finish_reason", finish);
            Json out = Json::object();
            out.set("id", Json(id)); out.set("object", Json("text_completion")); out.set("created", Json(created));
            out.set("model", Json(config.model_id));
            Json choices = Json::array(); choices.push(std::move(choice));
            out.set("choices", std::move(choices));
            if (final_usage && outcome) { out.set("usage", usage_json(*outcome)); out.set("strix", (config.stream_receipts ? receipt(outcome, note, body, limit, sampling_params) : engine_json(*outcome, note, body))); }
            if (stream && (config.stream_receipts || spec) && first_chunk && !outcome) out.set("strix", initial_receipt);
            first_chunk = false;
            return out;
        };
        if (!stream) {
            auto outcome = drive(ticket, stops, {}, [&] { return !response.client_gone(); }, {}, spec);
            if (!outcome.error.empty()) throw ApiError(outcome.stats.error_status ? outcome.stats.error_status : 500, "server_error", outcome.error, "", outcome.stats.error_code);
            response.send(200, "application/json", chunk(outcome.text, Json(outcome.finish == "cancelled" ? "stop" : outcome.finish), true, &outcome).dump());
            finish_entry(entry, outcome);
            return;
        }
        stream_admission(ticket, body, response, "server_error");
        if (!response.begin_stream(200, "text/event-stream")) { ticket->cancel(); return; }
        StreamWriter writer{response, interval, progress_interval};
        // with receipts: an empty first chunk carries the policy right after the headers (keepalive comments follow it)
        if (config.stream_receipts || spec) writer.write("data: " + chunk("", Json(), false, nullptr).dump() + "\n\n");
        auto outcome = drive(ticket, stops, [&](const std::string& text) {
            return writer.write("data: " + chunk(text, Json(), false, nullptr).dump() + "\n\n"); }, [&] { return !response.client_gone(); }, [&] { return writer.idle(*ticket); }, spec, {}, [&](const Json& score, bool) { pending_scores.push(score); });
        if (named_session(body) && !outcome.error.empty()) {
            auto error = stream_error(outcome);
            if (config.stream_receipts || spec) error.set("strix", receipt(&outcome, note, body, limit, sampling_params, spec));
            writer.write("data: " + error.dump() + "\n\n");
            response.end_stream();
            finish_entry(entry, outcome);
            return;
        }
        writer.write("data: " + chunk("", Json(outcome.finish == "cancelled" ? "stop" : outcome.finish), true, &outcome).dump() + "\n\n");
        writer.write("data: [DONE]\n\n");
        response.end_stream();
        finish_entry(entry, outcome);
    }

    // ---- Anthropic messages ----
    Json anthropic_to_chat(const Json& body, bool& thinking) const {
        Json messages = Json::array();
        if (const auto* system = body.find("system"); system && !system->is_null()) {
            std::string text;
            if (system->is_string()) text = system->as_string();
            else if (system->is_array()) for (const auto& block : system->as_array()) if (const auto* t = block.find("text"); t && t->is_string()) text += (text.empty() ? "" : "\n") + t->as_string();
            Json m = Json::object(); m.set("role", Json("system")); m.set("content", Json(text));
            messages.push(std::move(m));
        }
        const auto* input = body.find("messages");
        if (!input || !input->is_array() || input->size() == 0U) throw ApiError(400, "invalid_request_error", "messages must be a non-empty array", "messages");
        for (const auto& message : input->as_array()) {
            const auto* role = message.find("role");
            if (!role || !role->is_string()) throw ApiError(400, "invalid_request_error", "every message needs a role", "messages");
            const auto* content = message.find("content");
            if (content && content->is_string()) {
                Json m = Json::object(); m.set("role", *role); m.set("content", *content); messages.push(std::move(m));
                continue;
            }
            if (!content || !content->is_array()) throw ApiError(400, "invalid_request_error", "message content must be a string or an array", "messages");
            std::string text, reasoning;
            Json calls = Json::array();
            for (const auto& block : content->as_array()) {
                const auto* type = block.find("type");
                const auto kind = type && type->is_string() ? type->as_string() : std::string{};
                if (kind == "text") { if (const auto* t = block.find("text"); t && t->is_string()) text += t->as_string(); }
                else if (kind == "thinking") { if (const auto* t = block.find("thinking"); t && t->is_string()) reasoning += t->as_string(); }
                else if (kind == "tool_use") {
                    Json function = Json::object();
                    function.set("name", block.find("name") ? *block.find("name") : Json(""));
                    function.set("arguments", block.find("input") ? *block.find("input") : Json::object());
                    Json call = Json::object(); call.set("type", Json("function")); call.set("function", std::move(function));
                    calls.push(std::move(call));
                } else if (kind == "tool_result") {
                    std::string result;
                    if (const auto* c = block.find("content")) {
                        if (c->is_string()) result = c->as_string();
                        else if (c->is_array()) for (const auto& part : c->as_array()) if (const auto* t = part.find("text"); t && t->is_string()) result += t->as_string();
                    }
                    if (!text.empty()) { Json m = Json::object(); m.set("role", *role); m.set("content", Json(text)); messages.push(std::move(m)); text.clear(); }
                    Json m = Json::object(); m.set("role", Json("tool")); m.set("content", Json(result));
                    messages.push(std::move(m));
                } else if (kind == "redacted_thinking") {
                } else throw ApiError(400, "invalid_request_error", "unsupported content block type " + kind, "messages");
            }
            if (!text.empty() || calls.size() || !reasoning.empty() || role->as_string() == "assistant") {
                Json m = Json::object(); m.set("role", *role); m.set("content", Json(text));
                if (!reasoning.empty()) m.set("reasoning_content", Json(reasoning));
                if (calls.size()) m.set("tool_calls", std::move(calls));
                messages.push(std::move(m));
            }
        }
        Json chat = Json::object();
        chat.set("messages", std::move(messages));
        const auto* choice = body.find("tool_choice");
        const auto* choice_type = choice && choice->is_object() ? choice->find("type") : nullptr;
        if (choice_type && choice_type->is_string() && choice_type->as_string() == "none") chat.set("tool_choice", Json("none"));
        if (const auto* v = body.find("output_config"); v && !v->is_null()) {  // null members are absent
            if (!v->is_object()) throw ApiError(400, "invalid_request_error", "output_config must be an object", "output_config");
            for (const auto& [key, value] : v->as_object())
                if (!value.is_null())
                    throw ApiError(400, "invalid_request_error", "output_config." + key + " is not supported by this server yet", "output_config." + key);
        }
        if (const auto* tools = body.find("tools"); tools && tools->is_array() && tools->size()) {
            Json converted = Json::array();
            for (const auto& tool : tools->as_array()) {
                Json function = Json::object();
                if (const auto* n = tool.find("name")) function.set("name", *n);
                if (const auto* d = tool.find("description")) function.set("description", *d);
                if (const auto* s = tool.find("input_schema")) function.set("parameters", *s);
                Json t = Json::object(); t.set("type", Json("function")); t.set("function", std::move(function));
                converted.push(std::move(t));
            }
            chat.set("tools", std::move(converted));
        }
        thinking = config.default_thinking;
        if (const auto* t = body.find("thinking"); t && t->is_object())
            if (const auto* type = t->find("type"); type && type->is_string()) {
                if (type->as_string() != "enabled" && type->as_string() != "disabled")
                    throw ApiError(400, "invalid_request_error", "thinking.type must be enabled or disabled", "thinking.type");
                thinking = type->as_string() == "enabled";
            }
        chat.set("enable_thinking", Json(thinking));
        return chat;
    }

    void messages(const HttpRequest&, HttpResponseWriter& response, const Json& body) {
        refuse_unexecuted(body, false, false);
        bool thinking = false;
        auto chat = anthropic_to_chat(body, thinking);
        const auto prompt = chat_prompt(chat);
        std::string note;
        const auto sampling_params = sampling(body, note);
        if (const auto* t = body.find("thinking"); thinking && t && t->is_object() && t->find("budget_tokens")) {
            if (config.engine_thinking_budget && config.enforce_anthropic_thinking_budget) {
                Json options = Json::object(); options.set("thinking_policy", Json("thinking-budget-v1"));
                options.set("max_thinking_tokens", *t->find("budget_tokens")); chat.set("strix", std::move(options));
            } else note += std::string(note.empty() ? "" : "; ") + "thinking.budget_tokens not enforced: thinking ends by itself or at max_tokens"
                + (config.engine_thinking_budget ? " (the server enforces it with --http-anthropic-thinking-budget enforce)" : "");
        }
        const auto stops = stop_strings(body.find("stop_sequences"));
        const bool stream = bool_or(body, "stream", false);
        const auto interval = keepalive_seconds(body);
        const auto progress_interval = progress_seconds(body);
        const auto* max = body.find("max_tokens");
        if (!max || !max->is_number()) throw ApiError(400, "invalid_request_error", "max_tokens is required", "max_tokens");
        const auto limit = max_tokens(body, prompt.tokens.size());
        const auto spec = constraint(chat, prompt, sampling_params, stops, limit);
        const auto initial_receipt = config.stream_receipts || spec ? receipt(nullptr, note, body, limit, sampling_params, spec) : Json();
        auto ticket = submit(prompt.tokens, limit, sampling_params, body, prompt.stable_prefix, spec);
        queue_headers(response, *ticket);
        const auto id = random_id("msg_");
        RecentRequest entry{id, "messages", "ok", unix_now()};
        const auto stop_reason = [&](const Outcome& o, bool tools) {
            if (!o.matched.empty()) return std::string("stop_sequence");
            if (tools) return std::string("tool_use");
            return o.finish == "length" ? std::string("max_tokens") : std::string("end_turn");
        };
        const auto usage = [&](const Outcome& o) {
            Json u = Json::object();
            const auto reused = o.stats.cached_tokens + o.stats.imported_tokens;  // prefix cache + imported state
            u.set("input_tokens", Json(o.stats.prompt_tokens - std::min(o.stats.prompt_tokens, reused)));
            u.set("cache_read_input_tokens", Json(reused));
            u.set("cache_creation_input_tokens", Json(0));
            u.set("output_tokens", Json(o.output_tokens));
            return u;
        };
        if (!stream) {
            auto outcome = drive(ticket, stops, {}, [&] { return !response.client_gone(); }, {}, spec);
            if (!outcome.error.empty()) {
                ApiError error(outcome.stats.error_status ? outcome.stats.error_status : 500, "api_error", outcome.error, "", outcome.stats.error_code);
                if (spec && outcome.stats.constraint) {
                    error.strix = Json::object(); selection_json(error.strix, *spec, *outcome.stats.constraint);
                }
                throw error;
            }
            const auto parsed = parsed_output(outcome, prompt);
            Json content = Json::array();
            if (prompt.thinking && !parsed.reasoning.empty()) {
                Json b = Json::object(); b.set("type", Json("thinking")); b.set("thinking", Json(sanitize_utf8(parsed.reasoning))); b.set("signature", Json(""));
                content.push(std::move(b));
            }
            if (!parsed.content.empty()) {
                Json b = Json::object(); b.set("type", Json("text")); b.set("text", Json(sanitize_utf8(parsed.content)));
                content.push(std::move(b));
            }
            for (const auto& call : parsed.tool_calls.as_array()) {
                Json b = Json::object(); b.set("type", Json("tool_use")); b.set("id", Json(random_id("toolu_")));
                b.set("name", *call.find("name")); b.set("input", *call.find("arguments"));
                content.push(std::move(b));
            }
            Json out = Json::object();
            out.set("id", Json(id)); out.set("type", Json("message")); out.set("role", Json("assistant"));
            out.set("model", Json(config.model_id)); out.set("content", std::move(content));
            out.set("stop_reason", Json(stop_reason(outcome, parsed.tool_calls.size() > 0U)));
            out.set("stop_sequence", outcome.matched.empty() ? Json() : Json(outcome.matched));
            out.set("usage", usage(outcome));
            out.set("strix", (config.stream_receipts ? receipt(&outcome, note, body, limit, sampling_params) : engine_json(outcome, note, body)));
            response.send(200, "application/json", out.dump());
            finish_entry(entry, outcome);
            return;
        }
        stream_admission(ticket, body, response, "api_error");
        if (!response.begin_stream(200, "text/event-stream")) { ticket->cancel(); return; }
        StreamWriter writer{response, interval, progress_interval};
        const auto event = [&](const char* name, const Json& data) {
            return writer.write(std::string("event: ") + name + "\ndata: " + data.dump() + "\n\n");
        };
        bool stream_started = false;
        const auto start_stream = [&] {
            if (stream_started) return true;
            stream_started = true;
            Json message = Json::object();
            message.set("id", Json(id)); message.set("type", Json("message")); message.set("role", Json("assistant"));
            message.set("model", Json(config.model_id)); message.set("content", Json::array());
            message.set("stop_reason", Json()); message.set("stop_sequence", Json());
            Json u = Json::object(); u.set("input_tokens", Json(prompt.tokens.size())); u.set("output_tokens", Json(0));
            message.set("usage", std::move(u));
            if (config.stream_receipts || spec) message.set("strix", initial_receipt);
            Json start = Json::object(); start.set("type", Json("message_start")); start.set("message", std::move(message));
            return event("message_start", start);
        };
        std::int64_t index = -1;
        std::string open_kind;
        const auto close_block = [&] {
            if (open_kind.empty()) return;
            Json stop = Json::object(); stop.set("type", Json("content_block_stop")); stop.set("index", Json(index));
            event("content_block_stop", stop);
            open_kind.clear();
        };
        const auto open_block = [&](const std::string& kind, Json block) {
            close_block();
            ++index;
            open_kind = kind;
            Json start = Json::object(); start.set("type", Json("content_block_start")); start.set("index", Json(index));
            start.set("content_block", std::move(block));
            event("content_block_start", start);
        };
        start_stream();  // the first event right after the headers, as before; keepalive comments follow it
        ChatSplitter splitter(prompt.thinking, config.tool_call_marker);
        const auto send_deltas = [&](const std::vector<ChatSplitter::Delta>& deltas) {
            if (!start_stream()) return false;
            for (const auto& d : deltas) {
                const std::string kind = d.reasoning ? "thinking" : "text";
                if (open_kind != kind) {
                    Json block = Json::object(); block.set("type", Json(kind));
                    if (d.reasoning) { block.set("thinking", Json("")); block.set("signature", Json("")); } else block.set("text", Json(""));
                    open_block(kind, std::move(block));
                }
                Json delta = Json::object();
                delta.set("type", Json(d.reasoning ? "thinking_delta" : "text_delta"));
                delta.set(d.reasoning ? "thinking" : "text", Json(sanitize_utf8(d.text)));
                Json e = Json::object(); e.set("type", Json("content_block_delta")); e.set("index", Json(index)); e.set("delta", std::move(delta));
                if (!event("content_block_delta", e)) return false;
            }
            return true;
        };
        auto outcome = drive(ticket, stops, [&](const std::string& text) { return send_deltas(splitter.push(text)); },
            [&] { return !response.client_gone(); }, [&] { return writer.idle(*ticket); }, spec, send_deltas);
        if (named_session(body) && !outcome.error.empty()) {
            auto error = stream_error(outcome, true);
            if (config.stream_receipts || spec) error.set("strix", receipt(&outcome, note, body, limit, sampling_params, spec));
            writer.write("event: error\ndata: " + error.dump() + "\n\n");
            response.end_stream();
            finish_entry(entry, outcome);
            return;
        }
        if (!spec || (!spec->program && !spec->thinking)) send_deltas(splitter.finish());
        bool tools = false;
        if (splitter.saw_tools()) {
            const auto parsed = parsed_output(outcome, prompt);
            for (const auto& call : parsed.tool_calls.as_array()) {
                tools = true;
                Json block = Json::object(); block.set("type", Json("tool_use")); block.set("id", Json(random_id("toolu_")));
                block.set("name", *call.find("name")); block.set("input", Json::object());
                open_block("tool_use", std::move(block));
                Json delta = Json::object(); delta.set("type", Json("input_json_delta")); delta.set("partial_json", Json(call.find("arguments")->dump()));
                Json e = Json::object(); e.set("type", Json("content_block_delta")); e.set("index", Json(index)); e.set("delta", std::move(delta));
                event("content_block_delta", e);
            }
        }
        close_block();
        Json delta = Json::object();
        delta.set("stop_reason", Json(stop_reason(outcome, tools)));
        delta.set("stop_sequence", outcome.matched.empty() ? Json() : Json(outcome.matched));
        Json m = Json::object(); m.set("type", Json("message_delta")); m.set("delta", std::move(delta)); m.set("usage", usage(outcome));
        event("message_delta", m);
        Json stop = Json::object(); stop.set("type", Json("message_stop"));
        if (config.stream_receipts || spec) stop.set("strix", receipt(&outcome, note, body, limit, sampling_params, spec));
        else if (!note.empty()) { Json s = Json::object(); s.set("note", Json(note)); stop.set("strix", std::move(s)); }  // a note is never dropped
        event("message_stop", stop);
        response.end_stream();
        finish_entry(entry, outcome);
    }

    void finish_entry(RecentRequest& entry, const Outcome& o) {
        entry.status = !o.error.empty() ? "error" : o.finish == "cancelled" ? "cancelled" : "done";
        entry.stop_reason = o.finish;
        entry.prompt = o.stats.prompt_tokens; entry.cached = o.stats.cached_tokens; entry.output = o.output_tokens;
        entry.cycles = o.stats.cycles; entry.accepted = o.stats.accepted; entry.proposed = o.stats.proposed;
        entry.queue_ms = o.stats.queue_ms; entry.prefill_ms = o.stats.prefill_ms; entry.decode_ms = o.stats.decode_ms;
        entry.imported = o.stats.imported_tokens;  // every record (its prefill rate); reported as a field only when observed
        if (o.observed) { entry.armed = true; entry.first_token_ms = o.stats.first_token_ms; }
        record(entry);
    }

    // ---- service endpoints ----
    // An arming request: arms (a new period when the last one expired); the period lasts this service's expiry more.
    void arm() {
        if (observe_arm(expiry_seconds)) { const std::lock_guard lock(mutex); armed_unix = unix_now(); }
    }
    Json status() {
        const auto q = queue.snapshot();
        const bool armed = observe_armed();
        const auto ob = armed ? queue.observed() : ObserveSnapshot{};
        Json out = Json::object();
        out.set("schema", Json("strix-serve-status-v1"));
        if (config.capabilities) out.set("capabilities_id", Json(capabilities_id));
        out.set("model", Json(config.model_id));
        out.set("engine", Json(config.engine_description));
        out.set("context_tokens", Json(context_tokens));
        out.set("uptime_seconds", Json(std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count()));
        std::lock_guard lock(mutex);
        out.set("state", Json(q.running ? std::string("generating") : q.queued ? std::string("queued") : state));
        Json queue_json = Json::object();
        queue_json.set("queued", Json(q.queued)); queue_json.set("running", Json(q.running));
        queue_json.set("completed", Json(q.completed)); queue_json.set("failed", Json(q.failed)); queue_json.set("cancelled", Json(q.cancelled));
        out.set("queue", std::move(queue_json));
        Json totals = Json::object();
        totals.set("requests", Json(requests_total)); totals.set("prompt_tokens", Json(q.prompt_tokens));
        totals.set("cached_tokens", Json(q.cached_tokens)); totals.set("generated_tokens", Json(q.generated_tokens));
        totals.set("busy_seconds", Json(q.busy_seconds));
        out.set("totals", std::move(totals));
        Json list = Json::array();
        for (const auto& r : recent) {
            Json e = Json::object();
            e.set("id", Json(r.id)); e.set("route", Json(r.route)); e.set("time", Json(r.time)); e.set("status", Json(r.status));
            e.set("stop_reason", Json(r.stop_reason));
            e.set("prompt_tokens", Json(r.prompt)); e.set("reused_tokens", Json(r.cached)); e.set("output_tokens", Json(r.output));
            e.set("queue_ms", Json(r.queue_ms)); e.set("prefill_ms", Json(r.prefill_ms)); e.set("decode_ms", Json(r.decode_ms));
            e.set("duration_ms", Json(r.queue_ms + r.prefill_ms + r.decode_ms));
            // the rate of the rows this engine computed: prompt - cached - imported (imported rows were not prefilled here)
            e.set("prefill_tok_s", Json(r.prefill_ms > 0 ? 1000.0 * static_cast<double>(r.prompt - std::min(r.prompt, r.cached + r.imported)) / r.prefill_ms : 0.0));
            e.set("decode_tok_s", Json(r.decode_ms > 0 ? 1000.0 * static_cast<double>(r.output) / r.decode_ms : 0.0));
            e.set("mtp_cycles", Json(r.cycles)); e.set("mtp_accepted", Json(r.accepted)); e.set("mtp_proposed", Json(r.proposed));
            if (r.armed) {  // C8 and C1 fields of observed requests (others: absent, never zero)
                const auto restored = r.cached + r.imported;
                e.set("cached_tokens", Json(r.cached)); e.set("imported_tokens", Json(r.imported));
                e.set("computed_tokens", Json(r.output != 0U && r.prompt > restored ? r.prompt - restored : std::uint64_t{0}));
                e.set("first_token_ms", r.first_token_ms ? Json(*r.first_token_ms) : Json());
                if (!r.response_id.empty()) e.set("response_id", Json(r.response_id));
            }
            list.push(std::move(e));
        }
        out.set("requests", std::move(list));
        {  // always present (no arming): tells an engine restart from a counter reset; fixed at startup, no token path
            Json instance_json = Json::object();
            instance_json.set("id", Json(instance)); instance_json.set("start_time", Json(start_unix));
            out.set("instance", std::move(instance_json));
        }
        if (armed) {
            out.set("armed", Json(true));
            out.set("armed_since", armed_unix != 0 ? Json(armed_unix) : Json());
            Json o = Json::object();
            o.set("since", Json("armed"));
            o.set("requests", Json(ob.requests)); o.set("prompt_tokens", Json(ob.prompt_tokens));
            o.set("cached_tokens", Json(ob.cached_tokens)); o.set("imported_tokens", Json(ob.imported_tokens));
            o.set("computed_tokens", Json(ob.computed_tokens));
            o.set("ttft_seconds", histogram_json(ob.ttft_seconds, ttft_seconds_bounds));
            o.set("decode_tokens_per_second", histogram_json(ob.decode_tokens_per_second, decode_tok_s_bounds));
            out.set("observed", std::move(o));
        }
        if (config.status_extra) config.status_extra(out);
        deployment_status(out, config.deployment, queue, config.http_slots);
        return out;
    }
    std::string metrics() {
        const auto q = queue.snapshot();
        std::ostringstream m;
        // exact values: integers print as integers, doubles round-trip (the default 6 significant digits rounded every
        // count above 10^6)
        m.precision(std::numeric_limits<double>::max_digits10);
        const auto counter = [&](const char* name, const char* help, auto value, const char* type = "counter") {
            m << "# HELP " << name << ' ' << help << "\n# TYPE " << name << ' ' << type << '\n' << name << ' ' << value << '\n';
        };
        counter("strix_requests_completed_total", "Generations completed.", q.completed);
        counter("strix_requests_failed_total", "Generations that failed.", q.failed);
        counter("strix_requests_cancelled_total", "Generations cancelled (client gone or stop string).", q.cancelled);
        counter("strix_prompt_tokens_total", "Prompt tokens of finished generations.", q.prompt_tokens);
        counter("strix_prompt_tokens_cached_total", "Prompt tokens restored from the prefix state cache.", q.cached_tokens);
        counter("strix_generated_tokens_total", "Generated tokens.", q.generated_tokens);
        counter("strix_engine_busy_seconds_total", "Seconds the engine spent on generations.", q.busy_seconds);
        counter("strix_queue_waiting", "Generations waiting in the queue.", q.queued, "gauge");
        counter("strix_queue_running", "Generations running.", q.running, "gauge");
        if (observe_armed()) {
            const auto ob = queue.observed();
            counter("strix_instrumentation_armed", "1 while the instrumentation is armed (a scrape arms it; it expires after the last one).", 1, "gauge");
            m << "# HELP strix_engine_info Engine instance of this process.\n# TYPE strix_engine_info gauge\nstrix_engine_info{instance=\""
              << instance << "\",model=\"" << label_value(config.model_id) << "\"} 1\n";
            counter("strix_engine_start_time_seconds", "Unix time the engine process started serving.", start_unix, "gauge");
            counter("strix_prompt_tokens_imported_total", "Prompt tokens whose state was imported from another device (since armed).",
                ob.imported_tokens);
            counter("strix_prompt_tokens_computed_total", "Prompt tokens prefilled by this engine: prompt - cached - imported (since armed).",
                ob.computed_tokens);
            histogram_text(m, "strix_ttft_seconds", "Engine-observed time from submit to the first published token (since armed).",
                ob.ttft_seconds, ttft_seconds_bounds);
            histogram_text(m, "strix_decode_tokens_per_second", "Per-request decode rate: generated tokens / decode time (since armed).",
                ob.decode_tokens_per_second, decode_tok_s_bounds);
        }
        if (config.metrics_extra) m << config.metrics_extra();
        return m.str();
    }
};

ApiService::ApiService(ApiConfig config, std::shared_ptr<const BpeTokenizer> tokenizer, GenerationQueue& queue,
    std::uint64_t context_tokens, bool engine_samples)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(tokenizer), queue, context_tokens, engine_samples)) {}
ApiService::~ApiService() = default;

std::vector<std::uint32_t> ApiService::chat_prompt(const Json& body) const { return impl_->chat_prompt(body).tokens; }
void ApiService::set_state(std::string state) { const std::lock_guard lock(impl_->mutex); impl_->state = std::move(state); }

void ApiService::handle(const HttpRequest& request, HttpResponseWriter& raw_response) {
    auto& impl = *impl_;
    // C1: a valid X-Request-Id names this request's ring record and is echoed on its response
    struct CurrentId {
        explicit CurrentId(const std::string& id) : previous(std::move(current_request_id)) {
            current_request_id = valid_request_id(id) ? id : std::string{};
        }
        ~CurrentId() { current_request_id = std::move(previous); }
        std::string previous;
        CurrentId(const CurrentId&) = delete;
        CurrentId& operator=(const CurrentId&) = delete;
    } const current_id(request.header("x-request-id"));
    // the accepted id is echoed in the response head; the body and stream writes go to the connection directly
    if (!current_request_id.empty()) raw_response.add_response_header("X-Request-Id", current_request_id);
    HttpResponseWriter& response = raw_response;
    const bool anthropic = request.path == "/v1/messages";
    const auto error_body = [&](const ApiError& e) {
        Json detail = Json::object();
        if (anthropic) {
            detail.set("type", Json(e.type())); detail.set("message", Json(std::string(e.what())));
            Json out = Json::object(); out.set("type", Json("error")); out.set("error", std::move(detail));
            return out.dump();
        }
        detail.set("message", Json(std::string(e.what()))); detail.set("type", Json(e.type()));
        detail.set("param", e.param().empty() ? Json() : Json(e.param())); detail.set("code", e.code().empty() ? Json() : Json(e.code()));
        if (!e.keyword().empty()) detail.set("keyword", Json(e.keyword()));
        Json out = Json::object(); out.set("error", std::move(detail));
        if (!e.strix.is_null()) out.set("strix", e.strix);
        return out.dump();
    };
    try {
        if (request.method == "GET" && (request.path == "/health" || request.path == "/v1/health")) {
            std::string state;
            { const std::lock_guard lock(impl.mutex); state = impl.state; }
            if (impl.queue.draining() && state != "failed") state = "draining";
            if (impl.config.health_record && serve_health_record_enabled()) {
                BackendHealth evidence;
                try {
                    if (impl.config.health_record->backend_probe_enabled && impl.config.health_provider)
                        evidence = impl.config.health_provider();
                } catch (...) { evidence.state = BackendState::failed; }
                auto health = render_health(*impl.config.health_record, evidence, state,
                    impl.config.model_id, impl.queue.snapshot(),
                    impl.config.capabilities ? std::string_view(impl.capabilities_id) : std::string_view{});
                deployment_health(health.body, impl.config.deployment);
                response.send(health.status, "application/json", health.body.dump());
                return;
            }
            const bool failed = state == "failed", draining = state == "draining";
            Json out = Json::object(); out.set("status", Json(failed ? "failed" : draining ? "draining" : "ok")); out.set("model", Json(impl.config.model_id));
            if (impl.config.capabilities) out.set("capabilities_id", Json(impl.capabilities_id));
            deployment_health(out, impl.config.deployment);
            response.send(failed || draining ? 503 : 200, "application/json", out.dump());
            return;
        }
        if (request.method == "GET" && request.path == "/metrics") { impl.arm(); response.send(200, "text/plain; version=0.0.4", impl.metrics()); return; }
        if (request.method == "GET" && request.path == "/api/status") {
            if (detail_requested(request.query)) impl.arm();
            response.send(200, "application/json", impl.status().dump());
            return;
        }
        impl.authorize(request);
        if (handle_deployment_route(request, response, impl.queue, impl.config.deployment, impl.config.on_drain)) return;
        if (request.method == "GET" && request.path == "/v1/cache/stats" && impl.config.state_cache_stats) {
            response.send(200,"application/json",impl.config.state_cache_stats()); return;
        }
        if (impl.config.engine_routes && impl.config.engine_routes(request, response)) return;
        if (request.method == "GET" && request.path == "/v1/capabilities" && impl.config.capabilities) {
            response.send(200, "application/json", impl.capabilities_document.dump());
            return;
        }
        if (request.method == "GET" && request.path == "/v1/models") {
            Json model = Json::object();
            model.set("id", Json(impl.config.model_id)); model.set("object", Json("model")); model.set("created", Json(unix_now()));
            if (impl.config.capabilities) model.set("capabilities_id", Json(impl.capabilities_id));
            model.set("owned_by", Json("strix")); model.set("context_length", Json(impl.context_tokens));
            Json data = Json::array(); data.push(std::move(model));
            Json out = Json::object(); out.set("object", Json("list")); out.set("data", std::move(data));
            response.send(200, "application/json", out.dump());
            return;
        }
        if (request.method != "POST") throw ApiError(404, "not_found_error", "unknown route " + request.method + " " + request.path);
        if (request.path == "/v1/responses" && !impl.config.responses)
            throw ApiError(404, "not_found_error", "unknown route " + request.path);
        Json body;
        try { body = parse_request_json(request.body); }
        catch (const JsonError& e) { throw ApiError(400, "invalid_request_error", std::string("invalid JSON body: ") + e.what()); }
        if (!body.is_object()) throw ApiError(400, "invalid_request_error", "the request body must be a JSON object");
        if (request.path == "/v1/chat/completions") impl.chat_completions(request, response, body);
        else if (request.path == "/v1/completions") impl.completions(request, response, body);
        else if (request.path == "/v1/responses" && impl.config.responses) impl.responses(request, response, body);
        else if (request.path == "/v1/messages") impl.messages(request, response, body);
        else if (request.path == "/tokenize") {
            std::vector<std::uint32_t> ids;
            if (const auto* messages = body.find("messages")) { static_cast<void>(messages); ids = impl.chat_prompt(body).tokens; }
            else if (const auto* text = body.find("prompt"); text && text->is_string()) ids = impl.tokenizer->encode(text->as_string(), true);
            else throw ApiError(400, "invalid_request_error", "tokenize takes a prompt string or chat messages");
            Json tokens = Json::array();
            for (const auto id : ids) tokens.push(Json(static_cast<std::uint64_t>(id)));
            Json out = Json::object(); out.set("tokens", std::move(tokens)); out.set("count", Json(static_cast<std::uint64_t>(ids.size())));
            response.send(200, "application/json", out.dump());
        } else if (request.path == "/detokenize") {
            const auto* tokens = body.find("tokens");
            if (!tokens || !tokens->is_array()) throw ApiError(400, "invalid_request_error", "detokenize takes a tokens array");
            std::vector<std::uint32_t> ids;
            for (const auto& id : tokens->as_array()) {
                if (!id.is_number() || id.as_int() < 0 || static_cast<std::uint64_t>(id.as_int()) >= impl.tokenizer->vocab_size())
                    throw ApiError(400, "invalid_request_error", "token IDs must be in the vocabulary");
                ids.push_back(static_cast<std::uint32_t>(id.as_int()));
            }
            Json out = Json::object(); out.set("text", Json(sanitize_utf8(impl.tokenizer->decode(ids))));
            response.send(200, "application/json", out.dump());
        } else throw ApiError(404, "not_found_error", "unknown route " + request.path);
    } catch (const ApiError& e) {
        RecentRequest entry{"", request.path, "error", unix_now()};
        if (e.status() >= 500) impl.record(entry);
        response.send(e.status(), "application/json", error_body(e));
    } catch (const std::exception& e) {
        impl.record(RecentRequest{"", request.path, "error", unix_now()});
        response.send(500, "application/json", error_body(ApiError(500, anthropic ? "api_error" : "server_error", e.what())));
    }
}

}  // namespace strix::serve
