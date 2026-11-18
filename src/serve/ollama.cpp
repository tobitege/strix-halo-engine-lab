#include "strix/serve/ollama.hpp"

#include "strix/env_flags.hpp"

#include <array>
#include <cstdio>
#include <ctime>
#include <set>
#include <string_view>

namespace strix::serve::ollama {
namespace {

[[noreturn]] void bad(const std::string& code, const std::string& message) { throw RequestError(400, code, message); }

std::int64_t integer_option(const Json& value, std::string_view key, bool allow_unbounded = false) {
    std::int64_t n = 0;
    try { n = value.as_int(); }
    catch (const JsonError&) { bad("invalid_option", "options." + std::string(key) + " must be an integer in the signed 64-bit range"); }
    if (n < 1 && !(allow_unbounded && (n == -1 || n == -2)))
        bad("invalid_option", "options." + std::string(key) + (allow_unbounded
            ? " must be positive, -1 (unbounded) or -2 (fill the context)" : " must be a positive integer"));
    return n;
}

Json obj(std::initializer_list<Json::Member> members) { return Json(Json::Object(members)); }

std::string rfc3339(std::chrono::system_clock::time_point t) {
    const auto since = t.time_since_epoch();
    const auto secs = std::chrono::duration_cast<std::chrono::seconds>(since);
    const auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(since - secs).count();
    const std::time_t tt = static_cast<std::time_t>(secs.count());
    std::tm tm{};
#if defined(_WIN32)
    gmtime_s(&tm, &tt);
#else
    gmtime_r(&tt, &tm);
#endif
    std::array<char, 48> buffer{};
    std::snprintf(buffer.data(), buffer.size(), "%04d-%02d-%02dT%02d:%02d:%02d.%09lldZ", tm.tm_year + 1900, tm.tm_mon + 1,
        tm.tm_mday, tm.tm_hour, tm.tm_min, tm.tm_sec, static_cast<long long>(nanos));
    return buffer.data();
}

bool flag(const Json& request, std::string_view key, bool fallback) {
    const auto* v = request.find(key);
    if (!v || v->is_null()) return fallback;
    if (!v->is_bool()) bad("invalid_field", std::string(key) + " must be a boolean");
    return v->as_bool();
}

std::string text_field(const Json& request, std::string_view key) {
    const auto* v = request.find(key);
    if (!v || v->is_null()) return {};
    if (!v->is_string()) bad("invalid_field", std::string(key) + " must be a string");
    return v->as_string();
}

double number_or(const Json& options, std::string_view key, double neutral) {
    const auto* v = options.find(key);
    if (!v || v->is_null()) return neutral;
    if (!v->is_number()) bad("invalid_option", "options." + std::string(key) + " must be a number");
    return v->as_double();
}

// Raw base64 image (the Ollama form carries no media type): sniff the container from the first bytes.
std::string data_url(const std::string& base64) {
    static constexpr std::array<std::pair<std::string_view, std::string_view>, 4> kinds{{
        {"iVBORw0KGgo", "image/png"}, {"/9j/", "image/jpeg"}, {"R0lGOD", "image/gif"}, {"UklGR", "image/webp"}}};
    for (const auto& [prefix, type] : kinds)
        if (std::string_view(base64).starts_with(prefix)) return "data:" + std::string(type) + ";base64," + base64;
    bad("invalid_image", "images must be base64 PNG, JPEG, GIF or WebP");
}

Json content_with_images(const std::string& text, const Json* images, const Served& served) {
    if (!images || images->is_null() || (images->is_array() && images->size() == 0U)) return Json(text);
    if (!images->is_array()) bad("invalid_field", "images must be an array of base64 strings");
    if (!served.images) bad("images_unsupported", "this engine has no image adapter loaded; images cannot be executed");
    Json parts = Json::array();
    if (!text.empty()) parts.push(obj({{"type", Json("text")}, {"text", Json(text)}}));
    for (const auto& image : images->as_array()) {
        if (!image.is_string()) bad("invalid_field", "images must be an array of base64 strings");
        parts.push(obj({{"type", Json("image_url")}, {"image_url", obj({{"url", Json(data_url(image.as_string()))}})}}));
    }
    return parts;
}

Json chat_message(const Json& in, std::size_t index, const Served& served) {
    if (!in.is_object()) bad("invalid_field", "messages[" + std::to_string(index) + "] must be an object");
    static const std::set<std::string, std::less<>> known{"role", "content", "thinking", "images", "tool_calls", "tool_name"};
    for (const auto& [key, value] : in.as_object())
        if (!known.contains(key)) bad("unknown_field", "messages[" + std::to_string(index) + "]." + key + " is not part of the chat message contract");
    Json out = Json::object();
    const auto role = text_field(in, "role");
    if (role.empty()) bad("invalid_field", "messages[" + std::to_string(index) + "].role is required");
    out.set("role", Json(role));
    out.set("content", content_with_images(text_field(in, "content"), in.find("images"), served));
    if (const auto thinking = text_field(in, "thinking"); !thinking.empty()) out.set("reasoning_content", Json(thinking));
    if (const auto name = text_field(in, "tool_name"); !name.empty()) out.set("name", Json(name));
    if (const auto* calls = in.find("tool_calls"); calls && !calls->is_null()) {
        if (!calls->is_array()) bad("invalid_field", "tool_calls must be an array");
        Json list = Json::array();
        std::size_t n = 0;
        for (const auto& call : calls->as_array()) {
            const auto* function = call.is_object() ? call.find("function") : nullptr;
            if (!function || !function->is_object()) bad("invalid_field", "tool_calls[].function must be an object");
            const auto* arguments = function->find("arguments");
            const std::string args = !arguments || arguments->is_null() ? "{}" : arguments->is_string() ? arguments->as_string() : arguments->dump();
            list.push(obj({{"id", Json("call_" + std::to_string(n++))}, {"type", Json("function")},
                {"function", obj({{"name", Json(text_field(*function, "name"))}, {"arguments", Json(args)}})}}));
        }
        out.set("tool_calls", std::move(list));
    }
    return out;
}

void options_into(const Json* options, const Served& served, Json& body, Translated& out) {
    if (!options || options->is_null()) return;
    if (!options->is_object()) bad("invalid_field", "options must be an object");
    // Same meaning as the OpenAI field: forwarded under the OpenAI name and validated by the shared resolver.
    static const std::array<std::pair<std::string_view, std::string_view>, 8> forwarded{{
        {"temperature", "temperature"}, {"top_p", "top_p"}, {"top_k", "top_k"}, {"min_p", "min_p"}, {"seed", "seed"},
        {"repeat_penalty", "repetition_penalty"}, {"presence_penalty", "presence_penalty"}, {"frequency_penalty", "frequency_penalty"}}};
    // Resource hints of the Ollama runner: they cannot change the committed tokens of this engine.
    static const std::set<std::string, std::less<>> resource{"num_batch", "num_gpu", "main_gpu", "num_thread", "use_mmap",
        "use_mlock", "low_vram", "numa", "num_keep"};
    const bool penalties_neutral = number_or(*options, "repeat_penalty", 1.0) == 1.0 &&
        number_or(*options, "presence_penalty", 0.0) == 0.0 && number_or(*options, "frequency_penalty", 0.0) == 0.0;
    const bool mirostat_off = number_or(*options, "mirostat", 0.0) == 0.0;
    for (const auto& [key, value] : options->as_object()) {
        if (value.is_null()) continue;
        bool done = false;
        for (const auto& [from, to] : forwarded)
            if (key == from) { body.set(std::string(to), value); done = true; }
        if (done) continue;
        if (resource.contains(key)) { out.no_effect.push_back(key); continue; }
        if (key == "num_predict") {
            const auto n = integer_option(value, key, true);
            if (n > 0) body.set("max_tokens", Json(n));
        } else if (key == "stop") {
            if (value.is_string()) { Json list = Json::array(); list.push(value); body.set("stop", std::move(list)); }
            else if (value.is_array()) body.set("stop", value);
            else bad("invalid_option", "options.stop must be a string or an array of strings");
        } else if (key == "num_ctx") {
            const auto n = static_cast<std::uint64_t>(integer_option(value, key));
            // Below the served context it is a hard bound (no truncation, no context shift); at or above it, no effect.
            if (n < served.context_tokens) out.context_ceiling = n; else out.no_effect.push_back(key);
        } else if (key == "typical_p" || key == "tfs_z") {
            if (number_or(*options, key, 1.0) != 1.0) bad("unsupported_option", "options." + key + " != 1 changes the result and is not executed by this engine");
        } else if (key == "mirostat") {
            if (!mirostat_off) bad("unsupported_option", "options.mirostat sampling is not executed by this engine");
        } else if (key == "mirostat_tau" || key == "mirostat_eta") {
            if (!mirostat_off) bad("unsupported_option", "options.mirostat is not executed by this engine");
            out.no_effect.push_back(key);  // inert while mirostat is off
        } else if (key == "repeat_last_n") {
            if (!penalties_neutral) bad("unsupported_option", "options.repeat_last_n with active penalties is not executed by this engine (the penalty window is the whole output)");
            out.no_effect.push_back(key);
        } else if (key == "penalize_newline") {
            if (!value.is_bool()) bad("invalid_option", "options.penalize_newline must be a boolean");
            if (!penalties_neutral) bad("unsupported_option", "options.penalize_newline with active penalties is not executed by this engine");
            out.no_effect.push_back(key);
        } else bad("unknown_option", "options." + key + " is not a known option");
    }
}

void format_into(const Json* format, Json& body) {
    if (!format || format->is_null() || (format->is_string() && format->as_string().empty())) return;
    if (format->is_string() && format->as_string() == "json") body.set("response_format", obj({{"type", Json("json_object")}}));
    else if (format->is_object())
        body.set("response_format", obj({{"type", Json("json_schema")}, {"json_schema", obj({{"name", Json("response")}, {"schema", *format}})}}));
    else bad("invalid_field", "format must be \"json\" or a JSON schema object");
}

void think_into(const Json* think, Json& body) {
    if (!think || think->is_null()) return;
    if (think->is_bool()) body.set("chat_template_kwargs", obj({{"enable_thinking", *think}}));
    else if (think->is_string() && (think->as_string() == "low" || think->as_string() == "medium" || think->as_string() == "high"))
        body.set("reasoning_effort", *think);
    else bad("invalid_field", "think must be a boolean or one of \"low\", \"medium\", \"high\"");
}

Json details() {
    return obj({{"parent_model", Json("")}, {"format", Json("strix")}, {"family", Json("")}, {"families", Json::array()},
        {"parameter_size", Json("")}, {"quantization_level", Json("")}});
}

Json model_entry(const Served& served) {
    return obj({{"name", Json(served.model_id)}, {"model", Json(served.model_id)}, {"size", Json(0)}, {"digest", Json("")},
        {"details", details()}});
}

}  // namespace

bool enabled_by_default() {
    static const bool on = envcfg::flag("STRIX_SERVE_OLLAMA", true, EnvFlagMatch::exact);
    return on;
}

std::string rfc3339_now() { return rfc3339(std::chrono::system_clock::now()); }

std::string error_body(const std::string& message) { return obj({{"error", Json(message)}}).dump(); }

void require_model(const Json& request, const Served& served, bool allow_name_alias) {
    std::string model = text_field(request, "model");
    if (model.empty() && allow_name_alias) model = text_field(request, "name");
    if (model.empty()) bad("model_required", "model is required");
    if (model != served.model_id && model != served.model_id + ":latest")
        throw RequestError(404, "model_not_found", "model '" + model + "' not found");
}

Translated translate(Route route, const Json& request, const Served& served) {
    static const std::set<std::string, std::less<>> chat_fields{"model", "messages", "tools", "format", "options", "stream", "keep_alive", "think"};
    static const std::set<std::string, std::less<>> generate_fields{"model", "prompt", "suffix", "images", "format", "options", "system",
        "template", "context", "stream", "raw", "keep_alive", "think"};
    const auto& known = route == Route::chat ? chat_fields : generate_fields;
    for (const auto& [key, value] : request.as_object())
        if (!known.contains(key)) bad("unknown_field", key + " is not part of the " + (route == Route::chat ? "chat" : "generate") + " contract");
    require_model(request, served);
    Translated out;
    out.stream = flag(request, "stream", true);
    if (const auto* keep = request.find("keep_alive"); keep && !keep->is_null()) out.no_effect.emplace_back("keep_alive");  // weights stay resident
    Json body = Json::object();
    body.set("model", Json(served.model_id));
    if (route == Route::chat) {
        const auto* messages = request.find("messages");
        if (messages && !messages->is_null() && !messages->is_array()) bad("invalid_field", "messages must be an array");
        if (!messages || messages->is_null() || messages->size() == 0U) { out.load_only = true; return out; }
        Json list = Json::array();
        std::size_t i = 0;
        for (const auto& m : messages->as_array()) list.push(chat_message(m, i++, served));
        body.set("messages", std::move(list));
        if (const auto* tools = request.find("tools"); tools && !tools->is_null()) {
            if (!tools->is_array()) bad("invalid_field", "tools must be an array");
            if (tools->size()) body.set("tools", *tools);
        }
    } else {
        if (!text_field(request, "suffix").empty()) bad("suffix_unsupported", "suffix (fill-in-the-middle) is not executed by this engine");
        if (!text_field(request, "template").empty()) bad("template_unsupported", "template overrides would change the prompt; the model template is fixed");
        if (const auto* context = request.find("context"); context && !context->is_null() && (!context->is_array() || context->size()))
            bad("context_unsupported", "context token arrays are not accepted; send the conversation as /api/chat messages");
        out.raw = flag(request, "raw", false);
        const auto prompt = text_field(request, "prompt");
        const auto system = text_field(request, "system");
        const auto* images = request.find("images");
        const bool has_images = images && images->is_array() && images->size();
        if (prompt.empty() && !has_images) { out.load_only = true; return out; }
        if (out.raw) {
            if (has_images) bad("images_unsupported", "raw prompts cannot carry images");
            if (!system.empty()) bad("invalid_field", "system is not applied to raw prompts; put it in the prompt");
            body.set("prompt", Json(prompt));
        } else {
            Json list = Json::array();
            if (!system.empty()) list.push(obj({{"role", Json("system")}, {"content", Json(system)}}));
            list.push(obj({{"role", Json("user")}, {"content", content_with_images(prompt, images, served)}}));
            body.set("messages", std::move(list));
        }
    }
    format_into(request.find("format"), body);
    if (!out.raw) think_into(request.find("think"), body);
    else if (const auto* think = request.find("think"); think && !think->is_null()) bad("invalid_field", "think does not apply to raw prompts");
    options_into(request.find("options"), served, body, out);
    body.set("stream", Json(out.stream));
    if (out.stream) body.set("stream_options", obj({{"include_usage", Json(true)}}));
    out.body = std::move(body);
    return out;
}

Json tags(const Served& served, std::int64_t modified_unix) {
    auto entry = model_entry(served);
    entry.set("modified_at", Json(rfc3339(std::chrono::system_clock::time_point(std::chrono::seconds(modified_unix)))));
    Json models = Json::array(); models.push(std::move(entry));
    return obj({{"models", std::move(models)}});
}

Json ps(const Served& served) {
    auto entry = model_entry(served);
    entry.set("expires_at", Json("9999-12-31T23:59:59Z"));  // resident for the life of the server
    entry.set("size_vram", Json(0));
    entry.set("context_length", Json(served.context_tokens));
    Json models = Json::array(); models.push(std::move(entry));
    return obj({{"models", std::move(models)}});
}

Json show(const Served& served) {
    Json capabilities = Json::array();
    capabilities.push(Json("completion"));
    capabilities.push(Json("tools"));
    if (served.thinking) capabilities.push(Json("thinking"));
    if (served.images) capabilities.push(Json("vision"));
    if (served.embeddings) capabilities.push(Json("embedding"));
    return obj({{"modelfile", Json("")}, {"parameters", Json("")}, {"template", Json("")}, {"details", details()},
        {"model_info", obj({{"general.architecture", Json("strix")}, {"strix.context_length", Json(served.context_tokens)}})},
        {"capabilities", std::move(capabilities)}});
}

// ---- response re-encoding ----

ResponseWriter::ResponseWriter(HttpResponseWriter& inner, Route route, std::string model, std::vector<std::string> no_effect)
    : inner_(inner), route_(route), model_(std::move(model)), no_effect_(std::move(no_effect)) {
    if (!no_effect_.empty()) {
        std::string joined;
        for (const auto& key : no_effect_) joined += (joined.empty() ? "" : ",") + key;
        inner_.add_response_header("X-Strix-Ollama-No-Effect", joined);
    }
}

Json ResponseWriter::line(bool done) const {
    Json out = Json::object();
    out.set("model", Json(model_));
    out.set("created_at", Json(rfc3339_now()));
    if (route_ == Route::chat) out.set("message", obj({{"role", Json("assistant")}, {"content", Json("")}}));
    else out.set("response", Json(""));
    out.set("done", Json(done));
    return out;
}

void ResponseWriter::timing_from(const Json& out) {
    if (const auto* usage = out.find("usage"); usage && usage->is_object()) {
        if (const auto* p = usage->find("prompt_tokens"); p && p->is_number()) prompt_tokens_ = p->as_int();
        if (const auto* c = usage->find("completion_tokens"); c && c->is_number()) output_tokens_ = c->as_int();
    }
    const auto* strix = out.find("strix");
    if (!strix || !strix->is_object()) return;
    const auto take = [&](const Json& from) {
        if (const auto* v = from.find("prefill_ms"); v && v->is_number()) prefill_ms_ = v->as_double();
        if (const auto* v = from.find("decode_ms"); v && v->is_number()) decode_ms_ = v->as_double();
    };
    take(*strix);
    for (const auto& [key, value] : strix->as_object()) if (value.is_object()) take(value);
}

void ResponseWriter::finish(Json& done) const {
    using ns = std::chrono::nanoseconds;
    const auto now = std::chrono::steady_clock::now();
    const auto first = first_ == std::chrono::steady_clock::time_point{} ? now : first_;
    done.set("done_reason", Json(finish_ == "tool_calls" ? std::string("stop") : finish_));
    done.set("total_duration", Json(static_cast<std::int64_t>(std::chrono::duration_cast<ns>(now - start_).count())));
    done.set("load_duration", Json(0));  // weights are resident before the first request
    if (prompt_tokens_ >= 0) done.set("prompt_eval_count", Json(prompt_tokens_));
    done.set("prompt_eval_duration", Json(prefill_ms_ >= 0 ? static_cast<std::int64_t>(prefill_ms_ * 1e6)
        : static_cast<std::int64_t>(std::chrono::duration_cast<ns>(first - start_).count())));
    if (output_tokens_ >= 0) done.set("eval_count", Json(output_tokens_));
    done.set("eval_duration", Json(decode_ms_ >= 0 ? static_cast<std::int64_t>(decode_ms_ * 1e6)
        : static_cast<std::int64_t>(std::chrono::duration_cast<ns>(now - first).count())));
}

namespace {
Json tool_calls_out(const Json::Array& calls_in) {
    Json calls = Json::array();
    for (const auto& call : calls_in) {
        const auto* function = call.is_object() ? call.find("function") : nullptr;
        if (!function) continue;
        Json arguments = Json::object();
        if (const auto* a = function->find("arguments"); a && a->is_string() && !a->as_string().empty()) {
            try { arguments = parse_request_json(a->as_string()); } catch (const JsonError&) { arguments = *a; }
        } else if (a && a->is_object()) arguments = *a;
        const auto* name = function->find("name");
        calls.push(obj({{"function", obj({{"name", name && name->is_string() ? *name : Json("")}, {"arguments", std::move(arguments)}})}}));
    }
    return calls;
}
void set_text(Json& line, Route route, const char* field, const std::string& text) {
    if (route == Route::chat) {
        Json message = obj({{"role", Json("assistant")}, {"content", Json(std::string_view(field) == "content" ? text : std::string())}});
        if (std::string_view(field) == "thinking") message.set("thinking", Json(text));
        line.set("message", std::move(message));
    } else line.set(std::string_view(field) == "content" ? "response" : "thinking", Json(text));
}
}  // namespace

void ResponseWriter::send(int status, std::string_view, std::string_view body, const std::vector<std::pair<std::string, std::string>>& headers) {
    Json in;
    try { in = parse_request_json(body); } catch (const JsonError&) { in = Json::object(); }
    if (status != 200) {
        std::string message = "request failed";
        if (const auto* e = in.is_object() ? in.find("error") : nullptr) {
            if (e->is_string()) message = e->as_string();
            else if (const auto* m = e->is_object() ? e->find("message") : nullptr; m && m->is_string()) message = m->as_string();
        }
        inner_.send(status, "application/json", error_body(message), headers);
        return;
    }
    timing_from(in);
    Json done = line(true);
    std::string content, thinking;
    Json calls = Json::array();
    if (const auto* choices = in.find("choices"); choices && choices->is_array() && choices->size()) {
        const auto& choice = choices->as_array().front();
        if (const auto* f = choice.find("finish_reason"); f && f->is_string()) finish_ = f->as_string();
        const Json* message = choice.find("message");
        const Json& source = message && message->is_object() ? *message : choice;
        for (const char* key : {"content", "text"}) if (const auto* v = source.find(key); v && v->is_string()) content = v->as_string();
        if (const auto* r = source.find("reasoning_content"); r && r->is_string()) thinking = r->as_string();
        if (const auto* t = source.find("tool_calls"); t && t->is_array()) calls = tool_calls_out(t->as_array());
    }
    if (route_ == Route::chat) {
        Json message = obj({{"role", Json("assistant")}, {"content", Json(content)}});
        if (!thinking.empty()) message.set("thinking", Json(thinking));
        if (calls.size()) message.set("tool_calls", std::move(calls));
        done.set("message", std::move(message));
    } else {
        done.set("response", Json(content));
        if (!thinking.empty()) done.set("thinking", Json(thinking));
    }
    finish(done);
    inner_.send(200, "application/json", done.dump(), headers);
}

bool ResponseWriter::begin_stream(int status, std::string_view, const std::vector<std::pair<std::string, std::string>>& headers) {
    return inner_.begin_stream(status, "application/x-ndjson", headers);
}

std::string ResponseWriter::final_lines() {
    if (failed_ || finished_) return {};
    finished_ = true;
    std::string out;
    if (!tools_.empty()) {
        Json raw = Json::array();
        for (const auto& [index, call] : tools_)
            raw.push(obj({{"function", obj({{"name", Json(call.name)}, {"arguments", Json(call.arguments)}})}}));
        Json l = line(false);
        Json message = obj({{"role", Json("assistant")}, {"content", Json("")}, {"tool_calls", tool_calls_out(raw.as_array())}});
        if (route_ == Route::chat) l.set("message", std::move(message));
        out += l.dump() + "\n";
    }
    Json done = line(true);
    finish(done);
    return out + done.dump() + "\n";
}

std::string ResponseWriter::translate_frame(std::string_view payload) {
    if (payload == "[DONE]") return final_lines();
    Json in;
    try { in = parse_request_json(payload); } catch (const JsonError&) { return {}; }
    if (!in.is_object()) return {};
    if (const auto* e = in.find("error")) {
        failed_ = true;
        const auto* m = e->is_object() ? e->find("message") : nullptr;
        return error_body(m && m->is_string() ? m->as_string() : e->is_string() ? e->as_string() : std::string("generation failed")) + "\n";
    }
    if (in.find("usage")) timing_from(in);
    const auto* choices = in.find("choices");
    if (!choices || !choices->is_array() || !choices->size()) return {};
    const auto& choice = choices->as_array().front();
    if (const auto* f = choice.find("finish_reason"); f && f->is_string()) finish_ = f->as_string();
    const Json* delta = choice.find("delta");
    const Json& source = delta && delta->is_object() ? *delta : choice;
    std::string out;
    const auto emit = [&](const char* field, const std::string& text) {
        if (text.empty()) return;
        if (first_ == std::chrono::steady_clock::time_point{}) first_ = std::chrono::steady_clock::now();
        Json l = line(false);
        set_text(l, route_, field, text);
        out += l.dump() + "\n";
    };
    if (const auto* r = source.find("reasoning_content"); r && r->is_string()) emit("thinking", r->as_string());
    for (const char* key : {"content", "text"}) if (const auto* v = source.find(key); v && v->is_string()) emit("content", v->as_string());
    if (const auto* calls = source.find("tool_calls"); calls && calls->is_array()) {
        std::int64_t fallback = static_cast<std::int64_t>(tools_.size());
        for (const auto& call : calls->as_array()) {
            const auto* index = call.find("index");
            auto& slot = tools_[index && index->is_number() ? index->as_int() : fallback++];
            if (const auto* function = call.find("function"); function && function->is_object()) {
                if (const auto* n = function->find("name"); n && n->is_string()) slot.name += n->as_string();
                if (const auto* a = function->find("arguments"); a && a->is_string()) slot.arguments += a->as_string();
            }
        }
    }
    return out;
}

bool ResponseWriter::write(std::string_view data) {
    pending_.append(data);
    std::string out;
    for (std::size_t end; (end = pending_.find("\n\n")) != std::string::npos;) {
        const std::string_view frame(pending_.data(), end);
        if (frame.starts_with("data: ")) out += translate_frame(frame.substr(6));
        pending_.erase(0, end + 2U);
    }
    if (out.empty()) return !inner_.client_gone();
    return inner_.write(out);
}

void ResponseWriter::end_stream() { inner_.end_stream(); }

}  // namespace strix::serve::ollama
