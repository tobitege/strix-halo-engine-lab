#pragma once

// Resident test mode, phase 1: the engine-neutral host library behind a resident run (load the model
// once, run many arms). It parses and validates the JSON plan, groups arms by the levers that need a
// reload, expands the ABAB run order, computes the paired early-stop statistics and writes the per-run
// receipts and the summary. No engine wiring: the runner that applies levers, resets request state and
// agrees the plan hash / arm digest across TP2 ranks builds on this header in a later phase.
//
// Header-only; needs strix_serve_json (src/serve/json.cpp) at link time.

#include "strix/serve/json.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace strix::serve::resident {

class PlanError : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// ---------------------------------------------------------------------------------------------------
// Hashes

inline constexpr std::uint64_t kFnv64Offset = 0xcbf29ce484222325ULL;
inline constexpr std::uint64_t kFnv64Prime = 0x100000001b3ULL;
inline constexpr std::uint32_t kFnv32Offset = 0x811c9dc5U;
inline constexpr std::uint32_t kFnv32Prime = 0x01000193U;

[[nodiscard]] inline std::uint64_t fnv1a64(std::string_view bytes, std::uint64_t h = kFnv64Offset) noexcept {
    for (const char c : bytes) {
        h ^= static_cast<std::uint8_t>(c);
        h *= kFnv64Prime;
    }
    return h;
}

[[nodiscard]] inline std::uint32_t fnv1a32(std::string_view bytes, std::uint32_t h = kFnv32Offset) noexcept {
    for (const char c : bytes) {
        h ^= static_cast<std::uint8_t>(c);
        h *= kFnv32Prime;
    }
    return h;
}

[[nodiscard]] inline std::string hex_digits(std::uint64_t value, int digits) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out(static_cast<std::size_t>(digits), '0');
    for (int i = digits - 1; i >= 0; --i) {
        out[static_cast<std::size_t>(i)] = kHex[value & 0xfU];
        value >>= 4U;
    }
    return out;
}
[[nodiscard]] inline std::string hex64(std::uint64_t value) { return hex_digits(value, 16); }
[[nodiscard]] inline std::string hex32(std::uint32_t value) { return hex_digits(value, 8); }

// FNV-1a 64 over the little-endian bytes of the generated token ids.
template<class Token>
[[nodiscard]] inline std::uint64_t token_digest(const std::vector<Token>& tokens) noexcept {
    std::uint64_t h = kFnv64Offset;
    for (const Token t : tokens) {
        const auto u = static_cast<std::uint32_t>(t);
        for (unsigned shift = 0; shift < 32U; shift += 8U) {
            h ^= (u >> shift) & 0xffU;
            h *= kFnv64Prime;
        }
    }
    return h;
}

// ---------------------------------------------------------------------------------------------------
// Plan

enum class Order : std::uint8_t {
    abab,        // (A B) x repeats; more than two arms rotate per pass (ABC, BCA, CAB, ...)
    baba,        // (B A) x repeats; more than two arms: the reversed list, rotated per pass
    sequential,  // A x repeats, then B x repeats (reload-separated arms)
};

[[nodiscard]] inline std::string_view order_name(Order order) noexcept {
    switch (order) {
    case Order::abab: return "abab";
    case Order::baba: return "baba";
    case Order::sequential: return "sequential";
    }
    return "abab";
}

// Capture bits.
inline constexpr std::uint32_t kCaptureTiming = 1U;
inline constexpr std::uint32_t kCaptureTokens = 2U;       // generated ids in the receipt
inline constexpr std::uint32_t kCaptureLogits = 4U;       // last logits_last_n rows to a file
inline constexpr std::uint32_t kCaptureStateDigest = 8U;  // engine state digest after the run (leak check)

struct CaptureName {
    std::uint32_t bit;
    std::string_view name;
};
inline constexpr CaptureName kCaptureNames[] = {
    {kCaptureTiming, "timing"}, {kCaptureTokens, "tokens"}, {kCaptureLogits, "logits"},
    {kCaptureStateDigest, "state-digest"}};

// Metrics a plan can name (early_stop.metric) and that receipts/summaries carry.
inline constexpr std::string_view kMetricNames[] = {"ttft_ms", "prefill_ms", "decode_ms", "prefill_tok_s",
                                                    "decode_tok_s"};
[[nodiscard]] inline bool known_metric(std::string_view metric) noexcept {
    return std::find(std::begin(kMetricNames), std::end(kMetricNames), metric) != std::end(kMetricNames);
}

// Env overrides of one arm: (name, value), sorted by name, unique. JSON booleans become "1"/"0",
// numbers keep their source text.
using EnvSet = std::vector<std::pair<std::string, std::string>>;

struct Prompt {
    std::string name;
    std::string path;
};

struct Arm {
    std::string name;
    EnvSet env;
    std::vector<std::size_t> prompts;  // indices into Plan::prompts, plan order
    std::uint32_t steps = 0;
    std::uint32_t capture = kCaptureTiming;
};

struct EarlyStop {
    bool enabled = false;
    std::string metric = "decode_tok_s";
    std::uint32_t min_pairs = 3;
    double band_pct = 1.0;  // equivalence band on the relative paired difference, in percent
    double z = 0.0;         // > 0: fixed critical value instead of the t table
};

struct Plan {
    struct Request {
        std::size_t prompt = 0;
        std::uint32_t max_tokens = 0;
        std::uint64_t seed = 0;
    };
    struct Mtp {
        std::uint32_t depth = 0, rows = 0;
    };
    std::string engine;
    Order order = Order::abab;
    std::uint32_t repeats = 2;  // passes over the order pattern
    std::uint32_t warmup = 1;   // unmeasured runs per arm after each (re)load or arm switch
    std::vector<Prompt> prompts;
    std::uint32_t steps = 0;
    std::uint32_t capture = kCaptureTiming;
    std::uint32_t logits_last_n = 0;
    EarlyStop early_stop;
    bool cold_control = false;  // after all groups, reload and run the first arm once more
    std::vector<Arm> arms;
    // Presence is preserved: old plans keep their canonical bytes and receipts.
    std::optional<std::uint32_t> concurrency;
    std::optional<std::vector<Request>> requests;
    std::optional<std::string> batch_mode;
    std::optional<Mtp> mtp;
    std::optional<bool> solo_control;
    std::string canonical;  // normalized plan JSON: defaults filled in, keys sorted, compact
    std::uint64_t hash = 0;  // fnv1a64(canonical)
};

namespace detail {

inline constexpr std::uint32_t kMaxRepeats = 10000;
inline constexpr std::uint32_t kMaxWarmup = 1000;
inline constexpr std::uint32_t kMaxSteps = 1U << 24U;
inline constexpr std::size_t kMaxNameLen = 64;

[[noreturn]] inline void fail(std::string_view where, std::string_view what) {
    throw PlanError("resident plan: " + std::string(where) + ": " + std::string(what));
}

inline void check_keys(const Json& obj, std::string_view where, std::initializer_list<std::string_view> allowed) {
    for (const auto& member : obj.as_object()) {
        if (std::find(allowed.begin(), allowed.end(), std::string_view(member.first)) == allowed.end())
            fail(where, "unknown key \"" + member.first + "\"");
    }
}

inline const Json& object_at(const Json& value, std::string_view where) {
    if (!value.is_object()) fail(where, "expected an object");
    return value;
}

inline std::uint32_t uint_value(const Json& value, std::string_view where, std::uint32_t lo, std::uint32_t hi) {
    if (!value.is_number()) fail(where, "expected an integer");
    const double d = value.as_double();
    if (!(d >= static_cast<double>(lo) && d <= static_cast<double>(hi)) || d != std::floor(d))
        fail(where, "expected an integer in [" + std::to_string(lo) + ", " + std::to_string(hi) + "]");
    return static_cast<std::uint32_t>(d);
}

inline double number_value(const Json& value, std::string_view where, double lo, double hi) {
    if (!value.is_number()) fail(where, "expected a number");
    const double d = value.as_double();
    if (!(d >= lo && d <= hi)) fail(where, "number out of range");
    return d;
}

inline bool bool_value(const Json& value, std::string_view where) {
    if (!value.is_bool()) fail(where, "expected true or false");
    return value.as_bool();
}

inline const std::string& string_value(const Json& value, std::string_view where) {
    if (!value.is_string()) fail(where, "expected a string");
    return value.as_string();
}

// Arm and prompt names end up in receipt file names: [A-Za-z0-9._-], 1..64 bytes, not starting with '.'.
inline const std::string& safe_name(const Json& value, std::string_view where) {
    const std::string& s = string_value(value, where);
    if (s.empty() || s.size() > kMaxNameLen || s.front() == '.') fail(where, "bad name \"" + s + "\"");
    for (const char c : s) {
        const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' ||
                        c == '-' || c == '.';
        if (!ok) fail(where, "bad name \"" + s + "\" (allowed: letters, digits, '_', '-', '.')");
    }
    return s;
}

inline std::uint32_t capture_value(const Json& value, std::string_view where) {
    if (!value.is_array()) fail(where, "expected an array of capture names");
    std::uint32_t bits = 0;
    for (const Json& item : value.as_array()) {
        const std::string& s = string_value(item, where);
        std::uint32_t bit = 0;
        for (const CaptureName& c : kCaptureNames)
            if (c.name == s) bit = c.bit;
        if (bit == 0) fail(where, "unknown capture \"" + s + "\"");
        if ((bits & bit) != 0) fail(where, "duplicate capture \"" + s + "\"");
        bits |= bit;
    }
    return bits;
}

inline EnvSet env_value(const Json& value, std::string_view where) {
    if (!value.is_object()) fail(where, "expected an object of env overrides");
    EnvSet env;
    for (const auto& [key, v] : value.as_object()) {
        if (key.empty()) fail(where, "empty env name");
        for (const char c : key) {
            const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
            if (!ok) fail(where, "bad env name \"" + key + "\"");
        }
        std::string text;
        if (v.is_string()) {
            text = v.as_string();
        } else if (v.is_number()) {
            text = v.number_source();
        } else if (v.is_bool()) {
            text = v.as_bool() ? "1" : "0";
        } else {
            fail(where, "env \"" + key + "\": expected a string, number or boolean");
        }
        for (const char c : text)
            if (static_cast<unsigned char>(c) < 0x20U) fail(where, "env \"" + key + "\": control character in value");
        env.emplace_back(key, std::move(text));
    }
    std::sort(env.begin(), env.end());
    return env;
}

inline Json capture_json(std::uint32_t bits) {
    Json out = Json::array();
    for (const CaptureName& c : kCaptureNames)
        if ((bits & c.bit) != 0) out.push(Json(std::string(c.name)));
    return out;
}

inline Json env_json(const EnvSet& env) {
    Json out = Json::object();
    for (const auto& [k, v] : env) out.set(k, Json(v));
    return out;
}

// Recursively sorts object members by key (byte order) so dump() is canonical.
inline Json sorted(const Json& value) {
    if (value.is_array()) {
        Json out = Json::array();
        for (const Json& item : value.as_array()) out.push(sorted(item));
        return out;
    }
    if (value.is_object()) {
        Json::Object members;
        for (const auto& [k, v] : value.as_object()) members.emplace_back(k, sorted(v));
        std::sort(members.begin(), members.end(),
                  [](const Json::Member& a, const Json::Member& b) { return a.first < b.first; });
        return Json(std::move(members));
    }
    return value;
}

inline Json normalized_json(const Plan& plan) {
    Json root = Json::object();
    if (plan.concurrency) root.set("concurrency", Json(static_cast<std::uint64_t>(*plan.concurrency)));
    if (plan.requests) {
        Json requests = Json::array();
        for (const Plan::Request& r : *plan.requests) {
            Json item = Json::object();
            item.set("prompt", Json(static_cast<std::uint64_t>(r.prompt)));
            item.set("max_tokens", Json(static_cast<std::uint64_t>(r.max_tokens)));
            item.set("seed", Json(r.seed));
            requests.push(std::move(item));
        }
        root.set("requests", std::move(requests));
    }
    if (plan.batch_mode) root.set("batch_mode", Json(*plan.batch_mode));
    if (plan.mtp) {
        Json mtp = Json::object();
        mtp.set("depth", Json(static_cast<std::uint64_t>(plan.mtp->depth)));
        mtp.set("rows", Json(static_cast<std::uint64_t>(plan.mtp->rows)));
        root.set("mtp", std::move(mtp));
    }
    if (plan.solo_control) root.set("solo_control", Json(*plan.solo_control));
    root.set("engine", Json(plan.engine));
    root.set("order", Json(std::string(order_name(plan.order))));
    root.set("repeats", Json(static_cast<std::uint64_t>(plan.repeats)));
    root.set("warmup", Json(static_cast<std::uint64_t>(plan.warmup)));
    Json prompts = Json::array();
    for (const Prompt& p : plan.prompts) {
        Json item = Json::object();
        item.set("name", Json(p.name));
        item.set("path", Json(p.path));
        prompts.push(std::move(item));
    }
    root.set("prompts", std::move(prompts));
    root.set("steps", Json(static_cast<std::uint64_t>(plan.steps)));
    root.set("capture", capture_json(plan.capture));
    root.set("logits_last_n", Json(static_cast<std::uint64_t>(plan.logits_last_n)));
    Json es = Json::object();
    es.set("enabled", Json(plan.early_stop.enabled));
    es.set("metric", Json(plan.early_stop.metric));
    es.set("min_pairs", Json(static_cast<std::uint64_t>(plan.early_stop.min_pairs)));
    es.set("band_pct", Json(plan.early_stop.band_pct));
    es.set("z", Json(plan.early_stop.z));
    root.set("early_stop", std::move(es));
    root.set("cold_control", Json(plan.cold_control));
    Json arms = Json::array();
    for (const Arm& arm : plan.arms) {
        Json item = Json::object();
        item.set("name", Json(arm.name));
        item.set("env", env_json(arm.env));
        Json names = Json::array();
        for (const std::size_t i : arm.prompts) names.push(Json(plan.prompts[i].name));
        item.set("prompts", std::move(names));
        item.set("steps", Json(static_cast<std::uint64_t>(arm.steps)));
        item.set("capture", capture_json(arm.capture));
        arms.push(std::move(item));
    }
    root.set("arms", std::move(arms));
    return sorted(root);
}

}  // namespace detail

// Parses and validates a plan (plain JSON; duplicate keys and unknown keys are rejected so a typo never
// falls back to a default). The hash covers the normalized plan, so key order, whitespace and spelled-out
// defaults do not change it.
inline constexpr std::size_t kMaxPlanBytes = 1024U * 1024U;
inline constexpr std::size_t kMaxPromptBytes = 64U * 1024U * 1024U;

[[nodiscard]] inline Plan parse_plan(std::string_view text) {
    if (text.size() > kMaxPlanBytes) throw PlanError("resident plan: input exceeds 1 MiB");
    Json root;
    try {
        root = parse_json_unique(text);
    } catch (const JsonError& e) {
        throw PlanError(std::string("resident plan: invalid JSON: ") + e.what());
    }
    detail::object_at(root, "plan");
    detail::check_keys(root, "plan",
                       {"engine", "order", "repeats", "warmup", "prompts", "steps", "capture", "logits_last_n",
                        "early_stop", "cold_control", "arms", "concurrency", "requests", "batch_mode", "mtp",
                        "solo_control"});
    Plan plan;

    const Json* engine = root.find("engine");
    if (engine == nullptr) detail::fail("plan", "missing \"engine\"");
    plan.engine = detail::safe_name(*engine, "engine");

    if (const Json* v = root.find("order")) {
        const std::string& s = detail::string_value(*v, "order");
        if (s == "abab") plan.order = Order::abab;
        else if (s == "baba") plan.order = Order::baba;
        else if (s == "sequential") plan.order = Order::sequential;
        else detail::fail("order", "expected abab, baba or sequential");
    }
    if (const Json* v = root.find("repeats")) plan.repeats = detail::uint_value(*v, "repeats", 1, detail::kMaxRepeats);
    if (const Json* v = root.find("warmup")) plan.warmup = detail::uint_value(*v, "warmup", 0, detail::kMaxWarmup);

    const Json* prompts = root.find("prompts");
    if (prompts == nullptr || !prompts->is_array() || prompts->size() == 0)
        detail::fail("prompts", "expected a non-empty array");
    for (const Json& item : prompts->as_array()) {
        detail::object_at(item, "prompts[]");
        detail::check_keys(item, "prompts[]", {"name", "path"});
        const Json* name = item.find("name");
        const Json* path = item.find("path");
        if (name == nullptr || path == nullptr) detail::fail("prompts[]", "needs \"name\" and \"path\"");
        Prompt p{detail::safe_name(*name, "prompts[].name"), detail::string_value(*path, "prompts[].path")};
        if (p.path.empty()) detail::fail("prompts[].path", "empty path");
        for (const Prompt& q : plan.prompts)
            if (q.name == p.name) detail::fail("prompts", "duplicate prompt \"" + p.name + "\"");
        plan.prompts.push_back(std::move(p));
    }

    const Json* steps = root.find("steps");
    if (steps == nullptr) detail::fail("plan", "missing \"steps\"");
    plan.steps = detail::uint_value(*steps, "steps", 1, detail::kMaxSteps);
    if (const Json* v = root.find("concurrency"))
        plan.concurrency = detail::uint_value(*v, "concurrency", 1, 32);
    if (const Json* v = root.find("batch_mode")) {
        plan.batch_mode = detail::string_value(*v, "batch_mode");
        if (*plan.batch_mode != "lockstep" && *plan.batch_mode != "continuous")
            detail::fail("batch_mode", "expected lockstep or continuous");
    }
    if (const Json* v = root.find("mtp")) {
        detail::object_at(*v, "mtp");
        detail::check_keys(*v, "mtp", {"depth", "rows"});
        Plan::Mtp mtp;
        if (const Json* f = v->find("depth")) mtp.depth = detail::uint_value(*f, "mtp.depth", 0, 8);
        if (const Json* f = v->find("rows")) mtp.rows = detail::uint_value(*f, "mtp.rows", 0, 64);
        plan.mtp = mtp;
    }
    if (const Json* v = root.find("solo_control"))
        plan.solo_control = detail::bool_value(*v, "solo_control");
    if (const Json* v = root.find("requests")) {
        if (!plan.concurrency || !v->is_array() || v->size() != *plan.concurrency)
            detail::fail("requests", "expected an array of concurrency objects");
        plan.requests.emplace();
        for (const Json& item : v->as_array()) {
            detail::object_at(item, "requests[]");
            detail::check_keys(item, "requests[]", {"prompt", "max_tokens", "seed"});
            Plan::Request request{plan.requests->size() % plan.prompts.size(), plan.steps, 0};
            if (const Json* f = item.find("prompt")) {
                // Compare in size_t after validating the bounded integer; do not narrow the prompt count.
                request.prompt = detail::uint_value(*f, "requests[].prompt", 0,
                                                    std::numeric_limits<std::uint32_t>::max());
                if (request.prompt >= plan.prompts.size()) detail::fail("requests[].prompt", "out of range");
            }
            if (const Json* f = item.find("max_tokens"))
                request.max_tokens = detail::uint_value(*f, "requests[].max_tokens", 1, plan.steps);
            if (const Json* f = item.find("seed")) {
                if (!f->is_number()) detail::fail("requests[].seed", "expected a u64 integer");
                const std::string& source = f->number_source();
                const auto parsed = std::from_chars(source.data(), source.data() + source.size(), request.seed);
                if (parsed.ec != std::errc{} || parsed.ptr != source.data() + source.size())
                    detail::fail("requests[].seed", "expected a u64 integer");
            }
            plan.requests->push_back(request);
        }
    }
    if (const Json* v = root.find("capture")) plan.capture = detail::capture_value(*v, "capture");
    if (const Json* v = root.find("logits_last_n"))
        plan.logits_last_n = detail::uint_value(*v, "logits_last_n", 0, detail::kMaxSteps);

    if (const Json* v = root.find("early_stop")) {
        detail::object_at(*v, "early_stop");
        detail::check_keys(*v, "early_stop", {"enabled", "metric", "min_pairs", "band_pct", "z"});
        EarlyStop& es = plan.early_stop;
        if (const Json* f = v->find("enabled")) es.enabled = detail::bool_value(*f, "early_stop.enabled");
        if (const Json* f = v->find("metric")) {
            es.metric = detail::string_value(*f, "early_stop.metric");
            if (!known_metric(es.metric)) detail::fail("early_stop.metric", "unknown metric \"" + es.metric + "\"");
        }
        if (const Json* f = v->find("min_pairs"))
            es.min_pairs = detail::uint_value(*f, "early_stop.min_pairs", 2, detail::kMaxRepeats);
        if (const Json* f = v->find("band_pct")) es.band_pct = detail::number_value(*f, "early_stop.band_pct", 0.0, 100.0);
        if (const Json* f = v->find("z")) {
            // 0 (what the canonical form emits when z was omitted) selects the t table.
            es.z = detail::number_value(*f, "early_stop.z", 0.0, 100.0);
        }
    }
    if (const Json* v = root.find("cold_control")) plan.cold_control = detail::bool_value(*v, "cold_control");

    const Json* arms = root.find("arms");
    if (arms == nullptr || !arms->is_array() || arms->size() == 0) detail::fail("arms", "expected a non-empty array");
    for (const Json& item : arms->as_array()) {
        detail::object_at(item, "arms[]");
        detail::check_keys(item, "arms[]", {"name", "env", "steps", "prompts", "capture"});
        const Json* name = item.find("name");
        if (name == nullptr) detail::fail("arms[]", "missing \"name\"");
        Arm arm;
        arm.name = detail::safe_name(*name, "arms[].name");
        for (const Arm& other : plan.arms)
            if (other.name == arm.name) detail::fail("arms", "duplicate arm \"" + arm.name + "\"");
        const std::string where = "arm " + arm.name;
        if (const Json* v = item.find("env")) arm.env = detail::env_value(*v, where + ".env");
        arm.steps = plan.steps;
        if (const Json* v = item.find("steps")) arm.steps = detail::uint_value(*v, where + ".steps", 1, detail::kMaxSteps);
        arm.capture = plan.capture;
        if (const Json* v = item.find("capture")) arm.capture = detail::capture_value(*v, where + ".capture");
        if (const Json* v = item.find("prompts")) {
            if (!v->is_array() || v->size() == 0) detail::fail(where + ".prompts", "expected a non-empty array");
            std::vector<bool> used(plan.prompts.size(), false);
            for (const Json& p : v->as_array()) {
                const std::string& pname = detail::string_value(p, where + ".prompts");
                std::size_t idx = plan.prompts.size();
                for (std::size_t i = 0; i < plan.prompts.size(); ++i)
                    if (plan.prompts[i].name == pname) idx = i;
                if (idx == plan.prompts.size()) detail::fail(where + ".prompts", "unknown prompt \"" + pname + "\"");
                if (used[idx]) detail::fail(where + ".prompts", "duplicate prompt \"" + pname + "\"");
                used[idx] = true;
            }
            for (std::size_t i = 0; i < used.size(); ++i)
                if (used[i]) arm.prompts.push_back(i);
        } else {
            for (std::size_t i = 0; i < plan.prompts.size(); ++i) arm.prompts.push_back(i);
        }
        if ((arm.capture & kCaptureLogits) != 0 && plan.logits_last_n == 0)
            detail::fail(where + ".capture", "\"logits\" needs logits_last_n > 0");
        plan.arms.push_back(std::move(arm));
    }
    if (plan.logits_last_n > 0) {
        bool any = false;
        for (const Arm& arm : plan.arms) any = any || (arm.capture & kCaptureLogits) != 0;
        if (!any) detail::fail("logits_last_n", "set but no arm captures \"logits\"");
    }

    plan.canonical = detail::normalized_json(plan).dump();
    plan.hash = fnv1a64(plan.canonical);
    return plan;
}

// FNV-1a 32 over the arm index and its canonical env overrides; agreed across ranks at every arm switch.
[[nodiscard]] inline std::uint32_t arm_digest(const Plan& plan, std::size_t arm) {
    std::string bytes = std::to_string(arm);
    bytes.push_back('\x1e');
    for (const auto& [k, v] : plan.arms.at(arm).env) {
        bytes += k;
        bytes.push_back('\x1f');
        bytes += v;
        bytes.push_back('\x1e');
    }
    return fnv1a32(bytes);
}

// ---------------------------------------------------------------------------------------------------
// Lever classes and reload grouping

enum class LeverClass : std::uint8_t {
    runtime,    // read at dispatch time (getenv per call or a global setter)
    construct,  // resolved in a constructor; switchable through the engine's apply_levers()
    cached,     // static cache with a setter
    reload,     // allocation, weight layout or a startup probe: needs a model reload
};

struct LeverEntry {
    std::string_view name;
    LeverClass cls;
};
using LeverTable = std::vector<LeverEntry>;

// Undeclared levers are treated as reload levers: switching them in place is never assumed safe.
[[nodiscard]] inline LeverClass lever_class(const LeverTable& table, std::string_view name) noexcept {
    for (const LeverEntry& e : table)
        if (e.name == name) return e.cls;
    return LeverClass::reload;
}

// The overrides of an arm that need a reload (class reload or undeclared), sorted by name.
[[nodiscard]] inline EnvSet reload_key(const Arm& arm, const LeverTable& table) {
    EnvSet key;
    for (const auto& kv : arm.env)
        if (lever_class(table, kv.first) == LeverClass::reload) key.push_back(kv);
    return key;
}

struct ArmGroup {
    EnvSet reload_key;
    std::vector<std::size_t> arms;  // indices into Plan::arms, plan order
};

// Groups arms by reload key in first-appearance order; one load serves a whole group.
[[nodiscard]] inline std::vector<ArmGroup> group_arms(const Plan& plan, const LeverTable& table) {
    std::vector<ArmGroup> groups;
    for (std::size_t i = 0; i < plan.arms.size(); ++i) {
        EnvSet key = reload_key(plan.arms[i], table);
        auto it = std::find_if(groups.begin(), groups.end(), [&](const ArmGroup& g) { return g.reload_key == key; });
        if (it == groups.end()) {
            groups.push_back(ArmGroup{std::move(key), {i}});
        } else {
            it->arms.push_back(i);
        }
    }
    return groups;
}

[[nodiscard]] inline std::size_t group_of(const std::vector<ArmGroup>& groups, std::size_t arm) noexcept {
    for (std::size_t g = 0; g < groups.size(); ++g)
        if (std::find(groups[g].arms.begin(), groups[g].arms.end(), arm) != groups[g].arms.end()) return g;
    return groups.size();
}

// ---------------------------------------------------------------------------------------------------
// Run order

// Arm order of one pass of an abab/baba group. Two arms: fixed (A B or B A) every pass. More arms: the
// list (reversed for baba) rotated by the pass index, so every arm visits every position.
[[nodiscard]] inline std::vector<std::size_t> pass_order(const std::vector<std::size_t>& arms, Order order,
                                                         std::uint32_t pass) {
    std::vector<std::size_t> out = arms;
    if (order == Order::baba) std::reverse(out.begin(), out.end());
    if (out.size() > 2) {
        const std::size_t shift = static_cast<std::size_t>(pass) % out.size();
        std::rotate(out.begin(), out.begin() + static_cast<std::ptrdiff_t>(shift), out.end());
    }
    return out;
}

struct RunSlot {
    std::uint64_t seq = 0;  // position in the whole schedule (receipt order)
    std::size_t group = 0;
    std::size_t arm = 0;
    std::uint32_t pass = 0;
    std::size_t position = 0;  // index of the arm within its pass
    std::size_t prompt = 0;
    bool warmup = false;
    bool cold = false;  // cold-control run after all groups (fresh reload of arm 0)
};

// The whole schedule: groups in order; within a group, every arm occurrence runs all its prompts back to
// back, preceded by plan.warmup unmeasured runs (on its first prompt) whenever the arm differs from the
// previous occurrence or the group has just been loaded. With early stop, the runner skips the remaining
// passes of a group once decide_early_stop() says so; the schedule itself is static.
[[nodiscard]] inline std::vector<RunSlot> schedule(const Plan& plan, const std::vector<ArmGroup>& groups) {
    // Conservative bound includes warmups at every occurrence, even for sequential order.
    // Check before allocating: neither a time budget nor early stop bounds host memory.
    constexpr std::size_t max_slots = 65536;
    std::size_t bound = 0;
    const auto add = [&](const Arm& a, std::size_t repeats) {
        if (a.prompts.size() > max_slots || plan.warmup > max_slots - a.prompts.size())
            throw PlanError("resident plan: schedule exceeds 65536 slots");
        const auto per = a.prompts.size() + plan.warmup;
        if (per != 0 && repeats > (max_slots - bound) / per)
            throw PlanError("resident plan: schedule exceeds 65536 slots");
        bound += per * repeats;
    };
    for (const auto& g : groups)
        for (const auto a : g.arms) add(plan.arms.at(a), plan.repeats);
    if (plan.cold_control && !plan.arms.empty()) add(plan.arms.front(), 1);
    std::vector<RunSlot> out;
    out.reserve(bound);
    std::uint64_t seq = 0;
    const auto emit_occurrence = [&](std::size_t g, std::size_t arm, std::uint32_t pass, std::size_t position,
                                     bool switched, bool cold) {
        const Arm& a = plan.arms[arm];
        if (switched) {
            for (std::uint32_t w = 0; w < plan.warmup; ++w)
                out.push_back(RunSlot{seq++, g, arm, pass, position, a.prompts.front(), true, cold});
        }
        for (const std::size_t p : a.prompts) out.push_back(RunSlot{seq++, g, arm, pass, position, p, false, cold});
    };
    for (std::size_t g = 0; g < groups.size(); ++g) {
        const std::vector<std::size_t>& arms = groups[g].arms;
        std::size_t prev = plan.arms.size();
        if (plan.order == Order::sequential) {
            for (std::size_t i = 0; i < arms.size(); ++i) {
                for (std::uint32_t pass = 0; pass < plan.repeats; ++pass) {
                    emit_occurrence(g, arms[i], pass, i, arms[i] != prev, false);
                    prev = arms[i];
                }
            }
        } else {
            for (std::uint32_t pass = 0; pass < plan.repeats; ++pass) {
                const std::vector<std::size_t> order = pass_order(arms, plan.order, pass);
                for (std::size_t i = 0; i < order.size(); ++i) {
                    emit_occurrence(g, order[i], pass, i, order[i] != prev, false);
                    prev = order[i];
                }
            }
        }
    }
    if (plan.cold_control && !plan.arms.empty()) emit_occurrence(group_of(groups, 0), 0, 0, 0, true, true);
    return out;
}

// ---------------------------------------------------------------------------------------------------
// Results and early stop

inline constexpr double kNoValue = std::numeric_limits<double>::quiet_NaN();

struct RunResult {
    double ttft_ms = kNoValue;
    double prefill_ms = kNoValue;
    double decode_ms = kNoValue;
    double prefill_tok_s = kNoValue;
    double decode_tok_s = kNoValue;
    std::vector<std::int32_t> tokens;  // generated ids (digest always; ids in the receipt with "tokens")
    std::string state_digest;          // hex, with "state-digest"
    std::string logits_path;           // with "logits"
    std::string logits_fnv;            // fnv1a64 of the prefill logits row (this rank's slice), when the engine reports it
    std::string logits_sha256;         // sha256 of the same raw f32 row
    std::string error;                 // non-empty: the run failed and is excluded from statistics
    bool cold_run = false;             // first run after the load or an arm switch (warm-ups included)
    Json prompt_inputs = Json::array(); // ordered effective input counts and content digests
};

[[nodiscard]] inline double metric_value(const RunResult& r, std::string_view metric) noexcept {
    if (metric == "ttft_ms") return r.ttft_ms;
    if (metric == "prefill_ms") return r.prefill_ms;
    if (metric == "decode_ms") return r.decode_ms;
    if (metric == "prefill_tok_s") return r.prefill_tok_s;
    if (metric == "decode_tok_s") return r.decode_tok_s;
    return kNoValue;
}

struct RunRecord {
    RunSlot slot;
    RunResult result;
};

// Two-sided 95% critical value of Student's t; conservative (next lower table df) between table rows.
[[nodiscard]] inline double t95(std::size_t df) noexcept {
    static constexpr double kTable[] = {12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
                                        2.201,  2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
                                        2.080,  2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042};
    if (df == 0) return std::numeric_limits<double>::infinity();
    if (df <= 30) return kTable[df - 1];
    if (df < 40) return 2.042;
    if (df < 60) return 2.021;
    if (df < 120) return 2.000;
    if (df < 1000) return 1.980;
    return 1.960;
}

struct Comparison {
    std::size_t baseline = 0;  // arm indices
    std::size_t arm = 0;
    std::string metric;
    bool cross_reload = false;  // the arms are in different reload groups (load-to-load variance included)
    std::size_t n_pairs = 0;
    double mean_rel = kNoValue;  // mean of (arm - baseline) / baseline over matched (pass, prompt) pairs
    double sd_rel = kNoValue;
    double half_rel = kNoValue;  // critical value * sd / sqrt(n)
    std::string verdict;         // insufficient | separated | equivalent | open
    bool cold_run = false;
};

// Pairs `arm` with `baseline` on (pass, prompt); warmups, cold-control runs, errors and non-positive or
// missing baseline values are excluded.
[[nodiscard]] inline Comparison compare_pair(const Plan& plan, const std::vector<ArmGroup>& groups,
                                             std::size_t baseline, std::size_t arm,
                                             const std::vector<RunRecord>& records, bool cold_run = false) {
    Comparison c;
    c.baseline = baseline;
    c.arm = arm;
    c.metric = plan.early_stop.metric;
    c.cold_run = cold_run;
    c.cross_reload = group_of(groups, baseline) != group_of(groups, arm);
    std::vector<double> d;
    for (const RunRecord& a : records) {
        if (a.slot.arm != baseline || a.slot.warmup || a.slot.cold || !a.result.error.empty() ||
            a.result.cold_run != cold_run) continue;
        const double av = metric_value(a.result, c.metric);
        if (!std::isfinite(av) || av <= 0.0) continue;
        for (const RunRecord& b : records) {
            if (b.slot.arm != arm || b.slot.warmup || b.slot.cold || !b.result.error.empty() ||
                b.result.cold_run != cold_run) continue;
            if (b.slot.pass != a.slot.pass || b.slot.prompt != a.slot.prompt) continue;
            const double bv = metric_value(b.result, c.metric);
            if (std::isfinite(bv)) d.push_back((bv - av) / av);
            break;
        }
    }
    c.n_pairs = d.size();
    if (d.empty()) {
        c.verdict = "insufficient";
        return c;
    }
    double sum = 0.0;
    for (const double x : d) sum += x;
    const double n = static_cast<double>(d.size());
    c.mean_rel = sum / n;
    if (d.size() >= 2) {
        double ss = 0.0;
        for (const double x : d) ss += (x - c.mean_rel) * (x - c.mean_rel);
        c.sd_rel = std::sqrt(ss / (n - 1.0));
        const double crit = plan.early_stop.z > 0.0 ? plan.early_stop.z : t95(d.size() - 1);
        c.half_rel = crit * c.sd_rel / std::sqrt(n);
    }
    if (d.size() < plan.early_stop.min_pairs) {
        c.verdict = "insufficient";
    } else if (std::fabs(c.mean_rel) > c.half_rel) {
        c.verdict = "separated";  // the CI excludes 0
    } else if (std::fabs(c.mean_rel) + c.half_rel <= plan.early_stop.band_pct / 100.0) {
        c.verdict = "equivalent";  // the CI lies inside +-band_pct
    } else {
        c.verdict = "open";
    }
    return c;
}

// Plan arm 0 against every other arm.
[[nodiscard]] inline std::vector<Comparison> compare_all(const Plan& plan, const std::vector<ArmGroup>& groups,
                                                         const std::vector<RunRecord>& records, bool cold_run = false) {
    std::vector<Comparison> out;
    for (std::size_t i = 1; i < plan.arms.size(); ++i) out.push_back(compare_pair(plan, groups, 0, i, records, cold_run));
    return out;
}

struct StopDecision {
    bool stop = false;
    std::string reason;  // "" | separated | equivalent | mixed (some separated, some equivalent)
};

// Stop when early stop is enabled and every comparison is decided (separated or equivalent).
[[nodiscard]] inline StopDecision decide_early_stop(const Plan& plan, const std::vector<Comparison>& comparisons) {
    StopDecision out;
    if (!plan.early_stop.enabled || comparisons.empty()) return out;
    bool separated = false, equivalent = false;
    for (const Comparison& c : comparisons) {
        if (c.cold_run) return out;  // only warm evidence can stop a plan
        if (c.verdict == "separated") separated = true;
        else if (c.verdict == "equivalent") equivalent = true;
        else return out;
    }
    out.stop = true;
    out.reason = separated && equivalent ? "mixed" : (separated ? "separated" : "equivalent");
    return out;
}

// ---------------------------------------------------------------------------------------------------
// Receipts and summary

namespace detail {
inline Json number_or_null(double v) { return std::isfinite(v) ? Json(v) : Json(nullptr); }
}  // namespace detail

// <out>/run-<seq, 5+ digits>-<arm>-<prompt>.json
[[nodiscard]] inline std::string receipt_name(const Plan& plan, const RunSlot& slot) {
    char seq[24];
    std::snprintf(seq, sizeof(seq), "%05llu", static_cast<unsigned long long>(slot.seq));
    return "run-" + std::string(seq) + "-" + plan.arms.at(slot.arm).name + "-" + plan.prompts.at(slot.prompt).name +
           ".json";
}

[[nodiscard]] inline Json receipt_json(const Plan& plan, const RunRecord& record) {
    const RunSlot& s = record.slot;
    const RunResult& r = record.result;
    const Arm& arm = plan.arms.at(s.arm);
    Json out = Json::object();
    out.set("plan_hash", Json(hex64(plan.hash)));
    out.set("engine", Json(plan.engine));
    out.set("seq", Json(static_cast<std::uint64_t>(s.seq)));
    out.set("arm", Json(arm.name));
    out.set("arm_digest", Json(hex32(arm_digest(plan, s.arm))));
    out.set("group", Json(static_cast<std::uint64_t>(s.group)));
    out.set("pass", Json(static_cast<std::uint64_t>(s.pass)));
    out.set("position", Json(static_cast<std::uint64_t>(s.position)));
    out.set("prompt", Json(plan.prompts.at(s.prompt).name));
    out.set("warmup", Json(s.warmup));
    out.set("cold_control", Json(s.cold));
    out.set("cold_run", Json(r.cold_run));
    out.set("steps", Json(static_cast<std::uint64_t>(arm.steps)));
    out.set("prompt_inputs", r.prompt_inputs);
    for (const std::string_view m : kMetricNames) out.set(std::string(m), detail::number_or_null(metric_value(r, m)));
    out.set("n_tokens", Json(static_cast<std::uint64_t>(r.tokens.size())));
    out.set("tokens_fnv", Json(hex64(token_digest(r.tokens))));
    if ((arm.capture & kCaptureTokens) != 0) {
        Json ids = Json::array();
        for (const std::int32_t t : r.tokens) ids.push(Json(static_cast<std::int64_t>(t)));
        out.set("tokens", std::move(ids));
    }
    if ((arm.capture & kCaptureLogits) != 0 && !r.logits_path.empty()) out.set("logits_path", Json(r.logits_path));
    if (!r.logits_fnv.empty()) out.set("logits_fnv", Json(r.logits_fnv));
    if (!r.logits_sha256.empty()) out.set("logits_sha256", Json(r.logits_sha256));
    if ((arm.capture & kCaptureStateDigest) != 0 && !r.state_digest.empty())
        out.set("state_digest", Json(r.state_digest));
    if (!r.error.empty()) out.set("error", Json(r.error));
    return out;
}

namespace detail {
struct Stats {
    std::size_t n = 0;
    double mean = kNoValue;
    double sd = kNoValue;
};
inline Stats stats(const std::vector<double>& v) {
    Stats s;
    s.n = v.size();
    if (v.empty()) return s;
    double sum = 0.0;
    for (const double x : v) sum += x;
    s.mean = sum / static_cast<double>(v.size());
    if (v.size() >= 2) {
        double ss = 0.0;
        for (const double x : v) ss += (x - s.mean) * (x - s.mean);
        s.sd = std::sqrt(ss / static_cast<double>(v.size() - 1));
    }
    return s;
}
inline std::vector<double> resident_values(const std::vector<RunRecord>& records, std::size_t arm,
                                           std::string_view metric, bool cold, bool cold_run = false) {
    std::vector<double> v;
    for (const RunRecord& r : records) {
        if (r.slot.arm != arm || r.slot.warmup || r.slot.cold != cold || !r.result.error.empty()) continue;
        if (!cold && r.result.cold_run != cold_run) continue;
        const double x = metric_value(r.result, metric);
        if (std::isfinite(x)) v.push_back(x);
    }
    return v;
}
}  // namespace detail

// summary.json: per-arm means, the paired comparisons against arm 0, the early-stop outcome and the cold
// control (resident_drift when the cold run's mean lies outside the resident runs' 95% CI of arm 0).
[[nodiscard]] inline Json summary_json(const Plan& plan, const std::vector<ArmGroup>& groups,
                                       const std::vector<RunRecord>& records,
                                       const std::vector<Comparison>& comparisons, const StopDecision& stop) {
    Json out = Json::object();
    out.set("plan_hash", Json(hex64(plan.hash)));
    out.set("engine", Json(plan.engine));
    out.set("order", Json(std::string(order_name(plan.order))));
    out.set("metric", Json(plan.early_stop.metric));
    out.set("early_stop_population", Json("warm"));
    out.set("groups", Json(static_cast<std::uint64_t>(groups.size())));
    std::size_t errors = 0;
    for (const RunRecord& r : records) errors += r.result.error.empty() ? 0U : 1U;
    out.set("runs", Json(static_cast<std::uint64_t>(records.size())));
    out.set("errors", Json(static_cast<std::uint64_t>(errors)));

    Json arms = Json::array();
    for (std::size_t i = 0; i < plan.arms.size(); ++i) {
        Json a = Json::object();
        a.set("name", Json(plan.arms[i].name));
        a.set("arm_digest", Json(hex32(arm_digest(plan, i))));
        a.set("group", Json(static_cast<std::uint64_t>(group_of(groups, i))));
        Json means = Json::object();
        std::size_t n = 0;
        for (const std::string_view m : kMetricNames) {
            const detail::Stats s = detail::stats(detail::resident_values(records, i, m, false));
            n = std::max(n, s.n);
            means.set(std::string(m), detail::number_or_null(s.mean));
        }
        a.set("n_runs", Json(static_cast<std::uint64_t>(n)));
        a.set("mean", std::move(means));
        Json cold_means = Json::object();
        std::size_t cold_n = 0;
        for (const std::string_view m : kMetricNames) {
            const auto s = detail::stats(detail::resident_values(records, i, m, false, true));
            cold_n = std::max(cold_n, s.n);
            cold_means.set(std::string(m), detail::number_or_null(s.mean));
        }
        a.set("population", Json("warm"));
        a.set("cold_n_runs", Json(static_cast<std::uint64_t>(cold_n)));
        a.set("cold_mean", std::move(cold_means));
        arms.push(std::move(a));
    }
    out.set("arms", std::move(arms));

    const auto comparison_json = [&](const std::vector<Comparison>& values) {
      Json cmp = Json::array();
      for (const Comparison& c : values) {
        Json item = Json::object();
        item.set("baseline", Json(plan.arms.at(c.baseline).name));
        item.set("arm", Json(plan.arms.at(c.arm).name));
        item.set("metric", Json(c.metric));
        item.set("population", Json(c.cold_run ? "cold" : "warm"));
        item.set("cross_reload", Json(c.cross_reload));
        item.set("n_pairs", Json(static_cast<std::uint64_t>(c.n_pairs)));
        item.set("mean_rel", detail::number_or_null(c.mean_rel));
        item.set("half_rel", detail::number_or_null(c.half_rel));
        item.set("ci_lo", detail::number_or_null(c.mean_rel - c.half_rel));
        item.set("ci_hi", detail::number_or_null(c.mean_rel + c.half_rel));
        item.set("verdict", Json(c.verdict));
        cmp.push(std::move(item));
      }
      return cmp;
    };
    out.set("comparisons", comparison_json(comparisons));
    out.set("cold_comparisons", comparison_json(compare_all(plan, groups, records, true)));
    out.set("stopped_early", Json(stop.stop));
    out.set("stop_reason", Json(stop.reason));

    if (plan.cold_control && !plan.arms.empty()) {
        const std::string& m = plan.early_stop.metric;
        const detail::Stats resident = detail::stats(detail::resident_values(records, 0, m, false));
        const detail::Stats cold = detail::stats(detail::resident_values(records, 0, m, true));
        Json cc = Json::object();
        cc.set("arm", Json(plan.arms[0].name));
        cc.set("cold_mean", detail::number_or_null(cold.mean));
        cc.set("resident_mean", detail::number_or_null(resident.mean));
        double delta_rel = kNoValue, half_rel = kNoValue;
        if (std::isfinite(cold.mean) && std::isfinite(resident.mean) && resident.mean != 0.0) {
            delta_rel = (cold.mean - resident.mean) / resident.mean;
            if (std::isfinite(resident.sd))
                half_rel = t95(resident.n - 1) * resident.sd / std::sqrt(static_cast<double>(resident.n)) /
                           std::fabs(resident.mean);
        }
        cc.set("delta_rel", detail::number_or_null(delta_rel));
        cc.set("resident_half_rel", detail::number_or_null(half_rel));
        // Undecidable (no cold value or fewer than two resident runs) is not reported as drift.
        cc.set("resident_drift", Json(std::isfinite(delta_rel) && std::isfinite(half_rel) &&
                                      std::fabs(delta_rel) > half_rel));
        out.set("cold_control", std::move(cc));
    }
    return out;
}

}  // namespace strix::serve::resident
