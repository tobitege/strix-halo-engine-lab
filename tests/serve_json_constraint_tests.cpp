#include "strix/serve/json_constraint.hpp"
#include "strix/env_flags.hpp"

#include <memory>
#include <thread>
#include <atomic>
#include <algorithm>
#include <array>
#include <chrono>
#include <functional>
#include <iostream>
#include <random>
#include <regex>
#include <map>
#include <set>
#include <cstdio>
#include <condition_variable>
#include <cstdlib>
#include <mutex>
#include <new>
#ifdef __linux__
#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

// Sweep every allocation before compilation, including list insertion and index
// node/bucket insertion. Fail only once so exception cleanup can allocate.
namespace allocation_failure {
thread_local std::ptrdiff_t remaining = -1;

// Keep free out of new-expression call sites: GCC otherwise diagnoses the
// inlined replacement delete as mismatched, despite the malloc-backed new.
#if defined(_MSC_VER)
__declspec(noinline)
#elif defined(__GNUC__)
__attribute__((noinline))
#endif
void deallocate(void* p) noexcept { std::free(p); }
}
void* operator new(std::size_t bytes) {
    if (allocation_failure::remaining >= 0 && allocation_failure::remaining-- == 0) {
        allocation_failure::remaining = -1;
        throw std::bad_alloc();
    }
    if (void* p = std::malloc(bytes ? bytes : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t bytes) { return ::operator new(bytes); }
void operator delete(void* p) noexcept { allocation_failure::deallocate(p); }
void operator delete[](void* p) noexcept { allocation_failure::deallocate(p); }
void operator delete(void* p, std::size_t) noexcept { allocation_failure::deallocate(p); }
void operator delete[](void* p, std::size_t) noexcept { allocation_failure::deallocate(p); }

namespace strix::serve {
struct ConstraintProgramCacheTestAccess {
    static ConstraintProgramCache make(ConstraintProgramCacheConfig config, void (*hook)(void*), void* context) {
        return ConstraintProgramCache(config, hook, context);
    }
};
}

// Historical differential regression only. The semantic oracles below are
// the independent raw-byte validator and bounded PRODUCTIVITY search.
namespace reference {
using namespace strix::serve;
namespace {
struct Node {
    bool yes = true;
    std::set<std::string> types;
    std::vector<std::pair<std::string, std::size_t>> properties;
    std::set<std::string> required;
    bool additional = true;
    std::size_t items = 0;
    std::size_t min_items = 0, max_items = SIZE_MAX, min_length = 0, max_length = SIZE_MAX;
    std::vector<std::string> literals;
    bool literal_set = false;
    std::vector<std::size_t> alternatives;
    std::size_t ref = SIZE_MAX;
};
std::string join(const std::string& p, const std::string& k) { return p.empty() ? k : p + "." + k; }
[[noreturn]] void error(const std::string& p, const std::string& k, const char* message) {
    throw SchemaError(join(p, k), k, message);
}
struct Value {
    char kind = 0;
    bool done = false;
    std::size_t begin = 0, end = 0, scalars = 0;
    std::string text;
    std::string pending;
    bool expect_more = false;
    std::vector<Value> children;
    std::vector<std::string> keys;
    bool key_pending = false;
    std::string key_prefix;
    std::string key_pending_bytes;
};
struct Invalid {};
class PrefixParser {
public:
    PrefixParser(std::string_view s, const ConstraintLimits& limits) : s_(s), limits_(limits) {}
    Value root() {
        auto v = value(0);
        if (v.done) { gap(); if (at_ != s_.size()) throw Invalid{}; }
        return v;
    }
private:
    bool end() const { return at_ == s_.size(); }
    void gap() {
        unsigned count = 0;
        while (!end() && (s_[at_] == ' ' || s_[at_] == '\n' || s_[at_] == '\t')) {
            ++at_; if (++count > 2) throw Invalid{};
        }
    }
    unsigned hex(char c) {
        if (c >= '0' && c <= '9') return static_cast<unsigned>(c - '0');
        if (c >= 'a' && c <= 'f') return static_cast<unsigned>(c - 'a' + 10);
        if (c >= 'A' && c <= 'F') return static_cast<unsigned>(c - 'A' + 10);
        throw Invalid{};
    }
    void utf8(std::string& out, unsigned cp) {
        if (cp < 128) out += static_cast<char>(cp);
        else if (cp < 2048) { out += static_cast<char>(0xc0U | (cp >> 6U)); out += static_cast<char>(0x80U | (cp & 63U)); }
        else if (cp < 65536) { out += static_cast<char>(0xe0U | (cp >> 12U)); out += static_cast<char>(0x80U | ((cp >> 6U) & 63U)); out += static_cast<char>(0x80U | (cp & 63U)); }
        else { out += static_cast<char>(0xf0U | (cp >> 18U)); out += static_cast<char>(0x80U | ((cp >> 12U) & 63U)); out += static_cast<char>(0x80U | ((cp >> 6U) & 63U)); out += static_cast<char>(0x80U | (cp & 63U)); }
    }
    Value string() {
        Value v; v.kind = 's'; v.begin = at_++;
        while (!end()) {
            const auto scalar_begin = at_;
            const auto incomplete = [&] { v.pending = std::string(s_.substr(scalar_begin)); return v; };
            unsigned cp = static_cast<unsigned char>(s_[at_++]);
            if (cp == '"') { v.done = true; v.end = at_; return v; }
            if (cp < 32) throw Invalid{};
            if (++v.scalars > limits_.max_string_length) throw Invalid{};
            if (cp == '\\') {
                if (end()) return incomplete();
                const char e = s_[at_++];
                if (e == 'u') {
                    cp = 0;
                    for (unsigned i = 0; i < 4; ++i) { if (end()) return incomplete(); cp = (cp << 4U) | hex(s_[at_++]); if (i == 1 && cp >= 0xdc && cp <= 0xdf) throw Invalid{}; }
                    if (cp >= 0xdc00 && cp <= 0xdfff) throw Invalid{};
                    if (cp >= 0xd800 && cp <= 0xdbff) {
                        if (end()) return incomplete();
                        if (s_[at_++] != '\\') throw Invalid{};
                        if (end()) return incomplete();
                        if (s_[at_++] != 'u') throw Invalid{};
                        unsigned low = 0;
                        for (unsigned i = 0; i < 4; ++i) { if (end()) return incomplete(); low = (low << 4U) | hex(s_[at_++]); if ((i == 0 && low != 0xd) || (i == 1 && (low < 0xdc || low > 0xdf))) throw Invalid{}; }
                        if (low < 0xdc00 || low > 0xdfff) throw Invalid{};
                        cp = 0x10000U + ((cp - 0xd800U) << 10U) + low - 0xdc00U;
                    }
                } else {
                    const std::string escapes = "\"\\/bfnrt";
                    const auto pos = escapes.find(e);
                    if (pos == std::string::npos) throw Invalid{};
                    cp = static_cast<unsigned char>(std::string("\"\\/\b\f\n\r\t")[pos]);
                }
                utf8(v.text, cp);
            } else if (cp >= 128) {
                const unsigned lead = cp;
                unsigned count = 0, minimum = 0;
                if (cp >= 0xc2 && cp <= 0xdf) { count = 1; minimum = 128; cp &= 31U; }
                else if (cp >= 0xe0 && cp <= 0xef) { count = 2; minimum = 2048; cp &= 15U; }
                else if (cp >= 0xf0 && cp <= 0xf4) { count = 3; minimum = 65536; cp &= 7U; }
                else throw Invalid{};
                for (unsigned i = 0; i < count; ++i) {
                    if (end()) return incomplete();
                    const auto b = static_cast<unsigned char>(s_[at_++]);
                    if (b < 0x80 || b > 0xbf || (i == 0 && ((lead == 0xe0 && b < 0xa0) || (lead == 0xed && b > 0x9f) || (lead == 0xf0 && b < 0x90) || (lead == 0xf4 && b > 0x8f)))) throw Invalid{};
                    cp = (cp << 6U) | (b & 63U);
                }
                if (cp < minimum || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) throw Invalid{};
                utf8(v.text, cp);
            } else v.text += static_cast<char>(cp);
        }
        return v;
    }
    Value value(std::size_t depth) {
        if (depth > limits_.depth) throw Invalid{};
        gap(); Value v; v.begin = at_;
        if (end()) return v;
        const char c = s_[at_];
        if (c == '"') return string();
        if (c == '{' || c == '[') {
            v.kind = c; ++at_; gap(); if (end()) return v;
            const char close = c == '{' ? '}' : ']';
            if (s_[at_] == close) { ++at_; v.done = true; v.end = at_; return v; }
            for (;;) {
                if (c == '{') {
                    gap(); if (end()) { v.key_pending = true; return v; }
                    if (s_[at_] != '"') throw Invalid{};
                    auto key = string();
                    if (!key.done) { v.key_pending = true; v.key_prefix = key.text; v.key_pending_bytes = key.pending; return v; }
                    if (std::find(v.keys.begin(), v.keys.end(), key.text) != v.keys.end()) throw Invalid{};
                    v.keys.push_back(key.text);
                    gap(); if (end()) return v;
                    if (s_[at_++] != ':') throw Invalid{};
                }
                auto child = value(depth + 1); v.expect_more = false; v.children.push_back(std::move(child));
                if (!v.children.back().done) return v;
                gap(); if (end()) return v;
                if (s_[at_] == close) { ++at_; v.done = true; v.end = at_; return v; }
                if (s_[at_++] != ',') throw Invalid{};
                v.expect_more = true;
                gap(); if (end()) { if (c == '{') v.key_pending = true; return v; }
            }
        }
        if (c == 't' || c == 'f' || c == 'n') {
            v.kind = c == 'n' ? '0' : 'b';
            const std::string_view word = c == 't' ? "true" : c == 'f' ? "false" : "null";
            for (char b : word) { if (end()) return v; if (s_[at_++] != b) throw Invalid{}; }
            v.done = true; v.end = at_; return v;
        }
        v.kind = 'n';
        if (s_[at_] == '-') { ++at_; if (end()) return v; }
        if (s_[at_] == '0') ++at_;
        else {
            if (s_[at_] < '1' || s_[at_] > '9') throw Invalid{};
            while (!end() && s_[at_] >= '0' && s_[at_] <= '9') ++at_;
        }
        if (!end() && s_[at_] == '.') {
            ++at_; if (end()) return v;
            if (s_[at_] < '0' || s_[at_] > '9') throw Invalid{};
            while (!end() && s_[at_] >= '0' && s_[at_] <= '9') ++at_;
        }
        if (!end() && (s_[at_] == 'e' || s_[at_] == 'E')) {
            ++at_; if (end()) return v;
            if (s_[at_] == '+' || s_[at_] == '-') { ++at_; if (end()) return v; }
            if (s_[at_] < '0' || s_[at_] > '9') throw Invalid{};
            while (!end() && s_[at_] >= '0' && s_[at_] <= '9') ++at_;
        }
        v.text = std::string(s_.substr(v.begin, at_ - v.begin));
        v.done = true; v.end = at_; return v;
    }
    std::string_view s_;
    const ConstraintLimits& limits_;
    std::size_t at_ = 0;
};
}  // namespace
struct ReferenceData {
    ConstraintLimits limits;
    std::vector<Node> nodes{Node{}};
    std::size_t root = 0;
};
namespace {
class ReferenceCompiler {
public:
    explicit ReferenceCompiler(const ConstraintLimits& limits) { data->limits = limits; }
    std::shared_ptr<ReferenceData> data = std::make_shared<ReferenceData>();
    std::map<std::string, std::size_t> defs;
    std::map<std::string, const Json*> sources;
    std::set<std::string> active;
    std::size_t definition(const std::string& name, const std::string& path) {
        if (active.contains(name)) error(path, "$ref", "recursive reference");
        if (active.size() >= data->limits.depth) error(path, "$ref", "reference depth limit");
        if (defs.contains(name)) return defs.at(name);
        if (!sources.contains(name)) error(path, "$ref", "unknown local reference");
        active.insert(name);
        const auto id = node(*sources.at(name), join("$defs", name), 1);
        active.erase(name); defs[name] = id; return id;
    }
    std::size_t node(const Json& schema, const std::string& path, std::size_t depth) {
        if (depth > data->limits.depth || data->nodes.size() > data->limits.nodes) error(path, "limits", "schema resource limit");
        const auto id = data->nodes.size(); data->nodes.emplace_back(); Node n;
        if (schema.is_bool()) { n.yes = schema.as_bool(); data->nodes[id] = n; return id; }
        if (!schema.is_object()) error(path, "schema", "expected schema object or boolean");
        const auto size_value = [&](const Json& x, const std::string& k) {
            if (!x.is_number() || x.number_source().find_first_not_of("0123456789") != std::string::npos) error(path, k, "expected nonnegative integer");
            try { return static_cast<std::size_t>(std::stoull(x.number_source())); }
            catch (const std::exception&) { error(path, k, "integer out of range"); }
        };
        for (const auto& [k, x] : schema.as_object()) {
            if (k == "type") {
                const auto add = [&](const Json& t) {
                    if (!t.is_string()) error(path, k, "expected type name");
                    const auto& s = t.as_string();
                    if (s != "object" && s != "array" && s != "string" && s != "number" && s != "integer" && s != "boolean" && s != "null") error(path, k, "unknown type");
                    if (!n.types.insert(s).second) error(path, k, "duplicate type");
                };
                if (x.is_array()) { if (x.size() == 0) error(path, k, "empty types"); for (const auto& t : x.as_array()) add(t); }
                else add(x);
            } else if (k == "properties") {
                if (!x.is_object() || x.size() > data->limits.properties) error(path, k, "properties limit or shape");
                for (const auto& [name, sub] : x.as_object()) n.properties.emplace_back(name, node(sub, join(join(path, k), name), depth + 1));
            } else if (k == "required") {
                if (!x.is_array() || x.size() > data->limits.properties) error(path, k, "required limit or shape");
                for (const auto& item : x.as_array()) if (!item.is_string() || !n.required.insert(item.as_string()).second) error(path, k, "invalid required name");
            } else if (k == "additionalProperties") {
                if (!x.is_bool()) error(path, k, "expected boolean");
                n.additional = x.as_bool();
            } else if (k == "items") n.items = node(x, join(path, k), depth + 1);
            else if (k == "minItems") n.min_items = size_value(x, k);
            else if (k == "maxItems") n.max_items = size_value(x, k);
            else if (k == "minLength") n.min_length = size_value(x, k);
            else if (k == "maxLength") n.max_length = size_value(x, k);
            else if (k == "enum" || k == "const") {
                std::vector<std::string> values;
                if (k == "enum") {
                    if (!x.is_array() || x.size() == 0 || x.size() > data->limits.enum_values) error(path, k, "enum limit or shape");
                    for (const auto& item : x.as_array()) values.push_back(item.dump());
                } else values.push_back(x.dump());
                if (n.literal_set) {
                    std::vector<std::string> common;
                    for (const auto& a : n.literals) if (std::find(values.begin(), values.end(), a) != values.end()) common.push_back(a);
                    n.literals = std::move(common);
                } else n.literals = std::move(values);
                n.literal_set = true;
            } else if (k == "anyOf") {
                if (!x.is_array() || x.size() == 0) error(path, k, "expected nonempty alternatives");
                std::size_t i = 0;
                for (const auto& item : x.as_array()) n.alternatives.push_back(node(item, join(join(path, k), std::to_string(i++)), depth + 1));
            } else if (k == "$ref") {
                if (!x.is_string() || !x.as_string().starts_with("#/$defs/") || x.as_string().find('/', 8) != std::string::npos) error(path, k, "expected local definition reference");
                std::string name;
                const auto encoded = std::string_view(x.as_string()).substr(8);
                for (std::size_t i = 0; i < encoded.size(); ++i) {
                    if (encoded[i] != '~') name += encoded[i];
                    else {
                        if (++i == encoded.size() || (encoded[i] != '0' && encoded[i] != '1')) error(path, k, "invalid reference escape");
                        name += encoded[i] == '0' ? '~' : '/';
                    }
                }
                n.ref = definition(name, path);
            } else if (k == "$defs") {
                if (!x.is_object()) error(path, k, "expected definitions object");
                if (!path.empty()) for (const auto& [name, sub] : x.as_object()) node(sub, join(join(path, k), name), depth + 1);
            } else if (k == "title" || k == "description" || k == "$comment") {
                if (!x.is_string()) error(path, k, "expected annotation string");
            } else error(path, k, "unsupported schema keyword");
        }
        data->nodes[id] = std::move(n); return id;
    }
};
bool key_viable(const std::string& key, const Value& v) {
    if (!key.starts_with(v.key_prefix)) return false;
    if (v.key_pending_bytes.empty()) return true;
    const auto rest = std::string_view(key).substr(v.key_prefix.size());
    if (rest.empty()) return false;
    unsigned cp = static_cast<unsigned char>(rest[0]); std::size_t len = 1;
    if (cp >= 128) {
        len = cp < 0xe0 ? 2 : cp < 0xf0 ? 3 : 4;
        cp &= len == 2 ? 31U : len == 3 ? 15U : 7U;
        if (rest.size() < len) return false;
        for (std::size_t i = 1; i < len; ++i) cp = (cp << 6U) | (static_cast<unsigned char>(rest[i]) & 63U);
    }
    std::vector<std::string> forms;
    if (cp >= 32 && cp != '"' && cp != '\\') forms.emplace_back(rest.substr(0, len));
    std::string quoted; append_json_string(quoted, std::string(rest.substr(0, len)));
    forms.push_back(quoted.substr(1, quoted.size() - 2));
    if (cp == '/') forms.emplace_back("\\/");
    const auto unit = [](unsigned x) {
        std::string out = "\\u0000"; const char* digits = "0123456789abcdef";
        for (int i = 5; i >= 2; --i) { out[static_cast<std::size_t>(i)] = digits[x & 15U]; x >>= 4U; }
        return out;
    };
    if (cp < 65536) forms.push_back(unit(cp));
    else { cp -= 65536; forms.push_back(unit(0xd800U + (cp >> 10U)) + unit(0xdc00U + (cp & 1023U))); }
    for (const auto& form : forms) {
        if (form.size() < v.key_pending_bytes.size()) continue;
        bool same = true;
        for (std::size_t i = 0; i < v.key_pending_bytes.size(); ++i) {
            char b = v.key_pending_bytes[i];
            if (form.starts_with("\\u") && b >= 'A' && b <= 'F') b = static_cast<char>(b - 'A' + 'a');
            if (b != form[i]) { same = false; break; }
        }
        if (same) return true;
    }
    return false;
}
bool matches(const ReferenceData& d, std::size_t id, const Value& v, std::string_view bytes, bool final, std::size_t& steps) {
    if (++steps > d.limits.compilation_steps) throw ConstraintError("constraint evaluation resource limit");
    const auto& n = d.nodes[id];
    if (!n.yes) return false;
    if (n.ref != SIZE_MAX && !matches(d, n.ref, v, bytes, final, steps)) return false;
    if (!n.alternatives.empty() && std::none_of(n.alternatives.begin(), n.alternatives.end(), [&](auto a) { return matches(d, a, v, bytes, final, steps); })) return false;
    if (v.kind == 0) return true;
    const std::string type = v.kind == '{' ? "object" : v.kind == '[' ? "array" : v.kind == 's' ? "string" : v.kind == 'n' ? "number" : v.kind == 'b' ? "boolean" : "null";
    if (!n.types.empty() && !n.types.contains(type)) {
        if (v.kind != 'n' || !n.types.contains("integer")) return false;
        const auto raw = bytes.substr(v.begin, (v.done ? v.end : bytes.size()) - v.begin);
        if (raw.find_first_of(".eE") != std::string_view::npos) return false;
    }
    if (n.literal_set) {
        const auto raw = bytes.substr(v.begin, (v.done ? v.end : bytes.size()) - v.begin);
        if (std::none_of(n.literals.begin(), n.literals.end(), [&](const auto& s) { return v.done && (final || v.kind != 'n' || v.end != bytes.size()) ? s == raw : std::string_view(s).starts_with(raw); })) return false;
    }
    if (v.kind == 's' && (v.scalars > n.max_length || (v.done && v.scalars < n.min_length))) return false;
    if (v.kind == '[') {
        if ((v.expect_more && v.children.size() >= n.max_items) || v.children.size() > n.max_items || (v.done && v.children.size() < n.min_items)) return false;
        for (const auto& child : v.children) if (!matches(d, n.items, child, bytes, final, steps)) return false;
    }
    if (v.kind == '{') {
        for (std::size_t i = 0; i < v.keys.size(); ++i) {
            const auto p = std::find_if(n.properties.begin(), n.properties.end(), [&](const auto& x) { return x.first == v.keys[i]; });
            if (p == n.properties.end() && !n.additional) return false;
            if (i < v.children.size() && !matches(d, p == n.properties.end() ? 0 : p->second, v.children[i], bytes, final, steps)) return false;
        }
        if (v.key_pending && !n.additional && std::none_of(n.properties.begin(), n.properties.end(), [&](const auto& p) { return key_viable(p.first, v) && std::find(v.keys.begin(), v.keys.end(), p.first) == v.keys.end(); })) return false;
        if (v.done) for (const auto& key : n.required) if (std::find(v.keys.begin(), v.keys.end(), key) == v.keys.end()) return false;
    }
    return true;
}
// Expand unions only while evaluating a conjunction. Keeping sibling constraints
// together prevents admitting a prefix whose alternatives have no common value.
class Viability {
public:
    explicit Viability(const ReferenceData& data) : d(data) {}
    bool evaluate(std::vector<std::size_t> ids, const Value* value, std::string_view bytes, bool final = false, std::size_t depth = 0) {
        return expand({}, std::move(ids), value, bytes, final, depth);
    }
private:
    const ReferenceData& d;
    std::size_t steps = 0;
    void step() {
        if (++steps > d.limits.compilation_steps) throw ConstraintError("constraint evaluation resource limit");
    }
    bool expand(std::vector<std::size_t> own, std::vector<std::size_t> pending, const Value* v, std::string_view bytes, bool final, std::size_t depth) {
        step();
        while (!pending.empty()) {
            const auto id = pending.back(); pending.pop_back(); const auto& n = d.nodes[id];
            if (!n.yes) return false;
            own.push_back(id);
            if (n.ref != SIZE_MAX) pending.push_back(n.ref);
            if (!n.alternatives.empty()) {
                for (auto a : n.alternatives) {
                    auto branch = pending; branch.push_back(a);
                    if (expand(own, std::move(branch), v, bytes, final, depth)) return true;
                }
                return false;
            }
            step();
        }
        return conjunction(own, v, bytes, final, depth);
    }
    bool conjunction(const std::vector<std::size_t>& ids, const Value* v, std::string_view bytes, bool final, std::size_t depth) {
        if (v && !v->kind) v = nullptr;
        if (depth > d.limits.depth) return false;
        // A literal supplies a finite set of witnesses, checked against every
        // sibling constraint. This also handles incompatible enum/const pairs.
        for (auto id : ids) if (d.nodes[id].literal_set) {
            for (const auto& literal : d.nodes[id].literals) {
                step();
                if (v && v->kind) {
                    const auto raw = bytes.substr(v->begin, (v->done ? v->end : bytes.size()) - v->begin);
                    const bool exact = v->done && (final || v->kind != 'n' || v->end != bytes.size());
                    if (exact ? raw != literal : !std::string_view(literal).starts_with(raw)) continue;
                }
                try {
                    const auto candidate = PrefixParser(literal, d.limits).root();
                    if (!candidate.done) continue;
                    if (std::all_of(ids.begin(), ids.end(), [&](auto x) { return matches(d, x, candidate, literal, true, steps); })) return true;
                } catch (const Invalid&) {}
            }
            return false;
        }
        for (const char kind : {'0', 'b', 'n', 's', '[', '{'}) {
            if (v && v->kind && v->kind != kind) continue;
            const std::string type = kind == '0' ? "null" : kind == 'b' ? "boolean" : kind == 'n' ? "number" : kind == 's' ? "string" : kind == '[' ? "array" : "object";
            bool possible = true, integer = false;
            for (auto id : ids) {
                const auto& types = d.nodes[id].types;
                if (!types.empty() && !types.contains(type)) {
                    if (kind == 'n' && types.contains("integer")) integer = true;
                    else { possible = false; break; }
                }
            }
            if (!possible) continue;
            if (kind == 'n' && integer && v && v->kind) {
                const auto raw = bytes.substr(v->begin, (v->done ? v->end : bytes.size()) - v->begin);
                if (raw.find_first_of(".eE") != std::string_view::npos) continue;
            }
            if (kind == 's') {
                std::size_t minimum = 0, maximum = d.limits.max_string_length;
                for (auto id : ids) { minimum = std::max(minimum, d.nodes[id].min_length); maximum = std::min(maximum, d.nodes[id].max_length); }
                if (minimum > maximum || (v && (v->scalars > maximum || (v->done && v->scalars < minimum)))) continue;
            }
            if (kind == '[') {
                std::size_t minimum = 0, maximum = SIZE_MAX; std::vector<std::size_t> items;
                for (auto id : ids) { minimum = std::max(minimum, d.nodes[id].min_items); maximum = std::min(maximum, d.nodes[id].max_items); items.push_back(d.nodes[id].items); }
                if (minimum > maximum) continue;
                if (v && (v->children.size() > maximum || (v->expect_more && v->children.size() >= maximum) || (v->done && v->children.size() < minimum))) continue;
                bool valid = true;
                if (v) for (const auto& child : v->children) if (!evaluate(items, &child, bytes, final, depth + 1)) { valid = false; break; }
                if (!valid) continue;
                if ((!v && minimum > 0) || (v && !v->done && (v->children.size() < minimum || v->expect_more))) {
                    if (!evaluate(items, nullptr, {}, false, depth + 1)) continue;
                }
            }
            if (kind == '{') {
                std::set<std::string> required; const Node* closed = nullptr;
                for (auto id : ids) { const auto& n = d.nodes[id]; required.insert(n.required.begin(), n.required.end()); if (!n.additional) closed = &n; }
                const auto subs = [&](const std::string& key, std::vector<std::size_t>& out) {
                    for (auto id : ids) {
                        const auto& n = d.nodes[id];
                        const auto p = std::find_if(n.properties.begin(), n.properties.end(), [&](const auto& x) { return x.first == key; });
                        if (p != n.properties.end()) out.push_back(p->second);
                        else if (!n.additional) return false;
                    }
                    if (out.empty()) out.push_back(0);
                    return true;
                };
                bool valid = true;
                for (const auto& key : required) {
                    const bool seen = v && std::find(v->keys.begin(), v->keys.end(), key) != v->keys.end();
                    if (seen) continue;
                    if (v && v->done) { valid = false; break; }
                    std::vector<std::size_t> sub;
                    if (!subs(key, sub) || !evaluate(sub, nullptr, {}, false, depth + 1)) { valid = false; break; }
                }
                if (!valid) continue;
                if (v) {
                    for (std::size_t i = 0; i < v->keys.size(); ++i) {
                        std::vector<std::size_t> sub;
                        if (!subs(v->keys[i], sub) || !evaluate(sub, i < v->children.size() ? &v->children[i] : nullptr, bytes, final, depth + 1)) { valid = false; break; }
                    }
                    if (!valid) continue;
                    if (v->key_pending && closed) {
                        bool found = false;
                        for (const auto& [key, ignored] : closed->properties) {
                            (void)ignored;
                            if (std::find(v->keys.begin(), v->keys.end(), key) != v->keys.end() || !key_viable(key, *v)) continue;
                            std::vector<std::size_t> sub;
                            if (subs(key, sub) && evaluate(sub, nullptr, {}, false, depth + 1)) { found = true; break; }
                        }
                        if (!found) continue;
                    }
                }
            }
            return true;
        }
        return false;
    }
};
bool inspect(const ReferenceData& d, std::string_view bytes, bool* complete = nullptr) {
    try {
        const auto v = PrefixParser(bytes, d.limits).root();
        if (!Viability(d).evaluate({d.root}, &v, bytes)) return false;
        if (complete) *complete = v.done && Viability(d).evaluate({d.root}, &v, bytes, true);
        return true;
    } catch (const Invalid&) { return false; }
}
}  // namespace
struct Oracle {
    std::shared_ptr<ReferenceData> data;
    std::string bytes;
    explicit Oracle(const Json& schema, const ConstraintLimits& limits = {}) {
        ReferenceCompiler c(limits);
        if (const auto* defs = schema.find("$defs"))
            for (const auto& [name, sub] : defs->as_object()) c.sources[name] = &sub;
        c.data->root = c.node(schema, "", 0);
        data = c.data;
    }
    bool advance(unsigned char b) {
        if (bytes.size() >= data->limits.max_output_bytes) return false;
        auto next = bytes; next += static_cast<char>(b);
        if (!inspect(*data, next)) return false;
        bytes = std::move(next); return true;
    }
    bool complete() const { bool result = false; inspect(*data, bytes, &result); return result; }
};
}  // namespace reference

namespace {
using namespace strix::serve;
int failures = 0;
void check(bool ok, const std::string& message) {
    if (!ok) { ++failures; std::cerr << "FAIL: " << message << '\n'; }
}
ConstraintProgram schema(std::string_view s) { return compile_json_schema(parse_json(s)); }
bool accepts(const ConstraintProgram& p, std::string_view s) {
    ConstraintCursor c(p);
    for (unsigned char b : s) if (!c.advance(b)) return false;
    return c.complete();
}
void cases(std::string_view spec, std::initializer_list<std::string> yes, std::initializer_list<std::string> no) {
    const auto p = schema(spec);
    for (const auto& s : yes) check(accepts(p, s), "accept " + std::string(spec) + " : " + s);
    for (const auto& s : no) check(!accepts(p, s), "reject " + std::string(spec) + " : " + s);
}
void profile() {
    for (const char* keyword : {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "pattern", "format", "oneOf", "allOf", "not", "if", "then", "else", "dependentRequired", "dependentSchemas", "prefixItems", "contains", "minContains", "maxContains", "unevaluatedItems", "unevaluatedProperties", "patternProperties", "propertyNames", "unknown"}) {
        Json sub = Json::object(); sub.set(keyword, 1);
        Json props = Json::object(); props.set("age", sub);
        Json root = Json::object(); root.set("properties", props);
        try { (void)compile_json_schema(root); check(false, "unsupported accepted"); }
        catch (const SchemaError& e) { check(e.keyword == keyword && e.path == std::string("properties.age.") + keyword, "keyword path " + std::string(keyword)); }
        root = Json::object(); Json defs = Json::object(); defs.set("unused", sub); root.set("$defs", defs);
        try { (void)compile_json_schema(root); check(false, "unused definition accepted"); }
        catch (const SchemaError& e) { check(e.keyword == keyword && e.path == std::string("$defs.unused.") + keyword, "unused definition path"); }
    }
    cases(R"({"type":"object","properties":{"minimum":{"type":"integer"},"oneOf":{"type":"null"}},"title":"T","description":"D","$comment":"C"})", {R"({"minimum":1,"oneOf":null})"}, {});
    for (const char* bad : {"false", R"({"additionalProperties":{}})", R"({"items":[]})", R"({"type":"bad"})", R"({"type":[]})", R"({"type":["string","string"]})", R"({"required":[1]})", R"({"required":["a","a"]})", R"({"enum":[]})", R"({"anyOf":[]})", R"({"minLength":-1})", R"({"maxItems":1.5})", R"({"$ref":"https://invalid/schema"})", R"({"$ref":"#"})", R"({"$ref":"#/$defs/missing"})", R"({"$defs":{"x":{"$ref":"#/$defs/x"}}})", R"({"$defs":{"x":{"$ref":"#/$defs/y"},"y":{"$ref":"#/$defs/x"}}})", R"({"title":3})", R"({"$defs":[]})", R"({"properties":[]})", R"({"anyOf":[{"minimum":1}]})", R"({"items":{"minimum":1}})", R"({"properties":{"x":{"$defs":{"unused":{"minimum":1}}}}})"}) {
        try { (void)schema(bad); check(false, std::string("invalid schema accepted ") + bad); } catch (const SchemaError&) {}
    }
    cases(R"({"$defs":{"x":{"type":"integer"}},"$ref":"#/$defs/x"})", {"1", "-2"}, {"true", "1.0"});
    cases(R"({"$defs":{"a/b~c":{"type":"null"}},"$ref":"#/$defs/a~1b~0c"})", {"null"}, {"0"});
    cases(R"({"properties":{"x":{"$defs":{"unused":true},"type":"string"}}})", {R"({"x":"a"})"}, {R"({"x":3})"});
    cases(R"({"type":["string","null"]})", {"null", R"("a")"}, {"1", "false"});
    cases(R"({"anyOf":[false,{"type":"integer"},{"type":"boolean"}]})", {"1", "true", "false"}, {"null", "1.5"});
    cases(R"({"type":"string","anyOf":[{"const":"a"},{"const":1}]})", {R"("a")"}, {"1", R"("b")"});
    for (const char* empty : {R"({"type":"string","minLength":3,"maxLength":2})", R"({"type":"array","minItems":2,"maxItems":1})", R"({"type":"array","minItems":1,"items":false})", R"({"type":"object","required":["x"],"additionalProperties":false})", R"({"type":"object","required":["x"],"properties":{"x":false}})", R"({"anyOf":[false,false]})", R"({"enum":[1],"const":2})", R"({"type":"object","properties":{"a":{"type":"string"}},"required":["a"],"$ref":"#/$defs/o","$defs":{"o":{"properties":{"a":{"type":"integer"}}}}})"}) {
        try { (void)schema(empty); check(false, std::string("empty schema accepted ") + empty); } catch (const SchemaError&) {}
    }
    auto p = schema(R"({"type":["array","string"],"minItems":3,"maxItems":2})");
    ConstraintCursor c(p); check(!c.advance('[') && c.advance('"'), "empty branch pruned");
    cases(R"({"type":"object","required":["a"],"properties":{"a":{"type":["string","integer"]}},"$ref":"#/$defs/o","$defs":{"o":{"properties":{"a":{"type":"integer"}}}}})", {R"({"a":1})"}, {R"({"a":"s"})"});
    Json original = parse_json(R"({"type":"string"})"); auto snapshot = compile_json_schema(original); original.set("type", "integer");
    check(accepts(snapshot, R"("x")") && !accepts(snapshot, "1"), "compiled schema snapshot");
}
void language() {
    cases("true", {"{}", "[]", "null", "true", "false", "0", "-0", "1.2e-30", R"("\"\\\/\b\f\n\r\t")", " \n{}\t "}, {"", "01", "+1", ".1", "1.", "1e", "--1", "NaN", "{}{}", "[1,]", R"({"x":1,})", "\r{}", "   {}", "{}   ", "[ \n\t1]", "[truefalse]"});
    cases(R"({"type":"integer"})", {"0", "-0", "123", "-98765432109876543210987654321"}, {"1.0", "1e2", "01", "-", "false"});
    cases(R"({"type":"number"})", {"1", "-0.2", "1E+9", "1e0"}, {"1e+", "0x1", "00"});
    cases(R"({"type":"boolean"})", {"true", "false"}, {"null", "0", "True"});
    cases(R"({"type":"null"})", {"null"}, {"false", "nul"});
    cases(R"({"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"string"}},"required":["a"],"additionalProperties":false})", {R"({"a":1})", R"({"b":"s","a":2})", R"({"\u0061":1})"}, {"{}", R"({"a":1,"a":2})", R"({"a":1,"\u0061":2})", R"({"a":1,"c":0})", R"({"a":"x"})"});
    cases(R"({"type":"object","required":["unknown"]})", {R"({"unknown":[{"a":null}]})"}, {"{}"});
    cases(R"({"type":"object","properties":{"x":false}})", {"{}", R"({"y":0})"}, {R"({"x":null})"});
    cases(R"({"type":"object","properties":{"\u00e9":{"type":"null"},"\ud83d\ude00":{"type":"boolean"}},"additionalProperties":false})", {R"({"\u00E9":null,"\uD83d\udE00":true})", "{\"\xc3\xa9\":null,\"\xf0\x9f\x98\x80\":false}"}, {R"({"\u00e8":null})", "{\"\xc3\xa9x\":null}"});
    cases(R"({"type":"array","items":{"type":"array","items":{"type":"integer"}},"minItems":1,"maxItems":2})", {"[[]]", "[[1],[2,3]]"}, {"[]", "[[],[],[]]", "[[true]]"});
    cases(R"({"type":"array","items":false})", {"[]"}, {"[null]"});
    cases(R"({"type":"string","minLength":1,"maxLength":2})", {R"("a")", R"("\n")", R"("\u0061b")", R"("\uD83D\uDE00")", "\"\xc3\xa9\xf0\x9f\x98\x80\""}, {R"("")", R"("abc")", R"("\u0061bc")", R"("\uD800")", R"("\uDC00")", R"("\uD800\u0061")", "\"\xc0\xaf\"", "\"\xed\xa0\x80\"", "\"\xf4\x90\x80\x80\"", "\"\x80\"", "\"\xe0\x80\x80\"", "\"\xc3\"", "\"\n\""});
    cases(R"({"type":"string","maxLength":0})", {R"("")"}, {R"("a")", R"("\n")"});
    cases(R"({"enum":[12,"x",{"b":1,"a":2}]})", {"12", R"("x")", R"({"b":1,"a":2})"}, {"1", "120", R"("\u0078")", R"({"a":2,"b":1})", R"({"b": 1,"a":2})"});
    cases(R"({"const":[true,null]})", {"[true,null]"}, {"[true, null]", "[]"});
    cases(R"({"enum":[1,2],"const":2})", {"2"}, {"1"});
    auto p = compile_json_object(); check(accepts(p, "{}") && !accepts(p, "[]"), "object root mode");
    auto n = schema(R"({"const":12})"); ConstraintCursor c(n);
    check(!c.complete() && !c.dead(), "initial cursor"); check(c.advance('1') && !c.complete(), "numeric prefix incomplete");
    check(c.advance('2') && c.complete(), "numeric exact complete"); check(!c.advance('3') && c.complete(), "refusal atomic");
    check(c.advance(' ') && c.complete(), "trailing gap complete"); auto copy = c;
    check(c.advance('\n') && !c.advance(' '), "whitespace bound"); check(copy.advance('\t'), "cursor copy independent");
    auto text = schema(R"({"type":"string"})"); ConstraintCursor str(text);
    for (unsigned char b : std::string("\"</think></tool_call>")) check(str.advance(b), "framing spelling is string data");
    check(!str.complete() && str.advance('"') && str.complete(), "string completes at closing quote");
    for (const auto* value : {"{}", "[]", "true", "false", "null", "\"a\""}) {
        c = ConstraintCursor(schema("true"));
        const std::string word(value);
        for (std::size_t i = 0; i < word.size(); ++i) { check(c.advance(static_cast<unsigned char>(word[i])), "complete prefix byte"); check(c.complete() == (i + 1 == word.size()), "exact completion point"); }
    }
}
std::shared_ptr<const BpeTokenizer> tokenizer() {
    std::vector<std::string> tokens; std::vector<std::int32_t> types; int extra = 0;
    for (int b = 0; b < 256; ++b) {
        const bool direct = (b >= '!' && b <= '~') || (b >= 0xa1 && b <= 0xac) || b >= 0xae;
        const int cp = direct ? b : 256 + extra++;
        std::string s;
        if (cp < 128) s += static_cast<char>(cp);
        else { s += static_cast<char>(0xc0 | (cp >> 6)); s += static_cast<char>(0x80 | (cp & 63)); }
        tokens.push_back(s); types.push_back(1);
    }
    for (const char* s : {"{}", "{}x", "true", "trueX", "\\u", "\\uD83D\\uDE00", "\"a\"", "\"a\"x", "{}"}) { tokens.emplace_back(s); types.push_back(1); }
    tokens.push_back(tokens[0xf0] + tokens[0x9f]); types.push_back(1);
    tokens.push_back(tokens[0x98] + tokens[0x80] + "\""); types.push_back(1);
    tokens.push_back(tokens[0x98] + "x"); types.push_back(1);
    for (const auto* special : {"<|im_end|>", "<tool_call>", "</think>"}) { tokens.emplace_back(special); types.push_back(special[1] == '|' ? 3 : 4); }
    tokens.emplace_back("unknown-marker"); types.push_back(2);
    tokens.emplace_back("unused-marker"); types.push_back(5);
    tokens.emplace_back("byte-marker"); types.push_back(6);
    tokens.emplace_back(""); types.push_back(3);
    return BpeTokenizer::from_parts(std::move(tokens), std::move(types), {}, "qwen35");
}
void tokens() {
    auto t = tokenizer(); TokenByteTrie trie(*t); auto p = schema("true"); ConstraintCursor c(p);
    const std::vector<std::uint32_t> eos{static_cast<std::uint32_t>(t->vocab_size() - 1), 256};
    auto mask = allowed(p, c, trie, eos);
    check(!mask.contains(256) && !mask.contains(eos[0]) && !mask.contains(257) && mask.contains(258) && !mask.contains(259) && mask.contains(262) && !mask.contains(263) && mask.contains(264), "whole tokens, duplicates, EOS override");
    auto done = advance_token(c, "{}"); mask = allowed(p, done, trie, eos); check(mask.contains(eos[0]) && mask.contains(256), "EOS only at completion");
    for (const auto* prefix : {"[", "{\"a\":", "\"", "tru"}) check(!allowed(p, advance_token(c, prefix), trie, eos).contains(eos[0]), "no premature EOS");
    try { (void)advance_token(c, "{}x"); check(false, "partial token accepted"); } catch (const ConstraintError&) { check(!c.complete(), "token helper atomic"); }
    try { (void)advance_token(c, ""); check(false, "empty token"); } catch (const ConstraintError&) {}
    auto q = schema(R"({"type":"string"})"); auto str = advance_token(ConstraintCursor(q), "\"");
    mask = allowed(q, str, trie); check(mask.contains(260) && mask.contains(261) && mask.contains(265), "merged escapes and UTF8 start");
    str = advance_token(str, std::string("\xf0\x9f")); mask = allowed(q, str, trie);
    check(mask.contains(0x98) && mask.contains(266) && !mask.contains(267) && !mask.contains('a') && !mask.contains('"'), "merged UTF8 suffix all bytes");
    auto open = advance_token(ConstraintCursor(q), "\"");
    const auto markers = allowed(q, open, trie);
    for (const auto* special : {"<|im_end|>", "<tool_call>", "</think>", "unknown-marker", "unused-marker"})
        check(!markers.contains(t->find(special)), std::string("exclude marker token ") + special);
    check(markers.contains(t->find("byte-marker")), "byte token remains eligible");
    const std::vector<std::uint32_t> exclusions{static_cast<std::uint32_t>('a'), t->find("byte-marker")};
    TokenByteTrie filtered(*t, exclusions);
    check(!allowed(q, open, filtered).contains('a') && !allowed(q, open, filtered).contains(t->find("byte-marker")), "explicit tokenizer exclusions");
    const std::vector<std::string> fixtures{"a", "b"}; const std::vector<std::uint32_t> fixture_exclusions{0};
    TokenByteTrie fixture_trie(fixtures, fixture_exclusions);
    check(!allowed(q, open, fixture_trie).contains(0) && allowed(q, open, fixture_trie).contains(1), "fixture exclusions");
    AllowedTokens small{4, {0}}; small.set(1); small.set(2);
    const std::vector<float> logits{100, 3, 4, 200}; check(constrained_greedy(logits, small) == 2, "highest allowed");
    check(constrained_greedy(std::vector<float>{100, 4, 4, 200}, small) == 1, "lowest ID tie");
    small.set(1, false); small.set(2, false);
    try { (void)constrained_greedy(logits, small); check(false, "empty mask fail closed"); } catch (const ConstraintError&) {}
    try { (void)allowed(q, c, trie); check(false, "wrong program"); } catch (const ConstraintError&) {}
    for (const auto* prefix : {"", "[", "{\"a\":", "\"", "1", "{}"}) {
        auto state = ConstraintCursor(p); if (*prefix) state = advance_token(state, prefix);
        auto actual = allowed(p, state, trie);
        for (std::size_t id = 0; id < t->vocab_size(); ++id) {
            auto trial = state; const auto bytes = t->token_bytes(static_cast<std::uint32_t>(id)); bool ok = !bytes.empty() && (t->type(static_cast<std::uint32_t>(id)) == BpeTokenizer::TokenType::normal || t->type(static_cast<std::uint32_t>(id)) == BpeTokenizer::TokenType::byte);
            for (unsigned char b : bytes) if (!trial.advance(b)) { ok = false; break; }
            check(actual.contains(id) == ok, "trie agrees with all-byte admission");
        }
    }
}
// Independent complete-output oracle: a token scanner records raw value spans,
// with strict scalar spelling checks. Schema predicates operate
// on this tree, without production compilation, prefix parsing or viability.
struct RawValue {
    Json value;
    std::string_view bytes;
    std::vector<RawValue> children;
    std::vector<std::string> names;
};
class RawValidator {
public:
    RawValidator(std::string_view bytes, const ConstraintLimits& limits) : bytes_(bytes), limits_(limits) {}
    RawValue document() {
        if (bytes_.size() > limits_.max_output_bytes) throw JsonError("output limit");
        auto v = value(0); gap();
        if (at_ != bytes_.size()) throw JsonError("trailing bytes");
        return v;
    }
private:
    std::string_view bytes_;
    const ConstraintLimits& limits_;
    std::size_t at_ = 0;
    void gap() {
        const auto begin = at_;
        while (at_ < bytes_.size() && (bytes_[at_] == ' ' || bytes_[at_] == '\n' || bytes_[at_] == '\t')) ++at_;
        if (at_ - begin > 2) throw JsonError("gap limit");
    }
    void expect(char c) {
        gap(); if (at_ == bytes_.size() || bytes_[at_++] != c) throw JsonError("delimiter");
    }
    Json text() {
        const auto begin = at_;
        if (at_ == bytes_.size() || bytes_[at_++] != '"') throw JsonError("string");
        bool closed = false;
        while (at_ < bytes_.size()) {
            const auto b = bytes_[at_++];
            if (b == '"') { closed = true; break; }
            if (b == '\\') { if (at_ == bytes_.size()) throw JsonError("escape"); ++at_; }
        }
        if (!closed) throw JsonError("unterminated string");
        const auto spelling = std::string(bytes_.substr(begin, at_ - begin));
        static const std::regex strings(R"rx("([^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u([0-9a-cA-Ce-fE-F][0-9a-fA-F]{3}|[dD][0-7][0-9a-fA-F]{2})|\\u[dD][89aAbB][0-9a-fA-F]{2}\\u[dD][c-fC-F][0-9a-fA-F]{2})*")rx");
        if (!std::regex_match(spelling, strings)) throw JsonError("scalar escape grammar");
        auto v = parse_json(spelling);
        static const std::regex unicode(R"rx(([\x00-\x7f]|[\xc2-\xdf][\x80-\xbf]|\xe0[\xa0-\xbf][\x80-\xbf]|[\xe1-\xec\xee-\xef][\x80-\xbf]{2}|\xed[\x80-\x9f][\x80-\xbf]|\xf0[\x90-\xbf][\x80-\xbf]{2}|[\xf1-\xf3][\x80-\xbf]{3}|\xf4[\x80-\x8f][\x80-\xbf]{2})*)rx");
        if (!std::regex_match(v.as_string(), unicode)) throw JsonError("scalar UTF8 grammar");
        std::size_t scalars = 0;
        for (unsigned char b : v.as_string()) scalars += (b & 0xc0U) != 0x80U;
        if (scalars > limits_.max_string_length) throw JsonError("scalar limit");
        return v;
    }
    RawValue value(std::size_t depth) {
        if (depth > limits_.depth) throw JsonError("depth limit");
        gap(); const auto begin = at_;
        if (at_ == bytes_.size()) throw JsonError("missing value");
        RawValue out; const char first = bytes_[at_];
        if (first == '"') out.value = text();
        else if (first == '[' || first == '{') {
            const char close = first == '[' ? ']' : '}'; ++at_; gap();
            if (at_ == bytes_.size()) throw JsonError("container");
            std::set<std::string> names;
            if (bytes_[at_] != close) for (;;) {
                if (first == '{') {
                    gap(); auto key = text().as_string();
                    if (!names.insert(key).second) throw JsonError("duplicate name");
                    out.names.push_back(key); expect(':');
                }
                out.children.push_back(value(depth + 1)); gap();
                if (at_ == bytes_.size()) throw JsonError("container");
                if (bytes_[at_] == close) break;
                expect(',');
            }
            expect(close);
            out.value = parse_json(bytes_.substr(begin, at_ - begin));
        } else {
            while (at_ < bytes_.size() && std::string_view(" \t\n,]}").find(bytes_[at_]) == std::string_view::npos) ++at_;
            const auto raw = std::string(bytes_.substr(begin, at_ - begin));
            static const std::regex atom("(null|true|false|-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?)");
            if (!std::regex_match(raw, atom)) throw JsonError("scalar atom grammar");
            out.value = parse_json(raw);
        }
        out.bytes = bytes_.substr(begin, at_ - begin); return out;
    }
};
bool raw_matches(const Json& spec, const RawValue& v, const Json& root) {
    if (spec.is_bool()) return spec.as_bool();
    if (const auto* ref = spec.find("$ref")) {
        std::string name;
        const auto encoded = std::string_view(ref->as_string()).substr(8);
        for (std::size_t i = 0; i < encoded.size(); ++i) {
            if (encoded[i] == '~') { name += encoded[++i] == '0' ? '~' : '/'; }
            else name += encoded[i];
        }
        if (!raw_matches(*root.find("$defs")->find(name), v, root)) return false;
    }
    if (const auto* branches = spec.find("anyOf")) {
        bool found = false;
        for (const auto& branch : branches->as_array()) found |= raw_matches(branch, v, root);
        if (!found) return false;
    }
    if (const auto* type = spec.find("type")) {
        const auto matches = [&](const Json& t) {
            const auto& name = t.as_string();
            if (name == "null") return v.value.is_null();
            if (name == "boolean") return v.value.is_bool();
            if (name == "number") return v.value.is_number();
            if (name == "integer") return v.value.is_number() && v.bytes.find_first_of(".eE") == std::string_view::npos;
            if (name == "string") return v.value.is_string();
            if (name == "array") return v.value.is_array();
            return v.value.is_object();
        };
        bool found = false;
        if (type->is_array()) { for (const auto& t : type->as_array()) found |= matches(t); }
        else found = matches(*type);
        if (!found) return false;
    }
    if (const auto* literal = spec.find("const")) if (literal->dump() != v.bytes) return false;
    if (const auto* literals = spec.find("enum")) {
        bool found = false;
        for (const auto& literal : literals->as_array()) found |= literal.dump() == v.bytes;
        if (!found) return false;
    }
    if (v.value.is_string()) {
        std::size_t scalars = 0;
        for (unsigned char b : v.value.as_string()) scalars += (b & 0xc0U) != 0x80U;
        if (const auto* n = spec.find("minLength")) if (scalars < static_cast<std::size_t>(n->as_int())) return false;
        if (const auto* n = spec.find("maxLength")) if (scalars > static_cast<std::size_t>(n->as_int())) return false;
    }
    if (v.value.is_array()) {
        if (const auto* n = spec.find("minItems")) if (v.children.size() < static_cast<std::size_t>(n->as_int())) return false;
        if (const auto* n = spec.find("maxItems")) if (v.children.size() > static_cast<std::size_t>(n->as_int())) return false;
        if (const auto* items = spec.find("items")) for (const auto& child : v.children) if (!raw_matches(*items, child, root)) return false;
    }
    if (v.value.is_object()) {
        if (const auto* required = spec.find("required")) for (const auto& name : required->as_array())
            if (std::find(v.names.begin(), v.names.end(), name.as_string()) == v.names.end()) return false;
        for (std::size_t i = 0; i < v.names.size(); ++i) {
            const auto* props = spec.find("properties"); const auto* child = props ? props->find(v.names[i]) : nullptr;
            if (child) { if (!raw_matches(*child, v.children[i], root)) return false; }
            else if (const auto* extra = spec.find("additionalProperties")) if (!extra->as_bool()) return false;
        }
    }
    return true;
}
bool raw_valid(const Json& spec, std::string_view bytes, const ConstraintLimits& limits = {}) {
    try { return raw_matches(spec, RawValidator(bytes, limits).document(), spec); }
    catch (const JsonError&) { return false; }
}
Json random_schema(std::mt19937& rng, int depth) {
    const auto kind = rng() % (depth ? 5U : 3U);
    if (kind == 0) return parse_json(R"({"type":"integer","enum":[-1,0,1,12]})");
    if (kind == 1) return parse_json(R"({"type":"string","minLength":1,"maxLength":3})");
    if (kind == 2) return parse_json(R"({"type":"boolean"})");
    Json s = Json::object();
    if (kind == 3) {
        s.set("type", "array"); s.set("items", random_schema(rng, depth - 1)); s.set("minItems", static_cast<int>(rng() % 2)); s.set("maxItems", 2);
    } else {
        s.set("type", "object"); Json props = Json::object(); props.set("a", random_schema(rng, depth - 1)); props.set("b", random_schema(rng, depth - 1));
        s.set("properties", props); s.set("required", parse_json(rng() % 2 ? "[\"a\"]" : "[]")); s.set("additionalProperties", false);
    }
    return s;
}
void random_walks() {
    std::vector<std::string> bytes;
    for (unsigned i = 0; i < 128; ++i) bytes.emplace_back(1, static_cast<char>(i));
    for (const auto* x : {"true", "false", "\"a\":", "\"b\":", "12", "\"x\"", "\"\\u0061\"", "\"\xc3\xa9\""}) bytes.emplace_back(x);
    bytes.emplace_back(); TokenByteTrie trie(bytes); std::mt19937 rng(47193); std::size_t completed = 0;
    const std::vector<std::uint32_t> eos{static_cast<std::uint32_t>(bytes.size() - 1)};
    for (int run = 0; run < 250; ++run) {
        auto s = random_schema(rng, 2);
        if (run % 4 == 0) { Json branches = Json::array(); branches.push(s); branches.push(parse_json("{\"const\":null}")); s = Json::object(); s.set("anyOf", branches); }
        else if (run % 4 == 1) { Json defs = Json::object(); defs.set("T", s); s = Json::object(); s.set("$defs", defs); s.set("$ref", "#/$defs/T"); }
        else if (run % 4 == 2) s = parse_json(R"({"type":["boolean","integer"],"enum":[false,true,0,1]})");
        const auto p = compile_json_schema(s); auto c = ConstraintCursor(p); std::string out;
        for (unsigned step = 0; step < 1024; ++step) {
            const auto mask = allowed(p, c, trie, eos); std::vector<std::uint32_t> choices;
            for (std::size_t i = 0; i < bytes.size(); ++i) if (mask.contains(i)) choices.push_back(static_cast<std::uint32_t>(i));
            check(!choices.empty(), "random nonempty mask: " + s.dump() + " prefix=" + out);
            if (choices.empty()) break;
            const auto id = choices[rng() % choices.size()];
            if (id == eos[0]) { check(c.complete(), "random EOS complete"); break; }
            c = advance_token(c, bytes[id]); out += bytes[id];
            if (c.complete()) {
                try { check(raw_valid(s, out), "random raw-byte schema validation"); ++completed; }
                catch (const JsonError&) { check(false, "random complete parses"); }
                break;
            }
        }
    }
    check(completed >= 200, "random walks complete"); std::cout << "random walks completed=" << completed << "/250\n";
}
void equivalence() {
    std::mt19937 rng(871294);
    std::vector<Json> specs{
        parse_json("true"),
        parse_json(R"({"type":"object","required":["unknown"]})"),
        parse_json(R"({"type":"object","properties":{"a":false}})"),
        parse_json(R"({"type":"object","properties":{"a":{"type":"integer"},"\u00e9":{"type":"null"},"\ud83d\ude00":{"type":"boolean"}},"additionalProperties":false})"),
        parse_json(R"({"anyOf":[{"type":"object","properties":{"a":{"const":1},"b":{"const":2}},"additionalProperties":false},{"type":"object","properties":{"a":{"const":2},"b":{"const":1}},"additionalProperties":false}]})"),
        parse_json(R"({"enum":[1,12,123,"x",{"b":1,"a":2},[true,null]]})"),
        parse_json(R"({"type":"array","minItems":1,"maxItems":3,"items":{"anyOf":[{"const":1},{"const":12},{"type":"string","maxLength":2}]}})")
    };
    for (int i = 0; i < 100; ++i) specs.push_back(random_schema(rng, 2));
    std::size_t probes = 0, accepted = 0;
    for (const auto& spec : specs) {
        const auto p = compile_json_schema(spec);
        for (int walk = 0; walk < 4; ++walk) {
            ConstraintCursor cursor(p); reference::Oracle oracle(spec);
            for (int step = 0; step < 90; ++step) {
                std::vector<unsigned char> choices;
                for (unsigned byte = 0; byte < 256; ++byte) {
                    auto actual = cursor; auto expected = oracle;
                    const auto a = actual.advance(static_cast<unsigned char>(byte));
                    const auto e = expected.advance(static_cast<unsigned char>(byte));
                    ++probes;
                    if (a != e || actual.complete() != expected.complete()) {
                        check(false, "oracle byte=" + std::to_string(byte) + " schema=" + spec.dump() + " prefix=" + oracle.bytes);
                        return;
                    }
                    if (e) choices.push_back(static_cast<unsigned char>(byte));
                }
                if (choices.empty() || (oracle.complete() && rng() % 3 == 0)) break;
                auto byte = choices[rng() % choices.size()];
                check(cursor.advance(byte) && oracle.advance(byte), "oracle random selected byte"); ++accepted;
            }
        }
    }
    // Directed walks exercise all escape prefixes, complete/extendable enum
    // numbers, unknown-key duplicates, and correlations across object members.
    const std::vector<std::pair<std::string, std::string>> paths{
        {"true", R"({"long-name":0,"long\u002dname":1})"},
        {"true", R"({"x":0,"y":1,"z":2,"\u0078":3})"},
        {"true", R"("\uD83D\uDE00\u0061\n")"},
        {R"({"properties":{"\u00e9":true,"\ud83d\ude00":true},"additionalProperties":false})", R"({"\u00E9":null,"\uD83D\uDE00":0})"},
        {R"({"enum":[1,12,123]})", "123 "},
        {R"({"const":[{"a":[1,2]}]})", R"([{"a":[1,2]}])"},
        {specs[4].dump(), R"({"a":1,"b":1})"}
    };
    for (const auto& [spec, path] : paths) {
        auto json = parse_json(spec); auto p = compile_json_schema(json);
        ConstraintCursor cursor(p); reference::Oracle oracle(json);
        for (unsigned char next : path) {
            for (unsigned byte = 0; byte < 256; ++byte) {
                auto a = cursor; auto e = oracle;
                check(a.advance(static_cast<unsigned char>(byte)) == e.advance(static_cast<unsigned char>(byte)), "directed oracle byte " + std::to_string(byte));
                check(a.complete() == e.complete(), "directed oracle completion"); ++probes;
            }
            const bool a = cursor.advance(next), e = oracle.advance(next);
            check(a == e, "directed walk admission");
            if (!e) break;
        }
    }
    std::cout << "incremental/reference equivalence probes=" << probes << " random_bytes=" << accepted << '\n';
}
void limits() {
    const auto test = [](const Json& s, const ConstraintLimits& l) {
        try { (void)compile_json_schema(s, l); check(false, "limit accepted"); } catch (const SchemaError&) {}
    };
    ConstraintLimits l; l.schema_bytes = 2; test(parse_json("{\"type\":\"string\"}"), l);
    l = {}; l.nodes = 1; test(parse_json("{\"items\":true}"), l);
    l = {}; l.depth = 1; test(parse_json("{\"items\":{\"items\":true}}"), l);
    l = {}; l.properties = 1; test(parse_json("{\"properties\":{\"a\":true,\"b\":true}}"), l);
    l = {}; l.enum_values = 1; test(parse_json("{\"enum\":[1,2]}"), l);
    l = {}; l.compilation_steps = 1; test(Json(true), l);
    l = {}; l.max_string_length = 2; auto p = compile_json_schema(Json(true), l); check(!accepts(p, "\"abc\""), "runtime string bound");
    l = {}; l.max_output_bytes = 2; p = compile_json_schema(Json(true), l); auto c = advance_token(ConstraintCursor(p), "{}");
    check(!c.advance(' ') && c.complete(), "runtime byte bound atomic");
    l = {}; l.max_output_bytes = 1; p = compile_json_schema(Json(true), l); c = ConstraintCursor(p);
    check(!c.advance('"') && c.advance('0') && c.complete(), "byte limit prunes unfinishable string");
    l = {}; l.depth = 2; p = compile_json_schema(Json(true), l); check(!accepts(p, "[[[[]]]]"), "runtime depth bound");
    l = {}; l.depth = 256; l.compilation_steps = 100000;
    Json branches = Json::array();
    for (int i = 0; i < 512; ++i) { Json branch = Json::object(); branch.set("type", "array"); branch.set("minItems", i); branches.push(branch); }
    Json wide = Json::object(); wide.set("anyOf", branches);
    p = compile_json_schema(wide, l); c = ConstraintCursor(p); bool exhausted = false;
    for (int i = 0; i < 256; ++i) {
        try { (void)c.advance('['); }
        catch (const ConstraintError&) { exhausted = true; check(!c.dead(), "evaluation exhaustion preserves cursor"); break; }
    }
    check(exhausted, "runtime evaluation bound explicit");
}
bool continuation(ConstraintCursor cursor, std::string& bytes, std::size_t left, std::size_t& visits) {
    if (++visits > 20000) return false;
    if (cursor.complete()) return true;
    if (!left) return false;
    const std::string order = "]}\"0ab1-2:,[ truefalsnue\\6789cdefABCDEF /\t\n";
    std::array<bool, 256> tried{};
    for (unsigned i = 0; i < order.size() + 256; ++i) {
        const auto b = i < order.size() ? static_cast<unsigned char>(order[i]) : static_cast<unsigned char>(i - order.size());
        if (tried[b]) continue;
        tried[b] = true;
        auto next = cursor;
        if (!next.advance(b)) continue;
        bytes.push_back(static_cast<char>(b));
        if (continuation(std::move(next), bytes, left - 1, visits)) return true;
        bytes.pop_back();
    }
    return false;
}
void productivity() {
    std::mt19937 rng(715230); std::size_t prefixes = 0, searches = 0;
    const auto walk = [&](const Json& spec, const ConstraintLimits& limits, int walks) {
        ConstraintProgram p = compile_json_schema(spec, limits);
        for (int run = 0; run < walks; ++run) {
            ConstraintCursor cursor(p); std::string prefix;
            for (unsigned step = 0; step <= limits.max_output_bytes; ++step) {
                auto completed = prefix; std::size_t visits = 0;
                const bool found = continuation(cursor, completed, limits.max_output_bytes - prefix.size(), visits);
                searches += visits; ++prefixes;
                check(found, "PRODUCTIVITY schema=" + spec.dump() + " prefix=" + prefix);
                if (!found) return;
                check(raw_valid(spec, completed, limits), "PRODUCTIVITY independent raw-byte witness: " + completed);
                std::vector<unsigned char> choices;
                for (unsigned b = 0; b < 256; ++b) { auto next = cursor; if (next.advance(static_cast<unsigned char>(b))) choices.push_back(static_cast<unsigned char>(b)); }
                if (choices.empty()) { check(cursor.complete(), "terminal prefix complete"); break; }
                const auto b = choices[rng() % choices.size()];
                check(cursor.advance(b), "random productive step"); prefix.push_back(static_cast<char>(b));
            }
        }
    };
    for (int i = 0; i < 100; ++i) {
        auto spec = random_schema(rng, 1);
        if (i % 4 == 0) { Json branches = Json::array(); branches.push(spec); branches.push(parse_json(R"({"const":null})")); spec = Json::object(); spec.set("anyOf", branches); }
        if (i % 4 == 1) { Json defs = Json::object(); defs.set("T", spec); spec = Json::object(); spec.set("$defs", defs); spec.set("$ref", "#/$defs/T"); }
        ConstraintLimits limits; limits.max_output_bytes = 32; limits.depth = 2; limits.max_string_length = 3;
        walk(spec, limits, 3);
    }
    for (const auto depth : {0U, 1U, 2U}) for (const auto length : {0U, 1U, 2U}) {
        ConstraintLimits limits; limits.max_output_bytes = 18; limits.depth = depth; limits.max_string_length = length;
        walk(Json(true), limits, 8);
    }
    for (const auto* spec : {R"({"type":"array","items":{"const":[0]}})", R"({"type":"array","items":{"const":{"a":0}}})"}) {
        ConstraintLimits limits; limits.depth = 1;
        const auto p = compile_json_schema(parse_json(spec), limits); auto cursor = advance_token(ConstraintCursor(p), "[");
        check(!cursor.advance('[') && !cursor.advance('{') && cursor.advance(']') && cursor.complete(), "omit depth-incompatible literal");
        limits.max_output_bytes = 18; walk(parse_json(spec), limits, 4);
    }
    Json nested(0);
    for (unsigned i = 0; i < 64; ++i) { Json outer = Json::array(); outer.push(nested); nested = outer; }
    Json item = Json::object(); item.set("const", nested); Json outer = Json::object(); outer.set("type", "array"); outer.set("items", item);
    auto depth_program = compile_json_schema(outer); auto depth_cursor = advance_token(ConstraintCursor(depth_program), "[");
    check(!depth_cursor.advance('[') && depth_cursor.advance(']') && depth_cursor.complete(), "default-depth literal witness omitted");
    outer.set("minItems", 1);
    try { (void)compile_json_schema(outer); check(false, "required depth-incompatible item"); } catch (const SchemaError&) {}
    ConstraintLimits limits; limits.max_string_length = 0;
    auto object = parse_json(R"({"type":"object","properties":{"x":true},"additionalProperties":false})");
    auto p = compile_json_schema(object, limits); auto cursor = advance_token(ConstraintCursor(p), "{");
    check(!cursor.advance('"') && cursor.advance('}') && cursor.complete(), "prune oversized optional name before quote");
    const auto free_object = compile_json_schema(parse_json(R"({"type":"object"})"), limits);
    auto empty_name = advance_token(ConstraintCursor(free_object), R"({"":0)");
    check(!empty_name.advance(',') && empty_name.advance('}') && empty_name.complete(), "empty-name scalar space exhausted before comma");
    object.set("required", parse_json("[\"x\"]"));
    try { (void)compile_json_schema(object, limits); check(false, "oversized required name"); } catch (const SchemaError&) {}
    for (const auto& invalid : {std::string("\x80"), std::string("\xc0\xaf"), std::string("\xed\xa0\x80")}) {
        Json props = Json::object(); props.set(invalid, true); Json spec = Json::object(); spec.set("properties", props);
        try { (void)compile_json_schema(spec); check(false, "malformed property Unicode"); } catch (const SchemaError&) {}
        spec = Json::object(); Json required = Json::array(); required.push(invalid); spec.set("required", required);
        try { (void)compile_json_schema(spec); check(false, "malformed required Unicode"); } catch (const SchemaError&) {}
    }
    // Pending escaped and multibyte scalars must retain a viable exact name.
    limits.max_string_length = 1;
    auto named = parse_json(R"({"type":"object","properties":{"\u00e9":true,"\ud83d\ude00":true,"xy":true},"additionalProperties":false})");
    p = compile_json_schema(named, limits); cursor = advance_token(ConstraintCursor(p), "{\"\\u00");
    check(!cursor.advance('7') && cursor.advance('e'), "pending escape prunes impossible key");
    cursor = advance_token(ConstraintCursor(p), std::string("{\"\xc3"));
    check(!cursor.advance(0xa8) && cursor.advance(0xa9), "pending UTF8 viable key");
    limits.max_output_bytes = 24; walk(named, limits, 8);
    for (unsigned size = 1; size <= 12; ++size) { limits = {}; limits.max_output_bytes = size; limits.max_string_length = 1; limits.depth = 1; walk(Json(true), limits, 4); }
    std::cout << "PRODUCTIVITY prefixes=" << prefixes << " bounded_search_states=" << searches << " (all witnesses independently raw-byte validated)\n";
}
void raw_oracle() {
    const std::vector<std::pair<std::string, std::vector<std::string>>> fixtures{
        {R"({"const":1})", {"1", "1.0", "1e0", " 1 ", "01"}},
        {R"({"enum":["x",{"b":1,"a":2}]})", {R"("x")", R"("\u0078")", R"({"b":1,"a":2})", R"({"a":2,"b":1})", R"({"b": 1,"a":2})"}},
        {R"({"type":"object","properties":{"x":{"const":1}},"additionalProperties":false})", {"{}", R"({"x":1})", R"({"x":1.0})", R"({"x":1,"\u0078":1})", R"({"y":1})"}},
        {"true", {"[1,]", "[true,false]", "[   1]", "\r{}", "\r0", "0\r", "\"\\uD800\"", "\"\\uD83D\\uDE00\"", "\"\xc0\xaf\"", "\"\xe0\xa0\x80\""}}
    };
    std::size_t probes = 0;
    for (const auto& [text, outputs] : fixtures) {
        const auto spec = parse_json(text); const auto p = compile_json_schema(spec);
        for (const auto& out : outputs) { check(accepts(p, out) == raw_valid(spec, out), "independent complete-output oracle: " + out); ++probes; }
    }
    check(!raw_valid(parse_json(R"({"const":1})"), "1.0"), "raw oracle rejects numeric value equality");
    std::cout << "independent raw-byte oracle directed_probes=" << probes << '\n';
}
void adversarial_compilation() {
    const auto measure = [](const char* name, const std::function<Json()>& make, bool refuse, ConstraintLimits limits = {}) {
        const auto start = std::chrono::steady_clock::now();
        const auto run = [&] {
            const auto spec = make();
            bool refused = false;
            try { (void)compile_json_schema(spec, limits); } catch (const SchemaError&) { refused = true; }
            return refused == refuse;
        };
        bool ok; long rss = 0;
#ifdef __linux__
        const auto pid = fork();
        if (pid == 0) { try { _exit(run() ? 0 : 1); } catch (...) { _exit(2); } }
        int status = 0; rusage usage{};
        ok = pid > 0 && wait4(pid, &status, 0, &usage) == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0;
        rss = usage.ru_maxrss;
#else
        ok = run();
#endif
        const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        check(ok, std::string("aggregate budget ") + name);
        check(ms < 2000 && rss < 65536, std::string("bounded time/memory ") + name);
        std::cout << "adversarial compile shape=" << name << " expected=" << (refuse ? "SchemaError" : "compiled") << " ms=" << ms << " peak_rss_kib=" << rss << '\n';
    };
    measure("repeated-ref-1024-enum-512-branches-46628-bytes", [] {
        Json values = Json::array();
        for (int i = 0; i < 1024; ++i) { char index[5]; std::snprintf(index, sizeof(index), "%04x", i); values.push(std::string(index) + std::string(28, 'a')); }
        Json e = Json::object(); e.set("enum", values); Json defs = Json::object(); defs.set("E", e);
        Json branches = Json::array(); for (int i = 0; i < 512; ++i) branches.push(parse_json(R"({"$ref":"#/$defs/E"})"));
        Json root = Json::object(); root.set("$defs", defs); root.set("anyOf", branches);
        if (root.dump().size() != 46628) throw std::runtime_error("adversarial schema size");
        return root;
    }, true);
    measure("deep-ref-chain", [] {
        Json defs = Json::object();
        for (int i = 0; i < 100; ++i) { Json next = Json::object(); next.set("$ref", "#/$defs/T" + std::to_string(i + 1)); defs.set("T" + std::to_string(i), next); }
        defs.set("T100", true); Json root = Json::object(); root.set("$defs", defs); root.set("$ref", "#/$defs/T0"); return root;
    }, true);
    measure("wide-anyOf", [] {
        Json branches = Json::array();
        for (int i = 0; i < 2500; ++i) { Json branch = Json::object(); branch.set("const", i); branches.push(branch); }
        Json root = Json::object(); root.set("anyOf", branches); return root;
    }, true);
    measure("many-large-enums", [] {
        Json props = Json::object();
        for (int i = 0; i < 16; ++i) { Json values = Json::array(); for (int j = 0; j < 128; ++j) values.push(std::to_string(j) + std::string(16, static_cast<char>('a' + i))); Json child = Json::object(); child.set("enum", values); props.set(std::to_string(i), child); }
        Json root = Json::object(); root.set("properties", props); return root;
    }, true);
    // Fits only when identical branches and their literal trie are shared.
    ConstraintLimits small; small.trie_nodes = 12; small.alternative_expansions = 8;
    measure("deduplicated-ref-alternatives", [] {
        auto root = parse_json(R"({"$defs":{"E":{"enum":["abc","abd"]}}})");
        Json branches = Json::array(); for (int i = 0; i < 512; ++i) branches.push(parse_json(R"({"$ref":"#/$defs/E"})"));
        root.set("anyOf", branches); return root;
    }, false, small);
    small = {}; small.trie_nodes = 12; small.alternative_expansions = 8;
    measure("deduplicated-enum-order", [] { return parse_json(R"({"anyOf":[{"enum":["abc","abd"]},{"enum":["abd","abc"]}]})"); }, false, small);
    small = {}; small.trie_nodes = 8;
    measure("shared-literal-trie", [] { return parse_json(R"({"anyOf":[{"type":"integer","enum":[1,2]},{"type":"number","enum":[1,2]}]})"); }, false, small);
    small = {}; small.trie_nodes = 32;
    measure("aggregate-trie-growth", [] { return parse_json(R"({"anyOf":[{"const":"aaaaaaaaaa"},{"const":"bbbbbbbbbb"},{"const":"cccccccccc"}]})"); }, true, small);
    small = {}; small.literal_bytes = 96;
    measure("aggregate-literal-scans", [] { return parse_json(R"({"anyOf":[{"const":"aaaaaaaaaa"},{"const":"bbbbbbbbbb"}]})"); }, true, small);
    small = {}; small.literal_bytes = 8;
    measure("literal-byte-cap", [] { return parse_json(R"({"const":"abcdefghijkl"})"); }, true, small);
    small = {}; small.trie_nodes = 4;
    measure("trie-node-cap", [] { return parse_json(R"({"const":"abcdefghijkl"})"); }, true, small);
    small = {}; small.alternative_expansions = 2;
    measure("alternative-cap", [] { return parse_json(R"({"anyOf":[{"const":1},{"const":2}]})"); }, true, small);
}
void branching_expansion() {
    for (const unsigned levels : {11U, 12U, 20U}) {
        Json defs = Json::object();
        for (unsigned i = 0; i < levels; ++i) {
            const auto ref = "#/$defs/T" + std::to_string(i + 1);
            Json next = Json::object(); next.set("$ref", ref);
            Json alternative = Json::object(); alternative.set("$ref", ref);
            Json alternatives = Json::array(); alternatives.push(alternative);
            next.set("anyOf", alternatives); defs.set("T" + std::to_string(i), next);
        }
        defs.set("T" + std::to_string(levels), true);
        Json spec = Json::object(); spec.set("$defs", defs); spec.set("$ref", "#/$defs/T0");
        check(spec.dump().size() == (levels == 11 ? 696U : levels == 12 ? 758U : 1254U), "branching fixture size");
        ConstraintCompilationStats stats;
        const auto start = std::chrono::steady_clock::now();
        const auto p = compile_json_schema(spec, {}, &stats);
        const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
        long rss = 0;
#ifdef __linux__
        rusage usage{}; getrusage(RUSAGE_SELF, &usage); rss = usage.ru_maxrss;
#endif
        check(accepts(p, "true") && accepts(p, "{}"), "branching expansion preserves language");
        check(stats.expansion_recursion_high_water == 1, "constant native expansion recursion");
        check(stats.expansion_stack_high_water == 1, "single-alternative work stack bounded");
        check(stats.compilation_steps < 20000 && ms < 2000 && rss < 65536, "branching expansion bounded work/time/memory");
        std::cout << "branching expansion levels=" << levels << " schema_bytes=" << spec.dump().size()
                  << " ms=" << ms << " peak_rss_kib=" << rss
                  << " recursion_high_water=" << stats.expansion_recursion_high_water
                  << " work_stack_high_water=" << stats.expansion_stack_high_water
                  << " charged_steps=" << stats.compilation_steps << '\n';
        ConstraintLimits limited; limited.alternative_expansions = 8;
        try { (void)compile_json_schema(spec, limited, &stats); check(false, "branch expansion budget charged"); }
        catch (const SchemaError&) {}
    }
}
void exhaustive_lexemes() {
    const auto number = schema(R"({"type":"number"})"); const auto integer = schema(R"({"type":"integer"})");
    const std::regex number_re("-?(0|[1-9][0-9]*)(\\.[0-9]+)?([eE][+-]?[0-9]+)?");
    const std::regex integer_re("-?(0|[1-9][0-9]*)"); std::size_t count = 0;
    std::function<void(std::string)> walk = [&](std::string word) {
        check(accepts(number, word) == std::regex_match(word, number_re), "number lexeme " + word);
        check(accepts(integer, word) == std::regex_match(word, integer_re), "integer lexeme " + word);
        ++count; if (word.size() == 5) return;
        for (char c : std::string("01-.e+")) walk(word + c);
    };
    walk(""); auto text = schema(R"({"type":"string"})");
    for (unsigned b = 0; b < 256; ++b) {
        const std::string word = "\"" + std::string(1, static_cast<char>(b)) + "\"";
        check(accepts(text, word) == (b >= 32 && b < 128 && b != '"' && b != '\\'), "single string byte " + std::to_string(b));
    }
    std::cout << "exhaustive numeric lexemes=" << count << "; string bytes=256\n";
}
void cache_exactness() {
    ConstraintLimits limits; limits.mask_cache_bytes = 8192;
    const auto spec = parse_json("true");
    const auto cached = compile_json_schema(spec, limits), plain = compile_json_schema(spec);
    std::vector<std::string> vocabulary;
    for (unsigned byte = 0; byte < 256; ++byte) vocabulary.emplace_back(1, static_cast<char>(byte));
    vocabulary.emplace_back(); TokenByteTrie trie(vocabulary);
    std::size_t masks = 0;
    // Reconstruct the same states independently and repeat in reverse order:
    // cache equality must cover parents, counters, bits and partial scalars.
    std::vector<std::string> prefixes{"", " ", "  ", "[", "[1", "[12", "[1,", "[1,2", "[1]", "[1] ", "\"", "\"a", "\"\\", "\"\\u", "\"\\uD83D\\u", "{", "{\"a", "{\"a\":", "{\"a\":1,", "{\"b\":1,", "{\"a\":\"", "{}", "{}\t"};
    for (int repeat = 0; repeat < 4; ++repeat) {
        for (const auto& prefix : prefixes) {
            const auto a = prefix.empty() ? ConstraintCursor(cached) : advance_token(ConstraintCursor(cached), prefix);
            const auto b = prefix.empty() ? ConstraintCursor(plain) : advance_token(ConstraintCursor(plain), prefix);
            for (const std::vector<std::uint32_t>& eos : {std::vector<std::uint32_t>{}, {256}, {static_cast<std::uint32_t>('x'), 256}}) {
                const auto actual = allowed(cached, a, trie, eos), expected = allowed(plain, b, trie, eos);
                check(actual.words == expected.words, "cache exact across states and EOS"); ++masks;
            }
        }
        std::reverse(prefixes.begin(), prefixes.end());
    }
    // Declared names exercise serialized property bits rather than the
    // additional-name bypass used above.
    auto obj = parse_json(R"({"type":"object","properties":{"a":true,"b":true},"additionalProperties":false})");
    auto p = compile_json_schema(obj, limits), q = compile_json_schema(obj);
    for (const auto* prefix : {"{", "{\"a", "{\"a\":1,", "{\"b\":1,", "{\"a\":\"\\u", "{\"a\":1,\"b\":2}"}) {
        const auto a = advance_token(ConstraintCursor(p), prefix), b = advance_token(ConstraintCursor(q), prefix);
        for (int repeat = 0; repeat < 3; ++repeat) { check(allowed(p, a, trie).words == allowed(q, b, trie).words, "cache declared-property state"); ++masks; }
    }
    // Vocabulary ownership, including equal-sized tries and destruction/address
    // reuse; a cached mask never belongs to a different vocabulary.
    for (int i = 0; i < 40; ++i) {
        const std::vector<std::string> bytes = i % 2 ? std::vector<std::string>{"{}", "x"} : std::vector<std::string>{"x", "{}"};
        TokenByteTrie other(bytes);
        check(allowed(cached, ConstraintCursor(cached), other).words == allowed(plain, ConstraintCursor(plain), other).words, "cache vocabulary identity"); ++masks;
    }
    limits.mask_cache_bytes = 1;
    const auto tiny = compile_json_schema(spec, limits);
    check(allowed(tiny, ConstraintCursor(tiny), trie).words == allowed(plain, ConstraintCursor(plain), trie).words, "undersized cache bypass");
    std::cout << "cache exactness masks=" << masks << " budget_bytes=8192 (eviction, EOS, trie identity, bypass)\n";
}
void benchmark() {
    std::vector<std::string> bytes; bytes.reserve(151000);
    for (unsigned i = 0; i < 151000; ++i) {
        switch (i % 4) {
        case 0: bytes.push_back(std::to_string(i)); break;
        case 1: bytes.push_back("word" + std::to_string(i)); break;
        case 2: bytes.push_back("\"key" + std::to_string(i) + "\":"); break;
        default: bytes.push_back("}" + std::to_string(i)); break;
        }
    }
    TokenByteTrie trie(bytes); const auto p = schema(R"({"type":"object","properties":{"a":{"type":"integer"}},"additionalProperties":false})");
    for (const auto* prefix : {"", "{", "{\"a\":", "{\"a\":1}"}) {
        auto c = ConstraintCursor(p); if (*prefix) c = advance_token(c, prefix);
        const auto start = std::chrono::steady_clock::now(); std::size_t count = 0;
        for (int i = 0; i < 3; ++i) { auto mask = allowed(p, c, trie); for (auto word : mask.words) count += static_cast<std::size_t>(word != 0); }
        const auto ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count() / 3;
        std::cout << "mask benchmark vocab=151000 prefix='" << prefix << "' ms=" << ms << " nonzero_words=" << count / 3 << '\n';
    }
    {
        Json properties = Json::object(), required = Json::array();
        std::vector<std::string> names;
        std::string output = "{";
        for (unsigned i = 0; i < 128; ++i) {
            char index[4]; std::snprintf(index, sizeof(index), "%03u", i);
            const auto name = std::string(50, 'a') + index;
            properties.set(name, true); required.push(name); names.push_back(name);
            if (i) output += ',';
            append_json_string(output, name); output += ":true";
        }
        output += '}';
        Json spec = Json::object(); spec.set("type", "object"); spec.set("properties", properties);
        spec.set("required", required); spec.set("additionalProperties", false);
        check(spec.dump().size() == 15050, "shared-prefix fixture size");
        ConstraintLimits limits; limits.max_output_bytes = output.size();
        const auto required_program = compile_json_schema(spec, limits);
        check(accepts(required_program, output), "cached required cost exact byte boundary");
        std::reverse(names.begin(), names.end());
        std::string reverse = "{";
        for (const auto& name : names) { if (reverse.size() > 1) reverse += ','; append_json_string(reverse, name); reverse += ":true"; }
        reverse += '}';
        check(accepts(required_program, reverse), "cached required cost independent of member order");
        limits.max_output_bytes = output.size() - 1;
        const auto short_program = compile_json_schema(spec, limits); // numbers provide a shorter completion
        auto short_cursor = ConstraintCursor(short_program); bool refused = false;
        for (unsigned char b : output) if (!short_cursor.advance(b)) { refused = true; break; }
        check(refused, "cached required cost refuses over-budget completion");
        auto shared_bytes = bytes;
        for (const auto& name : names) shared_bytes.push_back(name + "\":true");
        shared_bytes.push_back(std::string(50, 'a')); shared_bytes.push_back("a");
        shared_bytes.push_back("000\":true");
        TokenByteTrie shared_trie(shared_bytes);
        for (const unsigned position : {0U, 1U, 25U, 49U, 50U}) {
            const auto cursor = advance_token(ConstraintCursor(required_program), "{\"" + std::string(position, 'a'));
            for (int i = 0; i < 3; ++i) (void)allowed(required_program, cursor, shared_trie);
            std::vector<double> timings;
            for (int i = 0; i < 11; ++i) {
                const auto start = std::chrono::steady_clock::now();
                const auto mask = allowed(required_program, cursor, shared_trie);
                timings.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
                check(mask.contains(shared_bytes.size() - 1) == (position == 50), "shared-prefix candidate mask");
            }
            std::sort(timings.begin(), timings.end());
            std::cout << "shared-prefix mask required=128 schema_bytes=15050 vocab=" << shared_bytes.size()
                      << " shared_prefix_position=" << position << " median_ms=" << timings[5] << '\n';
            check(timings[5] < 100, "shared-prefix mask bounded time");
        }
        auto cursor = advance_token(ConstraintCursor(required_program), "{\"");
        check(!cursor.advance('b'), "shared-prefix impossible candidate refusal");
        // Correlated branches share compiled children but still visit every
        // candidate at runtime. The quote must exhaust the transition budget
        // explicitly, without committing a partial cursor update.
        Json choices = Json::array(), arrays = Json::array();
        for (int i = 0; i < 8; ++i) {
            Json choice = Json::object(); choice.set("$ref", "#/$defs/Required"); choice.set("minItems", i);
            choices.push(choice);
        }
        for (int i = 0; i < 512; ++i) {
            Json array = Json::object(); array.set("type", "array"); array.set("minItems", i);
            array.set("items", parse_json(R"({"$ref":"#/$defs/Choices"})")); arrays.push(array);
        }
        Json choice_schema = Json::object(); choice_schema.set("anyOf", choices);
        Json defs = Json::object(); defs.set("Required", spec); defs.set("Choices", choice_schema);
        Json branched = Json::object(); branched.set("$defs", defs); branched.set("anyOf", arrays);
        limits = {}; limits.compilation_steps = 400000; limits.max_output_bytes = 16 * 1048576;
        ConstraintCompilationStats stats;
        const auto budgeted = compile_json_schema(branched, limits, &stats);
        cursor = advance_token(ConstraintCursor(budgeted), "[{");
        bool exhausted = false;
        try { (void)cursor.advance('"'); }
        catch (const ConstraintError& e) { exhausted = std::string(e.what()) == "constraint key candidate resource limit"; }
        check(exhausted && !cursor.dead() && !cursor.advance('}'), "candidate traversal charged and exhaustion atomic");
        std::cout << "shared-prefix candidate budget=400000 compiled_steps=" << stats.compilation_steps
                  << " runtime_branches=4096 quote_exhausted=" << exhausted << '\n';
    }
    const auto array_schema = parse_json(R"({"type":"array","items":{"type":"object","properties":{"a":{"type":"integer"}},"additionalProperties":false}})");
    const auto array = compile_json_schema(array_schema);
    const auto prefix_at = [](std::size_t size) {
        const auto items = (size - 6) / 8, padding = (size - 6) % 8;
        std::string prefix(std::min<std::size_t>(padding, 2), ' '); prefix += '[';
        prefix.append(std::min<std::size_t>(padding > 2 ? padding - 2 : 0, 2), ' ');
        for (std::size_t i = 0; i < items; ++i) prefix += "{\"a\":0},";
        prefix += '{'; prefix.append(padding > 4 ? padding - 4 : 0, ' '); prefix += "\"a\":";
        return prefix;
    };
    for (const auto size : {10U, 1000U, 10000U}) {
        const auto prefix = prefix_at(size); check(prefix.size() == size, "scaling prefix size");
        const auto c = advance_token(ConstraintCursor(array), prefix);
        for (int i = 0; i < 3; ++i) (void)allowed(array, c, trie);
        std::vector<double> timings; std::size_t count = 0;
        for (int i = 0; i < 11; ++i) {
            const auto start = std::chrono::steady_clock::now(); auto mask = allowed(array, c, trie);
            timings.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count());
            for (auto word : mask.words) count += static_cast<std::size_t>(word != 0);
        }
        std::sort(timings.begin(), timings.end());
        std::cout << "scaling mask vocab=151000 prior_bytes=" << size << " median_ms=" << timings[timings.size() / 2] << " nonzero_words=" << count / timings.size() << '\n';
        constexpr std::size_t copies = 1000000;
        const auto start = std::chrono::steady_clock::now(); std::size_t completed = 0;
        for (std::size_t i = 0; i < copies; ++i) { auto copy = c; completed += static_cast<std::size_t>(copy.complete()); }
        const auto ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - start).count() / copies;
        std::cout << "cursor copy prior_bytes=" << size << " ns_per_copy_and_complete=" << ns << " sizeof_cursor=" << sizeof(ConstraintCursor) << " completed=" << completed << '\n';
    }
    std::string workload = "[";
    for (int i = 0; i < 20000; ++i) workload += i ? ", {\"a\":123456789}" : "{\"a\":123456789}";
    workload += ']';
    const auto start = std::chrono::steady_clock::now(); auto c = ConstraintCursor(array);
    std::size_t admitted = 0;
    for (unsigned char b : workload) if (c.advance(b)) ++admitted;
    const auto seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    check(admitted == workload.size() && c.complete(), "throughput workload complete");
    std::cout << "advance throughput bytes=" << admitted << " seconds=" << seconds << " bytes_per_second=" << static_cast<double>(admitted) / seconds << '\n';
    ConstraintLimits cached_limits; cached_limits.mask_cache_bytes = 131072;
    const auto cached = compile_json_schema(array_schema, cached_limits);
    const auto state = advance_token(ConstraintCursor(cached), prefix_at(10000));
    (void)allowed(cached, state, trie);
    const auto cache_start = std::chrono::steady_clock::now();
    for (int i = 0; i < 100; ++i) { auto mask = allowed(cached, state, trie); check(mask.contains(4), "cached benchmark admission"); }
    std::cout << "cached mask vocab=151000 prior_bytes=10000 budget_bytes=131072 mean_ms="
              << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - cache_start).count() / 100 << '\n';
}
}  // namespace
// Literal validation of a const over a long preloaded reference chain is bounded (refused, never a native stack overflow).
void literal_reference_chain() {
    for (const int length : {50, 300}) {
        Json defs = Json::object();
        char name[32];
        for (int i = 0; i < length; ++i) {
            std::snprintf(name, sizeof(name), "T%05d", i);
            if (i == 0) { defs.set(name, Json(true)); continue; }
            char target[32]; std::snprintf(target, sizeof(target), "#/$defs/T%05d", i - 1);
            Json ref = Json::object(); ref.set("$ref", Json(target)); defs.set(name, std::move(ref));
        }
        Json load = Json::object(); load.set("$defs", defs);
        Json properties = Json::object(); properties.set("load", std::move(load));
        std::snprintf(name, sizeof(name), "#/$defs/T%05d", length - 1);
        Json spec = Json::object(); spec.set("$defs", defs); spec.set("properties", std::move(properties));
        spec.set("$ref", Json(name)); spec.set("const", Json(true));
        bool refused = false, compiled = false;
        try { const auto p = compile_json_schema(spec); compiled = accepts(p, "true") && !accepts(p, "false"); }
        catch (const SchemaError&) { refused = true; }
        if (length == 50) check(compiled, "short literal reference chain compiles and keeps the const");
        else check(refused || compiled, "long literal reference chain refused or compiled, never a crash");
        std::cout << "literal reference chain length=" << length << (refused ? " refused" : " compiled") << '\n';
    }
}

struct CompilationBarrier {
    std::mutex mutex;
    std::condition_variable changed;
    std::size_t entered = 0;
    bool released = false;
    static void pause(void* context) {
        auto& self = *static_cast<CompilationBarrier*>(context);
        std::unique_lock lock(self.mutex);
        ++self.entered;
        self.changed.notify_all();
        self.changed.wait(lock, [&] { return self.released; });
    }
    bool wait(std::size_t count) {
        std::unique_lock lock(mutex);
        return changed.wait_for(lock, std::chrono::seconds(10), [&] { return entered >= count; });
    }
    void release() {
        std::lock_guard lock(mutex);
        released = true;
        changed.notify_all();
    }
};

void program_cache_allocation_failure() {
    const auto spec = parse_json(R"({"const":0})");
    bool reached_compile = false;
    const auto hook = [](void* context) {
        *static_cast<bool*>(context) = true;
        allocation_failure::remaining = -1;
    };
    std::size_t injected = 0;
    bool swept = false;
    for (std::ptrdiff_t nth = 0; nth < 256; ++nth) {
        auto cache = ConstraintProgramCacheTestAccess::make({}, hook, &reached_compile);
        reached_compile = false;
        bool threw = false;
        allocation_failure::remaining = nth;
        try { (void)cache.json_schema(spec); } catch (const std::bad_alloc&) { threw = true; }
        allocation_failure::remaining = -1;
        if (reached_compile) { swept = true; break; }
        ++injected;
        check(threw, "allocation sweep must throw before compilation");
        check(cache.stats().programs == 0 && cache.stats().key_bytes == 0, "failed insertion leaves no node or charge");
        cache.clear();
        check(cache.stats().programs == 0 && cache.stats().key_bytes == 0, "clear after allocation failure");
        const auto reused = cache.json_schema(spec);
        check(reused == cache.json_schema(spec), "reuse after allocation failure");
        cache.clear();
        check(cache.stats().programs == 0 && cache.stats().key_bytes == 0, "clear after reuse");
    }
    check(swept && injected > 0, "swept all allocations through list and index insertion");
}

void program_cache_pending_bounds() {
    ConstraintProgramCache probe;
    (void)probe.json_schema(parse_json(R"({"const":0})"));
    const auto key_size = probe.stats().key_bytes;
    for (const bool byte_bound : {false, true}) {
        ConstraintProgramCacheConfig config;
        config.max_programs = byte_bound ? 16 : 2;
        config.key_bytes = byte_bound ? key_size * 4 : config.key_bytes;
        const std::size_t retained = byte_bound ? 4 : 2;
        CompilationBarrier barrier;
        auto cache = ConstraintProgramCacheTestAccess::make(config, CompilationBarrier::pause, &barrier);
        std::vector<std::thread> threads;
        std::atomic<unsigned> errors{0};
        for (unsigned i = 0; i < 8; ++i) threads.emplace_back([&, i] {
            try { (void)cache.json_schema(parse_json("{\"const\":" + std::to_string(i) + "}")); }
            catch (...) { ++errors; }
        });
        check(barrier.wait(8), "all distinct compilers held pending");
        const auto pending = cache.stats();
        check(pending.programs == retained && pending.key_bytes == retained * key_size, "pending retention uses bounded capacity");
        check(pending.programs <= config.max_programs && pending.key_bytes <= config.key_bytes, "both bounds hold before any completion");
        check(pending.uncached == 8 - retained, "pending capacity bypasses retention");
        cache.clear();
        check(cache.stats().programs == retained, "clear preserves admitted pending futures");
        barrier.release();
        for (auto& thread : threads) thread.join();
        check(errors == 0, "retained and bypassed compilers succeed");
        cache.clear();
        check(cache.stats().programs == 0 && cache.stats().key_bytes == 0, "pending cache clears after completion");
        const auto spec = parse_json(R"({"const":0})");
        check(cache.json_schema(spec) == cache.json_schema(spec), "pending cache remains reusable");
    }
}

void program_cache_joiners() {
    for (const bool invalid : {false, true}) {
        const auto spec = parse_json(invalid ? R"({"type":"invalid"})" : R"({"const":0})");
        CompilationBarrier barrier;
        auto cache = ConstraintProgramCacheTestAccess::make({}, CompilationBarrier::pause, &barrier);
        std::array<std::shared_ptr<const ConstraintProgram>, 8> got;
        std::array<std::string, 8> errors;
        const auto request = [&](std::size_t i) {
            try { got[i] = cache.json_schema(spec); }
            catch (const SchemaError& e) { errors[i] = e.path + ":" + e.keyword + ":" + e.what(); }
            catch (...) { errors[i] = "unexpected exception"; }
        };
        std::vector<std::thread> threads;
        threads.emplace_back(request, 0);
        check(barrier.wait(1), "leader held before compilation");
        for (std::size_t i = 1; i < got.size(); ++i) threads.emplace_back(request, i);
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (cache.stats().joined != 7 && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
        check(cache.stats().joined == 7 && cache.stats().hits == 0, "all seven followers joined a pending future");
        barrier.release();
        for (auto& thread : threads) thread.join();
        for (std::size_t i = 0; i < got.size(); ++i) {
            if (invalid) check(!got[i] && !errors[i].empty() && errors[i] != "unexpected exception" && errors[i] == errors[0], "failed joiners receive the same schema error");
            else check(got[i] && got[i] == got[0] && errors[i].empty(), "successful joiners share one program");
        }
        const auto stats = cache.stats();
        check(stats.misses == 1 && stats.failures == (invalid ? 1U : 0U), "one joined compilation outcome");
        if (invalid) {
            check(stats.programs == 0 && stats.key_bytes == 0, "failed joined compilation removes its entry");
            try { (void)cache.json_schema(spec); } catch (const SchemaError&) {}
            check(cache.stats().failures == 2, "failed compilation is retried");
        }
    }
}

void program_cache_environment() {
    const auto& cache = shared_constraint_program_cache();
    const auto entries = strix::envcfg::registered();
    bool found = false;
    for (const auto& entry : entries) if (entry.name == "STRIX_CONSTRAINT_MASK_MEMO_BYTES")
        found = entry.value == std::to_string(cache.config().mask_cache_bytes);
    check(found, "mask memo byte setting records its resolved value");
}

// The shared program cache: one program per (schema, limits), masks equal to a per-request compilation,
// bounded, errors not kept, one compilation for concurrent identical requests.
void program_cache() {
    const char* text = R"({"type":"object","properties":{"name":{"type":"string","maxLength":40},"kind":{"enum":["alpha","beta","gamma"]},
        "count":{"type":"integer"},"tags":{"type":"array","items":{"type":"string","maxLength":12},"maxItems":4},
        "nested":{"type":"object","properties":{"ok":{"type":"boolean"},"score":{"type":"number"}},"required":["ok"],"additionalProperties":false}},
        "required":["name","kind"],"additionalProperties":false})";
    const Json spec = parse_json(text);
    std::vector<std::string> vocabulary;
    for (unsigned byte = 0; byte < 256; ++byte) vocabulary.emplace_back(1, static_cast<char>(byte));
    for (const char* piece : {"{\"", "\":", "\",\"", "name", "kind", "alpha", "true", "false", "},", "\"}"}) vocabulary.emplace_back(piece);
    const TokenByteTrie trie(vocabulary), other(vocabulary);
    const std::vector<std::uint32_t> eos{static_cast<std::uint32_t>(vocabulary.size() - 1U)};
    {
        ConstraintProgramCache cache;
        const auto a = cache.json_schema(spec), b = cache.json_schema(parse_json(text));
        check(a && a == b, "same schema shares one program");
        ConstraintLimits tighter; tighter.max_output_bytes = 4096;
        const auto c = cache.json_schema(spec, tighter);
        check(c != a && cache.json_schema(spec, tighter) == c, "limits are part of the key");
        const auto reordered = cache.json_schema(parse_json(R"({"additionalProperties":false,"type":"object"})"));
        const auto ordered = cache.json_schema(parse_json(R"({"type":"object","additionalProperties":false})"));
        check(reordered != ordered, "member order is part of the key");
        check(cache.json_object() == cache.json_object(), "json_object shares one program");
        auto stats = cache.stats();
        check(stats.misses == 5 && stats.hits == 3 && stats.programs == 5 && stats.failures == 0, "program cache counts");
        // Masks: the shared program (mask memo on, second pass served from it) against a compilation per request.
        const auto plain = compile_json_schema(spec);
        (void)allowed(*a, ConstraintCursor(*a), trie);
        const auto cold_memo = a->memo_stats();
        check(cold_memo.misses == 1 && cold_memo.hits == 0, "first memo lookup misses");
        (void)allowed(*a, ConstraintCursor(*a), trie, eos);
        const auto warm_memo = a->memo_stats();
        check(warm_memo.misses == 1 && warm_memo.hits == 1, "repeated state with EOS overlay hits memo");
        std::size_t masks = 0;
        const std::vector<std::string> prefixes{"", "{", "{\"", "{\"name", "{\"name\":", "{\"name\":\"", "{\"name\":\"a", "{\"name\":\"a\",",
            "{\"kind\":\"", "{\"kind\":\"al", "{\"kind\":\"alpha\",\"name\":\"x\"", "{\"kind\":\"beta\",\"name\":\"x\",\"count\":1",
            "{\"name\":\"x\",\"kind\":\"gamma\",\"tags\":[", "{\"name\":\"x\",\"kind\":\"gamma\",\"tags\":[\"a\",",
            "{\"name\":\"x\",\"kind\":\"gamma\",\"nested\":{\"ok\":tr", "{\"name\":\"x\",\"kind\":\"gamma\"}"};
        for (int pass = 0; pass < 3; ++pass)
            for (const auto& prefix : prefixes)
                for (const TokenByteTrie* t : {&trie, &other})
                    for (const std::vector<std::uint32_t>& stop : {std::vector<std::uint32_t>{}, eos}) {
                        const auto shared = prefix.empty() ? ConstraintCursor(*a) : advance_token(ConstraintCursor(*a), prefix);
                        const auto fresh = prefix.empty() ? ConstraintCursor(plain) : advance_token(ConstraintCursor(plain), prefix);
                        check(allowed(*a, shared, *t, stop).words == allowed(plain, fresh, *t, stop).words, "shared program mask differs");
                        check(shared.complete() == fresh.complete(), "shared program completion differs");
                        ++masks;
                    }
        // Cursors of two requests on one program do not see each other.
        ConstraintCursor first(*a), second(*a);
        first = advance_token(first, "{\"name\":\"a\"");
        check(allowed(*a, second, trie).words == allowed(plain, ConstraintCursor(plain), trie).words, "cursor state leaked between requests");
        // A schema that does not compile throws the compiler's error every time and leaves no entry.
        const Json invalid = parse_json(R"({"type":"object","properties":{"a":{"$ref":"#/$defs/missing"}}})");
        for (int i = 0; i < 2; ++i) {
            bool threw = false;
            try { (void)cache.json_schema(invalid); } catch (const SchemaError&) { threw = true; }
            check(threw, "invalid schema must throw SchemaError");
        }
        stats = cache.stats();
        check(stats.failures == 2 && stats.programs == 5, "failed compilation was kept");
        cache.clear();
        check(cache.stats().programs == 0 && cache.json_schema(spec) != a, "clear drops entries");
        check(allowed(*a, ConstraintCursor(*a), trie).words == allowed(plain, ConstraintCursor(plain), trie).words, "held program after clear");
        std::cout << "program cache masks=" << masks << " (two tries, EOS overlay, three passes)\n";
    }
    {
        ConstraintProgramCacheConfig small; small.max_programs = 3;
        ConstraintProgramCache cache(small);
        std::vector<std::shared_ptr<const ConstraintProgram>> held;
        for (int i = 0; i < 5; ++i) held.push_back(cache.json_schema(parse_json("{\"const\":" + std::to_string(i) + "}")));
        auto stats = cache.stats();
        check(stats.programs == 3 && stats.evictions == 2, "entry bound");
        check(cache.json_schema(parse_json("{\"const\":4}")) == held[4] && cache.json_schema(parse_json("{\"const\":0}")) != held[0], "least recently used leaves first");
        check(accepts(*held[0], "0") && !accepts(*held[0], "1"), "evicted program still valid for its holder");
        ConstraintProgramCacheConfig off; off.max_programs = 0;
        ConstraintProgramCache none(off);
        check(none.json_schema(spec) != none.json_schema(spec) && none.stats().uncached == 2 && none.stats().programs == 0, "sharing off");
        ConstraintProgramCacheConfig narrow; narrow.key_bytes = 256;
        ConstraintProgramCache bypass(narrow);
        check(bypass.json_schema(spec) != bypass.json_schema(spec) && bypass.stats().uncached == 2, "oversized schema text bypasses the cache");
        ConstraintLimits tiny; tiny.schema_bytes = 8;
        bool refused = false;
        try { (void)bypass.json_schema(spec, tiny); } catch (const SchemaError&) { refused = true; }
        check(refused, "schema size limit applies before the cache");
    }
    {
        // Eight requests for one new schema at once: one compilation, one program.
        ConstraintProgramCache cache;
        std::vector<std::shared_ptr<const ConstraintProgram>> got(8);
        std::vector<std::thread> threads;
        std::atomic<int> ready{0};
        for (std::size_t i = 0; i < got.size(); ++i)
            threads.emplace_back([&, i] {
                ++ready;
                while (ready.load() < 8) std::this_thread::yield();
                got[i] = cache.json_schema(spec);
                (void)allowed(*got[i], ConstraintCursor(*got[i]), trie, eos); // the mask memo under concurrent use
            });
        for (auto& thread : threads) thread.join();
        const auto stats = cache.stats();
        bool same = true;
        for (const auto& program : got) same = same && program == got[0];
        check(same && stats.misses == 1 && stats.hits + stats.joined == 7, "concurrent identical requests compiled more than once");
    }
}
// --bench-program-cache: compile + cursor + first mask per request for repeated identical-schema requests,
// per request (previous) against the shared cache, with a 151000-token vocabulary. Not a test.
int program_cache_bench() {
    using Clock = std::chrono::steady_clock;
    const auto ms = [](Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); };
    std::vector<std::string> bytes; bytes.reserve(151000);
    for (unsigned i = 0; i < 151000; ++i) {
        switch (i % 4) {
        case 0: bytes.push_back(std::to_string(i)); break;
        case 1: bytes.push_back(" word" + std::to_string(i)); break;
        case 2: bytes.push_back("\"key" + std::to_string(i) + "\":"); break;
        default: bytes.push_back(i < 1024 ? std::string(1, static_cast<char>(i / 4)) : "}" + std::to_string(i)); break;
        }
    }
    const TokenByteTrie trie(bytes);
    Json properties = Json::object(), required = Json::array();
    for (unsigned i = 0; i < 24; ++i) {
        const std::string name = "field_" + std::to_string(i);
        properties.set(name, parse_json(i % 3 == 0 ? R"({"type":"string","maxLength":64})"
                                        : i % 3 == 1 ? R"({"enum":["low","medium","high","critical"]})"
                                                     : R"({"type":"object","properties":{"id":{"type":"integer"},"ok":{"type":"boolean"}},"required":["id"],"additionalProperties":false})"));
        if (i % 2 == 0) required.push(Json(name));
    }
    Json spec = Json::object();
    spec.set("type", Json("object")); spec.set("properties", properties); spec.set("required", required); spec.set("additionalProperties", Json(false));
    const std::vector<std::uint32_t> eos{7};
    constexpr int requests = 20;
    bool identical = true;
    std::vector<std::uint64_t> reference, string_reference;
    for (const bool shared : {false, true, false, true}) {
        ConstraintProgramCache cache;
        double compile_first = 0, mask_first = 0, string_first = 0, compile_rest = 0, mask_rest = 0, string_rest = 0;
        for (int request = 0; request < requests; ++request) {
            const auto begin = Clock::now();
            const std::shared_ptr<const ConstraintProgram> program =
                shared ? cache.json_schema(spec) : std::make_shared<const ConstraintProgram>(compile_json_schema(spec));
            const auto compiled = Clock::now();
            const auto mask = allowed(*program, ConstraintCursor(*program), trie, eos);
            const auto done = Clock::now();
            // The first mask inside a string value: the state whose walk visits most of the vocabulary.
            const auto in_string = allowed(*program, advance_token(ConstraintCursor(*program), "{\"field_0\":\""), trie, eos);
            const auto string_done = Clock::now();
            if (reference.empty()) { reference = mask.words; string_reference = in_string.words; }
            identical = identical && mask.words == reference && in_string.words == string_reference;
            (request == 0 ? string_first : string_rest) += ms(string_done - done);
            (request == 0 ? compile_first : compile_rest) += ms(compiled - begin);
            (request == 0 ? mask_first : mask_rest) += ms(done - compiled);
        }
        const auto stats = cache.stats();
        std::cout << "program cache bench " << (shared ? "shared     " : "per-request") << " schema_bytes " << spec.dump().size()
                  << " request1 compile_ms " << compile_first << " first_mask_ms " << mask_first << " string_mask_ms " << string_first
                  << " | requests2.." << requests << " mean compile_ms " << compile_rest / (requests - 1) << " first_mask_ms "
                  << mask_rest / (requests - 1) << " string_mask_ms " << string_rest / (requests - 1) << " total_ms "
                  << (compile_rest + mask_rest + string_rest) / (requests - 1) << " | hits " << stats.hits << " misses " << stats.misses << '\n';
    }
    std::cout << "first and string-state masks " << (identical ? "identical" : "DIFFER") << " in every arm\n";
    return identical ? 0 : 1;
}
int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--bench-program-cache") return program_cache_bench();
    program_cache_allocation_failure(); program_cache_pending_bounds(); program_cache_joiners(); program_cache_environment();
    program_cache();
    branching_expansion(); literal_reference_chain(); adversarial_compilation(); profile(); language(); tokens(); raw_oracle(); random_walks(); equivalence(); limits(); productivity(); exhaustive_lexemes(); cache_exactness(); benchmark();
    std::cout << "json constraint tests: " << failures << " failures\n"; return failures ? 1 : 0;
}
