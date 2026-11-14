#include "strix/serve/json_constraint.hpp"

#include "strix/env_flags.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <functional>
#include <future>
#include <limits>
#include <map>
#include <set>
#include <list>
#include <mutex>
#include <unordered_map>

namespace strix::serve {
SchemaError::SchemaError(std::string p, std::string k, std::string message)
    : std::runtime_error(std::move(message)), path(std::move(p)), keyword(std::move(k)) {}
namespace constraint_detail {
struct CompilationBudget {
    const ConstraintLimits& limits;
    std::size_t steps = 0, literals = 0, tries = 0, alternatives = 0;
    ConstraintCompilationStats* stats = nullptr;
    std::size_t expansion_depth = 0;
    static void charge(std::size_t& used, std::size_t amount, std::size_t limit) {
        if (amount > limit || used > limit - amount) error_limit();
        used += amount;
    }
    [[noreturn]] static void error_limit() { throw SchemaError("limits", "limits", "aggregate compilation resource limit"); }
    void step(std::size_t n = 1) {
        charge(steps, n, limits.compilation_steps);
        if (stats) stats->compilation_steps = steps;
    }
    void literal(std::size_t n) { step(n); charge(literals, n, limits.literal_bytes); }
    void trie() { step(); charge(tries, 1, limits.trie_nodes); }
    void alternative() { step(); charge(alternatives, 1, limits.alternative_expansions); }
};
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
    auto operator<=>(const Node&) const = default;
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
    std::size_t consumed() const { return at_; }
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
namespace constraint_detail { struct PlanSet; struct MaskCache; }
using namespace constraint_detail;
struct ConstraintData {
    ConstraintLimits limits;
    std::vector<Node> nodes{Node{}};
    std::size_t root = 0;
    std::shared_ptr<const constraint_detail::PlanSet> plan;
    std::shared_ptr<constraint_detail::MaskCache> cache;
};
namespace constraint_detail {
class Compiler {
public:
    Compiler(const ConstraintLimits& limits, CompilationBudget& b) : budget(b) { data->limits = limits; }
    CompilationBudget& budget;
    std::map<std::string, std::size_t> intern;
    std::map<Node, std::size_t> canonical;
    std::size_t stack = 0;
    void name(const std::string& text, const std::string& path, const std::string& keyword) {
        budget.literal(text.size());
        std::string quoted; append_json_string(quoted, text);
        auto limits = data->limits; limits.max_string_length = SIZE_MAX;
        try { (void)PrefixParser(quoted, limits).root(); }
        catch (const Invalid&) { error(path, keyword, "invalid Unicode name"); }
    }
    std::shared_ptr<ConstraintData> data = std::make_shared<ConstraintData>();
    std::map<std::string, std::size_t> defs;
    std::map<std::string, const Json*> sources;
    std::set<std::string> active;
    std::size_t definition(const std::string& name, const std::string& path) {
        if (active.contains(name)) error(path, "$ref", "recursive reference");
        if (active.size() >= std::min<std::size_t>(256, data->limits.depth)) error(path, "$ref", "reference depth limit");
        if (defs.contains(name)) return defs.at(name);
        if (!sources.contains(name)) error(path, "$ref", "unknown local reference");
        active.insert(name);
        const auto id = node(*sources.at(name), join("$defs", name), 1);
        active.erase(name); defs[name] = id; return id;
    }
    std::size_t node(const Json& schema, const std::string& path, std::size_t depth) {
        budget.step();
        if (++stack > 256) error(path, "limits", "schema compiler stack limit");
        struct StackGuard { std::size_t& size; ~StackGuard() { --size; } } guard{stack};
        if (depth > std::min<std::size_t>(256, data->limits.depth) || data->nodes.size() > data->limits.nodes) error(path, "limits", "schema resource limit");
        const auto identity = schema.dump(); budget.step(identity.size());
        if (auto it = intern.find(identity); it != intern.end()) return it->second;
        const auto id = data->nodes.size(); data->nodes.emplace_back(); Node n;
        if (schema.is_bool()) { n.yes = schema.as_bool(); data->nodes[id] = n; canonical.emplace(n, id); intern.emplace(identity, id); return id; }
        if (!schema.is_object()) error(path, "schema", "expected schema object or boolean");
        const auto size_value = [&](const Json& x, const std::string& k) {
            if (!x.is_number() || x.number_source().find_first_not_of("0123456789") != std::string::npos) error(path, k, "expected nonnegative integer");
            try { return static_cast<std::size_t>(std::stoull(x.number_source())); }
            catch (const std::exception&) { error(path, k, "integer out of range"); }
        };
        for (const auto& [k, x] : schema.as_object()) {
            budget.step();
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
                for (const auto& [key, sub] : x.as_object()) { name(key, path, k); n.properties.emplace_back(key, node(sub, join(join(path, k), key), depth + 1)); }
            } else if (k == "required") {
                if (!x.is_array() || x.size() > data->limits.properties) error(path, k, "required limit or shape");
                for (const auto& item : x.as_array()) {
                    if (!item.is_string() || !n.required.insert(item.as_string()).second) error(path, k, "invalid required name");
                    name(item.as_string(), path, k);
                }
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
                    for (const auto& item : x.as_array()) { auto bytes = item.dump(); budget.literal(bytes.size()); values.push_back(std::move(bytes)); }
                } else { auto bytes = x.dump(); budget.literal(bytes.size()); values.push_back(std::move(bytes)); }
                std::sort(values.begin(), values.end(), [&](const auto& a, const auto& b) { budget.literal(std::min(a.size(), b.size())); return a < b; });
                values.erase(std::unique(values.begin(), values.end(), [&](const auto& a, const auto& b) { budget.literal(std::min(a.size(), b.size())); return a == b; }), values.end());
                if (n.literal_set) {
                    std::vector<std::string> common;
                    for (const auto& a : n.literals) { budget.literal(a.size()); if (std::binary_search(values.begin(), values.end(), a, [&](const auto& lhs, const auto& rhs) { budget.literal(std::min(lhs.size(), rhs.size())); return lhs < rhs; })) common.push_back(a); }
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
        std::sort(n.alternatives.begin(), n.alternatives.end());
        n.alternatives.erase(std::unique(n.alternatives.begin(), n.alternatives.end()), n.alternatives.end());
        std::sort(n.properties.begin(), n.properties.end());
        if (auto it = canonical.find(n); it != canonical.end()) { intern.emplace(identity, it->second); return it->second; }
        canonical.emplace(n, id); data->nodes[id] = std::move(n); intern.emplace(identity, id); return id;
    }
};
std::size_t key_spelling(std::string_view rest, std::string_view pending, std::span<const std::size_t> suffix) {
    if (pending.empty()) return suffix[0];
    if (rest.empty()) return SIZE_MAX / 4;
    unsigned cp = static_cast<unsigned char>(rest[0]); std::size_t len = 1;
    if (cp >= 128) {
        len = cp < 0xe0 ? 2 : cp < 0xf0 ? 3 : 4;
        cp &= len == 2 ? 31U : len == 3 ? 15U : 7U;
        if (rest.size() < len) return SIZE_MAX / 4;
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
    std::size_t best = SIZE_MAX / 4;
    for (const auto& form : forms) {
        if (form.size() < pending.size()) continue;
        bool same = true;
        for (std::size_t i = 0; i < pending.size(); ++i) {
            char b = pending[i];
            if (form.starts_with("\\u") && b >= 'A' && b <= 'F') b = static_cast<char>(b - 'A' + 'a');
            if (b != form[i]) { same = false; break; }
        }
        if (same) {
            best = std::min(best, form.size() - pending.size() + suffix[len]);
        }
    }
    return best;
}
// Literal validation follows references and nesting natively: bounded like the compiler's own stacks.
bool matches(const ConstraintData& d, std::size_t id, const Value& v, std::string_view bytes, bool final, CompilationBudget& budget,
    std::size_t depth = 0) {
    if (depth > 256) throw SchemaError("limits", "limits", "literal validation depth limit");
    budget.step(); budget.literal(v.end - v.begin);
    const auto& n = d.nodes[id];
    if (!n.yes) return false;
    if (n.ref != SIZE_MAX && !matches(d, n.ref, v, bytes, final, budget, depth + 1)) return false;
    if (!n.alternatives.empty() && std::none_of(n.alternatives.begin(), n.alternatives.end(), [&](auto a) { return matches(d, a, v, bytes, final, budget, depth + 1); })) return false;
    if (v.kind == 0) return true;
    const std::string type = v.kind == '{' ? "object" : v.kind == '[' ? "array" : v.kind == 's' ? "string" : v.kind == 'n' ? "number" : v.kind == 'b' ? "boolean" : "null";
    if (!n.types.empty() && !n.types.contains(type)) {
        if (v.kind != 'n' || !n.types.contains("integer")) return false;
        const auto raw = bytes.substr(v.begin, (v.done ? v.end : bytes.size()) - v.begin);
        if (raw.find_first_of(".eE") != std::string_view::npos) return false;
    }
    if (n.literal_set) {
        const auto raw = bytes.substr(v.begin, v.end - v.begin);
        const auto at = std::lower_bound(n.literals.begin(), n.literals.end(), raw, [&](const auto& literal, auto candidate) {
            budget.literal(std::min(literal.size(), candidate.size())); return std::string_view(literal) < candidate;
        });
        if (at == n.literals.end()) return false;
        budget.literal(std::min(at->size(), raw.size()));
        if (*at != raw) return false;
    }
    if (v.kind == 's' && (v.scalars > n.max_length || (v.done && v.scalars < n.min_length))) return false;
    if (v.kind == '[') {
        if ((v.expect_more && v.children.size() >= n.max_items) || v.children.size() > n.max_items || (v.done && v.children.size() < n.min_items)) return false;
        for (const auto& child : v.children) if (!matches(d, n.items, child, bytes, final, budget, depth + 1)) return false;
    }
    if (v.kind == '{') {
        for (std::size_t i = 0; i < v.keys.size(); ++i) {
            const auto p = std::find_if(n.properties.begin(), n.properties.end(), [&](const auto& x) { return x.first == v.keys[i]; });
            if (p == n.properties.end() && !n.additional) return false;
            if (i < v.children.size() && !matches(d, p == n.properties.end() ? 0 : p->second, v.children[i], bytes, final, budget, depth + 1)) return false;
        }
        if (v.done) for (const auto& key : n.required) if (std::find(v.keys.begin(), v.keys.end(), key) == v.keys.end()) return false;
    }
    return true;
}
}  // namespace
// Compiled plans and incremental pushdown machine. No generated output or
// parsed value tree is retained by a cursor.
namespace constraint_detail {
constexpr std::size_t infinity = SIZE_MAX / 4;
std::size_t plus(std::size_t a, std::size_t b) { return a >= infinity || b >= infinity || a > infinity - b ? infinity : a + b; }
std::size_t times(std::size_t a, std::size_t b) { return !a || !b ? 0 : a >= infinity || b >= infinity || a > infinity / b ? infinity : a * b; }
struct ByteTrie {
    struct Entry {
        std::vector<std::pair<unsigned char, std::size_t>> edges;
        std::vector<std::size_t> descendants;
        std::size_t terminal = SIZE_MAX, depth = 0, suffix = infinity;
        char kind = 0;
    };
    std::vector<Entry> nodes{1};
    std::size_t next(std::size_t at, unsigned char b) const {
        if (at == SIZE_MAX) return SIZE_MAX;
        for (auto [byte, to] : nodes[at].edges) if (byte == b) return to;
        return SIZE_MAX;
    }
    void insert(std::string_view text, std::size_t id, CompilationBudget& budget, char kind = 0) {
        budget.literal(text.size());
        std::size_t at = 0;
        nodes[at].descendants.push_back(id);
        for (const char b_char : text) { unsigned char b = static_cast<unsigned char>(b_char);
            auto to = next(at, b);
            if (to == SIZE_MAX) {
                budget.trie(); to = nodes.size();
                const auto depth = nodes[at].depth + 1;
                nodes[at].edges.emplace_back(b, to);
                nodes.emplace_back(); nodes.back().depth = depth;
            }
            at = to; nodes[at].descendants.push_back(id);
        }
        nodes[at].terminal = id; nodes[at].kind = kind;
        nodes[at].suffix = 0;
    }
    void seal() {
        for (std::size_t i = nodes.size(); i-- > 0;)
            for (auto [byte, to] : nodes[i].edges) { (void)byte; nodes[i].suffix = std::min(nodes[i].suffix, plus(1, nodes[to].suffix)); }
    }
};
struct Plan {
    char kind = 0;
    bool integer = false, additional = true;
    std::size_t minimum = 0, maximum = SIZE_MAX, bytes = infinity, required_cost = 0;
    std::shared_ptr<const PlanSet> items, unknown;
    struct Property {
        std::string name;
        std::shared_ptr<const PlanSet> value;
        bool required = false;
        std::size_t scalars = 0, spelling = 0, required_cost = 0;
        std::vector<std::size_t> suffix{};
    };
    std::vector<Property> properties;
    ByteTrie trie;
};
struct PlanSet { std::vector<std::shared_ptr<const Plan>> alternatives; std::size_t bytes = infinity; };

class PlanCompiler {
public:
    PlanCompiler(const ConstraintData& data, CompilationBudget& b) : d(data), budget(b) {}
    std::shared_ptr<const PlanSet> build(std::vector<std::size_t> ids, std::size_t depth) {
        const auto root = request(std::move(ids), depth);
        std::vector<std::shared_ptr<Build>> stack;
        if (!root->done) stack.push_back(root);
        while (!stack.empty()) {
            auto task = stack.back();
            if (!task->prepared) {
                task->conjunctions = expand(std::move(task->ids));
                dependencies(*task);
                task->prepared = true;
            }
            while (task->next < task->children.size() && task->children[task->next]->done) ++task->next;
            if (task->next < task->children.size()) { stack.push_back(task->children[task->next]); continue; }
            for (auto& own : task->conjunctions) conjunction(std::move(own), task->depth, *task->result);
            task->done = true;
            task->conjunctions.clear(); task->children.clear();
            stack.pop_back();
        }
        return root->result;
    }
private:
    struct Build {
        std::vector<std::size_t> ids;
        std::size_t depth = 0, next = 0;
        std::shared_ptr<PlanSet> result = std::make_shared<PlanSet>();
        std::vector<std::vector<std::size_t>> conjunctions;
        std::vector<std::shared_ptr<Build>> children;
        bool prepared = false, done = false;
    };
    const ConstraintData& d;
    std::map<std::vector<std::size_t>, std::shared_ptr<Build>> memo;
    CompilationBudget& budget;
    std::map<std::vector<std::size_t>, std::shared_ptr<PlanSet>> compiled;
    std::map<std::vector<std::string>, std::shared_ptr<Plan>> literal_plans;
    std::shared_ptr<Build> request(std::vector<std::size_t> ids, std::size_t depth) {
        budget.step(ids.size() + 1);
        std::sort(ids.begin(), ids.end()); ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        auto key = ids; key.push_back(SIZE_MAX); key.push_back(depth);
        if (auto it = memo.find(key); it != memo.end()) return it->second;
        auto task = std::make_shared<Build>(); task->ids = std::move(ids); task->depth = depth;
        memo.emplace(std::move(key), task);
        if (depth > d.limits.depth) { task->done = true; return task; }
        if (depth > 256) throw ConstraintError("constraint plan stack resource limit");
        return task;
    }
    std::shared_ptr<const PlanSet> child(std::vector<std::size_t> ids, std::size_t depth) {
        auto task = request(std::move(ids), depth);
        if (!task->done) throw ConstraintError("constraint plan dependency incomplete");
        return task->result;
    }
    // Discover child plans before emitting this value's plans. Dependencies
    // strictly increase value depth, and the work stack finishes them first.
    void dependencies(Build& task) {
        const auto add = [&](std::vector<std::size_t> ids) {
            budget.step(); task.children.push_back(request(std::move(ids), task.depth + 1));
        };
        for (const auto& own : task.conjunctions) {
            budget.step(own.size() + 1);
            if (std::any_of(own.begin(), own.end(), [&](auto id) { return d.nodes[id].literal_set; })) continue;
            const auto permits = [&](const char* type) {
                return std::all_of(own.begin(), own.end(), [&](auto id) {
                    const auto& types = d.nodes[id].types;
                    return types.empty() || types.contains(type);
                });
            };
            if (permits("array")) {
                std::vector<std::size_t> ids; std::size_t minimum = 0, maximum = SIZE_MAX;
                for (auto id : own) {
                    const auto& n = d.nodes[id]; ids.push_back(n.items);
                    minimum = std::max(minimum, n.min_items); maximum = std::min(maximum, n.max_items);
                }
                if (minimum <= maximum) add(std::move(ids));
            }
            if (!permits("object")) continue;
            std::set<std::string> names;
            for (auto id : own) {
                const auto& n = d.nodes[id];
                for (const auto& [name, ignored] : n.properties) { (void)ignored; names.insert(name); }
                names.insert(n.required.begin(), n.required.end());
            }
            for (const auto& name : names) {
                budget.literal(name.size());
                std::vector<std::size_t> ids; bool permitted = true;
                for (auto id : own) {
                    const auto& n = d.nodes[id];
                    const auto it = std::find_if(n.properties.begin(), n.properties.end(), [&](const auto& x) { return x.first == name; });
                    if (it != n.properties.end()) ids.push_back(it->second);
                    else if (!n.additional) permitted = false;
                }
                if (permitted) add(std::move(ids));
            }
            add({});
        }
    }
    void reserve_plan() { budget.alternative(); budget.trie(); }
    std::vector<std::vector<std::size_t>> expand(std::vector<std::size_t> ids) {
        ++budget.expansion_depth;
        struct Guard { std::size_t& depth; ~Guard() { --depth; } } guard{budget.expansion_depth};
        if (budget.stats) budget.stats->expansion_recursion_high_water =
            std::max(budget.stats->expansion_recursion_high_water, budget.expansion_depth);
        struct Work {
            std::vector<std::size_t> own, pending;
            std::set<std::size_t> expanded;
        };
        std::vector<Work> stack;
        std::vector<std::vector<std::size_t>> result;
        budget.alternative(); budget.step(ids.size() + 1);
        stack.push_back({{}, std::move(ids), {}});
        while (!stack.empty()) {
            if (budget.stats) budget.stats->expansion_stack_high_water =
                std::max(budget.stats->expansion_stack_high_water, stack.size());
            auto branch = std::move(stack.back()); stack.pop_back();
            bool finished = true;
            while (!branch.pending.empty()) {
                budget.step();
                const auto id = branch.pending.back(); branch.pending.pop_back();
                // Repeating a conjunct cannot further constrain this branch.
                // Charge even duplicate visits, but never expand them again.
                if (!branch.expanded.insert(id).second) continue;
                const auto& n = d.nodes[id];
                if (!n.yes) { finished = false; break; }
                if (!n.types.empty() || n.literal_set || !n.properties.empty() || !n.required.empty() ||
                    !n.additional || n.items || n.min_items || n.max_items != SIZE_MAX ||
                    n.min_length || n.max_length != SIZE_MAX) branch.own.push_back(id);
                if (n.ref != SIZE_MAX) { budget.step(); branch.pending.push_back(n.ref); }
                if (n.alternatives.empty()) continue;
                for (auto alt : n.alternatives) {
                    budget.alternative();
                    budget.step(branch.own.size() + branch.pending.size() + branch.expanded.size() + 1);
                    auto more = branch; more.pending.push_back(alt);
                    stack.push_back(std::move(more));
                }
                finished = false; break;
            }
            if (finished) { budget.step(branch.own.size() + 1); result.push_back(std::move(branch.own)); }
        }
        return result;
    }
    void conjunction(std::vector<std::size_t> own, std::size_t depth, PlanSet& out) {
        std::sort(own.begin(), own.end()); own.erase(std::unique(own.begin(), own.end()), own.end());
        // A union can reach the same conjunction through many references.
        // Share its plans/trie rather than rebuilding or repeating runtime branches.
        budget.step(own.size() + 2);
        auto identity = own; identity.push_back(SIZE_MAX); identity.push_back(depth);
        if (auto it = compiled.find(identity); it != compiled.end()) {
            for (const auto& p : it->second->alternatives)
                if (std::find(out.alternatives.begin(), out.alternatives.end(), p) == out.alternatives.end()) { out.alternatives.push_back(p); out.bytes = std::min(out.bytes, p->bytes); }
            return;
        }
        auto result = std::make_shared<PlanSet>(); compiled.emplace(std::move(identity), result);
        const auto emit = [&](std::shared_ptr<Plan> p) {
            p->trie.seal();
            switch (p->kind) {
            case 'L': p->bytes = p->trie.nodes[0].suffix; break;
            case 'n': p->bytes = 1; break;
            case '0': case 'b': p->bytes = 4; break;
            case 's': p->bytes = plus(2, p->minimum); break;
            case '[': p->bytes = plus(2, plus(times(p->minimum, p->items->bytes), p->minimum ? p->minimum - 1 : 0)); break;
            case '{': {
                p->bytes = 2;
                for (auto& property : p->properties) {
                    for (const char b_char : property.name) { unsigned char b = static_cast<unsigned char>(b_char); property.scalars += (b & 0xc0U) != 0x80U; }
                    property.suffix.resize(property.name.size() + 1); property.suffix.back() = 1;
                    for (std::size_t i = property.name.size(); i-- > 0;) {
                        const auto b = static_cast<unsigned char>(property.name[i]);
                        const auto cost = b == '"' || b == '\\' ? 2U : b >= 32 ? 1U : b == '\b' || b == '\f' || b == '\n' || b == '\r' || b == '\t' ? 2U : 6U;
                        property.suffix[i] = property.suffix[i + 1] + cost;
                    }
                    property.spelling = property.suffix[0] + 1;
                    if (property.required) {
                        // Include a leading comma; the first member's comma is
                        // removed by required_tail() or the complete plan cost.
                        property.required_cost = plus(property.spelling + 2, property.value->bytes);
                        p->required_cost = plus(p->required_cost, property.required_cost);
                    }
                }
                if (p->required_cost) p->bytes = plus(1, p->required_cost);
                break;
            }
            }
            if (p->bytes >= infinity || p->bytes > d.limits.max_output_bytes) return;
            result->bytes = std::min(result->bytes, p->bytes); out.bytes = std::min(out.bytes, p->bytes);
            result->alternatives.push_back(p);
            if (std::find(out.alternatives.begin(), out.alternatives.end(), p) == out.alternatives.end()) out.alternatives.push_back(std::move(p));
        };
        for (auto id : own) if (d.nodes[id].literal_set) {
            std::vector<std::string> witnesses;
            for (const auto& literal : d.nodes[id].literals) {
                budget.literal(literal.size());
                try {
                    auto limits = d.limits; limits.depth -= depth;
                    const auto v = PrefixParser(literal, limits).root();
                    if (v.done && std::all_of(own.begin(), own.end(), [&](auto x) {
                        return matches(d, x, v, literal, true, budget);
                    })) witnesses.push_back(literal);
                } catch (const Invalid&) {}
            }
            if (witnesses.empty()) return;
            // Sibling constraints can differ yet select the same canonical
            // language. Intern the resulting literal plan across all builds.
            if (auto it = literal_plans.find(witnesses); it != literal_plans.end()) { emit(it->second); return; }
            reserve_plan(); auto p = std::make_shared<Plan>(); p->kind = 'L';
            for (const auto& literal : witnesses) {
                const char first = literal[0];
                const char kind = first == '"' ? 's' : first == '[' || first == '{' ? first : first == 'n' ? '0' : first == 't' || first == 'f' ? 'b' : 'n';
                p->trie.insert(literal, 0, budget, kind);
            }
            literal_plans.emplace(std::move(witnesses), p); emit(std::move(p));
            return;
        }
        for (const char kind : {'0', 'b', 'n', 's', '[', '{'}) {
            budget.step();
            const std::string type = kind == '0' ? "null" : kind == 'b' ? "boolean" : kind == 'n' ? "number" : kind == 's' ? "string" : kind == '[' ? "array" : "object";
            if (std::any_of(own.begin(), own.end(), [&](auto id) {
                const auto& types = d.nodes[id].types;
                return !types.empty() && !types.contains(type) && !(kind == 'n' && types.contains("integer"));
            })) continue;
            reserve_plan();
            auto p = std::make_shared<Plan>(); p->kind = kind;
            if (kind == 'n') for (auto id : own) {
                const auto& types = d.nodes[id].types;
                if (!types.empty() && !types.contains("number") && types.contains("integer")) p->integer = true;
            }
            if (kind == 's' || kind == '[') {
                if (kind == 's') p->maximum = d.limits.max_string_length;
                std::vector<std::size_t> children;
                for (auto id : own) {
                    const auto& n = d.nodes[id];
                    p->minimum = std::max(p->minimum, kind == 's' ? n.min_length : n.min_items);
                    p->maximum = std::min(p->maximum, kind == 's' ? n.max_length : n.max_items);
                    children.push_back(n.items);
                }
                if (p->minimum > p->maximum) continue;
                if (kind == '[') {
                    p->items = child(std::move(children), depth + 1);
                    if (p->items->alternatives.empty()) { if (p->minimum) continue; p->maximum = 0; }
                }
            }
            if (kind == '{') {
                std::set<std::string> names;
                for (auto id : own) {
                    const auto& n = d.nodes[id]; p->additional &= n.additional;
                    for (const auto& [name, ignored] : n.properties) { (void)ignored; names.insert(name); }
                    names.insert(n.required.begin(), n.required.end());
                }
                bool viable = true;
                for (const auto& name : names) {
                    budget.literal(name.size());
                    std::vector<std::size_t> sub; bool permitted = true, required = false;
                    for (auto id : own) {
                        const auto& n = d.nodes[id]; required |= n.required.contains(name);
                        const auto it = std::find_if(n.properties.begin(), n.properties.end(), [&](const auto& x) { return x.first == name; });
                        if (it != n.properties.end()) sub.push_back(it->second);
                        else if (!n.additional) permitted = false;
                    }
                    auto child_plan = permitted ? child(std::move(sub), depth + 1) : std::make_shared<PlanSet>();
                    std::size_t scalars = 0;
                    for (const char byte_char : name) { unsigned char byte = static_cast<unsigned char>(byte_char); scalars += (byte & 0xc0U) != 0x80U; }
                    if (scalars > d.limits.max_string_length || child_plan->alternatives.empty()) {
                        if (required) viable = false;
                        // Keep an empty value plan for a prohibited declared name.
                        p->trie.insert(name, p->properties.size(), budget);
                        p->properties.push_back({name, std::make_shared<PlanSet>(), required});
                        continue;
                    }
                    p->trie.insert(name, p->properties.size(), budget);
                    p->properties.push_back({name, std::move(child_plan), required});
                }
                if (!viable) continue;
                p->unknown = child({}, depth + 1);
                p->additional &= !p->unknown->alternatives.empty();
            }
            emit(std::move(p));
        }
    }
};

// Strict scalar decoder. The bounded pending spelling permits early key-trie
// pruning even in the middle of UTF-8 and UTF-16 surrogate escape sequences.
struct StringState {
    enum Mode : unsigned char { plain, escape, hex, low_slash, low_u, low_hex, continuation };
    Mode mode = plain;
    unsigned value = 0, high = 0, digits = 0, remaining = 0;
    unsigned char lead = 0;
    std::size_t scalars = 0;
    std::array<char, 12> pending{};
    unsigned pending_size = 0;
    bool feed(unsigned char b, std::size_t maximum, std::string& scalar, bool& closed) {
        closed = false; scalar.clear();
        if (mode == plain) {
            if (b == '"') { closed = true; return true; }
            if (b < 32 || scalars >= maximum) return false;
            ++scalars; pending_size = 0;
        }
        if (pending_size >= pending.size()) return false;
        pending[pending_size++] = static_cast<char>(b);
        const auto emit = [&] {
            if (value < 128) scalar += static_cast<char>(value);
            else if (value < 2048) { scalar += static_cast<char>(0xc0U | (value >> 6U)); scalar += static_cast<char>(0x80U | (value & 63U)); }
            else if (value < 65536) { scalar += static_cast<char>(0xe0U | (value >> 12U)); scalar += static_cast<char>(0x80U | ((value >> 6U) & 63U)); scalar += static_cast<char>(0x80U | (value & 63U)); }
            else { scalar += static_cast<char>(0xf0U | (value >> 18U)); scalar += static_cast<char>(0x80U | ((value >> 12U) & 63U)); scalar += static_cast<char>(0x80U | ((value >> 6U) & 63U)); scalar += static_cast<char>(0x80U | (value & 63U)); }
            mode = plain; pending_size = 0;
        };
        if (mode == plain) {
            value = b;
            if (b == '\\') { mode = escape; return true; }
            if (b < 128) { emit(); return true; }
            lead = b;
            if (b >= 0xc2 && b <= 0xdf) { remaining = 1; value &= 31U; }
            else if (b >= 0xe0 && b <= 0xef) { remaining = 2; value &= 15U; }
            else if (b >= 0xf0 && b <= 0xf4) { remaining = 3; value &= 7U; }
            else return false;
            mode = continuation; return true;
        }
        if (mode == continuation) {
            if (b < 0x80 || b > 0xbf || (pending_size == 2 && ((lead == 0xe0 && b < 0xa0) || (lead == 0xed && b > 0x9f) || (lead == 0xf0 && b < 0x90) || (lead == 0xf4 && b > 0x8f)))) return false;
            value = (value << 6U) | (b & 63U);
            if (--remaining == 0) emit();
            return true;
        }
        if (mode == escape) {
            if (b == 'u') { mode = hex; digits = 0; value = 0; return true; }
            const std::string_view escapes = "\"\\/bfnrt";
            const auto at = escapes.find(static_cast<char>(b));
            if (at == std::string_view::npos) return false;
            value = static_cast<unsigned char>(std::string_view("\"\\/\b\f\n\r\t")[at]); emit(); return true;
        }
        if (mode == low_slash) { if (b != '\\') return false; mode = low_u; return true; }
        if (mode == low_u) { if (b != 'u') return false; mode = low_hex; digits = 0; value = 0; return true; }
        unsigned nibble;
        if (b >= '0' && b <= '9') nibble = b - '0';
        else if (b >= 'a' && b <= 'f') nibble = b - 'a' + 10U;
        else if (b >= 'A' && b <= 'F') nibble = b - 'A' + 10U;
        else return false;
        value = (value << 4U) | nibble; ++digits;
        if (mode == hex && digits == 2 && value >= 0xdc && value <= 0xdf) return false;
        if (mode == low_hex && ((digits == 1 && value != 0xd) || (digits == 2 && (value < 0xdc || value > 0xdf)))) return false;
        if (digits != 4) return true;
        if (mode == hex && value >= 0xd800 && value <= 0xdbff) { high = value; mode = low_slash; return true; }
        if (mode == low_hex) value = 0x10000U + ((high - 0xd800U) << 10U) + value - 0xdc00U;
        emit(); return true;
    }
};

// Intern decoded additional-name prefixes in a persistent Patricia map. An edge
// is (prefix ID, next UTF-8 byte); a reserved byte marks a seen complete name.
// IDs are exact, not hashes. Lookup and path copying take at most 64 bit tests,
// regardless of key length or previously generated object members. This avoids
// both copying long key prefixes and rescanning a duplicate at its closing quote.
struct NameMap {
    std::uint64_t key = 0, value = 0;
    unsigned split = 64; // Leaf; internal splits strictly decrease down a path.
    std::shared_ptr<const NameMap> left, right;
};
using Names = std::shared_ptr<const NameMap>;
std::uint64_t name_lookup(Names tree, std::uint64_t key) {
    while (tree && tree->split != 64) tree = ((key >> tree->split) & 1U) ? tree->right : tree->left;
    return tree && tree->key == key ? tree->value : 0;
}
Names name_join(Names tree, Names leaf, unsigned split) {
    if (tree->split == 64 || tree->split < split) {
        auto node = std::make_shared<NameMap>(); node->split = split;
        if (((leaf->key >> split) & 1U) == 0) { node->left = leaf; node->right = tree; }
        else { node->left = tree; node->right = leaf; }
        return node;
    }
    auto node = std::make_shared<NameMap>(*tree);
    if (((leaf->key >> tree->split) & 1U) == 0) node->left = name_join(tree->left, leaf, split);
    else node->right = name_join(tree->right, leaf, split);
    return node;
}
Names name_insert(Names tree, std::uint64_t key, std::uint64_t value) {
    auto leaf = std::make_shared<NameMap>(); leaf->key = key; leaf->value = value;
    if (!tree) return leaf;
    auto found = tree;
    while (found->split != 64) found = ((key >> found->split) & 1U) ? found->right : found->left;
    if (found->key == key) return tree;
    const auto split = static_cast<unsigned>(std::bit_width(found->key ^ key) - 1);
    return name_join(tree, std::move(leaf), split);
}
struct ObjectState {
    std::vector<std::uint64_t> seen;
    std::size_t remaining_required_cost = 0;
    Names names;
    std::uint64_t next_name = 1;
    bool has_additional = false;
    bool contains(std::size_t i) const { return (seen[i / 64] & (std::uint64_t{1} << (i % 64))) != 0; }
};
struct Frame {
    enum Phase : unsigned char { start, string, number, literal, finite, key_or_close, key, colon, value, comma_or_close, after_comma };
    std::shared_ptr<const Plan> plan;
    std::shared_ptr<const Frame> parent;
    std::shared_ptr<const ObjectState> object;
    std::shared_ptr<const PlanSet> child;
    Phase phase = start;
    std::size_t count = 0, trie = 0, depth = 0, after = 0;
    unsigned char gap = 0, lex = 0, progress = 0;
    char word = 0;
    StringState text;
    std::uint64_t name = 0;
};
bool whitespace(unsigned char b) { return b == ' ' || b == '\n' || b == '\t'; }
bool number_done(unsigned char state) { return state == 2 || state == 3 || state == 5 || state == 8; }
bool number_byte(Frame& f, unsigned char b) {
    const bool digit = b >= '0' && b <= '9';
    switch (f.lex) {
    case 0: if (b == '-') { f.lex = 1; return true; } [[fallthrough]];
    case 1: if (b == '0') { f.lex = 2; return true; } if (b >= '1' && b <= '9') { f.lex = 3; return true; } return false;
    case 2: case 3:
        if (f.lex == 3 && digit) return true;
        if (f.plan->integer) return false;
        if (b == '.') { f.lex = 4; return true; }
        if (b == 'e' || b == 'E') { f.lex = 6; return true; }
        return false;
    case 4: if (digit) { f.lex = 5; return true; } return false;
    case 5: if (digit) return true; if (b == 'e' || b == 'E') { f.lex = 6; return true; } return false;
    case 6: if (b == '+' || b == '-') { f.lex = 7; return true; } [[fallthrough]];
    case 7: if (digit) { f.lex = 8; return true; } return false;
    case 8: return digit;
    default: return false;
    }
}
bool required_done(const Frame& f) {
    return f.object->remaining_required_cost == 0;
}
// Shortest suffix of a valid pending scalar. A scalar is charged on its
// first byte, so pending bytes do not consume another scalar slot.
std::size_t decoder_need(const StringState& text) {
    switch (text.mode) {
    case StringState::plain: return 0;
    case StringState::escape: return 1;
    case StringState::continuation: return text.remaining;
    case StringState::low_slash: return 6;
    case StringState::low_u: return 5;
    case StringState::low_hex: return 4 - text.digits;
    case StringState::hex:
        return 4 - text.digits + ((text.digits == 2 && text.value >= 0xd8 && text.value <= 0xdb) ||
                                  (text.digits == 3 && text.value >= 0xd80 && text.value <= 0xdbf) ? 6 : 0);
    }
    return infinity;
}
std::size_t required_tail(const Frame& f, std::size_t except = SIZE_MAX, bool comma = true) {
    auto cost = f.object->remaining_required_cost;
    if (except != SIZE_MAX && !f.object->contains(except)) cost -= f.plan->properties[except].required_cost;
    return 1 + cost - static_cast<std::size_t>(!comma && cost != 0); // closing brace
}
// Search only an additional name's spelling, never prior output. Once neither
// exact name trie has this prefix, every valid scalar completion is fresh.
// Iterative length bounds give the shortest spelling; work remains bounded by
// the transition budget even for a heavily populated name space.
bool unknown_name(const Frame& f, StringState text, std::size_t trie, std::uint64_t name,
                  std::size_t left, const ConstraintLimits& limits, std::size_t& work) {
    if (++work > limits.compilation_steps) throw ConstraintError("constraint key search resource limit");
    const auto need = decoder_need(text) + 1;
    if (need > left) return false;
    if (trie == SIZE_MAX && name == UINT64_MAX) return true;
    if (text.mode == StringState::plain) {
        const bool declared = trie != SIZE_MAX && f.plan->trie.nodes[trie].terminal != SIZE_MAX;
        const bool seen = name != UINT64_MAX && name_lookup(f.object->names, (name << 9U) | 256U);
        if (!declared && !seen) return true;
        if (text.scalars >= limits.max_string_length || left <= 1) return false;
    }
    // Printable scalars first, then escape/UTF-8/hex continuation bytes.
    for (unsigned i = 0; i < 256; ++i) {
        const unsigned char b = static_cast<unsigned char>(i < 95 ? i + 32 : i < 127 ? i - 95 : i);
        auto next = text; std::string scalar; bool closed;
        if (!next.feed(b, limits.max_string_length, scalar, closed) || closed) continue;
        auto to = trie; auto next_name = name;
        for (const char byte_char : scalar) { unsigned char byte = static_cast<unsigned char>(byte_char);
            to = f.plan->trie.next(to, byte);
            if (next_name != UINT64_MAX) {
                const auto id = name_lookup(f.object->names, (next_name << 9U) | byte);
                next_name = id ? id : UINT64_MAX;
            }
        }
        if (unknown_name(f, next, to, next_name, left - 1, limits, work)) return true;
    }
    return false;
}
std::size_t key_cost(const Frame& f, const ConstraintLimits& limits, std::size_t& work) {
    std::size_t best = infinity;
    if (f.trie != SIZE_MAX) {
        const auto& node = f.plan->trie.nodes[f.trie];
        for (auto id : node.descendants) {
            if (++work > limits.compilation_steps) throw ConstraintError("constraint key candidate resource limit");
            const auto& property = f.plan->properties[id];
            if (f.object->contains(id) || property.value->alternatives.empty() || property.scalars > limits.max_string_length) continue;
            const auto suffix = key_spelling(std::string_view(property.name).substr(node.depth),
                                           std::string_view(f.text.pending.data(), f.text.pending_size),
                                           std::span<const std::size_t>(property.suffix).subspan(node.depth));
            best = std::min(best, plus(suffix, plus(1, plus(property.value->bytes, required_tail(f, id)))));
        }
    }
    if (f.plan->additional) {
        const auto tail = plus(1, plus(f.plan->unknown->bytes, required_tail(f)));
        for (std::size_t bytes = decoder_need(f.text) + 1; plus(bytes, tail) < best && plus(bytes, tail) <= limits.max_output_bytes; ++bytes) {
            if (unknown_name(f, f.text, f.trie, f.name, bytes, limits, work)) { best = plus(bytes, tail); break; }
            if (f.text.mode == StringState::plain && f.text.scalars == limits.max_string_length) break;
        }
    }
    return best;
}
std::size_t remaining(const Frame& f, const ConstraintLimits& limits, std::size_t& work) {
    switch (f.phase) {
    case Frame::start: return f.plan->bytes;
    case Frame::finite: return f.plan->trie.nodes[f.trie].suffix;
    case Frame::number: return number_done(f.lex) ? 0 : 1;
    case Frame::literal: return (f.word == 'f' ? 5 : 4) - f.progress;
    case Frame::string: return plus(1 + decoder_need(f.text), f.plan->minimum > f.text.scalars ? f.plan->minimum - f.text.scalars : 0);
    default: break;
    }
    if (f.plan->kind == '{') {
        if (f.phase == Frame::key) return key_cost(f, limits, work);
        if (f.phase == Frame::colon || f.phase == Frame::value) return plus(f.phase == Frame::colon ? 1 : 0, plus(f.child->bytes, required_tail(f)));
        if (f.phase == Frame::after_comma) {
            auto key = f; key.text = {}; key.trie = 0; key.name = 0;
            return plus(1, key_cost(key, limits, work)); // opening quote
        }
        return required_tail(f, SIZE_MAX, f.count != 0);
    }
    const auto count = std::max(f.plan->minimum > f.count ? f.plan->minimum - f.count : 0, std::size_t(f.phase == Frame::after_comma));
    return plus(1, plus(times(count, f.plan->items->bytes), count ? count - 1 + std::size_t(f.phase == Frame::comma_or_close) : 0));
}
bool key_possible(const Frame& f, const ConstraintLimits& limits, std::size_t& work) { return key_cost(f, limits, work) < infinity; }
bool boundary(const Frame& f) {
    if (f.phase == Frame::number) return number_done(f.lex);
    if (f.phase == Frame::finite) return f.plan->trie.nodes[f.trie].terminal != SIZE_MAX;
    return false;
}
struct MaskCache {
    struct Entry {
        std::string key;
        std::shared_ptr<const unsigned char> trie;
        AllowedTokens mask;
        std::size_t charge;
    };
    std::mutex mutex;
    std::list<Entry> lru;
    std::unordered_multimap<std::size_t, std::list<Entry>::iterator> index;
    std::size_t used = 0, budget;
    ConstraintMemoStats totals;
    explicit MaskCache(std::size_t bytes) : budget(bytes) {}
    bool get(const std::string& key, AllowedTokens& mask) {
        std::lock_guard lock(mutex);
        const auto [first, last] = index.equal_range(std::hash<std::string>{}(key));
        for (auto it = first; it != last; ++it) if (it->second->key == key) {
            mask = it->second->mask;
            lru.splice(lru.begin(), lru, it->second); ++totals.hits; return true;
        }
        ++totals.misses;
        return false;
    }
    void put(std::string key, std::shared_ptr<const unsigned char> trie, const AllowedTokens& mask) {
        // Charge both payloads and a fixed allowance exceeding the list/index
        // node overhead. Oversized entries are never retained. No cursor or
        // dynamic name graph is retained by a cache entry.
        const auto charge = sizeof(Entry) + 256 + key.capacity() + mask.words.size() * sizeof(std::uint64_t);
        if (charge > budget) return;
        std::lock_guard lock(mutex);
        const auto hash = std::hash<std::string>{}(key);
        const auto [first, last] = index.equal_range(hash);
        for (auto it = first; it != last; ++it) if (it->second->key == key) return;
        while (used > budget - charge) {
            const auto victim = std::prev(lru.end());
            const auto [a, b] = index.equal_range(std::hash<std::string>{}(victim->key));
            for (auto it = a; it != b; ++it) if (it->second == victim) { index.erase(it); break; }
            used -= victim->charge; lru.erase(victim);
        }
        lru.push_front({std::move(key), std::move(trie), mask, charge});
        index.emplace(hash, lru.begin()); used += charge;
    }
};
}  // namespace

struct ConstraintState {
    std::vector<constraint_detail::Frame> branches;
    std::size_t bytes = 0;
    unsigned char trailing = 0;
    bool done = false;
};

namespace {
using namespace constraint_detail;
// Equality is byte-for-byte equality of the complete bounded serialization,
// including byte budget, every correlated branch/parent, lexer state and seen
// property bits. Hash collisions only select a bucket. Trie ownership prevents
// address reuse from confusing two vocabulary instances. Unknown names bypass
// the cache rather than replacing their exact identity with a digest.
std::string cache_key(const ConstraintState& state, const void* trie, std::size_t budget) {
    std::string key;
    const auto cap = std::min<std::size_t>(4096, budget / 4);
    bool fits = true;
    const auto append = [&](std::uint64_t x) {
        if (!fits || key.size() + sizeof(x) > cap) { fits = false; return; }
        for (unsigned i = 0; i < 8; ++i) key.push_back(static_cast<char>(x >> (i * 8U)));
    };
    append(reinterpret_cast<std::uintptr_t>(trie));
    append(state.bytes); append(state.trailing); append(state.done); append(state.branches.size());
    for (const auto& branch : state.branches) for (const Frame* f = &branch; f; f = f->parent.get()) {
        if (!fits || (f->object && f->object->names)) return {};
        append(reinterpret_cast<std::uintptr_t>(f->plan.get()));
        append(f->phase); append(f->count); append(f->trie); append(f->depth);
        append(f->after); append(f->gap); append(f->lex); append(f->progress); append(static_cast<unsigned char>(f->word));
        append(reinterpret_cast<std::uintptr_t>(f->child.get()));
        append(f->text.mode); append(f->text.value); append(f->text.high); append(f->text.digits);
        append(f->text.remaining); append(f->text.lead); append(f->text.scalars); append(f->text.pending_size);
        for (unsigned i = 0; i < f->text.pending_size; ++i) append(static_cast<unsigned char>(f->text.pending[i]));
        append(f->name);
        append(f->object != nullptr);
        if (f->object) {
            append(f->object->remaining_required_cost);
            append(f->object->seen.size());
            for (auto word : f->object->seen) append(word);
        }
        append(f->parent != nullptr);
    }
    return fits ? key : std::string{};
}
// A branch is one correlated conjunction of schema alternatives. Parent frames
// are immutable and shared. Completing a child restores its already-pruned
// parent; alternatives from different object members cannot be mixed together.
bool finish(Frame& f, ConstraintState& out) {
    if (!f.parent) { out.done = true; return true; }
    auto parent = f.parent;
    f = *parent;
    f.phase = Frame::comma_or_close; f.gap = 0; ++f.count;
    return false;
}
void consume(Frame f, unsigned char b, const ConstraintLimits& limits, ConstraintState& out, std::size_t& work) {
    const auto emit = [&] {
        if (plus(f.after, remaining(f, limits, work)) <= limits.max_output_bytes - out.bytes) out.branches.push_back(std::move(f));
    };
    for (;;) {
        if (++work > limits.compilation_steps) throw ConstraintError("constraint transition resource limit");
        if (f.phase == Frame::start) {
            if (whitespace(b)) { if (++f.gap <= 2) emit(); return; }
            f.gap = 0;
            switch (f.plan->kind) {
            case 'L': f.phase = Frame::finite; break;
            case 's': if (b == '"') { f.phase = Frame::string; emit(); } return;
            case 'n': f.phase = Frame::number; break;
            case '0': case 'b':
                if (b != (f.plan->kind == '0' ? 'n' : 't') && !(f.plan->kind == 'b' && b == 'f')) return;
                f.word = static_cast<char>(b); f.progress = 1; f.phase = Frame::literal; emit(); return;
            case '{':
                if (b != '{') return;
                f.phase = Frame::key_or_close;
                { auto object = std::make_shared<ObjectState>(); object->seen.resize((f.plan->properties.size() + 63) / 64); object->remaining_required_cost = f.plan->required_cost; f.object = std::move(object); }
                emit(); return;
            case '[': if (b == '[') { f.phase = Frame::key_or_close; emit(); } return;
            default: return;
            }
        }
        if (f.phase == Frame::string) {
            std::string scalar; bool closed;
            if (!f.text.feed(b, f.plan->maximum, scalar, closed)) return;
            if (closed) {
                if (f.text.scalars < f.plan->minimum) return;
                if (!finish(f, out)) emit();
            } else emit();
            return;
        }
        if (f.phase == Frame::literal) {
            const std::string_view word = f.word == 't' ? "true" : f.word == 'f' ? "false" : "null";
            if (b != static_cast<unsigned char>(word[f.progress])) return;
            if (++f.progress == word.size()) { if (!finish(f, out)) emit(); }
            else emit();
            return;
        }
        if (f.phase == Frame::finite) {
            auto to = f.plan->trie.next(f.trie, b);
            if (to != SIZE_MAX) {
                f.trie = to;
                const auto& node = f.plan->trie.nodes[to];
                if (node.terminal != SIZE_MAX && node.kind != 'n') { if (!finish(f, out)) emit(); }
                else emit();
                return;
            }
            if (!boundary(f)) return;
            if (!f.parent) {
                if (whitespace(b)) { out.done = true; out.trailing = 1; }
                return;
            }
            finish(f, out);
            continue; // Delimiter belongs to the restored parent.
        }
        if (f.phase == Frame::number) {
            if (number_byte(f, b)) { emit(); return; }
            if (!number_done(f.lex)) return;
            if (!f.parent) {
                if (whitespace(b)) { out.done = true; out.trailing = 1; }
                return;
            }
            finish(f, out);
            continue;
        }
        if (f.phase == Frame::key) {
            std::string scalar; bool closed;
            if (!f.text.feed(b, limits.max_string_length, scalar, closed)) return;
            if (!closed) {
                auto object = f.plan->additional && !scalar.empty() ? std::make_shared<ObjectState>(*f.object) : nullptr;
                for (const char byte_char : scalar) { unsigned char byte = static_cast<unsigned char>(byte_char);
                    f.trie = f.plan->trie.next(f.trie, byte);
                    if (object) {
                        if (f.name > (UINT64_MAX >> 9U) || object->next_name > (UINT64_MAX >> 9U)) return;
                        const auto edge = (f.name << 9U) | byte;
                        auto id = name_lookup(object->names, edge);
                        if (!id) { id = object->next_name++; object->names = name_insert(object->names, edge, id); }
                        f.name = id;
                    }
                }
                if (object) f.object = std::move(object);
                emit();
                return;
            }
            const auto id = f.trie == SIZE_MAX ? SIZE_MAX : f.plan->trie.nodes[f.trie].terminal;
            auto object = std::make_shared<ObjectState>(*f.object);
            if (id != SIZE_MAX) {
                if (object->contains(id)) return;
                f.child = f.plan->properties[id].value;
                object->seen[id / 64] |= std::uint64_t{1} << (id % 64);
                object->remaining_required_cost -= f.plan->properties[id].required_cost;
            } else {
                if (!f.plan->additional || f.name > (UINT64_MAX >> 9U)) return;
                const auto terminal = (f.name << 9U) | 256U;
                if (name_lookup(object->names, terminal)) return;
                f.child = f.plan->unknown;
                object->names = name_insert(object->names, terminal, 1); object->has_additional = true;
            }
            if (f.child->alternatives.empty()) return;
            f.object = std::move(object); f.name = 0; f.text = {};
            f.phase = Frame::colon; f.gap = 0; emit(); return;
        }
        if (whitespace(b)) { if (++f.gap <= 2) emit(); return; }
        f.gap = 0;
        const bool object = f.plan->kind == '{';
        const unsigned char close = object ? '}' : ']';
        if (f.phase == Frame::comma_or_close) {
            if (b == close) {
                if (object ? !required_done(f) : f.count < f.plan->minimum) return;
                if (!finish(f, out)) emit();
                return;
            }
            if (b != ',') return;
            if (!object && f.count >= f.plan->maximum) return;
            f.phase = Frame::after_comma; f.trie = 0; f.text = {};
            if (object && !key_possible(f, limits, work)) return;
            if (!object && f.plan->items->alternatives.empty()) return;
            emit(); return;
        }
        if (f.phase == Frame::colon) {
            if (b == ':') { f.phase = Frame::value; emit(); }
            return;
        }
        if (f.phase == Frame::key_or_close && b == close) {
            if (object ? !required_done(f) : f.count < f.plan->minimum) return;
            if (!finish(f, out)) emit();
            return;
        }
        if (object && (f.phase == Frame::key_or_close || f.phase == Frame::after_comma)) {
            if (b != '"') return;
            f.phase = Frame::key; f.trie = 0; f.text = {}; f.name = 0;
            emit();
            return;
        }
        if (!object && f.count >= f.plan->maximum) return;
        const auto child = object ? f.child : f.plan->items;
        if (!child) return;
        // Creating a child consumes one unit per ancestor, bounding both depth
        // and total alternative work independently of completed output length.
        work += f.depth;
        if (work >= limits.compilation_steps) throw ConstraintError("constraint transition resource limit");
        auto completed = f; completed.phase = Frame::comma_or_close; ++completed.count;
        const auto after = plus(f.after, remaining(completed, limits, work));
        auto parent = std::make_shared<const Frame>(std::move(f));
        for (const auto& plan : child->alternatives) {
            Frame next; next.plan = plan; next.parent = parent; next.depth = parent->depth + 1; next.after = after;
            consume(std::move(next), b, limits, out, work);
        }
        return;
    }
}
}  // namespace

namespace {
// Bound the serialization before allocating it, including caller-created Json
// trees that did not pass through the request parser.
void schema_size(const Json& value, std::size_t& bytes, std::size_t limit, std::size_t depth = 0) {
    if (depth > 256) error("", "limits", "schema stack limit");
    const auto add = [&](std::size_t n) { CompilationBudget::charge(bytes, n, limit); };
    const auto text = [&](std::string_view s) {
        add(2); if (s.size() > limit - bytes) error("", "limits", "schema byte limit");
        for (const char b_char : s) { unsigned char b = static_cast<unsigned char>(b_char); add(b == '"' || b == '\\' ? 2 : b >= 32 ? 1 : b == '\b' || b == '\f' || b == '\n' || b == '\r' || b == '\t' ? 2 : 6); }
    };
    if (value.is_string()) text(value.as_string());
    else if (value.is_number()) add(value.number_source().size());
    else if (value.is_bool()) add(value.as_bool() ? 4 : 5);
    else if (value.is_null()) add(4);
    else {
        add(2); bool first = true;
        if (value.is_array()) for (const auto& child : value.as_array()) { if (!first) add(1); first = false; schema_size(child, bytes, limit, depth + 1); }
        else for (const auto& [key, child] : value.as_object()) { if (!first) add(1); first = false; text(key); add(1); schema_size(child, bytes, limit, depth + 1); }
    }
}
}
ConstraintProgram compile_json_schema(const Json& schema, const ConstraintLimits& limits, ConstraintCompilationStats* stats) try {
    if (stats) *stats = {};
    std::size_t bytes = 0; schema_size(schema, bytes, limits.schema_bytes);
    CompilationBudget budget{limits};
    budget.stats = stats;
    Compiler c(limits, budget);
    if (const auto* defs = schema.find("$defs")) {
        if (!defs->is_object()) error("", "$defs", "expected definitions object");
        for (const auto& [name, sub] : defs->as_object()) c.sources[name] = &sub;
    }
    c.data->root = c.node(schema, "", 0);
    for (const auto& [name, sub] : c.sources) { (void)sub; c.definition(name, ""); }
    try {
        c.data->plan = PlanCompiler(*c.data, budget).build({c.data->root}, 0);
        if (c.data->plan->alternatives.empty()) error("", "schema", "empty root schema");
    } catch (const ConstraintError&) { error("", "limits", "schema evaluation resource limit"); }
    if (limits.mask_cache_bytes) c.data->cache = std::make_shared<MaskCache>(limits.mask_cache_bytes);
    return ConstraintProgram(c.data);
} catch (const std::bad_alloc&) { error("", "limits", "schema allocation resource limit"); }
ConstraintProgram compile_json_object() { return compile_json_schema(parse_json("{\"type\":\"object\"}")); }

struct ConstraintProgramCache::Impl {
    using Program = std::shared_ptr<const ConstraintProgram>;
    struct Entry {
        std::string key;
        std::shared_future<Program> program;
        bool ready = false;
    };
    ConstraintProgramCacheConfig config;
    mutable std::mutex mutex;
    std::list<Entry> lru; // most recently used first
    std::unordered_multimap<std::size_t, std::list<Entry>::iterator> index;
    ConstraintProgramCacheStats totals;
    void (*before_compile)(void*) = nullptr;
    void* compile_context = nullptr;

    std::list<Entry>::iterator find(std::size_t hash, const std::string& key) {
        const auto [first, last] = index.equal_range(hash);
        for (auto it = first; it != last; ++it) if (it->second->key == key) return it->second;
        return lru.end();
    }
    std::list<Entry>::iterator erase(std::list<Entry>::iterator entry) {
        const auto [first, last] = index.equal_range(std::hash<std::string>{}(entry->key));
        for (auto it = first; it != last; ++it) if (it->second == entry) { index.erase(it); break; }
        totals.key_bytes -= entry->key.size();
        return lru.erase(entry);
    }
    // Finished entries leave from the cold end; an entry that is still compiling has waiters and stays.
    bool reserve(std::size_t bytes) {
        const auto fits = [&] {
            return lru.size() < config.max_programs && totals.key_bytes <= config.key_bytes - bytes;
        };
        auto it = lru.end();
        while (it != lru.begin() && !fits()) {
            --it;
            if (!it->ready) continue;
            it = erase(it);
            ++totals.evictions;
        }
        return fits();
    }
    static void field(std::string& key, std::size_t value) { key += std::to_string(value); key.push_back(','); }
    static std::string key_of(const Json& schema, const ConstraintLimits& l) {
        std::string key = "v";
        field(key, ConstraintProgramCache::semantics_version);
        for (const std::size_t value : {l.schema_bytes, l.nodes, l.depth, l.properties, l.enum_values, l.max_string_length,
                                        l.max_output_bytes, l.compilation_steps, l.literal_bytes, l.trie_nodes,
                                        l.alternative_expansions, l.mask_cache_bytes})
            field(key, value);
        key.push_back('|');
        key += schema.dump();
        return key;
    }
    Program compile(const Json& schema, const ConstraintLimits& limits, bool cached) {
        if (before_compile) before_compile(compile_context);
        ConstraintLimits effective = limits;
        if (cached && effective.mask_cache_bytes == 0) effective.mask_cache_bytes = config.mask_cache_bytes;
        return std::make_shared<const ConstraintProgram>(compile_json_schema(schema, effective));
    }
};
// The limits struct is serialized field by field into the key; a new field must be added there too.
static_assert(sizeof(ConstraintLimits) == 12 * sizeof(std::size_t), "ConstraintProgramCache key: add the new limit");

ConstraintProgramCache::ConstraintProgramCache(ConstraintProgramCacheConfig config)
    : ConstraintProgramCache(config, nullptr, nullptr) {}
ConstraintProgramCache::ConstraintProgramCache(ConstraintProgramCacheConfig config, void (*before_compile)(void*), void* context)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
    impl_->before_compile = before_compile;
    impl_->compile_context = context;
}
ConstraintProgramCache::~ConstraintProgramCache() = default;
const ConstraintProgramCacheConfig& ConstraintProgramCache::config() const { return impl_->config; }
ConstraintProgramCacheStats ConstraintProgramCache::stats() const {
    std::lock_guard lock(impl_->mutex);
    auto out = impl_->totals;
    out.programs = impl_->lru.size();
    return out;
}
void ConstraintProgramCache::clear() {
    std::lock_guard lock(impl_->mutex);
    for (auto it = impl_->lru.begin(); it != impl_->lru.end();) {
        const auto entry = it++;
        if (entry->ready) (void)impl_->erase(entry);
    }
}
std::shared_ptr<const ConstraintProgram> ConstraintProgramCache::json_object() {
    static const Json object = parse_json("{\"type\":\"object\"}");
    return json_schema(object);
}
std::shared_ptr<const ConstraintProgram> ConstraintProgramCache::json_schema(const Json& schema, const ConstraintLimits& limits) {
    Impl& impl = *impl_;
    // The size check of the compiler comes first, so an oversized schema is refused before it is serialized.
    if (impl.config.max_programs != 0) { std::size_t bytes = 0; schema_size(schema, bytes, limits.schema_bytes); }
    std::string key = impl.config.max_programs != 0 ? Impl::key_of(schema, limits) : std::string{};
    if (impl.config.max_programs == 0 || key.size() > impl.config.key_bytes / 4U) {
        { std::lock_guard lock(impl.mutex); ++impl.totals.uncached; }
        return impl.compile(schema, limits, false);
    }
    const std::size_t hash = std::hash<std::string>{}(key);
    std::promise<Impl::Program> promise;
    {
        std::unique_lock lock(impl.mutex);
        if (const auto hit = impl.find(hash, key); hit != impl.lru.end()) {
            impl.lru.splice(impl.lru.begin(), impl.lru, hit);
            ++(hit->ready ? impl.totals.hits : impl.totals.joined);
            const auto program = hit->program;
            lock.unlock();
            return program.get(); // rethrows the compilation error of the request that started it
        }
        ++impl.totals.misses;
        if (!impl.reserve(key.size())) {
            ++impl.totals.uncached;
            lock.unlock();
            return impl.compile(schema, limits, false);
        }
        impl.lru.push_front({key, promise.get_future().share(), false});
        try {
            impl.index.emplace(hash, impl.lru.begin());
        } catch (...) {
            impl.lru.pop_front();
            throw;
        }
        impl.totals.key_bytes += key.size();
    }
    const auto begin = std::chrono::steady_clock::now();
    const auto elapsed = [&] {
        return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - begin).count());
    };
    try {
        auto program = impl.compile(schema, limits, true);
        std::lock_guard lock(impl.mutex);
        impl.totals.compile_ns += elapsed();
        impl.find(hash, key)->ready = true;
        promise.set_value(program);
        return program;
    } catch (...) {
        std::lock_guard lock(impl.mutex);
        impl.totals.compile_ns += elapsed();
        ++impl.totals.failures;
        promise.set_exception(std::current_exception());
        (void)impl.erase(impl.find(hash, key));
        throw;
    }
}
ConstraintProgramCache& shared_constraint_program_cache() {
    static ConstraintProgramCache cache([] {
        ConstraintProgramCacheConfig config;
        static const bool enabled = envcfg::flag("STRIX_CONSTRAINT_PROGRAM_CACHE", true, EnvFlagMatch::exact);
        if (!enabled) config.max_programs = 0;
        config.mask_cache_bytes = static_cast<std::size_t>(
            env_bytes("STRIX_CONSTRAINT_MASK_MEMO_BYTES", config.mask_cache_bytes, 0, std::uint64_t{1} << 30U));
        envcfg::record("STRIX_CONSTRAINT_MASK_MEMO_BYTES", envcfg::text("STRIX_CONSTRAINT_MASK_MEMO_BYTES"),
                       std::to_string(config.mask_cache_bytes));
        return config;
    }());
    return cache;
}
ConstraintCursor::ConstraintCursor(const ConstraintProgram& program) : data_(program.data_) {
    auto state = std::make_shared<ConstraintState>();
    for (const auto& plan : data_->plan->alternatives) { Frame f; f.plan = plan; state->branches.push_back(std::move(f)); }
    state_ = std::move(state);
}
bool ConstraintCursor::advance(unsigned char byte) {
    if (state_->bytes >= data_->limits.max_output_bytes) return false;
    auto next = std::make_shared<ConstraintState>(); next->bytes = state_->bytes + 1;
    if (state_->done && whitespace(byte) && state_->trailing < 2) {
        next->done = true; next->trailing = static_cast<unsigned char>(state_->trailing + 1);
    }
    std::size_t work = 0;
    for (const auto& branch : state_->branches) consume(branch, byte, data_->limits, *next, work);
    if (!next->done && next->branches.empty()) return false;
    state_ = std::move(next); return true;
}
bool ConstraintCursor::complete() const {
    if (state_->done) return true;
    for (const auto& branch : state_->branches) if (!branch.parent && boundary(branch)) return true;
    return false;
}
bool ConstraintCursor::dead() const {
    return (state_->branches.empty() && !state_->done) || (state_->bytes >= data_->limits.max_output_bytes && !complete());
}
ConstraintCursor advance_token(ConstraintCursor cursor, std::string_view bytes) {
    if (bytes.empty()) throw ConstraintError("empty constraint token");
    for (const char b_char : bytes) { unsigned char b = static_cast<unsigned char>(b_char); if (!cursor.advance(b)) throw ConstraintError("token refused by constraint"); }
    return cursor;
}
TokenByteTrie::TokenByteTrie(const BpeTokenizer& tokenizer, std::span<const std::uint32_t> excluded) : vocab_size_(tokenizer.vocab_size()) {
    for (std::size_t i = 0; i < vocab_size_; ++i) {
        const auto id = static_cast<std::uint32_t>(i); const auto type = tokenizer.type(id);
        if ((type == BpeTokenizer::TokenType::normal || type == BpeTokenizer::TokenType::byte) && std::find(excluded.begin(), excluded.end(), id) == excluded.end()) insert(tokenizer.token_bytes(id), id);
    }
}
TokenByteTrie::TokenByteTrie(std::span<const std::string> bytes, std::span<const std::uint32_t> excluded) : vocab_size_(bytes.size()) {
    for (std::size_t i = 0; i < bytes.size(); ++i) if (std::find(excluded.begin(), excluded.end(), i) == excluded.end()) insert(bytes[i], static_cast<std::uint32_t>(i));
}
void TokenByteTrie::insert(std::string_view bytes, std::uint32_t id) {
    if (bytes.empty()) return;
    if (members_.size() < vocab_size_) members_.resize(vocab_size_, false);
    members_.at(id) = true;
    std::size_t at = 0;
    for (const char b_char : bytes) { unsigned char b = static_cast<unsigned char>(b_char);
        const auto it = std::find_if(nodes_[at].edges.begin(), nodes_[at].edges.end(), [&](auto e) { return e.first == b; });
        if (it != nodes_[at].edges.end()) at = it->second;
        else {
            const auto next = nodes_.size(); nodes_[at].edges.emplace_back(b, next); nodes_.emplace_back(); at = next;
        }
    }
    nodes_[at].ids.push_back(id);
}
bool AllowedTokens::contains(std::size_t id) const { return id < vocab_size && (words[id / 64] & (std::uint64_t{1} << (id % 64))) != 0; }
void AllowedTokens::set(std::size_t id, bool value) {
    if (id >= vocab_size) throw ConstraintError("token ID outside vocabulary");
    const auto bit = std::uint64_t{1} << (id % 64);
    if (value) words[id / 64] |= bit; else words[id / 64] &= ~bit;
}
struct ConstraintAccess {
    static bool same(const ConstraintProgram& p, const ConstraintCursor& c) { return p.data_ == c.data_; }
    static auto cache(const ConstraintProgram& p) { return p.data_->cache; }
    static auto identity(const TokenByteTrie& t) { return t.identity_; }
    static const auto& state(const ConstraintCursor& c) { return *c.state_; }
};
ConstraintMemoStats ConstraintProgram::memo_stats() const {
    const auto cache = ConstraintAccess::cache(*this);
    if (!cache) return {};
    std::lock_guard lock(cache->mutex);
    return cache->totals;
}
namespace {
std::optional<AllowedTokens> walk_allowed(const ConstraintProgram& program, const ConstraintCursor& cursor, const TokenByteTrie& trie,
                                          std::span<const std::uint32_t> eos_ids, const std::atomic<bool>* stop);
}
AllowedTokens allowed(const ConstraintProgram& program, const ConstraintCursor& cursor, const TokenByteTrie& trie, std::span<const std::uint32_t> eos_ids) {
    return std::move(*walk_allowed(program, cursor, trie, eos_ids, nullptr));
}
std::optional<AllowedTokens> allowed_until(const ConstraintProgram& program, const ConstraintCursor& cursor, const TokenByteTrie& trie,
                                           std::span<const std::uint32_t> eos_ids, const std::atomic<bool>& stop) {
    return walk_allowed(program, cursor, trie, eos_ids, &stop);
}
namespace {
std::optional<AllowedTokens> walk_allowed(const ConstraintProgram& program, const ConstraintCursor& cursor, const TokenByteTrie& trie,
                                          std::span<const std::uint32_t> eos_ids, const std::atomic<bool>* stop) {
    if (!ConstraintAccess::same(program, cursor)) throw ConstraintError("cursor belongs to a different program");
    AllowedTokens mask{trie.vocab_size(), std::vector<std::uint64_t>((trie.vocab_size() + 63) / 64)};
    const auto cache = ConstraintAccess::cache(program);
    const auto identity = ConstraintAccess::identity(trie);
    const auto key = cache ? cache_key(ConstraintAccess::state(cursor), identity.get(), cache->budget) : std::string{};
    if (!key.empty() && cache->get(key, mask)) {
        for (auto id : eos_ids) mask.set(id, cursor.complete());
        return mask;
    }
    // Iterative DFS: token length never consumes the native call stack.
    std::vector<std::pair<std::size_t, ConstraintCursor>> pending; pending.emplace_back(0, cursor);
    for (std::size_t visited = 0; !pending.empty(); ++visited) {
        if (stop && (visited & 255U) == 0 && stop->load(std::memory_order_relaxed)) return std::nullopt;
        auto [at, state] = std::move(pending.back()); pending.pop_back();
        for (auto id : trie.nodes()[at].ids) mask.set(id);
        for (auto [b, next] : trie.nodes()[at].edges) { auto copy = state; if (copy.advance(b)) pending.emplace_back(next, std::move(copy)); }
    }
    if (!key.empty()) cache->put(key, identity, mask);
    for (auto id : eos_ids) mask.set(id, cursor.complete());
    return mask;
}
}  // namespace
std::uint32_t constrained_greedy(std::span<const float> logits, const AllowedTokens& mask) {
    if (logits.size() != mask.vocab_size) throw ConstraintError("logit vocabulary mismatch");
    std::uint32_t best = UINT32_MAX; float score = -std::numeric_limits<float>::infinity();
    for (std::size_t i = 0; i < logits.size(); ++i) if (mask.contains(i)) {
        if (std::isnan(logits[i])) throw ConstraintError("NaN allowed logit");
        if (best == UINT32_MAX || logits[i] > score) { best = static_cast<std::uint32_t>(i); score = logits[i]; }
    }
    if (best == UINT32_MAX) throw ConstraintError("empty constraint mask");
    return best;
}
}  // namespace strix::serve
