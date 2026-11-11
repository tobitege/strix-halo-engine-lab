#include "strix/deploy/spec.hpp"
#include "strix/model_description.hpp"

#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>

namespace strix::deploy {
namespace {
using Value = model_description::Value;
using Table = Value::Table;
const Value* find(const Table& t, const std::string& k) {
    const auto it = t.find(k);
    return it == t.end() ? nullptr : &it->second;
}
std::string string(const Value& v, const std::string& k) {
    if (const auto* s = std::get_if<std::string>(&v.data)) return *s;
    throw std::invalid_argument(k + ": expected string");
}
std::uint64_t integer(const Value& v, const std::string& k) {
    if (const auto* n = std::get_if<std::int64_t>(&v.data); n && *n >= 0) return static_cast<std::uint64_t>(*n);
    throw std::invalid_argument(k + ": expected nonnegative integer");
}
void keys(const Table& t, const std::set<std::string>& allowed, const std::string& prefix = "") {
    for (const auto& [k, v] : t) {
        (void)v;
        if (!allowed.contains(k)) throw std::invalid_argument(prefix + k + ": unknown deployment key");
    }
}
} // namespace

void validate_spec(const Spec& s) {
    if (!s.context) throw std::invalid_argument("context must be positive");
    if (s.nodes > 256 || s.slots > 65536) throw std::invalid_argument("nodes or slots exceeds deployment search bounds");
    if (s.memory.policy != "kv-first") throw std::invalid_argument("memory.policy: only kv-first is implemented");
    if (!std::isfinite(s.memory.margin_gib) || s.memory.margin_gib < 0 || s.memory.margin_gib >= 17179869184.0)
        throw std::invalid_argument("memory.margin_gib: invalid byte size");
    if (!std::set<std::string>{"auto", "single", "replica", "replicas", "tp2", "pipeline", "disaggregated"}.contains(s.topology))
        throw std::invalid_argument("topology: unsupported mode " + s.topology);
    if (s.format.resident_layers > s.format.total_layers)
        throw std::invalid_argument("format.resident_layers exceeds total_layers");
    if (s.link.empty() || s.profile.empty() || s.format.decode.empty() || s.format.prefill.empty())
        throw std::invalid_argument("empty deployment selection");
}

Spec parse_spec(std::string_view text) {
    Spec s;
    if (text == "auto") return s;
    const auto d = model_description::parse(text);
    keys(d.fields, {"schema", "model", "nodes", "link", "topology", "format", "context", "slots", "profile", "memory", "hardware", "pins", "devices"});
    const auto* schema = d.find("schema");
    if (!schema || string(*schema, "schema") != "strix-deploy-v1") throw std::invalid_argument("schema must be strix-deploy-v1");
    const auto set_string = [&](const std::string& key, std::string& target) {
        if (const auto* v = d.find(key)) { target = string(*v, key); s.pins[key] = target; }
    };
    const auto set_u32 = [&](const std::string& key, std::uint32_t& target) {
        if (const auto* v = d.find(key)) {
            const auto n = integer(*v, key);
            if (!n || n > std::numeric_limits<std::uint32_t>::max()) throw std::invalid_argument(key + ": expected positive uint32");
            target = static_cast<std::uint32_t>(n); s.pins[key] = std::to_string(n);
        }
    };
    const auto table_keys = [&](const std::string& name, const std::set<std::string>& allowed) {
        if (const auto* v = d.find(name)) {
            const auto* t = std::get_if<Table>(&v->data);
            if (!t) throw std::invalid_argument(name + ": expected table");
            keys(*t, allowed, name + ".");
        }
    };
    set_string("model", s.model);
    table_keys("hardware", {"nodes", "link"});
    table_keys("memory", {"context", "slots", "margin_gib", "policy"});
    table_keys("format", {"decode", "prefill", "resident_layers", "total_layers", "jit_slots"});
    for (const auto& [plain, nested] : std::vector<std::pair<std::string, std::string>>{
             {"nodes", "hardware.nodes"}, {"link", "hardware.link"}, {"context", "memory.context"}, {"slots", "memory.slots"}})
        if (d.find(plain) && d.find(nested)) throw std::invalid_argument(plain + ": duplicate deployment selection");
    set_u32("nodes", s.nodes); set_u32("hardware.nodes", s.nodes);
    set_string("link", s.link); set_string("hardware.link", s.link);
    set_u32("context", s.context); set_u32("memory.context", s.context);
    set_u32("slots", s.slots); set_u32("memory.slots", s.slots);
    if (const auto* v = d.find("topology"); v && std::holds_alternative<std::string>(v->data)) set_string("topology", s.topology);
    else { table_keys("topology", {"mode", "ranks"}); set_string("topology.mode", s.topology); set_u32("topology.ranks", s.nodes); }
    if (d.find("topology.ranks") && (d.find("nodes") || d.find("hardware.nodes"))) {
        const auto* n = d.find("nodes") ? d.find("nodes") : d.find("hardware.nodes");
        if (integer(*n, "nodes") != s.nodes) throw std::invalid_argument("topology.ranks conflicts with nodes");
    }
    if (const auto* v = d.find("profile"); v && std::holds_alternative<std::string>(v->data)) set_string("profile", s.profile);
    else {
        table_keys("profile", {"name", "allow_pending"}); set_string("profile.name", s.profile);
        if (const auto* pending = d.find("profile.allow_pending")) {
            const auto* allow = std::get_if<bool>(&pending->data);
            if (!allow || *allow) throw std::invalid_argument("profile.allow_pending: deployment requires qualified candidates");
            s.pins["profile.allow_pending"] = "false";
        }
    }
    set_string("format.decode", s.format.decode); set_string("format.prefill", s.format.prefill);
    // Zero is meaningful for partial/JIT choices.
    for (auto [key, target] : std::vector<std::pair<std::string, std::uint32_t*>>{
             {"resident_layers", &s.format.resident_layers}, {"total_layers", &s.format.total_layers}, {"jit_slots", &s.format.jit_slots}}) {
        if (const auto* v = d.find("format." + key)) {
            const auto n = integer(*v, key);
            if (n > std::numeric_limits<std::uint32_t>::max()) throw std::invalid_argument(key + ": uint32 overflow");
            *target = static_cast<std::uint32_t>(n); s.pins["format." + key] = std::to_string(n);
        }
    }
    set_string("memory.policy", s.memory.policy);
    if (const auto* v = d.find("memory.margin_gib")) {
        if (const auto* n = std::get_if<double>(&v->data)) s.memory.margin_gib = *n;
        else s.memory.margin_gib = static_cast<double>(integer(*v, "memory.margin_gib"));
    }
    // pins is a string table of canonical scalar paths; apply through the same schema validation.
    if (const auto* v = d.find("pins")) {
        const auto* t = std::get_if<Table>(&v->data);
        if (!t) throw std::invalid_argument("pins: expected table");
        for (const auto& [k, val] : *t) {
            const auto value = string(val, "pins." + k);
            if (!s.pins.contains(k) || s.pins.at(k) != value)
                throw std::invalid_argument("pins." + k + ": must match an explicit selection");
        }
    }
    if (const auto* v = d.find("devices")) {
        const auto* a = std::get_if<Value::Array>(&v->data);
        if (!a || a->size() > 256) throw std::invalid_argument("devices: expected bounded array of tables");
        for (const auto& item : *a) {
            const auto* t = std::get_if<Table>(&item.data);
            if (!t) throw std::invalid_argument("devices: expected table");
            keys(*t, {"gfx_target", "cu_count", "mem_total_bytes", "mem_free_bytes", "unified", "source"}, "devices.");
            DeviceFacts f;
            for (const char* k : {"gfx_target", "cu_count", "mem_total_bytes", "mem_free_bytes", "unified", "source"})
                if (!find(*t, k)) throw std::invalid_argument(std::string("devices: missing ") + k);
            f.gfx_target = string(*find(*t, "gfx_target"), "gfx_target");
            const auto cu = integer(*find(*t, "cu_count"), "cu_count");
            if (cu > std::numeric_limits<std::uint32_t>::max()) throw std::invalid_argument("cu_count: uint32 overflow");
            f.cu_count = static_cast<std::uint32_t>(cu);
            f.mem_total_bytes = integer(*find(*t, "mem_total_bytes"), "mem_total_bytes");
            f.mem_free_bytes = integer(*find(*t, "mem_free_bytes"), "mem_free_bytes");
            const auto* unified = std::get_if<bool>(&find(*t, "unified")->data);
            if (!unified) throw std::invalid_argument("unified: expected bool");
            f.unified = *unified;
            f.source = string(*find(*t, "source"), "source");
            if (!f.mem_total_bytes || f.mem_free_bytes > f.mem_total_bytes || f.source.empty())
                throw std::invalid_argument("devices: invalid measured memory");
            s.devices.push_back(std::move(f));
        }
    }
    validate_spec(s);
    return s;
}

Spec load_spec(const std::string& path) {
    if (path == "auto") return {};
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot open deployment spec");
    std::ostringstream text;
    text << file.rdbuf();
    return parse_spec(text.str());
}
} // namespace strix::deploy
