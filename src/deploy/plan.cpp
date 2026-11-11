#include "strix/deploy/plan.hpp"
#include "strix/deploy/binding.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>

namespace strix::deploy {
namespace {
std::string quote(const std::string& s) {
    static constexpr char hex[] = "0123456789abcdef";
    std::string out = "\"";
    for (const char ch : s) {
        const auto c = static_cast<unsigned char>(ch);
        if (c == '"' || c == '\\') { out += '\\'; out += ch; }
        else if (c < 32) { out += "\\u00"; out += hex[c >> 4]; out += hex[c & 15]; }
        else out += ch;
    }
    return out + '"';
}
std::string format_json(const Format& f) {
    return "{\"decode\":" + quote(f.decode) + ",\"prefill\":" + quote(f.prefill) +
        ",\"resident_layers\":" + std::to_string(f.resident_layers) + ",\"total_layers\":" + std::to_string(f.total_layers) +
        ",\"jit_slots\":" + std::to_string(f.jit_slots) + "}";
}
std::string canonical(const Plan& p) {
    std::ostringstream o;
    o.imbue(std::locale::classic());
    const auto& s = p.requested;
    o << "{\"schema\":\"strix-deployment-v1\",\"model\":" << quote(s.model)
      << ",\"refused\":" << (p.refused ? "true" : "false") << ",\"refusal\":" << quote(p.refusal)
      << ",\"requested\":{\"nodes\":" << s.nodes << ",\"link\":" << quote(s.link)
      << ",\"topology\":" << quote(s.topology) << ",\"context\":" << s.context << ",\"slots\":" << s.slots
      << ",\"profile\":" << quote(s.profile) << ",\"format\":" << format_json(s.format)
      << ",\"memory\":{\"policy\":" << quote(s.memory.policy) << ",\"margin_gib\":"
      << (std::isfinite(s.memory.margin_gib) ? [&] {
             std::ostringstream n; n.imbue(std::locale::classic()); n << std::setprecision(17) << s.memory.margin_gib; return n.str();
         }() : "null") << "},\"pins\":{";
    bool comma = false;
    for (const auto& [k, v] : s.pins) { if (comma) o << ','; comma = true; o << quote(k) << ':' << quote(v); }
    o << "}},\"chosen\":{\"topology\":" << quote(p.chosen.topology) << ",\"format\":" << format_json(p.chosen.format)
      << ",\"context\":" << p.chosen.context << ",\"slots\":" << p.chosen.slots << ",\"ranks\":" << p.chosen.ranks << "},\"devices\":[";
    comma = false;
    for (const auto& d : p.devices) {
        if (comma) o << ',';
        comma = true;
        o << "{\"gfx_target\":" << quote(d.gfx_target) << ",\"cu_count\":" << d.cu_count
          << ",\"mem_total_bytes\":" << d.mem_total_bytes << ",\"mem_free_bytes\":" << d.mem_free_bytes
          << ",\"unified\":" << (d.unified ? "true" : "false") << ",\"source\":" << quote(d.source) << '}';
    }
    o << "],\"ranks\":[";
    comma = false;
    for (const auto& r : p.ranks) {
        if (comma) o << ',';
        comma = true;
        o << "{\"rank\":" << r.rank << ",\"node\":" << r.node << ",\"total_bytes\":" << r.total_bytes
          << ",\"free_bytes\":" << r.free_bytes << ",\"margin_bytes\":" << r.margin_bytes << ",\"budget_bytes\":" << r.budget_bytes
          << ",\"kv_bytes\":" << r.kv_bytes << ",\"after_kv_bytes\":" << r.after_kv_bytes
          << ",\"after_workspace_bytes\":" << r.after_workspace_bytes << ",\"required_bytes\":" << r.required_bytes
          << ",\"free_after_bytes\":" << r.free_after_bytes << ",\"fits\":" << (r.fits ? "true" : "false")
          << ",\"reason\":" << quote(r.reason) << ",\"footprint\":{\"weights\":" << r.footprint.weights
          << ",\"kv_per_token\":" << r.footprint.kv_per_token << ",\"kv_total\":" << r.footprint.kv_total
          << ",\"workspace\":" << r.footprint.workspace << ",\"staging\":" << r.footprint.staging << ",\"extra\":[";
        bool more = false;
        for (const auto& [name, bytes] : r.footprint.extra) {
            if (more) o << ',';
            more = true;
            o << "{\"name\":" << quote(name) << ",\"bytes\":" << bytes << '}';
        }
        o << "]}}";
    }
    o << "],\"reasons\":[";
    comma = false;
    for (const auto& reason : p.reasons) { if (comma) o << ','; comma = true; o << quote(reason); }
    o << "]}";
    return o.str();
}
bool matches(const Spec& s, const Format& f) {
    return (s.format.decode == "auto" || s.format.decode == f.decode) &&
        (s.format.prefill == "auto" || s.format.prefill == f.prefill) &&
        (!s.pins.contains("format.resident_layers") || s.format.resident_layers == f.resident_layers) &&
        (!s.pins.contains("format.total_layers") || s.format.total_layers == f.total_layers) &&
        (!s.pins.contains("format.jit_slots") || s.format.jit_slots == f.jit_slots);
}
} // namespace

std::string Plan::hash() const {
    std::uint64_t h = 14695981039346656037ULL;
    for (const char c : canonical(*this)) { h ^= static_cast<unsigned char>(c); h *= 1099511628211ULL; }
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::hex << std::setfill('0') << std::setw(16) << h;
    return out.str();
}
std::string Plan::to_json() const {
    auto json = canonical(*this);
    json.pop_back();
    return json + ",\"deployment_id\":" + quote(hash()) + "}";
}

Plan resolve(const Spec& input, const std::vector<DeviceFacts>& devices, const FootprintProvider& provider) {
    Plan p;
    p.requested = input;
    p.devices = devices;
    const auto refuse = [&](const std::string& reason) { p.refusal = reason; return p; };
    try { validate_spec(input); } catch (const std::exception& e) { return refuse(e.what()); }
    if (input.model.empty()) p.requested.model = provider.model();
    else if (input.model != provider.model()) return refuse("model: provider does not support pinned model");
    if (devices.empty()) return refuse("devices: no measured node inventory");
    if (devices.size() > 256) return refuse("devices: inventory exceeds search bound");
    if (input.nodes && input.nodes != devices.size()) return refuse("nodes: pinned count differs from measured inventory");
    for (const auto& d : devices)
        if (!d.mem_total_bytes || d.mem_free_bytes > d.mem_total_bytes)
            return refuse("devices: missing or inconsistent device-usable memory");
    const auto max_slots = provider.max_slots();
    if (!max_slots || max_slots > 65536) return refuse("provider: invalid maximum slots");
    if (input.slots > max_slots) return refuse("slots: pinned count exceeds provider limit");
    const auto margin = static_cast<std::uint64_t>(std::ceil(input.memory.margin_gib * 1073741824.0));
    const auto topologies = provider.topologies();
    std::size_t attempts = 0;
    // KV capacity takes priority over organization. Pins never shrink.
    for (std::uint32_t slots = input.slots ? input.slots : max_slots; slots; --slots) {
        for (const auto& advertised : topologies) {
            const auto topology = advertised == "replicas" ? "replica" : advertised;
            const auto pinned = input.topology == "replicas" ? "replica" : input.topology;
            if (pinned != "auto" && pinned != topology) continue;
            const bool pair = topology == "tp2";
            const bool multi = pair || topology == "pipeline" || topology == "disaggregated";
            if (topology != "single" && topology != "replica" && !multi) continue;
            if (pair && devices.size() != 2) { p.reasons.push_back("tp2: requires exactly two measured nodes"); continue; }
            if (multi && (devices.size() < 2 || input.link == "none" || input.link == "auto")) {
                p.reasons.push_back(topology + ": requires multiple nodes and an explicit link"); continue;
            }
            const auto count = topology == "single" ? 1U : static_cast<std::uint32_t>(devices.size());
            // Single candidates may select any measured node, in supplied inventory order.
            const auto placements = topology == "single" ? devices.size() : std::size_t{1};
            for (const auto& format : provider.formats(topology)) {
                if (!matches(input, format)) continue;
                for (std::size_t placement = 0; placement < placements; ++placement) {
                    if (++attempts > 100000) return refuse("search: candidate limit exceeded");
                    Candidate c{topology, format, input.context, slots, count};
                    std::vector<RankBudget> ranks;
                    bool fits = true;
                    for (std::uint32_t rank = 0; rank < count; ++rank) {
                        RankBudget b;
                        b.rank = rank;
                        b.node = topology == "single" ? static_cast<std::uint32_t>(placement) : rank;
                        const auto& d = devices[b.node];
                        b.total_bytes = d.mem_total_bytes; b.free_bytes = d.mem_free_bytes; b.margin_bytes = margin;
                        b.budget_bytes = b.free_bytes > margin ? b.free_bytes - margin : 0;
                        if (b.free_bytes < margin) b.reason = "margin exceeds measured free memory";
                        try {
                            b.footprint = provider.footprint(input, c, rank);
                            const auto& f = b.footprint;
                            b.kv_bytes = f.kv_total ? f.kv_total : checked_mul(checked_mul(f.kv_per_token, c.context), c.slots);
                            auto needed = b.kv_bytes;
                            if (b.reason.empty() && needed > b.budget_bytes) b.reason = "KV reservation exceeds budget";
                            b.after_kv_bytes = needed <= b.budget_bytes ? b.budget_bytes - needed : 0;
                            needed = checked_add(needed, f.workspace);
                            if (b.reason.empty() && needed > b.budget_bytes) b.reason = "workspace exceeds remaining budget";
                            b.after_workspace_bytes = needed <= b.budget_bytes ? b.budget_bytes - needed : 0;
                            needed = checked_add(checked_add(needed, f.weights), f.staging);
                            for (const auto& extra : f.extra) needed = checked_add(needed, extra.second);
                            // KV/slots and every mandatory owner win over optional cache RAM.
                            for (const auto& [name, cap] : f.elastic_extra) {
                                const auto bytes = fit_elastic_extra(cap, needed, b.budget_bytes);
                                b.footprint.extra.emplace_back(name, bytes);
                                needed = checked_add(needed, bytes);
                            }
                            b.footprint.elastic_extra.clear();
                            b.required_bytes = needed;
                            b.fits = b.free_bytes >= margin && needed <= b.budget_bytes;
                            if (!b.fits && b.reason.empty()) b.reason = "weights/staging/extras exceed remaining budget";
                            if (b.fits) b.free_after_bytes = b.budget_bytes - needed;
                        } catch (const std::exception& e) { b.reason = e.what(); }
                        if (!b.fits) {
                            fits = false;
                            p.reasons.push_back(topology + "/" + format.decode + "/" + format.prefill + " slots=" +
                                std::to_string(slots) + " rank=" + std::to_string(rank) + ": " + b.reason);
                        }
                        ranks.push_back(std::move(b));
                    }
                    p.chosen = c;
                    p.ranks = std::move(ranks);
                    if (fits) {
                        p.refused = false;
                        p.reasons.push_back("admitted: KV first, then workspace, weights, staging and extras; margin preserved");
                        return p;
                    }
                }
            }
        }
        if (input.slots) break;
    }
    return refuse(attempts ? "no candidate fits the measured per-rank budgets" : "no supported candidate matches pinned topology/format/link/nodes");
}
} // namespace strix::deploy
