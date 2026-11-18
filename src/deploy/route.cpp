#include "strix/deploy/route.hpp"

#include "strix/model_description.hpp"
#include "strix/pair/balance.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <iomanip>
#include <limits>
#include <locale>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>

namespace strix::deploy {
namespace {
using Value = model_description::Value;
using Table = Value::Table;

[[noreturn]] void bad(const std::string& key, const std::string& what) { throw std::invalid_argument(key + ": " + what); }
// The key is a view: a std::string argument (a temporary) makes GCC flag the returned reference.
const Table& table(const Value& v, std::string_view key) {
    if (const auto* t = std::get_if<Table>(&v.data)) return *t;
    bad(std::string(key), "expected a table");
}
std::string text(const Value& v, const std::string& key) {
    if (const auto* s = std::get_if<std::string>(&v.data)) return *s;
    bad(key, "expected a string");
}
double number(const Value& v, const std::string& key) {
    double out = -1;
    if (const auto* d = std::get_if<double>(&v.data)) out = *d;
    else if (const auto* n = std::get_if<std::int64_t>(&v.data)) out = static_cast<double>(*n);
    if (!std::isfinite(out) || out < 0) bad(key, "expected a nonnegative number");
    return out;
}
std::uint64_t count(const Value& v, const std::string& key) {
    if (const auto* n = std::get_if<std::int64_t>(&v.data); n && *n >= 0) return static_cast<std::uint64_t>(*n);
    bad(key, "expected a nonnegative integer");
}
const Value* find(const Table& t, std::string_view key) {
    const auto it = t.find(key);
    return it == t.end() ? nullptr : &it->second;
}
const Value::Array& array(const Value& v, std::string_view key) {
    if (const auto* a = std::get_if<Value::Array>(&v.data)) return *a;
    bad(std::string(key), "expected an array");
}

std::string join(const std::vector<std::string>& unit) {
    std::string out;
    for (const auto& id : unit) out += (out.empty() ? "" : "+") + id;
    return out;
}
bool unique_unit(const std::vector<std::string>& unit) {
    return !unit.empty() && std::set<std::string>(unit.begin(), unit.end()).size() == unit.size();
}
bool overlaps(const std::vector<std::string>& a, const std::vector<std::string>& b) {
    return std::any_of(a.begin(), a.end(), [&](const auto& id) { return std::find(b.begin(), b.end(), id) != b.end(); });
}
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
std::string num(double v) {
    if (!std::isfinite(v)) return "null";
    std::ostringstream o;
    o.imbue(std::locale::classic());
    o << std::fixed << std::setprecision(3) << v;
    return o.str();
}
std::string ms_text(double v) { return num(v) + " ms"; }

// ---- cost estimation -----------------------------------------------------------------------------------------------
struct Estimate {
    double ms = 0, fixed_ms = 0, rel = 0;
    const StageCost* entry = nullptr; // the bracketing entry closest to the prompt
    std::string note;
};
std::vector<const StageCost*> entries_for(const StageCostTable& costs, const std::string& unit, const std::string& identity,
                                          std::uint32_t chunk) {
    std::vector<const StageCost*> warm, cold;
    for (const auto& e : costs.entries)
        if (join(e.unit) == unit && e.identity == identity && (chunk == 0 || e.chunk == chunk)) (e.warm ? warm : cold).push_back(&e);
    auto& out = warm.empty() ? cold : warm;
    std::sort(out.begin(), out.end(), [](const StageCost* a, const StageCost* b) { return a->context < b->context; });
    return out;
}
// Whole-prompt time of `rows` rows from receipts at other lengths: linear between receipts, the outer receipt's
// per-row rate beyond them (a stated hypothesis with wider uncertainty, not a demonstration of scaling).
std::optional<Estimate> estimate(const std::vector<const StageCost*>& points, std::uint64_t rows) {
    if (points.empty() || rows == 0) return std::nullopt;
    const auto per_row = [](const StageCost* e) { return (e->total_ms - e->fixed_ms) / static_cast<double>(e->context); };
    const double n = static_cast<double>(rows);
    Estimate out;
    for (std::size_t i = 0; i < points.size(); ++i) {
        const auto* e = points[i];
        if (e->context == rows) {
            out = {e->total_ms, e->fixed_ms, e->spread, e, "measured at " + std::to_string(rows)};
            return out;
        }
        if (e->context > rows) {
            if (i == 0) {
                out = {e->fixed_ms + per_row(e) * n, e->fixed_ms, e->spread + 0.10, e,
                       "below calibration " + std::to_string(e->context) + ": its per-row rate"};
                return out;
            }
            const auto* lo = points[i - 1];
            const double t = (n - static_cast<double>(lo->context)) / static_cast<double>(e->context - lo->context);
            const auto* near = t < 0.5 ? lo : e;
            out = {lo->total_ms + t * (e->total_ms - lo->total_ms), near->fixed_ms, std::max(lo->spread, e->spread) + 0.05, near,
                   "interpolated " + std::to_string(lo->context) + ".." + std::to_string(e->context)};
            return out;
        }
    }
    const auto* e = points.back();
    out = {e->fixed_ms + per_row(e) * n, e->fixed_ms, e->spread + 0.15, e,
           "beyond calibration " + std::to_string(e->context) + ": constant per-row rate assumed"};
    return out;
}

// ---- per-option checks ---------------------------------------------------------------------------------------------
struct Ctx {
    const RouteRequest& request;
    const Inventory& inventory;
    const RouteProvider& provider;
    const StageCostTable& costs;
    LinkGraph graph;
    std::uint64_t rows = 0, margin = 0;
    std::uint32_t slots = 1;
    mutable std::map<std::pair<std::string, std::string>, PathResult> paths;
};

const device::Descriptor* descriptor(const Ctx& c, const std::string& id) {
    for (const auto& d : c.inventory.declaration.devices)
        if (d.id == id) return &d;
    return nullptr;
}
// Empty when every device of `unit` may take the stage; `prefill` stages need the rated prefill role (the start
// gate's rule), the decoding unit needs decode, pair or replica.
std::string role_check(const Ctx& c, const std::vector<std::string>& unit, bool prefill_stage, bool decoding) {
    for (const auto& id : unit) {
        const auto* d = descriptor(c, id);
        if (!d) return "device " + id + " is not declared";
        if (d->excluded) return "device " + id + " is excluded" + (d->reason.empty() ? "" : ": " + d->reason);
        if (decoding) {
            bool ok = false;
            for (const auto* role : {"decode", "pair", "replica"})
                if (device::role_refusal(*d, role).empty()) ok = true;
            if (!ok) return "device " + id + " takes no decode, pair or replica role";
        }
        if (prefill_stage) {
            const auto why = device::role_refusal(*d, "prefill");
            if (!why.empty()) return why;
        }
    }
    return {};
}
std::string load_check(const Ctx& c, const std::vector<std::string>& unit) {
    for (const auto& id : unit) {
        const auto it = c.inventory.load.find(id);
        if (it == c.inventory.load.end()) continue;
        if (it->second.prefill_busy) return "live load: device " + id + " is running a prefill";
        if (it->second.queued) return "live load: device " + id + " has " + std::to_string(it->second.queued) + " queued requests";
    }
    return {};
}
std::optional<Format> format_for(const Ctx& c, const std::string& topology) {
    const auto& f = c.request.spec.format;
    for (const auto& x : c.provider.footprints().formats(topology))
        if ((f.decode == "auto" || f.decode == x.decode) && (f.prefill == "auto" || f.prefill == x.prefill)) return x;
    return std::nullopt;
}
// Memory admission through the model's FootprintProvider and the shared fit_rank(); a partial stage asks the
// provider for its range footprint and is refused when the provider cannot state it.
std::string memory_check(const Ctx& c, RouteStage& stage, bool whole_model) {
    const auto format = format_for(c, stage.topology);
    if (!format) return "provider advertises no format for topology " + stage.topology;
    const Candidate cand{stage.topology, *format, c.request.spec.context, c.slots, static_cast<std::uint32_t>(stage.unit.size())};
    for (std::uint32_t rank = 0; rank < stage.unit.size(); ++rank) {
        const auto& id = stage.unit[rank];
        const auto facts = c.inventory.facts.find(id);
        if (facts == c.inventory.facts.end()) return "memory of device " + id + " is not measured";
        std::optional<Footprint> partial;
        if (!whole_model) {
            partial = c.provider.range_footprint(c.request.spec, cand, rank, stage.begin, stage.end);
            if (!partial) return "provider states no footprint for layers [" + std::to_string(stage.begin) + "," +
                                 std::to_string(stage.end) + ")";
        }
        auto b = fit_rank(rank, rank, facts->second, c.margin, cand.context, cand.slots, [&] {
            return whole_model ? c.provider.footprints().footprint(c.request.spec, cand, rank) : *partial;
        });
        const bool fits = b.fits;
        const auto reason = b.reason;
        stage.ranks.push_back(std::move(b));
        if (!fits) return "device " + id + ": " + reason;
    }
    return {};
}
// Path between the first devices of two units (rank 0 receives; relay inside a unit is the model's own interconnect).
PathResult path(const Ctx& c, const std::vector<std::string>& from, const std::vector<std::string>& to) {
    const auto key = std::make_pair(from.front(), to.front());
    auto [it, inserted] = c.paths.try_emplace(key);
    if (inserted) it->second = best_path(c.graph, key.first, key.second);
    return it->second;
}
std::string rate_of(const Ctx& c, const LinkPath& p) {
    RateSource worst = RateSource::integrated;
    for (const auto i : p.edges) {
        const auto s = c.graph.edges[i].source;
        if (s == RateSource::nominal || (s == RateSource::measured && worst == RateSource::integrated)) worst = s;
    }
    return std::string(rate_source_name(worst));
}
RouteTransfer transfer(const Ctx& c, std::string what, const std::vector<std::string>& from, const std::vector<std::string>& to,
                       const LinkPath& p, std::uint64_t bytes, double ms) {
    RouteTransfer t{std::move(what), join(from), join(to), {}, bytes, ms, rate_of(c, p)};
    for (const auto i : p.edges) t.edges.push_back(c.graph.edges[i].name);
    return t;
}
std::uint64_t messages(const Protocol& p, std::uint64_t bytes) {
    return p.piece_bytes ? std::max<std::uint64_t>(1, bytes / p.piece_bytes + (bytes % p.piece_bytes != 0)) : 1;
}
// Time of several transfers that run together: each shared resource carries the sum of its transfers' bytes.
double shared_ms(const Ctx& c, const Protocol& protocol,
                 const std::vector<std::pair<LinkPath, std::uint64_t>>& moves, double* uncertainty = nullptr) {
    std::map<std::string, double> busy;
    double rtt = 0, worst_rel = 0;
    for (const auto& [p, bytes] : moves) {
        rtt = std::max(rtt, static_cast<double>(messages(protocol, bytes)) * p.rtt_us / 1e3);
        worst_rel = std::max(worst_rel, p.uncertainty);
        for (const auto i : p.edges) {
            const auto& e = c.graph.edges[i];
            if (e.source == RateSource::integrated) continue;
            busy[e.resource] += static_cast<double>(bytes) / (e.gbytes_ps * 1e6);
        }
    }
    double worst = 0;
    for (const auto& [name, ms] : busy) worst = std::max(worst, ms);
    if (uncertainty) *uncertainty = worst_rel;
    return worst + rtt;
}

// Identical full blocks followed by one possibly short block. Each operation waits for both its
// preceding operation and its own preceding block. Shared link demand also limits block launches.
double pipeline_ms(const std::vector<std::pair<double, double>>& operations, std::uint64_t blocks, double shared) {
    double prefix = 0, cycle = shared, finish = 0;
    for (const auto& [full, last] : operations) {
        prefix += full;
        cycle = std::max(cycle, full);
        const double previous = blocks > 1 ? prefix + static_cast<double>(blocks - 2) * cycle : 0;
        finish = std::max(finish, previous) + last;
    }
    return finish;
}

struct Unit { std::vector<std::string> unit; std::string identity, topology; };
std::vector<Unit> units_with(const StageCostTable& costs, const std::string& identity) {
    std::vector<Unit> out;
    for (const auto& e : costs.entries)
        if (e.identity == identity && std::none_of(out.begin(), out.end(), [&](const Unit& u) { return u.unit == e.unit; }))
            out.push_back({e.unit, e.identity, e.topology});
    return out;
}
// Relative layer work for a split: the entry's own vector, else one of the same identity, else any of the model.
std::optional<std::vector<double>> layer_work(const Ctx& c, const StageCost& entry, std::string& note, bool& borrowed_work) {
    borrowed_work = false;
    const auto n = c.provider.layers();
    if (entry.layers.size() == n) {
        auto w = entry.layers; w.push_back(entry.tail); return w;
    }
    const StageCost* borrowed = nullptr;
    for (const auto& e : c.costs.entries)
        if (e.layers.size() == n && (!borrowed || (e.identity == entry.identity && borrowed->identity != entry.identity))) borrowed = &e;
    if (!borrowed) return std::nullopt;
    note += std::string(note.empty() ? "" : "; ") + "layer work of " + entry.name + " taken from " + borrowed->name;
    borrowed_work = true;
    auto w = borrowed->layers; w.push_back(borrowed->tail);
    return w;
}
} // namespace

// ---- parsing -------------------------------------------------------------------------------------------------------
StageCostTable parse_stage_costs(std::string_view toml) {
    const auto parsed = model_description::parse(toml);
    const auto& root = parsed.fields;
    for (const auto& [k, v] : root)
        if (k != "schema" && k != "model" && k != "costs") bad(k, "unknown key");
    const auto* schema = find(root, "schema");
    if (!schema || text(*schema, "schema") != "strix-stage-costs-v1") bad("schema", "expected strix-stage-costs-v1");
    StageCostTable out;
    if (const auto* m = find(root, "model")) out.model = text(*m, "model");
    if (out.model.empty()) bad("model", "required");
    const auto* entries = find(root, "costs");
    if (!entries) return out;
    static const std::set<std::string_view> keys{"unit", "topology", "identity", "context", "chunk", "total_ms", "fixed_ms",
                                                 "layers", "tail", "spread", "warm", "receipt"};
    for (const auto& [name, value] : table(*entries, "costs")) {
        const auto key = "costs." + name;
        const auto& t = table(value, key);
        for (const auto& [k, v] : t)
            if (!keys.contains(k)) bad(key + "." + k, "unknown key");
        StageCost e;
        e.name = name;
        const auto* unit = find(t, "unit");
        if (!unit) bad(key + ".unit", "required");
        for (const auto& id : array(*unit, key + ".unit")) e.unit.push_back(text(id, key + ".unit"));
        if (!unique_unit(e.unit)) bad(key + ".unit", "needs distinct device ids and at least one device");
        if (const auto* x = find(t, "topology")) e.topology = text(*x, key + ".topology");
        if (e.unit.size() > 1 && !find(t, "topology")) bad(key + ".topology", "required for a unit of several devices");
        const auto* identity = find(t, "identity");
        if (!identity) bad(key + ".identity", "required");
        e.identity = text(*identity, key + ".identity");
        const auto* context = find(t, "context");
        if (!context || !(e.context = count(*context, key + ".context"))) bad(key + ".context", "required and positive");
        if (const auto* x = find(t, "chunk")) e.chunk = static_cast<std::uint32_t>(count(*x, key + ".chunk"));
        const auto* total = find(t, "total_ms");
        if (!total || !((e.total_ms = number(*total, key + ".total_ms")) > 0)) bad(key + ".total_ms", "required and positive");
        if (const auto* x = find(t, "fixed_ms")) e.fixed_ms = number(*x, key + ".fixed_ms");
        if (e.fixed_ms >= e.total_ms) bad(key + ".fixed_ms", "must be below total_ms");
        if (const auto* x = find(t, "layers")) {
            double sum = 0;
            for (const auto& w : array(*x, key + ".layers")) sum += e.layers.emplace_back(number(w, key + ".layers"));
            if (!(sum > 0)) bad(key + ".layers", "needs positive work");
        }
        if (const auto* x = find(t, "tail")) e.tail = number(*x, key + ".tail");
        if (const auto* x = find(t, "spread")) e.spread = number(*x, key + ".spread");
        if (const auto* x = find(t, "warm")) {
            const auto* b = std::get_if<bool>(&x->data);
            if (!b) bad(key + ".warm", "expected true or false");
            e.warm = *b;
        }
        if (const auto* x = find(t, "receipt")) e.receipt = text(*x, key + ".receipt");
        out.entries.push_back(std::move(e));
    }
    return out;
}
StageCostTable load_stage_costs(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot read stage costs " + path);
    std::string data;
    char buffer[4096];
    for (std::size_t n; (n = std::fread(buffer, 1, sizeof buffer, f)) > 0;) data.append(buffer, n);
    std::fclose(f);
    return parse_stage_costs(data);
}

std::string_view objective_name(Objective objective) noexcept { return objective == Objective::ttft ? "ttft" : "throughput"; }
Objective parse_objective(std::string_view text_value) {
    if (text_value == "ttft") return Objective::ttft;
    if (text_value == "throughput") return Objective::throughput;
    throw std::invalid_argument("objective: expected ttft or throughput");
}

// ---- planning ------------------------------------------------------------------------------------------------------
RoutePlan plan_route(const RouteRequest& request, const Inventory& inventory, const RouteProvider& provider,
                     const StageCostTable& costs) {
    RoutePlan plan;
    plan.request = request;
    const auto refuse = [&](std::string reason) { plan.reason = std::move(reason); return plan; };
    try { validate_spec(request.spec); } catch (const std::exception& e) { return refuse(e.what()); }
    if (request.prompt > request.spec.context) return refuse("prompt exceeds admitted context");
    const auto max_slots = provider.footprints().max_slots();
    if (!max_slots || max_slots > 65536) return refuse("provider: invalid maximum slots");
    if (request.spec.slots > max_slots) return refuse("slots: pinned count exceeds provider limit");
    if (!request.decode.empty() && !unique_unit(request.decode)) return refuse("decode unit needs distinct device ids");
    for (const auto& e : costs.entries)
        if (!unique_unit(e.unit)) return refuse("cost unit needs distinct device ids");
    const auto protocols = provider.protocols();
    for (const auto& p : protocols)
        if (std::find(p.chunks.begin(), p.chunks.end(), 0U) != p.chunks.end()) return refuse("protocol chunks must be positive");
    // Fail closed rather than select a winner from a truncated search. Limits also bound recursion depth.
    if (provider.layers() > 4096 || inventory.declaration.devices.size() > 256 || costs.entries.size() > 4096 ||
        protocols.size() > 128) return refuse("route planning size limit exceeded");
    for (const auto& p : protocols)
        if (p.stages.size() > 64 || p.chunks.size() > 64) return refuse("route planning size limit exceeded");
    std::size_t work_left = 100000;
    bool exhausted = false;
    const auto spend = [&] { if (work_left) { --work_left; return true; } exhausted = true; return false; };
    Ctx c{request, inventory, provider, costs, link_graph(inventory.declaration), 0, 0, 1, {}};
    c.rows = request.prompt ? request.prompt : request.spec.context;
    c.slots = request.spec.slots ? request.spec.slots : 1;
    c.margin = static_cast<std::uint64_t>(std::ceil(request.spec.memory.margin_gib * 1073741824.0));
    const auto model = provider.footprints().model();
    if (!costs.model.empty() && costs.model != model) { plan.reason = "stage costs are for model " + costs.model + ", not " + model; return plan; }
    if (!request.spec.model.empty() && request.spec.model != model) { plan.reason = "provider does not serve model " + request.spec.model; return plan; }
    const auto L = provider.layers();
    if (L < 1) { plan.reason = "provider states no layers"; return plan; }
    const double n = static_cast<double>(c.rows);

    // Common closing: uncertainty, gate, notes.
    const auto finish = [&](RouteOption& o, const Protocol& p, std::string refusal, const std::string& notes) {
        o.gate = p.gate;
        if (refusal.empty() && (p.gate.empty() || p.gate == "pending"))
            refusal = "not qualified: protocol " + p.identity + (p.gate.empty() ? " has no gate receipt" : " gate pending");
        o.selectable = refusal.empty() && o.predicted;
        o.reason = refusal.empty() ? notes : refusal + (notes.empty() ? "" : " (" + notes + ")");
        plan.options.push_back(std::move(o));
    };

    for (const auto& p : protocols) {
        if (!spend()) break;
        if (p.kind == "local" && p.stages.size() == 1) {
            for (const auto& u : units_with(costs, p.stages[0])) {
                if (!spend()) break;
                RouteOption o;
                o.kind = "local"; o.protocol = p.name; o.identity = p.identity; o.decode = u.unit;
                o.label = "local:" + join(u.unit) + "/" + p.identity;
                RouteStage s{u.unit, u.identity, u.topology, "", 0, L, 0, {}};
                std::string refusal, notes;
                if (!request.decode.empty() && request.decode != u.unit) refusal = "decode is placed on " + join(request.decode);
                const auto est = estimate(entries_for(costs, join(u.unit), u.identity, 0), c.rows);
                if (est) {
                    s.cost = est->entry->name; s.ms = est->ms; notes = est->note;
                    o.predicted = true; o.predicted_ms = est->ms; o.uncertainty_ms = est->ms * est->rel;
                    o.rows_per_s = n / est->ms * 1e3;
                }
                if (refusal.empty()) refusal = role_check(c, u.unit, false, true);
                if (refusal.empty()) refusal = load_check(c, u.unit);
                if (refusal.empty()) refusal = memory_check(c, s, true);
                o.stages.push_back(std::move(s));
                finish(o, p, refusal, notes);
            }
        } else if (p.kind == "handoff" && p.stages.size() == 2) {
            const auto producers = units_with(costs, p.stages[0]);
            auto consumers = units_with(costs, p.stages[1]);
            if (!request.decode.empty())
                consumers.erase(std::remove_if(consumers.begin(), consumers.end(), [&](const Unit& u) { return u.unit != request.decode; }),
                                consumers.end());
            for (const auto& pu : producers)
                for (const auto& cu : consumers) {
                    if (!spend()) break;
                    if (overlaps(pu.unit, cu.unit)) continue; // no shared-ownership contract
                    RouteOption o;
                    o.kind = "handoff"; o.protocol = p.name; o.identity = p.identity; o.decode = cu.unit;
                    o.label = "handoff:" + join(pu.unit) + "->" + join(cu.unit) + "/" + p.identity;
                    RouteStage s0{pu.unit, pu.identity, pu.topology, "", 0, L, 0, {}}, s1{cu.unit, cu.identity, cu.topology, "", 0, 0, 0, {}};
                    std::string refusal, notes;
                    const auto est = estimate(entries_for(costs, join(pu.unit), pu.identity, 0), c.rows);
                    const auto route = path(c, pu.unit, cu.unit);
                    const auto bytes = provider.state_bytes(c.rows, 0, L);
                    if (!route.path) refusal = "no link " + join(pu.unit) + " -> " + join(cu.unit) + ": " + route.reason;
                    if (est && route.path) {
                        s0.cost = est->entry->name; s0.ms = est->ms;
                        const double x = shared_ms(c, p, {{*route.path, bytes}});
                        const double import_ms = p.import_ms < 0 ? 0 : p.import_ms + p.import_ms_per_mb * static_cast<double>(bytes) / 1e6;
                        o.transfers.push_back(transfer(c, "state", pu.unit, cu.unit, *route.path, bytes, x));
                        o.bytes = bytes;
                        o.predicted = true;
                        o.predicted_ms = est->ms + x + import_ms;
                        o.uncertainty_ms = est->ms * est->rel + x * route.path->uncertainty + import_ms * 0.15;
                        o.rows_per_s = n / std::max({est->ms, x, import_ms}) * 1e3;
                        notes = est->note + "; state " + std::to_string(bytes) + " B over " + o.transfers.back().rate + " link";
                        if (p.import_ms < 0) refusal = "import cost of " + p.identity + " is not measured";
                    }
                    if (refusal.empty()) refusal = role_check(c, pu.unit, true, false);
                    if (refusal.empty()) refusal = role_check(c, cu.unit, false, true);
                    if (refusal.empty()) refusal = load_check(c, pu.unit);
                    if (refusal.empty()) refusal = memory_check(c, s0, true);
                    if (refusal.empty()) refusal = memory_check(c, s1, true);
                    o.stages.push_back(std::move(s0));
                    o.stages.push_back(std::move(s1));
                    finish(o, p, refusal, notes);
                }
        } else if (p.kind == "layers" && p.stages.size() >= 2 && p.stages.size() <= L) {
            const auto S = p.stages.size();
            std::vector<std::vector<Unit>> choice(S);
            for (std::size_t s = 0; s < S; ++s) choice[s] = units_with(costs, p.stages[s]);
            std::vector<std::uint32_t> legal;
            for (std::uint32_t k = 1; k < L; ++k)
                if (provider.seam_bytes(k, c.rows)) legal.push_back(k);
            std::vector<Unit> pick(S);
            std::function<void(std::size_t)> assign = [&](std::size_t s) {
                if (!spend()) return;
                if (s < S) {
                    for (const auto& u : choice[s]) {
                        if (!spend()) return;
                        bool reused = false;
                        for (std::size_t q = 0; q < s; ++q)
                            for (const auto& id : u.unit)
                                if (std::find(pick[q].unit.begin(), pick[q].unit.end(), id) != pick[q].unit.end()) reused = true;
                        if (reused) continue;
                        pick[s] = u;
                        assign(s + 1);
                    }
                    return;
                }
                const auto decode = request.decode.empty() ? pick.back().unit : request.decode;
                const auto chunks = p.chunks.empty() ? std::vector<std::uint32_t>{static_cast<std::uint32_t>(c.rows)} : p.chunks;
                for (const auto chunk : chunks) {
                    if (!spend()) return;
                    // Per-stage per-row rate at this prompt and relative layer work.
                    std::vector<Estimate> est(S);
                    std::vector<std::vector<double>> work(S);
                    std::vector<char> borrowed(S, 0);
                    std::string notes, missing;
                    for (std::size_t q = 0; q < S && missing.empty(); ++q) {
                        auto points = entries_for(costs, join(pick[q].unit), pick[q].identity, chunk);
                        if (points.empty()) {
                            points = entries_for(costs, join(pick[q].unit), pick[q].identity, 0);
                            notes += (notes.empty() ? "" : "; ") + std::string("no matching chunk calibration for ") + join(pick[q].unit);
                        }
                        const auto e = estimate(points, c.rows);
                        if (!e) { missing = "no stage cost for " + join(pick[q].unit) + "/" + pick[q].identity; break; }
                        est[q] = *e;
                        bool lent = false;
                        auto w = layer_work(c, *e->entry, notes, lent);
                        borrowed[q] = lent ? 1 : 0;
                        if (!w) missing = "no per-layer work for " + join(pick[q].unit) + ": a layer split cannot be predicted";
                        else work[q] = std::move(*w);
                    }
                    // Per-layer ms per row on each stage's own unit (heterogeneous costs for pair::predict_chain).
                    std::vector<std::vector<double>> per_layer(S);
                    std::vector<std::span<const double>> spans(S);
                    double tail = 0;
                    for (std::size_t q = 0; q < S && missing.empty(); ++q) {
                        double all = 0;
                        for (const auto w : work[q]) all += w;
                        const double scale = (est[q].ms - est[q].fixed_ms) / n / all;
                        per_layer[q].assign(work[q].begin(), work[q].begin() + L);
                        for (auto& w : per_layer[q]) w *= scale;
                        spans[q] = per_layer[q];
                        if (q + 1 == S) tail = work[q][L] * scale;
                    }
                    RouteOption best;
                    double best_score = std::numeric_limits<double>::infinity();
                    int best_rank = 3;
                    std::size_t evaluated = 0;
                    std::vector<std::uint32_t> cuts(S - 1);
                    const auto evaluate = [&] {
                        ++evaluated;
                        RouteOption o;
                        o.kind = "layers"; o.protocol = p.name; o.identity = p.identity; o.decode = decode; o.chunk = chunk;
                        const auto C = std::min<std::uint64_t>(chunk, c.rows);
                        const auto m = (c.rows + C - 1) / C;
                        const auto last = c.rows - (m - 1) * C;
                        std::string refusal;
                        double bottleneck = 0, fixed = 0, unc = 0;
                        std::vector<std::pair<double, double>> operations;
                        std::vector<std::pair<LinkPath, std::uint64_t>> per_block;
                        std::vector<std::uint32_t> ends(cuts);
                        ends.push_back(L);
                        const auto stage_rate = pair::predict_chain(spans, ends, tail);
                        for (std::size_t q = 0; q < S; ++q) {
                            const auto begin = q ? cuts[q - 1] : 0U, end = q + 1 < S ? cuts[q] : L;
                            const double rate = stage_rate[q]; // ms per row
                            RouteStage st{pick[q].unit, pick[q].identity, pick[q].topology, est[q].entry->name, begin, end,
                                          rate * static_cast<double>(C), {}};
                            bottleneck = std::max(bottleneck, st.ms);
                            operations.emplace_back(st.ms, rate * static_cast<double>(last));
                            fixed = std::max(fixed, est[q].fixed_ms);
                            // A stage rate taken from a whole-model receipt (not a block receipt at this size) and
                            // borrowed layer work are hypotheses: each widens the stage's uncertainty by 10 %.
                            const double derived = (est[q].entry->chunk == chunk ? 0.0 : 0.10) + (borrowed[q] ? 0.10 : 0.0);
                            unc += rate * n * (est[q].rel + derived);
                            o.label += (q ? "|" : "") + join(st.unit) + "[" + std::to_string(begin) + "," + std::to_string(end) + ")";
                            o.stages.push_back(std::move(st));
                            if (q + 1 < S) {
                                const auto seam = provider.seam_bytes(cuts[q], c.rows).value_or(0);
                                const auto route = path(c, pick[q].unit, pick[q + 1].unit);
                                if (!route.path) {
                                    if (refusal.empty()) refusal = "no link " + join(pick[q].unit) + " -> " + join(pick[q + 1].unit) + ": " + route.reason;
                                    continue;
                                }
                                const double x = shared_ms(c, p, {{*route.path, seam * C}});
                                operations.emplace_back(x, shared_ms(c, p, {{*route.path, seam * last}}));
                                unc += x * static_cast<double>(m) * route.path->uncertainty;
                                o.transfers.push_back(transfer(c, "seam", pick[q].unit, pick[q + 1].unit, *route.path, seam * c.rows, x));
                                o.bytes += seam * c.rows;
                                per_block.emplace_back(*route.path, seam * C);
                            }
                        }
                        o.label = "layers:" + o.label + "@" + std::to_string(chunk) + "/" + p.identity;
                        double rel = 0;
                        const double shared = shared_ms(c, p, per_block, &rel);
                        bottleneck = std::max(bottleneck, shared);
                        // Final state to the decoder: every stage not on it sends its layers' state; together on shared links.
                        std::vector<std::pair<LinkPath, std::uint64_t>> state;
                        std::uint64_t state_total = 0;
                        for (std::size_t q = 0; q < S; ++q) {
                            if (pick[q].unit == decode) continue;
                            const auto bytes = provider.state_bytes(c.rows, o.stages[q].begin, o.stages[q].end);
                            const auto route = path(c, pick[q].unit, decode);
                            if (!route.path) {
                                if (refusal.empty()) refusal = "no link " + join(pick[q].unit) + " -> " + join(decode) + ": " + route.reason;
                                continue;
                            }
                            o.transfers.push_back(transfer(c, "state", pick[q].unit, decode, *route.path, bytes,
                                                           shared_ms(c, p, {{*route.path, bytes}})));
                            state.emplace_back(*route.path, bytes);
                            state_total += bytes;
                        }
                        o.bytes += state_total;
                        double state_rel = 0;
                        const double state_ms = shared_ms(c, p, state, &state_rel);
                        const double import_ms = state.empty() || p.import_ms < 0 ? 0
                            : p.import_ms + p.import_ms_per_mb * static_cast<double>(state_total) / 1e6;
                        if (!state.empty() && p.import_ms < 0 && refusal.empty()) refusal = "import cost of " + p.identity + " is not measured";
                        o.predicted = refusal.empty() || refusal.rfind("import", 0) == 0;
                        o.predicted_ms = pipeline_ms(operations, m, shared) + fixed + state_ms + import_ms;
                        o.uncertainty_ms = unc + state_ms * state_rel + import_ms * 0.15;
                        o.rows_per_s = std::min(static_cast<double>(C) / bottleneck,
                                                state.empty() ? std::numeric_limits<double>::infinity() : n / (state_ms + import_ms)) * 1e3;
                        // Memory depends on the ranges: checked per cut, so a cut that does not fit never hides one that does.
                        for (std::size_t q = 0; q < S && refusal.empty(); ++q)
                            refusal = memory_check(c, o.stages[q], o.stages[q].unit == decode);
                        o.reason = refusal;
                        // Rank: clean before refused before unpredicted; then the objective.
                        const int rank = !o.predicted ? 2 : refusal.empty() ? 0 : 1;
                        const double value = request.objective == Objective::ttft ? o.predicted_ms : -o.rows_per_s;
                        if ((!request.pin.empty() && o.label == request.pin) || ((request.pin.empty() || best.label != request.pin) &&
                            (evaluated == 1 || rank < best_rank || (rank == best_rank && value < best_score)))) {
                            best_rank = rank;
                            best_score = value;
                            best = std::move(o);
                        }
                    };
                    std::function<void(std::size_t, std::size_t)> cut = [&](std::size_t q, std::size_t from) {
                        if (!spend()) return;
                        if (q + 1 == S) { evaluate(); return; }
                        for (std::size_t i = from; i < legal.size() && !exhausted; ++i) { cuts[q] = legal[i]; cut(q + 1, i + 1); }
                    };
                    if (missing.empty() && legal.size() + 1 >= S) cut(0, 0);
                    if (!evaluated) {
                        best.kind = "layers"; best.protocol = p.name; best.identity = p.identity; best.decode = decode; best.chunk = chunk;
                        std::string label;
                        for (std::size_t q = 0; q < S; ++q) label += (q ? "|" : "") + join(pick[q].unit);
                        best.label = "layers:" + label + "@" + std::to_string(chunk) + "/" + p.identity;
                        finish(best, p, missing.empty() ? "no legal cuts for " + std::to_string(S) + " stages" : missing, notes);
                        continue;
                    }
                    std::string refusal = best.reason;
                    const auto stage_notes = notes + (notes.empty() ? "" : "; ") + "balanced over " + std::to_string(evaluated) +
                        " legal cut sets; blocks " + std::to_string((c.rows + chunk - 1) / std::max<std::uint32_t>(chunk, 1));
                    for (std::size_t q = 0; q < S && refusal.empty(); ++q) {
                        refusal = role_check(c, best.stages[q].unit, true, best.stages[q].unit == decode);
                        if (refusal.empty()) refusal = load_check(c, best.stages[q].unit);
                    }
                    if (refusal.empty() && std::none_of(pick.begin(), pick.end(), [&](const Unit& u) { return u.unit == decode; })) {
                        std::string topology = decode.size() == 1 ? "single" : "";
                        for (const auto& e : costs.entries)
                            if (e.unit == decode) topology = e.topology;
                        RouteStage d{decode, p.stages.back(), topology, "", 0, 0, 0, {}};
                        refusal = topology.empty() ? "no stage cost states the topology of decode unit " + join(decode)
                                                   : role_check(c, decode, false, true);
                        for (const auto& u : pick)
                            if (overlaps(u.unit, decode)) refusal = "decoder overlaps a stage without shared ownership";
                        if (refusal.empty()) refusal = memory_check(c, d, true);
                        best.stages.push_back(std::move(d)); // explicit decoder admission, serialized with every rank
                    }
                    finish(best, p, refusal, stage_notes);
                }
            };
            assign(0);
        } else {
            RouteOption o;
            o.kind = p.kind; o.protocol = p.name; o.identity = p.identity; o.label = p.kind + ":?/" + p.identity;
            finish(o, p, "protocol " + p.identity + " has an unknown kind or stage count", "");
        }
    }

    if (exhausted) { plan.options.clear(); return refuse("route planning work limit exceeded"); }
    // Selection: the objective over selectable options; enumeration order breaks ties.
    const auto better = [&](const RouteOption& a, const RouteOption& b) {
        return request.objective == Objective::ttft ? a.predicted_ms < b.predicted_ms : a.rows_per_s > b.rows_per_s;
    };
    if (!request.pin.empty()) {
        const auto it = std::find_if(plan.options.begin(), plan.options.end(), [&](const RouteOption& o) { return o.label == request.pin; });
        if (it == plan.options.end()) plan.reason = "pin " + request.pin + ": no such option";
        else if (!it->selectable) plan.reason = "pin " + request.pin + ": " + it->reason;
        else { plan.chosen = static_cast<int>(it - plan.options.begin()); plan.reason = "pinned by the request"; }
        return plan;
    }
    for (std::size_t i = 0; i < plan.options.size(); ++i)
        if (plan.options[i].selectable && (plan.chosen < 0 || better(plan.options[i], plan.options[static_cast<std::size_t>(plan.chosen)])))
            plan.chosen = static_cast<int>(i);
    if (plan.chosen < 0) { plan.reason = plan.options.empty() ? "provider announces no protocol" : "no option is selectable"; return plan; }
    const auto& w = plan.options[static_cast<std::size_t>(plan.chosen)];
    plan.reason = std::string(objective_name(request.objective)) + ": " + w.label + " " + ms_text(w.predicted_ms) + " +- " +
                  ms_text(w.uncertainty_ms) + ", " + num(w.rows_per_s) + " rows/s";
    const RouteOption* next = nullptr;
    for (const auto& o : plan.options)
        if (&o != &w && o.selectable && (!next || better(o, *next))) next = &o;
    if (next) {
        plan.reason += "; next " + next->label + " " + ms_text(next->predicted_ms);
        if (std::abs(next->predicted_ms - w.predicted_ms) <= w.uncertainty_ms + next->uncertainty_ms)
            plan.reason += " (within the combined uncertainty: a receipt should confirm the order)";
    }
    return plan;
}

// ---- output --------------------------------------------------------------------------------------------------------
namespace {
std::string strings(const std::vector<std::string>& v) {
    std::string out = "[";
    for (std::size_t i = 0; i < v.size(); ++i) out += (i ? "," : "") + quote(v[i]);
    return out + "]";
}
std::string canonical(const RoutePlan& p) {
    const auto& r = p.request;
    std::string o = "{\"schema\":\"strix-route-v1\",\"request\":{\"model\":" + quote(r.spec.model) +
        ",\"prompt\":" + std::to_string(r.prompt ? r.prompt : r.spec.context) + ",\"context\":" + std::to_string(r.spec.context) +
        ",\"slots\":" + std::to_string(r.spec.slots) + ",\"margin_gib\":" + num(r.spec.memory.margin_gib) +
        ",\"objective\":" + quote(std::string(objective_name(r.objective))) + ",\"decode\":" + strings(r.decode) +
        ",\"pin\":" + quote(r.pin) + "},\"options\":[";
    for (std::size_t i = 0; i < p.options.size(); ++i) {
        const auto& x = p.options[i];
        o += std::string(i ? "," : "") + "{\"label\":" + quote(x.label) + ",\"kind\":" + quote(x.kind) + ",\"protocol\":" +
             quote(x.protocol) + ",\"identity\":" + quote(x.identity) + ",\"gate\":" + quote(x.gate) + ",\"decode\":" +
             strings(x.decode) + ",\"chunk\":" + std::to_string(x.chunk) + ",\"bytes\":" + std::to_string(x.bytes) +
             ",\"predicted\":" + (x.predicted ? "true" : "false") + ",\"predicted_ms\":" + num(x.predicted_ms) +
             ",\"uncertainty_ms\":" + num(x.uncertainty_ms) + ",\"rows_per_s\":" + num(x.rows_per_s) +
             ",\"selectable\":" + (x.selectable ? "true" : "false") + ",\"reason\":" + quote(x.reason) + ",\"stages\":[";
        for (std::size_t s = 0; s < x.stages.size(); ++s) {
            const auto& st = x.stages[s];
            o += std::string(s ? "," : "") + "{\"unit\":" + strings(st.unit) + ",\"identity\":" + quote(st.identity) +
                 ",\"topology\":" + quote(st.topology) + ",\"cost\":" + quote(st.cost) + ",\"layers\":[" +
                 std::to_string(st.begin) + "," + std::to_string(st.end) + "],\"ms\":" + num(st.ms) + ",\"ranks\":[";
            for (std::size_t k = 0; k < st.ranks.size(); ++k)
                o += std::string(k ? "," : "") + "{\"required_bytes\":" + std::to_string(st.ranks[k].required_bytes) +
                     ",\"budget_bytes\":" + std::to_string(st.ranks[k].budget_bytes) + ",\"fits\":" +
                     (st.ranks[k].fits ? "true" : "false") + "}";
            o += "]}";
        }
        o += "],\"transfers\":[";
        for (std::size_t t = 0; t < x.transfers.size(); ++t) {
            const auto& tr = x.transfers[t];
            o += std::string(t ? "," : "") + "{\"what\":" + quote(tr.what) + ",\"from\":" + quote(tr.from) + ",\"to\":" +
                 quote(tr.to) + ",\"edges\":" + strings(tr.edges) + ",\"bytes\":" + std::to_string(tr.bytes) +
                 ",\"ms\":" + num(tr.ms) + ",\"rate\":" + quote(tr.rate) + "}";
        }
        o += "]}";
    }
    o += "],\"chosen\":" + (p.chosen >= 0 ? quote(p.options[static_cast<std::size_t>(p.chosen)].label) : std::string("null")) +
         ",\"reason\":" + quote(p.reason) + "}";
    return o;
}
} // namespace

std::string RoutePlan::hash() const {
    std::uint64_t h = 1469598103934665603ULL;
    for (const unsigned char ch : canonical(*this)) { h ^= ch; h *= 1099511628211ULL; }
    char out[17];
    std::snprintf(out, sizeof out, "%016llx", static_cast<unsigned long long>(h));
    return out;
}
std::string RoutePlan::to_json() const {
    auto json = canonical(*this);
    json.pop_back();
    return json + ",\"route_id\":" + quote(hash()) + "}";
}

std::vector<Contrast> contrast(const RoutePlan& plan, const std::vector<Receipt>& receipts) {
    std::vector<Contrast> out;
    for (const auto& r : receipts) {
        Contrast x;
        x.label = r.label;
        x.measured_ms = r.measured_ms;
        for (const auto& o : plan.options)
            if (o.label == r.label && o.predicted) {
                x.found = true;
                x.predicted_ms = o.predicted_ms;
                x.uncertainty_ms = o.uncertainty_ms;
                x.error = r.measured_ms / o.predicted_ms - 1.0;
                x.within = std::abs(r.measured_ms - o.predicted_ms) <= o.uncertainty_ms;
                x.explain = !x.within && std::abs(x.error) > 0.30;
            }
        out.push_back(x);
    }
    return out;
}

} // namespace strix::deploy
