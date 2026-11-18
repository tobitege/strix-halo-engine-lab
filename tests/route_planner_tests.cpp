// Host tests of the shared route planner (deploy/route.hpp) with a synthetic inventory: one discrete GPU host and
// a two-node pair reached over the network. Rates are the published receipts quoted in the design notes; models
// are stand-ins with the stated seam and state sizes, not engine code.
#include "strix/deploy/route.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#include <crtdbg.h>
#endif

namespace {
using namespace strix::deploy;
constexpr std::uint64_t GiB = 1ULL << 30;
void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F f, const char* what) {
    bool rejected = false;
    try { (void)f(); } catch (const std::exception&) { rejected = true; }
    require(rejected, std::string("accepted invalid input: ") + what);
}
std::string replace(std::string text, const std::string& from, const std::string& to) {
    const auto at = text.find(from);
    if (at == std::string::npos) throw std::runtime_error("fixture line not found: " + from);
    return text.replace(at, from.size(), to);
}

// The one line that states the link to the pair.
const std::string kLink1g = R"(attach = { kind = "remote", link = "1gbe" })";

std::string declaration(const std::string& link_line) {
    return R"(schema = "strix-devices-v1"
machine = "synthetic"
[devices.rtx]
vendor = "nvidia"
uuid = "GPU-00000000-0000-0000-0000-000000000001"
roles = ["prefill"]
attach = { kind = "pcie", pcie_gen = 5, pcie_width = 16, h2d_gbytes_ps = 25.0 }
[devices.halo-a]
vendor = "amd"
pci = "0000:c3:00.0"
roles = ["pair", "decode", "prefill"]
)" + link_line + R"(
[devices.halo-b]
vendor = "amd"
pci = "0001:c3:00.0"
roles = ["pair", "decode", "prefill"]
)" + link_line + R"(
[links.pair]
between = ["halo-a", "halo-b"]
class = "usb4"
)";
}

DeviceFacts facts(std::uint64_t total, std::uint64_t free_bytes) {
    DeviceFacts f;
    f.mem_total_bytes = total; f.mem_free_bytes = free_bytes; f.source = "synthetic";
    return f;
}
Inventory inventory(const std::string& link_line) {
    Inventory inv;
    inv.declaration = strix::device::parse_declaration(declaration(link_line));
    inv.facts["rtx"] = facts(96 * GiB, 90 * GiB);
    inv.facts["halo-a"] = facts(117 * GiB, 110 * GiB);
    inv.facts["halo-b"] = facts(117 * GiB, 110 * GiB);
    return inv;
}

class Footprints final : public FootprintProvider {
public:
    Footprints(std::string model, std::uint64_t weights) : weights_(weights), model_(std::move(model)) {}
    std::string model() const override { return model_; }
    std::vector<std::string> topologies() const override { return {"single", "tp2", "pipeline"}; }
    std::vector<Format> formats(const std::string&) const override { return {{"q", "q", 0, 0, 0}}; }
    Footprint footprint(const Spec&, const Candidate& c, std::uint32_t) const override {
        Footprint f;
        f.weights = c.ranks > 1 && c.topology == "tp2" ? weights_ / 2 : weights_;
        f.kv_per_token = 4096; f.workspace = 2 * GiB;
        return f;
    }
    std::uint64_t weights_;
private:
    std::string model_;
};

// Stand-in model: L layers, constant seam bytes, state = a*rows + b split evenly over layers.
class Model final : public RouteProvider {
public:
    Model(std::string name, std::uint32_t layers, std::uint64_t weights, std::uint64_t seam, double state_a, double state_b,
          std::vector<Protocol> protocols)
        : protocols_(std::move(protocols)), fp_(std::move(name), weights), layers_(layers), seam_(seam), a_(state_a), b_(state_b) {}
    const FootprintProvider& footprints() const override { return fp_; }
    std::uint32_t layers() const override { return layers_; }
    std::optional<std::uint64_t> seam_bytes(std::uint32_t, std::uint64_t) const override { return seam_; }
    std::uint64_t state_bytes(std::uint64_t rows, std::uint32_t begin, std::uint32_t end) const override {
        return static_cast<std::uint64_t>((a_ * static_cast<double>(rows) + b_) * (end - begin) / layers_);
    }
    std::vector<Protocol> protocols() const override { return protocols_; }
    std::optional<Footprint> range_footprint(const Spec& s, const Candidate& c, std::uint32_t rank, std::uint32_t begin,
                                             std::uint32_t end) const override {
        auto f = fp_.footprint(s, c, rank);
        f.weights = f.weights * (end - begin) / layers_;
        return f;
    }
    std::vector<Protocol> protocols_;
private:
    Footprints fp_;
    std::uint32_t layers_;
    std::uint64_t seam_;
    double a_, b_;
};

// DS-like: RTX 2K 4.66 s / 32K 11.65 s; pair 2K 1.7 s / 32K 19.65 s (FAST); state ~894 B/row + 2.9 MB; seam 85.5 KB.
const char* kDsCosts = R"(schema = "strix-stage-costs-v1"
model = "ds-like"
[costs.rtx-2k]
unit = ["rtx"]
identity = "ds-cuda"
context = 2048
total_ms = 4660.0
spread = 0.02
[costs.rtx-32k]
unit = ["rtx"]
identity = "ds-cuda"
context = 32768
total_ms = 11650.0
spread = 0.02
[costs.pair-2k]
unit = ["halo-a", "halo-b"]
topology = "tp2"
identity = "ds-hip"
context = 2048
total_ms = 1700.0
spread = 0.03
[costs.pair-32k]
unit = ["halo-a", "halo-b"]
topology = "tp2"
identity = "ds-hip"
context = 32768
total_ms = 19650.0
spread = 0.03
)";
Model ds_model() {
    return Model("ds-like", 40, 80 * GiB, 85500, 894.0, 2.9e6, {
        {"local", "none", "ds-pair-v1", {"ds-hip"}, {}, "pair-gate"},
        {"local", "none", "ds-cuda-v1", {"ds-cuda"}, {}, "cuda-gate"},
        {"handoff", "handoff-sho1-v2/bp-v1", "ds-cuda-prefill-hip-v1", {"ds-cuda", "ds-hip"}, {}, "synthetic-gate", 300.0, 0.0, 0},
    });
}

// Qwen-like: RTX 32K 7.65 s, one node 17.7 s, pair pipeline 11.04 s; seam 25,616 B/row; state 29,952 B/row + 116.4 MB.
std::string qwen_costs() {
    std::string uniform = "[";
    for (int i = 0; i < 48; ++i) uniform += std::string(i ? ", " : "") + "1.0";
    uniform += "]";
    return R"(schema = "strix-stage-costs-v1"
model = "qwen-like"
[costs.rtx]
unit = ["rtx"]
identity = "q-cuda"
context = 32768
total_ms = 7650.0
spread = 0.03
layers = )" + uniform + R"(
[costs.node-a]
unit = ["halo-a"]
identity = "q-hip"
context = 32768
total_ms = 17700.0
spread = 0.03
[costs.node-b]
unit = ["halo-b"]
identity = "q-hip"
context = 32768
total_ms = 17700.0
spread = 0.03
[costs.pair]
unit = ["halo-a", "halo-b"]
topology = "pipeline"
identity = "q-hip-pair"
context = 32768
total_ms = 11040.0
spread = 0.03
)";
}
Model qwen_model(const std::string& layers_gate) {
    return Model("qwen-like", 48, 40 * GiB, 25616, 29952.0, 116.4e6, {
        {"local", "none", "q-pair-v1", {"q-hip-pair"}, {}, "pair-gate"},
        {"handoff", "handoff-sho1-v2/bp-v1", "q-cuda-hip-handoff-v1", {"q-cuda", "q-hip"}, {}, "handoff-gate", 200.0, 0.0, 0},
        {"layers", "pipe-crc-ack-v1", "q-cuda-hip-layer-v1", {"q-cuda", "q-hip", "q-hip"}, {2048, 4096, 8192}, layers_gate, 200.0, 0.0, 0},
    });
}

RouteRequest request(const std::string& model, std::uint64_t prompt, std::vector<std::string> decode) {
    RouteRequest r;
    r.spec.model = model;
    r.spec.context = 32768;
    r.spec.slots = 1;
    r.spec.memory.margin_gib = 4;
    r.prompt = prompt;
    r.decode = std::move(decode);
    return r;
}
const RouteOption& option(const RoutePlan& p, const std::string& prefix) {
    for (const auto& o : p.options)
        if (o.label.rfind(prefix, 0) == 0) return o;
    throw std::runtime_error("no option " + prefix + " in plan: " + p.to_json().substr(0, 400));
}
std::string chosen(const RoutePlan& p) { return p.selected() ? p.selected()->label : "<none: " + p.reason + ">"; }

void declaration_tests() {
    const auto d = strix::device::parse_declaration(declaration(kLink1g));
    require(d.links.size() == 1 && d.links[0].link == "usb4" && d.links[0].port == "link-pair", "peer link parsed");
    const auto dev = [&](const std::string& id) -> const strix::device::Descriptor& {
        for (const auto& x : d.devices) if (x.id == id) return x;
        throw std::runtime_error("no device " + id);
    };
    require(dev("halo-a").attach.link == "1gbe" && dev("halo-a").attach.port == "attach-remote", "remote attach defaults to shared port");
    require(dev("rtx").attach.port == "attach-rtx", "pcie attach is its own resource");
    rejects([] { return strix::device::parse_declaration(declaration(R"(attach = { kind = "remote", link = "3gbe" })")); }, "unknown class");
    rejects([] { return strix::device::parse_declaration(replace(declaration(kLink1g), R"(["halo-a", "halo-b"])", R"(["halo-a", "nope"])")); },
            "link to an undeclared device");
    const auto g = link_graph(d);
    const auto p = best_path(g, "rtx", "halo-b");
    require(p.path && p.path->edges.size() == 2 && std::abs(p.path->gbytes_ps - 0.118) < 1e-9, "rtx->halo-b via host NIC at 1 GbE");
    const auto unknown = link_graph(strix::device::parse_declaration(declaration(R"(attach = { kind = "remote" })")));
    const auto none = best_path(unknown, "rtx", "halo-a");
    require(!none.path && none.reason.find("no measured rate and no link class") != std::string::npos, "unknown link is not traversed");
    rejects([] { return parse_stage_costs("schema = \"strix-stage-costs-v1\"\nmodel = \"m\"\n[costs.x]\nunit = [\"a\", \"b\"]\n"
                                          "identity = \"i\"\ncontext = 1\ntotal_ms = 1.0\n"); }, "group without topology");
}

void ds_tests() {
    const auto model = ds_model();
    const auto costs = parse_stage_costs(kDsCosts);
    const auto inv = inventory(kLink1g);
    // 32K: hand-off beats the pair (prediction in the design: ~12.2 s against 19.4-19.9 s FAST).
    auto p = plan_route(request("ds-like", 32768, {"halo-a", "halo-b"}), inv, model, costs);
    const auto h = option(p, "handoff:rtx->halo-a+halo-b");
    require(chosen(p) == h.label, "ds 32K chooses the hand-off, got " + chosen(p));
    require(std::abs(h.predicted_ms - 12223) < 20, "ds 32K hand-off ~12.2 s, got " + std::to_string(h.predicted_ms));
    require(h.bytes == static_cast<std::uint64_t>(894.0 * 32768 + 2.9e6), "state bytes reported");
    require(h.transfers.size() == 1 && h.transfers[0].rate == "nominal" && h.transfers[0].edges.size() == 2, "nominal 1 GbE path via host");
    require(option(p, "local:rtx").reason.find("decode is placed on") == 0, "rtx local refused: decode is on the pair");
    // 2K: the pair's short-prompt cost wins (no extension of the long-prompt gain to short prompts).
    p = plan_route(request("ds-like", 2048, {"halo-a", "halo-b"}), inv, model, costs);
    require(chosen(p).rfind("local:halo-a+halo-b", 0) == 0, "ds 2K chooses the pair, got " + chosen(p));
    // Throughput objective: the hand-off's producer bound (2,813 rows/s) beats the pair (1,668).
    auto t = request("ds-like", 32768, {"halo-a", "halo-b"});
    t.objective = Objective::throughput;
    p = plan_route(t, inv, model, costs);
    require(chosen(p).rfind("handoff:", 0) == 0 && std::abs(p.selected()->rows_per_s - 2812.7) < 2, "ds throughput objective");
    // Unmeasured import: predicted, never selected.
    auto m2 = model;
    m2.protocols_[2].import_ms = -1;
    p = plan_route(request("ds-like", 32768, {"halo-a", "halo-b"}), inv, m2, costs);
    require(chosen(p).rfind("local:halo-a+halo-b", 0) == 0, "unmeasured import never selected");
    require(option(p, "handoff:").predicted && !option(p, "handoff:").selectable, "unmeasured import still predicted");
    // Unknown memory and memory that does not fit, through FootprintProvider + fit_rank.
    auto inv2 = inv;
    inv2.facts.erase("rtx");
    p = plan_route(request("ds-like", 32768, {"halo-a", "halo-b"}), inv2, model, costs);
    require(option(p, "handoff:").reason.rfind("memory of device rtx is not measured", 0) == 0, "unknown memory refused");
    inv2 = inv;
    inv2.facts["rtx"] = facts(96 * GiB, 60 * GiB);
    p = plan_route(request("ds-like", 32768, {"halo-a", "halo-b"}), inv2, model, costs);
    require(option(p, "handoff:").reason.find("device rtx: ") == 0 && chosen(p).rfind("local:", 0) == 0, "rtx without room refused");
    // Receipts contrast.
    p = plan_route(request("ds-like", 32768, {"halo-a", "halo-b"}), inv, model, costs);
    const auto x = contrast(p, {{h.label, 12500}, {h.label, 20000}, {"nope", 1}});
    require(x[0].found && x[0].within && !x[0].explain, "receipt within uncertainty: predicted " + std::to_string(x[0].predicted_ms) +
            " +- " + std::to_string(x[0].uncertainty_ms) + " found " + std::to_string(x[0].found));
    require(x[1].explain && std::abs(x[1].error - (20000 / h.predicted_ms - 1)) < 1e-9, "receipt miss flagged");
    require(!x[2].found, "unknown label");
    // Pin: an unselectable option is refused with its reason, never silently replaced.
    auto pin = request("ds-like", 32768, {"halo-a", "halo-b"});
    pin.pin = option(p, "local:rtx").label;
    p = plan_route(pin, inv, model, costs);
    require(p.chosen < 0 && p.reason.find("decode is placed on") != std::string::npos, "pinned unselectable option refused");
}

void idle_load_tests() {
    const auto model = ds_model();
    const auto costs = parse_stage_costs(kDsCosts);
    const auto cli = inventory(kLink1g);
    auto serve = cli;
    serve.load["rtx"] = {}; serve.load["halo-a"] = {2, 0, false}; // idle prefill, pair decoding two streams
    const auto r = request("ds-like", 32768, {"halo-a", "halo-b"});
    const auto a = plan_route(r, cli, model, costs), b = plan_route(r, serve, model, costs);
    require(a.hash() == b.hash() && a.to_json() == b.to_json(), "idle decode load leaves the planner deterministic");
    require(a.to_json().find("\"schema\":\"strix-route-v1\"") == 1 && a.to_json().find("\"route_id\":\"" + a.hash()) != std::string::npos,
            "route JSON schema and id");
    serve.load["rtx"].queued = 1; // the only explained difference: live load
    const auto c = plan_route(r, serve, model, costs);
    require(option(c, "handoff:").reason.rfind("live load: device rtx has 1 queued requests", 0) == 0 && chosen(c).rfind("local:halo", 0) == 0,
            "server load changes the decision and names it");
}

void qwen_tests() {
    const auto costs = parse_stage_costs(qwen_costs());
    const auto measured10g = R"(attach = { kind = "remote", link = "10gbe", gbps_measured = 8.8, rtt_us_measured = 100 })";
    const auto measured1g = R"(attach = { kind = "remote", link = "1gbe", gbps_measured = 0.944, rtt_us_measured = 200 })";
    // Layer split not qualified (no CUDA stage adapter yet): predicted, reported, never selected.
    auto p = plan_route(request("qwen-like", 32768, {"halo-a"}), inventory(measured10g), qwen_model(""), costs);
    const auto split = option(p, "layers:rtx[0,");
    require(split.predicted && !split.selectable && split.reason.find("not qualified") == 0, "ungated split reported only");
    require(chosen(p).rfind("handoff:rtx->halo-a", 0) == 0, "qwen 10 GbE without split chooses hand-off, got " + chosen(p));
    require(std::abs(p.selected()->predicted_ms - (7650 + 1097.9e6 / 1.1e6 + 0.105 + 200)) < 5, "qwen hand-off ~8.9-9.0 s at 10 GbE");
    // With a gate the split wins on 10 GbE and loses on 1 GbE: one declaration line decides.
    p = plan_route(request("qwen-like", 32768, {"halo-a"}), inventory(measured10g), qwen_model("synthetic-gate"), costs);
    require(chosen(p).rfind("layers:rtx[0,", 0) == 0, "qwen 10 GbE gated split chosen, got " + chosen(p));
    const auto& s = *p.selected();
    require(s.stages.size() == 3 && s.stages[0].end > 20 && s.stages[0].end < 30, "balanced first cut near 57 % of work");
    require(s.predicted_ms < 7650, "split beats RTX alone on 10 GbE");
    const auto cut10 = s.stages[0].end;
    p = plan_route(request("qwen-like", 32768, {"halo-a"}), inventory(measured1g), qwen_model("synthetic-gate"), costs);
    // At 1 GbE the RTX stage becomes link-bound: the balance moves almost all work to the nodes, and the 9.3 s raw
    // state transfer keeps the full RTX hand-off out.
    const auto s1 = option(p, "layers:rtx[0,");
    require(s1.stages[0].end < cut10 / 4 && chosen(p).rfind("handoff:", 0) != 0, "1 GbE moves the cut, got " + chosen(p));
    require(option(p, "handoff:rtx->halo-a").predicted_ms > 17000, "1 GbE hand-off pays ~9.3 s of raw state");
    require(s1.reason.find("taken from rtx") != std::string::npos, "borrowed layer work is stated");
    // Class-only link: the remote prefill role needs a measured rate (start-gate rule), so the split is refused.
    p = plan_route(request("qwen-like", 32768, {"halo-a"}), inventory(R"(attach = { kind = "remote", link = "10gbe" })"),
                   qwen_model("synthetic-gate"), costs);
    require(option(p, "layers:rtx[0,").reason.find("needs a measured rate") != std::string::npos, "class-only remote prefill refused");
    require(chosen(p).rfind("handoff:", 0) == 0, "class-only link still admits the hand-off to a decode unit");
    // No per-layer work for a stage and none to borrow: the split cannot be predicted.
    auto bare = costs;
    for (auto& e : bare.entries) e.layers.clear();
    p = plan_route(request("qwen-like", 32768, {"halo-a"}), inventory(measured10g), qwen_model("synthetic-gate"), bare);
    require(option(p, "layers:").reason.find("no per-layer work") == 0, "split without layer work refused");
}
// Small independent timing oracle: two one-ms/row stages, with no wire or import costs.
struct Tiny {
    Inventory inv = inventory(kLink1g);
    Model model{"tiny", 2, GiB, 0, 0, 0,
                {{"layers", "pipe", "split", {"a", "b"}, {10}, "synthetic", 0}}};
    StageCostTable costs{"tiny", {}};
    RouteRequest req = request("tiny", 11, {});
    Tiny() {
        for (auto& d : inv.declaration.devices) {
            d.attach = {};
            d.attach.kind = "integrated";
            d.roles = {"prefill", "decode"};
        }
        for (const auto& [unit, identity] : std::vector<std::pair<std::string, std::string>>{{"rtx", "a"}, {"halo-a", "b"}}) {
            StageCost e;
            e.name = identity; e.unit = {unit}; e.identity = identity; e.context = 11; e.chunk = 10;
            e.total_ms = 22; e.layers = {1, 1}; e.spread = 0;
            costs.entries.push_back(e);
        }
    }
    RoutePlan plan() const { return plan_route(req, inv, model, costs); }
};

void request_validation_tests() {
    Tiny t;
    t.req.spec.context = 10;
    require(!t.plan().selected(), "prompt must fit the admitted context");
    t.req.spec.context = 0; t.req.prompt = 0;
    require(!t.plan().selected(), "zero context refused before division");
    t.req.spec.context = 32;
    for (const double margin : {-1.0, std::numeric_limits<double>::quiet_NaN()}) {
        t.req.spec.memory.margin_gib = margin;
        require(!t.plan().selected(), "invalid margin refused before conversion");
    }
    t.req.spec.memory.margin_gib = 0; t.req.spec.slots = 2;
    require(!t.plan().selected(), "provider slot limit enforced");
    t.req.spec.slots = 1; t.model.protocols_[0].chunks = {0};
    require(!t.plan().selected(), "zero protocol chunk refused");
}

void ownership_tests() {
    rejects([] { return parse_stage_costs("schema = \"strix-stage-costs-v1\"\nmodel = \"m\"\n[costs.x]\n"
        "unit = [\"g\", \"g\"]\ntopology = \"tp2\"\nidentity = \"i\"\ncontext = 1\ntotal_ms = 1.0\n"); }, "duplicate devices");
    Tiny t;
    t.model.protocols_[0].kind = "handoff";
    t.costs.entries[1].unit = {"rtx", "halo-a"}; t.costs.entries[1].topology = "tp2";
    require(!t.plan().selected(), "overlapping handoff cannot double spend device memory");
    t.costs.entries[1].unit = {"halo-a", "halo-a"};
    require(!t.plan().selected(), "programmatic cost units also reject duplicates");
}

void dense_path_tests() {
    LinkGraph g;
    for (int i = 0; i < 16; ++i) g.nodes.push_back(std::to_string(i));
    for (std::size_t i = 0; i < g.nodes.size(); ++i)
        for (std::size_t j = i + 1; j < g.nodes.size(); ++j)
            g.edges.push_back({"e", g.nodes[i], g.nodes[j], "", "r", 1, 1, 0, RateSource::measured});
    const auto p = best_path(g, "0", "15");
    require(p.path && p.path->edges.size() == 1 && p.path->rtt_us == 1, "dense graph chooses direct equal-width path");
    // A faster prefix is not necessarily the shortest path after a later bottleneck.
    g = {{"a", "b", "c", "d"}, {
        {"ab", "a", "b", "", "ab", 10, 100, 0, RateSource::measured},
        {"ac", "a", "c", "", "ac", 5, 1, 0, RateSource::measured},
        {"cb", "c", "b", "", "cb", 5, 1, 0, RateSource::measured},
        {"bd", "b", "d", "", "bd", 5, 1, 0, RateSource::measured}}};
    const auto q = best_path(g, "a", "d");
    require(q.path && q.path->rtt_us == 3 && q.path->edges.size() == 3, "RTT optimized after bottleneck is fixed");
}

void planning_limit_tests() {
    Tiny t;
    t.model = Model("tiny", 500, GiB, 0, 0, 0,
                    {{"layers", "pipe", "split", {"a", "b", "c"}, {10}, "synthetic", 0}});
    auto third = t.costs.entries.back(); third.unit = {"halo-b"}; third.identity = "c";
    t.costs.entries.push_back(third);
    for (auto& e : t.costs.entries) e.layers.assign(500, 1);
    const auto p = t.plan();
    require(!p.selected() && p.options.empty() && p.reason.find("work limit") != std::string::npos,
            "cut search fails closed at its work bound");
    // Assignment product with no legal complete placement must also terminate.
    t.model.protocols_[0].stages.assign(12, "a");
    t.costs.entries.clear();
    for (int i = 0; i < 12; ++i) {
        auto e = third; e.unit = {"d" + std::to_string(i)}; e.identity = "a";
        t.costs.entries.push_back(e);
    }
    require(t.plan().reason.find("work limit") != std::string::npos, "assignment search is bounded");
}

void stage_role_tests() {
    Tiny t;
    for (auto& d : t.inv.declaration.devices) d.roles = {"prefill"};
    require(!t.plan().selected(), "last split stage must be decode eligible");
    for (auto& d : t.inv.declaration.devices) {
        d.roles = {"prefill", "decode"};
        if (d.id == "halo-a") { d.attach.kind = "remote"; d.attach.link = "1gbe"; }
    }
    const auto p = t.plan();
    require(!p.selected() && option(p, "layers:").reason.find("needs a measured rate") != std::string::npos,
            "decode permission cannot bypass measured prefill permission");
}

void uneven_tail_tests() {
    Tiny t;
    const auto p = t.plan();
    require(p.selected() && std::abs(p.selected()->predicted_ms - 21) < 1e-9,
            "11 rows in blocks of 10 finish at 21 ms, including preceding block drain");
    t.req.prompt = 20;
    const auto full = t.plan();
    require(full.selected() && std::abs(full.selected()->predicted_ms - 30) < 1e-9, "two full blocks finish at 30 ms");
    t.req.prompt = 1;
    require(std::abs(t.plan().selected()->predicted_ms - 2) < 1e-9, "one short block finishes at 2 ms");
}

void acknowledgement_tests() {
    Tiny t;
    for (auto& d : t.inv.declaration.devices) if (d.id == "rtx") {
        d.attach.kind = "pcie"; d.attach.h2d_gbytes_ps = 1; d.attach.rtt_us_measured = 10000;
    }
    t.model = Model("tiny", 2, GiB, 100, 0, 2000,
                    {{"layers", "pipe", "split", {"a", "b"}, {10}, "synthetic", 0, 0, 10}});
    const auto p = t.plan();
    const auto o = option(p, "layers:");
    require(o.selectable && o.transfers.size() == 2, "acknowledgement fixture admitted");
    require(std::abs(o.transfers[0].ms - 1000.001) < 1e-9, "seam full block pays 100 acknowledgements");
    require(std::abs(o.transfers[1].ms - 1000.001) < 1e-9, "state pays 100 acknowledgements");
    // With negligible wire time: seam [10,1010],[1010,1110]; stage1 [1010,1020],[1110,1111].
    require(std::abs(o.predicted_ms - 2111.0021) < 1e-7, "total includes seam dependency and every state acknowledgement");
}

void shared_port_tests() {
    Tiny t;
    t.inv.declaration.links.clear();
    for (auto& d : t.inv.declaration.devices) {
        if (d.id == "halo-b") continue;
        d.attach.kind = "pcie"; d.attach.port = "shared"; d.attach.h2d_gbytes_ps = 1;
        d.attach.rtt_us_measured = 1000;
    }
    t.model = Model("tiny", 2, GiB, 0, 0, 1e9,
                    {{"handoff", "pipe", "handoff", {"a", "b"}, {}, "synthetic", 0, 0, 10000000}});
    const auto p = t.plan();
    const auto o = option(p, "handoff:");
    require(o.selectable && std::abs(o.transfers[0].ms - 2200) < 1e-9,
            "one GB uses shared port twice plus 100 two-ms acknowledgements");
    require(std::abs(o.predicted_ms - 2222) < 1e-9 && std::abs(o.rows_per_s - 5) < 1e-9,
            "handoff total and throughput include repeated shared-port demand");
}

void chunk_calibration_tests() {
    Tiny t;
    t.req.prompt = 20;
    t.model.protocols_[0].chunks = {10, 20};
    for (auto& e : t.costs.entries) { e.context = 20; e.total_ms = 400; }
    const auto first = t.costs.entries;
    for (auto e : first) { e.name += "-20"; e.chunk = 20; e.total_ms = 40; t.costs.entries.push_back(e); }
    const auto p = t.plan();
    require(p.selected() && p.selected()->chunk == 20 && std::abs(p.selected()->predicted_ms - 40) < 1e-9,
            "matching block measurements change the winning chunk");
    require(p.selected()->stages[0].cost == "a-20", "receipt identifies matching chunk calibration");
    t.model.protocols_[0].chunks = {5};
    require(option(t.plan(), "layers:").reason.find("no matching chunk calibration") != std::string::npos,
            "missing chunk calibration is explicitly labelled");
}

void decoder_admission_tests() {
    Tiny t;
    t.req.decode = {"halo-b"};
    const auto a = t.plan();
    require(a.selected() && a.selected()->stages.size() == 3, "off-chain decoder retained");
    const auto& d = a.selected()->stages.back();
    require(d.unit == t.req.decode && d.identity == "b" && d.topology == "single" && d.ranks.size() == 1 && d.ranks[0].fits,
            "decoder has identity, topology and full-model rank admission");
    t.inv.facts["halo-b"].mem_free_bytes -= GiB;
    const auto b = t.plan();
    require(b.selected() && a.hash() != b.hash() && a.to_json() != b.to_json(), "decoder budget participates in canonical receipt");
}

void pinned_cut_tests() {
    Tiny t;
    t.model = Model("tiny", 4, GiB, 0, 0, 0,
                    {{"layers", "pipe", "split", {"a", "b"}, {10}, "synthetic", 0}});
    t.req.prompt = 20;
    for (auto& e : t.costs.entries) { e.layers.assign(4, 1); e.context = 20; e.total_ms = 80; }
    const auto original = t.plan();
    require(original.selected() && original.selected()->stages[0].end == 2, "original balanced cut is two");
    t.req.pin = original.selected()->label;
    t.costs.entries[0].total_ms = 800;
    const auto pinned = t.plan();
    require(pinned.selected() && pinned.selected()->label == t.req.pin, "legal pinned cut survives changed calibration");
    t.req.pin.clear();
    require(t.plan().selected()->stages[0].end == 1, "new optimum differs from pinned cut");
}
} // namespace

int main(int argc, char** argv) try {
#ifdef _WIN32
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    const std::string test = argc > 1 ? argv[1] : "all";
    const auto run = [&](const char* name, auto fn) { if (test == "all" || test == name) fn(); };
    run("existing", [] { declaration_tests(); ds_tests(); idle_load_tests(); qwen_tests(); });
    run("request_validation", request_validation_tests);
    run("ownership", ownership_tests);
    run("dense_path", dense_path_tests);
    run("planning_limit", planning_limit_tests);
    run("stage_role", stage_role_tests);
    run("uneven_tail", uneven_tail_tests);
    run("acknowledgement", acknowledgement_tests);
    run("shared_port", shared_port_tests);
    run("chunk_calibration", chunk_calibration_tests);
    run("decoder_admission", decoder_admission_tests);
    run("pinned_cut", pinned_cut_tests);
    std::cout << "route planner tests passed\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << "route planner test failed: " << e.what() << '\n';
    return 1;
}
