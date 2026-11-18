// Device catalogue (strix/device/profiles.hpp), declared budgets (budget.hpp, gate.hpp) and the shared memory plan
// (strix/deploy/memory_plan.hpp). Host only.
#include "strix/deploy/device.hpp"
#include "strix/deploy/memory_plan.hpp"
#include "strix/device/process_budget.hpp"
#include "strix/device/budget.hpp"
#include "strix/device/declaration.hpp"
#include "strix/device/gate.hpp"
#include "strix/device/profiles.hpp"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <utility>

namespace {

int failures = 0;
void require(bool ok, const char* message) {
    if (!ok) {
        ++failures;
        std::fprintf(stderr, "FAIL: %s\n", message);
    }
}
template <class F> bool throws(F&& f, const char* needle = nullptr) {
    try {
        f();
    } catch (const std::exception& e) {
        if (needle && std::string(e.what()).find(needle) == std::string::npos) {
            std::fprintf(stderr, "  (message was: %s)\n", e.what());
            return false;
        }
        return true;
    }
    return false;
}

using namespace strix;
constexpr std::uint64_t GiB = 1073741824ULL;
const char* kPro = "GPU-11111111-2222-3333-4444-555555555555";
constexpr std::uint64_t kProBytes = 102641958912ULL;

device::Visible pro_card() {
    device::Visible v;
    v.ordinal = 0;
    v.vendor = device::Vendor::nvidia;
    v.name = "NVIDIA RTX PRO 6000 Blackwell Workstation Edition";
    v.uuid = device::normalise_uuid(kPro);
    v.arch = "sm_120";
    v.memory_bytes = kProBytes;
    return v;
}

std::string decl(const std::string& body) {
    return "schema = \"strix-devices-v1\"\n" + body;
}

void catalogue() {
    std::set<std::string_view> names;
    for (const auto& p : device::catalogue()) require(names.insert(p.name).second, "catalogue names are unique");
    for (const char* n : {"rtx-pro-6000-blackwell-96", "rtx-5090-32", "rtx-5080-16", "rtx-4090-24", "strix-halo-128",
                          "dgx-spark-gb10-128", "rx-9070-xt-16", "custom"})
        require(device::find_profile(n) != nullptr, n);
    const auto* p5090 = device::find_profile("rtx-5090-32");
    require(p5090->units == 170 && p5090->arch == "sm_120" && p5090->memory_bytes == 32607ULL * 1048576ULL, "5090 entry");
    require(device::find_profile("rtx-pro-6000-blackwell-96")->memory_bytes == kProBytes, "PRO memory is the measured total");
    for (const auto& p : device::catalogue())
        if (p.name != "custom") require(p.memory_bytes > 0 && p.memory_gbytes_ps > 0 && p.units > 0, "every known profile is sized");
    require(device::find_interconnect("pcie5x16") && !device::find_interconnect("nvlink"), "interconnect classes");
    // The data file is the printed table.
    std::ifstream in(std::string(STRIX_SOURCE_ROOT) + "/data/devices/profiles.toml", std::ios::binary);
    std::ostringstream file;
    file << in.rdbuf();
    require(in.good() || in.eof(), "data/devices/profiles.toml is readable");
    require(file.str() == device::catalogue_toml(), "data/devices/profiles.toml equals catalogue_toml() (regenerate: strix-device-plan --toml)");
}

void declaration() {
    const auto d = device::parse_declaration(decl("[devices.main]\nuuid = \"" + std::string(kPro) +
                                                  "\"\nprofile = \"rtx-5090-32\"\nroles = [\"replica\"]\n"));
    require(d.devices.size() == 1 && d.devices[0].vendor == device::Vendor::nvidia, "vendor comes from the profile");
    require(d.devices[0].profile == "rtx-5090-32" && !d.devices[0].planned, "a profile with an identity binds");
    require(throws([] { (void)device::parse_declaration(decl("[devices.a]\nuuid = \"GPU-11111111-2222-3333-4444-555555555555\"\nprofile = \"rtx-9999\"\n")); }, "unknown profile"),
            "unknown profile refused with the known list");
    require(throws([] { (void)device::parse_declaration(decl("[devices.a]\nvendor = \"amd\"\nuuid = \"GPU-11111111-2222-3333-4444-555555555555\"\nprofile = \"rtx-5090-32\"\n")); }, "contradicts"),
            "vendor contradicting the profile refused");
    require(throws([] { (void)device::parse_declaration(decl("[devices.a]\nvendor = \"nvidia\"\nprofile = \"custom\"\n")); }, "memory_gib"),
            "custom needs memory_gib");
    const auto shape = device::parse_declaration(decl("[devices.pair]\nprofile = \"rtx-5090-32\"\ncount = 2\ninterconnect = \"pcie5x16\"\n"));
    require(shape.devices[0].planned && shape.devices[0].count == 2 && shape.devices[0].interconnect == "pcie5x16", "2x 5090 shape");
    require(throws([] { (void)device::parse_declaration(decl("[devices.p]\nprofile = \"rtx-5090-32\"\ncount = 2\n")); }, "interconnect"),
            "count > 1 needs an interconnect");
    require(throws([] { (void)device::parse_declaration(decl("[devices.p]\nprofile = \"rtx-5090-32\"\ncount = 2\ninterconnect = \"nvlink\"\n")); }, "NVLink"),
            "nvlink is not a catalogue interconnect");
    require(throws([] { (void)device::parse_declaration(decl("[devices.p]\nuuid = \"GPU-11111111-2222-3333-4444-555555555555\"\nprofile = \"rtx-5090-32\"\ncount = 2\ninterconnect = \"pcie5x16\"\n")); }, "no identity"),
            "a shape takes no identity");
    require(throws([] { (void)device::parse_declaration(decl("[devices.p]\nvendor = \"nvidia\"\n")); }, "needs an identity"),
            "no identity and no profile is still refused");
    require(throws([] { (void)device::parse_declaration(decl("[devices.host]\nkind = \"cpu\"\nprofile = \"rtx-5090-32\"\n")); }, "only a GPU"),
            "a cpu unit takes no profile");
    const auto host = device::parse_declaration(decl("[devices.host]\nkind = \"cpu\"\nmemory_gib = 64\n"));
    require(host.devices[0].budget_bytes == 64 * GiB, "cpu unit memory_gib = host budget");
}

void budget() {
    device::Descriptor g;
    g.id = "main";
    g.profile = "rtx-5090-32";
    auto b = device::resolve_budget(g, nullptr, kProBytes);
    require(b.device_bytes == 32607ULL * 1048576ULL && b.device_source == "profile rtx-5090-32", "profile gives the budget");
    require(b.emulated() && b.units == 170 && b.bandwidth_gbytes_ps == 1792, "below the card = emulation");
    g.budget_bytes = 30 * GiB;
    require(device::resolve_budget(g, nullptr, kProBytes).device_bytes == 30 * GiB, "memory_gib wins over the profile");
    device::BudgetOverrides o;
    o.vram_total_gib = 24.0;
    require(device::resolve_budget(g, nullptr, kProBytes, o).device_bytes == 24 * GiB, "the route's flag wins");
    o.vram_total_gib = -1.0;
    require(!device::resolve_budget(g, nullptr, kProBytes, o).device_capped(), "--vram-total-gib none removes the cap");
    device::Descriptor big;
    big.id = "x";
    big.profile = "strix-halo-128";
    require(throws([&] { (void)device::resolve_budget(big, nullptr, 32 * GiB); }, "cannot exceed"), "a profile cannot add memory");
    device::Descriptor none;
    none.id = "plain";
    require(!device::resolve_budget(none, nullptr, kProBytes).device_capped(), "no profile = no cap (today's path)");
    device::Descriptor cpu;
    cpu.id = "host";
    cpu.kind = device::Kind::cpu;
    cpu.budget_bytes = 64 * GiB;
    require(device::resolve_budget(g, &cpu, kProBytes).host_bytes == 64 * GiB, "host budget from the cpu unit");
    require(device::parse_total_gib("--x", "none") < 0 && device::parse_total_gib("--x", "32") == 32.0, "flag form");
    require(throws([] { (void)device::parse_total_gib("--x", "0"); }), "zero refused");
}

void gate() {
    const std::string path_free = decl("[devices.main]\nvendor = \"nvidia\"\nuuid = \"" + std::string(kPro) + "\"\nroles = [\"replica\"]\n");
    const auto plain = device::parse_declaration(path_free);
    device::Request r;
    r.vendor = device::Vendor::nvidia;
    const auto b0 = device::resolve(&plain, {pro_card()}, r);
    require(!b0.budget.device_capped() && device::receipt_json(b0).find("memory_budget") == std::string::npos,
            "no profile: receipt bytes unchanged");
    r.profile = "rtx-5090-32";
    const auto b1 = device::resolve(&plain, {pro_card()}, r);
    require(b1.budget.emulated() && b1.device.profile == "rtx-5090-32", "--device-profile emulates on the bound card");
    require(device::receipt_json(b1).find("strix-memory-budget-v1") != std::string::npos, "receipt carries the budget");
    require(device::receipt_line(b1).find("profile=rtx-5090-32") != std::string::npos, "receipt line names the profile");
    r.profile = "rtx-9999";
    require(throws([&] { (void)device::resolve(&plain, {pro_card()}, r); }, "unknown device profile"), "unknown --device-profile");
    // A declared profile above the card refuses at the gate.
    const auto too_big = device::parse_declaration(decl("[devices.main]\nuuid = \"" + std::string(kPro) +
                                                        "\"\nprofile = \"dgx-spark-gb10-128\"\nvendor = \"nvidia\"\nroles = [\"replica\"]\n"));
    r.profile.clear();
    require(throws([&] { (void)device::resolve(&too_big, {pro_card()}, r); }, "cannot exceed"), "profile above the card refused");
    // A planned shape is never bound.
    const auto mixed = device::parse_declaration(decl("[devices.main]\nvendor = \"nvidia\"\nuuid = \"" + std::string(kPro) +
                                                      "\"\nroles = [\"replica\"]\n[devices.pair]\nprofile = \"rtx-5090-32\"\ncount = 2\n"
                                                      "interconnect = \"pcie5x16\"\nroles = [\"replica\"]\n"));
    r.select = "pair";
    require(throws([&] { (void)device::resolve(&mixed, {pro_card()}, r); }, "planned shape"), "planned entry refused at bind");
    r.select.clear();
    require(device::resolve(&mixed, {pro_card()}, r).device.id == "main", "the bindable card still binds beside a shape");
    // ONE RESOLVER: the server / CLI bound path and the planning tool resolve the same request to the same budget.
    device::BudgetOverrides o;
    o.ram_total_gib = 48.0;
    const auto via_gate = device::route_budget(b1, o);
    const auto via_tool = device::resolve_budget(b1.device, nullptr, kProBytes, o);
    require(via_gate.json() == via_tool.json(), "gate path and planning path give the same budget");
    // start_gate records the budget for the planners that never see the binding (deploy::probe_device).
    const std::string file = "device_profiles_tests_devices.toml";
    {
        std::ofstream out(file, std::ios::binary);
        out << decl("[devices.main]\nuuid = \"" + std::string(kPro) + "\"\nprofile = \"rtx-5080-16\"\nroles = [\"replica\"]\n");
    }
    device::FakeEnumerator cards(device::Vendor::nvidia, {pro_card()});
    const auto via_start = device::start_gate(cards, {}, file);
    std::remove(file.c_str());
    require(via_start.budget.device_bytes == 16303ULL * 1048576ULL &&
                device::process_device_budget_bytes().load() == via_start.budget.device_bytes,
            "start_gate records the process budget");
    deploy::DeviceFacts f;
    f.mem_total_bytes = 96 * GiB;
    f.mem_free_bytes = 90 * GiB;
    deploy::apply_budget(f, 32 * GiB);
    require(f.mem_total_bytes == 32 * GiB && f.mem_free_bytes == 26 * GiB, "apply_budget: free = budget - in use");
    deploy::DeviceFacts g = {};
    g.mem_total_bytes = 16 * GiB;
    g.mem_free_bytes = 15 * GiB;
    deploy::apply_budget(g, 32 * GiB);
    require(g.mem_total_bytes == 16 * GiB && g.mem_free_bytes == 15 * GiB, "apply_budget: a budget above the device changes nothing");
}

deploy::RouteNeeds moe_route() {
    deploy::RouteNeeds n;
    n.route = "moe";
    n.device_fixed = 4 * GiB;
    n.weights_resident = 10 * GiB;
    n.tier_units = 1000;
    n.tier_unit_bytes = 64ULL << 20; // 62.5 GiB of experts
    n.min_device_units = 16;
    n.host_tier = n.file_tier = true;
    n.host_fixed = 8 * GiB;
    n.session_per_position = 64 << 10;
    n.positions = 32768;
    n.min_positions = 2560;
    n.slots = 4;
    n.margin_bytes = 1 * GiB;
    return n;
}

void plan() {
    device::MemoryBudget b;
    b.device_bytes = 32 * GiB;
    b.host_bytes = 40 * GiB;
    b.device_source = "test";
    auto n = moe_route();
    const auto p = deploy::plan_memory(b, n);
    // 4 + 10 + 1 + 4 x 2 GiB sessions = 23 GiB, 9 GiB -> 144 units in VRAM; host 32 GiB -> 512 units; 344 to the file.
    require(!p.refused && p.slots == 4 && p.positions == 32768, "sessions kept");
    require(p.units_device == 144 && p.units_host == 512 && p.units_file == 344, "VRAM / pinned / file split");
    b.device_bytes = 18 * GiB;
    const auto q = deploy::plan_memory(b, n);
    require(!q.refused && q.positions == 32768 && q.slots == 1, "slots drop before positions");
    b.device_bytes = 17 * GiB;
    const auto r = deploy::plan_memory(b, n);
    require(!r.refused && r.slots == 1 && r.positions < 32768 && r.positions >= 2560, "then positions halve");
    b.device_bytes = 15 * GiB;
    const auto s = deploy::plan_memory(b, n);
    require(s.refused && s.min_device_bytes > 15 * GiB && s.refusal.find("needs at least") != std::string::npos, "refusal names the minimum");
    // A route without tiers (all weights resident): refuses below the weights and says the minimum.
    deploy::RouteNeeds dense;
    dense.route = "dense";
    dense.weights_resident = 82 * GiB;
    dense.device_fixed = 2 * GiB;
    dense.session_per_position = 31616;
    dense.session_fixed = 233ULL << 20;
    dense.positions = 34816;
    dense.slots = 3;
    b.device_bytes = 32 * GiB;
    const auto d = deploy::plan_memory(b, dense);
    require(d.refused && d.min_device_bytes > 84 * GiB, "dense route refuses on 32 GiB with the minimum");
    b.device_bytes = 96 * GiB;
    require(!deploy::plan_memory(b, dense).refused, "and fits 96 GiB");
    // Needs file round trip.
    const auto parsed = deploy::parse_route_needs("schema = \"strix-route-needs-v1\"\nroute = \"x\"\nweights_resident = 1024\nhost_tier = true\ndecode_efficiency = 0.5\n");
    require(parsed.route == "x" && parsed.weights_resident == 1024 && parsed.host_tier && parsed.decode_efficiency == 0.5, "needs parse");
    require(throws([] { (void)deploy::parse_route_needs("schema = \"strix-route-needs-v1\"\nroute = \"x\"\nbogus = 1\n"); }, "unknown key"),
            "needs unknown key");
    // 2x RTX 5090 over PCIe 5: each card holds half, the prediction counts one hop.
    device::Descriptor pair;
    pair.id = "pair";
    pair.profile = "rtx-5090-32";
    pair.count = 2;
    pair.interconnect = "pcie5x16";
    auto m = moe_route();
    m.bytes_per_token = 11ULL * 1000000000ULL;
    m.tiered_bytes_per_token = 6ULL * 1000000000ULL;
    m.hop_bytes_per_token = 16384;
    m.decode_efficiency = 0.5;
    const auto two = deploy::plan_shape(pair, nullptr, m);
    require(!two.per_device.refused && two.per_device.units_device > 0, "shape fits per device");
    require(two.predicted_decode_tok_s > 0 && two.hop_ms > 0, "shape prediction");
    pair.count = 1;
    const auto one = deploy::plan_shape(pair, nullptr, m);
    require(one.per_device.units_device < two.per_device.units_device * 2, "two cards hold more experts than one");
}

} // namespace

int main() {
    for (const auto& [name, run] : {std::pair<const char*, void (*)()>{"catalogue", catalogue}, {"declaration", declaration},
                                    {"budget", budget}, {"gate", gate}, {"plan", plan}}) {
        try {
            run();
        } catch (const std::exception& e) {
            ++failures;
            std::fprintf(stderr, "FAIL: %s threw: %s\n", name, e.what());
        }
    }
    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("device profiles tests: OK\n");
    return 0;
}
