#include "strix/deploy/plan.hpp"
#include "strix/serve/json.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace strix::deploy;
constexpr std::uint64_t GiB = 1ULL << 30;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F f) {
    bool rejected = false;
    try { f(); } catch (const std::exception&) { rejected = true; }
    require(rejected, "invalid input accepted");
}
void env(const char* value) {
#ifdef _WIN32
    _putenv_s("STRIX_DEPLOY_FAKE_MEM_GIB", value);
#else
    if (*value) setenv("STRIX_DEPLOY_FAKE_MEM_GIB", value, 1);
    else unsetenv("STRIX_DEPLOY_FAKE_MEM_GIB");
#endif
}
struct Tree {
    std::filesystem::path root = std::filesystem::temp_directory_path() /
        ("strix-deploy-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ~Tree() { std::error_code ec; std::filesystem::remove_all(root, ec); }
    void put(const std::string& name, const std::string& value) {
        const auto path = root / name;
        std::filesystem::create_directories(path.parent_path());
        std::ofstream(path) << value;
    }
    void gpu(std::uint64_t total, bool unified, const std::string& node = "1", unsigned minor = 128) {
        put("sys/class/kfd/kfd/topology/nodes/" + node + "/properties",
            "drm_render_minor " + std::to_string(minor) + "\ngfx_target_version 110501\nsimd_count 80\nsimd_per_cu 2\nlocal_mem_size " +
            std::to_string(unified ? 0 : total) + "\n");
        const auto device = "sys/class/drm/renderD" + std::to_string(minor) + "/device/";
        put(device + "mem_info_gtt_total", std::to_string(unified ? total : 16 * GiB));
        put(device + "mem_info_gtt_used", std::to_string(4 * GiB));
        put(device + "mem_info_vram_total", std::to_string(unified ? GiB : total));
        put(device + "mem_info_vram_used", std::to_string(unified ? 0 : 2 * GiB));
    }
};
void probes() {
    env("");
    for (const auto gib : {64ULL, 128ULL, 192ULL}) {
        Tree t; t.gpu(gib * GiB, true);
        t.put("sys/class/kfd/kfd/topology/nodes/0/properties", "cpu_cores_count 16\n");
        const auto d = probe_device(t.root.string());
        require(d.unified && d.gfx_target == "gfx1151" && d.cu_count == 40, "APU facts");
        require(d.mem_total_bytes == gib * GiB && d.mem_free_bytes == (gib - 4) * GiB, "APU GTT budget");
        require(d.source == "kfd+drm", "probe provenance");
    }
    Tree t; t.gpu(32 * GiB, false);
    auto d = probe_device(t.root.string());
    require(!d.unified && d.mem_total_bytes == 32 * GiB && d.mem_free_bytes == 30 * GiB, "discrete VRAM budget");
    t.put("sys/class/drm/renderD128/device/mem_info_vram_used", "-1");
    require(probe_device(t.root.string()).mem_free_bytes == 0, "malformed used counter");
    t.put("sys/class/drm/renderD128/device/mem_info_vram_used", std::to_string(33 * GiB));
    require(probe_device(t.root.string()).mem_total_bytes == 0, "inconsistent counter refused");
    std::filesystem::remove(t.root / "sys/class/drm/renderD128/device/mem_info_vram_used");
    require(probe_device(t.root.string()).mem_free_bytes == 0, "missing used counter");
    Tree empty;
    require(probe_device(empty.root.string()).mem_total_bytes == 0, "missing sysfs");
    t.gpu(64 * GiB, true, "2", 129);
    require(probe_device(t.root.string()).source == "ambiguous-device-selection", "ambiguous device selection");
    require(probe_devices(t.root.string()).size() == 2, "inventory");
    d = probe_device(t.root.string(), (t.root / "sys/class/drm/renderD129/device").string());
    require(d.mem_total_bytes == 64 * GiB && d.unified, "explicit selection among devices");
    env("96.5");
    require(probe_device(empty.root.string()).mem_free_bytes == 96 * GiB + GiB / 2, "fake memory");
    env("nan"); rejects([&] { (void)probe_device(empty.root.string()); });
    env("");
    Tree card;
    card.put("sys/class/drm/card2/device/mem_info_gtt_total", "4096");
    card.put("sys/class/drm/card2/device/mem_info_gtt_used", "512");
    require(!first_drm_device(card.root.string()).empty(), "legacy card lookup");
    d = probe_device("", first_drm_device(card.root.string()));
    require(d.gtt_total_bytes == 4096 && d.gtt_used_bytes == 512, "explicit DRM telemetry");
}
void specs() {
    require(parse_spec("auto").topology == "auto", "auto spec");
    const auto s = parse_spec(R"(schema = "strix-deploy-v1"
model = "synthetic"
[hardware]
nodes = 2
link = "usb4auto"
[topology]
mode = "tp2"
[format]
decode = "int4"
prefill = "int4-stored"
[memory]
context = 65536
slots = 2
margin_gib = 2.5
[profile]
name = "fast"
)");
    require(s.nodes == 2 && s.context == 65536 && s.slots == 2 && s.memory.margin_gib == 2.5, "TOML selections");
    require(s.pins.contains("memory.context") && s.profile == "fast", "pins recorded");
    require(parse_spec("schema='strix-deploy-v1'\n[profile]\nname='default'\nallow_pending=false").profile == "default", "qualified profile selection");
    for (const std::string input : {
             "", "schema='unknown'", "schema='strix-deploy-v1'\ncontext=0", "schema='strix-deploy-v1'\nslots='two'",
             "schema='strix-deploy-v1'\ncontext=1\ncontext=2", "schema='strix-deploy-v1'\nunknown=1",
             "schema='strix-deploy-v1'\n[memory]\nmargin_gib=-1", "schema='strix-deploy-v1'\n[hardware]\nmin_gtt_gib=160",
             "schema='strix-deploy-v1'\n[format]\ndecdoe='int4'", "schema='strix-deploy-v1'\n[memory]\npolicy='formats-first'",
             "schema='strix-deploy-v1'\nnodes=2\n[topology]\nranks=1", "schema='strix-deploy-v1'\n[profile]\nallow_pending=true"})
        rejects([&] { (void)parse_spec(input); });
    const auto inventory = parse_spec(R"(schema='strix-deploy-v1'
devices=[{gfx_target='gfx1151',cu_count=40,mem_total_bytes=65536,mem_free_bytes=32768,unified=true,source='measured'}]
)");
    require(inventory.devices.size() == 1 && inventory.devices[0].mem_free_bytes == 32768, "orchestrator inventory");
    require(parse_spec("schema='strix-deploy-v1'\ncontext=4096\n[pins]\ncontext='4096'").context == 4096, "explicit matching pins");
    rejects([] { (void)parse_spec("schema='strix-deploy-v1'\ncontext=4096\n[pins]\ncontext='8192'"); });
}
struct Synthetic final : FootprintProvider {
    bool partial = false, overflow = false;
    std::string model() const override { return "synthetic"; }
    std::vector<std::string> topologies() const override { return {"single", "tp2", "replica", "pipeline", "disaggregated"}; }
    std::vector<Format> formats(const std::string&) const override {
        std::vector<Format> result{{"int4", "int4-stored", 8, 8, 0}};
        if (partial) result.push_back({"exl3", "int4-partial", 4, 8, 2});
        return result;
    }
    std::uint32_t max_slots() const override { return 4; }
    Footprint footprint(const Spec&, const Candidate& c, std::uint32_t) const override {
        Footprint f;
        f.weights = (c.format.resident_layers == 8 ? 80 : 48) * GiB;
        if (c.topology == "tp2") f.weights /= 2;
        f.kv_per_token = overflow ? std::numeric_limits<std::uint64_t>::max() : 128 * 1024;
        f.workspace = 4 * GiB;
        f.staging = GiB;
        f.extra = {{"metadata", GiB}};
        return f;
    }
};
DeviceFacts device(std::uint64_t gib) {
    DeviceFacts d; d.gfx_target = "gfx1151"; d.cu_count = 40;
    d.mem_total_bytes = d.mem_free_bytes = gib * GiB; d.unified = true; d.source = "fixture";
    return d;
}
void decisions() {
    Synthetic provider;
    Spec s; s.context = 65536; s.slots = 2; s.link = "usb4auto";
    auto p = resolve(s, {device(128)}, provider);
    require(!p.refused && p.chosen.topology == "single" && p.chosen.context == 65536 && p.chosen.slots == 2, "single admission");
    require(p.ranks[0].kv_bytes == 16 * GiB && p.ranks[0].required_bytes == 102 * GiB && p.ranks[0].budget_bytes == 120 * GiB, "budget accounting");
    p = resolve(s, {device(80), device(80)}, provider);
    require(!p.refused && p.chosen.topology == "tp2" && p.ranks.size() == 2, "pair needed");
    require(p.ranks[1].node == 1 && p.ranks[1].required_bytes == 62 * GiB, "rank budgets");
    provider.partial = true;
    p = resolve(s, {device(80)}, provider);
    require(!p.refused && p.chosen.format.resident_layers == 4 && p.chosen.format.jit_slots == 2, "partial INT4");
    s.format = {"int4", "int4-stored"};
    require(resolve(s, {device(80)}, provider).refused, "pinned format honored");
    s.format = {};
    s.topology = "tp2";
    require(resolve(s, {device(80), device(32)}, provider).refused, "weakest rank refuses");
    s.link = "none";
    require(resolve(s, {device(128), device(128)}, provider).refused, "missing link refuses");
    s.topology = "single";
    p = resolve(s, {device(32), device(128)}, provider);
    require(!p.refused && p.ranks[0].node == 1, "single placement searches devices");
    s.nodes = 3;
    require(resolve(s, {device(128)}, provider).refused, "pinned node count");
    s.nodes = 0; s.slots = 0;
    p = resolve(s, {device(112)}, provider);
    require(!p.refused && p.chosen.slots == 4 && p.chosen.format.resident_layers == 4, "KV slots before full residency");
    provider.partial = false;
    p = resolve(s, {device(112)}, provider);
    require(!p.refused && p.chosen.slots == 2, "automatic slots search");
    s.slots = 4;
    require(resolve(s, {device(112)}, provider).refused, "pinned slots never shrink");
    s.slots = 2; provider.overflow = true;
    p = resolve(s, {device(192)}, provider);
    require(p.refused && p.reasons[0].find("overflow") != std::string::npos, "overflow refusal");
    provider.overflow = false;
    s.topology = "replica"; s.link = "none";
    require(!resolve(s, {device(128), device(128)}, provider).refused, "independent replicas do not require a link");
    s.topology = "pipeline";
    require(resolve(s, {device(128), device(128)}, provider).refused, "pipeline requires a link");
    s.link = "tcp";
    require(!resolve(s, {device(128), device(128)}, provider).refused, "provider-supported pipeline");
    s.topology = "disaggregated";
    require(!resolve(s, {device(128), device(128)}, provider).refused, "provider-supported disaggregated topology");
    const auto a = parse_spec("schema='strix-deploy-v1'\ncontext=65536\nslots=2\nmodel='synthetic'");
    const auto b = parse_spec("model='synthetic'\nslots=2\ncontext=65536\nschema='strix-deploy-v1'");
    const auto one = resolve(a, {device(128)}, provider), two = resolve(b, {device(128)}, provider);
    require(one.hash().size() == 16 && one.hash() == two.hash() && one.to_json() == two.to_json(), "canonical JSON/hash stability");
    require(one.hash() == "3b6bb5984acc1ac5", "deployment-v1 golden hash changed");
    require(one.to_json().find("\"schema\":\"strix-deployment-v1\"") != std::string::npos &&
            one.to_json().find("\"deployment_id\":\"" + one.hash() + "\"") != std::string::npos, "JSON identity");
    auto changed = device(127);
    require(one.hash() != resolve(a, {changed}, provider).hash(), "measured facts affect hash");
    require(strix::serve::parse_json(one.to_json()).is_object(), "deployment JSON parses");
    auto escaped = one;
    escaped.devices[0].drm_device = "ignored-observation-path";
    require(escaped.hash() == one.hash() && escaped.to_json().find("ignored-observation-path") == std::string::npos, "local probe paths omitted");
    escaped.devices[0].source = std::string("a\0\n\"\\", 5);
    require(strix::serve::parse_json(escaped.to_json()).is_object(), "JSON control escaping");
    auto invalid = a; invalid.memory.margin_gib = std::numeric_limits<double>::quiet_NaN();
    const auto refusal = resolve(invalid, {device(128)}, provider);
    require(refusal.refused && strix::serve::parse_json(refusal.to_json()).is_object(), "invalid spec refusal remains valid JSON");
    std::cout << "deployment fixture: " << one.hash() << '\n';
}
} // namespace
int main() try {
    probes(); specs(); decisions();
    std::cout << "deploy host tests passed\n";
    return 0;
} catch (const std::exception& e) { env(""); std::cerr << e.what() << '\n'; return 1; }
