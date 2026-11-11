// Compile the actual serve binding against host sizing/transport doubles. No HIP headers or model weights.
#include "strix/deploy/binding.hpp"
#include "strix/deepseek41_prefill.hpp"
#include "strix/deepseek41_serve_sched.hpp"
#include "strix/dist/transport.hpp"
#include "strix/engine_profile.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

namespace strix::deepseek41::pair_defaults {
const profile::Resolution& profile_resolution() { static const profile::Resolution r; return r; }
}
namespace strix::ds41::expert_tp { bool enabled_from_env() { return false; } }
namespace strix::deepseek41 {
std::uint64_t DevicePrefill::owned_weight_bytes(const std::filesystem::path&, const PrefillOptions&) { return 2ULL << 30; }
}
namespace {
struct Args {
    std::string model = "fixture.gguf", model_id = "deepseek-v4.1-flash", deploy = "auto", transport = "usb4auto";
    std::string dspark, hetero_listen;
    std::uint32_t context = 4096, slots = 4, layers = 40, prefill_rows = 4096;
    std::uint64_t prefix_cache_mb = 0, session_ram_bytes = 0;
    bool engram_direct = true, device_prefill = true, prefill_rows_auto = false;
    std::shared_ptr<const strix::deploy::Spec> deploy_spec;
    std::shared_ptr<const strix::deploy::Plan> deployment_plan;
};
struct NativeOptions { std::uint32_t context, slots; };
NativeOptions forward_options(const Args& a, void*, std::uint32_t) { return {a.context, a.slots}; }
struct BatchForward {
    struct Footprint { std::uint64_t weights, state_per_slot, workspace, pool, trellis; };
    static std::uint32_t configured_row_limit() { return 8; }
    static Footprint footprint(const std::string&, const NativeOptions& o) {
        return {40ULL << 30, std::uint64_t{o.context} * 1024, 1ULL << 30, 0, 0};
    }
    static std::map<std::string, std::uint64_t> file_form_weight_sizes(const std::string&, const NativeOptions&) { return {}; }
};
#include "../apps/deepseek41_deploy.inc"

void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void fake_memory(const char* value) {
#ifdef _WIN32
    _putenv_s("STRIX_DEPLOY_FAKE_MEM_GIB", value);
#else
    if (*value) setenv("STRIX_DEPLOY_FAKE_MEM_GIB", value, 1);
    else unsetenv("STRIX_DEPLOY_FAKE_MEM_GIB");
#endif
}
class Peer final : public strix::dist::Transport {
public:
    bool mismatch = false;
    void send(std::span<const std::byte>) override { throw std::logic_error("unexpected send"); }
    void receive(std::span<std::byte>) override { throw std::logic_error("unexpected receive"); }
    std::string name() const override { return "host-fixture"; }
    void exchange(std::span<const std::byte> out, std::span<std::byte> in) override {
        require(out.size() == in.size(), "exchange size");
        std::copy(out.begin(), out.end(), in.begin());
        if (out.size() > 16) { // device-facts packet (shared binding wire)
            std::uint64_t rank = 0;
            std::memcpy(&rank, out.data() + 8, 8);
            rank = 1 - rank;
            std::memcpy(in.data() + 8, &rank, 8);
        } else if (out.size() == 16 && mismatch) in[0] ^= std::byte{1};
        else require(out.size() == 16, "unexpected wire packet");
    }
};
} // namespace
int main() try {
    Args a;
    prepare_deployment(a);
    require(a.deploy_spec->nodes == 2 && a.deploy_spec->context == a.context && a.deploy_spec->slots == a.slots, "CLI defaults preserved");
    strix::deploy::DeviceFacts d;
    d.mem_total_bytes = d.mem_free_bytes = 128ULL << 30; d.source = "fixture";
    auto plan = strix::deploy::resolve(*a.deploy_spec, {d, d}, Ds41FootprintProvider(a));
    require(!plan.refused && plan.ranks.size() == 2 && plan.chosen.topology == "tp2", "DS41 target admission");
    require(plan.ranks[0].footprint.weights == 40ULL << 30 && plan.ranks[0].kv_bytes == 16ULL << 20, "native describe accounting");
    a.prefix_cache_mb = 512;
    plan = strix::deploy::resolve(*a.deploy_spec, {d, d}, Ds41FootprintProvider(a));
    require(plan.refused && plan.reasons[0].find("prefix/session cache") != std::string::npos, "unknown peak refuses");
    a.prefix_cache_mb = 0;
    a.context = 131072;
    prepare_deployment(a);
    require(strix::deploy::resolve(*a.deploy_spec, {d, d}, Ds41FootprintProvider(a)).refused, "unmeasured prefill estimate refused");
    a.device_prefill = false;
    require(!strix::deploy::resolve(*a.deploy_spec, {d, d}, Ds41FootprintProvider(a)).refused, "rows route header-only footprint");
    fake_memory("128");
    Peer peer;
    resolve_live_deployment(a, 0, peer);
    require(a.deployment_plan && !a.deployment_plan->refused && a.deployment_plan->devices.size() == 2, "live measured agreement");
    peer.mismatch = true;
    bool refused = false;
    try { resolve_live_deployment(a, 0, peer); } catch (const std::exception&) { refused = true; }
    require(refused, "peer hash mismatch refused");
    fake_memory("");
    std::cout << "DS41 deployment host binding tests passed\n";
    return 0;
} catch (const std::exception& e) { fake_memory(""); std::cerr << e.what() << '\n'; return 1; }
