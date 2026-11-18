#include "strix/ds41_host_pin_plan.hpp"
#include <cstdio>
#include <string>

using namespace strix::ds41;
enum class ExpertTier { device, host, file };
struct CudaPrefillOptions {
    HostPinOrder host_pin_order = HostPinOrder::automatic;
    double device_expert_gib = 1.0, host_expert_gib = 1.0;
    std::uint32_t first_layer = 0, end_layer = 5, slots = 2, file_ring_mib = 0;
    bool expert_residency = false;
    std::vector<ExpertTier> tiers;
};
struct Model {
    std::vector<ExpertTier> tiers;
    std::string host_pin_reason;
};
void check(bool ok) { if (!ok) throw std::runtime_error("whole-layer integration check failed"); }
#include "default_tiers.inc"

void whole_layer_streamer(std::uint32_t requested_ring) {
    CudaPrefillOptions o;
    o.file_ring_mib = requested_ring;
    Model m;
    struct { std::uint32_t num_hidden_layers = 5; } c;
    struct { std::uint32_t staging_pages = 4; } sp;
    const std::uint64_t layer_bytes = 1ULL << 30;
#include "host_pin_fallback.inc"
    (void)host_pin_fallback;
    // Actual driver branch used by --no-expert-residency --xfer-group 8.
#include "whole_layer_route.inc"
    check(o.host_pin_order == HostPinOrder::layer);
    check(m.tiers == std::vector<ExpertTier>{ExpertTier::host, ExpertTier::file, ExpertTier::file,
                                           ExpertTier::file, ExpertTier::device});
    check(o.file_ring_mib == (requested_ring ? requested_ring : 256U));
    check(sp.staging_pages == (requested_ring ? requested_ring / 64U : 4U));
}

int main() try {
    whole_layer_streamer(0);
    whole_layer_streamer(1024);
    CudaPrefillOptions o;
    o.file_ring_mib = 1024;
    Model m;
#include "host_pin_fallback.inc"
    host_pin_fallback("provider fallback");
    check(o.file_ring_mib == 1024 && o.host_pin_order == HostPinOrder::layer);
    o.host_expert_gib = 1.0 - 5e-10;
    check(default_tiers(o, 5, 1ULL << 30)[0] == ExpertTier::host);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 1;
}
