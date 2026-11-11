#pragma once

#include "strix/deploy/spec.hpp"
#include <algorithm>
#include <utility>
#include <vector>

namespace strix::deploy {

// Mandatory owners and the caller's margin always win over an optional cap.
[[nodiscard]] inline std::uint64_t fit_elastic_extra(std::uint64_t cap,
    std::uint64_t mandatory, std::uint64_t budget) {
    return mandatory < budget ? std::min(cap, budget - mandatory) : 0;
}

struct Footprint {
    std::uint64_t weights = 0, kv_per_token = 0, kv_total = 0;
    std::uint64_t workspace = 0, staging = 0;
    std::vector<std::pair<std::string, std::uint64_t>> extra;
    // Optional caches are fitted only after the complete fixed candidate fits.
    // Each pair is a name and requested cap; the resolved bytes go into extra.
    std::vector<std::pair<std::string, std::uint64_t>> elastic_extra;
    // kv_total is the complete reservation when nonzero; otherwise context * slots * kv_per_token.
};
struct Candidate {
    std::string topology;
    Format format;
    std::uint32_t context = 0, slots = 0, ranks = 0;
};

// A provider advertises only implemented choices, ordered by preference. It must
// account for every live and peak allocation without loading model weights or devices.
// INT4 hook. Stored ("int4-stored": every MoE layer resident) and JIT ("int4-jit")
// INT4 MoE prefill formats are advertised only by a provider whose engine links the
// FAST_I4 runtime and its controls. Until that runtime is on main, providers leave
// these candidates out, so no plan can select a format the binary cannot execute.
// Ordered by preference: resident where it fits, JIT otherwise.
[[nodiscard]] inline std::vector<Format> int4_moe_formats(const std::string& decode, std::uint32_t moe_layers,
                                                          std::uint32_t jit_slots) {
    return {{decode, "int4-stored", moe_layers, moe_layers, 0}, {decode, "int4-jit", 0, moe_layers, jit_slots}};
}

class FootprintProvider {
public:
    virtual ~FootprintProvider() = default;
    [[nodiscard]] virtual std::string model() const = 0;
    [[nodiscard]] virtual std::vector<std::string> topologies() const = 0;
    [[nodiscard]] virtual std::vector<Format> formats(const std::string& topology) const = 0;
    [[nodiscard]] virtual std::uint32_t max_slots() const { return 1; }
    [[nodiscard]] virtual Footprint footprint(const Spec&, const Candidate&, std::uint32_t rank) const = 0;
};

} // namespace strix::deploy
