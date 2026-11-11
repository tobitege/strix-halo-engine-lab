#pragma once

#include "strix/deploy/device.hpp"
#include <cstdint>
#include <map>
#include <string>
#include <string_view>

namespace strix::deploy {

struct Format {
    std::string decode = "auto", prefill = "auto";
    // Provider-defined resident layer count; partial residency is an explicit candidate.
    std::uint32_t resident_layers = 0, total_layers = 0, jit_slots = 0;
    bool operator==(const Format&) const = default;
};
struct Spec {
    std::string model, link = "auto", topology = "auto", profile = "auto";
    std::uint32_t nodes = 0; // 0: use supplied measured inventory
    Format format;
    std::uint32_t context = 8192, slots = 0; // slots 0: search provider's supported range
    struct Memory { double margin_gib = 8; std::string policy = "kv-first"; } memory;
    // Explicit TOML selections; engine adapters use these to preserve implicit CLI defaults.
    std::map<std::string, std::string> pins;
    // Optional orchestrator-supplied measured inventory, one selected device per node.
    // No memory tiers or inferred remote memory. CLI execution remeasures the local entry.
    std::vector<DeviceFacts> devices;
};

[[nodiscard]] Spec parse_spec(std::string_view toml); // "auto" is also accepted
[[nodiscard]] Spec load_spec(const std::string& file_or_auto);
void validate_spec(const Spec& spec);

} // namespace strix::deploy
