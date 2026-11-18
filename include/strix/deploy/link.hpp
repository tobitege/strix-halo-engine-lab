#pragma once

#include "strix/device/declaration.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// The link graph of one device declaration: which declared units can exchange bytes, at what rate, and which
// physical resource each transfer uses. Shared by every engine's placement (deploy/route.hpp); no model knowledge.
//
// Nodes are "host" (the declaring machine) and every declared, non-excluded GPU/NPU id. Edges come from
// devices.<id>.attach (host <-> device) and links.<name> (device <-> device). A rate is MEASURED when the declaration
// carries the probe (h2d_gbytes_ps for PCIe, gbps_measured for remote and links), NOMINAL when it comes from the link
// class (device/link_class.hpp) or the negotiated PCIe gen/width, UNKNOWN otherwise. Unknown is not infinite: an
// unknown edge is never traversed, and the path reports which edge is missing. Rates are symmetric until the
// declaration carries a per-direction probe.
namespace strix::deploy {

enum class RateSource { measured, nominal, unknown, integrated };
[[nodiscard]] std::string_view rate_source_name(RateSource source) noexcept;

struct LinkEdge {
    std::string name;       // "attach:<id>" or "link:<name>"
    std::string a, b;       // node ids
    std::string link;       // class name, "" when derived from PCIe keys or integrated
    std::string resource;   // shared bandwidth resource (attach.port / links.port)
    double gbytes_ps = 0;   // useful payload GB/s (decimal); 0 = unknown; ignored for integrated
    double rtt_us = 0;
    double uncertainty = 0; // relative half-width of gbytes_ps
    RateSource source = RateSource::unknown;
};

struct LinkGraph {
    std::vector<std::string> nodes; // "host" first, then devices in declaration order
    std::vector<LinkEdge> edges;
};

[[nodiscard]] LinkGraph link_graph(const device::Declaration& declaration);

struct LinkPath {
    std::vector<std::size_t> edges; // into LinkGraph::edges, in order from -> to; empty = same node
    double gbytes_ps = 0;           // bottleneck of the non-integrated edges; 0 with no such edge (no copy)
    double rtt_us = 0;              // sum over edges
    double uncertainty = 0;         // largest edge uncertainty
    [[nodiscard]] bool copy_free() const noexcept { return gbytes_ps == 0; }
};

struct PathResult {
    std::optional<LinkPath> path;
    std::string reason; // why there is no usable path (unknown edge, no edge at all)
};

// Best path: the largest bottleneck rate, then the smallest round trip, then the fewest edges. Pieces stream, so a
// multi-hop transfer on independent resources is limited by its slowest edge. Route timing additionally
// accounts for repeated/shared resource use. Search is polynomial (widest path, then RTT/hop minimization).
[[nodiscard]] PathResult best_path(const LinkGraph& graph, std::string_view from, std::string_view to);

// Isolated-path lower bound, assuming independent hop resources (one round trip per message).
// The route planner uses graph resource accounting when resources are shared or repeated.
[[nodiscard]] double transfer_ms(const LinkPath& path, std::uint64_t bytes, std::uint64_t messages = 1) noexcept;

} // namespace strix::deploy
