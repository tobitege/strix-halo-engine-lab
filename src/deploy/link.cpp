#include "strix/deploy/link.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace strix::deploy {
namespace {
// Useful PCIe payload per lane and direction (GB/s) by generation; 0.8 of it is assumed usable for bulk copies
// (nominal, uncertainty 0.3) until h2d_gbytes_ps is probed.
constexpr std::array<double, 8> pcie_lane_gbytes_ps{0.0, 0.25, 0.5, 0.985, 1.969, 3.938, 7.877, 15.754};

LinkEdge from_class(std::string name, std::string a, std::string b, const std::string& cls, std::string resource) {
    LinkEdge e{std::move(name), std::move(a), std::move(b), cls, std::move(resource)};
    if (const auto* c = device::find_link_class(cls)) {
        e.gbytes_ps = c->gbytes_ps;
        e.rtt_us = c->rtt_us;
        e.uncertainty = c->uncertainty;
        e.source = RateSource::nominal;
    }
    return e;
}
void measured(LinkEdge& e, double gbytes_ps, std::uint64_t rtt_us) {
    if (gbytes_ps > 0) { e.gbytes_ps = gbytes_ps; e.uncertainty = 0.10; e.source = RateSource::measured; }
    if (rtt_us > 0) e.rtt_us = static_cast<double>(rtt_us);
}
} // namespace

std::string_view rate_source_name(RateSource source) noexcept {
    switch (source) {
    case RateSource::measured: return "measured";
    case RateSource::nominal: return "nominal";
    case RateSource::integrated: return "integrated";
    case RateSource::unknown: break;
    }
    return "unknown";
}

LinkGraph link_graph(const device::Declaration& declaration) {
    LinkGraph g;
    g.nodes.push_back("host");
    for (const auto& d : declaration.devices) {
        if (d.excluded || d.kind == device::Kind::cpu) continue;
        g.nodes.push_back(d.id);
        const auto& at = d.attach;
        if (at.kind.empty()) continue; // no attach: not reachable from this machine
        auto e = from_class("attach:" + d.id, "host", d.id, at.link, at.port);
        if (at.kind == "integrated") {
            e.source = RateSource::integrated;
            e.uncertainty = 0;
        } else if (at.kind == "remote") {
            measured(e, at.gbps_measured / 8.0, at.rtt_us_measured);
        } else {
            if (e.source == RateSource::unknown && at.pcie_gen > 0 && at.pcie_width > 0) {
                e.gbytes_ps = 0.8 * pcie_lane_gbytes_ps[at.pcie_gen] * at.pcie_width;
                e.rtt_us = 5.0;
                e.uncertainty = 0.30;
                e.source = RateSource::nominal;
            }
            measured(e, at.h2d_gbytes_ps, at.rtt_us_measured);
        }
        g.edges.push_back(std::move(e));
    }
    const auto present = [&](const std::string& id) { return std::find(g.nodes.begin(), g.nodes.end(), id) != g.nodes.end(); };
    for (const auto& l : declaration.links) {
        if (!present(l.a) || !present(l.b)) continue; // an excluded end takes no traffic
        auto e = from_class("link:" + l.name, l.a, l.b, l.link, l.port);
        measured(e, l.gbps_measured / 8.0, l.rtt_us_measured);
        g.edges.push_back(std::move(e));
    }
    return g;
}

PathResult best_path(const LinkGraph& graph, std::string_view from, std::string_view to) {
    PathResult out;
    const auto has = [&](std::string_view id) { return std::find(graph.nodes.begin(), graph.nodes.end(), id) != graph.nodes.end(); };
    if (!has(from) || !has(to)) {
        out.reason = "no node " + std::string(has(from) ? to : from) + " in the link graph";
        return out;
    }
    if (from == to) { out.path = LinkPath{}; return out; }
    const auto N = graph.nodes.size();
    const auto index = [&](std::string_view id) {
        return static_cast<std::size_t>(std::find(graph.nodes.begin(), graph.nodes.end(), id) - graph.nodes.begin());
    };
    const auto source = index(from), target = index(to);
    const double inf = std::numeric_limits<double>::infinity();
    struct Arc { std::size_t next, edge; double rate; };
    std::vector<std::vector<Arc>> adj(N);
    std::string blocked;
    for (std::size_t i = 0; i < graph.edges.size(); ++i) {
        const auto& e = graph.edges[i];
        if (e.source == RateSource::unknown) {
            if (blocked.empty()) blocked = e.name + " has no measured rate and no link class";
            continue;
        }
        const auto a = index(e.a), b = index(e.b);
        if (a == N || b == N) continue;
        const double rate = e.source == RateSource::integrated ? inf : e.gbytes_ps;
        if (!(rate > 0) || !(e.rtt_us >= 0)) continue;
        adj[a].push_back({b, i, rate});
        adj[b].push_back({a, i, rate});
    }
    // Widest path, then shortest RTT/fewest hops restricted to that bottleneck.
    // Two label-setting passes avoid enumerating exponentially many simple paths.
    std::vector<double> width(N, 0), distance(N, inf);
    std::vector<bool> done(N, false);
    width[source] = inf;
    for (std::size_t step = 0; step < N; ++step) {
        std::size_t at = N;
        for (std::size_t i = 0; i < N; ++i)
            if (!done[i] && (at == N || width[i] > width[at])) at = i;
        if (at == N || width[at] == 0) break;
        done[at] = true;
        for (const auto& a : adj[at]) width[a.next] = std::max(width[a.next], std::min(width[at], a.rate));
    }
    if (width[target] == 0) {
        out.reason = blocked.empty() ? "no declared link joins " + std::string(from) + " and " + std::string(to) : blocked;
        return out;
    }
    std::fill(done.begin(), done.end(), false);
    std::vector<std::size_t> hops(N, N), previous(N, N), edge(N, 0);
    distance[source] = 0; hops[source] = 0;
    for (std::size_t step = 0; step < N; ++step) {
        std::size_t at = N;
        for (std::size_t i = 0; i < N; ++i)
            if (!done[i] && (at == N || distance[i] < distance[at] ||
                            (distance[i] == distance[at] && hops[i] < hops[at]))) at = i;
        if (at == N || distance[at] == inf) break;
        done[at] = true;
        for (const auto& a : adj[at]) {
            if (done[a.next] || a.rate < width[target]) continue;
            const double d = distance[at] + graph.edges[a.edge].rtt_us;
            if (d < distance[a.next] || (d == distance[a.next] && hops[at] + 1 < hops[a.next])) {
                distance[a.next] = d; hops[a.next] = hops[at] + 1;
                previous[a.next] = at; edge[a.next] = a.edge;
            }
        }
    }
    LinkPath p;
    for (auto at = target; at != source; at = previous[at]) p.edges.push_back(edge[at]);
    std::reverse(p.edges.begin(), p.edges.end());
    p.gbytes_ps = width[target] == inf ? 0 : width[target];
    p.rtt_us = distance[target];
    for (const auto i : p.edges) p.uncertainty = std::max(p.uncertainty, graph.edges[i].uncertainty);
    out.path = std::move(p);
    return out;
}

double transfer_ms(const LinkPath& path, std::uint64_t bytes, std::uint64_t messages) noexcept {
    const double wire = path.copy_free() ? 0.0 : static_cast<double>(bytes) / (path.gbytes_ps * 1e6);
    return wire + static_cast<double>(messages) * path.rtt_us / 1e3;
}

} // namespace strix::deploy
