#include "strix/session/parallel_job.hpp"

#include <map>
#include <set>
#include <stdexcept>
#include <utility>

namespace strix::session {
namespace {
void require(bool ok, const char* message) {
    if (!ok) throw std::invalid_argument(message);
}
}
std::span<const std::uint32_t> TokenReference::tokens() const {
    const auto all = buffer.tokens();
    require(offset <= all.size() && length <= all.size() - offset, "token reference bounds");
    return all.subspan(static_cast<std::size_t>(offset), static_cast<std::size_t>(length));
}
void validate_job(const ParallelJob& job) {
    require(!job.request_id.empty(), "missing request ID");
    require(job.policy.token_budget > 0 && job.policy.max_lanes > 0, "invalid job budget or lanes");
    require(!job.points.empty(), "missing points");
    (void)job.prefix.tokens();
    std::map<std::string, std::size_t> ids;
    std::uint64_t remaining = job.policy.token_budget;
    for (std::size_t i = 0; i < job.points.size(); ++i) {
        const auto& p = job.points[i];
        require(!p.id.empty() && ids.emplace(p.id, i).second, "missing or duplicate point ID");
        const auto& c = p.contract;
        require(c.max_tokens > 0 && c.min_tokens <= c.max_tokens &&
                c.max_tokens <= job.policy.token_budget, "invalid point lengths");
        require(c.min_tokens <= remaining, "point minima exceed request budget");
        remaining -= c.min_tokens;
        (void)p.suffix.tokens();
    }
    const auto index = [&](const std::string& id) {
        const auto it = ids.find(id);
        require(it != ids.end(), "unknown point ID");
        return it->second;
    };
    std::vector<std::vector<std::size_t>> next(ids.size());
    std::vector<std::size_t> degree(ids.size());
    std::set<std::pair<std::size_t, std::size_t>> edges;
    for (const auto& e : job.dependencies) {
        const auto a = index(e.before), b = index(e.after);
        require(a != b && edges.emplace(a, b).second, "self or duplicate dependency");
        next[a].push_back(b); ++degree[b];
    }
    std::vector<std::size_t> ready;
    for (std::size_t i = 0; i < degree.size(); ++i) if (!degree[i]) ready.push_back(i);
    for (std::size_t i = 0; i < ready.size(); ++i)
        for (auto b : next[ready[i]]) if (!--degree[b]) ready.push_back(b);
    require(ready.size() == ids.size(), "dependency cycle");
    require(job.merge.point_order.size() == ids.size(), "incomplete merge order");
    std::vector<std::size_t> order(ids.size(), ids.size());
    for (std::size_t i = 0; i < job.merge.point_order.size(); ++i) {
        auto p = index(job.merge.point_order[i]);
        require(order[p] == ids.size(), "duplicate merge point"); order[p] = i;
    }
    for (const auto& e : edges) require(order[e.first] < order[e.second], "merge violates dependency order");
    std::map<std::string, std::size_t> digests;
    for (const auto& d : job.digests) {
        require(!d.id.empty() && digests.emplace(d.id, index(d.producer)).second,
                "missing or duplicate digest ID");
        (void)d.evidence.tokens();
    }
    for (std::size_t consumer = 0; consumer < job.points.size(); ++consumer) {
        std::set<std::string> used;
        for (const auto& id : job.points[consumer].contract.evidence_ids) {
            const auto d = digests.find(id);
            require(d != digests.end() && used.insert(id).second, "unknown or duplicate evidence ID");
            // Iterative reachability avoids recursion on caller-sized DAGs.
            std::vector<bool> seen(ids.size());
            std::vector<std::size_t> work{d->second}; seen[d->second] = true;
            for (std::size_t n = 0; n < work.size(); ++n)
                for (auto b : next[work[n]]) if (!seen[b]) { seen[b] = true; work.push_back(b); }
            require(d->second != consumer && seen[consumer], "evidence producer is not a predecessor");
        }
    }
}
} // namespace strix::session
