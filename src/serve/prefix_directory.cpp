#include "strix/serve/prefix_directory.hpp"

#include "strix/env_flags.hpp"

#include <algorithm>
#include <numeric>

namespace strix::serve {
namespace {

std::uint64_t mix(std::uint64_t x) noexcept {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

std::uint64_t common_length(std::span<const std::uint32_t> a, std::span<const std::uint32_t> b) noexcept {
    const auto head = a.first(std::min(a.size(), b.size()));
    return static_cast<std::uint64_t>(std::mismatch(head.begin(), head.end(), b.begin()).first - head.begin());
}

} // namespace

std::uint64_t prefix_block_hash(std::uint64_t chain, std::span<const std::uint32_t> block) noexcept {
    std::uint64_t h = mix(chain + 0x9e3779b97f4a7c15ULL);
    for (const std::uint32_t token : block) h = mix(h ^ (static_cast<std::uint64_t>(token) + 0x9e3779b97f4a7c15ULL));
    return h;
}

bool prefix_fork_enabled_from_env() {
    return env_flag_default_off("STRIX_SERVE_PREFIX_FORK");
}

PrefixDirectory::PrefixDirectory(std::uint32_t block_tokens, std::uint64_t min_tokens, PrefixBlockHash hash)
    : block_tokens_(std::max<std::uint32_t>(block_tokens, 1)),
      min_tokens_(std::max<std::uint64_t>(min_tokens, 1)),
      hash_(hash != nullptr ? hash : &prefix_block_hash) {}

std::uint64_t PrefixDirectory::seed(const PrefixKey& key) const noexcept {
    return mix(mix(mix(key.model_id) ^ key.template_id) ^ key.format_id);
}

void PrefixDirectory::chain_of(const PrefixKey& key, std::span<const std::uint32_t> tokens,
                               std::vector<std::uint64_t>& chain) const {
    const std::size_t blocks = tokens.size() / block_tokens_;
    chain.clear();
    chain.reserve(blocks + 1);
    chain.push_back(seed(key));
    for (std::size_t b = 0; b < blocks; ++b)
        chain.push_back(hash_(chain.back(), tokens.subspan(b * block_tokens_, block_tokens_)));
}

void PrefixDirectory::unindex(std::uint64_t id, const Entry& entry, std::size_t from_depth) noexcept {
    for (std::size_t d = from_depth; d < entry.chain.size(); ++d) {
        const auto range = index_.equal_range(entry.chain[d]);
        for (auto it = range.first; it != range.second; ++it) {
            if (it->second == id) {
                index_.erase(it);
                break;
            }
        }
    }
}

PrefixMatch PrefixDirectory::lookup(const PrefixKey& key, std::span<const std::uint32_t> tokens) const {
    PrefixMatch best;
    if (tokens.size() < 2 || entries_.empty()) return best;
    const std::uint64_t limit = tokens.size() - 1; // one suffix token stays with the caller
    if (limit < min_tokens_) return best;
    std::vector<std::uint64_t> chain;
    chain_of(key, tokens, chain);
    std::vector<std::uint64_t> seen;
    for (std::size_t d = chain.size(); d-- > 0;) {
        const auto range = index_.equal_range(chain[d]);
        for (auto it = range.first; it != range.second; ++it) {
            const std::uint64_t id = it->second;
            if (std::find(seen.begin(), seen.end(), id) != seen.end()) continue;
            seen.push_back(id);
            const auto found = entries_.find(id);
            if (found == entries_.end()) continue;
            const Entry& entry = found->second;
            if (!(entry.key == key)) continue;
            const std::uint64_t common = common_length(entry.tokens, tokens);
            std::uint64_t usable = 0;
            if (entry.state == PrefixState::sealed) {
                if (common == entry.tokens.size() && common <= limit) usable = common;
            } else if (entry.state == PrefixState::inflight) {
                usable = std::min(common, limit);
            }
            if (usable < min_tokens_) continue;
            const bool inflight = entry.state == PrefixState::inflight;
            const bool better = usable > best.tokens ||
                (usable == best.tokens && ((best.inflight && !inflight) ||
                                           (best.inflight == inflight && id < best.entry)));
            if (best.entry == 0 || better) best = PrefixMatch{id, usable, inflight};
        }
        // Every entry indexed at depth d shares d verified blocks or was rejected above, and an entry
        // absent from depth d shares fewer than d blocks, so a match here cannot be beaten below.
        if (best.entry != 0 && best.tokens >= static_cast<std::uint64_t>(d) * block_tokens_) break;
    }
    return best;
}

std::uint64_t PrefixDirectory::announce(const PrefixKey& key, std::span<const std::uint32_t> tokens,
                                        std::uint32_t lane) {
    if (tokens.size() < min_tokens_) return 0;
    Entry entry;
    entry.key = key;
    entry.tokens.assign(tokens.begin(), tokens.end());
    chain_of(key, tokens, entry.chain);
    entry.lane = lane;
    const std::uint64_t id = next_++;
    for (const std::uint64_t value : entry.chain) index_.emplace(value, id);
    entries_.emplace(id, std::move(entry));
    return id;
}

void PrefixDirectory::publish(std::uint64_t entry, std::uint64_t frontier_tokens) {
    const auto found = entries_.find(entry);
    if (found == entries_.end()) return;
    Entry& e = found->second;
    if (e.state == PrefixState::withdrawn) return;
    const bool valid = frontier_tokens != 0 && frontier_tokens >= min_tokens_ &&
        frontier_tokens <= e.tokens.size() &&
        (e.state == PrefixState::inflight || frontier_tokens == e.tokens.size());
    if (!valid) {
        withdraw(entry);
        return;
    }
    if (e.state == PrefixState::sealed) return;
    const std::size_t depth = static_cast<std::size_t>(frontier_tokens / block_tokens_) + 1;
    unindex(entry, e, depth);
    e.chain.resize(depth);
    e.tokens.resize(static_cast<std::size_t>(frontier_tokens));
    e.tokens.shrink_to_fit();
    e.state = PrefixState::sealed;
}

void PrefixDirectory::withdraw(std::uint64_t entry) noexcept {
    const auto found = entries_.find(entry);
    if (found == entries_.end()) return;
    Entry& e = found->second;
    if (e.state != PrefixState::withdrawn) {
        unindex(entry, e, 0);
        e.state = PrefixState::withdrawn;
        std::vector<std::uint32_t>().swap(e.tokens);
        std::vector<std::uint64_t>().swap(e.chain);
    }
    if (e.leases == 0) entries_.erase(found);
}

bool PrefixDirectory::acquire(std::uint64_t entry) noexcept {
    const auto found = entries_.find(entry);
    if (found == entries_.end() || found->second.state == PrefixState::withdrawn) return false;
    ++found->second.leases;
    return true;
}

void PrefixDirectory::release(std::uint64_t entry) noexcept {
    const auto found = entries_.find(entry);
    if (found == entries_.end() || found->second.leases == 0) return;
    if (--found->second.leases == 0 && found->second.state == PrefixState::withdrawn) entries_.erase(found);
}

PrefixStatus PrefixDirectory::status(std::uint64_t entry) const noexcept {
    const auto found = entries_.find(entry);
    if (found == entries_.end()) return {};
    const Entry& e = found->second;
    return PrefixStatus{e.state, e.tokens.size(), e.lane, e.leases};
}

std::vector<PrefixDirectory::Group> PrefixDirectory::plan_groups(
    std::span<const std::span<const std::uint32_t>> prompts) const {
    std::vector<Group> groups;
    if (prompts.size() < 2) return groups;
    // Sorted order puts every set of prompts sharing a prefix in one contiguous run.
    std::vector<std::uint32_t> order(prompts.size());
    std::iota(order.begin(), order.end(), 0U);
    std::sort(order.begin(), order.end(), [&](std::uint32_t a, std::uint32_t b) {
        const auto& x = prompts[a];
        const auto& y = prompts[b];
        if (std::lexicographical_compare(x.begin(), x.end(), y.begin(), y.end())) return true;
        if (std::lexicographical_compare(y.begin(), y.end(), x.begin(), x.end())) return false;
        return a < b;
    });
    const auto shared = [&](std::uint32_t a, std::uint32_t b) -> std::uint64_t {
        const auto& x = prompts[a];
        const auto& y = prompts[b];
        if (x.empty() || y.empty()) return 0;
        return std::min<std::uint64_t>(common_length(x, y), std::min(x.size(), y.size()) - 1);
    };
    while (order.size() >= 2) {
        std::vector<std::uint64_t> adjacent(order.size() - 1);
        for (std::size_t i = 0; i + 1 < order.size(); ++i) adjacent[i] = shared(order[i], order[i + 1]);
        std::uint64_t best_saved = 0, best_tokens = 0;
        std::size_t best_first = 0, best_last = 0;
        for (std::size_t first = 0; first + 1 < order.size(); ++first) {
            std::uint64_t tokens = adjacent[first];
            for (std::size_t last = first + 1; last < order.size(); ++last) {
                tokens = std::min(tokens, adjacent[last - 1]);
                if (tokens < min_tokens_) break;
                const std::uint64_t saved = tokens * static_cast<std::uint64_t>(last - first);
                if (saved > best_saved || (saved == best_saved && tokens > best_tokens)) {
                    best_saved = saved;
                    best_tokens = tokens;
                    best_first = first;
                    best_last = last;
                }
            }
        }
        if (best_saved == 0) break;
        Group group;
        group.tokens = best_tokens;
        group.members.assign(order.begin() + static_cast<std::ptrdiff_t>(best_first),
                             order.begin() + static_cast<std::ptrdiff_t>(best_last) + 1);
        std::sort(group.members.begin(), group.members.end());
        group.producer = group.members.front();
        groups.push_back(std::move(group));
        order.erase(order.begin() + static_cast<std::ptrdiff_t>(best_first),
                    order.begin() + static_cast<std::ptrdiff_t>(best_last) + 1);
    }
    std::sort(groups.begin(), groups.end(),
              [](const Group& a, const Group& b) { return a.producer < b.producer; });
    return groups;
}

} // namespace strix::serve
