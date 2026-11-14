#pragma once
// Host-only lane snapshots: a thin adapter over strix::serve::ConversationKeep (index, retention, byte ledger, leases,
// pressure plans). This file only owns the payload bytes and the GLM-specific key (MTP variant). Rank 0 picks the
// victims, rank 1 replays them; both ranks make the same ledger decisions.
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <list>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "strix/serve/conversation_keep.hpp"

namespace strix::glm::serve {

inline std::uint64_t prefix_budget_bytes(const std::string& v) {
    std::size_t end = 0;
    const double g = std::stod(v, &end);
    if (end != v.size() || !std::isfinite(g) || g < 0 || g >= 17179869184.0)
        throw std::invalid_argument("--prefix-cache-gib: finite GiB >= 0, less than 17179869184");
    const auto bytes = static_cast<std::uint64_t>(g * 1073741824.0);
    if (g > 0 && bytes == 0) throw std::invalid_argument("--prefix-cache-gib: budget below one byte");
    return bytes;
}
inline constexpr std::uint64_t kPrefixMemoryFloor = 16ULL * 1024 * 1024 * 1024;  // MemAvailable a snapshot store needs
// MemAvailable in bytes; nullopt when the line is missing or malformed.
inline std::optional<std::uint64_t> prefix_memory_bytes(std::istream& f) {
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("MemAvailable:", 0) != 0) continue;
        std::istringstream in(line.substr(13));
        std::string value, unit, extra;
        in >> value >> unit;
        if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos || unit != "kB" || (in >> extra)) return std::nullopt;
        try { return std::stoull(value) * 1024ULL; }
        catch (...) { return std::nullopt; }
    }
    return std::nullopt;
}
inline bool prefix_memory_available(std::istream& f) {
    const auto b = prefix_memory_bytes(f);
    return b && *b >= kPrefixMemoryFloor;
}
inline bool prefix_memory_available() {
    std::ifstream f("/proc/meminfo");
    return prefix_memory_available(f);
}
inline std::optional<std::uint64_t> prefix_memory_bytes() {
    std::ifstream f("/proc/meminfo");
    return prefix_memory_bytes(f);
}

inline std::uint64_t prefix_snapshot_bound(std::uint64_t p) {
    // Full 45-layer layout + MTP, world=1 and unsharded: upper bound for every served layout.
    // 34 KDA layers; 11 DSA plus one MTP DSA. Include the seven-word shard header.
    constexpr std::uint64_t fixed = 34ULL * (64ULL * 128 * 128 * 4 + 3ULL * 64 * 128 * 3 * 4) + 7 * 8;
    constexpr std::uint64_t per_row = 12ULL * (512 * 2 + 128 * 2 * 2 + 128);
    if (p > (UINT64_MAX - fixed) / per_row) return UINT64_MAX;
    return fixed + p * per_row;
}

class PrefixCache {
public:
    using Kind = strix::serve::KeepKind;
    // What a store prepares: key ids (the prompt, plus the MTP pending token), the rows the snapshot restores, bytes.
    struct Entry {
        std::uint64_t id = 0, identity = 0, frontier = 0;
        bool mtp = false;
        Kind kind = Kind::prompt_end;
        std::vector<std::uint32_t> ids;
        std::vector<std::uint8_t> state;
    };
    explicit PrefixCache(std::uint64_t budget = 0, strix::serve::PrefixBlockHash hash = nullptr) : keep_(budget, hash) {}

    static strix::serve::KeepIdentity key_of(bool mtp, std::uint64_t identity) { return {identity, mtp ? 1ULL : 0ULL}; }
    const strix::serve::KeepEntry* find(std::span<const std::uint32_t> prompt, bool mtp, std::uint64_t identity) const {
        return keep_.find(key_of(mtp, identity), prompt);
    }
    const strix::serve::KeepEntry* lookup(std::span<const std::uint32_t> prompt, bool mtp, std::uint64_t identity) {
        return keep_.lookup(key_of(mtp, identity), prompt);
    }
    const strix::serve::KeepEntry* get(std::uint64_t id) const { return keep_.get(id); }
    const std::vector<std::uint8_t>* state(std::uint64_t id) const {
        for (const auto& p : payload_) if (p.first == id) return &p.second;
        return nullptr;
    }
    std::uint64_t replacement_id() const { return keep_.replacement_id(); }
    std::uint64_t oldest() const { return keep_.oldest(); }
    bool erase(std::uint64_t id, bool pressure = false) {
        if (!keep_.erase(id, pressure ? strix::serve::KeepEvict::pressure : strix::serve::KeepEvict::lru)) return false;
        payload_.remove_if([id](const auto& p) { return p.first == id; });
        return true;
    }
    void touch(std::uint64_t id) { keep_.touch(id); }
    bool acquire(std::uint64_t id) { return keep_.acquire(id); }
    void release(std::uint64_t id) { keep_.release(id); }
    bool prepare(Entry e) {
        commit(false);
        strix::serve::KeepCandidate c;
        c.id = e.id;
        c.identity = key_of(e.mtp, e.identity);
        c.kind = e.kind;
        c.ids = std::move(e.ids);
        c.frontier = e.frontier;
        c.payload_bytes = e.state.size();
        if (!keep_.can_stage(c.payload_bytes, c.ids.size())) return false;
        try {
            pending_payload_.emplace_back(e.id, std::move(e.state));
            if (!keep_.prepare(std::move(c))) { commit(false); return false; }
        } catch (...) { commit(false); return false; }
        return true;
    }
    // Ordered eviction list that makes room for the prepared entry (lead only; the follower replays the erasures).
    bool victims(std::vector<std::uint64_t>& out) const { return keep_.victims(out); }
    bool pending_fits() const { return keep_.pending_fits(); }
    void commit(bool accept) noexcept {
        if (keep_.commit(accept)) payload_.splice(payload_.end(), pending_payload_);
        pending_payload_.clear(); // destroy the owner, including vector capacity
    }
    bool can_stage(std::uint64_t bytes, std::size_t tokens) const { return keep_.can_stage(bytes, tokens); }
    std::vector<std::uint64_t> reclaim_plan(std::uint64_t target) const { return keep_.reclaim_plan(target); }
    std::vector<std::uint64_t> shrink_plan() const { return keep_.shrink_plan(); }
    void set_budget(std::uint64_t b) { keep_.set_budget(b); }
    void clear() { commit(false); keep_.clear(); payload_.clear(); }
    std::uint64_t bytes() const { return keep_.bytes(); }
    std::uint64_t pending_bytes() const { return keep_.pending_bytes(); }
    std::uint64_t budget() const { return keep_.budget(); }
    std::size_t size() const { return keep_.size(); }
    strix::serve::KeepStats stats() const { return keep_.stats(); }
    strix::serve::ConversationKeep& keep() { return keep_; }
    const strix::serve::ConversationKeep& keep() const { return keep_; }

private:
    strix::serve::ConversationKeep keep_;
    std::list<std::pair<std::uint64_t, std::vector<std::uint8_t>>> payload_, pending_payload_;
};

}  // namespace strix::glm::serve
