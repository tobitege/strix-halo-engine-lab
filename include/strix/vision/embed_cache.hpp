#pragma once
// vision::EmbedCache: LRU of projected image rows by (adapter identity, image digest), byte-budgeted. A conversation
// that re-sends its images every turn pays the encoder once. Thread-safe; entries are immutable and shared.
#include "strix/vision/adapter.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

namespace strix::vision {

using Digest = std::array<std::uint8_t, 32>;  // SHA-256 of the encoded image bytes

class EmbedCache {
public:
    static constexpr std::size_t kDefaultBudget = std::size_t{256} << 20U;
    struct Stats {
        std::uint64_t hits = 0, misses = 0, evictions = 0, rejected = 0;
        std::size_t bytes = 0, entries = 0, budget = 0;
    };
    using Rows = std::shared_ptr<const ProjectedRows>;

    explicit EmbedCache(std::size_t budget_bytes = kDefaultBudget);
    static std::size_t budget_from_env();  // STRIX_VISION_CACHE_MIB, default 256; 0 disables storing

    Rows find(std::string_view identity, const Digest&);  // null on a miss; a hit becomes most recent
    // Stores and returns the rows; an entry for the key already present wins (its rows are returned). Rows larger
    // than the budget are returned but not stored.
    Rows insert(std::string_view identity, const Digest&, ProjectedRows&&);
    void clear();
    Stats stats() const;

private:
    struct Entry {
        std::string key;
        Rows rows;
        std::size_t bytes;
    };
    static std::string key_of(std::string_view identity, const Digest&);

    mutable std::mutex mutex_;
    std::list<Entry> lru_;  // front = most recent
    std::unordered_map<std::string, std::list<Entry>::iterator> index_;
    Stats stats_;
};

struct CachedEncode {
    EmbedCache::Rows rows;
    bool hit = false;
};
// One image through cache -> backend -> projector. `cache` may be null.
CachedEncode encode_cached(VisionAdapter&, EncoderBackend&, const EncoderWeights&, EmbedCache* cache, const Digest&,
                           const PatchGrid&);

}  // namespace strix::vision
