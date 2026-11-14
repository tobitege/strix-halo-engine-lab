#include "strix/vision/embed_cache.hpp"

#include <cstdlib>
#include <stdexcept>
#include <utility>

namespace strix::vision {

EmbedCache::EmbedCache(std::size_t budget_bytes) { stats_.budget = budget_bytes; }

std::size_t EmbedCache::budget_from_env() {
    const char* v = std::getenv("STRIX_VISION_CACHE_MIB");
    if (!v || !*v) return kDefaultBudget;
    char* end = nullptr;
    const unsigned long long mib = std::strtoull(v, &end, 10);
    if (end == v || *end != '\0' || mib > (std::size_t{1} << 20U)) throw std::invalid_argument("STRIX_VISION_CACHE_MIB");
    return static_cast<std::size_t>(mib) << 20U;
}

std::string EmbedCache::key_of(std::string_view identity, const Digest& digest) {
    std::string key;
    key.reserve(identity.size() + 1U + digest.size());
    key.append(identity);
    key.push_back('\0');
    key.append(reinterpret_cast<const char*>(digest.data()), digest.size());
    return key;
}

EmbedCache::Rows EmbedCache::find(std::string_view identity, const Digest& digest) {
    const std::string key = key_of(identity, digest);
    const std::lock_guard<std::mutex> lock(mutex_);
    const auto it = index_.find(key);
    if (it == index_.end()) {
        ++stats_.misses;
        return nullptr;
    }
    ++stats_.hits;
    lru_.splice(lru_.begin(), lru_, it->second);
    return it->second->rows;
}

EmbedCache::Rows EmbedCache::insert(std::string_view identity, const Digest& digest, ProjectedRows&& rows) {
    std::string key = key_of(identity, digest);
    const std::size_t bytes = rows.bytes() + key.size();
    Rows shared = std::make_shared<const ProjectedRows>(std::move(rows));
    const std::lock_guard<std::mutex> lock(mutex_);
    if (const auto it = index_.find(key); it != index_.end()) {
        lru_.splice(lru_.begin(), lru_, it->second);
        return it->second->rows;
    }
    if (bytes > stats_.budget) {
        ++stats_.rejected;
        return shared;
    }
    while (stats_.bytes + bytes > stats_.budget && !lru_.empty()) {
        const auto& last = lru_.back();
        stats_.bytes -= last.bytes;
        index_.erase(last.key);
        lru_.pop_back();
        ++stats_.evictions;
    }
    lru_.push_front(Entry{key, shared, bytes});
    index_.emplace(std::move(key), lru_.begin());
    stats_.bytes += bytes;
    return shared;
}

void EmbedCache::clear() {
    const std::lock_guard<std::mutex> lock(mutex_);
    lru_.clear();
    index_.clear();
    stats_.bytes = 0;
}

EmbedCache::Stats EmbedCache::stats() const {
    const std::lock_guard<std::mutex> lock(mutex_);
    Stats s = stats_;
    s.entries = lru_.size();
    return s;
}

CachedEncode encode_cached(VisionAdapter& adapter, EncoderBackend& backend, const EncoderWeights& weights, EmbedCache* cache,
                           const Digest& digest, const PatchGrid& grid) {
    // The rows depend on the arithmetic that produced them, so the backend is part of the key.
    const std::string identity = adapter.identity() + "|" + backend.arithmetic();
    if (cache)
        if (auto rows = cache->find(identity, digest)) return {std::move(rows), true};
    const EncoderConfig config = adapter.config();
    EncoderOutput hidden = backend.encode(grid, adapter.positions(config, weights, grid));
    ProjectedRows projected;
    adapter.project(hidden, grid, backend, projected);
    if (projected.bf16.size() != std::size_t{projected.rows} * projected.width) throw std::logic_error("vision adapter: projected row shape");
    for (const auto& d : projected.deep)
        if (d.size() != projected.bf16.size()) throw std::logic_error("vision adapter: deepstack row shape");
    if (cache) return {cache->insert(identity, digest, std::move(projected)), false};
    return {std::make_shared<const ProjectedRows>(std::move(projected)), false};
}

}  // namespace strix::vision
