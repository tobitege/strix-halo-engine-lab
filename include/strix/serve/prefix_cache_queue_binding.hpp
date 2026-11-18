#pragma once
// Binds QueuedPrefixProtection to a PrefixStateCache: preparation = host-only prepare_lookup (prompt-only block
// digests, the same limit the restore lookup uses), protection = protect_longest under a byte cap.
// The store is held weakly: a released store makes preparation and protection no-ops, and live handles
// then release nothing. `identity` must stay callable while the protector lives.
#include "strix/prefix_state_cache.hpp"
#include "strix/serve/queued_prefix_protection.hpp"

#include <functional>
#include <memory>

namespace strix::serve {

// identity: the lookup identity the engine's restore will use for this prompt (called at preparation).
// limit: restorable rows for a prompt of n tokens (the restore lookup's limit; default n - 1).
[[nodiscard]] inline QueuedPrefixProtection::Binding prefix_cache_queue_binding(std::weak_ptr<PrefixStateCache> store,
    std::function<ModelContentIdentity()> identity, std::uint32_t block_rows, std::uint64_t max_bytes = 0,
    std::function<std::uint64_t(std::uint64_t)> limit = {}) {
    QueuedPrefixProtection::Binding b;
    b.prepare = [store, identity = std::move(identity), block_rows, limit = std::move(limit)](
                    std::span<const std::uint32_t> prompt) -> QueuedPrefixProtection::Prepared {
        const auto cache = store.lock();
        if (!cache || !block_rows || prompt.size() < 2 || !identity) return {};
        const auto rows = limit ? limit(prompt.size()) : prompt.size() - 1;
        if (rows < block_rows) return {};
        return std::make_shared<const PrefixLookupTicket>(cache->prepare_lookup(identity(), prompt, block_rows, rows, false));
    };
    b.protect = [store = std::move(store), max_bytes](const QueuedPrefixProtection::Prepared& prepared) -> QueuedPrefixProtection::Handle {
        const auto cache = store.lock();
        if (!cache || !prepared) return {};
        auto p = cache->protect_longest(*std::static_pointer_cast<const PrefixLookupTicket>(prepared), max_bytes);
        if (!p) return {};
        return std::make_shared<PrefixStateCache::QueueProtection>(std::move(p));
    };
    return b;
}

} // namespace strix::serve
