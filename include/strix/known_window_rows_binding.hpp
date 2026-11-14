#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace strix {

// Owner-held, like preflight_slots. Clear with that memo before replacing resources.
// Bound values must own their metadata: never retain a call-local dispatch context.
class KnownWindowRowsBindingCache {
    struct Slot {
        std::array<std::uintptr_t, 2> preflight{};
        const void* type = nullptr;
        std::shared_ptr<const void> value;
    };
    struct Plan {
        const void* key = nullptr;
        const void* slots = nullptr;
        std::size_t rows = 0;
        std::vector<Slot> steps;
    };
    std::vector<Plan> plans_;
    friend class KnownWindowRowsBindingScope;
public:
    void clear() { plans_.clear(); }
};

// Only the shared runtime admits reuse. Other callers bind locally as before.
// Scope restoration also masks an outer cache when a nested call has no memo.
class KnownWindowRowsBindingScope {
    using Slot = KnownWindowRowsBindingCache::Slot;
    inline static thread_local Slot* current_ = nullptr;
    template <class Bound> inline static char type_ = 0;
    Slot* previous_ = current_;
public:
    KnownWindowRowsBindingScope(KnownWindowRowsBindingCache* cache, const void* key,
        const void* slots, std::size_t step, std::size_t rows, std::size_t steps,
        std::array<std::uintptr_t, 2> preflight) {
        Slot* next = nullptr;
        if (cache && key && slots && step < steps) {
            auto* plan = [&]() -> KnownWindowRowsBindingCache::Plan* {
                for (auto& p : cache->plans_) if (p.key == key) return &p;
                cache->plans_.push_back({key, slots, rows, {}});
                return &cache->plans_.back();
            }();
            if (plan->slots != slots || plan->rows != rows || plan->steps.size() != steps) {
                plan->slots = slots;
                plan->rows = rows;
                plan->steps.assign(steps, {});
            }
            next = &plan->steps[step];
            if (next->preflight != preflight) {
                *next = {};
                next->preflight = preflight;
            }
        }
        current_ = next;
    }
    ~KnownWindowRowsBindingScope() { current_ = previous_; }
    KnownWindowRowsBindingScope(const KnownWindowRowsBindingScope&) = delete;
    KnownWindowRowsBindingScope& operator=(const KnownWindowRowsBindingScope&) = delete;

    template <class Bound, class Bind>
    static const Bound& get(std::optional<Bound>& local, Bind&& bind) {
        auto* slot = current_;
        if (!slot) return local.emplace(std::forward<Bind>(bind)());
        if (!slot->value || slot->type != &type_<Bound>) {
            auto value = std::make_shared<const Bound>(std::forward<Bind>(bind)());
            slot->value = std::move(value);
            slot->type = &type_<Bound>;
        }
        return *static_cast<const Bound*>(slot->value.get());
    }
};

} // namespace strix
