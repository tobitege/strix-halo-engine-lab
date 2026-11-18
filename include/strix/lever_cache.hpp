#pragma once

// A lever value read from the environment once and kept for the process, like a function-local
// `static const`, but re-resolvable per run: the resident runner (strix/serve/resident_session.hpp) calls
// lever_environment_changed() after it sets the environment of an arm, and the next read of every cache
// resolves again through the same reader (normally a strix::envcfg call, so the effective config still
// records it). An ordinary process never bumps the generation, so a cache resolves exactly once, at its
// first read, to the value the old `static const` had: default behaviour is unchanged.
//
//   static strix::lever_cache<bool> fast{[] { return strix::envcfg::flag("STRIX_X_FAST", true); }};
//   if (fast()) ...
//
// set(v) pins a value until the next environment change (the old setters keep working through it).
// Small trivially copyable values are lock-free; other types (std::vector<bool> tables) take a mutex.
// Host-only, no HIP/CUDA types.

#include <atomic>
#include <cstdint>
#include <mutex>
#include <type_traits>

namespace strix {

namespace detail {
inline std::atomic<std::uint64_t>& lever_generation_counter() noexcept {
    static std::atomic<std::uint64_t> g{0};
    return g;
}
}  // namespace detail

// 0 for the whole life of an ordinary process.
[[nodiscard]] inline std::uint64_t lever_generation() noexcept {
    return detail::lever_generation_counter().load(std::memory_order_acquire);
}

// The environment changed (resident runner, after an arm's set/unset): every lever_cache resolves again.
inline void lever_environment_changed() noexcept {
    detail::lever_generation_counter().fetch_add(1, std::memory_order_acq_rel);
}

namespace detail {
template<class T, bool Small = std::is_trivially_copyable_v<T> && sizeof(T) <= sizeof(std::uint64_t)>
struct lever_lock_free : std::false_type {};
template<class T>
struct lever_lock_free<T, true> : std::bool_constant<std::atomic<T>::is_always_lock_free> {};
}  // namespace detail

template<class T>
class lever_cache {
public:
    using Resolve = T (*)();

    explicit lever_cache(Resolve resolve) noexcept : resolve_(resolve) {}
    lever_cache(const lever_cache&) = delete;
    lever_cache& operator=(const lever_cache&) = delete;

    // The value for the current environment generation (resolved on the first read of each generation).
    [[nodiscard]] T get() {
        const std::uint64_t want = lever_generation() + 1U;  // seen_ == 0: never resolved
        if constexpr (kLockFree) {
            if (seen_.load(std::memory_order_acquire) != want) {
                std::lock_guard<std::mutex> lock(mutex_);
                const auto current = lever_generation() + 1U;
                if (seen_.load(std::memory_order_relaxed) != current) {
                    value_.store(resolve_(), std::memory_order_relaxed);
                    seen_.store(current, std::memory_order_release);
                }
            }
            return value_.load(std::memory_order_relaxed);
        } else {
            std::lock_guard<std::mutex> lock(mutex_);
            const auto current = lever_generation() + 1U;
            if (seen_.load(std::memory_order_relaxed) != current) {
                value_ = resolve_();
                seen_.store(current, std::memory_order_relaxed);
            }
            return value_;
        }
    }
    [[nodiscard]] T operator()() { return get(); }

    // Pins `v` until the next lever_environment_changed().
    void set(T v) {
        std::lock_guard<std::mutex> lock(mutex_);
        const std::uint64_t want = lever_generation() + 1U;
        if constexpr (kLockFree) {
            value_.store(v, std::memory_order_relaxed);
            seen_.store(want, std::memory_order_release);
        } else {
            value_ = std::move(v);
            seen_.store(want, std::memory_order_relaxed);
        }
    }

private:
    static constexpr bool kLockFree = detail::lever_lock_free<T>::value;

    Resolve resolve_;
    std::atomic<std::uint64_t> seen_{0};
    std::conditional_t<kLockFree, std::atomic<T>, T> value_{};
    std::mutex mutex_;
};

}  // namespace strix
