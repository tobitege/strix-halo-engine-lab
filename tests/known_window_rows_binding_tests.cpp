#include "strix/known_window_rows_binding.hpp"

#include <iostream>
#include <stdexcept>

namespace {
struct Bound { int resource; };
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
void test_bindings() {
    using strix::KnownWindowRowsBindingCache;
    using strix::KnownWindowRowsBindingScope;
    KnownWindowRowsBindingCache cache;
    int plan = 0, other_plan = 0, slots = 0, other_slots = 0;
    int binds = 0, resource = 17;
    const auto bind = [&] { ++binds; return Bound{resource}; };
    const auto read = [&] {
        std::optional<Bound> local;
        return KnownWindowRowsBindingScope::get(local, bind).resource;
    };
    const auto run = [&](const void* key, const void* owner_slots, std::size_t step,
        std::size_t rows, std::size_t steps, std::array<std::uintptr_t, 2> signature) {
        KnownWindowRowsBindingScope scope(&cache, key, owner_slots, step, rows, steps, signature);
        return read();
    };
    {
        KnownWindowRowsBindingScope scope(&cache, &plan, &slots, 0, 8, 2, {1, 2});
        require(read() == 17, "first bind");
        // Preflight and launch have different binder lambdas, but the same bound type.
        std::optional<Bound> local;
        const auto& launch = KnownWindowRowsBindingScope::get(local, [&] { return bind(); });
        require(launch.resource == 17 && binds == 1, "preflight/launch must reuse binding");
        {
            KnownWindowRowsBindingScope uncached(nullptr, nullptr, nullptr, 0, 8, 2, {});
            read();
            read();
            require(binds == 3, "unmemoized callers must bind each time");
        }
        read();
        require(binds == 3, "nested scope must restore cache");
    }
    run(&plan, &slots, 0, 8, 2, {1, 2});
    require(binds == 3, "next window must reuse binding");
    run(&plan, &slots, 1, 8, 2, {1, 2});
    require(binds == 4, "each step owns its binding");
    run(&other_plan, &slots, 0, 8, 2, {1, 2});
    require(binds == 5, "new preflight_key must bind");
    run(&plan, &slots, 0, 8, 2, {1, 3});
    require(binds == 6, "changed config slot must bind");
    run(&plan, &slots, 0, 8, 2, {4, 3});
    require(binds == 7, "changed preflight function must bind");
    run(&plan, &slots, 0, 4, 2, {4, 3});
    require(binds == 8, "changed rows must bind");
    run(&plan, &slots, 0, 4, 3, {4, 3});
    require(binds == 9, "changed steps must bind");
    run(&plan, &other_slots, 0, 4, 3, {4, 3});
    require(binds == 10, "changed slot owner must bind");
    // Resource replacement follows the existing owner memo invalidation contract.
    resource = 23;
    cache.clear();
    require(run(&plan, &slots, 0, 8, 2, {1, 2}) == 23 && binds == 11,
        "owner reset must discard old resource addresses");
    {
        KnownWindowRowsBindingScope scope(&cache, &plan, &slots, 1, 8, 2, {1, 2});
        std::optional<Bound> local;
        bool failed = false;
        try {
            KnownWindowRowsBindingScope::get(local, []() -> Bound { throw std::runtime_error("bind failure"); });
        } catch (const std::runtime_error&) { failed = true; }
        require(failed, "bind failure must propagate");
        require(read() == 23 && binds == 12, "failed bind must not publish a value");
    }
    read();
    read();
    require(binds == 14, "scope exit must disable reuse");
    {
        KnownWindowRowsBindingScope scope(&cache, &plan, &slots, 0, 8, 2, {1, 2});
        std::optional<int> local;
        require(KnownWindowRowsBindingScope::get(local, [] { return 31; }) == 31,
            "a different bound type must replace the old value safely");
        require(read() == 23 && binds == 15, "returning to the original bound type must bind again");
    }
    KnownWindowRowsBindingCache other_cache;
    {
        KnownWindowRowsBindingScope scope(&other_cache, &plan, &slots, 0, 8, 2, {1, 2});
        require(read() == 23 && binds == 16, "owners must never share cached values");
    }
}
} // namespace

int main() {
    try { test_bindings(); }
    catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
    return 0;
}
