// strix::lever_cache: an ordinary process (no environment change signalled) keeps the first value, exactly like the
// `static const` it replaces; after lever_environment_changed() the next read resolves again; set() pins a value
// until the next change; a non-trivial value type takes the locked path.

#include "strix/lever_cache.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
#include <thread>
#include <future>
#include <chrono>

namespace {

int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++failures;
    }
}

int g_source = 7;     // what the "environment" says
int g_resolves = 0;   // resolver calls
int resolve_int() {
    ++g_resolves;
    return g_source;
}

std::vector<bool> resolve_table() {
    std::vector<bool> t(4, false);
    t[static_cast<std::size_t>(g_source & 3)] = true;
    return t;
}

std::atomic<int> entered{0};
std::atomic<bool> release_resolver{false};
int slow_resolve() {
    ++entered;
    while (!release_resolver.load()) std::this_thread::yield();
    return 7;
}

void concurrent_miss_and_pin() {
    using namespace std::chrono_literals;
    {
        strix::lever_cache<int> c{slow_resolve};
        std::vector<std::future<int>> reads;
        for (int i = 0; i < 8; ++i) reads.push_back(std::async(std::launch::async, [&] { return c.get(); }));
        while (entered.load() == 0) std::this_thread::yield();
        std::this_thread::sleep_for(50ms);
        release_resolver = true;
        for (auto& r : reads) check(r.get() == 7, "concurrent first read value");
        check(entered == 1, "concurrent first reads resolve exactly once");
    }
    entered = 0;
    release_resolver = false;
    strix::lever_cache<int> c{slow_resolve};
    auto read = std::async(std::launch::async, [&] { return c.get(); });
    while (entered.load() == 0) std::this_thread::yield();
    std::promise<void> started;
    auto pin = std::async(std::launch::async, [&] { started.set_value(); c.set(42); });
    started.get_future().wait();
    const bool serialized = pin.wait_for(50ms) == std::future_status::timeout;
    release_resolver = true;
    (void)read.get();
    pin.get();
    check(serialized, "pin publication waits for in-flight resolution");
    check(c.get() == 42 && entered == 1, "resolution cannot overwrite a completed pin");
}

}  // namespace

int main() {
    check(strix::lever_generation() == 0U, "generation starts at 0");
    {
        strix::lever_cache<int> c{resolve_int};
        check(g_resolves == 0, "lazy: nothing resolved before the first read");
        check(c() == 7, "first read resolves");
        g_source = 9;  // the environment changes, but nobody signals it: the old value stays (static const behaviour)
        check(c() == 7 && c.get() == 7, "unsignalled change keeps the old value");
        check(g_resolves == 1, "resolved exactly once without a signal");
        strix::lever_environment_changed();
        check(c() == 9, "signalled change resolves again");
        check(g_resolves == 2, "one resolve per generation");
        check(c() == 9 && g_resolves == 2, "second read of a generation is cached");
        c.set(3);
        check(c() == 3 && g_resolves == 2, "set pins without resolving");
        strix::lever_environment_changed();
        check(c() == 9 && g_resolves == 3, "next change drops the pin");
    }
    {
        g_source = 2;
        strix::lever_cache<std::vector<bool>> t{resolve_table};
        check(t()[2] && !t()[1], "vector value resolves");
        g_source = 1;
        check(t()[2], "vector unsignalled change keeps the old value");
        strix::lever_environment_changed();
        check(t()[1] && !t()[2], "vector re-resolves after a change");
        t.set(std::vector<bool>(4, true));
        check(t()[0] && t()[3], "vector set pins");
    }
    {
        bool raised = false;
        static bool first = true;
        strix::lever_cache<int> bad{[]() -> int {
            if (first) {
                first = false;
                throw std::runtime_error("bad value");
            }
            return 5;
        }};
        try {
            (void)bad();
        } catch (const std::exception&) {
            raised = true;
        }
        check(raised, "a throwing resolver propagates");
        check(bad() == 5, "a throwing resolver is retried on the next read (like a static initialiser)");
    }
    concurrent_miss_and_pin();
    if (failures == 0) std::printf("lever_cache: all checks passed\n");
    return failures == 0 ? 0 : 1;
}
