#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
using hipStream_t = void*;
constexpr int hipMemcpyHostToDevice = 0;
int attempts = 0, fail_at = 0, live = 0;
int hipMalloc(void** p, std::size_t bytes) {
    if (++attempts == fail_at) return 1;
    *p = std::malloc(bytes);
    if (*p == nullptr) return 1;
    ++live;
    return 0;
}
int hipFree(void* p) { if (p != nullptr) { --live; std::free(p); } return 0; }
int hipMemsetAsync(void* p, int value, std::size_t bytes, hipStream_t) { std::memset(p, value, bytes); return 0; }
int hipMemcpyAsync(void* dst, const void* src, std::size_t bytes, int, hipStream_t) { std::memcpy(dst, src, bytes); return 0; }
void check(int status, const char* message) { require(status == 0, message); }
#include "shared_dev.inc"

struct Scratch { Dev<float> sh_gate, sh_up; };
void allocate(Scratch& m) {
    constexpr unsigned kSharedFastRows = 2048U, I = 128U;
#include "shared_alloc.inc"
}
void scenario(bool growing) {
    Scratch m;
    if (growing) { m.sh_gate = Dev<float>(32); m.sh_up = Dev<float>(32); }
    auto* const old_gate = m.sh_gate.p;
    auto* const old_up = m.sh_up.p;
    const auto old_n = m.sh_gate.n;
    const int old_live = live;
    fail_at = attempts + 2;
    bool failed = false;
    try { allocate(m); } catch (const std::runtime_error&) { failed = true; }
    require(failed, "second allocation failure not exercised");
    const bool preserved = m.sh_gate.p == old_gate && m.sh_up.p == old_up && m.sh_gate.n == old_n &&
                           m.sh_up.n == old_n && live == old_live;
    fail_at = 0;
    allocate(m);
    require(m.sh_gate.p != nullptr && m.sh_up.p != nullptr && m.sh_gate.n == 2048U * 128U &&
            m.sh_up.n == m.sh_gate.n, "retry left incomplete shared scratch");
    require(preserved, "failed transaction changed scratch or leaked memory");
    const int allocated = attempts;
    allocate(m);
    require(attempts == allocated, "sufficient scratch allocated again");
}
int main() {
    try {
        scenario(false);
        require(live == 0, "initial scratch leaked");
        scenario(true);
        require(live == 0, "grown scratch leaked");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
