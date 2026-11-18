// pair::state: one copy per owner, prefix first. Pins every family's hand-off bytes to the MINSTATE tables
// (wave4 next1113, Tarea 1) and checks the planner's owner, relay, prefix and recompute rules and the shared resolver.
#include "strix/pair/state_decls.hpp"
#include "strix/pair/state_relay.hpp"

#include <cstdio>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

namespace {

int failures = 0;
#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                          \
        }                                                                        \
    } while (0)

namespace st = strix::pair::state;
namespace dc = strix::pair::state::decl;

st::Plan full(const st::Family& f, std::uint64_t rows, bool once, std::vector<std::uint32_t> stages = {}) {
    st::Options o;
    o.shared_once = once;
    o.stage_first = std::move(stages);
    return st::plan(f, rows, o, st::Links{});
}

void minimax_tables() {
    for (const std::uint64_t n : {32768ULL, 131072ULL}) {
        const std::uint64_t p = n - 1;
        const st::Plan today = full(dc::minimax, p, false), once = full(dc::minimax, p, true);
        CHECK(today.host_bytes == 153600ULL * p);
        CHECK(once.host_bytes == 138240ULL * p);
        CHECK(today.relay_bytes == 0);
        CHECK(once.relay_bytes == 60ULL * 256ULL * p);  // IK forwarded rank 0 -> rank 1
        CHECK(once.host_bytes_to[0] == 60ULL * 1280ULL * p);
        CHECK(once.host_bytes_to[1] == 60ULL * 1024ULL * p);
    }
    // 32K: 5,033.01 -> 4,529.71 MB (MINSTATE MiniMax table)
    CHECK(full(dc::minimax, 32767, false).host_bytes == 5033011200ULL);
    CHECK(full(dc::minimax, 32767, true).host_bytes == 4529710080ULL);
}

void glm_tables() {
    const std::uint64_t fg = 152633344ULL;
    for (const std::uint64_t n : {32768ULL, 131072ULL}) {
        const std::uint64_t p = n - 1;
        CHECK(full(dc::glm, p, false).host_bytes == 39936ULL * n - 49152ULL + fg);  // v1
        CHECK(full(dc::glm, p, true).host_bytes == 19968ULL * n - 24576ULL + fg);   // v2, DSA once
    }
    CHECK(full(dc::glm, 32767, false).host_bytes == 1461207040ULL);  // 1,461.21 MB
    CHECK(full(dc::glm, 32767, true).host_bytes == 806920192ULL);    // 806.92 MB
    CHECK(full(dc::glm, 131071, true).relay_bytes == 2617221120ULL);  // 2,617.22 MB: the v1 second copy, now on the pair link
}

// Capacity estimates only; serialized fixtures are checked in pair_state_serialized_tests.cpp.
void ds41_tables() {
    for (const std::uint64_t n : {32768ULL, 131072ULL}) {
        CHECK(full(dc::ds41, n, false).host_bytes == 2ULL * (894ULL * n + 2900992ULL));  // a copy per rank
        const st::Plan once = full(dc::ds41, n, true);                                  // the shipped relay
        CHECK(once.host_bytes == 894ULL * n + 2900992ULL);
        CHECK(once.host_bytes_to[1] == 0);
    }
    CHECK(full(dc::ds41, 32768, true).host_bytes == 32195584ULL);  // 32.20 MB
}

void qwen_tables() {
    const std::uint64_t fq = 116400144ULL;
    for (const std::uint64_t n : {32768ULL, 131072ULL}) {
        CHECK(full(dc::qwen_single, n, false).host_bytes == 29952ULL * n - 2304ULL + fq);
        // Pair stages cut at layer 24: every layer's state goes to its stage only, the position to both (+8 B).
        const st::Plan pair = full(dc::qwen_pair, n, true, {0, 24});
        CHECK(pair.host_bytes == 29952ULL * n - 2304ULL + fq + 8ULL);
        CHECK(pair.relay_bytes == 0);
        // Stage B holds QSA layers 27..47 (6 of 12) and 18 GDN layers.
        CHECK(pair.host_bytes_to[1] == 6ULL * 2304ULL * n + 18ULL * (3145728ULL + 81920ULL) + 8ULL);
    }
    CHECK(full(dc::qwen_single, 32768, false).host_bytes == 1097864976ULL);  // 1,097.86 MB
}

void prefix_agreement() {
    // MiniMax-shaped family that has the invariance receipt; block 64.
    st::Family f = dc::minimax;
    f.prefix_rows_invariant = true;
    std::vector<std::uint64_t> chain(100);
    for (std::size_t k = 0; k < chain.size(); ++k) chain[k] = 1000 + k;
    const std::uint64_t rows = 6400 - 1;
    // Both ranks hold 64 blocks with the right digests: boundary 4096.
    std::vector<st::Holding> h{{4096, chain[63], false}, {4096, chain[63], false}};
    st::Agreement a = st::agree(f, rows, h, chain);
    CHECK(a.boundary == 4096);
    CHECK(!a.fixed_at);
    // One rank holds less: the minimum.
    h[1] = {1024, chain[15], false};
    CHECK(st::agree(f, rows, h, chain).boundary == 1024);
    // Wrong digest on one rank: no reuse.
    h[1] = {4096, 7, false};
    CHECK(st::agree(f, rows, h, chain).boundary == 0);
    // Not block-aligned: no reuse.
    h[1] = {4000, chain[62], false};
    CHECK(st::agree(f, rows, h, chain).boundary == 0);
    // A rank did not answer: no reuse.
    CHECK(st::agree(f, rows, std::span<const st::Holding>(h.data(), 1), chain).boundary == 0);
    // Without the receipt: never.
    h[1] = h[0];
    CHECK(st::agree(dc::minimax, rows, h, chain).boundary == 0);
    // Holding more rows than the prompt ships is clipped to whole blocks of the prompt.
    std::vector<st::Holding> big{{6400, chain[99], true}, {6400, chain[99], true}};
    CHECK(st::agree(f, rows, big, chain).boundary == 6336);

    // Suffix plan: rows [4096, P) only; nothing fixed in MiniMax.
    h = {{4096, chain[63], false}, {4096, chain[63], false}};
    a = st::agree(f, rows, h, chain);
    st::Options o;
    o.shared_once = true;
    o.prefix_first = true;
    const st::Plan s = st::plan(f, rows, o, st::Links{}, a);
    CHECK(s.mode == st::Mode::suffix);
    CHECK(s.host_bytes == 138240ULL * (rows - 4096));
    // Recompute is never chosen with an unmeasured rate, nor without the boundary's fixed state.
    st::Links fast;
    fast.recompute_rows_per_s = 1e9;
    CHECK(st::plan(f, rows, o, fast, a).mode == st::Mode::suffix);
    a.fixed_at = true;
    const st::Plan ids = st::plan(f, rows, o, fast, a);
    CHECK(ids.mode == st::Mode::ids);
    CHECK(ids.host_bytes == 4ULL * (rows - 4096) * 2ULL);
    st::Links slow;
    slow.recompute_rows_per_s = 10;  // 230 s of recompute loses to a 5 s send
    CHECK(st::plan(f, rows, o, slow, a).mode == st::Mode::suffix);
}

void fixed_state_survives_prefix() {
    // GLM: a prefix hit saves DSA rows only; the KDA final state (152.6 MB) always ships.
    st::Family g = dc::glm;
    g.prefix_rows_invariant = true;
    const std::uint64_t p = 32767;
    const std::uint64_t all_rows = st::host_bytes(g, p, 0, true);
    const std::uint64_t hit = st::host_bytes(g, p, p - 63, true);
    CHECK(all_rows == 806920192ULL);
    CHECK(hit >= 152633344ULL);
    CHECK(hit < 152633344ULL + 12ULL * 2048ULL * 64ULL);
}

void resolver() {
    // CLI wins over environment wins over family default; every switch recorded.
    st::Resolved r = st::resolve(dc::glm, -1, -1, nullptr, nullptr, false);
    CHECK(r.shared_once);  // GLM default: DSA once
    CHECK(!r.prefix_first);
    r = st::resolve(dc::minimax, -1, -1, nullptr, nullptr, false);
    CHECK(!r.shared_once);  // MiniMax: opt-in until its pair receipt
    r = st::resolve(dc::minimax, -1, -1, "1", nullptr, false);
    CHECK(r.shared_once);
    r = st::resolve(dc::glm, 0, -1, "1", nullptr, false);
    CHECK(!r.shared_once);
    r = st::resolve(dc::glm, -1, 1, nullptr, "0", false);
    CHECK(r.prefix_first);
    bool threw = false;
    try {
        (void)st::resolve(dc::glm, -1, -1, "yes", nullptr, false);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    CHECK(threw);
}

// Two ranks on threads over a symmetric in-memory exchange (one message each way per seq); `flip` corrupts one byte of
// the source's message with that seq.
struct FakeLink {
    std::mutex m;
    std::condition_variable cv;
    std::map<std::uint64_t, std::vector<std::byte>> box[2];
    std::uint64_t flip = ~0ULL;
    void xchg(unsigned r, std::span<const std::byte> local, std::span<std::byte> remote, std::uint64_t seq) {
        std::unique_lock<std::mutex> lk(m);
        box[r][seq].assign(local.begin(), local.end());
        if (r == 0 && seq == flip && !box[r][seq].empty()) box[r][seq][0] ^= std::byte{1};
        cv.notify_all();
        cv.wait(lk, [&] { return box[1 - r].count(seq) != 0; });
        const auto& in = box[1 - r][seq];
        if (!remote.empty()) std::memcpy(remote.data(), in.data(), std::min(remote.size(), in.size()));
    }
};
std::uint64_t fnv(const std::uint8_t* p, std::size_t n) {
    std::uint64_t h = 1469598103934665603ULL;
    for (std::size_t i = 0; i < n; ++i) h = (h ^ p[i]) * 1099511628211ULL;
    return h;
}
// Returns {rank 0 error, rank 1 error}; `dst` gets what rank 1 received.
std::pair<std::string, std::string> run_relay(FakeLink& link, std::vector<std::uint8_t> src, std::vector<std::uint8_t>& dst,
                                              std::size_t chunk, std::uint64_t* next = nullptr) {
    std::string e0, e1;
    std::thread t0([&] {
        try {
            const std::uint64_t n = st::relay_piece(true, true, std::span(src), chunk,
                [&](auto l, auto r, std::uint64_t q) { link.xchg(0, l, r, q); }, fnv, 10);
            if (next) *next = n;
        } catch (const std::exception& e) { e0 = e.what(); }
    });
    std::thread t1([&] {
        try {
            (void)st::relay_piece(false, true, std::span(dst), chunk, [&](auto l, auto r, std::uint64_t q) { link.xchg(1, l, r, q); },
                                  fnv, 10);
        } catch (const std::exception& e) { e1 = e.what(); }
    });
    t0.join();
    t1.join();
    return {e0, e1};
}

void replay_mode() {
    st::SharedOnceMode mode;
    const auto enabled = [&](unsigned repeat) {
        const bool once = st::resolve(dc::minimax, mode.cli, -1, "1", nullptr, false).shared_once;
        return mode.ab ? repeat % 2 == 1 : once;
    };
    mode.set(true, true);  // --ik-once ab
    CHECK(!enabled(0) && enabled(1));
    mode.set(false);       // --no-ik-once overrides AB and environment ON
    CHECK(!mode.ab && !enabled(0) && !enabled(1));
    mode.set(true, true);
    mode.set(true);        // a later plain --ik-once also cancels AB
    CHECK(!mode.ab && enabled(0) && enabled(1));
    mode.set(false);
    mode.set(true, true);  // last option may re-enable AB
    CHECK(!enabled(0) && enabled(1));
}

void relay_identity() {
    // Same-sized K/V and IK, but opposite layer order: refuse before either layer is imported.
    // Also bind the session, prompt, row interval and terminal frame, not just the ordinal exchange.
    for (unsigned field = 0; field <= 7; ++field) {
        FakeLink link;
        std::string errors[2];
        unsigned imported[2] = {};
        std::thread peers[2];
        for (unsigned r = 0; r < 2; ++r) peers[r] = std::thread([&, r] {
            try {
                for (std::uint64_t layer = 0; layer < 2; ++layer) {
                    st::RelayIdentity id{7, 99, layer, 0, 128, 123};
                    if (r == 1 && field < 6) {
                        if (field == 2) id[2] = 1 - layer;
                        else ++id[field];
                    }
                    if (r == 1 && field == 7) id[2] = 0xffffffffU;  // early end marker
                    st::relay_agree(id, [&](auto a, auto b, auto q) { link.xchg(r, a, b, q); }, layer);
                    ++imported[r];
                }
            } catch (const std::exception& e) { errors[r] = e.what(); }
        });
        for (auto& peer : peers) peer.join();
        if (field == 6) CHECK(errors[0].empty() && errors[1].empty() && imported[0] == 2 && imported[1] == 2);
        else CHECK(!errors[0].empty() && !errors[1].empty() && imported[0] == 0 && imported[1] == 0);
    }
}

void relay() {
    std::vector<std::uint8_t> src(5U << 20);
    for (std::size_t i = 0; i < src.size(); ++i) src[i] = static_cast<std::uint8_t>((i * 2654435761U) >> 13);
    {
        FakeLink link;
        std::vector<std::uint8_t> dst(src.size());
        std::uint64_t next = 0;
        const auto [e0, e1] = run_relay(link, src, dst, 1U << 20, &next);
        CHECK(e0.empty() && e1.empty());
        CHECK(dst == src);
        CHECK(next == 10 + 2 + 5);  // header + five 1 MiB chunks + verification vote
    }
    {
        FakeLink link;  // a corrupted chunk fails both ranks before either returns
        link.flip = 13;
        std::vector<std::uint8_t> dst(src.size());
        const auto [e0, e1] = run_relay(link, src, dst, 1U << 20);
        CHECK(e0.find("hash mismatch") != std::string::npos);
        CHECK(e1.find("hash mismatch") != std::string::npos);
    }
    {
        FakeLink link;  // a size disagreement fails both ranks before any payload moves
        std::vector<std::uint8_t> dst(src.size() - 1);
        const auto [e0, e1] = run_relay(link, src, dst, 1U << 20);
        CHECK(e0.find("piece size") != std::string::npos);
        CHECK(e1.find("piece size") != std::string::npos);
    }
}

}  // namespace

int main() {
    minimax_tables();
    glm_tables();
    ds41_tables();
    qwen_tables();
    prefix_agreement();
    fixed_state_survives_prefix();
    resolver();
    replay_mode();
    relay_identity();
    relay();
    if (failures == 0) std::printf("pair state owners: all checks passed\n");
    return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
