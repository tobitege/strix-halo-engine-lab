// Block drafting (strix/spec/block_drafter.hpp) against a deterministic fake target and a fake chain drafter with a
// controlled accuracy: whatever the sources / windows / accuracy, the emitted ids equal serial greedy decode and the
// target's committed length equals prompt + emitted - 1 (the anchor is never committed). Costs are a virtual clock.
#include "strix/spec/block_drafter.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace sp = strix::spec;

namespace {
#define BD_CHECK(c, ...) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); std::exit(1); } } while (0)

std::uint64_t mix(std::uint64_t x) {
    x ^= x >> 33; x *= 0xff51afd7ed558ccdULL; x ^= x >> 33; x *= 0xc4ceb9fe1a85ec53ULL; x ^= x >> 33;
    return x;
}

double g_clock = 0.0;  // virtual ms
double g_chain_step = 1.5;  // virtual ms per chain step

// The "model": response[i] is the greedy token at response position i (a pure function of the position: a valid
// deterministic target). Prompt = `prompt` tokens, which the response re-emits with sparse edits (code-edit class).
struct World {
    std::vector<std::uint32_t> prompt, response;
};

World make_world(const std::string& cls, std::uint32_t prompt_len, std::uint32_t n, std::uint64_t seed) {
    World w;
    for (std::uint32_t i = 0; i < prompt_len; ++i) w.prompt.push_back(std::uint32_t(mix(seed + i) % 900) + 10);
    for (std::uint32_t i = 0; i < n; ++i) {
        std::uint32_t t;
        if (cls == "edit") {  // copy of the prompt, an edit every ~23 tokens, wraps around
            t = w.prompt[i % prompt_len];
            if (mix(seed * 7 + i) % 23 == 0) t = std::uint32_t(mix(seed * 13 + i) % 900) + 10;
        } else if (cls == "loop") {  // short period
            t = w.prompt[i % 12];
        } else {  // prose: no structure
            t = std::uint32_t(mix(seed * 31 + i) % 900) + 10;
        }
        w.response.push_back(t);
    }
    return w;
}

struct FakeTarget final : sp::VerifyTarget {
    const World& w;
    std::uint32_t committed_rows = 0;  // positions in the target (anchor excluded)
    std::vector<std::uint32_t> last;
    std::uint32_t max_rows_seen = 0;
    std::uint64_t verifies = 0;
    explicit FakeTarget(const World& world) : w(world) {}
    // row i sits at stream position prompt + committed_rows + i; its logits predict response[committed_rows + i]
    void verify(std::span<const std::uint32_t> tokens, std::vector<std::uint32_t>& greedy) override {
        BD_CHECK(tokens.size() >= 1 && tokens.size() <= 32, "verify rows %zu", tokens.size());
        max_rows_seen = std::max<std::uint32_t>(max_rows_seen, std::uint32_t(tokens.size()));
        ++verifies;
        greedy.clear();
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            const auto pos = committed_rows + i;
            greedy.push_back(pos + 1 < w.response.size() ? w.response[pos + 1] : 7);
        }
        last.assign(tokens.begin(), tokens.end());
        g_clock += 10.0 + 0.15 * double(tokens.size());  // memory-bound verify: nearly flat in rows
    }
    void commit(std::uint32_t k) override {
        BD_CHECK(k >= 1 && k <= last.size(), "commit %u of %zu", k, last.size());
        committed_rows += k;
        g_clock += 0.2;
    }
};

struct FakeChain final : sp::ChainDrafter {
    const World& w;
    double accuracy;
    std::uint32_t stream_len = 0;  // response tokens emitted so far (the anchor is the last of them)
    std::uint64_t calls = 0;
    FakeChain(const World& world, double acc) : w(world), accuracy(acc) {}
    void draft(std::span<const std::uint32_t> prefix, std::uint32_t k, std::vector<std::uint32_t>& out) override {
        ++calls;
        out.clear();
        for (std::uint32_t j = 0; j < k; ++j) {
            const std::uint64_t pos = std::uint64_t(stream_len) + prefix.size() + j;  // response position predicted
            const bool right = double(mix(pos * 977 + 5) % 1000) / 1000.0 < accuracy;
            out.push_back(pos < w.response.size() && right ? w.response[pos] : 5);
        }
        g_clock += g_chain_step * double(prefix.size() + k);
    }
    void accept(std::span<const std::uint32_t> emitted) override { stream_len += std::uint32_t(emitted.size()); }
};

struct Run {
    std::vector<std::uint32_t> out;
    std::uint64_t steps = 0, hybrid = 0, chain_blocks = 0, copy_blocks = 0, serial_steps = 0;
    std::uint32_t max_rows = 0;
};

Run run(const World& w, std::uint32_t n_tokens, double accuracy, bool copy, bool chain, bool hybrid, bool seed_costs) {
    sp::BlockConfig cfg;
    cfg.copy = copy; cfg.chain = chain; cfg.hybrid = hybrid;
    cfg.clock_ms = [] { return g_clock; };
    FakeChain fc(w, accuracy);
    sp::BlockDrafter drafter(cfg, chain || hybrid ? &fc : nullptr);
    if (seed_costs)
        for (const std::uint32_t r : {1U, 2U, 4U, 8U, 16U, 32U}) drafter.seed_verify_ms(r, 10.0 + 0.15 * double(r));
    FakeTarget target(w);
    // prefill: the first token (greedy after the prompt) is the anchor; the stream = prompt + anchor
    drafter.commit(w.prompt);
    Run r;
    r.out.push_back(w.response[0]);
    drafter.commit(std::span<const std::uint32_t>(&r.out[0], 1));
    fc.stream_len = 1;
    sp::BlockSpeculator spec(drafter, target);
    spec.start(r.out[0]);
    target.committed_rows = 0;
    while (r.out.size() < n_tokens) {
        sp::StepParams p;
        p.remaining = n_tokens - std::uint32_t(r.out.size());
        p.context = w.prompt.size() + r.out.size();
        const auto before = r.out.size();
        const auto rec = spec.step(r.out, p);
        BD_CHECK(rec.emitted >= 1 && rec.emitted <= p.remaining, "emitted %u remaining %u", rec.emitted, p.remaining);
        BD_CHECK(r.out.size() == before + rec.emitted, "out size");
        ++r.steps;
        r.max_rows = std::max(r.max_rows, rec.rows);
        r.hybrid += rec.copy > 0 && rec.ext > 0;
        r.chain_blocks += rec.source == sp::Source::model;
        r.copy_blocks += rec.source == sp::Source::lookup;
        r.serial_steps += rec.source == sp::Source::serial;
        // bookkeeping: committed target rows = emitted so far - 1 (the anchor is not in the target)
        BD_CHECK(target.committed_rows + 1 == r.out.size(), "committed %u vs emitted %zu", target.committed_rows, r.out.size());
    }
    return r;
}
}  // namespace

int main() {
    const std::uint32_t n = 1500;
    int cases = 0;
    std::uint64_t hybrids = 0, chains = 0, copies = 0;
    double tok_ms_serial = 0.0, tok_ms_all_loop = 0.0, tok_ms_all_prose = 0.0;
    for (const std::string cls : {"edit", "loop", "prose"})
        for (const std::uint32_t plen : {12U, 40U, 700U}) {
            const auto w = make_world(cls, plen, n, 100 + plen);
            for (const double acc : {0.0, 0.55, 0.9})
                for (int mode = 0; mode < 5; ++mode) {  // serial, copy, chain, copy+chain, all
                    const bool copy = mode == 1 || mode >= 3, chain = mode == 2 || mode >= 3, hybrid = mode == 4;
                    for (const bool seed : {false, true}) {
                        g_clock = 0.0;
                        const auto r = run(w, n, acc, copy, chain, hybrid, seed);
                        for (std::uint32_t i = 0; i < n; ++i)
                            BD_CHECK(r.out[i] == w.response[i], "%s plen %u acc %.2f mode %d: token %u differs", cls.c_str(), plen, acc, mode, i);
                        BD_CHECK(r.max_rows <= 32, "rows");
                        if (mode == 0) BD_CHECK(r.steps == n - 1 && r.serial_steps == n - 1, "serial mode must not draft");
                        if (mode == 1) BD_CHECK(r.chain_blocks == 0 && r.hybrid == 0, "copy only");
                        if (mode == 2) BD_CHECK(r.copy_blocks == 0, "chain only");
                        ++cases;
                        hybrids += r.hybrid; chains += r.chain_blocks; copies += r.copy_blocks;
                        if (plen == 700 && acc == 0.9 && seed) {
                            if (cls == "edit" && mode == 0) tok_ms_serial = g_clock / n;
                            if (cls == "edit" && mode == 4) tok_ms_all_loop = g_clock / n;
                            if (cls == "prose" && mode == 4) tok_ms_all_prose = g_clock / n;
                        }
                    }
                }
        }
    {  // a cheap chain makes the hybrid extension pay: short copy spans (period 12) extended by chain tokens
        g_chain_step = 0.02;
        for (const std::string cls : {"edit", "loop"}) {
            const auto w = make_world(cls, 12, n, 77);
            g_clock = 0.0;
            const auto r = run(w, n, 0.9, true, true, true, true);
            for (std::uint32_t i = 0; i < n; ++i) BD_CHECK(r.out[i] == w.response[i], "cheap chain %s: token %u differs", cls.c_str(), i);
            hybrids += r.hybrid;
        }
        g_chain_step = 1.5;
    }
    BD_CHECK(copies > 100 && chains > 100, "sources never used: copy %llu chain %llu", (unsigned long long)copies, (unsigned long long)chains);
    BD_CHECK(hybrids > 0, "hybrid never chosen");
    // The mixed drafter must pay on structure (>= 3x fewer ms per token than serial on the edit class)...
    BD_CHECK(tok_ms_all_loop * 3.0 < tok_ms_serial, "edit class ms/token %.2f vs serial %.2f", tok_ms_all_loop, tok_ms_serial);
    // ...and must not lose to serial on structureless text with an accurate chain (acc 0.9) nor by more than 10 % at acc 0.
    BD_CHECK(tok_ms_all_prose < tok_ms_serial, "prose class ms/token %.2f vs serial %.2f", tok_ms_all_prose, tok_ms_serial);
    {  // inaccurate chain on prose: the arbiter must bar it (probe cost only)
        const auto w = make_world("prose", 700, n, 3);
        g_clock = 0.0;
        const auto r = run(w, n, 0.0, true, true, true, true);
        BD_CHECK(g_clock / n < tok_ms_serial * 1.10, "useless chain costs %.2f ms/token vs serial %.2f", g_clock / n, tok_ms_serial);
        (void)r;
    }
    {  // a chain at ~66 % per-token acceptance on prose must keep drafting: misses in a row are routine, not a retreat
        const auto w = make_world("prose", 700, n, 5);
        g_clock = 0.0;
        const auto r = run(w, n, 0.66, true, true, true, true);
        BD_CHECK(double(r.serial_steps) < 0.15 * double(r.steps), "66 %% chain: %llu of %llu passes serial", (unsigned long long)r.serial_steps, (unsigned long long)r.steps);
    }
    std::printf("block drafter tests ok: %d cases, copy blocks %llu, chain blocks %llu, hybrid blocks %llu; edit %.2f vs serial %.2f ms/token\n",
                cases, (unsigned long long)copies, (unsigned long long)chains, (unsigned long long)hybrids, tok_ms_all_loop, tok_ms_serial);
    return 0;
}
