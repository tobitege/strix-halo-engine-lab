// Shared MTP chain glue (strix/spec/mtp_chain.hpp) against a fake engine whose "device" is host memory: checks the lazy
// catch-up bookkeeping (position / successor token / hidden row pairing of every mtp_forward call, chunking, carry hidden of
// chain steps, accept() after a partial verify) without any GPU.
#include "strix/spec/mtp_chain.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace sp = strix::spec;

#define MC_CHECK(c, ...) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); std::exit(1); } } while (0)

namespace {
constexpr unsigned kH = 4;
struct FakeRows {
    const std::uint32_t* top = nullptr;
    const float* hidden_post = nullptr;
    const float* hidden_pre = nullptr;
};
struct FakeMtpOut {
    const std::uint32_t* top = nullptr;
    const float* hidden_post = nullptr;
    const float* hidden_pre = nullptr;
};
struct Call {
    std::vector<std::uint32_t> tokens;
    std::vector<float> hidden0;  // first element of each hidden row
    unsigned pos0 = 0;
    bool want_top = false;
};
struct Cfg { unsigned hidden = kH; };
struct FakeEngine {
    using Rows = FakeRows;
    using MtpOut = FakeMtpOut;
    Cfg cfg;
    std::vector<Call> calls;
    std::vector<float> out_h;       // mtp hidden_post rows of the last call: row r first element = 1000 + call index * 10 + r
    std::vector<std::uint32_t> out_top;
    const Cfg& config() const { return cfg; }
    FakeMtpOut mtp_forward(const std::uint32_t* t, const float* hidden, unsigned rows, unsigned pos0, bool want_top) {
        Call c;
        c.tokens.assign(t, t + rows);
        for (unsigned r = 0; r < rows; ++r) c.hidden0.push_back(hidden[r * kH]);
        c.pos0 = pos0;
        c.want_top = want_top;
        calls.push_back(c);
        out_h.assign(rows * kH, 0.0F);
        out_top.assign(rows, 0);
        for (unsigned r = 0; r < rows; ++r) {
            out_h[r * kH] = 1000.0F + float(calls.size()) * 10.0F + float(r);
            out_top[r] = 500U + unsigned(calls.size()) * 10U + r;  // prediction of the call's row r
        }
        return {out_top.data(), out_h.data(), out_h.data()};
    }
    void read_top(const FakeRows& o, unsigned rows, std::uint32_t* host) const { std::memcpy(host, o.top, rows * 4U); }
};
struct HostDev {
    static float* alloc(FakeEngine&, std::size_t n) { return new float[n](); }
    static void release(FakeEngine&, float* p) { delete[] p; }
    static void copy(FakeEngine&, float* d, const float* s, std::size_t n) { std::memcpy(d, s, n * sizeof(float)); }
};
}  // namespace

int main() {
    FakeEngine e;
    sp::MtpChain<FakeEngine, HostDev> chain(e, false, 3, 8);  // chunk 3 rows, ring 8
    // prefill: positions 0..4 with hidden first-element 10*pos, tokens 11..15, anchor 99 follows position 4
    const std::uint32_t prompt[5] = {11, 12, 13, 14, 15};
    std::vector<float> hid(5 * kH, 0.0F);
    for (unsigned i = 0; i < 5; ++i) hid[i * kH] = 10.0F * float(i);
    FakeRows pf;
    pf.hidden_post = pf.hidden_pre = hid.data();
    chain.observe_prefill(pf, 5, prompt, 99);
    MC_CHECK(e.calls.empty(), "lazy: nothing before the first draft");

    std::vector<std::uint32_t> out;
    chain.draft({}, 2, out);
    // flush: successors 12,13,14 | 15,99 at positions 0..4; the last chunk wants the top
    MC_CHECK(e.calls.size() == 2 + 1, "flush chunks 3+2 then one chain step: got %zu calls", e.calls.size());
    MC_CHECK(e.calls[0].pos0 == 0 && e.calls[0].tokens == std::vector<std::uint32_t>({12, 13, 14}) && !e.calls[0].want_top, "chunk 0");
    MC_CHECK(e.calls[0].hidden0 == std::vector<float>({0.0F, 10.0F, 20.0F}), "chunk 0 hidden rows");
    MC_CHECK(e.calls[1].pos0 == 3 && e.calls[1].tokens == std::vector<std::uint32_t>({15, 99}) && e.calls[1].want_top, "chunk 1");
    MC_CHECK(e.calls[1].hidden0 == std::vector<float>({30.0F, 40.0F}), "chunk 1 hidden rows");
    // d1 = prediction of the last flushed row (call 2, row 1) = 500 + 20 + 1; step feeds d1 at position 5 with carry = that row's hidden
    MC_CHECK(out.size() == 2 && out[0] == 521U, "first draft token is the last flushed row's prediction (%u)", out.empty() ? 0U : out[0]);
    MC_CHECK(e.calls[2].pos0 == 5 && e.calls[2].tokens == std::vector<std::uint32_t>({521}) && e.calls[2].want_top, "chain step");
    MC_CHECK(e.calls[2].hidden0[0] == 1000.0F + 20.0F + 1.0F, "carry = last flushed row hidden (%g)", double(e.calls[2].hidden0[0]));
    MC_CHECK(out[1] == 530U, "second draft = chain step prediction (%u)", out[1]);

    // verify of [anchor 99, 521, 530] accepted 2 rows: emitted = {521, 77}; the next draft catches up positions 5..6 (not 7)
    std::vector<float> vh(3 * kH, 0.0F);
    vh[0] = 100.0F; vh[kH] = 110.0F; vh[2 * kH] = 120.0F;
    FakeRows vr;
    vr.hidden_post = vr.hidden_pre = vh.data();
    chain.observe_verify(vr, 3);
    const std::uint32_t emitted[2] = {521, 77};
    chain.accept(emitted);
    const auto before = e.calls.size();
    chain.draft({}, 1, out);
    MC_CHECK(e.calls.size() == before + 1 && out.size() == 1, "one catch-up call, no chain step for k = 1");
    MC_CHECK(e.calls[before].pos0 == 5 && e.calls[before].tokens == std::vector<std::uint32_t>({521, 77}), "catch-up overwrites the speculative position");
    MC_CHECK(e.calls[before].hidden0 == std::vector<float>({100.0F, 110.0F}), "catch-up hidden rows are the verify rows");
    // prefix (copy tokens) is consumed by the chain: only the last step's prediction is used
    chain.draft(std::vector<std::uint32_t>({8, 9}), 1, out);
    const auto n = e.calls.size();
    MC_CHECK(e.calls[n - 2].pos0 == 7 && e.calls[n - 2].tokens[0] == 8 && !e.calls[n - 2].want_top, "prefix step 1 (no top)");
    MC_CHECK(e.calls[n - 1].pos0 == 8 && e.calls[n - 1].tokens[0] == 9 && e.calls[n - 1].want_top, "prefix step 2 (top)");
    MC_CHECK(out.size() == 1 && out[0] == 500U + unsigned(n) * 10U, "draft after a prefix = last prefix step prediction");
    // ring overflow flushes in order
    chain.reset();
    std::vector<float> big(20 * kH, 1.0F);
    std::vector<std::uint32_t> toks(20, 3);
    FakeRows br;
    br.hidden_post = br.hidden_pre = big.data();
    e.calls.clear();
    chain.observe_prefill(br, 20, toks.data(), 4);
    MC_CHECK(!e.calls.empty() && e.calls[0].pos0 == 0, "ring full forces an in-order flush from position 0");
    std::puts("spec_mtp_chain_tests: PASS");
    return 0;
}
