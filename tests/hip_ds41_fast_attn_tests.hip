// FAST lever ds41-attn-f16wmma-v1 vs the exact attention: error bounds on synthetic K/V and timing of both.
// Usage: strix-hip-ds41-fast-attn-tests [--no-timing] [--reps N]
#include "strix/hip_ds41_attention.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace a = strix::ds41::attn;
namespace g = strix::ds41::attn::hip;
namespace {
int failures = 0;
void expect(bool ok, const std::string& what) {
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL %s\n", what.c_str()); }
}
void check(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
template <class T> struct Dev {
    T* p = nullptr; std::size_t n = 0;
    explicit Dev(std::size_t count) : n(count) { check(hipMalloc(&p, std::max<std::size_t>(count, 1) * sizeof(T))); check(hipMemset(p, 0, std::max<std::size_t>(count, 1) * sizeof(T))); }
    explicit Dev(const std::vector<T>& h) : Dev(h.size()) { put(h); }
    ~Dev() { (void)hipFree(p); }
    Dev(const Dev&) = delete;
    void put(const std::vector<T>& h) { check(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice)); }
    std::vector<T> get() const { std::vector<T> h(n); check(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost)); return h; }
};
std::mt19937 rng(20261031);
float uni(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }
bool timing_on = true;
int reps = 20;
// median of reps timed launches of f (one warm-up first)
template <class F> float timing(F f) {
    f();
    check(hipDeviceSynchronize());
    if (!timing_on) return 0.0F;
    hipEvent_t b, e;
    check(hipEventCreate(&b)); check(hipEventCreate(&e));
    std::vector<float> t;
    for (int j = 0; j < reps; ++j) {
        check(hipEventRecord(b)); f(); check(hipEventRecord(e)); check(hipEventSynchronize(e));
        float ms; check(hipEventElapsedTime(&ms, b, e)); t.push_back(ms);
    }
    check(hipEventDestroy(b)); check(hipEventDestroy(e));
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}
struct Err { double rel_l2 = 0, max_abs = 0, max_ref = 0; std::size_t nonfinite = 0; };
Err compare(const std::vector<float>& ref, const std::vector<float>& got) {
    Err e; double num = 0, den = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i])) { ++e.nonfinite; continue; }
        const double d = double(got[i]) - ref[i];
        num += d * d; den += double(ref[i]) * ref[i];
        e.max_abs = std::max(e.max_abs, std::fabs(d)); e.max_ref = std::max(e.max_ref, std::fabs(double(ref[i])));
    }
    e.rel_l2 = std::sqrt(num / std::max(den, 1e-30));
    return e;
}
// FNV-1a over the raw bytes: a FAST output fingerprint for cross-build bitwise receipts.
template <class T>
unsigned long long bits_hash(const std::vector<T>& v) {
    unsigned long long h = 1469598103934665603ULL;
    const auto* b = reinterpret_cast<const unsigned char*>(v.data());
    for (std::size_t i = 0; i < v.size() * sizeof(T); ++i) h = (h ^ b[i]) * 1099511628211ULL;
    return h;
}
// Case options: also compare the A8 output (codes / deltas / sums of o); a NaN code in one window key (both kernels must
// put NaN in exactly the same outputs).
bool case_a8 = false, case_nan = false, case_big_key = false;
// One case: `rows` consecutive positions ending at `last`, `heads` query heads, `nc` selected compressed keys per row
// (0: window-only layer), q uniform in +-qmax (larger qmax = peakier softmax), window scale bytes from smin .. smax.
void run_case(const char* name, std::uint64_t last, unsigned rows, unsigned heads, unsigned nc, float qmax, unsigned smin, unsigned smax,
              bool tickets, bool checked = true) {
    const unsigned cap = a::kWindow + std::max(rows, a::kMaxRows), ncache = 8192U;
    std::vector<std::uint8_t> ring(std::size_t{cap} * a::kWindowEntryBytes), kv(std::size_t{ncache} * a::kKvEntryBytes);
    for (std::size_t s = 0; s < cap; ++s) {
        for (unsigned d = 0; d < 512; ++d) { std::uint8_t c; do c = static_cast<std::uint8_t>(rng()); while ((c & 0x7FU) == 0x7FU); ring[s * 528 + d] = c; }
        for (unsigned b = 0; b < 16; ++b) ring[s * 528 + 512 + b] = static_cast<std::uint8_t>(smin + rng() % (smax - smin + 1));
    }
    for (std::size_t t = 0; t < ncache; ++t) {
        for (unsigned i = 0; i < 256; ++i) kv[t * 288 + i] = static_cast<std::uint8_t>(rng());
        for (unsigned g2 = 0; g2 < 32; ++g2) kv[t * 288 + 256 + g2] = static_cast<std::uint8_t>(0x30 + rng() % 16);
    }
    std::vector<std::uint64_t> pos(rows);
    for (unsigned r = 0; r < rows; ++r) pos[r] = last - (rows - 1) + r;
    if (case_nan) ring[static_cast<std::size_t>((last - 3U) % cap) * 528 + 77] = 0x7F;   // a window key every row sees
    if (case_big_key) {   // one all-zero window key with scale byte 150 (shift 16) among ordinary keys (Sol r2 P1)
        const std::size_t at = static_cast<std::size_t>((last - 3U) % cap) * 528;
        std::fill(ring.begin() + static_cast<std::ptrdiff_t>(at), ring.begin() + static_cast<std::ptrdiff_t>(at + 512), std::uint8_t{0});
        std::fill(ring.begin() + static_cast<std::ptrdiff_t>(at + 512), ring.begin() + static_cast<std::ptrdiff_t>(at + 528), std::uint8_t{150});
    }
    std::vector<float> q(std::size_t{rows} * heads * 512), sink(heads);
    for (auto& x : q) x = uni(-qmax, qmax);
    for (auto& x : sink) x = uni(-2.0F, 2.0F);
    std::vector<std::uint32_t> idx(std::size_t{rows} * a::kTopK), cnt(rows, nc);
    for (unsigned r = 0; r < rows; ++r) {
        std::vector<std::uint32_t> pick(ncache);
        for (std::uint32_t i = 0; i < ncache; ++i) pick[i] = i;
        std::shuffle(pick.begin(), pick.end(), rng);
        std::sort(pick.begin(), pick.begin() + nc);
        std::copy(pick.begin(), pick.begin() + nc, &idx[std::size_t{r} * a::kTopK]);
    }
    std::vector<a::RopeRow> rope;
    for (const auto p : pos) rope.push_back(a::rope_row(a::RopeKind::compressed, p));
    Dev<std::uint8_t> dring(ring), dkv(kv);
    Dev<float> dq(q), dsink(sink), dpart(std::size_t{rows} * heads * a::kMaxChunks * 514), dout(std::size_t{rows} * heads * 512);
    Dev<std::uint64_t> dpos(pos);
    Dev<std::uint32_t> didx(idx), dcnt(cnt), dtick(std::size_t{rows} * g::kAttnGroupCount);
    Dev<a::RopeRow> drope(rope);
    g::AttentionView v{};
    v.rows = rows; v.capacity = cap; v.topk = a::kTopK; v.q = dq.p; v.sink = dsink.p; v.rope = drope.p; v.pos = dpos.p;
    v.ring = dring.p; v.kv_cache = dkv.p; v.idx = didx.p; v.idx_count = nc != 0U ? dcnt.p : nullptr; v.partial = dpart.p; v.o = dout.p;
    v.heads = heads; v.tickets = tickets ? dtick.p : nullptr;
    Dev<std::int8_t> dcodes(dout.n);
    Dev<float> ddeltas(dout.n / 32U);
    Dev<std::int32_t> dsums(dout.n / 32U);
    if (case_a8) { v.codes = dcodes.p; v.deltas = ddeltas.p; v.sums = dsums.p; }
    const float exact_ms = timing([&] { g::enqueue_attention(v, nullptr); });
    const auto ref = dout.get();
    const auto ref_codes = dcodes.get();
    check(hipMemset(dout.p, 0, dout.n * sizeof(float)));
    const float fast_ms = timing([&] { g::enqueue_attention_f16wmma_prefill(v, nullptr); });
    const auto got = dout.get();
    const Err e = compare(ref, got);
    // twice: the lever is deterministic (same bits on a repeat)
    g::enqueue_attention_f16wmma_prefill(v, nullptr);
    check(hipDeviceSynchronize());
    const auto again = dout.get();
    if (checked) {
        expect(std::memcmp(got.data(), again.data(), got.size() * sizeof(float)) == 0, std::string(name) + " repeat bits");
        if (case_nan) {
            std::size_t ref_nan = 0, same = 0;
            for (std::size_t i = 0; i < ref.size(); ++i) {
                ref_nan += std::isnan(ref[i]) ? 1U : 0U;
                same += std::isnan(ref[i]) == std::isnan(got[i]) ? 1U : 0U;
            }
            expect(ref_nan > 0U && same == ref.size(), std::string(name) + " NaN in the same outputs as the exact kernel");
        } else {
            expect(e.nonfinite == 0, std::string(name) + " finite");
            expect(e.rel_l2 < 2e-3, std::string(name) + " relative L2 < 2e-3");
        }
        if (case_a8) {
            const auto codes = dcodes.get();
            std::size_t far = 0, diff = 0;
            for (std::size_t i = 0; i < codes.size(); ++i) {
                diff += codes[i] != ref_codes[i] ? 1U : 0U;
                far += std::abs(int(codes[i]) - int(ref_codes[i])) > 1 ? 1U : 0U;
            }
            expect(far == 0U && diff * 10U < codes.size(), std::string(name) + " A8 codes within 1 step, < 10 % differ");
        }
    }
    if (tickets) {
        const auto tk = dtick.get();
        expect(std::all_of(tk.begin(), tk.end(), [](std::uint32_t t) { return t == 0U; }), std::string(name) + " tickets left zero");
    }
    std::printf("%-22s pos=%llu rows=%u heads=%u nc=%u qmax=%.2f scales=%u..%u tickets=%d exact_ms=%.4f fast_ms=%.4f speedup=%.2f rel_l2=%.3g max_abs=%.3g max_ref=%.3g fast_bits=%016llx codes_bits=%016llx\n",
                name, static_cast<unsigned long long>(last), rows, heads, nc, qmax, smin, smax, tickets ? 1 : 0, exact_ms, fast_ms,
                fast_ms > 0 ? exact_ms / fast_ms : 0.0, e.rel_l2, e.max_abs, e.max_ref, bits_hash(got),
                case_a8 ? bits_hash(dcodes.get()) : 0ULL);
    std::fflush(stdout);
}
// Hand-built window-only cases (rows 1, 32 heads, positions 0 .. last, every key in the window): the ring entries come
// from `fill(position, entry)`, q[0] = q0 in every head (other dims 0), sink 0; exact vs FAST.
template <class Fill>
void window_case(const char* name, std::uint64_t last, float q0, Fill fill) {
    const unsigned rows = 1, heads = 32, cap = a::kWindow + a::kMaxRows;
    std::vector<std::uint8_t> ring(std::size_t{cap} * a::kWindowEntryBytes, 0);
    for (std::uint64_t p = 0; p <= last; ++p) fill(p, ring.data() + (p % cap) * 528);
    std::vector<float> q(std::size_t{rows} * heads * 512, 0.0F), sink(heads, 0.0F);
    for (unsigned h = 0; h < heads; ++h) q[std::size_t{h} * 512] = q0;
    const std::vector<std::uint64_t> pos{last};
    std::vector<a::RopeRow> rope{a::rope_row(a::RopeKind::compressed, last)};
    Dev<std::uint8_t> dring(ring), dkv(std::size_t{a::kKvEntryBytes});
    Dev<float> dq(q), dsink(sink), dpart(std::size_t{rows} * heads * a::kMaxChunks * 514), dout(std::size_t{rows} * heads * 512);
    Dev<std::uint64_t> dpos(pos);
    Dev<std::uint32_t> didx(std::size_t{a::kTopK}), dtick(std::size_t{rows} * g::kAttnGroupCount);
    Dev<a::RopeRow> drope(rope);
    g::AttentionView v{};
    v.rows = rows; v.capacity = cap; v.topk = a::kTopK; v.q = dq.p; v.sink = dsink.p; v.rope = drope.p; v.pos = dpos.p;
    v.ring = dring.p; v.kv_cache = dkv.p; v.idx = didx.p; v.idx_count = nullptr; v.partial = dpart.p; v.o = dout.p;
    v.heads = heads; v.tickets = dtick.p;
    g::enqueue_attention(v, nullptr);
    check(hipDeviceSynchronize());
    const auto ref = dout.get();
    g::enqueue_attention_f16wmma_prefill(v, nullptr);
    check(hipDeviceSynchronize());
    const auto got = dout.get();
    const Err e = compare(ref, got);
    expect(std::isfinite(ref[0]) && std::fabs(ref[0]) > 0.0F && e.nonfinite == 0 && e.rel_l2 < 2e-3, std::string(name) + ": FAST == exact within 2e-3");
    std::printf("%-22s exact o[0]=%.7g fast o[0]=%.7g rel_l2=%.3g nonfinite=%zu fast_bits=%016llx\n", name, ref[0], got[0], e.rel_l2,
                e.nonfinite, bits_hash(got));
    std::fflush(stdout);
}
void sol_cases() {
    // Sol r3: 31 keys with dimension 0 = 57344 (E4M3 0x7e, scale byte 134, shift 0), one all-zero key with scale byte
    // 142 (shift 8), q[0] = -3/512: their tiny probabilities (~3.6e-7) times the large values make o[0] ~ 0.316
    window_case("sol-r3-band-boundary", 31, -3.0F / 512.0F, [](std::uint64_t p, std::uint8_t* e) {
        if (p != 5U) e[0] = 0x7E;
        for (unsigned b = 0; b < 16; ++b) e[512 + b] = p == 5U ? 142 : 134;
    });
    // Sol r4: tile 1 = 32 keys with dimension 0 = 2^110 (E4M3 0x38 = 1, scale byte 237), tile 2 = 32 zero keys with scale
    // byte 127, q = 0: the accumulator holds ~2^115 after tile 1 and must not be scaled up at tile 2 (exact ~6.39e32)
    window_case("sol-r4-acc-overflow", 63, 0.0F, [](std::uint64_t p, std::uint8_t* e) {
        if (p < 32U) e[0] = 0x38;
        for (unsigned b = 0; b < 16; ++b) e[512 + b] = p < 32U ? 237 : 127;
    });
}
}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--no-timing") timing_on = false;
        else if (std::string(argv[i]) == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
    }
    try {
        // window-only rows (early prompt), partial first chunk
        run_case("window-only-early", 40, 8, 32, 0, 0.5F, 118, 121, true);
        run_case("window-only-full", 2000, 8, 32, 0, 0.5F, 118, 121, true);
        // compressed + window, short selection and the full top-512 (8K / 32K prompts), TP rank (32 heads) and all 64
        run_case("sel-100", 3000, 8, 32, 100, 0.5F, 118, 121, true);
        run_case("sel-512-tp", 8000, 8, 32, 512, 0.5F, 118, 121, true);
        run_case("sel-512-all", 8000, 8, 64, 512, 0.5F, 118, 121, true);
        run_case("sel-512-peaky", 30000, 8, 32, 512, 2.0F, 118, 121, true);
        run_case("sel-512-combine-k", 30000, 8, 32, 512, 0.5F, 118, 121, false);
        // window scales beyond f16 range (ek shift) and far below it (f16 subnormal keys)
        run_case("win-scale-high", 30000, 8, 32, 512, 0.5F, 130, 140, true);
        run_case("win-scale-low", 30000, 8, 32, 512, 0.5F, 106, 112, true);
        // larger calls (slab windows)
        run_case("sel-512-rows32", 30000, 32, 32, 512, 0.5F, 118, 121, true);
        run_case("sel-512-rows64", 30000, 64, 32, 512, 0.5F, 118, 121, true);
        // window / selected boundary inside a key pair (odd window: position 100 sees 101 window keys), A8 output,
        // window scales up to 150 (key shift 16: the probabilities must stay finite), large queries (f16 range), NaN key
        run_case("mixed-odd-window", 100, 8, 32, 50, 0.5F, 118, 121, true);
        case_a8 = true;
        run_case("sel-512-a8", 8000, 8, 32, 512, 0.5F, 118, 121, true);
        case_a8 = false;
        run_case("win-scale-150", 30000, 8, 32, 512, 0.5F, 140, 150, true);
        run_case("q-large", 30000, 8, 32, 512, 100000.0F, 118, 121, true);
        run_case("q-huge", 30000, 8, 32, 512, 1.0e33F, 118, 121, true);   // above the old 3e38 guard in spirit: k > 100
        case_big_key = true;
        run_case("big-key-among-ordinary", 30000, 8, 32, 512, 2.0F, 118, 121, true);
        run_case("big-key-window-only", 2000, 8, 32, 0, 2.0F, 118, 121, true);
        case_big_key = false;
        sol_cases();
        case_nan = true;
        run_case("nan-window-key", 30000, 8, 32, 512, 0.5F, 118, 121, true);
        case_nan = false;
        // every chunks-per-block / heads-per-block form, checked (timed only with timing on)
        for (const unsigned cpb : {1U, 2U, 5U, 10U})
            for (const unsigned hpb : {16U, 32U}) {
                g::set_fast_attn_tuning(cpb, 0U, hpb);
                std::printf("-- chunks per block %u, heads per block %u\n", cpb, hpb);
                run_case("cfg-sel-512-tp", 8000, 8, 32, 512, 0.5F, 118, 121, true);
                run_case("cfg-sel-512-all", 8000, 8, 64, 512, 0.5F, 118, 121, false);
                run_case("cfg-mixed-odd-window", 100, 8, 32, 50, 0.5F, 118, 121, true);
            }
        g::set_fast_attn_tuning(2U, 0U, 16U);
        if (timing_on) {
            // chunks per block (each checked), then ablations (timing only: bit 0 key loads, 1 dequant, 2 QK, 3 PV, 4 Q)
            for (const unsigned cpb : {1U, 2U, 5U})
                for (const unsigned hpb : {16U, 32U}) {
                    g::set_fast_attn_tuning(cpb, 0U, hpb);
                    std::printf("-- chunks per block %u, heads per block %u\n", cpb, hpb);
                    run_case("cpb-sel-512-tp", 8000, 8, 32, 512, 0.5F, 118, 121, true);
                    run_case("cpb-sel-512-all", 8000, 8, 64, 512, 0.5F, 118, 121, true);
                    run_case("cpb-sel-512-rows64", 30000, 64, 32, 512, 0.5F, 118, 121, true);
                    run_case("cpb-window-only", 2000, 8, 32, 0, 0.5F, 118, 121, true);
                }
            for (const unsigned mask : {2U, 4U, 8U, 15U}) {
                g::set_fast_attn_tuning(2U, mask);
                std::printf("-- ablate %u\n", mask);
                run_case("abl-sel-512-rows64", 30000, 64, 32, 512, 0.5F, 118, 121, true, false);
            }
            g::set_fast_attn_tuning(2U, 0U);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
    std::printf("%s (%d failures)\n", failures == 0 ? "PASS" : "FAIL", failures);
    return failures == 0 ? 0 : 1;
}
