// GPU test of tiered expert residency (src/cuda/tiered_experts.cu): the routed MoE of a layer on the card with every expert
// resident (qwen_moe::enqueue, today's route) against the tiered route (publish -> routed kernels on slot routes -> the
// host executor through TierWorker -> gather -> combine) with only some experts resident. Random weights for two layers
// (Q4_K/Q5_1 and Q5_K/Q8_0), rows 1, 8 and 5 (padded to 16): the raw dots and the routed output of every live row must be
// bitwise equal; a non-finite expert on the host side sets the same row status bits and job summary as on the card; a CUDA
// graph of the tiered sequence replayed across a table change stays bitwise equal.
//   usage: qwen_tiered_moe_tests <tables dir>      (needs the card; never loads a model)
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "strix/cuda/qwen_moe.hpp"
#include "strix/cuda/tiered_experts.hpp"
#include "strix/cuda/amd_math_tables.hpp"

namespace mo = strix::cuda::qwen_moe;
namespace am = strix::cuda::amdmath;
namespace ti = strix::cuda::tiered;
namespace cx = strix::cpu::expert;
using u8 = std::uint8_t; using u16 = std::uint16_t; using u32 = std::uint32_t; using i8 = std::int8_t; using i32 = std::int32_t;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(2); } } while (0)
static int g_fail = 0;
static unsigned long long g_checked = 0;
static void expect(bool ok, const std::string& what) {
    if (!ok) { ++g_fail; std::printf("FAIL %s\n", what.c_str()); }
}

constexpr u32 kHidden = mo::hidden, kFfn = mo::ffn, kTop = mo::top_k, kAlloc = 48;  // experts with weights (ids 0..47)
static u32 b_of(float f) { u32 b; std::memcpy(&b, &f, 4); return b; }
static u16 half_of(float f) {
    const u32 b = b_of(f), s = (b >> 31) << 15;
    const float a = std::fabs(f);
    if (a == 0.0F) return static_cast<u16>(s);
    if (a < 0x1p-14F) return static_cast<u16>(s | static_cast<u32>(std::nearbyint(a * 0x1p24F)));
    int e; const float m = std::frexp(a, &e);
    u32 mant = static_cast<u32>(std::nearbyint((m * 2.0F - 1.0F) * 1024.0F));
    int he = e - 1 + 15;
    if (mant == 1024) { mant = 0; ++he; }
    return static_cast<u16>(s | (static_cast<u32>(he) << 10) | mant);
}
static u16 bf16_of(float f) { const u32 b = b_of(f); return static_cast<u16>((b + 0x7fffU + ((b >> 16) & 1U)) >> 16); }

struct LayerData {
    u32 gate_type, down_type;
    std::vector<u8> gate, up, down;
    u32 gb() const { return gate_type == mo::type_q5_k ? 176 : 144; }
    u32 db() const { return down_type == mo::type_q5_1 ? 24 : 34; }
    std::size_t gate_bytes() const { return std::size_t{kFfn} * 10 * gb(); }     // per expert
    std::size_t down_bytes() const { return std::size_t{kHidden} * 20 * db(); }
};

static LayerData make_layer(u32 gate_type, u32 down_type, std::mt19937& rng) {
    LayerData w{gate_type, down_type, {}, {}, {}};
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> uni(0.0F, 1.0F);
    auto scale = [&](float lo, float hi) {
        const float r = uni(rng);
        if (r < 0.01F) return 0.0F;
        const float v = std::exp2(lo + (hi - lo) * uni(rng));
        return uni(rng) < 0.1F ? -v : v;
    };
    const u32 GB = w.gb(), DB = w.db();
    auto fill_gate = [&](std::vector<u8>& m) {
        m.resize(std::size_t{kAlloc} * w.gate_bytes());
        for (auto& x : m) x = static_cast<u8>(byte(rng));
        for (std::size_t blk = 0; blk < m.size() / GB; ++blk) {
            const u16 d = half_of(scale(-12, -4)), dm = half_of(scale(-12, -5));
            std::memcpy(m.data() + blk * GB, &d, 2); std::memcpy(m.data() + blk * GB + 2, &dm, 2);
        }
    };
    fill_gate(w.gate); fill_gate(w.up);
    w.down.resize(std::size_t{kAlloc} * w.down_bytes());
    for (auto& x : w.down) x = static_cast<u8>(byte(rng));
    for (std::size_t blk = 0; blk < w.down.size() / DB; ++blk) {
        const u16 d = half_of(scale(-12, -5)); std::memcpy(w.down.data() + blk * DB, &d, 2);
        if (down_type == mo::type_q5_1) { const u16 m = half_of(scale(-10, -4) * -1.0F); std::memcpy(w.down.data() + blk * DB + 2, &m, 2); }
    }
    return w;
}

template <class T> static T* dz(std::size_t n) {
    T* p = nullptr;
    CK(cudaMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)));
    CK(cudaMemset(p, 0, std::max<std::size_t>(n, 1) * sizeof(T)));
    return p;
}
template <class T> static T* upload(const std::vector<T>& v) {
    T* p = dz<T>(v.size());
    CK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

// One set of device buffers + the Rows view of `pad` rows.
struct Set {
    u32 pad;
    u16 *input = nullptr, *output = nullptr;
    u32 *routes = nullptr, *status = nullptr;
    float *rw = nullptr, *deltas = nullptr, *acc = nullptr, *ddeltas = nullptr, *raw = nullptr;
    i8 *codes = nullptr, *dcodes = nullptr;
    i32 *sums = nullptr, *dsums = nullptr;
    mo::Csr csr;
    std::vector<void*> owned;
    template <class T> T* grab(std::size_t n) { T* q = dz<T>(n); owned.push_back(q); return q; }
    explicit Set(u32 p) : pad(p) {
        input = grab<u16>(std::size_t{p} * kHidden); output = grab<u16>(std::size_t{p} * kHidden);
        routes = grab<u32>(std::size_t{p} * kTop); status = grab<u32>(std::size_t{p} * 4);
        rw = grab<float>(std::size_t{p} * kTop); deltas = grab<float>(std::size_t{p} * kHidden / 32);
        acc = grab<float>(std::size_t{p} * kHidden); ddeltas = grab<float>(std::size_t{p} * kTop * kFfn / 32);
        raw = grab<float>(std::size_t{p} * kTop * kHidden);
        codes = grab<i8>(std::size_t{p} * kHidden); dcodes = grab<i8>(std::size_t{p} * kTop * kFfn);
        sums = grab<i32>(std::size_t{p} * kHidden / 32); dsums = grab<i32>(std::size_t{p} * kTop * kFfn / 32);
        const auto sz = mo::csr_sizes(p);
        csr.summary = grab<u32>(32); csr.offsets = grab<u32>(mo::experts + 1); csr.touched = grab<u32>(mo::experts);
        csr.rows = grab<u32>(sz.assignments); csr.slots = grab<u32>(sz.assignments);
        csr.batch_experts = grab<u32>(sz.max_batches); csr.batch_begins = grab<u32>(sz.max_batches + 1);
        csr.tile_count = grab<u32>(1); csr.tile_experts = grab<u32>(sz.max_tiles); csr.tile_begins = grab<u32>(sz.max_tiles + 1);
    }
    ~Set() { for (auto* q : owned) cudaFree(q); }
    Set(const Set&) = delete;
    Set& operator=(const Set&) = delete;
    mo::Rows view(const u32* route_ptr) const {
        mo::Rows v;
        v.input = {input, kHidden * 2}; v.routes = {route_ptr, kTop * 4}; v.route_weights = {rw, kTop * 4};
        v.codes = {codes, kHidden}; v.deltas = {deltas, kHidden / 32 * 4}; v.sums = {sums, kHidden / 32 * 4};
        v.status = {status, 16}; v.intermediates = {nullptr, 0}; v.accumulation = {acc, kHidden * 4};
        v.down_codes = {dcodes, kTop * kFfn}; v.down_deltas = {ddeltas, kTop * kFfn / 32 * 4}; v.down_sums = {dsums, kTop * kFfn / 32 * 4};
        v.rawdots = {raw, kTop * kHidden * 4}; v.output = {output, kHidden * 2}; v.csr = csr;
        return v;
    }
};

struct Result {
    std::vector<float> raw;
    std::vector<u16> out;
    std::vector<u32> status, summary;
};
static Result fetch(const Set& s, u32 rows) {
    Result r;
    r.raw.resize(std::size_t{rows} * kTop * kHidden);
    r.out.resize(std::size_t{rows} * kHidden);
    r.status.resize(std::size_t{rows} * 4);
    r.summary.resize(32);
    CK(cudaMemcpy(r.raw.data(), s.raw, r.raw.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(r.out.data(), s.output, r.out.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(r.status.data(), s.status, r.status.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(r.summary.data(), s.csr.summary, 32 * 4, cudaMemcpyDeviceToHost));
    return r;
}

static std::vector<u32> random_routes(u32 rows, std::mt19937& rng, int forced_first) {
    std::vector<u32> routes(std::size_t{rows} * kTop);
    for (u32 r = 0; r < rows; ++r) {
        std::vector<u32> ids(kAlloc);
        for (u32 i = 0; i < kAlloc; ++i) ids[i] = i;
        std::shuffle(ids.begin(), ids.end(), rng);
        if (r == 0 && forced_first >= 0) {
            const auto at = std::find(ids.begin(), ids.end(), static_cast<u32>(forced_first));
            std::iter_swap(ids.begin(), at);
        }
        for (u32 k = 0; k < kTop; ++k) routes[std::size_t{r} * kTop + k] = ids[k];
    }
    return routes;
}

// Tiered side of one layer: slot buffers of the resident experts (+ the zero slot) and the device table.
struct TierLayer {
    u8 *gate = nullptr, *up = nullptr, *down = nullptr;
    i32* table = nullptr;
    u32 slots = 1;
    TierLayer(const LayerData& w, const std::vector<bool>& resident) {
        std::vector<u32> ids;
        for (u32 e = 0; e < kAlloc; ++e) if (resident[e]) ids.push_back(e);
        slots = std::max<u32>(1, static_cast<u32>(ids.size()));
        gate = dz<u8>(std::size_t{slots + 1} * w.gate_bytes()); up = dz<u8>(std::size_t{slots + 1} * w.gate_bytes());
        down = dz<u8>(std::size_t{slots + 1} * w.down_bytes());
        table = dz<i32>(mo::experts);
        std::vector<i32> t(mo::experts, -1);
        for (u32 s = 0; s < ids.size(); ++s) {
            const u32 e = ids[s];
            t[e] = static_cast<i32>(s);
            CK(cudaMemcpy(gate + std::size_t{s} * w.gate_bytes(), w.gate.data() + std::size_t{e} * w.gate_bytes(), w.gate_bytes(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(up + std::size_t{s} * w.gate_bytes(), w.up.data() + std::size_t{e} * w.gate_bytes(), w.gate_bytes(), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(down + std::size_t{s} * w.down_bytes(), w.down.data() + std::size_t{e} * w.down_bytes(), w.down_bytes(), cudaMemcpyHostToDevice));
        }
        table_set(t);
    }
    void table_set(const std::vector<i32>& t) { CK(cudaMemcpy(table, t.data(), t.size() * 4, cudaMemcpyHostToDevice)); }
    ~TierLayer() { cudaFree(gate); cudaFree(up); cudaFree(down); cudaFree(table); }
};

struct Env {
    am::Tables tables;
    ti::Exchange* ex = nullptr;
    std::vector<LayerData>* layers = nullptr;
    cudaStream_t stream = nullptr;
};

static mo::Weights full_weights(const LayerData& w, const u8* g, const u8* u, const u8* d) {
    mo::Weights x; x.gate_type = w.gate_type; x.down_type = w.down_type; x.gate = g; x.up = u; x.down = d; return x;
}

// Tiered sequence of one layer on `env.stream`.
static void enqueue_tiered(Env& env, u32 layer, Set& t, u32 rows, const TierLayer& tl, u32* slot_routes, const mo::Weights& w) {
    mo::Rows v = t.view(t.routes);
    mo::enqueue_prologue(v, t.pad, kHidden, env.stream);
    ti::enqueue_publish(env.ex->dev(), layer, rows, t.pad, t.routes, slot_routes, tl.table, tl.slots, t.codes, t.deltas, t.sums, nullptr, env.stream);
    v.routes = {slot_routes, kTop * 4};
    mo::enqueue_routed(v, t.pad, w, env.tables, env.stream);
    ti::enqueue_wait_gather(env.ex->dev(), rows, t.raw, kTop * kHidden * 4, t.status, 16, t.csr.summary, env.stream);
    mo::enqueue_combine(v, t.pad, nullptr, env.stream);
}

static void same_live(const Result& a, const Result& b, u32 rows, bool failing, const std::string& name) {
    bool ok = a.summary[0] == b.summary[0];
    for (u32 r = 0; r < rows; ++r) ok = ok && a.status[std::size_t{r} * 4] == b.status[std::size_t{r} * 4];
    g_checked += rows;
    if (!ok) { expect(false, name + ": status / summary differ"); return; }
    if (failing) {
        bool any = false;
        for (u32 r = 0; r < rows; ++r) any = any || (a.status[std::size_t{r} * 4] & 8U);
        expect(any && (a.summary[0] & 8U), name + ": expected a projection_nonfinite failure");
        return;
    }
    expect(a.summary[0] == 0U, name + ": unexpected failure on the full route");
    const std::size_t n = std::size_t{rows} * kTop * kHidden;
    g_checked += n;
    expect(std::memcmp(a.raw.data(), b.raw.data(), n * 4) == 0, name + ": raw dots differ");
    expect(std::memcmp(a.out.data(), b.out.data(), std::size_t{rows} * kHidden * 2) == 0, name + ": routed output differs");
}

enum class Resident { half, all, none };

static void one_case(Env& env, u32 layer, u32 rows, u32 pad, Resident mode, bool poison, bool graph, std::mt19937& rng, const std::string& label) {
    auto& L = (*env.layers)[layer];
    std::vector<bool> resident(kAlloc, false);
    for (u32 e = 0; e < kAlloc; ++e) resident[e] = mode == Resident::all || (mode == Resident::half && (e % 2 == 0));
    const u32 victim = 1;  // a non-resident (odd) expert for the poison case
    const auto routes = random_routes(rows, rng, poison ? static_cast<int>(victim) : -1);
    u8 saved[2] = {0, 0};
    if (poison) {  // gate scale of the first block of the victim's first output row = +inf (half 0x7C00): a non-finite gate
        u8* blk = L.gate.data() + std::size_t{victim} * L.gate_bytes();
        std::memcpy(saved, blk, 2);
        const u16 inf = 0x7C00;
        std::memcpy(blk, &inf, 2);
    }
    Set f(pad), t(pad);
    {
        std::normal_distribution<float> nd(0.0F, 1.0F);
        std::uniform_real_distribution<float> uni(0.0F, 1.0F);
        std::vector<u16> in(std::size_t{pad} * kHidden, 0);
        std::vector<float> rw(std::size_t{pad} * kTop, 0.0F);
        std::vector<u32> rt(std::size_t{pad} * kTop, 0);
        for (u32 r = 0; r < rows; ++r) {
            const float sc = std::exp2(-6.0F + 10.0F * uni(rng));
            for (u32 i = 0; i < kHidden; ++i) in[std::size_t{r} * kHidden + i] = bf16_of(nd(rng) * sc);
            for (u32 k = 0; k < kTop; ++k) { rt[std::size_t{r} * kTop + k] = routes[std::size_t{r} * kTop + k]; rw[std::size_t{r} * kTop + k] = uni(rng); }
        }
        for (Set* s : {&f, &t}) {
            CK(cudaMemcpy(s->input, in.data(), in.size() * 2, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(s->rw, rw.data(), rw.size() * 4, cudaMemcpyHostToDevice));
            CK(cudaMemcpy(s->routes, rt.data(), rt.size() * 4, cudaMemcpyHostToDevice));
        }
    }
    // full route: every expert on the card
    u8* fg = upload(L.gate); u8* fu = upload(L.up); u8* fd = upload(L.down);
    const auto wf = full_weights(L, fg, fu, fd);
    mo::Rows fv = f.view(f.routes);
    mo::enqueue(fv, pad, kHidden, wf, env.tables, nullptr, env.stream);
    CK(cudaStreamSynchronize(env.stream));
    const Result rf = fetch(f, rows);
    // tiered route
    TierLayer tl(L, resident);
    u32* slot_routes = dz<u32>(std::size_t{pad} * kTop);
    const auto wt = full_weights(L, tl.gate, tl.up, tl.down);
    if (!graph) {
        enqueue_tiered(env, layer, t, rows, tl, slot_routes, wt);
        CK(cudaStreamSynchronize(env.stream));
        same_live(rf, fetch(t, rows), rows, poison, label);
    } else {
        cudaGraph_t g = nullptr; cudaGraphExec_t ge = nullptr;
        CK(cudaStreamBeginCapture(env.stream, cudaStreamCaptureModeGlobal));
        enqueue_tiered(env, layer, t, rows, tl, slot_routes, wt);
        CK(cudaStreamEndCapture(env.stream, &g));
        CK(cudaGraphInstantiate(&ge, g, 0));
        for (int replay = 0; replay < 3; ++replay) {
            if (replay == 2) {  // between replays: evict two resident experts (their hits become host misses)
                std::vector<i32> table(mo::experts);
                CK(cudaMemcpy(table.data(), tl.table, table.size() * 4, cudaMemcpyDeviceToHost));
                u32 dropped = 0;
                for (u32 e = 0; e < kAlloc && dropped < 2; ++e) if (table[e] >= 0) { table[e] = -1; ++dropped; }
                tl.table_set(table);
            }
            CK(cudaMemsetAsync(t.status, 0, std::size_t{pad} * 16, env.stream));
            CK(cudaMemsetAsync(t.csr.summary, 0, 32 * 4, env.stream));
            CK(cudaMemsetAsync(t.raw, 0, std::size_t{pad} * kTop * kHidden * 4, env.stream));
            CK(cudaGraphLaunch(ge, env.stream));
            CK(cudaStreamSynchronize(env.stream));
            same_live(rf, fetch(t, rows), rows, poison, label + " graph replay " + std::to_string(replay));
        }
        cudaGraphExecDestroy(ge); cudaGraphDestroy(g);
    }
    if (poison) std::memcpy(L.gate.data() + std::size_t{victim} * L.gate_bytes(), saved, 2);
    cudaFree(fg); cudaFree(fu); cudaFree(fd); cudaFree(slot_routes);
}

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: qwen_tiered_moe_tests <tables dir>\n"); return 2; }
    CK(cudaSetDevice(0));
    Env env;
    env.tables = am::load_tables(argv[1]);
    CK(cudaStreamCreateWithFlags(&env.stream, cudaStreamNonBlocking));
    std::mt19937 rng(20261116);
    std::vector<LayerData> layers;
    layers.push_back(make_layer(mo::type_q4_k, mo::type_q5_1, rng));
    layers.push_back(make_layer(mo::type_q5_k, mo::type_q8_0, rng));
    env.layers = &layers;
    ti::Exchange ex(ti::TierLayout{kHidden, kFfn, kTop, mo::experts, 16});
    env.ex = &ex;
    cx::Gfx1151Exp exp(argv[1]);
    cx::Executor executor(8, exp);
    std::vector<cx::LayerWeights> host;
    for (const auto& L : layers) {
        cx::LayerWeights w;
        w.gate = L.gate.data(); w.up = L.up.data(); w.down = L.down.data();
        w.gate_format = L.gate_type == mo::type_q4_k ? cx::GateFormat::q4_k : cx::GateFormat::q5_k;
        w.down_format = L.down_type == mo::type_q5_1 ? cx::DownFormat::q5_1 : cx::DownFormat::q8_0;
        w.shape = {kHidden, kFfn};
        host.push_back(w);
    }
    ti::TierWorker worker(ex, host, executor);
    for (u32 layer = 0; layer < 2; ++layer)
        for (const auto [rows, pad] : {std::pair<u32, u32>{1, 1}, {8, 8}, {5, 16}}) {
            const std::string base = "layer " + std::to_string(layer) + " rows " + std::to_string(rows) + " pad " + std::to_string(pad);
            one_case(env, layer, rows, pad, Resident::half, false, false, rng, base + " half resident");
            one_case(env, layer, rows, pad, Resident::all, false, false, rng, base + " all resident (no host job)");
            one_case(env, layer, rows, pad, Resident::none, false, false, rng, base + " none resident");
            one_case(env, layer, rows, pad, Resident::half, true, false, rng, base + " non-finite host expert");
        }
    one_case(env, 0, 8, 8, Resident::half, false, true, rng, "graph layer 0 rows 8");
    one_case(env, 1, 5, 16, Resident::half, false, true, rng, "graph layer 1 rows 5 pad 16");
    const auto st = worker.stats();
    expect(st.jobs > 0 && st.misses > 0 && !worker.failed(), "worker answered jobs");
    std::printf("%s: %d failures, %llu items compared, host jobs %llu, host misses %llu, cpu %.1f ms (%s)\n", g_fail ? "FAIL" : "PASS", g_fail,
                g_checked, static_cast<unsigned long long>(st.jobs), static_cast<unsigned long long>(st.misses), st.cpu_ms, executor.isa());
    return g_fail ? 1 : 0;
}
