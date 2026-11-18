// Gate K (batch 2): GLM indexer + absorbed MLA device kernels == host emulation bit for bit.
//   * one slot, 3000 appended tokens with the real L3 indexer weights (cases/mla_l3_r0): every cache byte (latent,
//     indexer key, gate, pooled keys) equal after the append;
//   * canonical token lists at t = 0..4, 2047..2052, 2999 (P up to 750 > 512: real selection), real-scale and
//     all-tie (qI = 0) queries;
//   * attention outputs (32 rank heads, real L3 kv_b rows) at those positions;
//   * the top-k alone at P = 1 .. 32768 with heavy ties and +-0 scores.
// usage: hip_glm_dsa_tests CASES_DIR
#include "glm/host/glm_host.hpp"
#include "glm/kernels/glm_dsa.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

using namespace strix::glm;
namespace H = strix::glm::host;
namespace D = strix::glm::dev;

namespace {

int failures = 0, checks = 0;
void check(bool ok, const std::string& what) {
    ++checks;
    std::printf("%s %s\n", ok ? "OK  " : "FAIL", what.c_str());
    if (!ok) ++failures;
}
void hok(hipError_t e, const char* w) {
    if (e != hipSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, hipGetErrorString(e));
        std::exit(3);
    }
}
template <class T> T* up(const std::vector<T>& h) {
    T* p = nullptr;
    hok(hipMalloc(&p, std::max<std::size_t>(h.size(), 1) * sizeof(T)), "malloc");
    hok(hipMemcpy(p, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice), "h2d");
    return p;
}
template <class T> T* zalloc(std::size_t n) {
    T* p = nullptr;
    hok(hipMalloc(&p, std::max<std::size_t>(n, 1) * sizeof(T)), "malloc");
    hok(hipMemset(p, 0, n * sizeof(T)), "memset");
    return p;
}
template <class T> std::vector<T> down(const T* p, std::size_t n) {
    std::vector<T> h(n);
    hok(hipDeviceSynchronize(), "sync");
    hok(hipMemcpy(h.data(), p, n * sizeof(T), hipMemcpyDeviceToHost), "d2h");
    return h;
}
std::vector<float> rnd(std::size_t n, std::mt19937& g, float scale) {
    std::normal_distribution<float> d(0.0F, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = d(g);
    return v;
}

struct CaseDir {
    std::map<std::string, std::vector<char>> raw;
    explicit CaseDir(const std::string& dir) {
        std::ifstream m(dir + "/manifest.txt");
        if (!m) throw std::runtime_error("no manifest in " + dir);
        std::string line;
        while (std::getline(m, line)) {
            std::istringstream ss(line);
            std::string name, dt;
            std::size_t count = 0;
            ss >> name >> dt >> count;
            const std::size_t es = dt == "u16" ? 2 : (dt == "f64" ? 8 : 4);
            std::vector<char> b(count * es);
            std::ifstream f(dir + "/" + name + ".bin", std::ios::binary);
            f.read(b.data(), static_cast<std::streamsize>(b.size()));
            if (static_cast<std::size_t>(f.gcount()) != b.size()) throw std::runtime_error("short " + name);
            raw.emplace(name, std::move(b));
        }
    }
    template <class T> std::vector<T> v(const std::string& n) const {
        const auto& b = raw.at(n);
        std::vector<T> out(b.size() / sizeof(T));
        std::memcpy(out.data(), b.data(), b.size());
        return out;
    }
};

void test_topk_alone(std::mt19937& g) {
    bool all = true;
    for (unsigned P : {1U, 3U, 511U, 512U, 513U, 750U, 4096U, 32768U}) {
        std::uniform_int_distribution<int> lvl(0, 40);
        std::vector<float> sc(P);
        for (auto& v : sc) v = static_cast<float>(lvl(g)) * 0.125F;  // heavy ties
        if (P > 2) {
            sc[1] = -0.0F;
            sc[2] = 0.0F;
        }
        std::vector<std::uint32_t> keys(P);
        for (unsigned p = 0; p < P; ++p) keys[p] = ordered_key(sc[p]);
        std::uint32_t* dk = up(keys);
        std::uint32_t* dtok = zalloc<std::uint32_t>(kIdxTopk + 3);
        std::uint32_t* dn = zalloc<std::uint32_t>(1);
        const unsigned t = P * kKpool + 2;  // 2 tail tokens
        D::select_tokens(dk, P, t, dtok, dn, nullptr);
        const auto n = down(dn, 1)[0];
        const auto tok = down(dtok, n);
        std::vector<std::uint32_t> ref;
        for (std::uint32_t p : H::topk_ordered(sc, std::min(kIdxPools, P)))
            for (unsigned j = 0; j < kKpool; ++j) ref.push_back(p * kKpool + j);
        for (unsigned pos = P * kKpool; pos <= t; ++pos) ref.push_back(pos);
        all &= tok == ref;
        hok(hipFree(dk), "f");
        hok(hipFree(dtok), "f");
        hok(hipFree(dn), "f");
    }
    check(all, "R5 radix top-k + tail, P = 1..32768, heavy ties and +-0 (vs host stable order)");
}

void test_dsa(const std::string& cases, std::mt19937& g) {
    CaseDir c(cases + "/mla_l3_r0");
    const auto wk = c.v<std::uint16_t>("indexer.wk"), gate = c.v<std::uint16_t>("indexer.index_kpool_compress_gate");
    const auto wp = c.v<std::uint16_t>("indexer.weights_proj"), knw = c.v<std::uint16_t>("indexer.k_norm");
    const auto knb = c.v<std::uint16_t>("indexer.k_norm.bias");
    const auto ape = c.v<float>("indexer.index_kpool_compress_ape");
    const auto kvb_all = c.v<std::uint16_t>("kv_b_proj");
    constexpr unsigned Hh = 32, T = 3000, cap = 3072;
    const std::vector<std::uint16_t> kvb(kvb_all.begin(), kvb_all.begin() + static_cast<std::size_t>(Hh) * 2 * kMlaDim * kKvLora);
    const H::IndexerWeights hw{H::Mat{wk.data(), H::WType::F16, kIdxDim, kHidden}, H::Mat{gate.data(), H::WType::F16, kIdxDim, kHidden},
                               H::Mat{wp.data(), H::WType::F16, kIdxHeads, kHidden}, H::Vec{knw.data(), H::WType::F16, kIdxDim},
                               H::Vec{knb.data(), H::WType::F16, kIdxDim}, ape.data()};
    const D::IndexerDev dw{up(wk), up(gate), up(wp), up(knw), up(knb), up(ape)};
    D::DsaSlot slot{zalloc<std::uint16_t>(static_cast<std::size_t>(cap) * kKvLora), zalloc<std::uint16_t>(static_cast<std::size_t>(cap) * kIdxDim),
                    zalloc<std::uint16_t>(static_cast<std::size_t>(cap) * kIdxDim), zalloc<float>(static_cast<std::size_t>(cap / kKpool) * kIdxDim), cap};
    D::DsaSlot* dslot = up(std::vector<D::DsaSlot>{slot});
    unsigned* dt = zalloc<unsigned>(1);
    unsigned* flag = zalloc<unsigned>(1);
    float *dks = zalloc<float>(kIdxDim), *dgs = zalloc<float>(kIdxDim);
    H::DsaCache hc;
    std::vector<std::vector<float>> xs(T);
    for (unsigned t = 0; t < T; ++t) {  // append
        xs[t] = rnd(kHidden, g, 0.5F);
        const auto lat = rnd(kKvLora, g, 1.0F);
        H::dsa_append(hc, lat, xs[t], hw);
        float *dx = up(xs[t]), *dl = up(lat);
        hok(hipMemcpy(dt, &t, 4, hipMemcpyHostToDevice), "t");
        D::dsa_append(dslot, dt, dl, dx, dw, dks, dgs, flag, 1, nullptr);
        hok(hipFree(dx), "f");
        hok(hipFree(dl), "f");
    }
    const auto lat_d = down(slot.latent, static_cast<std::size_t>(T) * kKvLora);
    const auto key_d = down(slot.ikey, static_cast<std::size_t>(T) * kIdxDim);
    const auto gate_d = down(slot.igate, static_cast<std::size_t>(T) * kIdxDim);
    const auto pool_d = down(slot.pooled, static_cast<std::size_t>(T / kKpool) * kIdxDim);
    check(lat_d == hc.latent && key_d == hc.ikey && gate_d == hc.igate && pool_d == hc.pooled && down(flag, 1)[0] == 0,
          "R4/R12 append x3000 (real L3 indexer weights): latent, key, gate, pooled caches bitwise, no flag");
    // queries
    float *dwh = zalloc<float>(kIdxHeads), *dsc = zalloc<float>(cap / kKpool), *dq = zalloc<float>(Hh * kMlaDim);
    float *dqt = zalloc<float>(Hh * kKvLora), *dqI = zalloc<float>(kIdxHeads * kIdxDim), *dx = zalloc<float>(kHidden), *dout = zalloc<float>(Hh * kMlaDim);
    std::uint32_t *dwork = zalloc<std::uint32_t>(cap), *dtok = zalloc<std::uint32_t>(kIdxTopk + 3), *dn = zalloc<std::uint32_t>(1);
    std::uint16_t* dkvb = up(kvb);
    const H::Mat hkvb{kvb.data(), H::WType::F16, Hh * 2 * kMlaDim, kKvLora};
    bool lists = true, outs = true, ties = true;
    for (unsigned t : {0U, 1U, 2U, 3U, 4U, 2047U, 2048U, 2049U, 2050U, 2051U, 2052U, 2999U}) {
        for (int mode = 0; mode < 2; ++mode) {  // 0: real-scale qI, 1: qI = 0 (all scores tie at 0)
            const auto qI = mode == 0 ? rnd(kIdxHeads * kIdxDim, g, 1.0F) : std::vector<float>(kIdxHeads * kIdxDim, 0.0F);
            const auto q = rnd(Hh * kMlaDim, g, 1.0F);
            hok(hipMemcpy(dqI, qI.data(), qI.size() * 4, hipMemcpyHostToDevice), "q");
            hok(hipMemcpy(dx, xs[t].data(), kHidden * 4, hipMemcpyHostToDevice), "x");
            hok(hipMemcpy(dq, q.data(), q.size() * 4, hipMemcpyHostToDevice), "q");
            D::indexer_select(slot, t, dqI, dx, dw, dwh, dsc, dwork, dtok, dn, nullptr);
            const auto n = down(dn, 1)[0];
            const auto tok = down(dtok, n);
            const auto ref = H::indexer_select(hc, t, qI, xs[t], hw);
            (mode == 0 ? lists : ties) &= tok == ref;
            if (mode == 1 && t >= 2051) {  // all tie -> the lowest 512 pools
                bool low = ref.size() >= kIdxTopk;
                for (unsigned i = 0; low && i < kIdxTopk; ++i) low = ref[i] == i;
                ties &= low;
            }
            D::mla_attention(slot, dtok, dn, dq, dkvb, Hh, dqt, dout, nullptr);
            const auto od = down(dout, Hh * kMlaDim);
            std::vector<float> oh(Hh * kMlaDim);
            H::mla_attention(hc, ref, q, hkvb, Hh, oh);
            outs &= std::memcmp(od.data(), oh.data(), oh.size() * 4) == 0;
        }
    }
    check(lists, "R11/R5 + section 1.4 token lists at t = 0..4, 2047..2052, 2999 (P up to 750), real-scale queries");
    check(ties, "all-tie queries (qI = 0) select the lowest pools (lower-index rule)");
    check(outs, "R13 absorbed MLA, 32 rank heads, real L3 kv_b rows: outputs bitwise at all positions (n up to 2051)");
}

// Hand-off index tail (wire v3/v4, strix/block_summary_state.hpp): a slot restored at frontier P with only the open pool
// block's index key / gate rows [P - P % 4, P) continues bitwise like the full state. The closed blocks' rows are
// poisoned with BF16 NaN (0xFFFF): any read would turn a pooled key non-finite (flag) or change a list / output.
void test_index_tail(const std::string& cases, std::mt19937& g) {
    CaseDir c(cases + "/mla_l3_r0");
    const auto wk = c.v<std::uint16_t>("indexer.wk"), gate = c.v<std::uint16_t>("indexer.index_kpool_compress_gate");
    const auto wp = c.v<std::uint16_t>("indexer.weights_proj"), knw = c.v<std::uint16_t>("indexer.k_norm");
    const auto knb = c.v<std::uint16_t>("indexer.k_norm.bias");
    const auto ape = c.v<float>("indexer.index_kpool_compress_ape");
    const auto kvb_all = c.v<std::uint16_t>("kv_b_proj");
    constexpr unsigned Hh = 32, P0 = 2996, K = 13, cap = 3072;
    const std::vector<std::uint16_t> kvb(kvb_all.begin(), kvb_all.begin() + static_cast<std::size_t>(Hh) * 2 * kMlaDim * kKvLora);
    const D::IndexerDev dw{up(wk), up(gate), up(wp), up(knw), up(knb), up(ape)};
    auto slot_alloc = [&] {
        return D::DsaSlot{zalloc<std::uint16_t>(static_cast<std::size_t>(cap) * kKvLora), zalloc<std::uint16_t>(static_cast<std::size_t>(cap) * kIdxDim),
                          zalloc<std::uint16_t>(static_cast<std::size_t>(cap) * kIdxDim), zalloc<float>(static_cast<std::size_t>(cap / kKpool) * kIdxDim), cap};
    };
    const std::size_t lat_b = static_cast<std::size_t>(cap) * kKvLora * 2, idx_b = static_cast<std::size_t>(cap) * kIdxDim * 2,
                      pool_b = static_cast<std::size_t>(cap / kKpool) * kIdxDim * 4;
    auto copy = [&](const D::DsaSlot& to, const D::DsaSlot& from) {
        hok(hipMemcpy(to.latent, from.latent, lat_b, hipMemcpyDeviceToDevice), "c");
        hok(hipMemcpy(to.ikey, from.ikey, idx_b, hipMemcpyDeviceToDevice), "c");
        hok(hipMemcpy(to.igate, from.igate, idx_b, hipMemcpyDeviceToDevice), "c");
        hok(hipMemcpy(to.pooled, from.pooled, pool_b, hipMemcpyDeviceToDevice), "c");
    };
    const D::DsaSlot a = slot_alloc(), full = slot_alloc(), tail = slot_alloc();
    D::DsaSlot *da = up(std::vector<D::DsaSlot>{a}), *dfull = up(std::vector<D::DsaSlot>{full}), *dtail = up(std::vector<D::DsaSlot>{tail});
    unsigned *dt = zalloc<unsigned>(1), *flag = zalloc<unsigned>(1), *flag_full = zalloc<unsigned>(1), *flag_tail = zalloc<unsigned>(1);
    float *dks = zalloc<float>(kIdxDim), *dgs = zalloc<float>(kIdxDim), *dx = zalloc<float>(kHidden), *dl = zalloc<float>(kKvLora);
    float *dwh = zalloc<float>(kIdxHeads), *dsc = zalloc<float>(cap / kKpool), *dq = zalloc<float>(Hh * kMlaDim);
    float *dqt = zalloc<float>(Hh * kKvLora), *dqI = zalloc<float>(kIdxHeads * kIdxDim), *dout = zalloc<float>(Hh * kMlaDim);
    std::uint32_t *dwork = zalloc<std::uint32_t>(cap), *dtok = zalloc<std::uint32_t>(kIdxTopk + 3), *dn = zalloc<std::uint32_t>(1);
    std::uint16_t* dkvb = up(kvb);
    auto append = [&](D::DsaSlot* s, unsigned* f, unsigned t, const std::vector<float>& x, const std::vector<float>& lat) {
        hok(hipMemcpy(dt, &t, 4, hipMemcpyHostToDevice), "t");
        hok(hipMemcpy(dx, x.data(), kHidden * 4, hipMemcpyHostToDevice), "x");
        hok(hipMemcpy(dl, lat.data(), kKvLora * 4, hipMemcpyHostToDevice), "l");
        D::dsa_append(s, dt, dl, dx, dw, dks, dgs, f, 1, nullptr);
    };
    // One step's outputs: the selected tokens, then the MLA output bits.
    auto query = [&](const D::DsaSlot& s, unsigned t, const std::vector<float>& x, const std::vector<float>& qI, const std::vector<float>& q) {
        hok(hipMemcpy(dx, x.data(), kHidden * 4, hipMemcpyHostToDevice), "x");
        hok(hipMemcpy(dqI, qI.data(), qI.size() * 4, hipMemcpyHostToDevice), "q");
        hok(hipMemcpy(dq, q.data(), q.size() * 4, hipMemcpyHostToDevice), "q");
        D::indexer_select(s, t, dqI, dx, dw, dwh, dsc, dwork, dtok, dn, nullptr);
        const auto n = down(dn, 1)[0];
        auto tok = down(dtok, n);
        D::mla_attention(s, dtok, dn, dq, dkvb, Hh, dqt, dout, nullptr);
        const auto o = down(dout, Hh * kMlaDim);
        std::vector<std::uint32_t> bits(o.size());
        std::memcpy(bits.data(), o.data(), o.size() * 4);
        tok.insert(tok.end(), bits.begin(), bits.end());
        return tok;
    };
    for (unsigned t = 0; t < P0; ++t) append(da, flag, t, rnd(kHidden, g, 0.5F), rnd(kKvLora, g, 1.0F));
    bool steps = true, caches = true, holes = true, flags = true;
    unsigned compared = 0;
    for (unsigned P = P0; P < P0 + 4; ++P) {  // P % 4 = 0, 1, 2, 3
        copy(full, a);
        copy(tail, a);
        const unsigned first = P - P % kKpool;  // the open pool block's first row
        const std::vector<std::uint16_t> poison(static_cast<std::size_t>(first) * kIdxDim, 0xFFFFU);
        hok(hipMemcpy(tail.ikey, poison.data(), poison.size() * 2, hipMemcpyHostToDevice), "p");
        hok(hipMemcpy(tail.igate, poison.data(), poison.size() * 2, hipMemcpyHostToDevice), "p");
        hok(hipMemset(flag_full, 0, 4), "f");
        hok(hipMemset(flag_tail, 0, 4), "f");
        for (unsigned t = P; t < P + K; ++t) {  // continue: crosses 3..4 block closes
            const auto x = rnd(kHidden, g, 0.5F), lat = rnd(kKvLora, g, 1.0F);
            append(dfull, flag_full, t, x, lat);
            append(dtail, flag_tail, t, x, lat);
            for (int mode = 0; mode < 2; ++mode) {  // 0: real-scale qI, 1: qI = 0 (all scores tie)
                const auto qI = mode == 0 ? rnd(kIdxHeads * kIdxDim, g, 1.0F) : std::vector<float>(kIdxHeads * kIdxDim, 0.0F);
                const auto q = rnd(Hh * kMlaDim, g, 1.0F);
                steps &= query(full, t, x, qI, q) == query(tail, t, x, qI, q);
                ++compared;
            }
        }
        const unsigned end = P + K;
        caches &= down(full.latent, static_cast<std::size_t>(end) * kKvLora) == down(tail.latent, static_cast<std::size_t>(end) * kKvLora);
        const auto pooled_full = down(full.pooled, static_cast<std::size_t>(end / kKpool) * kIdxDim);
        const auto pooled_tail = down(tail.pooled, static_cast<std::size_t>(end / kKpool) * kIdxDim);
        caches &= std::memcmp(pooled_full.data(), pooled_tail.data(), pooled_full.size() * sizeof(float)) == 0;
        const auto kf = down(full.ikey, static_cast<std::size_t>(end) * kIdxDim), kt = down(tail.ikey, static_cast<std::size_t>(end) * kIdxDim);
        const auto gf = down(full.igate, static_cast<std::size_t>(end) * kIdxDim), gt = down(tail.igate, static_cast<std::size_t>(end) * kIdxDim);
        const auto f0 = static_cast<std::ptrdiff_t>(static_cast<std::size_t>(first) * kIdxDim);
        caches &= std::equal(kf.begin() + f0, kf.end(), kt.begin() + f0) && std::equal(gf.begin() + f0, gf.end(), gt.begin() + f0);
        holes &= std::all_of(kt.begin(), kt.begin() + f0, [](std::uint16_t v) { return v == 0xFFFFU; }) &&
                 std::all_of(gt.begin(), gt.begin() + f0, [](std::uint16_t v) { return v == 0xFFFFU; });
        flags &= down(flag_full, 1)[0] == 0 && down(flag_tail, 1)[0] == 0;
        append(da, flag, P, rnd(kHidden, g, 0.5F), rnd(kKvLora, g, 1.0F));  // the next frontier
    }
    check(steps && compared == 4 * K * 2, "index tail: token lists + MLA outputs bitwise vs the full state, P % 4 = 0..3, 13 steps each");
    check(caches, "index tail: latent, pooled and the open-block index rows bitwise after the continuation");
    check(holes && flags, "index tail: NaN-poisoned closed-block index rows never read (untouched, no non-finite flag)");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s CASES_DIR\n", argv[0]);
        return 2;
    }
    std::mt19937 g(20261024);
    test_topk_alone(g);
    test_dsa(argv[1], g);
    test_index_tail(argv[1], g);
    std::printf("checks: %d failures: %d\n", checks, failures);
    return failures ? 1 : 0;
}
