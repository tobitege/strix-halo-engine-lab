// Device self-test of the shared vision encoder: synthetic towers on the HIP runner against the host FP32 reference.
// Gate (per row of the hidden state and of every tap): cosine mean >= 0.9995, min >= 0.998, relative L2 <= 2 %.
// Needs a GPU; run it before any family adapter's ticket. Exit 0 = PASS. Every line carries the FNV-1a of the device
// output bytes. `--bench OFF_REFERENCE` checks tail linears against a separate OFF capture and adds a forward of a
// 32 x 1024 tower (SwiGLU 2816, 1,521 patches) with its time.
// Capture with both STRIX_F16_DENSE_TAIL=0 and STRIX_VISION_GEMM_EPILOGUE=0 using --bench-save-reference FILE.
#include "strix/vision/embed_cache.hpp"

#include "vision_encoder_toy.hpp"
#include "dense_fixture_checks.hpp"
#include "strix/env_flags.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

using namespace strix::vision;
using vision_toy::make_grid;
using vision_toy::make_tower;
using vision_toy::Tower;

namespace {

struct Score {
    double cos_mean = 0.0, cos_min = 1.0, rel_l2 = 0.0;
    bool pass() const { return cos_mean >= 0.9995 && cos_min >= 0.998 && rel_l2 <= 0.02; }
};
Score score(const std::vector<float>& got, const std::vector<float>& want, unsigned rows, unsigned width) {
    Score s;
    if (got.size() != want.size() || got.size() != std::size_t{rows} * width) return {0.0, 0.0, 1.0};
    double err = 0.0, ref = 0.0;
    for (unsigned r = 0; r < rows; ++r) {
        double ab = 0.0, aa = 0.0, bb = 0.0;
        for (unsigned k = 0; k < width; ++k) {
            const double a = static_cast<double>(got[std::size_t{r} * width + k]), b = static_cast<double>(want[std::size_t{r} * width + k]);
            ab += a * b;
            aa += a * a;
            bb += b * b;
            err += (a - b) * (a - b);
        }
        const double c = aa > 0.0 && bb > 0.0 ? ab / std::sqrt(aa * bb) : 0.0;
        s.cos_mean += c / rows;
        s.cos_min = std::min(s.cos_min, c);
        ref += bb;
    }
    s.rel_l2 = ref > 0.0 ? std::sqrt(err / ref) : 1.0;
    return s;
}

std::uint64_t fnv(const std::vector<float>& v, std::uint64_t h = 1469598103934665603ULL) {
    for (const float f : v) {
        std::uint32_t b;
        std::memcpy(&b, &f, 4);
        for (unsigned k = 0; k < 4U; ++k) h = (h ^ ((b >> (8U * k)) & 0xFFU)) * 1099511628211ULL;
    }
    return h;
}
unsigned long long hash_of(const EncoderOutput& o) {
    std::uint64_t h = fnv(o.hidden);
    for (const auto& t : o.taps) h = fnv(t, h);
    return h;
}

struct Case {
    const char* name;
    EncoderConfig cfg;
    unsigned gh, gw, tiles;
    bool split;
};

EncoderConfig tower(unsigned width, unsigned heads, unsigned mlp, EncoderConfig::Norm norm, EncoderConfig::Mlp act, EncoderConfig::Pos pos,
                    unsigned patch) {
    EncoderConfig c;
    c.layers = 3;
    c.width = width;
    c.heads = heads;
    c.mlp_width = mlp;
    c.norm = norm;
    c.mlp = act;
    c.pos = pos;
    c.pos_grid = c.has_learned() ? 6U : 0U;
    c.patch = patch;
    c.qkv_bias = c.out_bias = true;
    c.mlp_bias = act != EncoderConfig::Mlp::swiglu;
    return c;
}

bool run(const Case& k, std::uint64_t seed) {
    Tower t = make_tower(k.cfg, seed, 48);
    const PatchGrid g = make_grid(k.cfg, k.gh, k.gw, seed + 1U, k.tiles);
    const PositionInputs p = default_positions(k.cfg, t.w, g);
    const EncoderOutput want = reference_encode(k.cfg, t.w, g, p);
    EncoderOptions opt;
    opt.split_activations = k.split;
    opt.mlp_workspace_bytes = std::size_t{1} << 20U;  // several MLP chunks
    Encoder enc(k.cfg, opt);
    enc.load(t.w);
    EncodeStats stats;
    stats.per_layer = true;
    const EncoderOutput got = enc.encode(g, p, stats);
    const EncoderOutput twice = enc.encode(g, p);
    bool ok = got.hidden == twice.hidden && got.taps.size() == want.taps.size();  // deterministic, workspace reuse
    Score worst = score(got.hidden, want.hidden, g.rows, k.cfg.width);
    ok = ok && worst.pass();
    for (std::size_t i = 0; ok && i < want.taps.size(); ++i) {
        const Score s = score(got.taps[i], want.taps[i], g.rows, k.cfg.width);
        ok = s.pass();
        if (s.cos_min < worst.cos_min) worst = s;
    }
    // projector linear + cache path on the device backend
    vision_toy::ToyAdapter adapter(t);
    ReferenceEncoder ref(k.cfg, t.w);
    Digest d{};
    const CachedEncode dev = encode_cached(adapter, enc, t.w, nullptr, d, g), host = encode_cached(adapter, ref, t.w, nullptr, d, g);
    std::vector<float> a(dev.rows->bf16.size()), b(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
        a[i] = from_bf16(dev.rows->bf16[i]);
        b[i] = from_bf16(host.rows->bf16[i]);
    }
    const Score proj = score(a, b, g.rows, 48);
    ok = ok && proj.pass();
    const EncoderPlan pl = plan(k.cfg, g.rows, k.tiles, opt);
    ok = ok && enc.device_bytes() <= pl.weight_bytes + pl.workspace_bytes() + (std::size_t{1} << 20U);
    std::printf("%-22s rows %4u  cos mean %.6f min %.6f  relL2 %.5f  proj min %.6f  %.2f ms  dev %.1f MiB  fnv %016llx proj %016llx  %s\n",
                k.name, g.rows, worst.cos_mean, worst.cos_min, worst.rel_l2, proj.cos_min, stats.total_ms,
                static_cast<double>(enc.device_bytes()) / 1048576.0, hash_of(got), static_cast<unsigned long long>(fnv(a)), ok ? "PASS" : "FAIL");
    return ok;
}

// Projector linears at shapes that exercise the remainder launch and the last partial CTA, plain and split, and the
// non-finite report. Every output byte is compared with a separately saved OFF arm.
void bench_linears(Encoder& enc, const char* tag, std::istream* reference, std::ostream* save) {
    vision_toy::Rng rng{977U};
    const unsigned in = 1024;
    std::vector<std::vector<float>> weights;  // the encoder keys its panels by weight address: keep every one alive
    for (const unsigned n : {1024U, 272U, 16U, 1000U}) {
        weights.emplace_back(std::size_t{n} * in);
        std::vector<float>& w = weights.back();
        std::vector<float> b(n);
        for (auto& v : w) v = rng.next() * 0.05F;
        for (auto& v : b) v = rng.next();
        const Tensor tw{w.data(), Tensor::Type::f32, n, in}, tb{b.data(), Tensor::Type::f32, n, 1};
        for (const unsigned rows : {128U, 129U, 1U}) {
            std::vector<float> x(std::size_t{rows} * in), y(std::size_t{rows} * n, -7.0F);
            for (auto& v : x) v = rng.next() * 3.0F;
            enc.linear(tw, tb, x.data(), rows, y.data());
            dense_fixture::linear_reference(reference, save, y, rows, in, n);
            std::printf("LINEAR %s rows %3u in %u n %4u fnv %016llx\n", tag, rows, in, n, static_cast<unsigned long long>(fnv(y)));
            if (rows != 129U || n != 272U) continue;
            x[5U * in + 7U] = 3.0e38F;  // FP16 overflow in one row: the linear must report it
            dense_fixture::require_numeric_failure([&] {
                enc.linear(tw, tb, x.data(), rows, y.data());
            });
            std::printf("NONFINITE %s rows %u n %u reported 1\n", tag, rows, n);
        }
    }
}

void bench(std::istream* reference, std::ostream* save) {
    using C = EncoderConfig;
    {
        const C small = tower(128, 2, 352, C::Norm::rmsnorm, C::Mlp::swiglu, C::Pos::rope2d, 14);
        Tower t = make_tower(small, 31, 48);
        for (const bool split : {false, true}) {
            EncoderOptions opt;
            opt.split_activations = split;
            Encoder enc(small, opt);
            enc.load(t.w);
            bench_linears(enc, split ? "split" : "plain", reference, save);
        }
    }
    C big = tower(1024, 16, 2816, C::Norm::rmsnorm, C::Mlp::swiglu, C::Pos::rope2d, 14);
    big.layers = 32;
    Tower t = make_tower(big, 53, 0);
    const PatchGrid g = make_grid(big, 39, 39, 54, 1);
    const PositionInputs p = default_positions(big, t.w, g);
    for (const bool split : {false}) {  // split is covered by the small towers; one load of this tower is 0.8 GB of panels
        EncoderOptions opt;
        opt.split_activations = split;
        Encoder enc(big, opt);
        enc.load(t.w);
        std::vector<double> ms;
        unsigned long long h = 0;
        EncoderOutput first;
        for (unsigned i = 0; i < 16U; ++i) {
            EncodeStats st;
            const EncoderOutput o = enc.encode(g, p, st);
            const unsigned long long hi = hash_of(o);
            if (i == 0) first = o;
            dense_fixture::require_repeatable(o, first);
            h = hi;
            if (i) ms.push_back(st.total_ms);  // the first forward pays the workspace
        }
        std::sort(ms.begin(), ms.end());
        std::printf("FORWARD %s layers %u width %u rows %u fnv %016llx repeatable %d  ms min %.3f median %.3f max %.3f\n", split ? "split" : "plain",
                    big.layers, big.width, g.rows, h, 1, ms.front(), ms[ms.size() / 2U], ms.back());
    }
}

}  // namespace

int main(int argc, char** argv) {
    using C = EncoderConfig;
    const bool with_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
    const bool save_bench = argc > 1 && std::strcmp(argv[1], "--bench-save-reference") == 0;
    if (argc != 1 && (!(with_bench || save_bench) || argc != 3)) {
        std::fprintf(stderr, "usage: %s [--bench OFF_REFERENCE | --bench-save-reference OFF_REFERENCE]\n", argv[0]);
        return 1;
    }
    std::vector<Case> cases;
    cases.push_back({"d64-rms-swiglu-rope", tower(128, 2, 352, C::Norm::rmsnorm, C::Mlp::swiglu, C::Pos::rope2d, 14), 13, 13, 1, false});
    cases.push_back({"d64-rms-swiglu-tiles", tower(128, 2, 352, C::Norm::rmsnorm, C::Mlp::swiglu, C::Pos::rope2d, 14), 12, 12, 3, false});
    Case deep{"d72-ln-tanh-both-taps", tower(144, 2, 538, C::Norm::layernorm, C::Mlp::gelu_tanh, C::Pos::rope2d_plus_learned, 16), 10, 14, 1, false};
    deep.cfg.deepstack_taps = {0, 2};
    deep.cfg.final_norm = true;
    cases.push_back(deep);
    cases.push_back({"d80-ln-gelu-learned", tower(160, 2, 640, C::Norm::layernorm, C::Mlp::gelu, C::Pos::learned_interp, 14), 9, 7, 1, false});
    cases.push_back({"d64-split", tower(128, 2, 352, C::Norm::rmsnorm, C::Mlp::swiglu, C::Pos::rope2d, 14), 13, 13, 1, true});
    Case clamp{"d64-swiglu-limit-adj", tower(256, 4, 512, C::Norm::rmsnorm, C::Mlp::swiglu, C::Pos::rope2d, 14), 20, 17, 1, false};
    clamp.cfg.swiglu_limit = 1.0F;
    clamp.cfg.swiglu_gate_first = false;
    clamp.cfg.rope_pairing = C::RopePairing::adjacent;
    cases.push_back(clamp);
    bool ok = true;
    try {
        std::ifstream reference;
        std::ofstream save;
        if (save_bench) {
            dense_fixture::require(!strix::envcfg::flag("STRIX_F16_DENSE_TAIL", true) &&
                                   !strix::envcfg::flag("STRIX_VISION_GEMM_EPILOGUE", true),
                                   "reference capture requires both dense tail and vision epilogue switches OFF");
            save.open(argv[2], std::ios::binary | std::ios::trunc);
            dense_fixture::require(static_cast<bool>(save), "cannot open reference for writing");
        } else if (with_bench) {
            reference.open(argv[2], std::ios::binary);
            dense_fixture::require(static_cast<bool>(reference), "cannot open OFF reference");
        }
        std::uint64_t seed = 101;
        for (const auto& k : cases) ok = run(k, seed += 7U) && ok;
        if (with_bench || save_bench) bench(with_bench ? &reference : nullptr, save_bench ? &save : nullptr);
        if (with_bench) dense_fixture::finish_reference(reference);
        if (save_bench) {
            save.close();
            dense_fixture::require(static_cast<bool>(save), "reference close failed");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL exception: %s\n", e.what());
        return 1;
    }
    std::puts(ok ? "vision-encoder-hip: PASS" : "vision-encoder-hip: FAIL");
    return ok ? 0 : 1;
}
