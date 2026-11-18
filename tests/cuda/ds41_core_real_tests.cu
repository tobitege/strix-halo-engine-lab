// deepseek41 CUDA layer core on real weights: the decode data flow of layers 0-2 (embedding, Engram, hc mixes, collapse +
// norms, q_a / q_b / kv, a stand-in attention output through o_a / o_b, expansions, router, shared expert) and the head
// with its greedy selection, every site bitwise against the host reference with the reference's inputs.
//   STRIX_DS41_FIXTURE_GGUF=ds41-dense-0-1-2-20.gguf strix-cuda-ds41-core-real-tests [rows]
// The file needs the dense tensors of layers 0-2, the embedding and the head (the model or a subset). Engram rows are
// synthetic (the tables are not needed). Skips (exit 0) without the variable.

#include "strix/cuda/ds41_common.hpp"
#include "strix/cuda/ds41_core.hpp"
#include "strix/cuda/ds41_weights.hpp"

#include "strix/deepseek41_model.hpp"
#include "strix/ds41_formats.hpp"
#include "strix/ds41_layer.hpp"
#include "strix/ds41_moe.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace c = strix::ds41::cuda;
namespace d = strix::ds41;

namespace {

int failures = 0;
std::mt19937_64 rng(41);

void expect_bits(const std::string& what, const std::vector<float>& got, const std::vector<float>& want) {
    std::size_t bad = 0, first = 0;
    for (std::size_t i = 0; i < want.size(); ++i)
        if (i >= got.size() || std::memcmp(&got[i], &want[i], 4) != 0) {
            if (!bad) first = i;
            ++bad;
        }
    if (bad) {
        ++failures;
        std::printf("FAIL %-28s %zu of %zu differ (first %zu: %.9g vs %.9g)\n", what.c_str(), bad, want.size(), first, got[first], want[first]);
    } else {
        std::printf("ok   %-28s %7zu values, fnv %016llx\n", what.c_str(), want.size(),
                    static_cast<unsigned long long>(c::fnv1a(got.data(), got.size())));
    }
}

std::vector<float> random_floats(std::size_t n, float scale) {
    std::normal_distribution<float> g(0.0F, scale);
    std::vector<float> v(n);
    for (auto& x : v) x = g(rng);
    return v;
}

template <class T>
c::DeviceBuffer<T> dev(const std::vector<T>& v) {
    c::DeviceBuffer<T> b(v.size());
    b.upload(v);
    return b;
}
template <class T>
std::vector<T> host(const c::DeviceBuffer<T>& b) { return b.download(); }

std::vector<float> host_q8(const std::vector<std::uint8_t>& w, std::size_t width, std::size_t count, std::size_t per_group,
                           const std::vector<float>& x, std::size_t rows, std::size_t x_width) {
    std::vector<float> y(rows * count);
    for (std::size_t r = 0; r < rows; ++r)
        d::q8_0_a8_rows(w.data(), width, 0, count, per_group ? per_group : count,
                        d::quantize_a8(std::span<const float>(x.data() + r * x_width, x_width)), y.data() + r * count);
    return y;
}

c::A8Rows a8_of(const float* x, std::uint32_t rows, std::uint32_t width, c::Workspace& ws, cudaStream_t s) {
    c::A8Rows a{ws.take<std::int8_t>(std::size_t{rows} * width), ws.take<float>(std::size_t{rows} * width / 32U),
                ws.take<std::int32_t>(std::size_t{rows} * width / 32U), rows, width, width};
    c::enqueue_quantize_a8({x, rows, width, 0, a}, s);
    return a;
}

}  // namespace

int main(int argc, char** argv) try {
    const char* path = std::getenv("STRIX_DS41_FIXTURE_GGUF");
    if (!path) {
        std::printf("STRIX_DS41_FIXTURE_GGUF not set: skipped\n");
        return 0;
    }
    const std::uint32_t R = argc > 1 ? static_cast<std::uint32_t>(std::atoi(argv[1])) : 3U;
    const auto devinfo = c::select_device(0);
    c::TensorFile f(path);
    const auto cfg = f.config();
    const std::uint32_t H = cfg.hidden_size, N = 4U * H, V = cfg.vocab_size, I = cfg.moe_intermediate_size;
    const std::uint32_t E = cfg.n_routed_experts, K = cfg.num_experts_per_tok, QL = cfg.q_lora_rank, HD = cfg.head_dim;
    const std::uint32_t QO = cfg.num_attention_heads * HD, OL = cfg.o_groups * cfg.o_lora_rank;
    std::printf("device %s, %u rows, hidden %u, vocab %u\n", devinfo.name.c_str(), R, H, V);
    c::Stream st;
    const auto s = st.get();
    c::Workspace ws(512ULL << 20);
    const std::vector<std::uint32_t> tokens_all{0, 671, 128000, 4242, 77, 1, 9999, 31337, 12, 100000, 5, 6};
    const std::vector<std::uint32_t> tokens(tokens_all.begin(), tokens_all.begin() + R);

    // ---- embedding ----
    std::vector<float> stream(std::size_t{R} * N), pre_carry(std::size_t{R} * 4U);
    for (std::uint32_t r = 0; r < R; ++r) {
        const auto row = f.host("token_embd.weight", std::uint64_t{tokens[r]} * H * 2U, std::uint64_t{H} * 2U);
        d::embed_token(reinterpret_cast<const std::uint16_t*>(row.data()), H, 0,
                       std::span<float>(stream.data() + std::size_t{r} * N, N));
        const auto p = d::hc_identity_pre();
        std::copy(p.begin(), p.end(), pre_carry.begin() + r * 4);
    }
    {
        auto dtok = dev(tokens);
        c::DeviceBuffer<float> ds(stream.size()), dp(pre_carry.size());
        c::enqueue_embed({{f.device_f16("token_embd.weight"), V, H}, dtok.get(), R, ds.get(), dp.get()}, s);
        expect_bits("embedding", host(ds), stream);
        expect_bits("embedding pre_carry", host(dp), pre_carry);
        f.release("token_embd.weight");
    }

    for (std::uint32_t layer = 0; layer < 3; ++layer) {
        const std::string p = "blk." + std::to_string(layer) + ".";
        std::printf("-- layer %u\n", layer);
        // ---- Engram (layers with engram_kv), synthetic rows ----
        if (layer == 1) {
            const std::uint32_t cols = 24;
            std::vector<std::uint8_t> rows(std::size_t{R} * cols * 264U);
            std::uniform_int_distribution<int> code(0, 255), sc(118, 125);
            for (std::size_t k = 0; k < std::size_t{R} * cols; ++k) {
                for (int j = 0; j < 256; ++j) {
                    int v = code(rng);
                    if ((v & 0x7f) == 0x7f) v &= 0xfe;
                    rows[k * 264 + j] = static_cast<std::uint8_t>(v);
                }
                for (int j = 0; j < 8; ++j) rows[k * 264 + 256 + j] = static_cast<std::uint8_t>(sc(rng));
            }
            const auto kvw = f.host_as<std::uint16_t>(p + "engram_kv.weight");
            const auto qn = f.host_as<float>(p + "engram_q_norm.weight"), kn = f.host_as<float>(p + "engram_k_norm.weight");
            auto drows = dev(rows);
            auto dstream = dev(stream);
            c::DeviceBuffer<float> dkv(std::size_t{R} * 5U * H);
            c::EngramView v;
            v.row_bytes = drows.get();
            v.rows = R;
            v.cols = cols;
            v.hidden = H;
            v.kv = {f.device_f16(p + "engram_kv.weight"), 5U * H, cols * 256U};
            v.q_norm = f.device_f32(p + "engram_q_norm.weight");
            v.k_norm = f.device_f32(p + "engram_k_norm.weight");
            v.stream = dstream.get();
            v.kv_out = dkv.get();
            v.workspace = ws.take_bytes(c::engram_workspace_bytes(R, cols, H));
            c::enqueue_engram(v, s);
            std::vector<float> kv(std::size_t{R} * 5U * H);
            for (std::uint32_t r = 0; r < R; ++r) {
                std::vector<float> emb(std::size_t{cols} * 256U);
                d::engram_rows_to_bf16(std::span<const std::uint8_t>(rows.data() + std::size_t{r} * cols * 264U, std::size_t{cols} * 264U), emb);
                d::f16_rows(kvw.data(), emb.size(), 0, 5U * H, emb.data(), kv.data() + std::size_t{r} * 5U * H);
                d::engram_inject(std::span<float>(stream.data() + std::size_t{r} * N, N),
                                 std::span<const float>(kv.data() + std::size_t{r} * 5U * H, 5U * H), qn, kn, 1e-20F);
            }
            expect_bits("engram_kv (real F16)", host(dkv), kv);
            expect_bits("engram injection", host(dstream), stream);
            f.release(p + "engram_kv.weight");
            ws.reset();
        }
        // ---- hc mixes of both sites (the FFN one on the stream after a stand-in attention) ----
        auto mixes = [&](const std::string& site, const std::vector<float>& in) {
            const auto fn = f.host_as<std::uint16_t>(p + "hc_" + site + "_fn.weight");
            const auto base = f.host_as<float>(p + "hc_" + site + "_base.weight"), scale = f.host_as<float>(p + "hc_" + site + "_scale.weight");
            std::vector<float> want(std::size_t{R} * 24U);
            for (std::uint32_t r = 0; r < R; ++r) {
                const auto m = d::hc_mixes(std::span<const float>(in.data() + std::size_t{r} * N, N), d::HcSiteWeights{fn, base, scale},
                                           cfg.hc_sinkhorn_iters, static_cast<float>(cfg.hc_eps), 1e-20F);
                std::copy(m.pre.begin(), m.pre.end(), want.begin() + r * 24);
                std::copy(m.post.begin(), m.post.end(), want.begin() + r * 24 + 4);
                std::copy(m.comb.begin(), m.comb.end(), want.begin() + r * 24 + 8);
            }
            auto din = dev(in);
            c::DeviceBuffer<float> dm(want.size());
            c::HcMixView v;
            v.stream = din.get();
            v.rows = R;
            v.hidden = H;
            v.fn = f.device_f16(p + "hc_" + site + "_fn.weight");
            v.base = f.device_f32(p + "hc_" + site + "_base.weight");
            v.scale = f.device_f32(p + "hc_" + site + "_scale.weight");
            v.iters = cfg.hc_sinkhorn_iters;
            v.hc_eps = static_cast<float>(cfg.hc_eps);
            v.mixes = dm.get();
            v.partial = static_cast<float*>(ws.take_bytes(c::hc_mix_workspace_bytes(R, H)));
            c::enqueue_hc_mixes(v, s);
            expect_bits("hc_mixes " + site, host(dm), want);
            return want;
        };
        // collapse + norm (+ A8) of the stream with a pre of stride ps
        auto collapse_norm = [&](const std::vector<float>& in, const std::vector<float>& pre, std::uint32_t ps, const std::string& wname,
                                 const std::string& tag) {
            const auto wv = f.host_as<float>(wname);
            std::vector<float> col(std::size_t{R} * H), y(std::size_t{R} * H);
            for (std::uint32_t r = 0; r < R; ++r) {
                std::array<float, 4> pr{};
                std::copy_n(pre.begin() + r * ps, 4, pr.begin());
                d::hc_collapse(std::span<const float>(in.data() + std::size_t{r} * N, N), pr, std::span<float>(col.data() + std::size_t{r} * H, H));
                d::rms_norm(std::span<const float>(col.data() + std::size_t{r} * H, H), wv, 1e-20F, std::span<float>(y.data() + std::size_t{r} * H, H));
            }
            auto din = dev(in), dpre = dev(pre);
            c::DeviceBuffer<float> dcol(col.size()), dy(y.size());
            c::NormView v;
            v.hc_stream = din.get();
            v.pre = dpre.get();
            v.pre_stride = ps;
            v.weight = f.device_f32(wname);
            v.rows = R;
            v.n = H;
            v.out = dy.get();
            v.collapsed = dcol.get();
            c::enqueue_norm(v, s);
            expect_bits("hc_pre " + tag, host(dcol), col);
            expect_bits("rms_norm " + tag, host(dy), y);
            return y;
        };
        auto q8_site = [&](const std::string& name, std::uint32_t width, std::uint32_t count, std::uint32_t per_group,
                           const std::vector<float>& x, std::uint32_t x_width, const std::string& tag) {
            const auto w = f.host(name);
            const auto want = host_q8(w, width, count, per_group, x, R, x_width);
            auto dx = dev(x);
            c::DeviceBuffer<float> dy(want.size());
            c::Q8A8View v;
            v.weight = {f.device(name), count, width};
            v.count = count;
            v.per_group = per_group;
            v.x = a8_of(dx.get(), R, x_width, ws, s);
            v.out = dy.get();
            c::enqueue_q8_0_a8(v, s);
            expect_bits(tag, host(dy), want);
            f.release(name);
            return want;
        };

        const auto mix_attn = mixes("attn", stream);
        const auto x = collapse_norm(stream, pre_carry, 4, p + "attn_norm.weight", "attn");
        auto qr = q8_site(p + "attn_q_a.weight", H, QL, 0, x, H, "q_a");
        {  // q_a_norm in place
            const auto wv = f.host_as<float>(p + "attn_q_a_norm.weight");
            auto dq = dev(qr);
            c::NormView v;
            v.x = dq.get();
            v.weight = f.device_f32(p + "attn_q_a_norm.weight");
            v.rows = R;
            v.n = QL;
            v.out = dq.get();
            c::enqueue_norm(v, s);
            for (std::uint32_t r = 0; r < R; ++r) {
                std::vector<float> in(qr.begin() + r * QL, qr.begin() + (r + 1) * QL);
                d::rms_norm(in, wv, 1e-20F, std::span<float>(qr.data() + std::size_t{r} * QL, QL));
            }
            expect_bits("rms_norm q (in place)", host(dq), qr);
        }
        (void)q8_site(p + "attn_q_b.weight", QL, QO, 0, qr, QL, "q_b");
        (void)q8_site(p + "attn_kv.weight", H, HD, 0, x, H, "kv");
        // stand-in attention output o (the attention piece's), then o_a (grouped) and o_b (K-split 2)
        const auto o = random_floats(std::size_t{R} * QO, 0.05F);
        const auto olow = q8_site(p + "attn_output_a.weight", QO / cfg.o_groups, OL, cfg.o_lora_rank, o, QO, "o_a (grouped 8)");
        std::vector<float> y(std::size_t{R} * H);
        {
            const auto w = f.host(p + "attn_output_b.weight");
            for (std::uint32_t r = 0; r < R; ++r)
                d::q8_0_a8_split_rows(w.data(), OL, H, 2, d::quantize_a8(std::span<const float>(olow.data() + std::size_t{r} * OL, OL)),
                                      y.data() + std::size_t{r} * H);
            auto dx = dev(olow);
            c::DeviceBuffer<float> dy(y.size());
            const auto a = a8_of(dx.get(), R, OL, ws, s);
            for (std::uint32_t h = 0; h < 2; ++h) {
                c::Q8A8View v;
                v.weight = {f.device(p + "attn_output_b.weight"), H, OL};
                v.count = H;
                v.k_offset = h * OL / 2U;
                v.width = OL / 2U;
                v.x = {a.codes + h * OL / 2U, a.deltas + h * OL / 64U, a.sums + h * OL / 64U, R, OL / 2U, OL};
                v.out = dy.get();
                v.accumulate = h > 0;
                c::enqueue_q8_0_a8(v, s);
            }
            expect_bits("o_b (K-split 2)", host(dy), y);
            f.release(p + "attn_output_b.weight");
        }
        auto expand = [&](const std::vector<float>& sub, const std::vector<float>& mix, const std::string& tag) {
            std::vector<float> out(stream.size());
            for (std::uint32_t r = 0; r < R; ++r) {
                d::HcMix m;
                std::copy_n(mix.begin() + r * 24, 4, m.pre.begin());
                std::copy_n(mix.begin() + r * 24 + 4, 4, m.post.begin());
                std::copy_n(mix.begin() + r * 24 + 8, 16, m.comb.begin());
                d::hc_expand(std::span<const float>(sub.data() + std::size_t{r} * H, H), std::span<const float>(stream.data() + std::size_t{r} * N, N),
                             m, std::span<float>(out.data() + std::size_t{r} * N, N));
            }
            auto dsub = dev(sub), ds = dev(stream), dm = dev(mix);
            c::enqueue_hc_expand({dsub.get(), 0, ds.get(), dm.get(), R, H, ds.get()}, s);
            expect_bits("hc_post " + tag, host(ds), out);
            stream = out;
        };
        expand(y, mix_attn, "attn");
        // ---- FFN sublayer ----
        const auto mix_ffn = mixes("ffn", stream);
        const auto xf = collapse_norm(stream, mix_attn, 24, p + "ffn_norm.weight", "ffn");
        {  // router
            const auto wr = f.host_as<float>(p + "ffn_gate_inp.weight");
            const auto b = f.host_as<float>(p + "exp_probs_b.bias"), bi = f.host_as<float>(p + "exp_probs_b_vl.bias");
            const std::vector<std::uint8_t> image(R, 0);
            const auto ref = d::reference_router(xf, R, H, wr, E, b, bi, image, K, static_cast<float>(cfg.routed_scaling_factor));
            auto dx = dev(xf);
            c::DeviceBuffer<float> sc(std::size_t{R} * E), wts(std::size_t{R} * K);
            c::DeviceBuffer<std::uint32_t> ids(std::size_t{R} * K);
            c::RouterView v;
            v.x = dx.get();
            v.rows = R;
            v.weight = {f.device_f32(p + "ffn_gate_inp.weight"), E, H};
            v.bias = f.device_f32(p + "exp_probs_b.bias");
            v.bias_image = f.device_f32(p + "exp_probs_b_vl.bias");
            v.top_k = K;
            v.route_scale = static_cast<float>(cfg.routed_scaling_factor);
            v.scores = sc.get();
            v.ids = ids.get();
            v.weights = wts.get();
            c::enqueue_router(v, s);
            expect_bits("router scores", host(sc), ref.scores);
            const auto gid = ids.download();
            if (gid != ref.ids) { ++failures; std::printf("FAIL router ids\n"); }
            else std::printf("ok   router ids                   %u..: %u %u %u %u %u %u\n", R, gid[0], gid[1], gid[2], gid[3], gid[4], gid[5]);
            expect_bits("router weights", host(wts), ref.weights);
        }
        std::vector<float> shared(std::size_t{R} * H);
        {  // shared expert
            d::ExpertTensors t;
            t.gate_up_format = t.down_format = d::WeightFormat::q8_0;
            const auto g = f.host(p + "ffn_gate_shexp.weight"), u = f.host(p + "ffn_up_shexp.weight"), dn = f.host(p + "ffn_down_shexp.weight");
            t.gate = g;
            t.up = u;
            t.down = dn;
            t.hidden = H;
            t.inter = I;
            t.limit = static_cast<float>(cfg.swiglu_limit);
            for (std::uint32_t r = 0; r < R; ++r) {
                const std::vector<d::A8Activation> xs{d::quantize_a8(std::span<const float>(xf.data() + std::size_t{r} * H, H))};
                const std::vector<std::uint32_t> zero{0U};
                const std::vector<float> one{1.0F}, none(H, 0.0F);
                const auto raw = d::reference_window(t, xs, d::build_window_routes(zero, 1U, 1U, 1U), one, none, 1U).raw;
                std::copy(raw.begin(), raw.end(), shared.begin() + std::size_t{r} * H);
            }
            auto dx = dev(xf);
            c::DeviceBuffer<float> dy(shared.size());
            c::SharedExpertView v;
            v.gate = {f.device(p + "ffn_gate_shexp.weight"), I, H};
            v.up = {f.device(p + "ffn_up_shexp.weight"), I, H};
            v.down = {f.device(p + "ffn_down_shexp.weight"), H, I};
            v.limit = t.limit;
            v.x = a8_of(dx.get(), R, H, ws, s);
            v.out = dy.get();
            v.workspace = ws.take_bytes(c::shared_expert_workspace_bytes(R, I));
            c::enqueue_shared_expert(v, s);
            expect_bits("shared expert", host(dy), shared);
        }
        // stand-in combine: y = shared + a small routed term (the MoE piece's)
        auto routed = random_floats(shared.size(), 0.01F);
        for (std::size_t i = 0; i < routed.size(); ++i) routed[i] = shared[i] + routed[i];
        expand(routed, mix_ffn, "ffn");
        for (std::uint32_t r = 0; r < R; ++r) std::copy_n(mix_ffn.begin() + r * 24, 4, pre_carry.begin() + r * 4);
        ws.reset();
    }

    // ---- head: final collapse + output_norm + A8 + output (Q8_0, whole vocabulary) + greedy selection ----
    std::printf("-- head\n");
    {
        const auto wn = f.host_as<float>("output_norm.weight");
        std::vector<float> h(std::size_t{R} * H);
        for (std::uint32_t r = 0; r < R; ++r) {
            std::array<float, 4> pr{};
            std::copy_n(pre_carry.begin() + r * 4, 4, pr.begin());
            std::vector<float> col(H);
            d::hc_collapse(std::span<const float>(stream.data() + std::size_t{r} * N, N), pr, col);
            d::rms_norm(col, wn, 1e-20F, std::span<float>(h.data() + std::size_t{r} * H, H));
        }
        const auto w = f.host("output.weight");
        const auto logits = host_q8(w, H, V, 0, h, R, H);
        auto ds = dev(stream), dp = dev(pre_carry);
        c::DeviceBuffer<float> dl(logits.size());
        c::DeviceBuffer<d::ArgmaxCandidate> cand(R);
        c::A8Rows a{ws.take<std::int8_t>(std::size_t{R} * H), ws.take<float>(std::size_t{R} * H / 32U), ws.take<std::int32_t>(std::size_t{R} * H / 32U),
                    R, H, H};
        c::NormView v;
        v.hc_stream = ds.get();
        v.pre = dp.get();
        v.pre_stride = 4;
        v.weight = f.device_f32("output_norm.weight");
        v.rows = R;
        v.n = H;
        v.a8 = a;
        c::enqueue_norm(v, s);
        c::Q8A8View q;
        q.weight = {f.device("output.weight"), V, H};
        q.count = V;
        q.x = a;
        q.out = dl.get();
        c::enqueue_q8_0_a8(q, s);
        c::enqueue_argmax({dl.get(), R, V, 0, 0, cand.get()}, s);
        expect_bits("head logits", host(dl), logits);
        const auto got = cand.download();
        for (std::uint32_t r = 0; r < R; ++r) {
            const auto s0 = d::vocab_slice(V, 2, 0), s1 = d::vocab_slice(V, 2, 1);
            const float* l = logits.data() + std::size_t{r} * V;
            const auto want = d::combine_argmax(d::argmax_slice(std::span<const float>(l + s0.first, s0.count), s0.first),
                                                d::argmax_slice(std::span<const float>(l + s1.first, s1.count), s1.first));
            if (got[r].index != want.index || got[r].finite != want.finite) {
                ++failures;
                std::printf("FAIL greedy row %u: %u vs %u\n", r, got[r].index, want.index);
            } else {
                std::printf("ok   greedy row %u -> %u\n", r, want.index);
            }
        }
    }
    if (failures) {
        std::printf("%d check(s) FAILED\n", failures);
        return 1;
    }
    std::printf("ds41 cuda core real-weight tests passed\n");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
}
