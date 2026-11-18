// Timing of the deepseek41 CUDA layer core on real weights (not a test; seconds). One layer's core sites in decode
// order at R rows (attention and routed experts excluded: other pieces), each timed by events inside the sequence
// (weights stream from memory: the layer's weights exceed the L2), the whole sequence as one CUDA graph, and the head.
//   STRIX_DS41_FIXTURE_GGUF=ds41-dense-0-1-2-20.gguf strix-cuda-ds41-core-bench [rows] [layer] [iterations] [tiled]
// The L2 is rewritten (256 MB) before every timed iteration; "tiled": the tensor-core Q8_0 form.

#include "strix/cuda/ds41_common.hpp"
#include "strix/cuda/ds41_core.hpp"
#include "strix/cuda/ds41_weights.hpp"

#include "strix/ds41_layer.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

namespace c = strix::ds41::cuda;
namespace d = strix::ds41;

int main(int argc, char** argv) try {
    const char* path = std::getenv("STRIX_DS41_FIXTURE_GGUF");
    if (!path) throw std::runtime_error("set STRIX_DS41_FIXTURE_GGUF");
    const std::uint32_t R = argc > 1 ? static_cast<std::uint32_t>(std::atoi(argv[1])) : 1U;
    const std::uint32_t layer = argc > 2 ? static_cast<std::uint32_t>(std::atoi(argv[2])) : 1U;
    const int iters = argc > 3 ? std::atoi(argv[3]) : 30;
    const auto devinfo = c::select_device(0);
    c::TensorFile f(path);
    const auto cfg = f.config();
    const std::uint32_t H = cfg.hidden_size, N = 4U * H, V = cfg.vocab_size, I = cfg.moe_intermediate_size;
    const std::uint32_t E = cfg.n_routed_experts, K = cfg.num_experts_per_tok, QL = cfg.q_lora_rank, HD = cfg.head_dim;
    const std::uint32_t QO = cfg.num_attention_heads * HD, OL = cfg.o_groups * cfg.o_lora_rank;
    const std::string p = "blk." + std::to_string(layer) + ".";
    const bool engram = layer == 1 || layer == 14;
    c::Stream st;
    const auto s = st.get();
    c::Workspace ws(256ULL << 20);
    std::mt19937 rng(7);
    std::normal_distribution<float> g(0.0F, 1.0F);
    auto rnd = [&](std::size_t n, float sc) {
        std::vector<float> v(n);
        for (auto& x : v) x = g(rng) * sc;
        return v;
    };
    auto put = [&](const std::vector<float>& v) {
        float* d = ws.take<float>(v.size());
        STRIX_CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * 4U, cudaMemcpyHostToDevice));
        return d;
    };
    auto a8 = [&](std::uint32_t width) {
        return c::A8Rows{ws.take<std::int8_t>(std::size_t{R} * width), ws.take<float>(std::size_t{R} * width / 32U),
                         ws.take<std::int32_t>(std::size_t{R} * width / 32U), R, width, width};
    };
    // buffers
    float* stream = put(rnd(std::size_t{R} * N, 1.0F));
    float* pre_carry = put(std::vector<float>(std::size_t{R} * 4U, 0.25F));
    float* mix_attn = ws.take<float>(std::size_t{R} * 24U);
    float* mix_ffn = ws.take<float>(std::size_t{R} * 24U);
    float* x = ws.take<float>(std::size_t{R} * H);
    float* qr = ws.take<float>(std::size_t{R} * QL);
    float* q = ws.take<float>(std::size_t{R} * QO);
    float* kvrow = ws.take<float>(std::size_t{R} * HD);
    float* o = put(rnd(std::size_t{R} * QO, 0.05F));
    float* olow = ws.take<float>(std::size_t{R} * OL);
    float* y = ws.take<float>(std::size_t{R} * H);
    float* shared = ws.take<float>(std::size_t{R} * H);
    float* partial = static_cast<float*>(ws.take_bytes(c::hc_mix_workspace_bytes(R, H)));
    const auto xa = a8(H), qa = a8(QL), oa = a8(QO), la = a8(OL);
    float* scores = ws.take<float>(std::size_t{R} * E);
    auto* ids = ws.take<std::uint32_t>(std::size_t{R} * K);
    float* rw = ws.take<float>(std::size_t{R} * K);
    void* shw = ws.take_bytes(c::shared_expert_workspace_bytes(R, I));
    auto* erows = ws.take<std::uint8_t>(std::size_t{R} * 24U * 264U);
    STRIX_CUDA_CHECK(cudaMemset(erows, 0x38, std::size_t{R} * 24U * 264U));
    void* ew = ws.take_bytes(c::engram_workspace_bytes(R, 24, H));
    float* logits = ws.take<float>(std::size_t{R} * V);
    auto* cand = ws.take<d::ArgmaxCandidate>(R);
    std::printf("device %s; layer %u, %u rows, %d iterations (shared GPU: numbers under whatever else runs)\n", devinfo.name.c_str(), layer,
                R, iters);

    struct Op {
        std::string name;
        double bytes;
        std::function<void()> run;
    };
    std::vector<Op> ops;
    // tensor-core mode: every Q8_0 site from repacked tiles, q_a + kv and shared gate + up as one site each
    const bool tiled_mode = argc > 4 && std::string(argv[4]) == "tiled";
    std::map<std::string, c::Q8TiledBuffer> tiled;
    c::DeviceBuffer<float> split_scratch(4U << 20);
    auto tile_of = [&](const std::vector<std::string>& names, std::uint32_t rows, std::uint32_t width) {
        std::string key;
        for (const auto& n : names) key += n + "+";
        auto it = tiled.find(key);
        if (it != tiled.end()) return it->second.view();
        std::size_t total = 0;
        for (const auto& n : names) total += f.tensor(n).byte_size;
        c::DeviceBuffer<std::uint8_t> cat(total);
        std::size_t at = 0;
        for (const auto& n : names) {
            STRIX_CUDA_CHECK(cudaMemcpy(cat.get() + at, f.device(n), f.tensor(n).byte_size, cudaMemcpyDeviceToDevice));
            at += f.tensor(n).byte_size;
            f.release(n);
        }
        auto& b = tiled[key];
        b.repack(cat.get(), rows, width, s);
        st.sync();
        return b.view();
    };
    float* qkv = ws.take<float>(std::size_t{R} * (QL + HD));
    auto q8 = [&](const std::string& w, std::uint32_t rows, std::uint32_t width, c::A8Rows xin, float* out, std::uint32_t per_group = 0) {
        c::Q8A8View v;
        if (tiled_mode) {
            v.tiled = tile_of({w}, rows, width);
            v.split_scratch = split_scratch.get();
            v.split_scratch_floats = split_scratch.size();
        } else {
            v.weight = {f.device(w), rows, width};
        }
        v.count = rows;
        v.per_group = per_group;
        v.x = xin;
        v.out = out;
        c::enqueue_q8_0_a8(v, s);
    };
    auto mixes = [&](const std::string& site, float* out) {
        c::HcMixView v;
        v.stream = stream;
        v.rows = R;
        v.hidden = H;
        v.fn = f.device_f16(p + "hc_" + site + "_fn.weight");
        v.base = f.device_f32(p + "hc_" + site + "_base.weight");
        v.scale = f.device_f32(p + "hc_" + site + "_scale.weight");
        v.iters = cfg.hc_sinkhorn_iters;
        v.hc_eps = static_cast<float>(cfg.hc_eps);
        v.mixes = out;
        v.partial = partial;
        c::enqueue_hc_mixes(v, s);
    };
    auto site_norm = [&](const float* pre, std::uint32_t ps, const std::string& w) {
        c::NormView v;
        v.hc_stream = stream;
        v.pre = pre;
        v.pre_stride = ps;
        v.weight = f.device_f32(w);
        v.rows = R;
        v.n = H;
        v.out = x;
        v.a8 = xa;
        c::enqueue_norm(v, s);
    };
    const double q8b = 34.0 / 32.0;
    if (engram)
        ops.push_back({"engram (decode+kv F16+inject)", 2.0 * 5.0 * H * 24.0 * 256.0, [&] {
                           c::EngramView v;
                           v.row_bytes = erows;
                           v.rows = R;
                           v.hidden = H;
                           v.kv = {f.device_f16(p + "engram_kv.weight"), 5U * H, 24U * 256U};
                           v.q_norm = f.device_f32(p + "engram_q_norm.weight");
                           v.k_norm = f.device_f32(p + "engram_k_norm.weight");
                           v.stream = stream;
                           v.workspace = ew;
                           c::enqueue_engram(v, s);
                       }});
    ops.push_back({"hc_mix attn", 2.0 * 24.0 * N, [&] { mixes("attn", mix_attn); }});
    ops.push_back({"collapse+attn_norm+A8", 0, [&] { site_norm(pre_carry, 4, p + "attn_norm.weight"); }});
    if (tiled_mode) {
        ops.push_back({"q_a+kv (one site)", q8b * H * (QL + HD), [&] {
                           c::Q8A8View v;
                           v.tiled = tile_of({p + "attn_q_a.weight", p + "attn_kv.weight"}, QL + HD, H);
                           v.split_scratch = split_scratch.get();
                           v.split_scratch_floats = split_scratch.size();
                           v.count = QL + HD;
                           v.x = xa;
                           v.out = qkv;
                           c::enqueue_q8_0_a8(v, s);
                       }});
        ops.push_back({"q_a_norm+A8 (2 blocks)", 0, [&] {
                           c::NormView v;
                           v.x = qkv;
                           v.x_stride = QL + HD;
                           v.weight = f.device_f32(p + "attn_q_a_norm.weight");
                           v.rows = R;
                           v.n = QL;
                           v.out = qr;
                           v.a8 = qa;
                           c::enqueue_norm(v, s);
                       }});
    }
    if (!tiled_mode) ops.push_back({"q_a", q8b * H * QL, [&] { q8(p + "attn_q_a.weight", QL, H, xa, qr); }});
    if (!tiled_mode) ops.push_back({"q_a_norm+A8", 0, [&] {
                       c::NormView v;
                       v.x = qr;
                       v.weight = f.device_f32(p + "attn_q_a_norm.weight");
                       v.rows = R;
                       v.n = QL;
                       v.out = qr;
                       v.a8 = qa;
                       c::enqueue_norm(v, s);
                   }});
    ops.push_back({"q_b", q8b * QL * QO, [&] { q8(p + "attn_q_b.weight", QO, QL, qa, q); }});
    if (!tiled_mode) ops.push_back({"kv", q8b * H * HD, [&] { q8(p + "attn_kv.weight", HD, H, xa, kvrow); }});
    ops.push_back({"A8(o)+o_a", q8b * QO / cfg.o_groups * OL, [&] {
                       c::enqueue_quantize_a8({o, R, QO, 0, oa}, s);
                       q8(p + "attn_output_a.weight", OL, QO / cfg.o_groups, oa, olow, cfg.o_lora_rank);
                   }});
    ops.push_back({"A8+o_b (2 halves)", q8b * OL * H, [&] {
                       c::enqueue_quantize_a8({olow, R, OL, 0, la}, s);
                       for (std::uint32_t h = 0; h < 2; ++h) {
                           c::Q8A8View v;
                           if (tiled_mode) {
                               v.tiled = tile_of({p + "attn_output_b.weight"}, H, OL);
                               v.split_scratch = split_scratch.get();
                               v.split_scratch_floats = split_scratch.size();
                           } else {
                               v.weight = {f.device(p + "attn_output_b.weight"), H, OL};
                           }
                           v.count = H;
                           v.k_offset = h * OL / 2U;
                           v.width = OL / 2U;
                           v.x = {la.codes + h * OL / 2U, la.deltas + h * OL / 64U, la.sums + h * OL / 64U, R, OL / 2U, OL};
                           v.out = y;
                           v.accumulate = h > 0;
                           c::enqueue_q8_0_a8(v, s);
                       }
                   }});
    ops.push_back({"hc_post attn", 0, [&] { c::enqueue_hc_expand({y, 0, stream, mix_attn, R, H, stream}, s); }});
    ops.push_back({"hc_mix ffn", 2.0 * 24.0 * N, [&] { mixes("ffn", mix_ffn); }});
    ops.push_back({"collapse+ffn_norm+A8", 0, [&] { site_norm(mix_attn, 24, p + "ffn_norm.weight"); }});
    ops.push_back({"router", 4.0 * E * H, [&] {
                       c::RouterView v;
                       v.x = x;
                       v.rows = R;
                       v.weight = {f.device_f32(p + "ffn_gate_inp.weight"), E, H};
                       v.bias = f.device_f32(p + "exp_probs_b.bias");
                       v.bias_image = f.device_f32(p + "exp_probs_b_vl.bias");
                       v.top_k = K;
                       v.route_scale = static_cast<float>(cfg.routed_scaling_factor);
                       v.scores = scores;
                       v.ids = ids;
                       v.weights = rw;
                       c::enqueue_router(v, s);
                   }});
    ops.push_back({"shared expert", q8b * 3.0 * H * I, [&] {
                       c::SharedExpertView v;
                       if (tiled_mode) {
                           v.gate_up_tiled = tile_of({p + "ffn_gate_shexp.weight", p + "ffn_up_shexp.weight"}, 2U * I, H);
                           v.down_tiled = tile_of({p + "ffn_down_shexp.weight"}, H, I);
                       } else {
                           v.gate = {f.device(p + "ffn_gate_shexp.weight"), I, H};
                           v.up = {f.device(p + "ffn_up_shexp.weight"), I, H};
                           v.down = {f.device(p + "ffn_down_shexp.weight"), H, I};
                       }
                       v.limit = static_cast<float>(cfg.swiglu_limit);
                       v.x = xa;
                       v.out = shared;
                       v.workspace = shw;
                       c::enqueue_shared_expert(v, s);
                   }});
    ops.push_back({"hc_post ffn", 0, [&] { c::enqueue_hc_expand({shared, 0, stream, mix_ffn, R, H, stream}, s); }});
    Op head{"head (norm+A8+output+argmax)", q8b * H * V, [&] {
                c::NormView v;
                v.hc_stream = stream;
                v.pre = mix_ffn;
                v.pre_stride = 24;
                v.weight = f.device_f32("output_norm.weight");
                v.rows = R;
                v.n = H;
                v.a8 = xa;
                c::enqueue_norm(v, s);
                q8("output.weight", V, H, xa, logits);
                c::enqueue_argmax({logits, R, V, 0, 0, cand}, s);
            }};
    // load every weight once (first run), then warm up
    for (auto& op : ops) op.run();
    head.run();
    st.sync();
    std::vector<cudaEvent_t> ev(ops.size() + 1);
    for (auto& e : ev) STRIX_CUDA_CHECK(cudaEventCreate(&e));
    std::vector<std::vector<float>> t(ops.size());
    c::DeviceBuffer<std::uint8_t> flush(256U << 20);  // rewritten before every timed iteration: no weight stays in the L2
    for (int it = 0; it < iters; ++it) {
        STRIX_CUDA_CHECK(cudaMemsetAsync(flush.get(), it & 0xff, flush.size(), s));
        STRIX_CUDA_CHECK(cudaEventRecord(ev[0], s));
        for (std::size_t i = 0; i < ops.size(); ++i) {
            ops[i].run();
            STRIX_CUDA_CHECK(cudaEventRecord(ev[i + 1], s));
        }
        STRIX_CUDA_CHECK(cudaEventSynchronize(ev.back()));
        for (std::size_t i = 0; i < ops.size(); ++i) {
            float ms = 0;
            STRIX_CUDA_CHECK(cudaEventElapsedTime(&ms, ev[i], ev[i + 1]));
            t[i].push_back(ms);
        }
    }
    auto median = [](std::vector<float> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    double sum = 0;
    std::printf("%-30s %9s %9s\n", "op (in sequence)", "us", "GB/s");
    for (std::size_t i = 0; i < ops.size(); ++i) {
        const double us = 1000.0 * median(t[i]);
        sum += us;
        if (ops[i].bytes > 0) std::printf("%-30s %9.1f %9.0f\n", ops[i].name.c_str(), us, ops[i].bytes / us / 1000.0);
        else std::printf("%-30s %9.1f %9s\n", ops[i].name.c_str(), us, "-");
    }
    std::printf("%-30s %9.1f\n", "sum of ops (stream launches)", sum);
    // the whole sequence as a graph
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    STRIX_CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    for (auto& op : ops) op.run();
    STRIX_CUDA_CHECK(cudaStreamEndCapture(s, &graph));
    STRIX_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    c::EventTimer timer;
    std::vector<float> gt, ht;
    for (int it = 0; it < iters; ++it) {
        STRIX_CUDA_CHECK(cudaMemsetAsync(flush.get(), it & 0xff, flush.size(), s));
        timer.start(s);
        STRIX_CUDA_CHECK(cudaGraphLaunch(exec, s));
        gt.push_back(timer.stop(s));
    }
    std::printf("%-30s %9.1f\n", "layer core as one graph", 1000.0 * median(gt));
    for (int it = 0; it < iters; ++it) {
        STRIX_CUDA_CHECK(cudaMemsetAsync(flush.get(), it & 0xff, flush.size(), s));
        timer.start(s);
        head.run();
        ht.push_back(timer.stop(s));
    }
    std::printf("%-30s %9.1f %9.0f\n", head.name.c_str(), 1000.0 * median(ht), head.bytes / (1000.0 * median(ht)) / 1000.0);
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    for (auto& e : ev) cudaEventDestroy(e);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
}
