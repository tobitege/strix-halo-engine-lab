// Replays a host-forward capture (strix-ds41-tap-capture) on the CUDA layer core and compares every op it implements.
//   strix-cuda-ds41-core-tap-check WEIGHTS.gguf CAPTURE.tap [--chain] [--steps N]
// WEIGHTS.gguf holds (at least) the tensors of the captured layers, the embedding and the head under their model
// names (the full model or a subset). Each op runs on the device with its inputs taken from the captured buffers
// (isolated: the reference's inputs), its outputs are compared bit for bit with the captured buffers, or by op hash for
// the logits, the greedy token and the route ids. With --chain the core ops feed each other inside the step and only
// the ops of the other pieces (attention, routed experts) are taken from the capture. With --layer-core the steps run
// through LayerCore (the assembly's API: tiled weights, fused sites) with only the attention output o and the combined
// y taken from the capture, and every op output is compared through the LayerCore tap. Exit status 1 on any difference.

#include "strix/cuda/ds41_common.hpp"
#include "strix/cuda/ds41_core.hpp"
#include "strix/cuda/ds41_layer_core.hpp"
#include "strix/cuda/ds41_weights.hpp"

#include "strix/deepseek41_model.hpp"
#include "strix/deepseek41_plan.hpp"
#include "strix/ds41_layer.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace c = strix::ds41::cuda;
namespace d = strix::ds41;
namespace m = strix::deepseek41;

namespace {

// ---- capture file ----
struct StepCapture {
    std::uint32_t token = 0, position = 0;
    std::map<std::uint32_t, std::vector<std::uint8_t>> engram_rows;  // layer -> [column][264]
    std::vector<std::pair<std::string, std::vector<float>>> buffers;
    std::vector<std::pair<std::string, std::uint64_t>> hashes;
};
struct Capture {
    std::uint32_t layers = 0;
    std::vector<StepCapture> steps;
};
Capture read_capture(const std::string& path, std::uint32_t max_steps) {
    std::ifstream in(path, std::ios::binary);
    char magic[8];
    in.read(magic, 8);
    if (!in || std::memcmp(magic, "DS41TAP1", 8) != 0) throw std::runtime_error("not a DS41TAP1 capture: " + path);
    const auto u32 = [&] { std::uint32_t v = 0; in.read(reinterpret_cast<char*>(&v), 4); return v; };
    const auto u64 = [&] { std::uint64_t v = 0; in.read(reinterpret_cast<char*>(&v), 8); return v; };
    const auto label = [&] { std::string s(u32(), '\0'); in.read(s.data(), static_cast<std::streamsize>(s.size())); return s; };
    Capture cap;
    cap.layers = u32();
    for (;;) {
        const auto kind = u32();
        if (!in) throw std::runtime_error("truncated capture");
        if (kind == 0) break;
        if (kind == 1) {
            if (cap.steps.size() == max_steps) break;
            cap.steps.emplace_back();
            cap.steps.back().token = u32();
            cap.steps.back().position = u32();
        } else if (kind == 2) {
            const auto layer = u32(), cols = u32();
            std::vector<std::uint8_t> b(std::size_t{cols} * 264U);
            in.read(reinterpret_cast<char*>(b.data()), static_cast<std::streamsize>(b.size()));
            cap.steps.back().engram_rows[layer] = std::move(b);
        } else if (kind == 3) {
            auto l = label();
            std::vector<float> v(u64());
            in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(v.size() * 4U));
            cap.steps.back().buffers.emplace_back(std::move(l), std::move(v));
        } else if (kind == 4) {
            const auto n = u32();
            for (std::uint32_t i = 0; i < n; ++i) {
                auto l = label();
                cap.steps.back().hashes.emplace_back(std::move(l), u64());
            }
        } else {
            throw std::runtime_error("unknown capture record");
        }
    }
    return cap;
}

std::string label_of(const m::Step& s, m::Buffer b) {
    return std::string(m::to_string(s.op)) + (s.layer >= 0 ? "@" + std::to_string(s.layer) : "") + (s.site.empty() ? "" : "/" + s.site) +
           ":" + std::string(m::to_string(b));
}

struct Tally {
    std::uint32_t equal = 0, different = 0, skipped = 0;
};

}  // namespace


// LayerCore run of the capture: chained core ops, attention o and combine y from the capture, every tap compared.
int run_layer_core(c::TensorFile& w, const Capture& cap, const m::Config& cfg) {
    c::Stream st;
    const auto s = st.get();
    c::LayerCore core(w, 0, cap.layers, true, true, 1, s);
    const char* fuse = std::getenv("STRIX_DS41_CUDA_TAP_FUSE");
    if (fuse && std::string(fuse) == "1") core.set_decode_fusions(true, true);
    std::uint32_t failures = 0, equal = 0;
    const StepCapture* step = nullptr;
    std::map<std::string, const std::vector<float>*> want;
    std::map<std::string, std::uint64_t> want_hash;
    core.set_tap([&](const std::string& label, const float* dev, std::uint32_t rows, std::uint32_t count, std::uint32_t stride) {
        std::vector<float> v(std::size_t{rows} * count);
        STRIX_CUDA_CHECK(cudaMemcpy2DAsync(v.data(), count * 4U, dev, stride * 4U, count * 4U, rows, cudaMemcpyDeviceToHost, s));
        st.sync();
        bool same = false;
        if (label.ends_with(":route_ids")) {
            std::uint64_t h = 0;
            for (const float f : v) h = h * 1099511628211ULL + std::bit_cast<std::uint32_t>(f);
            same = want_hash.count(label) && want_hash.at(label) == h;
        } else if (label == "head:logits") {
            same = want_hash.count(label) && want_hash.at(label) == c::fnv1a(v.data(), v.size());
        } else if (label == "argmax:token") {
            same = want_hash.count(label) && want_hash.at(label) == std::bit_cast<std::uint32_t>(v[1]);
        } else {
            const auto it = want.find(label);
            if (it == want.end()) {
                std::printf("step %u: no captured buffer %s\n", step->position, label.c_str());
                ++failures;
                return;
            }
            same = it->second->size() == v.size() && std::memcmp(it->second->data(), v.data(), v.size() * 4U) == 0;
        }
        if (same) ++equal;
        else { ++failures; std::printf("DIFF step %u %s\n", step->position, label.c_str()); }
    });
    for (const auto& st_cap : cap.steps) {
        step = &st_cap;
        want.clear();
        want_hash.clear();
        for (const auto& [l, v] : st_cap.buffers) want[l] = &v;
        for (const auto& [l, h] : st_cap.hashes) want_hash[l] = h;
        core.embed(&st_cap.token, s);
        for (std::uint32_t l = 0; l < cap.layers; ++l) {
            const auto er = st_cap.engram_rows.find(l);
            core.attn_input(l, er == st_cap.engram_rows.end() ? nullptr : er->second.data(), s);
            core.attn_q(l, s);
            const auto& o = *want.at("attention@" + std::to_string(l) + ":o");
            STRIX_CUDA_CHECK(cudaMemcpyAsync(core.buffers().o.get(), o.data(), o.size() * 4U, cudaMemcpyHostToDevice, s));
            core.attn_output(l, s);
            core.ffn_input(l, s);
            core.ffn_core(l, s);
            const auto& y = *want.at("combine@" + std::to_string(l) + ":y");
            STRIX_CUDA_CHECK(cudaMemcpyAsync(core.buffers().y.get(), y.data(), y.size() * 4U, cudaMemcpyHostToDevice, s));
            core.ffn_output(l, s);
        }
        core.head(s);
        st.sync();
        std::printf("step %u token %u: %u core outputs equal so far, %u different\n", st_cap.position, st_cap.token, equal, failures);
    }
    (void)cfg;
    std::printf("%s\n", failures ? "LAYER-CORE TAP CHECK FAILED" : "layer-core tap check: every core output bitwise equal to the host forward");
    return failures ? 1 : 0;
}

int main(int argc, char** argv) try {
    std::string weights_path, capture_path;
    bool chain = false, layer_core = false;
    std::uint32_t max_steps = 0xffffffffU;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--chain") chain = true;
        else if (a == "--layer-core") layer_core = true;
        else if (a == "--steps" && i + 1 < argc) max_steps = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        else if (weights_path.empty()) weights_path = a;
        else if (capture_path.empty()) capture_path = a;
        else throw std::invalid_argument("unknown argument " + a);
    }
    if (capture_path.empty()) throw std::invalid_argument("usage: strix-cuda-ds41-core-tap-check WEIGHTS.gguf CAPTURE.tap [--chain | --layer-core] [--steps N]");
    const auto dev = c::select_device(0);
    const auto cap = read_capture(capture_path, max_steps);
    c::TensorFile w(weights_path);
    m::TargetModel model;
    model.config = w.config();
    model.layers = m::layer_attributes(model.config);
    if (layer_core) return run_layer_core(w, cap, model.config);
    const auto plan = m::build_decode_plan(model, 0, cap.layers);
    const auto& cfg = model.config;
    const std::uint32_t H = cfg.hidden_size, heads = cfg.num_attention_heads, hd = cfg.head_dim;
    std::printf("device %s; capture: %u layers, %zu steps; %zu plan steps per token; mode %s\n", dev.name.c_str(), cap.layers,
                cap.steps.size(), plan.steps.size(), chain ? "chain" : "isolated");

    c::Stream st;
    const auto s = st.get();
    c::Workspace ws(64ULL << 20);
    c::EventTimer timer;
    std::map<m::Buffer, std::vector<float>> state;
    for (const auto b : {m::Buffer::stream, m::Buffer::mix_attn, m::Buffer::mix_ffn, m::Buffer::pre_carry, m::Buffer::x, m::Buffer::qr,
                         m::Buffer::q, m::Buffer::kvrow, m::Buffer::o, m::Buffer::o_low, m::Buffer::y, m::Buffer::shared, m::Buffer::route_w,
                         m::Buffer::raw})
        state[b] = std::vector<float>(plan.buffer_elements(b), 0.0F);
    std::map<std::string, Tally> per_op;
    std::map<std::string, std::pair<double, std::uint32_t>> op_time;  // op -> (ms, count)
    std::uint32_t failures = 0;

    for (const auto& step : cap.steps) {
        std::size_t next_buffer = 0, next_hash = 0;
        std::uint32_t greedy = 0;
        std::uint64_t logits_hash = 0, ids_hash = 0;
        for (const auto& ps : plan.steps) {
            // the captured outputs of this op
            std::map<m::Buffer, std::vector<float>> want;
            std::map<m::Buffer, std::uint64_t> want_hash;
            for (const auto b : ps.outputs) {
                const auto label = label_of(ps, b);
                if (next_hash >= step.hashes.size() || step.hashes[next_hash].first != label)
                    throw std::runtime_error("capture hashes out of step with the plan at " + label);
                want_hash[b] = step.hashes[next_hash++].second;
                if (b == m::Buffer::logits || b == m::Buffer::token || b == m::Buffer::route_ids) continue;
                if (next_buffer >= step.buffers.size() || step.buffers[next_buffer].first != label)
                    throw std::runtime_error("capture buffers out of step with the plan at " + label);
                want[b] = step.buffers[next_buffer++].second;
            }
            const std::string L = std::to_string(ps.layer), p = "blk." + L + ".";
            const std::string opname = std::string(m::to_string(ps.op)) + (ps.site.empty() ? "" : "/" + ps.site);
            std::map<m::Buffer, std::vector<float>> got;
            std::map<m::Buffer, std::uint64_t> got_hash;
            bool ran = true;
            // device copies of the inputs
            auto up = [&](m::Buffer b) {
                const auto& v = state.at(b);
                float* dp = ws.take<float>(v.size());
                STRIX_CUDA_CHECK(cudaMemcpyAsync(dp, v.data(), v.size() * 4U, cudaMemcpyHostToDevice, s));
                return dp;
            };
            auto down = [&](const float* dp, std::size_t n) {
                std::vector<float> v(n);
                STRIX_CUDA_CHECK(cudaMemcpyAsync(v.data(), dp, n * 4U, cudaMemcpyDeviceToHost, s));
                st.sync();
                return v;
            };
            auto a8 = [&](const float* x, std::uint32_t width) {
                c::A8Rows a{ws.take<std::int8_t>(width), ws.take<float>(width / 32U), ws.take<std::int32_t>(width / 32U), 1, width, width};
                c::enqueue_quantize_a8({x, 1, width, 0, a}, s);
                return a;
            };
            auto q8 = [&](const std::string& name, std::uint32_t rows, std::uint32_t width, const c::A8Rows& x, float* out,
                          std::uint32_t per_group = 0) {
                c::Q8A8View v;
                v.weight = {w.device(name), rows, width};
                v.count = rows;
                v.per_group = per_group;
                v.x = x;
                v.out = out;
                c::enqueue_q8_0_a8(v, s);
            };
            ws.reset();
            // preload the weights of the op so that the timing covers the kernels only
            for (const auto& name : ps.weights)
                if (ps.op != m::Op::engram || name.find("engram_embd") == std::string::npos) {
                    if (ps.op == m::Op::window_write || ps.op == m::Op::compress || ps.op == m::Op::index || ps.op == m::Op::attention ||
                        ps.op == m::Op::routed_experts)
                        continue;
                    (void)w.device(name);
                }
            if (ps.op == m::Op::head) (void)w.device("output.weight"), (void)w.device("output_norm.weight");
            if (ps.op == m::Op::embed) (void)w.device("token_embd.weight");
            STRIX_CUDA_CHECK(cudaDeviceSynchronize());
            float ms = 0.0F;
            switch (ps.op) {
            case m::Op::embed: {
                auto* tok = ws.take<std::uint32_t>(1);
                STRIX_CUDA_CHECK(cudaMemcpyAsync(tok, &step.token, 4, cudaMemcpyHostToDevice, s));
                float* stream = ws.take<float>(4U * H);
                float* pre = ws.take<float>(4);
                timer.start(s);
                c::enqueue_embed({{w.device_f16("token_embd.weight"), cfg.vocab_size, H}, tok, 1, stream, pre}, s);
                ms = timer.stop(s);
                got[m::Buffer::stream] = down(stream, 4U * H);
                got[m::Buffer::pre_carry] = down(pre, 4);
                break;
            }
            case m::Op::engram: {
                const auto& rows = step.engram_rows.at(static_cast<std::uint32_t>(ps.layer));
                const std::uint32_t cols = static_cast<std::uint32_t>(rows.size() / 264U);
                auto* rb = ws.take<std::uint8_t>(rows.size());
                STRIX_CUDA_CHECK(cudaMemcpyAsync(rb, rows.data(), rows.size(), cudaMemcpyHostToDevice, s));
                float* stream = up(m::Buffer::stream);
                c::EngramView v;
                v.row_bytes = rb;
                v.rows = 1;
                v.cols = cols;
                v.hidden = H;
                v.kv = {w.device_f16(p + "engram_kv.weight"), 5U * H, cols * 256U};
                v.q_norm = w.device_f32(p + "engram_q_norm.weight");
                v.k_norm = w.device_f32(p + "engram_k_norm.weight");
                v.stream = stream;
                v.workspace = ws.take_bytes(c::engram_workspace_bytes(1, cols, H));
                timer.start(s);
                c::enqueue_engram(v, s);
                ms = timer.stop(s);
                got[m::Buffer::stream] = down(stream, 4U * H);
                break;
            }
            case m::Op::hc_mix: {
                const std::string q = p + "hc_" + ps.site;
                float* stream = up(m::Buffer::stream);
                float* mix = ws.take<float>(24);
                c::HcMixView v;
                v.stream = stream;
                v.rows = 1;
                v.hidden = H;
                v.fn = w.device_f16(q + "_fn.weight");
                v.base = w.device_f32(q + "_base.weight");
                v.scale = w.device_f32(q + "_scale.weight");
                v.iters = cfg.hc_sinkhorn_iters;
                v.hc_eps = static_cast<float>(cfg.hc_eps);
                v.mixes = mix;
                v.partial = static_cast<float*>(ws.take_bytes(c::hc_mix_workspace_bytes(1, H)));
                timer.start(s);
                c::enqueue_hc_mixes(v, s);
                ms = timer.stop(s);
                got[ps.outputs[0]] = down(mix, 24);
                break;
            }
            case m::Op::hc_pre: {
                float* stream = up(m::Buffer::stream);
                const bool attn = ps.site == "attn";
                float* pre = up(attn ? m::Buffer::pre_carry : m::Buffer::mix_attn);
                float* x = ws.take<float>(H);
                c::NormView v;
                v.hc_stream = stream;
                v.pre = pre;
                v.pre_stride = attn ? 4U : 24U;
                v.rows = 1;
                v.n = H;
                v.collapsed = x;
                timer.start(s);
                c::enqueue_norm(v, s);
                ms = timer.stop(s);
                got[m::Buffer::x] = down(x, H);
                break;
            }
            case m::Op::rms_norm: {
                const auto b = ps.site == "q" ? m::Buffer::qr : m::Buffer::x;
                float* x = up(b);
                const auto n = static_cast<std::uint32_t>(state.at(b).size());
                c::NormView v;
                v.x = x;
                v.weight = w.device_f32(ps.weights[0]);
                v.rows = 1;
                v.n = n;
                v.out = x;
                timer.start(s);
                c::enqueue_norm(v, s);
                ms = timer.stop(s);
                got[b] = down(x, n);
                break;
            }
            case m::Op::q_a:
            case m::Op::kv:
            case m::Op::q_b:
            case m::Op::o_a: {
                const bool qa = ps.op == m::Op::q_a, kvop = ps.op == m::Op::kv, qb = ps.op == m::Op::q_b;
                const auto in_b = qa || kvop ? m::Buffer::x : qb ? m::Buffer::qr : m::Buffer::o;
                const auto out_b = qa ? m::Buffer::qr : kvop ? m::Buffer::kvrow : qb ? m::Buffer::q : m::Buffer::o_low;
                const auto in_n = static_cast<std::uint32_t>(state.at(in_b).size());
                const std::uint32_t out_n = qa ? cfg.q_lora_rank : kvop ? hd : qb ? heads * hd : cfg.o_groups * cfg.o_lora_rank;
                float* x = up(in_b);
                float* y = ws.take<float>(out_n);
                timer.start(s);
                const auto a = a8(x, in_n);
                if (ps.op == m::Op::o_a) q8(ps.weights[0], out_n, in_n / cfg.o_groups, a, y, cfg.o_lora_rank);
                else q8(ps.weights[0], out_n, in_n, a, y);
                ms = timer.stop(s);
                got[out_b] = down(y, out_n);
                break;
            }
            case m::Op::o_b: {
                const std::uint32_t in_n = cfg.o_groups * cfg.o_lora_rank, part = in_n / 2U;
                float* x = up(m::Buffer::o_low);
                float* y = ws.take<float>(H);
                timer.start(s);
                const auto a = a8(x, in_n);
                for (std::uint32_t h = 0; h < 2U; ++h) {
                    c::Q8A8View v;
                    v.weight = {w.device(ps.weights[0]), H, in_n};
                    v.count = H;
                    v.k_offset = h * part;
                    v.width = part;
                    v.x = {a.codes + h * part, a.deltas + h * part / 32U, a.sums + h * part / 32U, 1, part, in_n};
                    v.out = y;
                    v.accumulate = h > 0;
                    c::enqueue_q8_0_a8(v, s);
                }
                ms = timer.stop(s);
                got[m::Buffer::y] = down(y, H);
                break;
            }
            case m::Op::hc_post: {
                const bool ffn = ps.site == "ffn";
                float* y = up(m::Buffer::y);
                float* stream = up(m::Buffer::stream);
                float* mix = up(ffn ? m::Buffer::mix_ffn : m::Buffer::mix_attn);
                timer.start(s);
                c::enqueue_hc_expand({y, 0, stream, mix, 1, H, stream}, s);
                ms = timer.stop(s);
                got[m::Buffer::stream] = down(stream, 4U * H);
                if (ffn) {
                    auto pre = state.at(m::Buffer::mix_ffn);
                    pre.resize(4);
                    got[m::Buffer::pre_carry] = pre;  // a copy (the next layer's collapse reads mix_ffn's pre)
                }
                break;
            }
            case m::Op::router: {
                float* x = up(m::Buffer::x);
                const std::uint32_t E = cfg.n_routed_experts, k = cfg.num_experts_per_tok;
                c::RouterView v;
                v.x = x;
                v.rows = 1;
                v.weight = {w.device_f32(ps.weights[0]), E, H};
                v.bias = w.device_f32(ps.weights[1]);
                v.bias_image = w.device_f32(ps.weights[2]);
                v.top_k = k;
                v.route_scale = static_cast<float>(cfg.routed_scaling_factor);
                v.scores = ws.take<float>(E);
                v.ids = ws.take<std::uint32_t>(k);
                v.weights = ws.take<float>(k);
                timer.start(s);
                c::enqueue_router(v, s);
                ms = timer.stop(s);
                std::vector<std::uint32_t> ids(k);
                STRIX_CUDA_CHECK(cudaMemcpyAsync(ids.data(), v.ids, k * 4U, cudaMemcpyDeviceToHost, s));
                got[m::Buffer::route_w] = down(v.weights, k);
                std::uint64_t h = 0;
                for (const auto id : ids) h = h * 1099511628211ULL + id;
                got_hash[m::Buffer::route_ids] = h;
                ids_hash = h;
                break;
            }
            case m::Op::shared_expert: {
                float* x = up(m::Buffer::x);
                const std::uint32_t I = cfg.moe_intermediate_size;
                float* y = ws.take<float>(H);
                c::SharedExpertView v;
                v.gate = {w.device(ps.weights[0]), I, H};
                v.up = {w.device(ps.weights[1]), I, H};
                v.down = {w.device(ps.weights[2]), H, I};
                v.limit = static_cast<float>(cfg.swiglu_limit);
                v.out = y;
                v.workspace = ws.take_bytes(c::shared_expert_workspace_bytes(1, I));
                timer.start(s);
                v.x = a8(x, H);
                c::enqueue_shared_expert(v, s);
                ms = timer.stop(s);
                got[m::Buffer::shared] = down(y, H);
                break;
            }
            case m::Op::head: {
                float* stream = up(m::Buffer::stream);
                float* pre = up(m::Buffer::pre_carry);
                const std::uint32_t V = cfg.vocab_size;
                float* logits = ws.take<float>(V);
                c::A8Rows a{ws.take<std::int8_t>(H), ws.take<float>(H / 32U), ws.take<std::int32_t>(H / 32U), 1, H, H};
                auto* cand = ws.take<d::ArgmaxCandidate>(1);
                c::NormView v;
                v.hc_stream = stream;
                v.pre = pre;
                v.pre_stride = 4;
                v.weight = w.device_f32("output_norm.weight");
                v.rows = 1;
                v.n = H;
                v.a8 = a;
                timer.start(s);
                c::enqueue_norm(v, s);
                q8("output.weight", V, H, a, logits);
                c::enqueue_argmax({logits, 1, V, 0, 0, cand}, s);
                ms = timer.stop(s);
                const auto l = down(logits, V);
                logits_hash = c::fnv1a(l.data(), l.size());
                got_hash[m::Buffer::logits] = logits_hash;
                d::ArgmaxCandidate best{};
                STRIX_CUDA_CHECK(cudaMemcpyAsync(&best, cand, sizeof(best), cudaMemcpyDeviceToHost, s));
                st.sync();
                greedy = best.index;
                break;
            }
            case m::Op::argmax:
                got_hash[m::Buffer::token] = greedy;
                break;
            default:
                ran = false;
                break;
            }
            auto& t = per_op[opname];
            if (!ran) {
                ++t.skipped;
            } else {
                auto& tm = op_time[opname];
                tm.first += ms;
                tm.second += 1;
                bool same = true;
                for (const auto& [b, v] : got) {
                    const auto& ref = want.at(b);
                    const bool eq = v.size() == ref.size() && std::memcmp(v.data(), ref.data(), v.size() * 4U) == 0;
                    if (!eq) {
                        same = false;
                        std::size_t first = 0;
                        while (first < v.size() && std::memcmp(&v[first], &ref[first], 4) == 0) ++first;
                        double max_abs = 0.0;
                        for (std::size_t i = 0; i < std::min(v.size(), ref.size()); ++i)
                            max_abs = std::max(max_abs, static_cast<double>(std::fabs(v[i] - ref[i])));
                        std::printf("DIFF step %u %s: first %zu (%.9g vs %.9g), max |d| %.3g\n", step.position, label_of(ps, b).c_str(),
                                    first, first < v.size() ? v[first] : 0.0F, first < ref.size() ? ref[first] : 0.0F, max_abs);
                    }
                }
                for (const auto& [b, h] : got_hash)
                    if (h != want_hash.at(b)) {
                        same = false;
                        std::printf("DIFF step %u %s: hash %016llx vs %016llx\n", step.position, label_of(ps, b).c_str(),
                                    static_cast<unsigned long long>(h), static_cast<unsigned long long>(want_hash.at(b)));
                    }
                if (same) ++t.equal;
                else { ++t.different; ++failures; }
            }
            // advance the state: captured outputs (isolated), or the device's for the core ops (chain)
            for (auto& [b, v] : want) state[b] = (chain && ran && got.count(b)) ? got.at(b) : v;
        }
        std::printf("step %u token %u: greedy %u (logits %016llx, last route ids %016llx)\n", step.position, step.token, greedy,
                    static_cast<unsigned long long>(logits_hash), static_cast<unsigned long long>(ids_hash));
    }
    std::printf("%-22s %8s %8s %8s %10s\n", "op", "equal", "differ", "skipped", "us/op(R=1)");
    for (const auto& [op, t] : per_op) {
        const auto it = op_time.find(op);
        const double us = it != op_time.end() && it->second.second ? 1000.0 * it->second.first / it->second.second : 0.0;
        std::printf("%-22s %8u %8u %8u %10.1f\n", op.c_str(), t.equal, t.different, t.skipped, us);
    }
    std::printf("%s\n", failures ? "TAP CHECK FAILED" : "tap check: every implemented op bitwise equal to the host forward");
    return failures ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
}
