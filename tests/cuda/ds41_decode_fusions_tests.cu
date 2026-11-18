// Exact GPU-vs-GPU gates and optional graph microbench of decode launch fusions.
// STRIX_DS41_FIXTURE_GGUF=... ds41-dense-0-1-2-20.gguf; [--bench] [--overlay FILE]
#include "strix/cuda/ds41_core.hpp"
#include "strix/cuda/ds41_layer_core.hpp"
#include "strix/cuda/ds41_weights.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace c = strix::ds41::cuda;
namespace {
int failures = 0;
std::mt19937 rng(19);

std::vector<float> random_values(std::size_t n, float scale = 1.0F) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> out(n);
    for (auto& x : out) x = dist(rng);
    return out;
}

template<class T> void same(const std::string& label, const T* got, const T* want, std::size_t n) {
    std::vector<T> a(n), b(n);
    STRIX_CUDA_CHECK(cudaDeviceSynchronize());
    STRIX_CUDA_CHECK(cudaMemcpy(a.data(), got, n * sizeof(T), cudaMemcpyDeviceToHost));
    STRIX_CUDA_CHECK(cudaMemcpy(b.data(), want, n * sizeof(T), cudaMemcpyDeviceToHost));
    std::size_t bad = 0;
    for (std::size_t i = 0; i < n; ++i) bad += std::memcmp(&a[i], &b[i], sizeof(T)) != 0;
    // Byte digest, including codes and integer sums (fnv1a(float*) is a word digest).
    const auto digest = [](const std::vector<T>& v) {
        std::uint64_t h = 14695981039346656037ULL;
        const auto* bytes = reinterpret_cast<const unsigned char*>(v.data());
        for (std::size_t i = 0; i < v.size() * sizeof(T); ++i) h = (h ^ bytes[i]) * 1099511628211ULL;
        return static_cast<unsigned long long>(h);
    };
    std::printf("%s %s values=%zu differ=%zu legacy=%016llx fused=%016llx\n",
                bad ? "FAIL" : "PASS", label.c_str(), n, bad, digest(b), digest(a));
    failures += bad != 0;
}

void same_a8(const std::string& label, c::A8Rows a, c::A8Rows b) {
    same(label + "/codes", a.codes, b.codes, std::size_t{a.rows} * a.stride);
    same(label + "/deltas", a.deltas, b.deltas, std::size_t{a.rows} * a.stride / 32U);
    same(label + "/sums", a.sums, b.sums, std::size_t{a.rows} * a.stride / 32U);
}

struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    std::size_t nodes = 0;
    Graph(cudaStream_t s, const std::function<void()>& op) {
        STRIX_CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
        op();
        STRIX_CUDA_CHECK(cudaStreamEndCapture(s, &graph));
        STRIX_CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &nodes));
        STRIX_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
    }
    void run(cudaStream_t s) { STRIX_CUDA_CHECK(cudaGraphLaunch(exec, s)); }
    ~Graph() { cudaGraphExecDestroy(exec); cudaGraphDestroy(graph); }
};

void bench(const std::string& label, cudaStream_t s, const std::function<void()>& legacy,
           const std::function<void()>& fused, c::DeviceBuffer<std::uint8_t>& flush) {
    struct Events {
        cudaEvent_t value[4];
        Events() { for (auto& e : value) STRIX_CUDA_CHECK(cudaEventCreate(&e)); }
        ~Events() { for (auto e : value) cudaEventDestroy(e); }
    } event_set;
    auto& events = event_set.value;
    Graph a(s, [&] {
        STRIX_CUDA_CHECK(cudaEventRecordWithFlags(events[0], s, cudaEventRecordExternal)); legacy();
        STRIX_CUDA_CHECK(cudaEventRecordWithFlags(events[1], s, cudaEventRecordExternal));
    });
    Graph b(s, [&] {
        STRIX_CUDA_CHECK(cudaEventRecordWithFlags(events[2], s, cudaEventRecordExternal)); fused();
        STRIX_CUDA_CHECK(cudaEventRecordWithFlags(events[3], s, cudaEventRecordExternal));
    });
    std::vector<float> ta, tb;
    for (int rep = 0; rep < 27; ++rep) {
        // Alternate order. Each graph is one site; flush outside the event interval for cold weights.
        for (int j = 0; j < 2; ++j) {
            const bool candidate = ((rep + j) & 1) != 0;
            STRIX_CUDA_CHECK(cudaMemsetAsync(flush.get(), rep, flush.size(), s));
            (candidate ? b : a).run(s);
            const int event = candidate ? 2 : 0;
            STRIX_CUDA_CHECK(cudaEventSynchronize(events[event + 1]));
            float ms = 0.0F;
            STRIX_CUDA_CHECK(cudaEventElapsedTime(&ms, events[event], events[event + 1]));
            if (rep >= 3) (candidate ? tb : ta).push_back(ms * 1000.0F);
        }
    }
    const auto median = [](std::vector<float> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2U]; };
    std::printf("BENCH %s nodes=%zu/%zu cold_graph_us=%.3f/%.3f samples=%zu\n",
                label.c_str(), a.nodes - 2U, b.nodes - 2U, median(ta), median(tb), ta.size());
}

c::Q8Tiled tile(c::TensorFile& f, const std::vector<std::string>& names, std::uint32_t rows, std::uint32_t width,
                c::Q8TiledBuffer& result, cudaStream_t s) {
    std::size_t bytes = 0;
    for (const auto& name : names) bytes += f.tensor(name).byte_size;
    c::DeviceBuffer<std::uint8_t> cat(bytes);
    std::size_t at = 0;
    for (const auto& name : names) {
        STRIX_CUDA_CHECK(cudaMemcpy(cat.get() + at, f.device(name), f.tensor(name).byte_size, cudaMemcpyDeviceToDevice));
        at += f.tensor(name).byte_size;
        f.release(name);
    }
    result.repack(cat.get(), rows, width, s);
    STRIX_CUDA_CHECK(cudaStreamSynchronize(s));
    return result.view();
}

void test_layer(c::TensorFile& f, std::uint32_t layer, bool timing, cudaStream_t s) {
    const auto cfg = f.config();
    const auto H = cfg.hidden_size, I = cfg.moe_intermediate_size;
    const std::string p = "blk." + std::to_string(layer) + ".";
    c::Q8TiledBuffer guw, dw;
    const auto gu = tile(f, {p + "ffn_gate_shexp.weight", p + "ffn_up_shexp.weight"}, 2U * I, H, guw, s);
    const auto down = tile(f, {p + "ffn_down_shexp.weight"}, H, I, dw, s);
    c::DeviceBuffer<std::uint8_t> flush(timing ? 256U << 20 : 0U);
    for (const auto R : {1U, 2U, 4U, 6U, 8U, 9U, 16U}) {
        c::Workspace ws(64U << 20);
        const auto floats = [&](std::size_t n) { return ws.take<float>(n); };
        const auto put = [&](std::vector<float> v) {
            auto* ptr = floats(v.size());
            STRIX_CUDA_CHECK(cudaMemcpy(ptr, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
            return ptr;
        };
        const auto a8 = [&](std::uint32_t width) {
            return c::A8Rows{ws.take<std::int8_t>(std::size_t{R} * width), floats(std::size_t{R} * width / 32U),
                            ws.take<std::int32_t>(std::size_t{R} * width / 32U), R, width, width};
        };
        float* stream = put(random_values(std::size_t{R} * 4U * H));
        float* pre = put(random_values(std::size_t{R} * 24U));
        float* sub = put(random_values(std::size_t{R} * H));
        float* mix[2] = {floats(R * 24U), floats(R * 24U)};
        float* norm[2] = {floats(std::size_t{R} * H), floats(std::size_t{R} * H)};
        float* collapse[2] = {floats(std::size_t{R} * H), floats(std::size_t{R} * H)};
        float* expanded[2] = {floats(std::size_t{R} * 4U * H), floats(std::size_t{R} * 4U * H)};
        float* carry[2] = {floats(R * 4U), floats(R * 4U)};
        c::A8Rows xa[2] = {a8(H), a8(H)};
        float* partial = static_cast<float*>(ws.take_bytes(c::hc_mix_workspace_bytes(R, H)));
        const auto label = "L" + std::to_string(layer) + "/R" + std::to_string(R);
        for (const std::string site : {"attn", "ffn"}) {
            c::HcMixView mv;
            mv.stream = stream; mv.rows = R; mv.hidden = H;
            mv.fn = f.device_f16(p + "hc_" + site + "_fn.weight");
            mv.base = f.device_f32(p + "hc_" + site + "_base.weight");
            mv.scale = f.device_f32(p + "hc_" + site + "_scale.weight");
            mv.iters = cfg.hc_sinkhorn_iters; mv.hc_eps = static_cast<float>(cfg.hc_eps); mv.partial = partial;
            c::NormView nv;
            nv.hc_stream = stream; nv.pre = pre; nv.pre_stride = site == "attn" ? 4U : 24U;
            nv.weight = f.device_f32(p + site + "_norm.weight"); nv.rows = R; nv.n = H;
            const auto run = [&](int arm) {
                auto m = mv; auto n = nv;
                m.mixes = mix[arm]; n.out = norm[arm]; n.collapsed = collapse[arm]; n.a8 = xa[arm];
                if (arm) c::enqueue_hc_mixes_norm(m, n, s);
                else { c::enqueue_hc_mixes(m, s); c::enqueue_norm(n, s); }
            };
            run(0); run(1);
            Graph legacy_graph(s, [&] { run(0); }), fused_graph(s, [&] { run(1); });
            if (legacy_graph.nodes != 3U || fused_graph.nodes != 2U) throw std::runtime_error("HC graph node count");
            for (int replay = 0; replay < 8; ++replay) { legacy_graph.run(s); fused_graph.run(s); }
            same(label + "/" + site + "/mix", mix[1], mix[0], R * 24U);
            same(label + "/" + site + "/collapse", collapse[1], collapse[0], std::size_t{R} * H);
            same(label + "/" + site + "/norm", norm[1], norm[0], std::size_t{R} * H);
            same_a8(label + "/" + site + "/A8", xa[1], xa[0]);
            if (timing && R <= 8U) bench(label + "/" + site + "/hc_norm", s, [&] { run(0); }, [&] { run(1); }, flush);
            // the clustered split/norm (arm 1 outputs poisoned first), against the same legacy outputs
            c::set_hc_norm_cluster(true);
            STRIX_CUDA_CHECK(cudaMemsetAsync(mix[1], 0xff, R * 24U * sizeof(float), s));
            STRIX_CUDA_CHECK(cudaMemsetAsync(collapse[1], 0xff, std::size_t{R} * H * sizeof(float), s));
            STRIX_CUDA_CHECK(cudaMemsetAsync(norm[1], 0xff, std::size_t{R} * H * sizeof(float), s));
            STRIX_CUDA_CHECK(cudaMemsetAsync(xa[1].codes, 0x5a, std::size_t{R} * H, s));
            STRIX_CUDA_CHECK(cudaMemsetAsync(xa[1].deltas, 0xff, std::size_t{R} * H / 32U * sizeof(float), s));
            STRIX_CUDA_CHECK(cudaMemsetAsync(xa[1].sums, 0xff, std::size_t{R} * H / 32U * sizeof(std::int32_t), s));
            run(1);
            Graph cluster_graph(s, [&] { run(1); });
            if (cluster_graph.nodes != 2U) throw std::runtime_error("HC cluster graph node count");
            for (int replay = 0; replay < 8; ++replay) { legacy_graph.run(s); cluster_graph.run(s); }
            same(label + "/" + site + "/cluster/mix", mix[1], mix[0], R * 24U);
            same(label + "/" + site + "/cluster/collapse", collapse[1], collapse[0], std::size_t{R} * H);
            same(label + "/" + site + "/cluster/norm", norm[1], norm[0], std::size_t{R} * H);
            same_a8(label + "/" + site + "/cluster/A8", xa[1], xa[0]);
            if (timing && R <= 8U) bench(label + "/" + site + "/hc_norm_cluster", s, [&] { run(0); }, [&] { run(1); }, flush);
            c::set_hc_norm_cluster(false);
        }
        const auto post = [&](int arm) {
            c::enqueue_hc_expand({sub, 0U, stream, mix[0], R, H, expanded[arm], arm ? carry[arm] : nullptr}, s);
            if (!arm) STRIX_CUDA_CHECK(cudaMemcpy2DAsync(carry[arm], 4U * sizeof(float), mix[0], 24U * sizeof(float),
                                                        4U * sizeof(float), R, cudaMemcpyDeviceToDevice, s));
        };
        post(0); post(1);
        Graph legacy_post(s, [&] { post(0); }), fused_post(s, [&] { post(1); });
        if (legacy_post.nodes != 2U || fused_post.nodes != 1U) throw std::runtime_error("post graph node count");
        for (int replay = 0; replay < 8; ++replay) { legacy_post.run(s); fused_post.run(s); }
        same(label + "/post/stream", expanded[1], expanded[0], std::size_t{R} * 4U * H);
        same(label + "/post/pre_carry", carry[1], carry[0], R * 4U);
        // In-place stream update, as used by decode.
        STRIX_CUDA_CHECK(cudaMemcpy(expanded[1], stream, std::size_t{R} * 4U * H * sizeof(float), cudaMemcpyDeviceToDevice));
        c::enqueue_hc_expand({sub, 0U, expanded[1], mix[0], R, H, expanded[1], carry[1]}, s);
        same(label + "/post/inplace", expanded[1], expanded[0], std::size_t{R} * 4U * H);
        if (timing && R <= 8U) bench(label + "/post", s, [&] { post(0); }, [&] { post(1); }, flush);

        float* gateup = floats(std::size_t{R} * 2U * I);
        c::A8Rows act[2] = {a8(I), a8(I)};
        float* shared[2] = {floats(std::size_t{R} * H), floats(std::size_t{R} * H)};
        void* scratch = ws.take_bytes(c::shared_expert_workspace_bytes(R, I));
        c::SharedExpertView e;
        e.gate_up_tiled = gu; e.down_tiled = down; e.x = xa[0]; e.workspace = scratch;
        e.limit = static_cast<float>(cfg.swiglu_limit);
        for (const int pattern : {0, 1, 2}) {
            if (pattern) {
                // Zero and underflow-sized groups test the special A8 path; large finite groups exercise clamping.
                auto v = random_values(std::size_t{R} * H, pattern == 1 ? 0.0F : 100.0F);
                if (pattern == 2) for (std::size_t i = 0; i < std::min<std::size_t>(32U, v.size()); ++i) v[i] = 0x1p-125F;
                STRIX_CUDA_CHECK(cudaMemcpy(norm[0], v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice));
                c::enqueue_quantize_a8({norm[0], R, H, 0U, xa[0]}, s);
            }
            c::Q8A8View g;
            g.tiled = gu; g.count = 2U * I; g.x = xa[0]; g.out = gateup;
            c::enqueue_q8_0_a8(g, s);
            c::enqueue_swiglu_a8({gateup, gateup + I, R, I, 2U * I, e.limit, nullptr, act[0]}, s);
            const auto tag = label + "/pattern" + std::to_string(pattern);
            c::enqueue_swiglu_a8_down(gateup, R, e.limit, down, shared[1], H, act[1], s);
            same_a8(tag + "/down_swiglu_A8", act[1], act[0]);
            e.out = shared[0]; c::enqueue_shared_expert(e, s);
            e.out = shared[1]; c::enqueue_shared_expert_fused(e, s);
            Graph legacy_shared(s, [&] { e.out = shared[0]; c::enqueue_shared_expert(e, s); });
            Graph fused_shared(s, [&] { e.out = shared[1]; c::enqueue_shared_expert_fused(e, s); });
            if (legacy_shared.nodes != 3U || fused_shared.nodes != 2U) throw std::runtime_error("shared graph node count");
            for (int replay = 0; replay < 8; ++replay) { legacy_shared.run(s); fused_shared.run(s); }
            same(tag + "/shared", shared[1], shared[0], std::size_t{R} * H);
        }
        // Restore the FFN norm's A8 for timing; the adversarial clamp pattern is a correctness case.
        e.x = xa[1];
        if (timing && R <= 8U) bench(label + "/shared", s,
            [&] { e.out = shared[0]; c::enqueue_shared_expert(e, s); },
            [&] { e.out = shared[1]; c::enqueue_shared_expert_fused(e, s); }, flush);
    }
}

// Exercise the public assembly API and its existing per-op tap labels. Attention and routed combine
// are supplied deterministic inputs; every core site, including Engram, embedding and head, is chained.
void test_chain(c::TensorFile& f, cudaStream_t s) {
    c::LayerCore core(f, 0U, 3U, true, true, 8U, s);
    const auto g = core.geometry();
    auto& b = core.buffers();
    struct Tap { std::string label; std::vector<float> values; };
    for (const auto rows : {1U, 4U, 8U}) {
        core.set_rows(rows);
        const auto attention = random_values(std::size_t{rows} * g.q_out, 0.1F);
        const std::vector<std::uint8_t> engram(std::size_t{rows} * g.engram_cols * 264U, 0x38U);
        std::vector<std::uint32_t> tokens(rows);
        for (std::uint32_t r = 0; r < rows; ++r) tokens[r] = 1000U + r;
        std::vector<Tap> reference;
        std::size_t index = 0;
        bool fused = false;
        core.set_tap([&](const std::string& label, const float* p, std::uint32_t R, std::uint32_t n, std::uint32_t stride) {
            std::vector<float> v(std::size_t{R} * n);
            STRIX_CUDA_CHECK(cudaMemcpy2DAsync(v.data(), n * sizeof(float), p, stride * sizeof(float), n * sizeof(float),
                                               R, cudaMemcpyDeviceToHost, s));
            STRIX_CUDA_CHECK(cudaStreamSynchronize(s));
            if (!fused) reference.push_back({label, std::move(v)});
            else {
                const auto& ref = reference.at(index++);
                const bool equal = ref.label == label && ref.values.size() == v.size() &&
                                   std::memcmp(ref.values.data(), v.data(), v.size() * sizeof(float)) == 0;
                std::printf("%s chain/R%u/%s values=%zu legacy=%016llx fused=%016llx\n", equal ? "PASS" : "FAIL",
                            rows, label.c_str(), v.size(), static_cast<unsigned long long>(c::fnv1a(ref.values.data(), ref.values.size())),
                            static_cast<unsigned long long>(c::fnv1a(v.data(), v.size())));
                failures += !equal;
            }
        });
        for (int arm = 0; arm < 2; ++arm) {
            fused = arm != 0;
            core.set_decode_fusions(fused, fused);
            core.embed(tokens.data(), s);
            for (std::uint32_t layer = 0; layer < 3U; ++layer) {
                core.attn_input(layer, engram.data(), s);
                core.attn_q(layer, s);
                STRIX_CUDA_CHECK(cudaMemcpyAsync(b.o.get(), attention.data(), attention.size() * sizeof(float), cudaMemcpyHostToDevice, s));
                core.attn_output(layer, s);
                core.ffn_input(layer, s);
                core.ffn_core(layer, s);
                STRIX_CUDA_CHECK(cudaMemcpyAsync(b.y.get(), b.shared.get(), std::size_t{rows} * g.hidden * sizeof(float),
                                                  cudaMemcpyDeviceToDevice, s));
                core.ffn_output(layer, s);
            }
            core.head(s);
        }
        if (index != reference.size()) throw std::runtime_error("chain tap count");
    }
}
} // namespace

int main(int argc, char** argv) try {
    const char* path = std::getenv("STRIX_DS41_FIXTURE_GGUF");
    if (!path) { std::puts("SKIP: set STRIX_DS41_FIXTURE_GGUF"); return 77; }
    bool timing = false;
    std::string overlay;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--bench") timing = true;
        else if (a == "--overlay" && i + 1 < argc) overlay = argv[++i];
        else throw std::runtime_error("usage: [--bench] [--overlay FILE]");
    }
    const auto device = c::select_device(0);
    if (device.name.find("RTX PRO 6000") == std::string::npos) throw std::runtime_error("requires RTX PRO 6000");
    std::printf("device=%s overlay=%s graph microbench=%d\n", device.name.c_str(), overlay.c_str(), timing);
    c::TensorFile f(path, overlay);
    c::Stream stream;
    for (auto layer : {0U, 1U, 2U, 20U}) test_layer(f, layer, timing, stream.get());
    test_chain(f, stream.get());
    std::printf("decode fusions: %s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
    return failures ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what()); return 2;
}
