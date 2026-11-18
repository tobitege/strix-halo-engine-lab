#include "strix/qwen35/cuda_engine.hpp"
#include "strix/cuda_core/dense_big.hpp"
#include "strix/spec/engine_target.hpp"
#include "../../src/core/dense_linear.hpp"
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace cc = strix::cuda_core;
namespace q = strix::qwen35;
namespace core = strix::core;
#define CK(expr) do { if ((expr) != cudaSuccess) throw std::runtime_error("mock CUDA failure"); } while (0)
namespace strix::cuda_core {
DenseArith process_policy = DenseArith::mma3, head_policy = DenseArith::mma3;
DenseArith dense_arith() { return process_policy; }
std::uint32_t nvfp4_split_k(std::uint32_t, std::uint32_t) { return 1; }
#include "audit_big.inc"
cudaError_t dense_big_reserve(std::uint32_t, std::uint32_t, std::uint64_t, int, BigCtx* ctx) {
    CK(bind_device(*ctx));
    return cudaMalloc(&ctx->planes[0], 32);
}
cudaError_t dense_rows(const DenseWeight&, const float*, float*, std::uint32_t, const float*, cudaStream_t, DenseArith a) {
    head_policy = a;
    return cudaSuccess;
}
std::size_t attention_scratch_floats(int, unsigned) { return 8; }
// attention.hpp (CUDA bf16 types) stays out of host builds; the engine Pf state holds e4m3 KV shadows by value.
struct KvShadowE4m3 { std::uint8_t *k8 = nullptr; float* ks = nullptr; std::uint8_t* v8t = nullptr; float* vs = nullptr; };
}
struct Engine {
    using PrefillArith = q::PrefillArith;
    q::EngineOptions opt;
    struct Config {
        unsigned hidden = 16, intermediate = 32, vocab = 8, head_dim = 256, heads = 1, layers = 1;
        unsigned q_gate_width() const { return 32; }
        unsigned kv_width() const { return 16; }
    } c;
    struct Gdn {
        unsigned key_dim = 128, value_dim = 128, value_heads = 16;
        unsigned v_width() const { return 16; }
        unsigned conv_channels() const { return 16; }
    } gd;
    struct LayerW { cc::DenseWeight qkv, gate, beta, alpha, out, q, k, v, o, ffn_gate, ffn_up, ffn_down; };
    std::vector<LayerW> L;
    static constexpr unsigned RM = 32;
    unsigned committed = 0, last_rows = 7, work = 0;
    int ad = 0;
    cudaStream_t s = nullptr;
    cc::DenseWeight head;
    float *xn = nullptr, *logits = nullptr;
#include "audit_engine.inc"
};

using WideScratch = core::DenseWideScratch;
using hipStream_t = void*;
struct Tensor { unsigned fmt = 30, inputs = 1, outputs = 1; };
unsigned limit = 0;
float* base_out = nullptr;
std::vector<unsigned> seen;
std::vector<const void*> scratch_seen;
struct Iu4Scratch { void *xh = nullptr, *g = nullptr, *u = nullptr; };
void linear16(const Tensor&, const float*, float* out, unsigned rows, hipStream_t, const Iu4Scratch&) {
    // Model the existing dispatcher's physical row rounding.
    unsigned physical = 1;
    while (physical < rows) physical *= 2;
    runtime_require(out >= base_out && out + physical <= base_out + limit);
    seen.push_back(rows);
}
void glu16(const Tensor& g, const Tensor&, const float* x, float* out, unsigned rows, hipStream_t s, const Iu4Scratch& iu4) {
    linear16(g, x, out, rows, s, iu4);
}
void wide_quant(const float*, unsigned, unsigned, hipStream_t, const WideScratch& w) { scratch_seen.push_back(w.codes); }
void wide_dense(const Tensor&, float*, unsigned, hipStream_t, const WideScratch& w) { scratch_seen.push_back(w.codes); }
// The IU4 wide windows take only type-200 weights; the audited tensors are bf16 / q8_0, as in the real predicate.
bool wide_iu4_ok(const Tensor& t, const WideScratch& w, unsigned) { return w.codes && t.fmt == 200U; }
void wide_iu4(const Tensor&, const float*, float*, unsigned, hipStream_t) { runtime_require(false); }
void silu_mul_host(const float*, const float*, float*, std::size_t) {}
#include "audit_hip.inc"
#include "audit_cli.inc"
#include "audit_lengths.inc"
std::map<std::pair<int, int>, unsigned> attributes;
constexpr int cudaFuncAttributeMaxDynamicSharedMemorySize = 1;
int cudaFuncSetAttribute(int kernel, int, int) { ++attributes[{current_device, kernel}]; return cudaSuccess; }
#include "audit_attributes.inc"
struct OtherEngine {
    enum class LogitRows { all };
    using Rows = int;  // forward_rows' output handle (EngineVerifyTarget observer contract)
    unsigned calls = 0;
    int forward_rows(const std::uint32_t*, std::uint32_t, LogitRows) { ++calls; return 0; }
    void read_top(int, std::uint32_t n, std::uint32_t* ids) { std::fill_n(ids, n, 7U); }
    void commit(std::uint32_t) {}
    void sync() {}
};
struct Bench {
    unsigned pf_bench = 8, R = 16;
    bool pf_twice = false, pf_sample = false, sampled = false;
    std::vector<std::uint32_t> ids = std::vector<std::uint32_t>(8);
    std::string label = "host";
    struct Runner { unsigned prefill_rows() { return 16; } } runner;
    Runner* E = &runner;
    static double now_ms() { static double clock = 0; return ++clock; }
    void prefill_ids(const std::vector<std::uint32_t>& v, std::uint32_t&) { runtime_require(v.size() <= ids.size()); }
#include "audit_bench.inc"
};
int failures = 0;
void check(bool ok, const char* name) { if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", name); } }
template<class F> bool rejects(F&& f) { try { f(); } catch (const std::exception&) { return true; } return false; }
int main() {
    Engine e;
    e.opt.quiet = true;
    std::vector<std::uint32_t> ids(160, 1);
    ids.back() = 8;
    check(rejects([&] { e.admit(ids.data(), 160); }) && e.last_rows == 7 && e.work == 0, "686-1 entire prompt admission before mutation");
    check(rejects([&] { e.admit(nullptr, 160); }), "686-1 null input");
    e.committed = 0xfffffff0U;
    check(rejects([&] { e.admit(ids.data(), 32); }), "686-1 capacity overflow");
    e.committed = 0;
    for (unsigned n = 1; n <= 65; ++n) {
        std::vector<float> v(n); limit = n; base_out = v.data(); seen.clear();
        linear(Tensor{}, v.data(), v.data(), n, nullptr, WideScratch{}, Iu4Scratch{});
        glu(Tensor{}, Tensor{}, v.data(), v.data(), n, nullptr, WideScratch{}, Iu4Scratch{});
        unsigned sum = 0; for (auto count : seen) sum += count;
        check(sum == 2 * n, "686-2 exact logical tails");
    }
    std::int8_t ca = 0, cb = 0;
    WideScratch a{&ca, nullptr, nullptr, nullptr}, b{&cb, nullptr, nullptr, nullptr};
    Tensor q8{8, 128, 16};
    linear(q8, nullptr, nullptr, 64, nullptr, a, Iu4Scratch{});
    linear(q8, nullptr, nullptr, 32, nullptr, b, Iu4Scratch{});
    linear(q8, nullptr, nullptr, 64, nullptr, a, Iu4Scratch{});
    check(scratch_seen == std::vector<const void*>{&ca, &ca, &cb, &cb, &ca, &ca}, "686-3 independent wide scratch");
    for (int device = 0; device < 2; ++device) {
        current_device = device;
        auto ctx = std::unique_ptr<cc::BigCtx>(cc::dense_big_ctx_create());
        CK(cc::bind_device(*ctx));
        const auto* table = cc::get_table(*ctx, cc::kChain, 16, 128, 1);
        check(table && live_allocations.at(table->kbs) == device, "686-4 device-local plan");
        check(cc::get_table(*ctx, cc::kChain, 16, 128, 1) == table, "686-4 plan reuse");
        current_device = 1 - device;
        check(cc::bind_device(*ctx) == cudaErrorInvalidValue, "686-4 reject cross-device context");
    }
    check(live_allocations.empty(), "686-4 table owner cleanup and device restore");
    for (int device = 0; device < 2; ++device) {
        current_device = device;
        CK(attr_big()); CK(attr_big8()); CK(attr_flash(0)); CK(attr_flash(1));
        for (int kernel = 1; kernel <= 3; ++kernel)
            check(attributes[{device, kernel}] != 0, "686-4 attributes initialized on each device");
    }
    current_device = 0;
    { cc::BigCtx ctx; CK(cc::bind_device(ctx)); fail_after = 1;
      check(cc::get_table(ctx, cc::kChain, 16, 128, 1) == nullptr && live_allocations.empty(), "686-4 partial plan allocation cleanup"); fail_after = -1; }
    auto worker = [] { auto& ctx = cc::scratch(); CK(cc::bind_device(ctx)); runtime_require(cc::get_table(ctx, cc::kChain, 16, 128, 1)); };
    std::thread t1(worker), t2(worker); t1.join(); t2.join();
    check(live_allocations.empty(), "686-4 concurrent implicit contexts released");
    for (auto policy : {cc::DenseArith::fp32, cc::DenseArith::mma1, cc::DenseArith::mma2, cc::DenseArith::mma3}) {
        e.opt.dense_arithmetic = policy;
        cc::process_policy = policy == cc::DenseArith::mma3 ? cc::DenseArith::fp32 : cc::DenseArith::mma3;
        check((e.big_fallback() != nullptr) == (policy == cc::DenseArith::fp32), "686-5 per-engine admission");
        check(e.pf_terms() == (policy == cc::DenseArith::mma3 ? 3 : policy == cc::DenseArith::mma2 ? 2 : 1), "686-5 per-engine terms");
        e.final_head(); check(cc::head_policy == policy, "686-5 per-engine final head");
    }
#if defined(STRIX_CUDA_FAST8_SM120A)
    check(cc::dense_big_fast8_supported(), "686-6 specific implementation available");
    device_prop.major = 8; check(!cc::dense_big_fast8_supported(), "686-6 unsupported device");
#else
    check(!cc::dense_big_fast8_supported(), "686-6 generic build unavailable");
#endif
    e.opt.prefill_arith = q::PrefillArith::fast8; ids.back() = 1;
    check(rejects([&] { e.admit(ids.data(), 160); }) && e.work == 0, "686-6 reject before state mutation");
    e.opt.prefill_arith = q::PrefillArith::exact;
    bool completed = false;
    for (int fail = 0; fail < 30; ++fail) {
        fail_after = fail;
        { Engine owned; lifetime.clear();
          if (rejects([&] { owned.pf_init(1); })) check(owned.pf.sl.empty(), "686-7 atomic slot publication");
          else completed = true; }
        check(live_allocations.empty() && streams.empty(), "686-7 partial/full slot cleanup");
        check(!lifetime.empty() && lifetime.front() == 'S' && lifetime.back() == 'D', "686-7 drain before release");
        if (completed) break;
    }
    fail_after = -1; check(completed, "686-7 all allocation failures exercised");
    { Engine owned; owned.pf_init(1); lifetime.clear();
      { Engine::Events pool{owned.pf, nullptr, nullptr, {}};
        pool.start = std::malloc(1); events.insert(pool.start);
        pool.ev.push_back(std::malloc(1)); events.insert(pool.ev.back()); }
      check(events.empty() && lifetime.front() == 'S' && lifetime.back() == 'E', "686-7 event cleanup drains work"); }
    Bench bench; bench.bench(); bench.pf_sample = true; bench.bench();
    check(!bench.sampled, "686-8 short sampled benchmark uses complete short workload");
    check(rejects([] { cli(0, 2); }), "686-9 reject zero repeats");
    check(rejects([] { cli(1, 0); }) && rejects([] { cli(1, 3); }), "686-design slot bounds");
    cli(1, 1); cli(2, 2);
    check(incomplete({1, 2}, {1, 2, 3}, 3) != 0 && incomplete({1, 2, 3}, {1, 2}, 3) != 0,
          "686-design reject incomplete output lengths");
    check(incomplete({1, 2, 3}, {1, 2, 3}, 3) == 0, "686-design complete output lengths");
    OtherEngine other;
    strix::spec::EngineVerifyTarget<OtherEngine> target(other);
    std::vector<std::uint32_t> greedy;
    target.verify(std::span<const std::uint32_t>(ids.data(), 2), greedy);
    check(other.calls == 1 && greedy == std::vector<std::uint32_t>{7, 7}, "686-design shared target accepts independent vocabulary");
    return failures ? 1 : 0;
}
