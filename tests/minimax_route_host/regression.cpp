// Real engine route predicates + real host workspace planner, no HIP calls.
#include "core/linear_plan.hpp"
#include "minimax/engine/route_identity.hpp"
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace route = strix::minimax::route;
namespace mmoe {
inline bool invariant = true;
bool prefill_order_invariant() { return invariant; }
using route::gemm_splits;
struct Exl3 {
    unsigned input = 6144, output = 256, bits = 4;
    const void* trellis = this;
    const void* suh = this;
    const void* svh = this;
};
struct PrefillDenseJob { Exl3 w; float* out; };
// Device metadata size is a stand-in: tests check admission and bounds, not ABI.
namespace pfk { struct Item { std::byte metadata[256]; }; }
#include "planner.inc"
}

struct Engine {
    bool gemv = false, linears_loaded = true, moe_opt = true;
    bool pf_dense_qkv = false, pf_dense_o = false;
    void* pf_ws = this;
    std::size_t pf_ws_bytes = std::size_t{1} << 30;
    struct Options { unsigned max_rows = 4096; } opt;
    float buffer = 0;
    float *q = &buffer, *k = &buffer, *v = &buffer, *part = &buffer;
    struct Layer { mmoe::Exl3 rq, rk, rv, ro; };
    std::vector<Layer> layers{2};
    Engine() {
        for (auto& y : layers) {
            y.rq.output = 4096;
            y.ro.input = 4096;
            y.ro.output = 6144;
        }
    }
#include "engine.inc"
};

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
template <class F> void rejects(F f, const char* message) {
    bool threw = false;
    try { f(); } catch (const std::exception&) { threw = true; }
    require(threw, message);
}
void startup(const Engine& e) {
    (void)e;
#include "startup.inc"
}
void startup_tests() {
    Engine e;
    startup(e);
    e.linears_loaded = false;
    rejects([&] { startup(e); }, "LINEARS=off accepted invariant identity");
    e.linears_loaded = true;
    e.moe_opt = false;
    rejects([&] { startup(e); }, "MOE=off accepted invariant identity");
    e.moe_opt = true;
    for (auto member : {&Engine::Layer::rq, &Engine::Layer::rk, &Engine::Layer::rv, &Engine::Layer::ro}) {
        auto& w = e.layers.back().*member;  // inspect all layers, not only layer zero
        w.bits = 2;
        rejects([&] { startup(e); }, "unsupported format accepted invariant identity");
        w.bits = 4;
        w.trellis = nullptr;
        rejects([&] { startup(e); }, "missing raw weights accepted invariant identity");
        w.trellis = &e;
    }
    e.pf_ws_bytes = 1;
    rejects([&] { startup(e); }, "insufficient workspace accepted invariant identity");
    e.pf_ws = nullptr;
    rejects([&] { startup(e); }, "missing workspace accepted invariant identity");
    mmoe::invariant = false;
    startup(e);  // adaptive keeps its existing fallback contract
}
void resources_tests() {
    Engine e;
    mmoe::PrefillDenseJob qkv[3] = {{e.layers[0].rq, e.q}, {e.layers[0].rk, e.k}, {e.layers[0].rv, e.v}};
    mmoe::PrefillDenseJob o1[1] = {{e.layers[0].ro, e.part}};
    for (unsigned rows : {1U, 8U, 256U}) {
        qkv[1].w.trellis = nullptr;
        rejects([&] { (void)e.dense_qkv(rows, qkv); }, "qkv silently fell back without raw weights");
        qkv[1].w.trellis = &e;
        e.pf_ws = nullptr;
        rejects([&] { (void)e.dense_qkv(rows, qkv); }, "qkv silently fell back without workspace");
        e.pf_ws = &e;
        qkv[2].w.bits = 2;
        rejects([&] { (void)e.dense_qkv(rows, qkv); }, "qkv silently fell back with unsupported bits");
        qkv[2].w.bits = 4;
    }
    o1[0].w.trellis = nullptr;
    rejects([&] { (void)e.dense_o(1, o1); }, "o silently fell back without raw weights");
    o1[0].w.trellis = &e;
    e.pf_ws_bytes = 1;
    rejects([&] { (void)e.dense_o(1, o1); }, "o silently fell back with small workspace");
    rejects([&] { (void)e.dense_qkv(256, qkv); }, "qkv silently fell back with small workspace");
}
void decode_tests() {
    Engine e;
    e.gemv = true;  // LINEARS=load: raw weights exist, optimized decode branch is bypassed
    e.pf_dense_qkv = e.pf_dense_o = true;
    const mmoe::PrefillDenseJob qkv[3] = {{e.layers[0].rq, e.q}, {e.layers[0].rk, e.k}, {e.layers[0].rv, e.v}};
    const mmoe::PrefillDenseJob o1[1] = {{e.layers[0].ro, e.part}};
    for (bool inv : {false, true}) {
        mmoe::invariant = inv;
        for (unsigned rows = 1; rows <= 8; ++rows) {
            require(!e.dense_qkv(rows, qkv), "decode/verification qkv entered prefill arithmetic");
            require(!e.dense_o(rows, o1), "decode/verification o entered prefill arithmetic");
        }
    }
}
void route_tests() {
    Engine e;
    for (unsigned bits : {3U, 4U, 5U, 6U, 8U}) {
        mmoe::PrefillDenseJob qkv[3] = {{e.layers[0].rq, e.q}, {e.layers[0].rk, e.k}, {e.layers[0].rv, e.v}};
        mmoe::PrefillDenseJob o1[1] = {{e.layers[0].ro, e.part}};
        for (auto& j : qkv) j.w.bits = bits;
        o1[0].w.bits = bits;
        for (unsigned rows : {1U, 8U, 9U, 17U, 256U, 1024U, 4096U}) {
            for (bool inv : {false, true}) {
                mmoe::invariant = inv;
                e.pf_dense_qkv = e.pf_dense_o = false;
                const bool qroute = e.dense_qkv(rows, qkv), oroute = e.dense_o(rows, o1);
                if (inv) {
                    for (const auto& j : qkv)
                        require(qroute || (rows > 8 && route::gemm_splits(j.w.input, j.w.output, bits, rows) == 1),
                                "engine retained off-contract qkv library route");
                    require(oroute || (rows > 8 && route::gemm_splits(o1[0].w.input, o1[0].w.output, bits, rows) == 1),
                            "engine retained off-contract o library route");
                } else {
                    require(!qroute && !oroute, "adaptive OFF routes changed");
                    e.pf_dense_qkv = e.pf_dense_o = true;
                    require(e.dense_qkv(rows, qkv) == (rows >= 1024), "adaptive qkv threshold changed");
                    require(e.dense_o(rows, o1) == (rows == 4096), "adaptive o threshold changed");
                }
            }
        }
    }
}
int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected test case");
        const std::string_view name(argv[1]);
        if (name == "startup") startup_tests();
        else if (name == "resources") resources_tests();
        else if (name == "decode") decode_tests();
        else if (name == "routes") route_tests();
        else throw std::runtime_error("unknown test case");
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
