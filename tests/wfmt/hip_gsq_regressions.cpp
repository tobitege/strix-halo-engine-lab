#include "../../src/hip/weight/gsq_plan.hpp"
#include "fixture.hpp"
#include "hip_gsq_checks.hpp"
#include <limits>
#include <string_view>
#ifdef _WIN32
#include <windows.h>
#endif

namespace {
namespace plan = strix::weight::hip;
using namespace strix::weight;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void refuses(F&& f, const char* message) {
    bool refused = false;
    try { f(); } catch (const std::runtime_error&) { refused = true; }
    require(refused, message);
}
void preparation() {
    struct Manifest { std::uint32_t ggml_type; } fused{12U}, gate{12U}, up{12U}, down{7U};
    struct Tensor { const Manifest* manifest = nullptr; };
    struct Routed { bool fused_gate_up; Tensor gate, up, gate_up, down; };
    const Routed tensors{true, {}, {}, {&fused}, {&down}};
    for (const bool q20 : {false, true}) {
        for (const auto type : {7U, 8U, 20U}) {
            down.ggml_type = type;
            bool refused = false;
            try { plan::gsq_prepare_a8_down(tensors, 2U, q20); }
            catch (const std::invalid_argument& e) {
                refused = std::string_view(e.what()) == "A8 down layer types differ from canonical fallback census";
            }
            require(refused, "fused preparation must refuse before reading null separate manifests");
        }
    }
    const Routed separate{false, {&gate}, {&up}, {}, {&down}};
    for (const unsigned layer : {0U, 2U, 4U, 30U, 46U, 47U}) {
        gate.ggml_type = up.ggml_type = layer == 2U ? 13U : 12U;
        down.ggml_type = layer == 0U ? 7U : 8U;
        plan::gsq_prepare_a8_down(separate, layer, false);
    }
    gate.ggml_type = up.ggml_type = 21U; down.ggml_type = 20U;
    plan::gsq_prepare_a8_down(separate, 0U, true);
    bool refused = false;
    try { plan::gsq_prepare_a8_down(separate, 0U, false); }
    catch (const std::invalid_argument&) { refused = true; }
    require(refused, "disabled Q20 exemption admitted");
}
void slices() {
    struct Format { unsigned type; const char* name; };
    for (const Format format : {Format{21U,"gguf.iq3_s"}, {18U,"gguf.iq3_xxs"},
                               {22U,"gguf.iq2_s"}, {20U,"gguf.iq4_nl"}, {42U,"gguf.q2_0.strix"}}) {
        const auto g = plan::gsq_geometry(format.type);
        WeightView w{};
        w.storage = {format.name, 1};
        w.shape = {{1U, 4U * g.elements, 0U, 0U}, 2};
        w.geometry = {g.elements, g.bytes, 32U, 1U, 1U, g.elements, 1U, true};
        // Complete owned source, with the descriptor's canonical geometry.
        w.payload = test::memory(std::vector<Byte>(4U * g.bytes, Byte{0}));
        plan::LaunchShape s{1U, 1U, 2U * g.elements, 4U * g.bytes, 2U * g.elements, 1U, 0U};
        require(bool(plan::gsq_f16_launch(w, s, format.type)), "whole source rejected");
        s.k_offset_bytes = g.bytes;
        require(bool(plan::validate_launch(w, s, g.elements, g.bytes, g.bytes)), "odd slice must be a valid whole-block envelope");
        require(!plan::gsq_f16_launch(w, s, format.type), "odd-block dword-misaligned slice admitted");
        s.k_offset_bytes = 2U * g.bytes;
        require(bool(plan::gsq_f16_launch(w, s, format.type)), "even-block slice rejected");
        s.k_offset_bytes += 4U;
        require(!plan::gsq_f16_launch(w, s, format.type), "partial block admitted");
        s.k_offset_bytes = 0U; --w.payload.range.bytes;
        s.k = s.activation_stride = 4U * g.elements;
        require(!plan::gsq_f16_launch(w, s, format.type), "truncated payload admitted");
    }
}
void fixture_checks() {
    using namespace strix::wfmt_gsq;
    const Snapshot good{{1.0F, 2.0F}, {0.5F}, {3.0F, 4.0F}, {1, 2}, {3}};
    check_snapshot(good, {0U, 0U}, 0U);
    refuses([&] { check_snapshot(good, {0U, 1U}, 0U); }, "row error ignored");
    refuses([&] { check_snapshot(good, {0U, 0U}, 1U); }, "CSR error ignored");
    auto bad = good; bad.raw = {0.0F, 0.0F};
    refuses([&] { check_snapshot(bad, {0U}, 0U); }, "dead launch accepted");
    bad = good; bad.output[1] = std::numeric_limits<float>::quiet_NaN();
    refuses([&] { check_snapshot(bad, {0U}, 0U); }, "nonfinite projection accepted");
    bad = good; bad.raw[1] = 0.0F;
    require(!same(bad, good), "partial wrong output accepted");
    bad = good; bad.dcodes[1] = 0;
    require(!same(bad, good), "wrong activation code accepted");
    // Original row 33 routes both slots to expert 5, in reversed CSR slot order.
    const std::vector<unsigned> words{33U, 33U, 1U, 0U, 5U, 0U, 2U};
    const auto local = row_csr(words, 1U, 0U, 2U, 4U, 5U, 33U);
    require(local[32] == 0U && local[33] == 0U && local[34] == 0U && local[35] == 1U &&
            local[36] == 5U && local[37] == 5U && local[40] == 2U, "tail/duplicate route remapping lost an assignment");
    refuses([&] { (void)row_csr(words, 1U, 0U, 2U, 4U, 5U, 32U); }, "missing token accepted");
}
}
int main(int argc, char** argv) try {
#ifdef _WIN32
    SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
#endif
    require(argc == 2, "choose a regression");
    const std::string_view name(argv[1]);
    if (name == "preparation") preparation();
    else if (name == "slices") slices();
    else if (name == "fixture-checks") fixture_checks();
    else throw std::runtime_error("unknown regression");
    return 0;
} catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
