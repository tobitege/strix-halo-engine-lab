#include "strix/cuda/qwen_hc.hpp"

#include <cstdio>
#include <initializer_list>
#include <stdexcept>

namespace hc = strix::cuda::qwen_hc;
static void require(bool condition) {
    if (!condition) throw std::runtime_error("HC workspace admission regression");
}
template <class F> static void rejects(F call) {
    try { call(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid staged workspace accepted");
}
int main() try {
    alignas(16) std::uint16_t input[16]{};
    alignas(16) float norm[8]{};
    hc::HcMixArgs a;
    a.rows = 65; a.mma = true;
    // Existing output-only fixture: normalized is supplied, input/norm are null.
    require(!hc::hc_compact_workspace(a, 4U, true, true));
    require(!hc::hc_compact_workspace(a, 2U, true, true));
    a.gates = norm;
    require(!hc::hc_compact_workspace(a, 12U, true, true));
    a.gates = nullptr; a.input = input; a.norm = norm;
    require(!hc::hc_compact_workspace(a, 15U, true, true));
    a.workspace_layout = hc::HcWorkspaceLayout::automatic;
    require(hc::hc_compact_workspace(a, 15U, true, true));
    require(!hc::hc_compact_workspace(a, 15U, false, true));
    rejects([&] { (void)hc::hc_compact_workspace(a, 4U, true, true); });
    for (bool mma : {false, true}) {
        a.mma = mma; a.rows = 16;
        a.input = input + 1;
        require(!hc::hc_compact_workspace(a, 15U, true, true));
        a.input = input; a.norm = norm + 1;
        require(!hc::hc_compact_workspace(a, 15U, true, true));
        a.input = input + 1;
        require(!hc::hc_compact_workspace(a, 15U, true, true));
        a.input = input; a.norm = norm;
        require(hc::hc_compact_workspace(a, 15U, true, true));
    }
    a.mma = false; a.rows = 1;
    require(!hc::hc_compact_workspace(a, 15U, true, true));
    require(hc::hc_compact_workspace(a, 15U, true, false));
    a.workspace_layout = hc::HcWorkspaceLayout::compact;
    for (unsigned stage : {1U, 2U, 4U, 15U})
        require(hc::hc_compact_workspace(a, stage, false, true));
    a.gates = norm;
    rejects([&] { (void)hc::hc_compact_workspace(a, 4U, true, true); });
    a.gates = nullptr; a.input = input + 1;
    rejects([&] { (void)hc::hc_compact_workspace(a, 2U, true, true); });
    a.input = input; a.norm = nullptr;
    rejects([&] { (void)hc::hc_compact_workspace(a, 4U, true, true); });
    std::puts("HC full/staged/compact/alignment admission passed");
    return 0;
} catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
}
