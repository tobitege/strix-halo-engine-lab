// Equivalence of the shared GSQ admission plan (src/hip/weight/gsq_plan.hpp) with the repeated raw
// predicates it replaced. The `old_*` functions are the PRE-P07 predicates copied verbatim from
// hip_moe_experts_launcher.hip, compiled_speculative_verify.cpp and hip_moe_q4k_a8_known_rows.hip
// (frozen reference: do not edit them to follow the plan). Host only, header-only plan, no HIP.
#include "../../src/hip/weight/gsq_plan.hpp"
#include <cstdint>
#include <iostream>
#include <vector>

namespace {
namespace gsq_a8 = strix::gsq_a8;
namespace plan = strix::weight::hip;
std::uint64_t checked = 0;
int failures = 0;
void same(bool a, bool b, const char* what, std::uint32_t t0, std::uint64_t t1 = 0, std::uint64_t t2 = 0) {
    ++checked;
    if (a != b && failures++ < 20)
        std::cerr << "mismatch " << what << " (" << t0 << ',' << t1 << ',' << t2 << ") old=" << a << " new=" << b << '\n';
}

// ---- frozen pre-P07 predicates ----
bool old_integer_down(std::uint32_t down_type, bool q51, bool q80, bool q20) {
    return (q51 && down_type == 7U) || (q80 && down_type == 8U) || (q80 && gsq_a8::down_type(down_type) && q20);
}
bool old_gate_refused(std::uint32_t gate_type, bool integer_down) {
    return gsq_a8::gate_type(gate_type) && !integer_down;
}
bool old_q20_layer(bool q20, bool fused, std::uint32_t g, std::uint32_t u, std::uint32_t d) {
    return q20 && !fused && gsq_a8::gate_type(g) && u == g && (gsq_a8::down_type(d) || d == 7U || d == 8U);
}
bool old_rows_gate_ineligible(std::uint32_t gate_type, bool q5k, bool q20) {
    return gate_type != 12U && !(gate_type == 13U && q5k) && !(gsq_a8::gate_type(gate_type) && q20);
}
bool old_q20_gate(std::uint32_t g, bool q20) { return gsq_a8::gate_type(g) && q20; }
bool old_window_admitted(bool down, std::uint32_t type, std::uint32_t gate_type, bool q80, bool q5k, bool q20) {
    return down ? type == 7U || (q80 && type == 8U) || (q20 && q80 && gsq_a8::down_type(type))
                : (type == 12U || (q5k && type == 13U) || (q20 && gsq_a8::gate_type(type))) && (!gate_type || type == gate_type);
}
bool old_window_dense(std::uint32_t type, std::uint64_t width, bool q20_enabled, bool window_dense) {
    return !(!q20_enabled || !window_dense || !gsq_a8::dense_type(type) || width % gsq_a8::dense_block_elements(type));
}
bool old_rows_ok(std::uint32_t type, std::uint32_t width) {
    return width % gsq_a8::block_elements(type) == 0U && gsq_a8::row_bytes(type, width) % 4U == 0U &&
           (gsq_a8::block_bytes(type) != 18U || (width / gsq_a8::block_elements(type)) % 2U == 0U);
}
bool old_f16(std::uint32_t g, std::uint32_t d, std::uint32_t hidden, std::uint32_t rows) {
    return gsq_a8::gate_type(g) && gsq_a8::down_type(d) && old_rows_ok(g, hidden) && old_rows_ok(d, rows);
}
}  // namespace

int main() {
    std::vector<std::uint32_t> types;
    for (std::uint32_t t = 0; t < 64U; ++t) types.push_back(t);
    for (std::uint32_t t : {64U, 100U, 255U, 256U, 0xFFFFFFFFU}) types.push_back(t);
    std::vector<std::uint32_t> widths{0U, 640U, 2560U, 4096U};
    for (std::uint32_t e : {18U, 32U, 64U, 110U, 256U})
        for (std::uint32_t k = 0; k <= 9U; ++k)
            for (std::uint32_t d : {0U, 1U, 2U, 3U, 4U, e / 2U, e - 1U}) widths.push_back(k * e + d);
    const bool flags[2]{false, true};

    for (auto t : types) {
        const auto tp = plan::gsq_type_plan(t);
        same(tp.gate, gsq_a8::gate_type(t), "gate", t);
        same(tp.down, gsq_a8::down_type(t), "down", t);
        same(tp.dense, gsq_a8::dense_type(t), "dense", t);
        same(tp.geometry.elements == gsq_a8::block_elements(t) && tp.geometry.bytes == gsq_a8::block_bytes(t), true, "geometry", t);
        const auto dg = plan::gsq_geometry(t, true);
        same(dg.elements == gsq_a8::dense_block_elements(t) && dg.bytes == gsq_a8::dense_block_bytes(t), true, "dense geometry", t);
        for (bool q20 : flags) {
            same(plan::gsq_q20_gate(t, q20), old_q20_gate(t, q20), "q20_gate", t, q20);
            for (bool q5k : flags)
                same(plan::gsq_rows_gate_ok(t, q5k, q20), !old_rows_gate_ineligible(t, q5k, q20), "rows_gate", t, q5k, q20);
            for (bool q51 : flags)
                for (bool q80 : flags) {
                    const bool a = old_integer_down(t, q51, q80, q20);
                    same(plan::gsq_integer_down(t, q51, q80, q20), a, "integer_down", t, q51 * 2U + q80, q20);
                    same(bool(plan::gsq_gate_needs_integer_down(t, a)), !old_gate_refused(t, a), "gate_needs_down", t, a);
                }
            for (bool wd : flags)
                for (auto w : widths) {
                    const auto v = plan::gsq_window_dense(t, w, q20, wd);
                    same(bool(v), old_window_dense(t, w, q20, wd), "window_dense", t, w, q20 * 2U + wd);
                    if (!v && !v.reason) ++failures;
                }
        }
        for (auto w : widths) same(plan::gsq_rows_whole_dword_paired(t, w), old_rows_ok(t, w), "rows_ok", t, w);
        for (auto g : types) {
            for (bool fused : flags)
                for (bool q20 : flags)
                    for (auto d : types) {
                        if (fused || !q20) {
                            same(plan::gsq_q20_layer(q20, fused, g, t, d), old_q20_layer(q20, fused, g, t, d), "q20_layer", g, t, d);
                            continue;
                        }
                        if (t != g && t != 0U && t != 12U) continue;  // up types: equal / differing representatives
                        same(plan::gsq_q20_layer(q20, fused, g, t, d), old_q20_layer(q20, fused, g, t, d), "q20_layer", g, t, d);
                    }
            // Legal gate/down pairs equal the old allowlist (same gate/up type, GSQ gate, GSQ/Q5_1/Q8_0 down).
            for (auto d : types)
                same(plan::gsq_gate_down_legal(g, d), old_q20_layer(true, false, g, g, d), "pair allowlist", g, d);
            for (auto d : {0U, 1U, 12U, 18U, 20U, 42U, 64U}) {
                same(plan::gsq_f16_prefill_capable(g, t, d, 640U), old_f16(g, t, d, 640U), "f16_prefill", g, t, d);
                same(plan::gsq_f16_prefill_capable(g, t, 2560U, d), old_f16(g, t, 2560U, d), "f16_prefill", g, t, d);
            }
            for (bool down : {false, true})
                for (bool q80 : flags)
                    for (bool q5k : flags)
                        for (bool q20 : flags) {
                            const auto ty = down ? t : g;
                            const auto prior = down ? 0U : t;  // prior gate type (0, equal or different)
                            for (auto pg : {0U, prior, g})
                                same(bool(plan::gsq_window_expert(down, ty, pg, {q80, q5k, q20})),
                                     old_window_admitted(down, ty, pg, q80, q5k, q20), "window_expert", ty, pg,
                                     (down * 8U) + q80 * 4U + q5k * 2U + q20);
                        }
        }
    }
    // Allowlist literals: the old list written out.
    const std::uint32_t gates[]{42U, 22U, 18U, 21U, 23U}, downs[]{42U, 20U};
    for (std::uint32_t t = 0; t < 64U; ++t) {
        bool g = false, d = false;
        for (auto x : gates) g |= x == t;
        for (auto x : downs) d |= x == t;
        same(plan::gsq_type_plan(t).gate, g, "gate list", t);
        same(plan::gsq_type_plan(t).down, d, "down list", t);
    }
    // IQ3_S geometry: 110 bytes / 256 elements.
    if (plan::gsq_geometry(21U).bytes != 110U || plan::gsq_geometry(21U).elements != 256U) ++failures;
    // Reason strings are exposed on refusal.
    if (plan::gsq_window_expert(true, 21U, 0U, {true, true, true}).reason == nullptr) ++failures;
    if (failures) {
        std::cerr << "FAIL: " << failures << " mismatches of " << checked << " checks\n";
        return 1;
    }
    std::cout << "gsq plan equivalent on " << checked << " checks\n";
    return 0;
}
