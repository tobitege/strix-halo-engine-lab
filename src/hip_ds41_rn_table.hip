// STRIX_DS41_RN_WIDE=2: the per site x format x rows table of hip_ds41_rn_table.hpp.

#include "strix/hip_ds41_rn_table.hpp"
#include "strix/lever_cache.hpp"

#include <cstdint>
#include <cstdlib>

namespace strix::ds41 {
namespace {

// One choice: old = the engine kernel, else the multi-row form with {waves, outputs per wave, lds, flat}.
struct Pick {
    bool old;
    RnWideShape shape;
};
constexpr Pick old_kernel{true, {}};
constexpr Pick rn(std::uint32_t w, std::uint32_t waves, bool lds) { return {false, {waves, w, lds, false}}; }
constexpr Pick flat(std::uint32_t w, std::uint32_t waves) { return {false, {waves, w, false, true}}; }
// Segment form (form 3: 8 lanes per output, terms in registers; attn_q_b's 40 groups only), activations in place.
constexpr Pick seg(std::uint32_t w, std::uint32_t waves) { return {false, {waves, w, false, false, 3}}; }

// A site: outputs per activation group N (per_group), 32-weight groups Kg (width / 32), and the picks of rows 2..8
// for Q8_0 rows (packed 0) and U6 rows (packed 6).
struct SiteRow {
    const char* name;
    std::uint32_t n, kg;
    Pick q8[7], u6[7];
};

// q_b slot: rows 2..8. The segment form 4,8,0,3 for U6 R3-R8 and Q8_0 R6-R8; the sweep picks elsewhere.
constexpr Pick qb_q8[7] = {rn(1, 8, true), rn(2, 8, true), rn(2, 8, false), rn(1, 16, false), seg(4, 8), seg(4, 8), seg(4, 8)};
constexpr Pick qb_u6[7] = {rn(2, 8, false), seg(4, 8), seg(4, 8), seg(4, 8), seg(4, 8), seg(4, 8), seg(4, 8)};

// Picks of rows 2..8 from the cold sweep (strix-hip-ds41-rn-sel-bench on one node, two sweeps: median of 4
// runs of the stream time; all variants bitwise equal). Rule: the engine kernel when it is within 1% of the best shape,
// else the =1 default shape when within 1% of the best, else the best. One tensor-parallel rank (--tp-attention); the
// single-process q_b (32,768 outputs) and head (129,280) rows reuse the TP picks (not measured).
constexpr SiteRow table[] = {
    // attn_q_a 5120 -> 1280
    {"q_a", 1280, 160,
     {rn(1, 16, true), rn(1, 16, true), rn(1, 16, true), rn(1, 16, true), rn(1, 16, true), rn(1, 16, true), rn(1, 16, true)},
     {rn(1, 16, true), rn(1, 16, true), rn(1, 16, true), rn(1, 8, false), rn(1, 16, true), rn(1, 16, false), rn(1, 8, false)}},
    // attn_kv 5120 -> 512
    {"kv", 512, 160,
     {rn(1, 16, false), rn(1, 16, false), rn(1, 16, false), rn(1, 16, false), rn(1, 16, false), rn(1, 16, false), rn(1, 16, false)},
     {rn(1, 16, true), rn(1, 16, true), rn(1, 16, true), rn(1, 16, true), rn(1, 16, false), rn(1, 8, false), rn(1, 8, false)}},
    // attn_q_b 1280 -> 16,384 (TP rank) / 32,768
    {"q_b", 16384, 40, {qb_q8[0], qb_q8[1], qb_q8[2], qb_q8[3], qb_q8[4], qb_q8[5], qb_q8[6]},
     {qb_u6[0], qb_u6[1], qb_u6[2], qb_u6[3], qb_u6[4], qb_u6[5], qb_u6[6]}},
    {"q_b", 32768, 40, {qb_q8[0], qb_q8[1], qb_q8[2], qb_q8[3], qb_q8[4], qb_q8[5], qb_q8[6]},
     {qb_u6[0], qb_u6[1], qb_u6[2], qb_u6[3], qb_u6[4], qb_u6[5], qb_u6[6]}},
    // attn_output_a: groups of 1,024 outputs of 4,096
    {"o_a", 1024, 128,
     {rn(1, 16, true), rn(1, 16, true), rn(1, 16, true), rn(1, 16, true), rn(1, 16, true), rn(1, 16, true), rn(1, 16, true)},
     {rn(1, 16, true), rn(1, 16, true), rn(1, 16, true), rn(1, 16, true), rn(1, 16, true), rn(1, 16, true), rn(1, 8, false)}},
    // attn_output_b: one K-half (4,096 of 8,192) of 5,120 rows
    {"o_b", 5120, 128,
     {rn(1, 16, true), rn(1, 16, true), rn(1, 16, true), rn(1, 16, true), rn(1, 16, true), rn(1, 16, true), rn(1, 16, true)},
     {rn(1, 16, true), rn(1, 16, true), rn(1, 16, true), rn(1, 16, true), rn(1, 16, true), rn(1, 16, true), rn(1, 16, true)}},
    // output (head): this rank's vocabulary half, 5120 -> 64,640 / 129,280
    {"head", 64640, 160,
     {rn(1, 16, true), rn(1, 16, true), rn(1, 8, true), old_kernel, rn(1, 16, true), rn(1, 16, true), rn(1, 16, false)},
     {flat(2, 8), flat(2, 8), old_kernel, old_kernel, rn(4, 16, false), rn(4, 16, false), rn(4, 16, false)}},
    {"head", 129280, 160,
     {rn(1, 16, true), rn(1, 16, true), rn(1, 8, true), old_kernel, rn(1, 16, true), rn(1, 16, true), rn(1, 16, false)},
     {flat(2, 8), flat(2, 8), old_kernel, old_kernel, rn(4, 16, false), rn(4, 16, false), rn(4, 16, false)}},
};

bool forced_shape() {
    static strix::lever_cache<bool> forced{[] { const char* e = std::getenv("STRIX_DS41_RN_SHAPE"); return e != nullptr && e[0] != 0; }};
    return forced.get();
}

} // namespace

namespace {
thread_local bool rn_wide_off = false;
}

RnWideOff::RnWideOff() : previous(rn_wide_off) { rn_wide_off = true; }
RnWideOff::~RnWideOff() { rn_wide_off = previous; }

int rn_wide_mode() {
    static strix::lever_cache<int> mode{[] {
        const char* e = std::getenv("STRIX_DS41_RN_WIDE");
        return e && e[0] == '0' ? 0 : e && e[0] == '1' ? 1 : 2;
    }};
    return rn_wide_off ? 0 : mode.get();
}

bool rn_table_select(const Q8A8View& view, RnWideShape& shape) {
    if (forced_shape()) { shape = rn_wide_default_shape(view); return true; }
    const std::uint32_t n = view.per_group ? view.per_group : view.count, kg = view.width / 32U;
    if (view.rows >= 2U && view.rows <= 8U && (view.packed == 0U || view.packed == 6U) && !view.planar)
        for (const auto& s : table)
            if (s.n == n && s.kg == kg) {
                const Pick& p = (view.packed == 0U ? s.q8 : s.u6)[view.rows - 2U];
                if (p.old) return false;
                shape = p.shape;
                return true;
            }
    shape = rn_wide_default_shape(view);  // not in the table: the =1 default
    return true;
}

} // namespace strix::ds41
