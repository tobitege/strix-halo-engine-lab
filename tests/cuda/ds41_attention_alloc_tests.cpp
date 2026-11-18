// CPU-only allocation and ring-capacity contracts; no CUDA runtime or device needed.
#include "strix/cuda/ds41_attention_alloc.hpp"
#include "strix/ds41_prefill.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace at = strix::ds41::cuda::attn;
namespace {
void check(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(what);
}
std::size_t named(const at::AllocationLedger& a, std::string_view name) {
    std::size_t n = 0;
    for (const auto& x : a.allocations) if (x.name == name) n += x.bytes;
    return n;
}
void table(std::uint32_t p, bool lean, double expected_mib) {
    const auto plan = at::released_plan();
    at::LayerAttentionOptions o;
    o.max_positions = p + 1U;
    o.max_rows = p;
    o.hidden = 5120;
    o.q_lora_rank = 1280;
    o = at::decode_attention_options(o, true, lean, 8);
    const auto a = at::attention_allocations(plan, o);
    const std::size_t rows = lean ? 8U : p, w = lean ? 8U : 64U;
    const std::size_t apos = std::size_t{p} + 1U, s = (apos + 7U) / 8U + 1U;
    const std::size_t entries = 5U * std::size_t{p} / 2U + 5U;
    const std::size_t history = 356U * entries + 3U * (4096U + 4U);
    const std::size_t ring = 40U * 528U * (lean ? 144U : p + 128U);
    const std::size_t survivors = 4U * w * ((apos + 4095U) / 4096U) * 1025U;
    check(named(a, "kv_cache") == 288U * entries, "KV history table");
    check(named(a, "idx_cache") == 68U * entries, "index history table");
    check(named(a, "ring") == ring, "ring table");
    check(named(a, "carry") == 3U * 4096U && named(a, "carry_valid") == 12U, "carry frontier table");
    check(named(a, "carry_rows") == 3U * 4096U * rows, "rollback table");
    check(named(a, "idx") == 8U * 2048U * rows && named(a, "idx_count") == 8U * 4U * rows, "selection table");
    check(named(a, "candidates") == rows * s && a.cand_stride == s, "candidate full key axis");
    check(named(a, "ckv") == 2048U * rows && named(a, "cscore") == 2048U * rows, "compressor workspace table");
    check(named(a, "iq") == 16384U * rows && named(a, "iw") == 128U * rows, "index projections table");
    check(named(a, "qprep") == 4736U * w, "query prep table");
    check(named(a, "scores") == 4U * w * apos, "scores full key axis");
    check(named(a, "block_keys") == 4U * w * s, "block keys full key axis");
    check(named(a, "survivors") == survivors, "survivor full slice axis");
    check(named(a, "partial") == 4U * w * 64U * 5160U, "partial stride table");
    check(named(a, "rope_window") + named(a, "rope_compressed") == 512U * apos, "full RoPE tables");
    check(a.total(at::AllocationClass::persistent_history) == history, "history class");
    check(a.total(at::AllocationClass::bounded_window) == ring, "window class");
    const auto working = rows * (3U * 4096U + 8U * 2052U + s + 2U * 2048U + 16384U + 128U) +
                         w * (4736U + 4U * apos + 4U * s + 4U * 64U * 5160U) + survivors + 512U * apos;
    check(a.total(at::AllocationClass::working) == working, "working class");
    check(a.total() == history + ring + working, "total table");
    const double mib = double(a.total()) / 1048576.0;
    check(std::abs(mib - expected_mib) < 0.00051, "rounded design total MiB");
    at::DecodeAllocationOptions d;
    d.cand_stride = a.cand_stride;
    d.moe_workspace = 12345;  // supplied by the existing numerical workspace helpers
    d.expert_tp = true;
    d.trellis = true;
    d.trellis_workspace = 6789;
    const auto driver = at::decode_allocations(d);
    check(driver.bytes("raw") == 983040U && driver.bytes("raw1") == 983040U, "raw decode rows");
    check(driver.bytes("pos") == 64U && driver.bytes("kvrows") == 16384U && driver.bytes("router_counters") == 8U, "fixed R8 ABI");
    check(driver.bytes("sec_dev") == 8U * (4U * 356U + 40U * 2052U + s) + 4096U, "boundary staging table");
    d.raw_rows = 4096;
    check(at::decode_allocations(d).bytes("raw") == 4096ULL * 6U * 5120U * 4U, "combined raw capacity unchanged");
    check(driver.bytes("moe_ws") == 12345 && driver.bytes("tr_ws") == 6789, "workspace helper results");
    check(driver.bytes("tr_gate") == 8U * 2304U * 4U && driver.bytes("tr_up") == driver.bytes("tr_gate") &&
          driver.bytes("tr_act") == driver.bytes("tr_gate"), "trellis rows");
    std::cout << "P=" << p << " mode=" << (lean ? "lean" : "current") << " bytes=" << at::allocation_json(a)
              << " MiB=" << std::fixed << std::setprecision(6) << mib << " boundary_bytes=" << driver.bytes("sec_dev") << '\n';
}
void prefill(std::uint32_t p) {
    const auto plan = at::released_plan();
    const auto f = std::min(p, 8192U);
    for (const auto d : {1U, 8U}) {
        at::LayerAttentionOptions o;
        o.max_positions = p + 1U;
        o.max_rows = p;
        o = at::prefill_attention_options(o, 8192, d);
        const auto a = at::attention_allocations(plan, o);
        check(o.max_rows == f && a.ring_rows == 136U + std::max(f, d), "prefill pass/ring bounds");
        check(named(a, "ring") == 40ULL * 528U * (136U + std::max(f, d)), "prefill ring ledger");
        check(named(a, "carry_rows") == 0 && a.bytes("ckv") == 0 && a.bytes("scores") == 0, "prefill no duplicate workspace/stash");
        check(named(a, "idx") == 8ULL * 2048U * f && a.bytes("candidates") == std::uint64_t{f} * ((p + 8ULL) / 8U + 1U),
              "prefill query/key axes");
        const auto seq = at::prefill_sequence_allocations(f, 5120, 1280, 6);
        check(seq.total() == std::uint64_t{f} * 165936U, "prefill independent row byte coefficient");
        strix::ds41::SuffixReplayGeometry geo;
        geo.draft_rows = 128;
        const auto cone = strix::ds41::plan_suffix_replay(geo, 0, p);
        const auto need = p - cone.encoder_output_begin / 64U * 64U;
        check(need + 64U <= f, "head/draft cone fits default pass");
        const auto passes = strix::ds41::plan_prefill_passes(p, 8192, 64, static_cast<std::uint32_t>(need));
        std::uint32_t end = 0;
        for (const auto& pass : passes) {
            check(pass.begin == end && pass.rows <= f, "bounded existing pass planner");
            end += pass.rows;
        }
        check(end == p && passes.back().rows >= need, "complete prompt and final cone");
        const auto first = cone.layers[20].out_begin / 64U * 64U;
        auto compact = o;
        compact.decoder_rows = std::min(f, static_cast<std::uint32_t>(p - first + 64U));
        compact.candidate_rows = compact.decoder_rows;
        const auto ca = at::attention_allocations(plan, compact);
        check(ca.total() <= a.total() && ca.ring_rows == a.ring_rows && named(ca, "carry_rows") == 0,
              "bounded cone compacts selections without changing ring/history");
        std::cout << "P=" << p << " mode=prefill F=" << f << " Dmax=" << d << " ring_rows=" << a.ring_rows
                  << " attention=" << at::allocation_json(a) << " compact_attention=" << at::allocation_json(ca) << " sequence=" << at::allocation_json(seq) << '\n';
    }
}
void bounds() {
    const auto plan = at::released_plan();
    at::LayerAttentionOptions tiny;
    tiny.max_rows = 4;
    check(at::prefill_attention_options(tiny, 8192, 8).ring_rows == 144, "prefill Dmax larger than F");
    at::LayerAttentionOptions o;
    o.max_positions = 1025;
    o.max_rows = 8;
    const auto full = at::attention_allocations(plan, o);
    check(full.bytes("survivors") == 4, "empty scratch minimum allocation");
    o.allocate_workspace = false;
    const auto bare = at::attention_allocations(plan, o);
    check(bare.bytes("survivors") == 0 && bare.bytes("ckv") == 0 && named(bare, "carry_rows") == 0, "no-workspace prefill has no rollback stash");
    o.first_layer = 22;
    o.end_layer = 30;
    const auto range = at::attention_allocations(plan, o);
    check(range.total(at::AllocationClass::bounded_window) == 8U * 136U * 528U, "range rings");
    check(range.total(at::AllocationClass::persistent_history) > 0, "external sources included");
    const auto unchanged = at::decode_attention_options(o, false, true, 4);
    check(unchanged.max_rows == o.max_rows && unchanged.ring_rows == o.ring_rows && !unchanged.allocate_workspace, "prefill untouched");
    const auto opted_out = at::decode_attention_options(o, true, false, 4);
    check(opted_out.max_rows == o.max_rows && opted_out.window_rows == o.window_rows && opted_out.ring_rows == 0, "opt-out restores sizing");
    o.ring_rows = 135;
    bool rejected = false;
    try { (void)at::attention_allocations(plan, o); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "undersized ring rejected");
    // Every kept prefix retains all 136 canonical slots after speculative writes, including wraparound.
    for (std::uint32_t d = 1; d <= 8; ++d) {
        at::LayerAttentionOptions base;
        base.max_positions = 131073;
        base.max_rows = 131072;
        const auto opts = at::decode_attention_options(base, true, true, d);
        const auto a = at::attention_allocations(plan, opts);
        const std::uint32_t cap = a.ring_rows;
        check(cap == 136U + d && opts.max_rows == d && opts.window_rows == d, "Dmax options");
        check(named(a, "carry_rows") == 3U * 4096U * d && named(a, "idx") == 8U * 2048U * d &&
              named(a, "candidates") == std::size_t{d} * a.cand_stride, "Dmax allocations");
        for (std::uint32_t start = cap; start < 3U * cap; ++start) {
            std::vector<std::uint32_t> ring(cap);
            for (std::uint32_t pos = 0; pos < start + d; ++pos) ring[pos % cap] = pos;
            for (std::uint32_t kept = 1; kept <= d; ++kept)
                for (std::uint32_t pos = start + kept - 136U; pos < start + kept; ++pos)
                    check(ring[pos % cap] == pos, "canonical ring rollback capacity");
        }
    }
}
}  // namespace
int main() {
    try {
        table(8192, false, 655.726); table(8192, true, 24.741);
        table(32768, false, 2467.634); table(32768, true, 58.655);
        table(131072, false, 11155.265); table(131072, true, 194.312);
        prefill(8192); prefill(32768); prefill(131072);
        bounds();
        std::cout << "PASS: allocation tables, classes, key axes, opt-out, range sources, scratch minimum and R1..R8 ring rollback\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return 1;
    }
}
