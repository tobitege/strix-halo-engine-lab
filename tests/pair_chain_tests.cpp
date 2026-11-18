// Shared heterogeneous chain planner (include/strix/pair/chain.hpp) and its Qwen binding (qwen_prefill_chain.hpp).
// Synthetic inventory: a discrete card (4,320 rows/s whole model) and two nodes whose pipeline totals 3,200 rows/s,
// as in the receipts the design cites; links are the declared classes (1 GbE measured, 2.5 GbE line rate, USB4).
#include "strix/qwen_prefill_chain.hpp"
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace {
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
namespace chain = strix::pair::chain;
std::vector<chain::Device> inventory() {
    return {strix::qwen_pp::chain_seed_device("card", 4320.0), strix::qwen_pp::chain_seed_device("node-a", 1600.0),
        strix::qwen_pp::chain_seed_device("node-b", 1600.0)};
}
std::vector<chain::Link> links(double card_bytes_per_ms) {
    return {{card_bytes_per_ms, card_bytes_per_ms, 0.3}, {4.58e6, 4.58e6, 0.05}};
}
std::uint32_t card_layers(const chain::Plan& plan) { return plan.first == 0U ? plan.cuts[1] : 0U; }
void malformed_plan_inputs() {
    const std::vector<chain::Device> devices{{"a", {1, 1}, 0, 0, 10}, {"b", {1, 1}, 0, 0, 10}};
    const chain::ModelShape model{2U, 0, {0, 0}, {0, 0}, {1, 1}};
    for (const auto& cuts : std::vector<std::vector<std::uint32_t>>{{0, 3, 2}, {0, 0, 2}, {0, 2, 2}}) {
        const auto plan = chain::evaluate(devices, {{1, 1, 0}}, model, 1, 0, 0, cuts, 1);
        require(!plan.valid && plan.reason == "cuts must be in bounds and strictly increasing", "malformed cuts rejected before sums");
    }
    const auto overflow = chain::evaluate(devices, {{1, 1, 0}}, model, 1,
        std::numeric_limits<std::uint32_t>::max(), 0, {0, 1, 2}, 1);
    require(!overflow.valid && overflow.reason == "device or link outside the declaration", "wrapped device range rejected");
    require(!chain::evaluate(devices, {{1, 1, 0}}, model, 1, 1, 0, {0, 1, 2}, 1).valid, "too many stages rejected");
}
void reverse_block_schedule() {
    const std::vector<chain::Device> devices{{"a", {1, 1}, 0, 0, 0}, {"b", {100, 100}, 0, 0, 0}};
    const chain::ModelShape model{2U, 0, {0, 100}, {0, 0}, {}};
    const auto one = chain::evaluate(devices, {{1, 1, 0}}, model, 1, 0, 0, {0, 1, 2}, 1);
    require(one.valid && one.makespan_ms == 101 && one.wall_ms == 201, "reverse waits for producing stage");
    // Forward completions: 104, 204. Reverse: 207, 310 (100 bytes + 3 ms each).
    const auto two = chain::evaluate(devices, {{1, 1, 3}}, model, 2, 0, 0, {0, 1, 2}, 1);
    require(two.valid && two.makespan_ms == 204 && two.wall_ms == 310 && two.fixed_state_ms == 0,
        "reverse blocks serialize and charge latency without phantom fixed messages");
    const std::vector<chain::Device> three{{"a", {1, 1, 1}, 0, 0, 0}, {"b", {2, 2, 2}, 0, 0, 0},
        {"c", {3, 3, 3}, 0, 0, 0}};
    const chain::ModelShape relay{3U, 0, {0, 5, 7}, {0, 0, 0}, {}};
    // Forward block completions: 10, 13. Hop 1 ends at 19, 28; hop 0 at 33, 47.
    const auto multi = chain::evaluate(three, {{1, 1, 2}, {1, 1, 2}}, relay, 2, 0, 0, {0, 1, 2, 3}, 1);
    require(multi.valid && multi.makespan_ms == 13 && multi.wall_ms == 47, "reverse relay waits and merges local state");
    const std::vector<chain::Device> handoff{{"decoder", {}, 0, 0, 0}, {"relay", {}, 0, 0, 0}, {"producer", {1}, 0, 0, 0}};
    const chain::ModelShape row{1U, 0, {10}, {0}, {}};
    // Two rows then one: releases 2, 3; hop 1 ends 24, 36; hop 0 ends 46, 58.
    const auto partial = chain::evaluate(handoff, {{1, 1, 2}, {1, 1, 2}}, row, 3, 2, 0, {0, 1}, 2);
    require(partial.valid && partial.wall_ms == 58, "partial reverse block and decoder outside stage run");
    auto fixed = row;
    fixed.state_fixed_bytes = {5};
    const auto with_fixed = chain::evaluate(handoff, {{1, 1, 2}, {1, 1, 2}}, fixed, 3, 2, 0, {0, 1}, 2);
    require(with_fixed.valid && with_fixed.fixed_state_ms == 14 && with_fixed.wall_ms == 72, "fixed relay follows row delivery");
}
void planning_work_limits() {
    const std::vector<chain::Device> devices{{"a", {1}, 0, 0, 0}};
    const chain::ModelShape model{1U, 0, {0}, {0}, {}};
    const auto rows = std::numeric_limits<std::uint64_t>::max();
    const auto extreme = chain::evaluate(devices, {}, model, rows, 0, 0, {0, 1}, 2);
    require(!extreme.valid && extreme.blocks == rows / 2U + 1U && extreme.reason == "plan work limit exceeded",
        "extreme rows counted without overflow and rejected before simulation");
    const auto large = chain::evaluate(devices, {}, model, chain::max_plan_work + 1U, 0, 0, {0, 1}, 1);
    require(!large.valid && large.reason == "plan work limit exceeded", "finite oversized simulation rejected");
    chain::Request request;
    request.rows = rows; request.blocks = {2}; request.decoders = {0};
    require(!chain::choose(devices, {}, model, request).best.valid, "search cannot select overflowed zero-time plan");
    request.rows = chain::max_plan_work;
    request.blocks.assign(static_cast<std::size_t>(chain::max_search_work / chain::max_plan_work + 1U), 1U);
    bool exhausted = false;
    try { (void)chain::choose(devices, {}, model, request); }
    catch (const std::invalid_argument& error) { exhausted = std::string(error.what()) == "chain search work limit exceeded"; }
    require(exhausted, "search exhaustion refuses a partial winner");
}
} // namespace

int main() try {
    malformed_plan_inputs();
    reverse_block_schedule();
    planning_work_limits();
    const auto shape = strix::qwen_pp::chain_shape();
    {   // Qwen state layout: 29,952 B/row and ~116.4 MB fixed, as the CUDA export sends
        double row = 0.0, fixed = 0.0;
        for (std::uint32_t l = 0; l < shape.layers; ++l) { row += shape.state_bytes_per_row[l]; fixed += shape.state_fixed_bytes[l]; }
        require(row == 29952.0, "Qwen state bytes per row");
        require(std::fabs(fixed - 116.4e6) < 0.2e6, "Qwen fixed state bytes");
        require(shape.seam_bytes_per_row == 25616.0, "Qwen seam row");
    }
    chain::Request request;
    request.rows = 32768U;
    request.blocks = {8192U, 2048U, 4096U};
    request.decoders = {0U};
    {   // 1 GbE (measured 118 MB/s): the seam alone needs 7.1 s at 32K, the card alone takes 7.59 s -> card alone
        const auto choice = chain::choose(inventory(), links(118000.0), shape, request);
        require(choice.best.valid && choice.best.stages() == 1U && choice.best.first == 0U, "1 GbE keeps the whole prefill on the card");
        require(std::fabs(choice.best.wall_ms - 32768.0 / 4.32) < 1.0, "card-alone wall is rows / rate");
        for (const auto& plan : choice.candidates)
            if (plan.valid && plan.stages() == 3U) require(plan.wall_ms > choice.best.wall_ms, "the 1 GbE chain is slower than the card");
        std::printf("1GbE: %s stages=%u wall=%.0f ms (%.0f rows/s) %s\n", inventory()[choice.best.first].name.c_str(),
            choice.best.stages(), choice.best.wall_ms, choice.best.rows_per_s(request.rows), choice.best.reason.c_str());
    }
    {   // 2.5 GbE: three stages, card about 57 % of the work (seed of the design), ~5.3 s at 32K
        const auto choice = chain::choose(inventory(), links(290000.0), shape, request);
        const auto& best = choice.best;
        require(best.valid && best.stages() == 3U, "2.5 GbE splits over the three devices");
        require(card_layers(best) >= 25U && card_layers(best) <= 30U, "card share near 4,320 / 7,520 of the layers");
        require(best.wall_ms < 5600.0 && best.wall_ms > 4357.0, "2.5 GbE wall between the compute bound and the design example");
        std::printf("2.5GbE: cuts %u/%u block=%u wall=%.0f ms (%.0f rows/s) %s\n", best.cuts[1], best.cuts[2], best.block,
            best.wall_ms, best.rows_per_s(request.rows), best.reason.c_str());
        // the internal pair cut is balanced too: both node stages within one layer of each other
        const auto a = best.cuts[2] - best.cuts[1], b = 48U - best.cuts[2];
        require(a + 1U >= b && b + 1U >= a, "pair-internal cut balanced");
    }
    {   // decode on node B after the prefill: the card's state must cross both hops; at 1 GbE the plan still has
        // to be valid and cannot beat the card + hand-off plan by more than what the seam saves
        auto decode_b = request;
        decode_b.decoders = {2U};
        const auto slow = chain::choose(inventory(), links(118000.0), shape, decode_b);
        require(slow.best.valid && slow.best.decoder == 2U, "1 GbE decode-on-node plan valid");
        std::printf("1GbE decode node-b: first=%u stages=%u wall=%.0f ms %s\n", slow.best.first, slow.best.stages(),
            slow.best.wall_ms, slow.best.reason.c_str());
        const auto fast = chain::choose(inventory(), links(290000.0), shape, decode_b);
        require(fast.best.valid && fast.best.stages() == 3U, "2.5 GbE decode-on-node still splits");
        require(card_layers(fast.best) < card_layers(chain::choose(inventory(), links(290000.0), shape, request).best),
            "moving the card's state to the decoder shrinks the card share");
        std::printf("2.5GbE decode node-b: cuts %u/%u block=%u wall=%.0f ms fixed-state=%.0f ms\n", fast.best.cuts[1],
            fast.best.cuts[2], fast.best.block, fast.best.wall_ms, fast.best.fixed_state_ms);
    }
    {   // two balanced stages with a free link reproduce the pair critical path (m + 1) / 2 x P(block)
        std::vector<chain::Device> pair{{"a", std::vector<double>(4, 0.25), 0, 0, 0}, {"b", std::vector<double>(4, 0.25), 0, 0, 0}};
        chain::ModelShape plain{4U, 0.0, std::vector<double>(4, 0.0), std::vector<double>(4, 0.0), {}};
        const auto plan = chain::evaluate(pair, {{1e12, 1e12, 0.0}}, plain, 8000U, 0U, 1U, {0U, 2U, 4U}, 1000U);
        require(plan.valid && std::fabs(plan.wall_ms - 9.0 / 2.0 * 1000.0) < 1e-6, "two-stage flow shop matches (m+1)/2 x P");
        // partial last block is charged its real rows
        const auto partial = chain::evaluate(pair, {{1e12, 1e12, 0.0}}, plain, 7500U, 0U, 1U, {0U, 2U, 4U}, 1000U);
        require(std::fabs(partial.wall_ms - (8.0 * 500.0 + 250.0)) < 1e-6, "partial last block");
        // an equal split never displaces the one-device plan (ties keep fewer stages)
        chain::Request tie;
        tie.rows = 1000U; tie.blocks = {1000U}; tie.decoders = {0U};
        const auto choice = chain::choose(pair, {{1e12, 1e12, 0.0}}, plain, tie);
        require(choice.best.stages() == 1U, "one block: no overlap, the one-device plan stays");
    }
    {   // split toggle: forced modes resolve through the same planner
        auto forced = request;
        forced.split = chain::Split::device_only;
        const auto only = chain::choose(inventory(), links(290000.0), shape, forced);
        require(only.best.valid && only.best.stages() == 1U && only.best.first == 0U && only.best.decoder == 0U, "device-only");
        forced.split = chain::Split::handoff;
        forced.decoders = {2U};
        const auto handoff = chain::choose(inventory(), links(290000.0), shape, forced);
        require(handoff.best.valid && handoff.best.stages() == 1U && handoff.best.first == 0U && handoff.best.decoder == 2U,
            "handoff-full runs the card and moves the state to node B");
        forced.split = chain::Split::layer_cut;
        forced.decoders = {0U};
        forced.cuts = {24U, 0U};
        const auto cut = chain::choose(inventory(), links(290000.0), shape, forced);
        require(cut.best.valid && cut.best.stages() == 3U && cut.best.cuts[1] == 24U && cut.best.cuts[2] > 24U, "layer-cut k=24, second cut auto");
        forced.cuts = {48U};
        bool threw = false;
        try { (void)chain::choose(inventory(), links(290000.0), shape, forced); } catch (const std::invalid_argument&) { threw = true; }
        require(threw, "a forced cut outside the model is refused");
        forced.split = chain::Split::device_only;
        forced.decoders = {0U};
        forced.cuts.clear();
        const auto slow = chain::choose(inventory(), links(118000.0), shape, forced);
        require(slow.best.reason.starts_with("chosen (split device-only)"), "the split toggle is in the reason");
    }
    {   // rejects: missing link, undeclared capacity, no costs
        auto devices = inventory();
        auto broken = links(290000.0);
        broken[0].forward_bytes_per_ms = 0.0;
        const auto choice = chain::choose(devices, broken, shape, request);
        require(choice.best.valid && choice.best.first == 0U && choice.best.stages() == 1U, "no card link: card alone");
        bool saw = false;
        for (const auto& plan : choice.candidates) saw = saw || plan.reason.starts_with("no forward link card->node-a");
        require(saw, "missing link is an observable reject reason");
        auto weighted = shape;
        weighted.weight_bytes.assign(48, 1.0e9);
        const auto memory = chain::evaluate(devices, links(290000.0), weighted, 32768U, 0U, 0U, {0U, 48U}, 4096U);
        require(!memory.valid && memory.reason == "capacity undeclared for card", "undeclared capacity rejects");
        devices[0].capacity_bytes = 30.0e9;
        const auto fit = chain::evaluate(devices, links(290000.0), weighted, 32768U, 0U, 0U, {0U, 48U}, 4096U);
        require(!fit.valid && fit.reason == "weights do not fit card", "48 GB of weights do not fit 30 GB");
        devices[1].layer_ms_per_row.clear();
        const auto none = chain::evaluate(devices, links(290000.0), shape, 32768U, 0U, 0U, {0U, 20U, 48U}, 4096U);
        require(!none.valid && none.reason == "no layer costs for node-a", "a device without costs never runs layers");
    }
    std::puts("PASS: heterogeneous chain planner");
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
