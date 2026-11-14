// Exercise the production host scheduler with independently interleaved device
// queues. Consumers must see their own layer even when the side queue runs first.
#include "core/layer_set.hpp"
#include "minimax/engine/mm_fast_i4.hpp"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

namespace fi = strix::minimax::engine::fast_i4;
void require(bool value) { if (!value) std::abort(); }
struct Queues {
    struct Node { std::vector<int> deps; unsigned slot, layer; int kind; };
    std::vector<Node> nodes;
    int tail[2] = {-1, -1}, free[2] = {-1, -1}, ready[2] = {-1, -1};
    unsigned conversions = 0, side_conversions = 0, waits = 0;
    int add(unsigned queue, unsigned s, unsigned L, int kind, int dependency = -1) {
        Node n{{}, s, L, kind};
        if (tail[queue] >= 0) n.deps.push_back(tail[queue]);
        if (dependency >= 0) n.deps.push_back(dependency);
        nodes.push_back(n);
        return tail[queue] = static_cast<int>(nodes.size()) - 1;
    }
    void convert(unsigned s, unsigned L, bool side) {
        ++conversions;
        if (side) ++side_conversions;
        const int id = add(side ? 1U : 0U, s, L, 0, side ? free[s] : -1);
        if (side) ready[s] = id;
    }
    void wait(unsigned s) {
        require(ready[s] >= 0);
        ++waits;
        add(0, s, 0, 2, ready[s]);
    }
    void run(unsigned seed) const {
        std::mt19937 random(seed);
        std::vector<bool> done(nodes.size(), false);
        int contents[2] = {-1, -1};
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            std::vector<std::size_t> runnable;
            for (std::size_t j = 0; j < nodes.size(); ++j) {
                if (!done[j] && std::all_of(nodes[j].deps.begin(), nodes[j].deps.end(),
                                          [&](int d) { return done[static_cast<std::size_t>(d)]; })) runnable.push_back(j);
            }
            require(!runnable.empty());
            const auto j = runnable[random() % runnable.size()];
            const auto& n = nodes[j];
            if (n.kind == 0) contents[n.slot] = static_cast<int>(n.layer);
            if (n.kind == 1) require(contents[n.slot] == static_cast<int>(n.layer));
            done[j] = true;
        }
    }
};
void range(fi::Schedule& schedule, Queues& q, unsigned first, unsigned end, unsigned chunks, bool prefetch) {
    const auto before = q.conversions;
    schedule.begin();
    for (unsigned L = first; L < end; ++L) {
        const unsigned s = schedule.acquire(L, end, prefetch,
            [&](unsigned slot, unsigned layer, bool side) { q.convert(slot, layer, side); },
            [&](unsigned slot) { q.wait(slot); });
        require(s == (L & 1U));
        for (unsigned c = 0; c < chunks; ++c) q.add(0, s, L, 1);
        q.free[s] = q.add(0, s, L, 2);  // release after every chunk
    }
    require(q.conversions - before == end - first);
}
int main() {
    require(fi::scope(nullptr) == fi::Scope::all && fi::scope("") == fi::Scope::all);
    require(fi::scope("all") == fi::Scope::all && fi::scope("gate_up") == fi::Scope::gate_up);
    for (const char* bad : {"ALL", "gate", "all ", "0"}) {
        bool refused = false;
        try { (void)fi::scope(bad); } catch (const std::invalid_argument&) { refused = true; }
        require(refused);
    }
    require(fi::min_rows(nullptr) == fi::kMinRows && fi::min_rows("") == fi::kMinRows && fi::min_rows("0") == 0 && fi::min_rows("8192") == 8192);
    for (const char* bad : {"-1", "4k", " 1", "1e3"}) {
        bool refused = false;
        try { (void)fi::min_rows(bad); } catch (const std::invalid_argument&) { refused = true; }
        require(refused);
    }
    {   // Nested child: the engine parks the parent task (move), the child begins its own task, the parent resumes
        // (move back). The row gate travels with the task, so a short child cannot turn the long parent's INT4 off.
        struct Task { bool open = false; fi::RowGate i4; };
        Task task, parked;
        const auto begin = [&](std::size_t rows) { task = Task{}; task.i4.begin(rows, fi::kMinRows); task.open = true; };
        begin(26000);
        require(task.i4.on);
        parked = std::move(task);
        task = Task{};
        begin(512);
        require(!task.i4.on && parked.i4.on);
        task = std::move(parked);
        parked = Task{};
        require(task.open && task.i4.on);
        begin(4095);
        require(!task.i4.on);
        fi::RowGate zero;
        zero.begin(1, 0);
        require(zero.on);
    }
    {   // STATS travel with the task like the row gate. A long parent runs IU4 launches, is parked, a short child runs
        // EXL3 launches below the gate with no timing sample and reports: the child's report has its own launches
        // only, clearing it leaves the parent's untouched, and the resumed parent still reports its work before the child.
        struct Task { bool open = false; fi::RowGate i4; fi::TaskStats<4> stats; };
        Task task, parked;
        const auto begin = [&](std::size_t rows) { task = Task{}; task.i4.begin(rows, fi::kMinRows); task.open = true; };
        begin(26000);
        task.stats.layer[1].count(task.i4.on, 3);
        task.stats.layer[1].sample(true, 2.0);
        task.stats.layer[1].converted(0.5);
        parked = std::move(task);
        task = Task{};
        begin(205);
        require(!task.stats.layer[1].any());
        task.stats.layer[1].count(task.i4.on, 1);
        const auto& child = task.stats.layer[1];
        require(child.any() && child.i4_calls == 0 && child.exl3_calls == 1 && child.i4_samples == 0 && child.exl3_samples == 0);
        require(fi::LayerStats::mean(child.exl3_ms, child.exl3_samples) == 0.0);  // no sample: counts still print
        task.stats = {};  // the child's report
        task = std::move(parked);
        parked = Task{};
        auto& parent = task.stats.layer[1];
        require(parent.i4_calls == 3 && parent.exl3_calls == 0 && parent.i4_samples == 1 && parent.converts == 1);
        parent.count(true, 2);
        parent.sample(false, 4.0);  // a mixed layer: time is charged to the arithmetic of the sampled group
        parent.sample(true, 4.0);
        require(parent.i4_calls == 5 && parent.i4_ms == 6.0 && parent.exl3_ms == 4.0 && parent.exl3_samples == 1);
        require(fi::LayerStats::mean(parent.i4_ms, parent.i4_samples) == 3.0 && !task.stats.layer[0].any());
    }
    {   // Shared per-layer policy set (core/layer_set.hpp): the keep-EXL3 / W4A8 layer lists of the FAST_I4 engines.
        namespace sc = strix::core;
        auto set = [](const char* v) { return sc::parse_layer_set(v, 3, 60, "TEST"); };
        require(set(nullptr).count() == 0 && set("").count() == 0 && set("none").count() == 0);
        const auto a = set("3,7-9,59");
        require(a.count() == 5 && a.test(3) && a.test(7) && a.test(8) && a.test(9) && a.test(59) && !a.test(4) && !a.test(60));
        const auto b = set("!3-32");
        require(b.count() == 27 && !b.test(3) && !b.test(32) && b.test(33) && b.test(59) && !b.test(2) && !b.test(60));
        require(set("!").count() == 57 && set("3-59").count() == 57 && set("!3-59").count() == 0);
        const auto keep = set(fi::kQualityExl3Layers);   // default: MoE layers 3-37 exact, 38-59 INT4
        require(keep.count() == 35 && keep.test(3) && keep.test(37) && !keep.test(38) && !keep.test(59));
        for (const char* bad : {"2", "60", "9-7", "3,", ",3", "3--4", "a", "3 ", "!none", "1000"}) {
            bool refused = false;
            try { (void)set(bad); } catch (const std::invalid_argument&) { refused = true; }
            require(refused);
        }
    }
    for (bool prefetch : {false, true}) for (unsigned chunks : {1U, 2U, 4U}) {
        Queues q;
        fi::Schedule schedule;
        // Full group, an OFF arm (no acquire), ungrouped tail, paused ranges,
        // parity changes and another prompt: no conversion is cached across units.
        range(schedule, q, 3, 60, chunks, prefetch);
        q.add(0, 0, 0, 2);
        range(schedule, q, 3, 60, 1, prefetch);
        range(schedule, q, 3, 4, chunks, prefetch);
        range(schedule, q, 4, 19, chunks, prefetch);
        range(schedule, q, 19, 60, chunks, prefetch);
        require(prefetch ? q.side_conversions > 0 && q.waits == q.side_conversions : q.side_conversions == 0 && q.waits == 0);
        for (unsigned seed = 0; seed < 32; ++seed) q.run(seed);
    }
    std::cout << "PASS: scope validation, parity slots, 1/2/4 chunks, OFF/resume/repeat, serial/prefetch; 192 queue interleavings\n";
}
