// Shared two-node mode selection (include/strix/pair/regime.hpp) and stage balance (balance.hpp).
#include "strix/pair/balance.hpp"
#include "strix/pair/producer_publish.hpp"
#include "strix/pair/regime.hpp"
#include <array>
#include <cstdio>
#include <limits>
#include <stdexcept>

namespace {
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
}
int main() try {
    {  // producer publication target: off by default, on once the channel flag is set
        strix::pair::ProducerPublish target;
        require(!target.on(), "a default producer target does not publish");
        std::uint64_t flag = 0;
        target.flag = &flag;
        require(target.on(), "a producer target with a flag publishes");
    }
    using strix::pair::Mode;
    const strix::pair::Policy policy;
    require(strix::pair::choose({2048, 4, 0, 0, true}, policy) == Mode::pipeline, "idle pair pipelines a multi-chunk prompt");
    require(strix::pair::choose({300, 1, 0, 0, true}, policy) == Mode::single, "one chunk stays single");
    require(strix::pair::choose({8192, 16, 1, 0, true}, policy) == Mode::single, "a queue keeps nodes independent");
    require(strix::pair::choose({8192, 16, 0, 0, false}, policy) == Mode::single, "a busy peer keeps nodes independent");
    require(strix::pair::choose({8192, 16, 0, 2, true}, policy) == Mode::disaggregate, "active decode disaggregates");
    require(strix::pair::choose({100, 1, 0, 2, true}, policy) == Mode::single, "short prompt beside decode stays local");
    const std::array<double, 6> layers{1, 1, 1, 1, 1, 1};
    const auto split = strix::pair::balance_split(layers, 1.0, [](std::uint32_t) { return true; });
    require(split.cut == 3 && split.a == 3 && split.b == 4, "balanced cut (tail on stage B)");
    const auto odd = strix::pair::balance_split(layers, 0.0, [](std::uint32_t cut) { return cut != 3; });
    require(odd.cut == 2 && odd.a == 2 && odd.b == 4, "illegal cut skipped, smallest best cut kept");
    bool threw = false;
    try { (void)strix::pair::balance_split(layers, 0.0, [](std::uint32_t) { return false; }); } catch (const std::invalid_argument&) { threw = true; }
    require(threw, "no legal cut throws");
    {  // interleaved virtual stages + layout gate (MMPIPE-PLAN section 6, PR-1)
        using strix::pair::PipelineCost;
        using strix::pair::interleaved_wall_ms;
        const auto near = [](double a, double b, double eps) { return a - b < eps && b - a < eps; };
        // (1) v = 1 without hop cost is the two-stage rule, on a rows x chunk grid (Qwen 14-nov costs)
        const PipelineCost qwen{168.0, 0.5474, 35.0, 28.0, 0.0028, 0.536};
        const auto infinity = std::numeric_limits<double>::infinity();
        require(interleaved_wall_ms(std::numeric_limits<std::uint64_t>::max(), 2U, 1U, qwen) == infinity,
            "ceil overflow is rejected before allocation");
        require(interleaved_wall_ms(8U, 1U, std::numeric_limits<std::uint32_t>::max(), qwen) == infinity,
            "stage count overflow is rejected before allocation");
        require(interleaved_wall_ms(strix::pair::max_stage_events, 1U, 1U, qwen) == infinity,
            "event budget bounds allocation products");
        require(interleaved_wall_ms(8U, 0U, 1U, qwen) == infinity &&
            interleaved_wall_ms(8U, 1U, 0U, qwen) == infinity &&
            interleaved_wall_ms(8U, 1U, 1U, qwen, 0U) == infinity, "zero dimensions rejected");
        require(strix::pair::valid_stage_shape(strix::pair::max_stage_events / 2U, 1U, 4U) &&
            !strix::pair::valid_stage_shape(strix::pair::max_stage_events / 2U + 1U, 1U, 4U),
            "inclusive event limit");
        for (const auto invalid : {std::array<std::uint32_t, 5>{0U, 0U, 1U, 1U, 0U},
                {0U, 0U, 1U, 0U, 4U}, {1U, 0U, 1U, 1U, 4U}, {0U, 1U, 1U, 1U, 4U}}) {
            bool rejected = false;
            try { (void)strix::pair::stage_sequence(invalid[0], invalid[1], invalid[2], invalid[3], invalid[4]); }
            catch (const std::invalid_argument&) { rejected = true; }
            require(rejected, "invalid stage sequence shape/index rejected");
        }
        for (const std::uint64_t rows : {1000ULL, 3072ULL, 8192ULL, 12345ULL, 32768ULL, 131072ULL})
            for (const std::uint32_t chunk : {512U, 1024U, 2048U, 4096U, 8192U})
                require(near(interleaved_wall_ms(rows, chunk, 1U, qwen), strix::pair::pipeline_wall_ms(rows, chunk, qwen), 1e-9),
                    "v=1 interleaved equals the two-stage wall");
        // (2) Qwen hand case: C = 2, v = 2, stage t = 545 ms, hop h = 20 ms => 2,785 ms (+ seam)
        PipelineCost hand{};
        hand.ms_per_row = 4.0 * 545.0 / 4096.0;
        hand.hop_latency_ms = 20.0;
        require(near(interleaved_wall_ms(8192U, 4096U, 2U, hand), 2785.0, 1e-6), "hand case C=2 v=2 = 2,785 ms");
        hand.seam_ms = 35.0;
        require(near(interleaved_wall_ms(8192U, 4096U, 2U, hand), 2820.0, 1e-6), "seam adds once");
        // (5) equal stages, no hop: makespan = sum of one rank's stages + the largest stage (Johnson)
        PipelineCost flat{};
        flat.ms_per_row = 0.2;  // P(1000) = 200 ms, t = 100 ms at v = 1
        require(near(interleaved_wall_ms(5000U, 1000U, 1U, flat), 5.0 * 100.0 + 100.0, 1e-9), "Johnson: sum + max");
        // two slots per direction: a depth-first group of 6 chunks cannot complete, 5 can; groups of 4 at C = 8
        require(interleaved_wall_ms(8000U, 1000U, 2U, flat, 6U) == std::numeric_limits<double>::infinity(), "group 6 stalls");
        require(interleaved_wall_ms(8000U, 1000U, 2U, flat, 5U) < 1e9, "group 5 completes");
        require(near(interleaved_wall_ms(8000U, 1000U, 2U, flat), interleaved_wall_ms(8000U, 1000U, 2U, flat, 4U), 1e-12), "default group 4");
        require(strix::pair::stage_sequence(5U, 2U, 8U, 2U) == 13U && strix::pair::stage_sequence(5U, 1U, 8U, 2U) == 5U &&
            strix::pair::stage_sequence(1U, 2U, 2U, 3U) == 3U, "stage sequence: group base x rounds + round x size + offset");
        // (3) MiniMax 8K (FAST stack): P(u) = 580 + 1.9946 u ms per chunk, FP32 seam 22 ms per 4,096 rows,
        // TP2 resident 9,810 ms. Whole stage layers need +41.6 GB/node with 31 GB free: tensor, layout reason.
        PipelineCost mm{580.0, 1.9946, 0.0, 0.0, 0.0, 0.0};
        mm.hop_ms_per_row = 22.0 / 4096.0;
        mm.tp_ms_per_row = 9810.0 / 8192.0;
        const std::array<std::uint32_t, 4> mm_chunks{512U, 1024U, 2048U, 4096U};
        const std::array<std::uint32_t, 6> mm_v{1U, 2U, 3U, 5U, 6U, 10U};
        const auto any = [](std::uint32_t) { return true; };
        const auto oversized = strix::pair::choose_pipeline(std::numeric_limits<std::uint64_t>::max(),
            qwen, mm_chunks, any, mm_v, any, {});
        require(oversized.mode == Mode::single && oversized.chunk == 0U, "oversized candidates are not selected");
        const auto fits60 = [](std::uint32_t v) { return 2U * v <= 60U; };
        strix::pair::LayoutDecl tp2{strix::pair::Residency::tensor_split, 31.0e9, 41.6e9, "engine.hip:89-91,1120-1131"};
        const auto mm8 = strix::pair::choose_pipeline(8192U, mm, mm_chunks, any, mm_v, fits60, tp2);
        require(mm8.mode == Mode::tensor && mm8.reason.rfind("layout: engine.hip:89-91,1120-1131 needs +41.6 GB/node, free 31.0 GB", 0) == 0,
            "MiniMax 8K stays tensor for the layout");
        require(mm8.pipeline_ms > 9500.0 && mm8.pipeline_ms < 9700.0 && near(mm8.alt_ms, 9810.0, 1e-6), "MiniMax best ~9.6 s vs TP 9.81 s");
        const auto rejected_line = strix::pair::effective_line(mm8);
        require(rejected_line.find("mode=tensor") != std::string::npos &&
            rejected_line.find(" candidate_pipeline_ms=") != std::string::npos &&
            rejected_line.find(" alt_ms=9810 ") != std::string::npos &&
            rejected_line.find(" predicted_ms=") == std::string::npos, "rejected candidate cost is labelled");
        tp2.node_free_bytes = 150.0e9;  // layout passed: the best pipeline is still inside run noise of TP2
        const auto mm8b = strix::pair::choose_pipeline(8192U, mm, mm_chunks, any, mm_v, fits60, tp2);
        require(mm8b.mode == Mode::tensor && mm8b.reason.rfind("min_gain:", 0) == 0, "MiniMax 8K under min_gain stays tensor");
        // (4) 32K with the layout passed: interleaved pipeline with v >= 5
        mm.tp_ms_per_row = 40360.0 / 32768.0;
        const std::array<std::uint32_t, 4> mm32_chunks{2048U, 4096U, 8192U, 16384U};
        const auto mm32 = strix::pair::choose_pipeline(32768U, mm, mm32_chunks, any, mm_v, fits60, tp2);
        require(mm32.mode == Mode::pipeline && mm32.virtual_stages >= 5U, "MiniMax 32K pipelines with v >= 5 once the layout fits");
        const auto line = strix::pair::effective_line(mm32);
        require(line.rfind("pair.pipeline mode=pipeline chunk=", 0) == 0 && line.find(" reason=length: ") != std::string::npos,
            "effective-config line");
        std::printf("pair.pipeline MiniMax 8K best %.0f ms (C=%llu v=%u); 32K %s\n", mm8.pipeline_ms,
            static_cast<unsigned long long>(mm8.chunks), mm8.virtual_stages, line.c_str());
        // Qwen whole-model pair: alternative is one node; at 8K the interleaved rule beats v = 1
        PipelineCost qv = qwen;
        qv.hop_ms_per_row = 20.0 / 4096.0;
        const std::array<std::uint32_t, 5> q_chunks{1024U, 2048U, 4096U, 8192U, 16384U};
        const std::array<std::uint32_t, 1> v1{1U};
        const std::array<std::uint32_t, 4> q_v{1U, 2U, 3U, 5U};
        const strix::pair::LayoutDecl whole{};
        const auto q8v1 = strix::pair::choose_pipeline(8192U, qv, q_chunks, any, v1, any, whole);
        const auto q8 = strix::pair::choose_pipeline(8192U, qv, q_chunks, any, q_v, any, whole);
        require(q8v1.mode == Mode::pipeline && q8.mode == Mode::pipeline && q8.pipeline_ms < q8v1.pipeline_ms && q8.virtual_stages > 1U,
            "Qwen 8K interleaved beats two stages");
        std::printf("Qwen 8K v=1 %.0f ms (C=%llu) -> %s\n", q8v1.pipeline_ms, static_cast<unsigned long long>(q8v1.chunks),
            strix::pair::effective_line(q8).c_str());
    }
    std::puts("PASS: pair regime selection and stage balance");
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
