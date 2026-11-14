// Shared two-node mode selection (include/strix/pair/regime.hpp) and stage balance (balance.hpp).
#include "strix/pair/balance.hpp"
#include "strix/pair/producer_publish.hpp"
#include "strix/pair/regime.hpp"
#include <array>
#include <cstdio>
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
    std::puts("PASS: pair regime selection and stage balance");
    return 0;
} catch (const std::exception& error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
