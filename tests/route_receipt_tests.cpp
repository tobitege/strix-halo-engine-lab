// Host tests of the shared route receipt (include/strix/route_receipt.hpp): counters, deltas, the printed block,
// the numerical-route fingerprint and the rank comparison.
#include "strix/route_receipt.hpp"
#include "strix/serve/json.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(_MSC_VER)
#include <crtdbg.h>
#include <cstdlib>
#endif

namespace {
namespace route = strix::route;
using route::Family;
using route::Product;
using route::Reason;

void require(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error("route receipt test failed: " + what);
}

void counters() {
    route::reset();
    route::set_enabled(true);
    route::launched(Product::dense, Family::exact, 512);
    route::launched(Product::dense, Family::exact, 512);
    route::launched(Product::dense, Family::f16_wmma, 2048);
    route::bypass(Product::dense, Family::f16_wmma, Reason::short_call, 512);
    route::bypass(Product::dense, Family::f16_wmma, Reason::short_call, 512);
    const auto t = route::snapshot();
    const auto& dense = t.product[static_cast<std::size_t>(Product::dense)];
    require(dense[static_cast<std::size_t>(Family::exact)].launches == 2 && dense[static_cast<std::size_t>(Family::exact)].rows == 1024, "exact cell");
    require(dense[static_cast<std::size_t>(Family::f16_wmma)].launches == 1 && dense[static_cast<std::size_t>(Family::f16_wmma)].rows == 2048, "wmma cell");
    const auto& by = t.bypass[static_cast<std::size_t>(Product::dense)][static_cast<std::size_t>(Reason::short_call)];
    require(by.launches == 2 && by.rows == 1024, "bypass cell");
    require(!t.empty(), "not empty");

    const auto json = strix::serve::parse_json(route::json(t));
    require(json.find("schema")->as_string() == "strix-route-receipt-v1", "schema");
    require(json.find("products")->find("dense")->find("exact")->find("rows")->as_int() == 1024, "json exact rows");
    require(json.find("products")->find("dense")->find("f16_wmma")->find("launches")->as_int() == 1, "json wmma launches");
    const auto& b = json.find("bypass")->as_array();
    require(b.size() == 1 && b[0].find("wanted")->as_string() == "f16_wmma" && b[0].find("reason")->as_string() == "short_call" &&
            b[0].find("calls")->as_int() == 2, "json bypass");
}

void off_switch() {
    route::reset();
    route::set_enabled(false);
    route::launched(Product::moe, Family::iu4, 4096);
    route::bypass(Product::moe, Family::iu4, Reason::disabled, 1);
    { const route::LayerScope scope(5, Reason::layer_mask); route::launched(Product::moe, Family::exact, 1); }
    require(route::snapshot().empty(), "off: nothing counted");
    route::lever("x.mode", "1", "1");
    require(route::fingerprint() != route::fingerprint({}), "off: levers and fingerprint stay");
    route::set_enabled(true);
}

void layers_and_delta() {
    route::reset();
    for (unsigned l = 3; l < 57; ++l) {
        const bool iu4 = l > 37;  // layers 3..37 keep the exact operator
        const route::LayerScope scope(l, iu4 ? Reason::selected : Reason::layer_mask);
        route::launched(Product::moe, iu4 ? Family::iu4 : Family::exact, 4096);
    }
    route::launched(Product::moe, Family::exact, 7);  // outside a scope: no layer attribution
    const auto first = route::snapshot();
    {
        const route::LayerScope scope(40, Reason::selected);
        route::launched(Product::moe, Family::iu4, 100);
    }
    const auto d = route::delta(route::snapshot(), first);
    require(d.product[static_cast<std::size_t>(Product::moe)][static_cast<std::size_t>(Family::iu4)].launches == 1 &&
            d.product[static_cast<std::size_t>(Product::moe)][static_cast<std::size_t>(Family::iu4)].rows == 100, "delta product");
    require(d.layer[40][static_cast<std::size_t>(Family::iu4)].rows == 100 && d.layer[39][static_cast<std::size_t>(Family::iu4)].rows == 0, "delta layer");

    const auto brief = strix::serve::parse_json(route::json(first, "request", false));
    require(brief.find("levers") == nullptr && brief.find("switches") == nullptr && brief.find("fingerprint") != nullptr &&
            brief.find("layers")->as_array().size() == 2, "brief block: counters and fingerprint only");
    const auto json = strix::serve::parse_json(route::json(first, "request"));
    require(json.find("scope")->as_string() == "request", "scope");
    const auto& runs = json.find("layers")->as_array();
    require(runs.size() == 2, "two layer runs");
    require(runs[0].find("layers")->as_string() == "3-37" && runs[0].find("reasons")->find("layer_mask")->as_int() == 35 &&
            runs[0].find("exact")->find("launches")->as_int() == 35 && runs[0].find("iu4") == nullptr, "exact run");
    require(runs[1].find("layers")->as_string() == "38-56" && runs[1].find("reasons")->find("selected")->as_int() == 19 &&
            runs[1].find("iu4")->find("rows")->as_int() == 19 * 4096, "iu4 run");
    // A later short request (every layer on the exact operator) adds its own reason; it does not relabel the long one.
    const auto long_request = route::snapshot();
    for (unsigned l = 3; l < 57; ++l) {
        const route::LayerScope scope(l, Reason::short_call);
        route::launched(Product::moe, Family::exact, 8);
    }
    const auto total = strix::serve::parse_json(route::json(route::snapshot()));
    const auto& all = total.find("layers")->as_array();
    require(all.size() == 2 && all[1].find("reasons")->find("selected")->as_int() == 20 && all[1].find("reasons")->find("short_call")->as_int() == 19 &&
            all[1].find("iu4")->find("launches")->as_int() == 20 && all[1].find("exact")->find("launches")->as_int() == 19, "process totals keep both reasons");
    const auto window = strix::serve::parse_json(route::json(route::delta(route::snapshot(), long_request), "request", false));
    const auto& short_runs = window.find("layers")->as_array();
    require(short_runs.size() == 1 && short_runs[0].find("layers")->as_string() == "3-56" && short_runs[0].find("reasons")->find("short_call")->as_int() == 54 &&
            short_runs[0].find("reasons")->find("selected") == nullptr && short_runs[0].find("iu4") == nullptr, "request window: only its own reason");
    const auto moe_exact = json.find("products")->find("moe")->find("exact");
    require(moe_exact->find("launches")->as_int() == 36 && moe_exact->find("rows")->as_int() == 35 * 4096 + 7, "unscoped launch counted once");
}

void levers_and_fingerprint() {
    route::reset();
    route::set_engine("test");
    route::set_rank(1);
    route::lever("a.format", "fast_i4", "iu4:all");
    route::lever("b.dense", "1", "0", Reason::short_call, true, "512-row chunks");
    route::lever("c.entry", "cli", "cli", Reason::selected, false);
    const auto base = route::fingerprint();

    // Order does not matter, non-numeric levers do not count, an effective value does.
    std::vector<route::Lever> other{{"b.dense", "auto", "0", "", Reason::disabled, true}, {"a.format", "", "iu4:all", "", Reason::selected, true}};
    require(route::fingerprint(other) == base, "order / requested / reason independent");
    other[0].effective = "1";
    require(route::fingerprint(other) != base, "effective value changes the fingerprint");
    route::lever("c.entry", "http", "http", Reason::selected, false);
    require(route::fingerprint() == base, "non-numeric lever is outside the fingerprint");
    route::lever("b.dense", "1", "1");
    require(route::fingerprint() != base && route::levers().size() == 3, "same name replaces");
    // "ab"+"c" and "a"+"bc" are different levers.
    require(route::fingerprint({{"ab", "", "c", "", Reason::selected, true}}) != route::fingerprint({{"a", "", "bc", "", Reason::selected, true}}), "name/value boundary");

    const auto json = strix::serve::parse_json(route::json());
    require(json.find("engine")->as_string() == "test" && json.find("rank")->as_int() == 1, "engine and rank");
    require(json.find("fingerprint")->as_string().size() == 16, "fingerprint text");
    const auto& l = json.find("levers")->as_array();
    require(l.size() == 3 && l[1].find("name")->as_string() == "b.dense" && l[1].find("effective")->as_string() == "1", "lever list");
    require(l[2].find("numeric") != nullptr && !l[2].find("numeric")->as_bool(), "non-numeric marked");
}

void rank_comparison() {
    route::reset();
    route::lever("m.format", "fast_i4", "iu4:all");
    const auto mine = route::fingerprint();
    // Same route on the peer: the exchange returns the same 16 bytes.
    const auto same = route::compare_ranks([](const void* out, void* in, std::size_t bytes) { std::memcpy(in, out, bytes); });
    require(same == mine, "compare returns the fingerprint");
    // A peer on another numerical route is an error, not a warning.
    bool thrown = false;
    try {
        (void)route::compare_ranks([](const void* out, void* in, std::size_t bytes) {
            std::memcpy(in, out, bytes);
            static_cast<std::uint64_t*>(in)[1] ^= 1U;
        });
    } catch (const std::runtime_error& e) {
        thrown = std::string(e.what()).find("differs between the ranks") != std::string::npos &&
                 std::string(e.what()).find("m.format=iu4:all") != std::string::npos;
    }
    require(thrown, "different fingerprint throws and names the local levers");
    thrown = false;
    try {
        (void)route::compare_ranks([](const void*, void* in, std::size_t bytes) { std::memset(in, 0, bytes); });
    } catch (const std::runtime_error&) { thrown = true; }
    require(thrown, "a peer without the receipt is refused");
    route::require_same_route(7, 7);
}

void threads() {
    route::reset();
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < 4; ++t)
        pool.emplace_back([t] {
            const route::LayerScope scope(10 + t, Reason::selected);
            for (unsigned i = 0; i < 10000; ++i) route::launched(Product::core, Family::fma_rows, 2);
        });
    for (auto& th : pool) th.join();
    const auto s = route::snapshot();
    require(s.product[static_cast<std::size_t>(Product::core)][static_cast<std::size_t>(Family::fma_rows)].launches == 40000, "threaded launches");
    for (unsigned t = 0; t < 4; ++t)
        require(s.layer[10 + t][static_cast<std::size_t>(Family::fma_rows)].rows == 20000, "layer scope is per thread");
}
}  // namespace

int main() {
#if defined(_MSC_VER)
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    try {
        counters();
        off_switch();
        layers_and_delta();
        levers_and_fingerprint();
        rank_comparison();
        threads();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
    std::puts("route receipt tests: OK");
    return 0;
}
