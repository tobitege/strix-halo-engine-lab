// The residency command line is one consumer (strix/residency/residency_cli.hpp) shared by the Qwen CUDA serve and
// qwen_decode_bench: the same arguments give the same Options in a serve-style loop (residency flags interleaved with
// the server's own) and in a bench-style tail, the environment fills only what the flags left, flags win.
#include "strix/residency/residency_cli.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <stdlib.h>
static void set_env(const char* k, const char* v) { _putenv_s(k, v ? v : ""); }
#else
static void set_env(const char* k, const char* v) { if (v && *v) setenv(k, v, 1); else unsetenv(k); }
#endif

using namespace strix::residency;
namespace {
int failures = 0;
void require(bool ok, const char* what, int line) {
    if (!ok) { std::cerr << "FAIL line " << line << ": " << what << "\n"; ++failures; }
}
#define REQUIRE(c) require((c), #c, __LINE__)

// serve-style: unknown (server) arguments are skipped by the caller's own branches, residency flags go to the consumer.
CliOptions serve_loop(const std::vector<std::string>& args) {
    CliOptions out;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const auto next = [&]() -> std::string {
            if (i + 1 >= args.size()) throw std::invalid_argument(args[i] + " needs a value");
            return args[++i];
        };
        if (args[i] == "--port") { (void)next(); continue; }  // a server option
        if (consume_cli_arg(args[i], next, out)) continue;
        throw std::invalid_argument("unknown option " + args[i]);
    }
    return out;
}
CliOptions bench_tail(const std::vector<std::string>& args) {
    CliOptions out;
    const auto stop = consume_cli_tail(args, 0, out);
    if (stop != args.size()) throw std::invalid_argument("unknown option " + args[stop]);
    return out;
}
bool same(const Options& a, const Options& b) { return a.mode == b.mode && a.vram_budget == b.vram_budget; }
bool throws(const std::vector<std::string>& args) {
    try { (void)serve_loop(args); } catch (const std::invalid_argument&) {
        try { (void)bench_tail(args); } catch (const std::invalid_argument&) { return true; }
    }
    return false;
}
}  // namespace

int main() {
    set_env("STRIX_EXPERT_RESIDENCY", "");
    set_env("STRIX_VRAM_BUDGET_GIB", "");
    for (const char* mode : {"full", "tiered"}) {
        set_env("STRIX_EXPERT_RESIDENCY", mode);
        REQUIRE(finish_cli(serve_loop({"--expert-residency", "auto"})).mode == Mode::automatic);
        REQUIRE(finish_cli(bench_tail({"--expert-residency", "auto"})).mode == Mode::automatic);
    }
    set_env("STRIX_EXPERT_RESIDENCY", "");
    const std::vector<std::vector<std::string>> cases = {
        {}, {"--expert-residency", "full"}, {"--expert-residency", "tiered", "--vram-budget-gib", "12"},
        {"--vram-budget-gib", "7.5"}, {"--expert-residency", "auto", "--vram-budget-gib", "24"}, {"--print-effective-config"}};
    for (const auto& args : cases) {
        const auto s = serve_loop(args), b = bench_tail(args);
        REQUIRE(same(finish_cli(s), finish_cli(b)));
        REQUIRE(s.print_effective_config == b.print_effective_config);
    }
    {  // interleaved with a server option: same Options as the bare flags
        const auto s = serve_loop({"--port", "8090", "--expert-residency", "tiered", "--vram-budget-gib", "8"});
        const auto b = bench_tail({"--expert-residency", "tiered", "--vram-budget-gib", "8"});
        REQUIRE(same(finish_cli(s), finish_cli(b)));
        REQUIRE(s.options.mode == Mode::tiered && s.options.vram_budget == 8ull << 30);
    }
    REQUIRE(bench_tail({"--vram-budget-gib", "7.5"}).options.vram_budget == (15ull << 29));
    REQUIRE(bench_tail({"--print-effective-config"}).print_effective_config);
    // environment fills what the flags left; flags win
    set_env("STRIX_EXPERT_RESIDENCY", "tiered");
    set_env("STRIX_VRAM_BUDGET_GIB", "16");
    {
        const auto s = serve_loop({}), b = bench_tail({});
        REQUIRE(same(finish_cli(s), finish_cli(b)));
        REQUIRE(finish_cli(s).mode == Mode::tiered && finish_cli(s).vram_budget == 16ull << 30);
        const auto f = serve_loop({"--expert-residency", "full", "--vram-budget-gib", "9"});
        const auto g = bench_tail({"--expert-residency", "full", "--vram-budget-gib", "9"});
        REQUIRE(same(finish_cli(f), finish_cli(g)));
        REQUIRE(finish_cli(f).mode == Mode::full && finish_cli(f).vram_budget == 9ull << 30);
    }
    set_env("STRIX_EXPERT_RESIDENCY", "");
    set_env("STRIX_VRAM_BUDGET_GIB", "");
    // the same malformed input is refused by both
    REQUIRE(throws({"--expert-residency", "banana"}));
    REQUIRE(throws({"--vram-budget-gib", "0"}));
    REQUIRE(throws({"--vram-budget-gib", "-3"}));
    REQUIRE(throws({"--vram-budget-gib", "lots"}));
    REQUIRE(throws({"--expert-residency"}));
    REQUIRE(throws({"--no-such-flag"}));
    if (failures) { std::cerr << failures << " failure(s)\n"; return 1; }
    std::cout << "expert residency parity tests passed\n";
    return 0;
}
