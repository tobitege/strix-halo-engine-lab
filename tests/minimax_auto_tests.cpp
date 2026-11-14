#include "minimax/dspark/minimax_auto.hpp"
#include "minimax/serve/minimax_dspark_policy.hpp"
#include "minimax/serve/minimax_control.hpp"
#include <iostream>

namespace ds = strix::minimax::dspark;
namespace ms = strix::minimax::serve;
namespace sv = strix::serve;
struct Peer {
    unsigned corrupt = 0, aborts = 0;
    strix::minimax::engine::PfProgress prefill_progress(bool = false) const { return {}; }
    void abort(const char*) { ++aborts; }
    unsigned rank() const { return 0; }
    void exchange_bytes(const void* out, void* in, std::size_t bytes) const {
        std::memcpy(in, out, bytes);
        ms::Control c; std::memcpy(&c, in, sizeof c);
        c.rank = 1U - c.rank;  // the echo answers as the other rank
        if (corrupt == 1) ++c.version;
        if (corrupt == 2) ++c.magic;
        if (corrupt == 3) ++c.active.generation;
        if (corrupt == 4) { c.command = 0; c.ack_round = 999; }
        if (corrupt == 5) ++c.round;            // peer a round ahead (duplicated / dropped control)
        if (corrupt == 6) c.rank = 0;           // both hosts started as rank 0
        std::memcpy(in, &c, sizeof c);
    }
};
int main() {
    unsigned failures = 0;
    const auto check = [&](bool ok, const char* name) {
        if (!ok) { ++failures; std::cerr << "FAIL " << name << '\n'; }
    };
    for (const auto& value : {"off", "0", "on", "1", "auto"})
        check(ds::parse_mode(ds::mode_name(ds::parse_mode(value))) == ds::parse_mode(value), "mode round trip");
    for (const auto& value : {"", "AUTO", "true", "false", "2", " auto", "auto ", "nan"}) {
        bool rejected = false;
        try { (void)ds::parse_mode(value); } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "strict mode parser");
    }
    check(ds::parse_inflight(nullptr) == 1 && ds::parse_inflight("8") == 8 && ds::parse_inflight("0") == 0, "inflight guard");
    for (const auto& value : {"", "-1", "9", "1.5", "nan", "4294967296", "1x"}) {
        bool rejected = false;
        try { (void)ds::parse_inflight(value); } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "strict inflight parser");
    }
    check(ms::dspark_mode_update(ds::Mode::automatic, sv::parse_json("{}")) == ds::Mode::automatic, "absent admin mode retains policy");
    for (const auto& value : {"true", "false"})
        check(ms::dspark_mode_update(ds::Mode::automatic, sv::parse_json(std::string("{\"dspark\":") + value + "}")) ==
              (std::string_view(value) == "true" ? ds::Mode::on : ds::Mode::off), "legacy boolean admin mode");
    for (const auto& value : {"off", "auto", "on"})
        check(ms::dspark_mode_update(ds::Mode::off, sv::parse_json(std::string("{\"dspark_mode\":\"") + value + "\"}")) == ds::parse_mode(value), "admin mode");
    for (const auto& body : {"{\"dspark\":true,\"dspark_mode\":\"on\"}", "{\"dspark\":0}", "{\"dspark_mode\":false}",
                             "{\"dspark_mode\":\"invalid\"}", "{\"dspark_mode\":\"1\"}"}) {
        bool rejected = false;
        try { (void)ms::dspark_mode_update(ds::Mode::automatic, sv::parse_json(body)); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "invalid/conflicting admin mode rejected");
    }
    for (unsigned corrupt = 0; corrupt <= 6; ++corrupt) {
        Peer peer; peer.corrupt = corrupt;
        ms::Control c; c.command = 8; c.round = 1;
        bool rejected = false;
        try { (void)ms::exchange(peer, c); } catch (const std::runtime_error&) { rejected = true; }
        check(rejected == (corrupt != 0) && (corrupt == 0 || peer.aborts > 0), "control version/magic/checkpoint/ack/round/rank validation");
    }
    ds::AutoAdmission a{ds::Mode::automatic, true, 0, false, 100, 4096, 256, true, ds::Prior::structural_v1, {}};
    const auto reason = [&](const char* expected) {
        ds::AutoRequest r; r.admit(a);
        check(r.state == ds::AutoRequest::State::plain_final && std::string_view(r.reason) == expected, expected);
    };
    reason("plain:structural_v1_prose");
    check(ds::parse_prior(nullptr) == ds::Prior::structural_v1 &&
          ds::parse_prior("unqualified-v1") == ds::Prior::unqualified_v1, "prior defaults and fallback");
    for (const char* v : {"", "structural-v2", "STRUCTURAL-V1"}) {
        bool bad = false; try { (void)ds::parse_prior(v); } catch (const std::invalid_argument&) { bad = true; }
        check(bad, "strict prior parser");
    }
    a.prior = ds::Prior::unqualified_v1; reason("plain:prior_unqualified"); a.prior = ds::Prior::structural_v1;
    const auto route = [&](std::string_view text, const char* expected, bool active = true) {
        a.structure = ds::features(text); ds::AutoRequest r; r.admit(a);
        check(std::string_view(r.reason) == expected && (r.state == ds::AutoRequest::State::active) == active, expected);
    };
    route(R"({"nombre":"José","valores":[1,2,3]})", "auto:structural_v1_json");
    a.prior = ds::Prior::unqualified_v1; reason("plain:prior_unqualified"); a.prior = ds::Prior::structural_v1;
    a.live = 1; reason("plain:concurrency"); a.live = 0;
    check(a.structure.keys == 2 && a.structure.pairs == 2 && a.structure.balanced, "JSON byte features");
    route("```rust\nfn main() {}\n```", "auto:structural_v1_code");
    route("int x=0; if(x==0){x=(x+1);}", "auto:structural_v1_code");
    route("def f(x):\n    if x:\n        return x\n    return 0\n", "auto:structural_v1_code");
    route("Once upon a time a traveler told a quiet story about the sea.", "plain:structural_v1_prose", false);
    route("Érase una vez una viajera que contaba una historia tranquila junto al mar.", "plain:structural_v1_prose", false);
    route(R"({"a": [1,2})", "plain:structural_v1_prose", false);
    route(R"({"a":"[{}]", "b":"\"x"})", "plain:structural_v1_prose", false);
    a.structure = {}; a.prompt = ds::long_prompt_tokens - 1; a.max_rows = ds::long_prompt_tokens;
    reason("plain:structural_v1_prose");
    a.prompt = ds::long_prompt_tokens;
    { ds::AutoRequest r; r.admit(a); check(r.state == ds::AutoRequest::State::active &&
        std::string_view(r.reason) == "auto:structural_v1_long", "long token boundary"); }
    a.prior = ds::Prior::unqualified_v1; reason("plain:prior_unqualified");
    a.prior = ds::Prior::structural_v1; a.prompt = 100; a.max_rows = 4096;
    a.live = 1; reason("plain:concurrency"); a.live = 0;
    a.competing = true; reason("plain:concurrency"); a.competing = false;
    a.prompt = 6551; reason("plain:grouped_taps_unqualified"); a.prompt = 100;
    a.room = false; reason("plain:capacity"); a.room = true;
    a.remaining = 1; reason("plain:short_request"); a.remaining = 256;
    a.assets = false; reason("plain:assets_missing"); a.assets = true;
    a.mode = ds::Mode::off; reason("plain:off"); a.mode = ds::Mode::automatic;
    ds::AutoRequest first, second;
    first.admit(a); first.retire("plain:concurrency");
    second.admit(a); first.admit(a);
    check(std::string_view(first.reason) == "plain:concurrency" && std::string_view(second.reason) == "plain:structural_v1_prose",
          "request isolation and no burst-drain reactivation");
    std::cout << "auto policy: " << failures << " failures\n";
    return failures == 0 ? 0 : 1;
}
