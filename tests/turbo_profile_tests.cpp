// TURBO product mechanics (strix/turbo.hpp, strix/engine_profile.hpp, strix/profiles/qwen.hpp): names, parser,
// turbo column, weight toggle, cache identity, rank agreement and the one resolution of CLI and serve.
#include "strix/engine_profile.hpp"
#include "strix/profiles/qwen.hpp"
#include "strix/turbo.hpp"
#include "../apps/hip_decode_options.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _MSC_VER
#include <crtdbg.h>
#endif

namespace {
namespace p = strix::profile;
namespace t = strix::profile::turbo;

int failures = 0;
void require(bool value, const char* message) {
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", message); ++failures; }
}
template <class F>
bool throws(F&& f) {
    try { f(); } catch (const std::exception&) { return true; }
    return false;
}

// Fixture table: a FAST lever inherited by turbo, a turbo-only lever, a FAST lever turbo keeps out, a CLI-only lever.
constexpr p::Lever kRows[] = {
    {.name = "flash", .env = "T_FLASH", .option = "--flash", .value = {"", "1", "", ""}, .builtin = "0",
     .kind = p::Kind::bit_changing, .ready = {true, true, true, false}, .runnable = {true, true, true, true},
     .target = "", .off = "0", .source = "fixture"},
    {.name = "replay", .env = "T_REPLAY", .option = "", .value = {"", "", "", "128"}, .builtin = "256",
     .kind = p::Kind::bit_changing, .ready = {true, true, true, false}, .runnable = {true, true, true, true},
     .target = "", .off = "256", .source = "fixture", .provisional = {false, false, false, true}},
    {.name = "dense", .env = "T_DENSE", .option = "", .value = {"", "1", "", "0"}, .builtin = "0",
     .kind = p::Kind::bit_changing, .ready = {true, true, true, true}, .runnable = {true, true, true, true},
     .target = "", .off = "0", .source = "fixture"},
    {.name = "head", .env = "", .option = "--head", .value = {"", "", "", "fp16"}, .builtin = "default",
     .kind = p::Kind::bit_changing, .ready = {true, true, true, false}, .runnable = {true, true, true, false},
     .target = "fp16 head", .off = "off", .source = "pending fixture"},
};
const p::Engine kEngine{"turbo-fixture", "", kRows, false, "fixture-turbo-v1"};
const p::Engine kNoTurbo{"no-turbo-fixture", "", kRows, false};

p::Lookup env_of(const std::map<std::string, std::string>& vars) {
    return [&vars](const char* key) -> const char* {
        const auto it = vars.find(key);
        return it == vars.end() ? nullptr : it->second.c_str();
    };
}
const p::Entry& entry(const p::Resolution& r, const char* name) {
    const auto* e = r.find(name);
    if (e == nullptr) throw std::logic_error(name);
    return *e;
}
// The CLI and the serve front end hand the shared parser differently ordered/spelled argument lists.
p::Resolution through(std::vector<std::string> args, const std::map<std::string, std::string>& vars) {
    p::Inputs in;
    in.env = env_of(vars);
    p::take_options(kEngine, args, in);
    return p::resolve(kEngine, in);
}

void names_and_parser() {
    require(p::parse("turbo") == p::Name::turbo && std::string(p::name(p::Name::turbo)) == "turbo", "turbo parses and prints");
    require(p::is_name("turbo") && !p::is_name("turbo-v1"), "is_name knows turbo");
    require(throws([] { (void)p::parse("turb"); }), "unknown profile throws");
    std::vector<std::string> args{"--profile", "turbo", "--turbo-weights=onfly", "--turbo-weight-bits", "4", "--keep"};
    p::Inputs in;
    p::take_options(kEngine, args, in);
    require(in.profile == p::Name::turbo && in.weights.source == std::string("onfly") && in.weights.bits == std::string("4"),
            "take_options reads profile and weight options");
    require(args.size() == 1 && args[0] == "--keep", "take_options removes profile and weight options");
    std::vector<std::string> typo{"--profile", "turbbo"};
    require(throws([&] { p::Inputs x; p::take_options(kEngine, typo, x); }), "a profile typo is refused");
    std::vector<std::string> timing{"--profile", "s:prompt.u32:8"};
    p::Inputs keep;
    p::take_options(kEngine, timing, keep);
    require(!keep.profile && timing.size() == 2, "a timing spec stays with the app (MiniMax --timing-profile compatibility)");
    p::Inputs none;
    none.profile = p::Name::turbo;
    require(throws([&] { (void)p::resolve(kNoTurbo, none); }), "an engine without a versioned turbo name refuses turbo");
    require(std::string(strix::qwen::profile::kTurboVersion) == "qwen-turbo-v1", "qwen versioned name");
}

void turbo_column() {
    const std::map<std::string, std::string> vars;
    const auto r = through({"--profile", "turbo"}, vars);
    require(entry(r, "flash").value == "1" && entry(r, "flash").status == p::Status::set, "empty turbo cell takes the fast cell");
    require(entry(r, "replay").value == "128" && entry(r, "replay").status == p::Status::provisional, "turbo-only provisional lever runs");
    require(entry(r, "dense").value == "0" && entry(r, "dense").status == p::Status::builtin, "turbo cell = off keeps a FAST lever out");
    require(entry(r, "head").status == p::Status::pending && entry(r, "head").value == "default", "unrunnable turbo cell is pending");
    require(r.describe().find("profile version fixture-turbo-v1") != std::string::npos, "describe prints the versioned name");
    require(r.describe().find("requested fp16") != std::string::npos, "describe prints requested beside effective");
    require(r.json().find("\"cache_namespace\":\"turbo=fixture-turbo-v1") != std::string::npos, "json carries the namespace");
    // Precedence CLI > environment > profile > builtin, per lever, inside turbo.
    const std::map<std::string, std::string> env_off{{"T_FLASH", "0"}};
    const auto e = through({"--profile", "turbo"}, env_off);
    require(entry(e, "flash").value == "0" && entry(e, "flash").status == p::Status::off, "environment switches a lever off");
    const auto c = through({"--profile", "turbo", "--flash", "1"}, env_off);
    require(entry(c, "flash").value == "1" && entry(c, "flash").status == p::Status::explicit_cli, "CLI beats environment");
}

void off_recovers_reference() {
    // Every lever explicitly off under turbo runs the reference arithmetic (same lever values as default) but keeps
    // its own cache space and rank identity.
    const std::map<std::string, std::string> all_off{{"T_FLASH", "0"}, {"T_REPLAY", "256"}, {"T_DENSE", "0"}};
    const auto off = through({"--profile", "turbo", "--head", "default"}, all_off);
    const auto ref = through({}, {});
    for (const auto& row : kRows)
        require(entry(off, std::string(row.name).c_str()).value == entry(ref, std::string(row.name).c_str()).value,
                "lever OFF under turbo recovers the reference value");
    require(ref.cache_namespace().empty(), "default exports no cache namespace (legacy identity unchanged)");
    require(!off.cache_namespace().empty(), "turbo has a cache namespace even with every lever off");
    require(off.fingerprint() != ref.fingerprint(), "turbo with every lever off is not the default run for rank agreement");
    // The legacy fingerprint of default is not touched by the turbo fields.
    p::Resolution legacy = ref;
    legacy.weights.origin = "cli";
    require(legacy.fingerprint() == ref.fingerprint(), "weight provenance outside turbo does not enter the fingerprint");
}

void weights_toggle() {
    const t::WeightCaps none{};
    t::WeightCaps four{};
    four.source = {true, true, true};
    four.bits[4] = true;
    const auto env0 = [](const char*) -> const char* { return nullptr; };
    auto w = t::resolve_weights(true, {}, env0, none);
    require(w.honoured() && w.identity() == "native" && w.origin == "profile", "turbo column of the toggle is native");
    w = t::resolve_weights(false, {std::string("onfly"), {}, {}}, env0, four);
    require(!w.honoured() && w.reason.find("requires --profile turbo") != std::string::npos, "onfly outside turbo is refused");
    w = t::resolve_weights(true, {std::string("onfly"), {}, {}}, env0, none);
    require(!w.honoured() && w.reason.rfind("pending:", 0) == 0 && w.identity() == "native", "unbuilt onfly is pending, never silent");
    require(throws([&] { w.require_honoured(); }), "require_honoured refuses an unhonoured request");
    require(w.describe().find("requested onfly:4 (cli) effective native") != std::string::npos, "describe: requested vs effective");
    w = t::resolve_weights(true, {std::string("onfly"), {}, {}}, env0, four);
    require(w.honoured() && w.identity() == "onfly-int4-rtn-g32", "onfly 4-bit identity names source, width and method");
    w = t::resolve_weights(true, {std::string("onfly"), std::string("3"), {}}, env0, four);
    require(!w.honoured() && w.reason.find("incompatible: 3-bit") != std::string::npos, "unsupported width is incompatible");
    w = t::resolve_weights(true, {std::string("converted"), {}, {}}, env0, four);
    require(!w.honoured() && w.reason.find("--turbo-weights-pack") != std::string::npos, "converted needs a pack");
    w = t::resolve_weights(true, {std::string("converted"), {}, std::string("/packs/m-int4")}, env0, four);
    require(w.honoured() && w.identity() == "converted-int4" && w.pack == "/packs/m-int4", "converted identity");
    w = t::resolve_weights(true, {std::nullopt, std::string("4"), {}}, env0, four);
    require(!w.honoured(), "a width without a low-bit source is refused");
    const auto env_onfly = [](const char* k) -> const char* {
        return std::strcmp(k, "STRIX_TURBO_WEIGHTS") == 0 ? "onfly" : nullptr;
    };
    w = t::resolve_weights(true, {}, env_onfly, four);
    require(w.identity() == "onfly-int4-rtn-g32" && w.origin == "env", "environment selects the source");
    w = t::resolve_weights(true, {std::string("native"), {}, {}}, env_onfly, four);
    require(w.identity() == "native" && w.origin == "cli", "CLI beats environment for the source");
    require(throws([&] { (void)t::resolve_weights(true, {std::string("int4"), {}, {}}, env0, four); }), "bad source throws");
    // The weight identity separates profiles' cache spaces and the ranks.
    const auto native = through({"--profile", "turbo"}, {});
    p::Resolution onfly = native;
    onfly.weights = t::resolve_weights(true, {std::string("onfly"), {}, {}}, env0, four);
    require(native.cache_namespace() != onfly.cache_namespace(), "weight sources never share a cache namespace");
    require(native.fingerprint() != onfly.fingerprint(), "weight sources never agree between ranks");
    const auto pending = through({"--profile", "turbo", "--turbo-weights", "onfly"}, {});
    require(!throws([&] {
        require(!p::exports(pending, std::map<std::string, std::string>{}).empty() &&
                pending.describe().find("onfly") != std::string::npos, "pending weight proposal contains exports and reason");
    }), "pending weights remain inspectable as a pure export proposal");
    require(throws([&] { p::apply(pending, [](const char*, const char*) { return true; }); }),
            "apply refuses an unhonoured weight request");
    namespace pq = strix::qwen::profile;
    // Element axis: the CUDA caps name the NVFP4 image; the INT4 identities above stay byte-identical.
    w = t::resolve_weights(true, {std::string("onfly"), {}, {}}, env0, pq::kCudaWeightCaps);
    require(w.honoured() && w.identity() == "onfly-nvfp4-rtn-g16", "CUDA onfly identity names the nvfp element");
    w = t::resolve_weights(true, {std::string("onfly"), std::string("3"), {}}, env0, pq::kCudaWeightCaps);
    require(!w.honoured() && w.reason.find("incompatible: 3-bit") != std::string::npos, "CUDA onfly is 4-bit only");
    w = t::resolve_weights(true, {std::string("converted"), {}, std::string("/packs/x")}, env0, pq::kCudaWeightCaps);
    require(!w.honoured() && w.reason.find("owner") != std::string::npos, "CUDA converted stays pending (owner's OK)");
    pq::Inputs cuda_inputs;
    cuda_inputs.weights.source = "onfly";
    cuda_inputs.caps = &pq::kCudaWeightCaps;
    const auto cuda = pq::resolve(pq::Name::turbo, cuda_inputs, {});
    require(cuda.weights.identity() == "onfly-nvfp4-rtn-g16" && cuda.cache_namespace() != onfly.cache_namespace(),
            "nvfp4 and int4 images never share a cache namespace");
}

void cache_keys() {
    t::CacheKey k{"ds41", "sha-a", "ds41-turbo-v1", "00ff", "native", "chunk=4096;order=fixed", "kv=fp8", "yarn-v1",
                  "hip-gfx1151;lineage=ep"};
    require(k.complete() && t::may_reuse(k, k), "a complete identical key may reuse");
    auto other = k;
    auto delimited = k;
    delimited.order = "a;state=b";
    delimited.state = "c";
    other.order = "a";
    other.state = "b;state=c";
    require(!t::may_reuse(delimited, other) && delimited.text() != other.text() && delimited.id() != other.id(),
            "delimiters in adjacent fields cannot alias complete cache keys");
    other = k;
    other.profile = "fast";
    require(!t::may_reuse(k, other) && k.id() != other.id(), "caches never cross profiles");
    for (auto field : {&t::CacheKey::model, &t::CacheKey::weights_sha, &t::CacheKey::levers, &t::CacheKey::weights,
                       &t::CacheKey::order, &t::CacheKey::state, &t::CacheKey::rotary, &t::CacheKey::backend}) {
        auto changed = k;
        changed.*field += "x";
        require(!t::may_reuse(k, changed), "every key field separates states");
        auto missing = k;
        (missing.*field).clear();
        require(!t::may_reuse(missing, missing), "an unknown field never proves compatibility");
    }
    require(!t::may_reuse(k, k.drafter()) && k.drafter().text().find("role=drafter") != std::string::npos,
            "drafter caches are separate from the target's");
    // Key mechanics only: this does not exercise a speculative engine or establish its cache correctness.
    const auto serial = k, verified = k;
    require(t::may_reuse(serial, verified) && !t::may_reuse(serial.drafter(), verified),
            "identical target keys share, while their drafter keys remain separate");
    require(t::drafter_namespace("turbo=x").find("role=drafter") != std::string::npos && t::drafter_namespace("").empty(),
            "drafter namespace derives from the target's");
    const std::map<std::string, std::string> vars{{"STRIX_TURBO_CACHE_NS", "turbo=q"}};
    require(t::exported_namespace(env_of(vars)) == "turbo=q" && t::exported_namespace(env_of({})).empty(),
            "exported namespace accessor");
}

void rank_agreement_and_one_resolution() {
    // Parser spelling equivalence only; actual CLI/serve front ends are not exercised by this fixture.
    const std::map<std::string, std::string> vars{{"T_DENSE", "1"}};
    const auto cli = through({"--profile", "turbo", "--flash", "0", "--turbo-weights", "native"}, vars);
    const auto serve = through({"--turbo-weights=native", "--flash=0", "--profile=turbo"}, vars);
    require(cli.fingerprint() == serve.fingerprint() && cli.describe() == serve.describe() &&
                cli.cache_namespace() == serve.cache_namespace(), "CLI and serve resolve the same turbo request");
    // Rank agreement over the 32-byte hello: same resolution passes, a different weight source fails.
    const auto swap = [](const p::Resolution& peer) {
        return [&peer](std::span<const std::byte> mine, std::span<std::byte> theirs) {
            std::vector<std::byte> captured;
            try {
                p::check_peer(peer, [&](std::span<const std::byte> o, std::span<std::byte>) {
                    captured.assign(o.begin(), o.end());
                    throw std::runtime_error("captured");
                });
            } catch (const std::runtime_error&) {}
            (void)mine;
            std::memcpy(theirs.data(), captured.data(), theirs.size());
        };
    };
    require(!throws([&] { p::check_peer(cli, swap(serve)); }), "ranks with one resolution agree");
    t::WeightCaps four{};
    four.source = {true, true, true};
    four.bits[4] = true;
    p::Resolution onfly = cli;
    onfly.weights = t::resolve_weights(true, {std::string("onfly"), {}, {}}, {}, four);
    require(throws([&] { p::check_peer(cli, swap(onfly)); }), "ranks with different weight sources disagree");
    const auto fast = through({"--profile", "fast", "--flash", "0"}, vars);
    require(throws([&] { p::check_peer(cli, swap(fast)); }), "turbo and fast ranks disagree");
}

void qwen_adapter() {
    namespace q = strix::qwen::profile;
    const std::map<std::string, std::string> vars;
    q::Inputs in;
    in.env = env_of(vars);
    const auto turbo = q::resolve(p::Name::turbo, in);
    const auto def = q::resolve(p::Name::standard, in);
    require(!turbo.cache_namespace().empty() && def.cache_namespace().empty(), "qwen turbo has its own cache space");
    require(turbo.cache_namespace().find("turbo=qwen-turbo-v1") == 0, "qwen namespace names qwen-turbo-v1");
    const auto* qsa = turbo.find("qsa_budget");
    require(qsa != nullptr && qsa->value == "1536", "qwen turbo inherits the FAST lever cells");
    // Every Qwen lever off: still a turbo namespace, never the default's.
    std::map<std::string, std::string> off;
    for (const auto& row : q::levers()) if (!row.env.empty() && !row.off.empty()) off[std::string(row.env)] = std::string(row.off);
    q::Inputs off_in;
    off_in.env = env_of(off);
    const auto all_off = q::resolve(p::Name::turbo, off_in);
    require(!all_off.cache_namespace().empty() && all_off.cache_namespace() != turbo.cache_namespace(),
            "qwen turbo with every lever off keeps a separate TURBO cache space");
    require(all_off.json().find("\"version\":\"qwen-turbo-v1\"") != std::string::npos, "qwen json carries the turbo block");
    // The qwen CLI option reader takes the weight toggle.
    const char* argv[] = {"app", "--turbo-weights", "onfly", "--profile", "turbo"};
    std::optional<p::Name> name;
    q::Inputs opts;
    for (int i = 1; i < 5; ++i) require(q::take_option(5, const_cast<char**>(argv), i, name, opts), "qwen takes the option");
    require(name == p::Name::turbo && opts.weights.source == std::string("onfly"), "qwen option values");
    const auto admitted = q::resolve(p::Name::turbo, opts);
    require(admitted.weights.honoured() && !throws([&] { q::apply(admitted, [](const char*, const char*) { return true; }); }),
            "qwen admits the implemented onfly loader");
    strix::decode_cli::Options route;
    route.turbo.source = strix::onfly_i4::Source::onfly;
    require(!throws([&] { strix::decode_cli::validate_onfly_i4_routes(route); }), "onfly single runner admitted");
    for (unsigned rows = 1; rows <= 8; ++rows)
        require(!throws([&] { strix::onfly_i4::require_serial_projection(rows, false); }), "onfly rows 1..8 admitted");
    for (unsigned rows : {0U, 9U})
        require(throws([&] { strix::onfly_i4::require_serial_projection(rows, false); }), "onfly unsupported rows refused");
    route.mtp.cohort.emplace_back();
    require(throws([&] { strix::decode_cli::validate_onfly_i4_routes(route); }), "onfly MTP cohorts refused");
    route.mtp.cohort.clear();
    route.http_serve = true;
    for (int batch : {-1, 1, 2}) {
        route.http_batch = batch;
        require(throws([&] { strix::decode_cli::validate_onfly_i4_routes(route); }), "onfly HTTP batch adapters refused");
    }
    route.http_batch = 0;
    require(!throws([&] { strix::decode_cli::validate_onfly_i4_routes(route); }), "onfly HTTP single runner admitted");
    opts.weights.source = "converted";
    opts.weights.pack = "model-int4.pack";
    const auto refused = q::resolve(p::Name::turbo, opts);
    require(!refused.weights.honoured() && throws([&] { q::apply(refused, [](const char*, const char*) { return true; }); }),
            "qwen refuses converted weights until its loader is in the build");
}

}  // namespace

int main() {
#ifdef _MSC_VER
    _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
    _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE);
    _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
#endif
    try {
        names_and_parser();
        turbo_column();
        off_recovers_reference();
        weights_toggle();
        cache_keys();
        rank_agreement_and_one_resolution();
        qwen_adapter();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: unexpected exception: %s\n", e.what());
        return 1;
    }
    if (failures != 0) return 1;
    std::puts("turbo profile tests: ok");
    return 0;
}
