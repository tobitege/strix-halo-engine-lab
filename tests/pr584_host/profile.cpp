#include "strix/deepseek41_pair_defaults.hpp"
#include "strix/ds41_fast_profile.hpp"
#include <algorithm>
#include <array>
#include <cstdio>
#include <map>
#include <stdexcept>
#include <string>

namespace pd = strix::deepseek41::pair_defaults;
using Env = std::map<std::string, std::string>;
void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
pd::Resolved resolve(Env env, bool fast = false) {
    env["STRIX_DS41_BASELINE"] = "1";
    return pd::resolve(pd::App::rank, fast ? std::vector<std::string>{"--profile", "fast"} : std::vector<std::string>{},
        [&](const char* key) -> const char* {
            const auto it = env.find(key);
            return it == env.end() ? nullptr : it->second.c_str();
        }, {}, {});
}
bool agrees(const pd::Resolved& a, const pd::Resolved& b) {
    std::array<std::byte, 32> hello{};
    strix::profile::check_peer(*b.profile, [&](auto out, auto in) {
        std::copy(out.begin(), out.end(), hello.begin());
        std::copy(out.begin(), out.end(), in.begin());
    });
    try {
        strix::profile::check_peer(*a.profile, [&](auto, auto in) {
            std::copy(hello.begin(), hello.end(), in.begin());
        });
        return true;
    } catch (const std::runtime_error&) { return false; }
}
std::string exported(const pd::Resolved& r, const char* key) {
    for (const auto& [k, v] : r.set_env) if (k == key) return v;
    throw std::runtime_error("missing canonical export");
}
int main() {
    try {
        const auto off = resolve({});
        const auto dense = resolve({{"STRIX_DS41_FAST_DENSE", "1"}});
        require(!agrees(off, dense) && !agrees(dense, off), "asymmetric dense peers admitted");
        require(!agrees(dense, resolve({{"STRIX_DS41_FAST_DENSE", "1"}, {"STRIX_DS41_FAST_DENSE_MIN_ROWS", "4096"}})),
                "asymmetric thresholds admitted");
        const auto bundle = resolve({{"STRIX_DS41_FAST_PREFILL", "1"}});
        require(!agrees(off, bundle) && !agrees(bundle, off), "asymmetric bundle admitted");
        require(exported(bundle, "STRIX_DS41_FAST_ATTN") == "2" && exported(bundle, "STRIX_DS41_FAST_MOE") == "1" &&
                exported(bundle, "STRIX_DS41_FAST_DENSE") == "1", "bundle not expanded before export");
        Env disabled{{"STRIX_DS41_FAST_PREFILL", "1"}, {"STRIX_DS41_FAST_ATTN", "0"},
                     {"STRIX_DS41_FAST_MOE", "0"}, {"STRIX_DS41_FAST_DENSE", "0"}};
        for (bool fast : {false, true}) {
            const auto r = resolve(disabled, fast);
            for (const char* key : {"STRIX_DS41_FAST_ATTN", "STRIX_DS41_FAST_MOE", "STRIX_DS41_FAST_DENSE"})
                require(exported(r, key) == "0", "explicit OFF lost to bundle/profile");
        }
        require(agrees(dense, resolve({{"STRIX_DS41_FAST_DENSE", "1"}, {"STRIX_DS41_FAST_DENSE_MIN_ROWS", "00128"}})),
                "equivalent thresholds disagree");
        require(agrees(off, resolve({{"STRIX_DS41_FAST_DENSE", "invalid"}, {"STRIX_DS41_FAST_PREFILL", "01"}})),
                "effective OFF not normalized");
        const auto negative = resolve({{"STRIX_DS41_FAST_DENSE_MIN_ROWS", "-1"}});
        const auto value = exported(negative, "STRIX_DS41_FAST_DENSE_MIN_ROWS");
        require(strix::ds41::fast_profile::dense_min_rows_value(value.c_str()) == static_cast<unsigned>(-1),
                "canonical unsigned export does not round trip");
        require(agrees(negative, resolve({{"STRIX_DS41_FAST_DENSE_MIN_ROWS", value}})), "unsigned normalization differs");
        // ds41-engram-f16wmma-v1: explicit only (the bundle does not select it), exported, and a peer without it disagrees
        const auto engram = resolve({{"STRIX_DS41_FAST_ENGRAM", "1"}});
        require(!agrees(off, engram) && !agrees(engram, off), "asymmetric Engram peers admitted");
        require(exported(engram, "STRIX_DS41_FAST_ENGRAM") == "1", "Engram lever not exported");
        require(std::none_of(bundle.set_env.begin(), bundle.set_env.end(),
                             [](const auto& kv) { return kv.first == "STRIX_DS41_FAST_ENGRAM" && kv.second == "1"; }),
                "bundle selects the unqualified Engram lever");
        require(exported(resolve({{"STRIX_DS41_FAST_ENGRAM", "1"}}, true), "STRIX_DS41_FAST_ENGRAM") == "1" &&
                    exported(resolve({{"STRIX_DS41_FAST_ENGRAM", "yes"}}), "STRIX_DS41_FAST_ENGRAM") == "0",
                "Engram lever not canonical (explicit value kept under --profile fast, anything but 1 is 0)");
        require(agrees(off, resolve({{"STRIX_DS41_FAST_ENGRAM", "0"}})), "explicit Engram OFF changes the pair identity");
        // ONE RESOLVER: the server and the prefill CLI resolve the same request to the same Engram value
        for (const pd::App app : {pd::App::serve, pd::App::prefill}) {
            const auto r = pd::resolve(app, {}, [](const char* key) -> const char* {
                return std::string(key) == "STRIX_DS41_FAST_ENGRAM" ? "1" : std::string(key) == "STRIX_DS41_BASELINE" ? "1" : nullptr;
            }, {}, {});
            require(exported(r, "STRIX_DS41_FAST_ENGRAM") == "1", "serve / prefill CLI resolve a different Engram value");
        }
        // ds41-shared-f16-v1: explicit only, exported, canonical, a peer without it disagrees; ONE RESOLVER across serve / CLI
        const auto shared = resolve({{"STRIX_DS41_FAST_SHARED", "1"}});
        require(!agrees(off, shared) && !agrees(shared, off), "asymmetric shared-expert peers admitted");
        require(exported(shared, "STRIX_DS41_FAST_SHARED") == "1", "shared-expert lever not exported");
        require(std::none_of(bundle.set_env.begin(), bundle.set_env.end(),
                             [](const auto& kv) { return kv.first == "STRIX_DS41_FAST_SHARED" && kv.second == "1"; }),
                "bundle selects the unqualified shared-expert lever");
        require(exported(resolve({{"STRIX_DS41_FAST_SHARED", "yes"}}), "STRIX_DS41_FAST_SHARED") == "0", "shared-expert lever not canonical");
        require(agrees(off, resolve({{"STRIX_DS41_FAST_SHARED", "0"}})), "explicit shared-expert OFF changes the pair identity");
        for (const pd::App app : {pd::App::serve, pd::App::prefill}) {
            const auto r = pd::resolve(app, {}, [](const char* key) -> const char* {
                return std::string(key) == "STRIX_DS41_FAST_SHARED" ? "1" : std::string(key) == "STRIX_DS41_BASELINE" ? "1" : nullptr;
            }, {}, {});
            require(exported(r, "STRIX_DS41_FAST_SHARED") == "1", "serve / prefill CLI resolve a different shared-expert value");
        }
        // ds41-moe-f16wmma-v1: the same explicit-lever contract
        const auto m16 = resolve({{"STRIX_DS41_FAST_MOE_F16", "1"}});
        require(!agrees(off, m16) && exported(m16, "STRIX_DS41_FAST_MOE_F16") == "1", "MoE FP16 lever not exported / admitted asymmetric");
        require(std::none_of(bundle.set_env.begin(), bundle.set_env.end(),
                             [](const auto& kv) { return kv.first == "STRIX_DS41_FAST_MOE_F16" && kv.second == "1"; }),
                "bundle selects the unqualified MoE FP16 lever");
        for (const pd::App app : {pd::App::serve, pd::App::prefill}) {
            const auto r = pd::resolve(app, {}, [](const char* key) -> const char* {
                return std::string(key) == "STRIX_DS41_FAST_MOE_F16" ? "1" : std::string(key) == "STRIX_DS41_BASELINE" ? "1" : nullptr;
            }, {}, {});
            require(exported(r, "STRIX_DS41_FAST_MOE_F16") == "1", "serve / prefill CLI resolve a different MoE FP16 value");
        }
        const auto report = pd::describe(pd::App::rank, bundle);
        require(report.find("STRIX_DS41_FAST_DENSE=1") != std::string::npos &&
                report.find("STRIX_DS41_FAST_DENSE_MIN_ROWS=128") != std::string::npos, "effective report omits dense policy");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
