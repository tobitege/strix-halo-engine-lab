#include "strix/deepseek41_prefix_cache.hpp"
#include "strix/ds41_fast_profile.hpp"
#include "strix/deepseek41_serve_cache_policy.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

void require(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void env(const char* name, const char* value) {
#ifdef _WIN32
    require(_putenv_s(name, value) == 0, "set environment");
#else
    require((*value ? setenv(name, value, 1) : unsetenv(name)) == 0, "set environment");
#endif
}
int main(int argc, char** argv) {
    try {
        namespace ds = strix::deepseek41;
        namespace fast = strix::ds41::fast_profile;
        env("STRIX_DS41_FAST_PREFILL", "0");
        env("STRIX_DS41_FAST_ATTN", "");
        env("STRIX_DS41_FAST_MOE", "");
        env("STRIX_DS41_FAST_DENSE", "1");
        env("STRIX_DS41_FAST_DENSE_MIN_ROWS", "");
        env("STRIX_DS41_FAST_SHARED", "0");
        if (argc > 1) {
            env("STRIX_DS41_FAST_DENSE", "0");
            const auto exact = ds::prefix_namespace_from_env(false);
            require(ds::prefix_continuation_decline().empty(), "exact continuation declined");
            env("STRIX_DS41_FAST_SHARED", "1");
            if (std::string(argv[1]) == "shared-continuation") {
                const auto why = ds::prefix_continuation_decline();
                require(why.find("ds41-shared-f16-v1") != std::string::npos &&
                        why.find("dense_min_rows=128") != std::string::npos, "shared-only continuation admitted");
                for (const bool explicit_tiers : {false, true}) {
                    ds::ServeCachePolicy policy;
                    if (explicit_tiers) {
                        require(policy.parse("--prefix-cache-mb", "512") && policy.parse("--prefix-cache-disk-mb", "1024") &&
                                policy.parse("--prefix-cache-dir", "cache") && policy.parse("--prefix-before-tail") &&
                                policy.parse("--prefix-session-snapshots") && policy.parse("--prefix-inrun-snapshots"), "parse cache tiers");
                    }
                    policy.continuation_decline = why;
                    policy.resolve(true, false, false);
                    require(policy.prefix_cache_mb == 0 && policy.prefix_disk_mb == 0 && policy.prefix_dir.empty() &&
                            !policy.prefix_before_tail && !policy.prefix_session && !policy.prefix_inrun, "shared cache tier stayed active");
                }
                env("STRIX_DS41_FAST_SHARED", "0");
                require(ds::prefix_continuation_decline().empty(), "shared OFF did not restore continuation");
            } else {
                const auto shared = ds::prefix_namespace_from_env(false);
                require(shared.tag != exact.tag, "shared arithmetic aliases exact namespace");
                env("STRIX_DS41_FAST_DENSE_MIN_ROWS", "4096");
                const auto large_shared = ds::prefix_namespace_from_env(false);
                require(shared.tag != large_shared.tag, "shared-only thresholds alias namespace");
                require(large_shared.text.find("dense_min_rows=4096;") != std::string::npos, "shared threshold not reported");
                env("STRIX_DS41_FAST_DENSE_MIN_ROWS", "004096");
                require(ds::prefix_namespace_from_env(false).tag == large_shared.tag, "shared threshold not normalized");
                env("STRIX_DS41_FAST_SHARED", "0");
                require(ds::prefix_namespace_from_env(false).tag == exact.tag, "inactive shared threshold changes namespace");
            }
            return 0;
        }
        const auto initial = ds::prefix_namespace_from_env(false);
        require(fast::dense_min_rows() == 128U, "default dispatch threshold");
        env("STRIX_DS41_FAST_DENSE_MIN_ROWS", "4096");
        const auto large = ds::prefix_namespace_from_env(false);
        require(fast::dense_min_rows() == 4096U, "explicit dispatch threshold");
        require(initial.tag != large.tag, "different dense arithmetic aliases the prefix namespace");
        require(large.text.find("dense_min_rows=4096;") != std::string::npos, "effective threshold not reported");
        const std::filesystem::path model = "pr584-model.bin";
        { std::ofstream fixture(model, std::ios::binary); fixture << "prefix identity fixture"; }
        const bool distinct_models = ds::prefix_model_id(model, 1U, initial) != ds::prefix_model_id(model, 1U, large);
        std::filesystem::remove(model);
        require(distinct_models, "persistent model IDs alias different thresholds");
        env("STRIX_DS41_FAST_DENSE_MIN_ROWS", "004096");
        require(ds::prefix_namespace_from_env(false).tag == large.tag, "threshold not normalized");
        env("STRIX_DS41_FAST_DENSE_MIN_ROWS", "0");
        require(fast::dense_min_rows() == 0U, "explicit zero threshold");
        env("STRIX_DS41_FAST_DENSE_MIN_ROWS", "-1");
        require(fast::dense_min_rows() == std::numeric_limits<unsigned>::max(), "legacy unsigned threshold");
        env("STRIX_DS41_FAST_DENSE", "0");
        const auto off = ds::prefix_namespace_from_env(false);
        env("STRIX_DS41_FAST_DENSE_MIN_ROWS", "128");
        require(ds::prefix_namespace_from_env(false).tag == off.tag, "inactive threshold changes exact namespace");
        env("STRIX_DS41_FAST_PREFILL", "1");
        require(!fast::dense_on() && fast::attention_flash() && fast::moe_on(), "explicit OFF must beat bundle");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
