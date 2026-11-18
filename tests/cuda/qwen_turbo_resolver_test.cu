// ONE RESOLVER (Qwen CUDA): the serve and CLI spellings of the same request resolve to the same effective values
// through strix/cuda/qwen_turbo.hpp; TURBO identities, refusals (never a silent native fallback), prefix identity
// namespace, and the HIP / INT4 identities unchanged.
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "strix/cuda/qwen_prefix_identity.hpp"
#include "strix/cuda/qwen_turbo.hpp"

namespace qt = strix::cuda::qwen_turbo;
namespace tb = strix::profile::turbo;
static int failures = 0;
#define CHECK(c, ...) do { if (!(c)) { ++failures; std::printf("FAIL %d: ", __LINE__); std::printf(__VA_ARGS__); std::printf("\n"); } } while (0)

static qt::Request parse(std::vector<std::string> args, std::vector<std::string>* rest = nullptr) {
    std::vector<char*> argv{const_cast<char*>("x")};
    for (auto& a : args) argv.push_back(a.data());
    qt::Request r;
    r.inputs.env = [](const char*) -> const char* { return nullptr; };
    for (int i = 1; i < static_cast<int>(argv.size()); ++i)
        if (!qt::take_option(static_cast<int>(argv.size()), argv.data(), i, r) && rest) rest->push_back(argv[i]);
    return r;
}
static bool throws(std::vector<std::string> args) {
    try { (void)qt::resolve(parse(std::move(args))); } catch (const std::exception&) { return true; }
    return false;
}

int main() {
    // Serve spelling (options first) and CLI spelling (options between positionals, = form) agree.
    std::vector<std::string> rest;
    const auto serve = qt::resolve(parse({"--profile", "turbo", "--turbo-weights", "onfly"}));
    const auto cli = qt::resolve(parse({"tables", "--profile=turbo", "rope", "--turbo-weights=onfly", "--turbo-weight-bits", "4"}, &rest));
    CHECK(rest.size() == 2 && rest[0] == "tables" && rest[1] == "rope", "positionals kept");
    CHECK(serve.effective_config() == cli.effective_config(), "serve != CLI:\n%s---\n%s", serve.effective_config().c_str(),
          cli.effective_config().c_str());
    CHECK(serve.onfly_nvfp4 && cli.onfly_nvfp4, "onfly effective");
    CHECK(serve.resolution.weights.identity() == "onfly-nvfp4-rtn-g16", "identity %s", serve.resolution.weights.identity().c_str());
    CHECK(serve.cache_namespace == cli.cache_namespace && serve.cache_namespace.find("weights=onfly-nvfp4-rtn-g16") != std::string::npos &&
          serve.cache_namespace.rfind("turbo=qwen-turbo-v1", 0) == 0, "namespace %s", serve.cache_namespace.c_str());
    strix::cuda::qwen_prefill::Config a, b;
    qt::apply(serve, a);
    qt::apply(cli, b);
    CHECK(a.turbo_onfly_nvfp4 && b.turbo_onfly_nvfp4 && a.turbo_namespace == b.turbo_namespace, "apply");
    // Turbo with native weights: no conversion, own namespace.
    const auto native = qt::resolve(parse({"--profile", "turbo"}));
    CHECK(!native.onfly_nvfp4 && native.resolution.weights.identity() == "native" &&
          native.cache_namespace.find("weights=native") != std::string::npos, "turbo native");
    // Default: nothing changes, no namespace, Config untouched.
    const auto def = qt::resolve(parse({}));
    strix::cuda::qwen_prefill::Config c;
    qt::apply(def, c);
    CHECK(!c.turbo_onfly_nvfp4 && c.turbo_namespace.empty() && def.cache_namespace.empty(), "default untouched");
    // Refusals: onfly outside turbo, 3-bit, converted (owner's OK), HIP lever options.
    CHECK(throws({"--turbo-weights", "onfly"}), "onfly needs --profile turbo");
    CHECK(throws({"--profile", "turbo", "--turbo-weights", "onfly", "--turbo-weight-bits", "3"}), "3-bit refused");
    CHECK(throws({"--profile", "turbo", "--turbo-weights", "converted", "--turbo-weights-pack", "x.gguf"}), "converted refused");
    for (const auto& row : strix::qwen::profile::levers()) {
        if (row.native_option || row.option.empty()) continue;
        bool refused = false;
        try { (void)parse({"--profile", "turbo", std::string(row.option), "1"}); } catch (const std::invalid_argument&) { refused = true; }
        CHECK(refused, "HIP lever option %s accepted on CUDA", std::string(row.option).c_str());
        break;
    }
    // INT4 identity of the HIP adapter unchanged (element axis default int).
    tb::Weights w;
    w.effective = tb::Source::onfly; w.effective_bits = 4; w.method = "rtn-g32";
    CHECK(w.identity() == "onfly-int4-rtn-g32", "int identity %s", w.identity().c_str());
    w.effective = tb::Source::converted;
    CHECK(w.identity() == "converted-int4", "converted identity %s", w.identity().c_str());
    // Prefix identity: the namespace is a field under turbo only.
    strix::cuda::qwen_prefill::Config p;
    const auto tables = std::filesystem::temp_directory_path() / "qwen_resolver_test_tables";
    std::filesystem::create_directories(tables);
    p.tables_dir = tables.string();
    const auto f0 = strix::cuda::qwen_prefix::canonical_identity(strix::cuda::qwen_prefix::identity_fields(p, 256));
    p.turbo_namespace = serve.cache_namespace;
    const auto f1 = strix::cuda::qwen_prefix::canonical_identity(strix::cuda::qwen_prefix::identity_fields(p, 256));
    CHECK(f0 != f1 && f1.find("turbo.namespace") != std::string::npos && f0.find("turbo") == std::string::npos, "prefix identity");
    std::printf("%s", serve.effective_config().c_str());
    std::printf(failures ? "qwen_turbo_resolver_test: %d FAILURES\n" : "qwen_turbo_resolver_test: PASS\n", failures);
    return failures ? 1 : 0;
}
