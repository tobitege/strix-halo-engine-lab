// Shared --serve-config TOML expander (src/serve/serve_config.cpp): a file and the equivalent command line resolve
// to the same effective options through the real CLI parser; the CLI wins; lists replace; relative paths; named
// refusals with file:line; secrets refused or redacted; untrusted files refused (POSIX). Host-only fixtures.
#include "../apps/hip_decode_options.hpp"
#include "strix/serve/serve_config.hpp"

#include <filesystem>
#include <algorithm>
#include <fstream>
#include <iostream>

#if !defined(_WIN32)
#include <sys/stat.h>
#endif

using namespace strix::decode_cli;
namespace cf = strix::serve::config_file;

namespace {
int failures = 0;
void check(bool ok, const char* what) {
    if (!ok) { ++failures; std::cerr << "FAIL: " << what << '\n'; }
}
std::string refusal(const std::string& text, const std::vector<std::string>& args, std::span<const cf::Option> table) {
    try { (void)cf::expand_text(text, "/cfg/serve.toml", args, table); } catch (const std::invalid_argument& e) { return e.what(); }
    return {};
}
#if !defined(STRIX_TEST_CONFIG_EXPANSION_ONLY)
Options parse(std::vector<std::string> args) {
    std::vector<char*> argv;
    for (auto& arg : args) argv.push_back(arg.data());
    return parse_options(static_cast<int>(argv.size()), argv.data());
}
#endif
std::string join(const std::vector<std::string>& v) { std::string s; for (const auto& a : v) s += a + " "; return s; }
}  // namespace

int main() try {
    const auto table = hip_serve_config_options();
    // 1. TOML and the equivalent argv give the same argv, hence the same parser result (one resolver).
    const std::string file = "context = 4096\n[prefill]\nexact_fp16 = true\ndisable = [\"hc_output_wmma\"]\n[http]\nport = 9001\nbind = \"0.0.0.0\"\nresponses = false\napi_key_file = \"keys/api.key\"\n";
    const auto expanded = cf::expand_text(file, "/cfg/serve.toml", {"decoder", "model.gguf", "1"}, table);
    const std::vector<std::string> by_hand{"decoder", "model.gguf", "1", "--context", "4096", "--http-api-key-file",
        std::filesystem::absolute("/cfg/keys/api.key").generic_string(),
        "--http-bind", "0.0.0.0", "--http-port", "9001", "--http-responses", "off", "--prefill-disable", "hc_output_wmma", "--prefill-exact-fp16"};
    check(expanded.argv == by_hand, "file expands to the equivalent command line");
    if (expanded.argv != by_hand) std::cerr << "  got: " << join(expanded.argv) << '\n';
#if !defined(STRIX_TEST_CONFIG_EXPANSION_ONLY)
    const auto a = parse(by_hand), b = parse(expanded.argv);
    check(a.context_tokens == 4096 && a.context_tokens == b.context_tokens && a.http_port == 9001 && a.http_port == b.http_port &&
        a.http_bind == b.http_bind && !b.http_responses && a.http_responses == b.http_responses && a.http_api_key_file == b.http_api_key_file &&
        b.prefill_exact_fp16 && a.prefill_exact_fp16 == b.prefill_exact_fp16 && a.prefill_disabled != 0U && a.prefill_disabled == b.prefill_disabled,
        "argv and TOML resolve to the same effective options");
#endif
    // 2. the command line wins, and a list given on the command line replaces the file's whole list
    const auto cli = cf::expand_text("http.port = 1\nprefill.disable = [\"hc_output_wmma\"]\n", "/cfg/serve.toml",
        {"decoder", "model.gguf", "--http-port", "2", "--prefill-disable=x"}, table);
    check(join(cli.argv) == "decoder model.gguf --http-port 2 --prefill-disable=x ", "command line wins; lists replace");
    check(cli.effective.find("http.port = \"2\"  # command line") != std::string::npos && cli.effective.find("= \"1\"") == std::string::npos,
        "effective output names the winning origin");
    const auto from_file = cf::expand_text("http.port = 7\n", "/cfg/serve.toml", {"decoder", "m"}, table);
    check(from_file.effective.find("http.port = \"7\"  # /cfg/serve.toml:1") != std::string::npos, "effective output names file:line");
    // 3. refusals: syntax/duplicate (line/column from the reader), unknown key, type, secrets, conflicting --config
    check(refusal("http.port = 1\nhttp.port = 2\n", {"d"}, table).find("/cfg/serve.toml") == 0, "duplicate key refused with the file named");
    check(refusal("context = 1\n\nhttp.prot = 2\n", {"d"}, table).starts_with("/cfg/serve.toml:3: http.prot: unknown"), "unknown key refused with its line");
    check(refusal("[http]\nresponses = \"maybe\"\n", {"d"}, table).starts_with("/cfg/serve.toml:2: http.responses"), "wrong type refused with its line");
    check(refusal("http.api_key = \"sk-123\"\n", {"d"}, table).find("inline credentials") != std::string::npos, "inline secret refused");
    check(refusal("prefill.exact_fp16 = true\n", {"d", "m", "--config", "p.toml"}, table).find("--config") != std::string::npos,
        "prefill settings in both the file and --config refused (no silent collision)");
    check(refusal("prefill.disable = \"x\"\n", {"d"}, table).find("expects an array") != std::string::npos, "list type checked");
    static constexpr cf::Option secret_table[] = {{"api_key", "--api-key", cf::Kind::secret}, {"port", "--port", cf::Kind::value}};
    const auto redacted = cf::expand_text("port = 1\n", "/cfg/s.toml", {"srv", "--api-key", "sk-live-9"}, secret_table);
    check(redacted.effective.find("sk-live-9") == std::string::npos && redacted.effective.find("<redacted>") != std::string::npos,
        "secrets from the command line are redacted in the effective output");
    check(refusal("api_key = \"x\"\n", {"srv"}, secret_table).find("inline credentials") != std::string::npos, "declared secret refused in the file");
    const std::vector<std::string> family_tail{"--", "--http-port", "1234", "--config", "family.toml", "--http-api-key", "family-secret"};
    std::vector<std::string> boundary_args{"decoder", "model.gguf"};
    boundary_args.insert(boundary_args.end(), family_tail.begin(), family_tail.end());
    const auto boundary = cf::expand_text("http.port = 9000\nprefill.exact_fp16 = true\n", "cfg/serve.toml", boundary_args, table);
    check(join(boundary.argv).find("--http-port 9000") != std::string::npos &&
        std::equal(family_tail.begin(), family_tail.end(), boundary.argv.end() - static_cast<std::ptrdiff_t>(family_tail.size())),
        "family tail neither overrides shell values nor triggers config conflicts");
    check(boundary.effective.find("1234") == std::string::npos && boundary.effective.find("family-secret") == std::string::npos,
        "effective options exclude family tail");
    const auto relative = cf::expand_text("mtp_bundle = \"bundles/b.toml\"\n", "cfg/serve.toml", {"decoder", "model.gguf"}, table);
    check(relative.argv.back() == std::filesystem::absolute("cfg/bundles/b.toml").lexically_normal().generic_string(),
        "relative config filename produces an absolute asset path");
    // 4. real files: relative paths, --print-serve-config, no config = unchanged argv, untrusted file
    const auto dir = std::filesystem::temp_directory_path() / "strix-serve-config-test";
    std::filesystem::create_directories(dir);
    const auto path = (dir / "serve.toml").string();
    { std::ofstream(path) << "mtp_bundle = \"bundles/b.toml\"\n"; }
#if !defined(_WIN32)
    ::chmod(path.c_str(), 0600);
#endif
    const std::string p = path;
    const char* argv1[] = {"decoder", "model.gguf", "--serve-config", p.c_str(), "--print-serve-config"};
    const auto real = cf::expand(5, argv1, table);
    check(real.used && real.print && real.argv.size() == 4U && real.argv[3] == (dir / "bundles/b.toml").lexically_normal().generic_string(),
        "relative path resolves against the file directory");
    const char* argv2[] = {"decoder", "model.gguf", "--http-port", "5"};
    const auto none = cf::expand(4, argv2, table);
    check(!none.used && none.argv == std::vector<std::string>({"decoder", "model.gguf", "--http-port", "5"}), "no --serve-config leaves argv unchanged");
    const char* tail_only[] = {"decoder", "--", "--serve-config", "missing.toml", "--print-serve-config", ""};
    const auto untouched = cf::expand(6, tail_only, table);
    check(!untouched.used && !untouched.print && untouched.argv == std::vector<std::string>(tail_only, tail_only + 6),
        "config extraction preserves all family arguments including empty strings");
    const char* with_tail[] = {"decoder", "--serve-config", p.c_str(), "--", "--serve-config=missing.toml", "--print-serve-config"};
    const auto extracted = cf::expand(6, with_tail, table);
    check(extracted.used && !extracted.print && extracted.argv.back() == "--print-serve-config" &&
        extracted.argv[extracted.argv.size() - 2] == "--serve-config=missing.toml", "family config flags remain verbatim with a shell config");
#if !defined(_WIN32)
    ::chmod(path.c_str(), 0666);
    bool refused = false;
    try { (void)cf::expand(5, argv1, table); } catch (const std::invalid_argument& e) { refused = std::string(e.what()).find("writable") != std::string::npos; }
    check(refused, "group/world-writable serve config refused");
#endif
    std::filesystem::remove_all(dir);
    std::cout << "serve config tests: failures=" << failures << '\n';
    return failures ? 1 : 0;
} catch (const std::exception& e) {
    std::cerr << "serve config tests: " << e.what() << '\n';
    return 1;
}
