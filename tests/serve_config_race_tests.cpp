// Linux linker wrappers replace the pathname immediately after validation.
// Both stat (the old implementation) and fstat are wrapped, so the same fixture
// deterministically fails the old check-then-open reader without a racing thread.
#include "strix/serve/serve_config.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>

namespace {
namespace fs = std::filesystem;
std::string target;
bool armed = false, swapped = false;
int failures = 0;
void check(bool ok, const char* message) {
    if (!ok) { ++failures; std::cerr << "FAIL: " << message << '\n'; }
}
void swap_path() {
    if (!armed) return;
    armed = false;
    fs::rename(target, target + ".validated");
    std::ofstream(target) << "port = 1234\n";
    if (::chmod(target.c_str(), 0666) != 0) throw std::runtime_error("chmod replacement failed");
    swapped = true;
}
}

extern "C" int __real_stat(const char*, struct stat*);
extern "C" int __real_fstat(int, struct stat*);
extern "C" int __wrap_stat(const char* path, struct stat* st) {
    const int result = __real_stat(path, st);
    if (result == 0) swap_path();
    return result;
}
extern "C" int __wrap_fstat(int fd, struct stat* st) {
    const int result = __real_fstat(fd, st);
    if (result == 0) swap_path();
    return result;
}

int main() try {
    namespace cf = strix::serve::config_file;
    constexpr cf::Option options[] = {{"port", "--port", cf::Kind::value}};
    const auto dir = fs::current_path() / "serve-config-race-fixture";
    fs::create_directories(dir);
    for (const bool symlink : {false, true}) {
        const auto original = dir / (symlink ? "symlink-original.toml" : "original.toml");
        std::ofstream(original) << "port = 9000\n";
        check(::chmod(original.c_str(), 0600) == 0, "trusted mode set");
        target = (dir / (symlink ? "link.toml" : "serve.toml")).string();
        if (symlink) fs::create_symlink(original, target);
        else fs::rename(original, target);
        const char* args[] = {"serve", "--serve-config", target.c_str()};
        armed = true;
        swapped = false;
        const auto result = cf::expand(3, args, options);
        check(swapped, "validation wrapper replaced the pathname");
        check(result.argv == std::vector<std::string>({"serve", "--port", "9000"}),
            "read uses validated descriptor after pathname or symlink replacement");
    }
    const auto directory = dir.string();
    const char* args[] = {"serve", "--serve-config", directory.c_str()};
    bool refused = false;
    try { (void)cf::expand(3, args, options); }
    catch (const std::invalid_argument& e) { refused = std::string(e.what()).find("regular file") != std::string::npos; }
    check(refused, "non-regular config refused");
    fs::remove_all(dir);
    std::cout << "serve config race tests: failures=" << failures << '\n';
    return failures ? 1 : 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
