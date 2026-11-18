#pragma once

// Eager GPU code-object loading, ONE home for every HIP binary (host only, no HIP headers).
//
// With the HIP default (deferred loading) a code object is loaded at the first launch of one of its kernels. On a
// unified-memory APU that load allocates the code segment as a KFD userptr BO on host pages (seen with the kfdevict
// probe: 22 MB / 7 MB / 6.6 MB BOs made by the first MoE / dense / trellis launches inside the first timed prefill),
// so the first (cold) request pays the loads and the new BOs join the host memory that page migration can evict.
// HIP_ENABLE_DEFERRED_LOADING=0 loads every code object while the runtime registers them, i.e. during static
// initialisation, before main(): a value set in main() is too late for the running image, so the process restarts its
// own image once with the value in its environment (same pid, same arguments; Windows has no exec and runs the program
// again as a child the first process waits for, strix/process_restart.hpp); the restarted image finds the value
// set and does not restart again. Kernel arithmetic does not change (bitwise): only when the code is loaded.
//
// Policy (STRIX_EAGER_CODE_LOAD): unset = the binary's default; "0" = deferred (HIP default); anything else = eager.
// An explicitly supplied HIP_ENABLE_DEFERRED_LOADING always wins, even empty, and is never overwritten.

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include "strix/env_flags.hpp"
#include "strix/process_environment.hpp"
#include "strix/process_restart.hpp"
#include <vector>

#if defined(__unix__)
#include <unistd.h>
#endif
#if defined(__linux__)
#include <sys/prctl.h>
#endif

namespace strix::device {

// Returns true when it set HIP_ENABLE_DEFERRED_LOADING=0 (the caller restarts the image for it to apply).
template<class Get, class Put>
inline bool apply_eager_code_loading(bool enabled, Get get, Put put) {
    const char* policy = get("STRIX_EAGER_CODE_LOAD");
    if (!enabled || (policy && policy[0] == '0') || get("HIP_ENABLE_DEFERRED_LOADING")) return false;
    if (!put("HIP_ENABLE_DEFERRED_LOADING", "0"))
        throw std::runtime_error("cannot configure HIP eager code loading");
    return true;
}

// The restart must repeat the process's own arguments: a runner that wraps an engine entry forwards a reduced argv,
// and restarting with that would drop the runner's options.
// Splits /proc/self/cmdline (NUL-terminated arguments); empty when the buffer is not complete.
inline std::vector<std::string> split_process_arguments(std::string_view cmdline) {
    std::vector<std::string> arguments;
    if (cmdline.empty() || cmdline.back() != '\0') return arguments;
    for (std::size_t begin = 0; begin < cmdline.size();) {
        const auto end = cmdline.find('\0', begin);
        arguments.emplace_back(cmdline.substr(begin, end - begin));
        begin = end + 1U;
    }
    return arguments;
}

// /proc/self/exe keeps the running inode even after a deployment replaces or unlinks its pathname.
// Linux names that restarted process "exe"; restore the executable's name separately, never as an exec target.
inline void restore_own_image_name() {
#if defined(__linux__)
    char comm[16]{};
    if (::prctl(PR_GET_NAME, comm) != 0 || std::strcmp(comm, "exe") != 0) return;
    char self[4096];
    const auto n = ::readlink("/proc/self/exe", self, sizeof(self) - 1);
    if (n <= 0) return;
    std::string path(self, static_cast<std::size_t>(n));
    constexpr std::string_view deleted = " (deleted)";
    if (path.ends_with(deleted)) path.resize(path.size() - deleted.size());
    const auto name = path.substr(path.find_last_of('/') + 1U);
    (void)::prctl(PR_SET_NAME, name.c_str());
#endif
}

// Returns the restart error only on failure (process_restart::failure_text names it). Success replaces this image
// (Linux) or ends this process with the exit code of the restarted one (Windows, strix/process_restart.hpp) and never
// returns.
inline int restart_own_image(char** argv) {
    std::fflush(nullptr);
#if defined(__unix__)
    std::ifstream cmdline("/proc/self/cmdline", std::ios::binary);
    const auto original = split_process_arguments(
        std::string((std::istreambuf_iterator<char>(cmdline)), std::istreambuf_iterator<char>()));
    std::vector<char*> restart;
    for (const auto& argument : original) restart.push_back(const_cast<char*>(argument.c_str()));
    restart.push_back(nullptr);
    return process_restart::run_again(original.empty() ? argv : restart.data());
#elif defined(_WIN32)
    return process_restart::run_again(argv);
#else
    (void)argv;
    return ENOSYS;
#endif
}

inline int& code_load_restart_error() {
    static int error = 0;
    return error;
}

// Own the inserted value until exec succeeds. A failed exec must not certify this already-registered image as eager.
template<class Get, class Put, class Erase, class Restart>
inline int try_eager_code_load_restart(bool enabled, Get get, Put put, Erase erase, Restart restart) {
    if (!apply_eager_code_loading(enabled, get, put)) return 0;
    const int error = restart();
    code_load_restart_error() = error;
    if (!erase("HIP_ENABLE_DEFERRED_LOADING"))
        throw std::runtime_error("cannot roll back HIP eager code loading after failed restart");
    return error;
}

inline void restart_with_eager_code_loading(bool enabled, char** argv) {
    const int error = try_eager_code_load_restart(enabled,
        [](const char* name) { return strix::envcfg::text(name); },
        [](const char* name, const char* value) { return process_environment::set(name, value, false); },
        [](const char* name) { return process_environment::unset(name); },
        [&] { return restart_own_image(argv); });
    if (error) std::fprintf(stderr, "eager code loading: restart failed (%s); continuing with deferred loading\n",
        process_restart::failure_text(error).c_str());
}

// Who decides for this binary. shared = the start gate applies the default below; engine = the binary resolves its own
// policy later (it has engine options that can decline eager loading) and calls restart_with_eager_code_loading.
enum class CodeLoad { shared, engine };

// The start gate's call: eager unless STRIX_EAGER_CODE_LOAD=0 or HIP_ENABLE_DEFERRED_LOADING is already set.
inline void eager_code_load_or_restart(char** argv, CodeLoad who = CodeLoad::shared) {
    restore_own_image_name();
    if (who != CodeLoad::shared) return;
    restart_with_eager_code_loading(true, argv);
}

// Receipt line of the resolved mode (the restarted image prints it): "code load: eager|deferred (<why>)".
inline std::string code_load_receipt(CodeLoad who = CodeLoad::shared) {
    const char* hip = strix::envcfg::text("HIP_ENABLE_DEFERRED_LOADING");
    const char* policy = strix::envcfg::text("STRIX_EAGER_CODE_LOAD");
    if (code_load_restart_error())
        return "code load: deferred (restart failed: " + process_restart::failure_text(code_load_restart_error()) + ")";
    const bool eager = hip && std::string_view(hip) == "0";
    std::string why = policy && policy[0] == '0' ? "STRIX_EAGER_CODE_LOAD=0"
        : who == CodeLoad::engine ? "engine policy"
        : hip ? (eager ? "HIP_ENABLE_DEFERRED_LOADING=0" : "HIP_ENABLE_DEFERRED_LOADING set") : "restart unavailable";
    return std::string("code load: ") + (eager ? "eager" : "deferred") + " (" + why + ")";
}

} // namespace strix::device
