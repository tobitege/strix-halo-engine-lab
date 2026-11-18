#include "strix/device/eager_code_load.hpp"

#include <filesystem>
#include <iostream>
#include <map>

#if defined(__linux__)
#include <linux/filter.h>
#include <linux/seccomp.h>
#include <sys/syscall.h>
#endif

namespace {
using namespace strix::device;
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

void transaction(bool explicit_hip) {
    for (const char* value : {"", "0", "1"}) {
        std::map<std::string, std::string> env;
        if (explicit_hip) env["HIP_ENABLE_DEFERRED_LOADING"] = value;
        bool restarted = false;
        bool erased = false;
        const int error = try_eager_code_load_restart(true,
            [&](const char* name) -> const char* {
                const auto it = env.find(name);
                return it == env.end() ? nullptr : it->second.c_str();
            },
            [&](const char* name, const char* text) { env[name] = text; return true; },
            [&](const char* name) { erased = true; return env.erase(name) == 1U; },
            [&] {
                restarted = true;
                require(env.at("HIP_ENABLE_DEFERRED_LOADING") == "0", "exec must inherit eager value");
                return EACCES;
            });
        if (explicit_hip) {
            require(error == 0 && !restarted && !erased, "explicit HIP must skip exec and rollback");
            require(env.at("HIP_ENABLE_DEFERRED_LOADING") == value, "explicit HIP must survive unchanged");
        } else {
            require(error == EACCES && restarted && erased, "exec failure must be returned and rolled back");
            require(env.empty(), "failed exec must remove its inserted value");
            const auto receipt = code_load_receipt();
            require(receipt.starts_with("code load: deferred (restart failed: "), "receipt must report failed exec");
            require(receipt.find(std::strerror(EACCES)) != std::string::npos, "receipt must include exec error");
        }
    }
}

#if defined(__linux__)
void native_restart(const std::string& mode, int argc, char** argv) {
    if (mode == "denied") {
        // Deny the actual production execv, not just the policy/environment callbacks.
        sock_filter filter[] = {
            BPF_STMT(BPF_LD | BPF_W | BPF_ABS, 0),
            BPF_JUMP(BPF_JMP | BPF_JEQ | BPF_K, __NR_execve, 0, 1),
            BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | EACCES),
            BPF_STMT(BPF_RET | BPF_K, SECCOMP_RET_ALLOW),
        };
        sock_fprog program{static_cast<unsigned short>(sizeof(filter) / sizeof(filter[0])), filter};
        require(::prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) == 0, "set no-new-privileges");
        require(::prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, &program) == 0, "install exec denial");
        eager_code_load_or_restart(argv);
        require(strix::envcfg::text("HIP_ENABLE_DEFERRED_LOADING") == nullptr, "native failure must roll back HIP");
        require(code_load_receipt().starts_with("code load: deferred (restart failed: "), "native failure receipt");
        return;
    }
    if (const char* hip = strix::envcfg::text("HIP_ENABLE_DEFERRED_LOADING"); hip && std::string_view(hip) == "0") {
        eager_code_load_or_restart(argv, CodeLoad::engine);
        require(argc == 3 && std::string_view(argv[2]).empty(), "restart must preserve complete process argv");
        char comm[16]{};
        require(::prctl(PR_GET_NAME, comm) == 0 && std::string_view(comm) == "engine716", "restore engine name");
        return;
    }
    const std::filesystem::path image(argv[0]);
    if (mode == "deleted") {
        require(std::filesystem::remove(image), "unlink running image");
    } else {
        const auto replacement = image.string() + ".next";
        { std::ofstream out(replacement); out << "#!/bin/sh\nexit 42\n"; }
        std::filesystem::permissions(replacement, std::filesystem::perms::owner_all);
        std::filesystem::rename(replacement, image);
    }
    char* reduced[] = {argv[0], nullptr};
    eager_code_load_or_restart(reduced);
    throw std::runtime_error("running inode must restart successfully");
}
#endif
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc >= 2, "test case required");
        const std::string mode(argv[1]);
        if (mode == "failure" || mode == "explicit") transaction(mode == "explicit");
#if defined(__linux__)
        else if (mode == "deleted" || mode == "replaced" || mode == "denied") native_restart(mode, argc, argv);
#endif
        else throw std::runtime_error("unknown test case");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
