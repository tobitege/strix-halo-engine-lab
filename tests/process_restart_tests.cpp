// strix::process_restart::run_again with real processes. This program plays three parts:
//   no arguments      the test: starts itself in the other parts and checks what comes back
//   restart ...       runs itself again; the restarted run checks its arguments and environment and exits 42
//   hold PIDFILE      runs itself again; the restarted run records its process ID and stays (Windows part)
#include "strix/process_environment.hpp"
#include "strix/process_restart.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>

#if !defined(_WIN32)
#include <sys/wait.h>
#endif

namespace {
constexpr const char* marker = "STRIX_PROCESS_RESTART_TEST";
constexpr int restarted_exit_code = 42;

int failures = 0;
void expect(bool condition, const char* what) {
    if (!condition) { std::cerr << "FAIL: " << what << '\n'; ++failures; }
}

bool restarted() {
    const char* value = std::getenv(marker);
    return value && std::string_view(value) == "set-before-restart";
}

// Returns only when the restart failed.
int run_again(char** argv) {
    if (!strix::process_environment::set(marker, "set-before-restart", true)) return 3;
    const auto failure = strix::process_restart::run_again(argv);
    std::cerr << "restart failed: " << failure << '\n';
    return 3;
}

// The exit code of this program started with the given arguments through the command interpreter.
int run_self(const std::string& self, const std::string& arguments) {
    std::string command = '"' + self + "\" " + arguments;
#if defined(_WIN32)
    command = '"' + command + '"'; // cmd /c strips one pair of quotes around the whole line
    return std::system(command.c_str());
#else
    const int status = std::system(command.c_str());
    return status != -1 && WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

#if defined(_WIN32)
// Ending the process that waits for the restarted program ends that program: under Linux both are one
// process, under Windows the job does it.
void restarted_program_ends_with_its_starter(const std::filesystem::path& self) {
    const auto pid_file = std::filesystem::temp_directory_path() /
        ("strix-process-restart-" + std::to_string(GetCurrentProcessId()) + ".pid");
    std::error_code ignored;
    std::filesystem::remove(pid_file, ignored);
    std::wstring command = L"\"" + self.wstring() + L"\" hold \"" + pid_file.wstring() + L"\"";
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION starter{};
    if (!CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE, 0, nullptr, nullptr, &startup, &starter)) {
        expect(false, "the starter process runs");
        return;
    }
    DWORD restarted_pid = 0;
    for (int attempt = 0; attempt < 300 && restarted_pid == 0; ++attempt) {
        std::ifstream input(pid_file);
        if (!(input >> restarted_pid)) { restarted_pid = 0; Sleep(100); }
    }
    const HANDLE restarted_process = restarted_pid ? OpenProcess(SYNCHRONIZE, FALSE, restarted_pid) : nullptr;
    expect(restarted_process != nullptr && restarted_pid != starter.dwProcessId,
        "the restarted program runs as a process of its own");
    TerminateProcess(starter.hProcess, 9U);
    expect(restarted_process != nullptr && WaitForSingleObject(restarted_process, 30000U) == WAIT_OBJECT_0,
        "ending the waiting process ends the restarted program");
    if (restarted_process != nullptr) CloseHandle(restarted_process);
    CloseHandle(starter.hThread);
    CloseHandle(starter.hProcess);
    std::filesystem::remove(pid_file, ignored);
}
#endif
} // namespace

int main(int argc, char** argv) {
    const std::string_view part = argc > 1 ? argv[1] : "";
    if (part == "restart") {
        if (!restarted()) return run_again(argv);
        const bool same_arguments = argc == 4 && std::string_view(argv[2]) == "alpha" &&
            std::string_view(argv[3]) == "beta gamma";
        return same_arguments ? restarted_exit_code : 4;
    }
#if defined(_WIN32)
    if (part == "hold" && argc == 3) {
        if (!restarted()) return run_again(argv);
        { std::ofstream output(argv[2]); output << GetCurrentProcessId() << '\n'; }
        Sleep(60000U); // ended by the job long before
        return 0;
    }
#endif
    if (argc != 1) return 2;

    expect(run_self(argv[0], "restart alpha \"beta gamma\"") == restarted_exit_code,
        "the restarted program sees the same arguments and the new environment, and its exit code comes back");
#if defined(_WIN32)
    restarted_program_ends_with_its_starter(std::filesystem::absolute(argv[0]));
#endif
    if (failures) return 1;
    std::cout << "process restart: arguments, environment and exit code carried over\n";
}
