#pragma once

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>

#if defined(_WIN32)
#include <windows.h>
#else
#include <unistd.h>
#endif

// Running the program again from its start, with the environment as it is now (a value the runtime reads
// while the image loads takes effect only this way).
namespace strix::process_restart {

// Replaces this process by a fresh run of its own program and returns only when that failed, with the reason.
// Linux: execv of the program with the given arguments (terminated by a null pointer); the process keeps its
// identity. Windows has no exec: the program runs again with the process's own command line (the arguments
// are not used) as a child that shares the console, the standard handles and the environment. The child is in
// a job that ends it when this process ends, however that happens, and this process waits and ends with the
// child's exit code. While it waits it leaves Ctrl+C and Ctrl+Break to the child.
[[nodiscard]] inline std::string run_again(char* const* arguments) {
#if defined(_WIN32)
    static_cast<void>(arguments);
    const auto failure = [] { return "Windows error " + std::to_string(GetLastError()); };
    std::wstring program(32768U, L'\0');
    program.resize(GetModuleFileNameW(nullptr, program.data(), static_cast<DWORD>(program.size())));
    if (program.empty()) return failure();
    const HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (job == nullptr) return failure();
    JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    std::wstring command_line(GetCommandLineW());
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    PROCESS_INFORMATION child{};
    // Suspended until it is in the job: it never runs outside it.
    if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits)) ||
        !CreateProcessW(program.c_str(), command_line.data(), nullptr, nullptr, TRUE, CREATE_SUSPENDED, nullptr, nullptr,
            &startup, &child)) {
        const auto reason = failure();
        CloseHandle(job);
        return reason;
    }
    // Ctrl+C and Ctrl+Break reach every process of the console. The restarted program decides what they mean
    // (a server stops cleanly); this process only waits for it, as the one process under Linux would. Closing
    // the console still ends this process, and the job ends the child with it. In place before the child runs.
    const PHANDLER_ROUTINE leave_to_child = [](DWORD event) -> BOOL {
        return event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT;
    };
    SetConsoleCtrlHandler(leave_to_child, TRUE);
    if (!AssignProcessToJobObject(job, child.hProcess) || ResumeThread(child.hThread) == static_cast<DWORD>(-1)) {
        const auto reason = failure();
        SetConsoleCtrlHandler(leave_to_child, FALSE);
        TerminateProcess(child.hProcess, 1U);
        CloseHandle(child.hThread);
        CloseHandle(child.hProcess);
        CloseHandle(job);
        return reason;
    }
    CloseHandle(child.hThread);
    DWORD code = 1U;
    if (WaitForSingleObject(child.hProcess, INFINITE) != WAIT_OBJECT_0 || !GetExitCodeProcess(child.hProcess, &code))
        code = 1U;
    std::_Exit(static_cast<int>(code)); // closes the job with this process
#else
    ::execv("/proc/self/exe", arguments);
    return std::strerror(errno);
#endif
}

} // namespace strix::process_restart
