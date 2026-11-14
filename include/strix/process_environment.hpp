#pragma once

#include <cstdlib>

// setenv/unsetenv for the process environment, with the Windows CRT equivalents.
// On Windows an empty value removes the variable (_putenv_s), so a variable can
// not be set to the empty string there.
namespace strix::process_environment {

[[nodiscard]] inline bool set(const char* name, const char* value, bool overwrite) noexcept {
#ifdef _WIN32
    if (!overwrite && std::getenv(name) != nullptr) return true;
    return _putenv_s(name, value) == 0;
#else
    return ::setenv(name, value, overwrite ? 1 : 0) == 0;
#endif
}

[[nodiscard]] inline bool unset(const char* name) noexcept {
#ifdef _WIN32
    return _putenv_s(name, "") == 0;
#else
    return ::unsetenv(name) == 0;
#endif
}

} // namespace strix::process_environment
