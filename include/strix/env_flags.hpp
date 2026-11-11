#pragma once

#include <cstdlib>
#include <cstring>

namespace strix {

// Most flags compare the entire value. Some legacy readers compare only byte 0;
// keep that choice explicit ("0tail" and "1tail" otherwise change meaning).
enum class EnvFlagMatch { exact, first_character };

namespace detail {
inline bool env_flag_matches(const char* value, const char* flag, EnvFlagMatch match) {
    return value && (match == EnvFlagMatch::first_character ? value[0] == flag[0] : std::strcmp(value, flag) == 0);
}
}  // namespace detail

// Unset and empty are ON; only "0" (or a leading '0' when selected) disables.
inline bool env_flag_default_on(const char* name, EnvFlagMatch match = EnvFlagMatch::exact) {
    return !detail::env_flag_matches(std::getenv(name), "0", match);
}

// Unset and empty are OFF; only "1" (or a leading '1' when selected) enables.
inline bool env_flag_default_off(const char* name, EnvFlagMatch match = EnvFlagMatch::exact) {
    return detail::env_flag_matches(std::getenv(name), "1", match);
}

}  // namespace strix
