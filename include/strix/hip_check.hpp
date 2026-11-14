#pragma once

// HIP TUs: the runtime header must precede the std headers so <cstring> etc. get device overloads.
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

// One HIP status check for the whole engine (R2-15). Replaces the per-file
// `check(hipError_t, const char*)` copies; message format is unchanged:
// "<what>: <hipGetErrorString>" or the bare HIP string for the one-argument form.
#include <hip/hip_runtime_api.h>

#include <stdexcept>
#include <string>

namespace strix {

inline void hip_check(hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(e));
}

inline void hip_check(hipError_t e) {
    if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e));
}

// Per-module prefix form: "<prefix><what>: <hipGetErrorString>".
inline void hip_check_prefixed(const char* prefix, hipError_t e, const char* what) {
    if (e != hipSuccess) throw std::runtime_error(std::string(prefix) + what + ": " + hipGetErrorString(e));
}

// Legacy form that reported only the caller's message.
inline void hip_check_message(hipError_t e, const char* message) {
    if (e != hipSuccess) throw std::runtime_error(message);
}

}  // namespace strix
