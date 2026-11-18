// Compile-only coverage of the HIP -> standard-library include boundary.
// No device is enumerated or bound by this test.
#include "strix/device/hip_enumerator.hpp"
#include "strix/power/sampler.hpp"

#include <chrono>
#include <format>

// Exercise the exact attribute token used by libstdc++ 16 even on GCC 15.
// HIP's empty host macro would turn this into an invalid [[__gnu__::]].
#if defined(__GNUC__) && !defined(__clang__)
[[__gnu__::__noinline__]]
#endif
int host_header_result() { return 0; }

int main() { return host_header_result(); }
