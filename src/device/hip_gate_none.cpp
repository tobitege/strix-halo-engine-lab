#include "strix/device/hip_gate.hpp"

#include <cstdlib>
#include <exception>

// Build without the HIP runtime: there is no device to bind. Mains still accept the two flags.
namespace strix::device {

const Bound& bind_hip(Request, const std::string&, std::FILE*) {
    throw Refusal("device gate: refusing to start: this build has no HIP runtime");
}

void bind_hip_main(int& argc, char** argv, std::FILE*) {
    try {
        (void)take_command_line(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        std::fflush(stderr);
        std::exit(2);
    }
}

HostWait set_host_wait(HostWait engine_choice, std::FILE*) { return resolve_host_wait(engine_choice, std::getenv("STRIX_HOST_WAIT")); }

const Bound* hip_bound() noexcept { return nullptr; }

} // namespace strix::device
