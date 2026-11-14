#pragma once

#include "strix/device/gate.hpp"

#include <cstdio>
#include <initializer_list>
#include <string>

// HIP side of the start gate, without HIP headers: any main can include this. bind_hip_main() is the one call a HIP
// binary makes before its first allocation.
//
// AMD devices are identified by PCI address. Without a declaration, a lone integrated device keeps working (receipt:
// declared=implicit); a discrete or a second device always needs the file. The current HIP device is per host thread
// and a new thread starts on index 0, so the bound device has to be index 0: when it is not, the gate refuses and names
// the HIP_VISIBLE_DEVICES value that makes it so.
//
// In a build without the HIP runtime the same functions exist: bind_hip_main() only removes the two flags, and
// bind_hip() refuses.
namespace strix::device {

// Resolves the device once per process (throws Refusal), makes it current for the calling thread and prints the
// receipt line on `receipt` (null = silent). Later calls return the same binding and only make the device current for
// their thread; their arguments are ignored.
const Bound& bind_hip(Request request = {}, const std::string& devices_path = {}, std::FILE* receipt = stderr);

// For a main(): takes --devices FILE / --device-id ID out of the argument vector and binds. On a refusal it prints the
// one-line reason and exits with status 2.
void bind_hip_main(int& argc, char** argv, std::FILE* receipt = stderr);

// The binding made by this process, or null before the first bind (and always null without the HIP runtime).
[[nodiscard]] const Bound* hip_bound() noexcept;

// Host wait policy: how a host thread waits for the device (stream / event / device synchronize). ONE policy home for
// every HIP binary: an engine names its choice once after the bind, and STRIX_HOST_WAIT=runtime|spin|yield|block
// overrides whatever the engine named.
//   runtime  : the HIP runtime's own schedule (no device flag is set)
//   spin     : busy-wait; lowest wake latency, one host core at 100 % while the device works
//   yield    : busy-wait that yields the core between polls
//   blocking : sleep in the driver until the device signals; the core idles while the device works
enum class HostWait { runtime, spin, yield, blocking };

[[nodiscard]] constexpr const char* host_wait_name(HostWait policy) noexcept {
    switch (policy) {
        case HostWait::spin: return "spin";
        case HostWait::yield: return "yield";
        case HostWait::blocking: return "block";
        default: return "runtime";
    }
}

// The policy in force for an engine choice and the STRIX_HOST_WAIT text (null or empty: the engine choice). Throws
// Refusal on any other text.
[[nodiscard]] inline HostWait resolve_host_wait(HostWait engine_choice, const char* override_text) {
    if (!override_text || !*override_text) return engine_choice;
    const std::string text(override_text);
    for (const auto policy : {HostWait::runtime, HostWait::spin, HostWait::yield, HostWait::blocking})
        if (text == host_wait_name(policy)) return policy;
    throw Refusal("host wait: STRIX_HOST_WAIT=" + text + " is not one of runtime|spin|yield|block");
}

// Applies the resolved policy to the calling thread's current device (call after bind_hip) and prints one receipt
// line (null = silent). Returns the policy in force. Without the HIP runtime it only resolves.
HostWait set_host_wait(HostWait engine_choice, std::FILE* receipt = stderr);

} // namespace strix::device
