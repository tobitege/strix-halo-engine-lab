#pragma once
#include <stdexcept>
#include <string_view>
#include "strix/cuda_core/attention_policy.hpp"

namespace strix::qwen35 {
// Host-only resolution shared by inspection and execution. Scheduling controls preserve
// arithmetic; attn8 is explicitly bit-changing and retains the engine's own identity/gates.
// Validate every value before changing the options, including inspection-only invocations.
template<class Options>
void resolve_flash_controls(Options& options, std::string_view pipe, std::string_view tail, std::string_view attn8) {
    if (!pipe.empty() && pipe != "pad2" && pipe != "swz3") throw std::runtime_error("--flash-pipe expects pad2|swz3");
    if (!tail.empty() && tail != "off" && tail != "split") throw std::runtime_error("--flash-tail expects split|off");
    if (!attn8.empty() && attn8 != "on" && attn8 != "off") throw std::runtime_error("--prefill-attn8 expects on|off");
    if (!pipe.empty()) options.flash_pipe = pipe == "swz3" ? cuda_core::FlashPipe::swz3 : cuda_core::FlashPipe::pad2;
    if (!tail.empty()) options.flash_tail = tail == "split" ? cuda_core::FlashTail::split : cuda_core::FlashTail::off;
    if (!attn8.empty()) options.prefill_attn8 = attn8 == "on";
}
}  // namespace strix::qwen35
