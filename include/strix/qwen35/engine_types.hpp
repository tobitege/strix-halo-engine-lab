#pragma once
// Backend-agnostic engine vocabulary shared by the CUDA and HIP qwen35 engines and the shared drafting binding.
// Home: shared (host): no device runtime types.
#include <cstdint>

namespace strix::qwen35 {
enum class LogitRows : std::uint8_t { none, last, all };
}  // namespace strix::qwen35
