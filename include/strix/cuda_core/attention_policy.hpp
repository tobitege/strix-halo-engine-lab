#pragma once
// Host-only attention scheduling policies of the CUDA core (no CUDA headers: engine option structs include this).
// Contracts: see attention.hpp attention_prefill_flash.
#include <cstdint>

namespace strix::cuda_core {

// FlashPipe = shared-memory pipeline of the K/V tiles only (scheduling): every pipe gives the SAME bits.
enum class FlashPipe : std::uint8_t { pad2 = 0, swz3 = 1 };
// FlashTail = wave-tail split of the flash grid (scheduling only, SAME bits).
enum class FlashTail : std::uint8_t { off = 0, split = 1 };

}  // namespace strix::cuda_core
