#include "cpu/arch/avx512/kernels.hpp"

namespace strix::cpu::arch::avx512 {
// Scaffold: no AVX-512 kernel yet. Planned first entry: gemv_q8_0_a8 with VPDPBUSD (sign-split
// int8 dot), the 32-lane accumulator array kept in two zmm registers, then the declared butterfly.
const Kernels* kernels() noexcept { return nullptr; }
}  // namespace strix::cpu::arch::avx512
