#pragma once
// AArch64 NEON (dotprod / i8mm) specialisation of the CPU backend; SVE registers here with its
// vector length from Caps::sve_bits. Null until a kernel exists: the selector keeps the reference.
// Each entry added must match src/cpu/core/reference.cpp bit for bit (tests/cpu_backend_host_tests.cpp).
#include "cpu/core/backend.hpp"

namespace strix::cpu::arch::neon {
const Kernels* kernels() noexcept;
}
