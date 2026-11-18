#pragma once
// x86-64 AVX-512 (BW/VNNI) specialisation of the CPU backend; AMX tiles register here too once the
// OS grants tile state. Null until a kernel exists: the selector then keeps the reference entries.
// Each entry added must match src/cpu/core/reference.cpp bit for bit (tests/cpu_backend_host_tests.cpp).
#include "cpu/core/backend.hpp"

namespace strix::cpu::arch::avx512 {
const Kernels* kernels() noexcept;
}
