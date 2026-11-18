#pragma once
// x86-64 AVX2 (+ AVX-VNNI when the CPU has it) row kernels of the host routed-expert executor. Null when the TU was
// not compiled for AVX2. Bit-identical to the scalar table (tests/cpu/routed_expert_host_tests.cpp).
#include "cpu/core/routed_expert_kernels.hpp"

namespace strix::cpu::arch::avx2 {
const expert::detail::Kernels* expert_kernels() noexcept;
}
