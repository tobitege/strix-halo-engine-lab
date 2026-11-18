#include "cpu/arch/neon/kernels.hpp"

namespace strix::cpu::arch::neon {
// Scaffold: no NEON kernel yet. Planned first entry: gemv_q8_0_a8 with SDOT, eight int32x4
// accumulators per 32 groups, then the declared butterfly.
const Kernels* kernels() noexcept { return nullptr; }
}  // namespace strix::cpu::arch::neon
