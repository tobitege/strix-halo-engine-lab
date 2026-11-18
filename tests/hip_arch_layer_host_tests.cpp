// Host tests for the AMD capability layer (strix/hip/arch.hpp). No HIP, no GPU.
#include "strix/hip/arch.hpp"

#include <cstdio>
#include <cstring>

namespace a = strix::hip::arch;

// The default stays gfx1151 with the literals the shared cores used before this layer.
static_assert(a::kTuningTarget == a::Target::Gfx1151);
static_assert(a::kCompileTarget == a::Target::Gfx1151, "host pass resolves to the default");
static_assert(a::Tuning::value.compute_units == 40U);
static_assert(a::Tuning::value.wave_size == 32U && a::Tuning::wave32);
static_assert(a::Tuning::value.lds_bytes == 65536U);
static_assert(a::Tuning::unified && a::Tuning::value.qualified);
static_assert(a::Tuning::value.matrix == a::Matrix::Wmma16Rdna3 && a::Tuning::value.matrix_int4 && !a::Tuning::value.matrix_fp8);
// Declared, unqualified targets.
static_assert(!a::caps(a::Target::Gfx1200).matrix_int4);
static_assert(!a::ArchTraits<a::Target::Gfx1201>::value.matrix_int4);
static_assert(a::ArchTraits<a::Target::Gfx1201>::value.matrix_fp8 && !a::ArchTraits<a::Target::Gfx1201>::unified);
static_assert(!a::ArchTraits<a::Target::Gfx942>::wave32 && a::caps(a::Target::Gfx942).matrix == a::Matrix::Mfma);
static_assert(a::caps(a::Target::Gfx950).lds_bytes > a::kGfx1151.lds_bytes);

template <a::Target T> constexpr unsigned lanes_per_tile() {
    if constexpr (a::ArchTraits<T>::wave32) return 32U; else return 64U;
}
static_assert(lanes_per_tile<a::Target::Gfx1151>() == 32U && lanes_per_tile<a::Target::Gfx950>() == 64U);

int main() {
    const a::Target all[] = {a::Target::Gfx1151, a::Target::Gfx1200, a::Target::Gfx1201, a::Target::Gfx942, a::Target::Gfx950};
    const char* names[] = {"gfx1151", "gfx1200", "gfx1201", "gfx942", "gfx950"};
    int fail = 0, qualified = 0;
    for (unsigned i = 0; i < 5U; ++i) {
        const a::ArchCaps& c = a::caps(all[i]);
        if (c.target != all[i] || std::strcmp(c.name, names[i]) != 0) { std::printf("table row %u\n", i); ++fail; }
        if ((c.wave_size == 64U) != (c.matrix == a::Matrix::Mfma)) { std::printf("wave/matrix %s\n", c.name); ++fail; }
        if (c.lds_bytes == 0U || c.compute_units == 0U) { std::printf("zero budget %s\n", c.name); ++fail; }
        qualified += c.qualified ? 1 : 0;
    }
    if (qualified != 1) { std::printf("only gfx1151 is qualified\n"); ++fail; }
    std::printf(fail ? "FAIL\n" : "ok\n");
    return fail ? 1 : 0;
}
