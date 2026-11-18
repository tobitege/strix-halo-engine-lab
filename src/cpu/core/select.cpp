// Capability detection and kernel-table selection (layer 3 is picked by capability, not by model).
#include "cpu/core/backend.hpp"
#include "cpu/arch/avx512/kernels.hpp"
#include "cpu/arch/neon/kernels.hpp"

#if defined(_MSC_VER) && (defined(_M_X64) || defined(_M_IX86))
#include <intrin.h>
#define STRIX_CPU_X86 1
#elif (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#define STRIX_CPU_X86 1
#endif

namespace strix::cpu {

namespace {
#if defined(STRIX_CPU_X86)
void cpuid(unsigned leaf, unsigned sub, unsigned (&r)[4]) {
#if defined(_MSC_VER)
    int v[4];
    __cpuidex(v, static_cast<int>(leaf), static_cast<int>(sub));
    for (int i = 0; i < 4; ++i) r[i] = static_cast<unsigned>(v[i]);
#else
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
#endif
}
#endif

Caps detect() noexcept {
    Caps c;
#if defined(STRIX_CPU_X86)
    // Feature bits only; OS state (XCR0 / AMX tile permission) is checked by the specialisation
    // that uses them before it registers.
    unsigned r[4];
    cpuid(0, 0, r);
    if (r[0] >= 7) {
        cpuid(7, 0, r);
        c.avx2 = (r[1] >> 5) & 1U;
        c.avx512f = (r[1] >> 16) & 1U;
        c.avx512bw = (r[1] >> 30) & 1U;
        c.avx512vnni = (r[2] >> 11) & 1U;
        c.amx_int8 = (r[3] >> 25) & 1U;
    }
#endif
#if defined(__ARM_NEON) || defined(_M_ARM64)
    c.neon = true;
#endif
#if defined(__ARM_FEATURE_DOTPROD)
    c.dotprod = true;
#endif
#if defined(__ARM_FEATURE_SVE)
    c.sve = true;
#endif
    return c;
}

void fill(Kernels& k, const Kernels* spec) {
    if (!spec) return;
    k.isa = spec->isa;
    if (spec->quantize_a8) k.quantize_a8 = spec->quantize_a8;
    if (spec->gemv_q8_0_a8) k.gemv_q8_0_a8 = spec->gemv_q8_0_a8;
    if (spec->rmsnorm_f32) k.rmsnorm_f32 = spec->rmsnorm_f32;
}
}  // namespace

Caps caps() noexcept {
    static const Caps c = detect();
    return c;
}

const char* isa_name(Isa isa) noexcept {
    switch (isa) {
    case Isa::scalar: return "scalar";
    case Isa::avx2: return "avx2";
    case Isa::avx512: return "avx512";
    case Isa::amx: return "amx";
    case Isa::neon: return "neon";
    case Isa::sve: return "sve";
    }
    return "unknown";
}

Kernels select(const Caps& c) noexcept {
    Kernels table = reference_kernels();
    if (c.avx512f && c.avx512bw) fill(table, arch::avx512::kernels());
    if (c.neon) fill(table, arch::neon::kernels());
    return table;
}

const Kernels* registered_weight_kernels(Isa isa) noexcept {
    switch (isa) {
    case Isa::scalar: return &reference_kernels();
    case Isa::avx512: return caps().avx512f && caps().avx512bw ? arch::avx512::kernels() : nullptr;
    case Isa::neon: return caps().neon ? arch::neon::kernels() : nullptr;
    default: return nullptr;
    }
}

}  // namespace strix::cpu
