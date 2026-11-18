#pragma once
// Capability layer for AMD architectures (HIP backend, layer 3 of MODULAR BY DEVICE).
// Pure constexpr C++20: no HIP include, usable from host code, device code and host tests.
//
// A kernel or a host-side tuning rule reads its architecture constants from ArchCaps instead of literals, and picks a
// specialisation with ArchTraits<Target> / if constexpr on the capability fields, never on the model.
// Instruction availability for a compile pass stays in strix/hip_arch_caps.hpp (macro level, fail closed);
// this header describes the hardware budget and the memory kind of a named target.
//
// Values are the hardware budget of each target; only Gfx1151 is qualified. Other rows are declarations so code can be
// written against them; they do not qualify any tuning or numerical path.
#include <cstdint>

namespace strix::hip::arch {

enum class Target : std::uint8_t { Gfx1151, Gfx1200, Gfx1201, Gfx942, Gfx950, Unknown };
enum class Family : std::uint8_t { Rdna35, Rdna4, Cdna3, Cdna4, Unknown };
// Matrix instruction family: WMMA (wave32, 16x16x16, RDNA) or MFMA (wave64, CDNA).
enum class Matrix : std::uint8_t { Wmma16Rdna3, Wmma16Rdna4, Mfma, None };
// Unified: device memory is host LPDDR (zero-copy and host-visible pointers are cheap, no PCIe hop).
// Discrete: device VRAM behind PCIe (host-visible pointers cross the link; weights need explicit residency).
enum class Memory : std::uint8_t { Unified, Discrete, Unknown };

struct ArchCaps {
    Target target;
    Family family;
    const char* name;
    unsigned wave_size;          // lanes per wave
    Matrix matrix;               // matrix instruction family
    unsigned matrix_m, matrix_n, matrix_k;  // f16 matrix tile of the primary instruction
    bool matrix_fp8;             // FP8 matrix inputs
    bool matrix_int4;            // packed IU4 matrix inputs
    unsigned lds_bytes;          // LDS per work-group
    unsigned vgprs_per_lane;     // architectural VGPR cap per lane
    unsigned compute_units;      // CUs of the reference part (tuning only)
    Memory memory;
    bool qualified;              // tuning and numerics measured on this target
};

inline constexpr ArchCaps kGfx1151{Target::Gfx1151, Family::Rdna35, "gfx1151", 32U, Matrix::Wmma16Rdna3, 16U, 16U, 16U,
                                   false, true, 65536U, 256U, 40U, Memory::Unified, true};
inline constexpr ArchCaps kGfx1200{Target::Gfx1200, Family::Rdna4, "gfx1200", 32U, Matrix::Wmma16Rdna4, 16U, 16U, 16U,
                                   true, false, 65536U, 256U, 32U, Memory::Discrete, false};
inline constexpr ArchCaps kGfx1201{Target::Gfx1201, Family::Rdna4, "gfx1201", 32U, Matrix::Wmma16Rdna4, 16U, 16U, 16U,
                                   true, false, 65536U, 256U, 64U, Memory::Discrete, false};
inline constexpr ArchCaps kGfx942{Target::Gfx942, Family::Cdna3, "gfx942", 64U, Matrix::Mfma, 16U, 16U, 16U,
                                  true, false, 65536U, 512U, 304U, Memory::Discrete, false};
inline constexpr ArchCaps kGfx950{Target::Gfx950, Family::Cdna4, "gfx950", 64U, Matrix::Mfma, 16U, 16U, 32U,
                                  true, false, 163840U, 512U, 256U, Memory::Discrete, false};
inline constexpr ArchCaps kUnknown{Target::Unknown, Family::Unknown, "unknown", 0U, Matrix::None, 0U, 0U, 0U,
                                    false, false, 0U, 0U, 0U, Memory::Unknown, false};

constexpr const ArchCaps& caps(Target t) {
    switch (t) {
        case Target::Gfx1200: return kGfx1200;
        case Target::Gfx1201: return kGfx1201;
        case Target::Gfx942: return kGfx942;
        case Target::Gfx950: return kGfx950;
        case Target::Gfx1151: return kGfx1151;
        case Target::Unknown: break;
    }
    return kUnknown;
}

// Compile-time dispatch: template kernels take a Target parameter and read ArchTraits<T>::caps.
template <Target T>
struct ArchTraits {
    static constexpr const ArchCaps& value = caps(T);
    static constexpr bool wave32 = value.wave_size == 32U;
    static constexpr bool unified = value.memory == Memory::Unified;
};

// The tuning target of host-side selection rules. Stays Gfx1151 until a second target is qualified; changing it
// changes tuning decisions and needs its own measurement.
inline constexpr Target kTuningTarget = Target::Gfx1151;

// Host passes use the default; unsupported device passes expose no capabilities.
#if !defined(__HIP_DEVICE_COMPILE__) || !__HIP_DEVICE_COMPILE__
inline constexpr Target kCompileTarget = Target::Gfx1151;
#elif defined(__gfx1151__)
inline constexpr Target kCompileTarget = Target::Gfx1151;
#elif defined(__gfx1200__)
inline constexpr Target kCompileTarget = Target::Gfx1200;
#elif defined(__gfx1201__)
inline constexpr Target kCompileTarget = Target::Gfx1201;
#elif defined(__gfx942__)
inline constexpr Target kCompileTarget = Target::Gfx942;
#elif defined(__gfx950__)
inline constexpr Target kCompileTarget = Target::Gfx950;
#else
inline constexpr Target kCompileTarget = Target::Unknown;
#endif

using Tuning = ArchTraits<kTuningTarget>;

}  // namespace strix::hip::arch
