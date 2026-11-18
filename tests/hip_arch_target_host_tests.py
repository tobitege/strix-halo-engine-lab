"""Host syntax tests for compile-target detection, not device compilation evidence."""
import argparse
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', required=True)
    parser.add_argument('--include-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'include')
    args = parser.parse_args()
    targets = ['gfx1151', 'gfx1200', 'gfx1201', 'gfx942', 'gfx950']
    cases = [('host', None, None, 'Gfx1151'),
             ('host_zero', 0, 'gfx1200', 'Gfx1151')]
    cases += [(target, 1, target, target.capitalize()) for target in targets]
    cases += [('unknown', 1, 'gfx9999', 'Unknown'),
              ('missing', 1, None, 'Unknown')]
    msvc = Path(args.compiler).name.lower() == 'cl.exe'
    flags = (['/nologo', '/std:c++20', '/Zs', '/W4', '/WX'] if msvc else
             ['-std=c++20', '-fsyntax-only', '-Wall', '-Wextra', '-Werror',
              '-Wshadow', '-Wsign-conversion'])
    header = args.include_dir.resolve() / 'strix/hip/arch.hpp'
    # Stay in the test working directory; each macro set gets a separate TU.
    with tempfile.TemporaryDirectory(prefix='strix-arch-target-', dir=Path.cwd()) as folder:
        for name, device, target, expected in cases:
            source = ''
            if device is not None:
                source += f'#define __HIP_DEVICE_COMPILE__ {device}\n'
            if target:
                source += f'#define __{target}__ 1\n'
            source += f'#include "{header.as_posix()}"\n'
            source += 'namespace a = strix::hip::arch;\n'
            source += f'static_assert(a::kCompileTarget == a::Target::{expected});\n'
            source += 'static_assert(a::kTuningTarget == a::Target::Gfx1151);\n'
            source += 'static_assert(a::Tuning::value.compute_units == 40U);\n'
            if expected == 'Unknown':
                source += '''using C = a::ArchTraits<a::kCompileTarget>;
static_assert(!C::value.qualified && !C::wave32 && !C::unified);
static_assert(!C::value.matrix_int4 && !C::value.matrix_fp8);
static_assert(C::value.matrix == a::Matrix::None);
static_assert(C::value.family == a::Family::Unknown);
static_assert(C::value.memory == a::Memory::Unknown);
static_assert(C::value.wave_size == 0U && C::value.lds_bytes == 0U);
static_assert(C::value.vgprs_per_lane == 0U && C::value.compute_units == 0U);
static_assert(C::value.matrix_m == 0U && C::value.matrix_n == 0U && C::value.matrix_k == 0U);
'''
            fixture = Path(folder) / f'{name}.cpp'
            fixture.write_text(source, encoding='utf-8')
            result = subprocess.run([args.compiler, *flags, str(fixture)],
                                    capture_output=True, text=True)
            if result.returncode:
                print(f'FAIL: {name}')
                print('\n'.join((result.stdout + result.stderr).splitlines()[-20:]))
                return 1
    print(f'PASS: {len(cases)} host/synthetic-device translation units')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
