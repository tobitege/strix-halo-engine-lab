# MiniMax RTX FP32 prefill core v3.1

The CUDA FP32 prefill identity defaults to v3.1. It replaces v3's scalar
byte-sum decode with unsigned `__dp4a(p, 0x01010101U, 1024U)`, where
`p = state * 0x83dcd12dU` wraps to 32 bits. The integer result is in
[1024, 2044], so conversion to FP32 is exact. The same explicitly rounded
FP32 FMA and round-to-nearest-even FP16 conversion produce the codebook value.
Tile geometry (64/128 rows by 128 columns, 256 threads), next-tile prefetch,
ascending accumulator chains, split tails and launch grids retain v3's order.
The HIP instantiation retains its scalar decoder.

`STRIX_MM_FP32_CORE=3` restores v3; `STRIX_MM_FP32_CORE=31` selects v3.1.
`GroupedCall.core_version` accepts 0 (environment/default), 3 or 31.
An explicit nonzero selection overrides the environment. Unset, empty or
unrecognized selections resolve to v3.1. The CUDA enqueue reads the environment
on each call; this selector does not change the arithmetic identity.

## Qualification evidence and release check

A model-level full gate found v3.1 states byte-identical to v3 and to the
FP32-identity references on **1K and 8K code prompts, on both ranks**. The interleaved v3/v3.1/v3 8K full-workload measurement
was 15.12 / 15.04 / 15.84 seconds: v3.1 was **2.8% faster** than the mean of
the bracketing v3 runs. This is evidence for that measured workload and identity,
not a universal speedup. Original receipts remain outside the public tree.

The native identity run on the target GPU passed every section. The CUDA identity
executable checks both decoders over all 65,536 states against the host reference,
then checks default, both environment selections and both explicit overrides against
the host linear reference. For each selection the grouped-core cases cover 64-row,
128-row and mixed-tail tiles, mixed bit rates and maps; combine and SwiGLU are also
checked but do not go through the core. Any mismatch returns nonzero.
The host route census checks the selector policy and source coverage; a passing
census is not native numerical qualification and does not close other open gates.

## Reproduce the public build

From a Windows checkout with CUDA 13.0 and Visual Studio 2022 Community, use
the public numerical build recipe (`sm_120`, `--fmad=false`, precise div/sqrt,
`/fp:strict`). Pass an absolute build directory to keep artifacts out of source:

```bat
src\cuda\models\minimax\build_tests.bat BUILD_DIR
src\cuda\models\minimax\build_bench.bat BUILD_DIR
cuobjdump --dump-sass BUILD_DIR\mm_fp32_core_bench.exe > BUILD_DIR\core.sass
python src\cuda\models\minimax\bench\count_core_sass.py BUILD_DIR\core.sass
```

The benchmark compares only v3 and v3.1, using production core dispatch and
split tails. It prints bitwise raw/output comparisons and medians of three
interleaved batches of ten iterations for transform/core/output stages.
It is synthetic evidence, separate from full-model qualification.

Static inner k-tile loop counts in this port's CUDA 13.0 `sm_120` benchmark:

| Rows | v3 instructions | v3.1 instructions | FFMA, either version | Registers, either version |
|---|---:|---:|---:|---:|
| 64 | 739 | 685 | 520 (512 accumulation) | 127 |
| 128 | 1278 | 1224 | 1032 (1024 accumulation) | 195 |

Both v3.1 loops contain eight `IDP.4A.U8.U8` instructions. Both versions have
zero stack frame, spill stores and spill loads at both heights. Counts include
predicated instructions, barriers and NOPs, excluding never-executed `@!PT`
instructions. They describe compiler output, not dynamic throughput or a GPU pass.
Run `mm_fp32_identity_tests.exe` on the intended PRO device before release;
retain the complete output, exit code and executable digest privately.
