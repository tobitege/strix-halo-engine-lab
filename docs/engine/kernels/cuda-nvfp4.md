# Shared CUDA NVFP4 dense core

Home: shared (backend core CUDA / architecture specialisation SM120).
The model-free interface is `include/strix/cuda/core/nvfp4.hpp`; CUDA enqueue
functions are in `nvfp4_cuda.hpp`. The CMake targets are `strix_nvfp4_reference`
and, with `STRIX_NVFP4_CUDA=ON`, `strix_nvfp4` in `src/cuda/core`.
TURBO+ and Qwen dense adapters can consume these same operands and plans.
This change supplies the operator; neither adapter is bound in this checkout.

Weights and explicitly quantised activations use E2M1 low-nibble-first codes,
positive E4M3FN block scales for each 16 K elements, and a positive finite FP32
tensor scale. Code and scale rows have independent byte strides, both divisible
by four; base pointers are four-byte aligned. Scales must be finite unsigned
encodings 0 through 126. K is positive and divisible by 64; M and N have tails.
The operation is `C[M,N] = A[M,K] * transpose(B[N,K])`. Output strides are in
float elements. Output and residual may alias exactly; other overlaps are not
supported. The caller supplies every buffer, stream and 256-byte-aligned scratch
allocation. Enqueue functions perform no allocation, transfer or synchronization.

`quantize` and `quantize_enqueue` implement the same finite-input activation
policy: block scale is RNE E4M3 of `min(448, (amax/6)/tensor_scale)`;
codes are RNE E2M1 of input divided by the rounded block scale times tensor scale,
clamped to [-6,6]. A zero rounded block scale produces zero codes. This lossy
activation policy is explicit and has no equivalence claim to FP16 activations.
Tensor-scale selection belongs to the caller; weight conversion can use the
same host quantizer without changing the format contract.

Identity `nvfp4-mma-f32-v1` accumulates native FP32 MMA partials. M<=8 selects
the transposed decode orientation and `min(16,K/64)` contiguous K splits;
other M selects an unsplit 64x32 CTA tile. Split partials add in ascending order,
then the product of the two tensor scales multiplies once, and optional residual
adds once. Plans fix shape, orientation, split count and workspace size.
The scalar oracle uses ascending FP32 FMA inside each split. General native
accumulation is checked within `2e-6 * sum(abs(products)) + 1e-5`; exact dyadic
fixtures and repeated identical native plans are checked bitwise. Different
plans are not promised bitwise equality.

CUDA 13 requires **sm_120a**, including its architecture-specific feature set,
for `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` with
E2M1 inputs and UE4M3 scales. Plain `sm_120` rejects that instruction. The
specialisation uses this warp instruction, not `tcgen05` (the installed CUDA 13
headers gate that form to SM100/101). Selection checks compute capability 12.0
on the caller's current device. The inspected native opcode is
`OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X`.
Instruction semantics and types follow NVIDIA's CUDA 13 headers and PTX ISA.
No third-party kernel implementation is incorporated.

## Qualification and measurement

From an MSVC x64 developer shell, host only:

```powershell
cmake -S tests/nvfp4 -B build/nvfp4-host -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/nvfp4-host --config Release
ctest --test-dir build/nvfp4-host -C Release --output-on-failure
```

For native tests, add `-DSTRIX_NVFP4_CUDA=ON` to configuration, with CUDA 13
nvcc selected. Alternatively configure the repository root with
`-DSTRIX_CUDA_NVFP4_ONLY=ON -DSTRIX_NVFP4_CUDA=ON`. Before any GPU execution:

```powershell
$env:CUDA_VISIBLE_DEVICES='0'
ctest --test-dir build/nvfp4 -C Release --output-on-failure
./build/nvfp4/nvfp4_device.exe --bench 8192 17408 5120
./scripts/bench/nvfp4/run.ps1 -Executable build/nvfp4/nvfp4_device.exe -OutputDir build/nvfp4-results
```

The reusable kit measures M=8192 and M=1..8 for (N,K)=(5120,5120),
(17408,5120), (5120,17408), and records device-0 clocks and utilization through
NVML when available. GPU selection uses the current CUDA device's PCI identity.
Each process reports its first cold launch, then a median of three warm batches
of at least two graph replays of the complete operator, including split merge.
Cold timing uses a direct enqueue; graph construction is outside warm timing.
GEMM reports 2MNK FLOPs; useful GB/s counts packed
operands plus output, excluding repeated cache traffic and split workspace.
Packing, allocation, uploads and activation quantization are outside those times.
NVML utilization is a coarse device-wide sample. The reported busy clock cycles
per row estimate includes that busy fraction; it is not a per-token measurement
of an engine or an isolated attribution of graphics/system activity.
Keep generated receipts in the build/output directory. Unit tests have a
110-second timeout and refuse any visible-device setting other than `0`.
These operator gates do not qualify model fidelity or complete workloads;
production numerical defaults and model integration remain pending those gates.

## Measured limits (review, graph replay, RTX PRO 6000, device 0)

Both activations and weights are FP4 here (W4A4), so the arithmetic identity is
tolerance-qualified, not an FP16-activation GEMV: against a double-precision dot
of the same dequantised weights and FP32 activations the relative RMS error is
about 9.5e-2 for 5120 and 17408 inputs, versus about 2e-4 for an FP16-activation
(W4A16) kernel on the same weights. An ABAB comparison with a W4A16 GEMV on the
Qwen3.8-27B shapes (rows 1 and 8) shows this decode route at 0.4-0.9 TB/s; rows
above 8 take the GEMM route, which is 4-13x slower than the decode route for
small M, so M 9..32 needs a decode-shaped route before a model adapter binds it.
