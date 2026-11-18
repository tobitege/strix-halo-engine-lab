# Compressed weights on CUDA — implementation report

Status: **experimental prototype; NOT READY for the model or the exact FRONT**.

## Delivered

`STRIX_ENABLE_CUDA_TRELLIS` is OFF by default and does not change the serving or
model dispatch. The API in `include/strix/cuda/trellis.hpp` takes views of
compressed weights on the device. It implements:

- Circular bitstream K1–K8/mul1, exact decoding to FP16 and complete scales.
- Input/output H128 with the declared FP16/FP32 roundings.
- GEMV R1–R8: decoding in registers reused across rows, explicit K16
  partitions, a four-lane tree and an ascending sum of partials.
- FP16/FP32 prefill with MMA: M16/N64, per-tile decoding into shared memory,
  four warps, row tails and explicit strides.
- Enqueue without allocation or synchronization, graph compatible; prior
  validation of sizes, overflow, alignment and aliasing of writable buffers.
- An independent CPU reference and a bounded fixture generator (fixtures kept
  outside Git).

The linear call does not expand Q globally. The global diagnostic decoder only
serves to check bits. Workspace: `align256(2*R*I)+4*S*R*N` bytes.
The test matrices, source inspections and full logs are kept outside the
repository.

## Evidence

The final small native run on the RTX PRO 6000 (sm_120) passed 20 fixtures:
16 synthetic K1–K8 in both rectangular orientations, one K4 projection
5120→512, K2 gate/down 5120→2304 and 2304→5120, and one complete H128 crop
512→512 of the K5 head (not the whole head). It checked:

- All 65,536 mul1 states, exact in FP16 bits.
- Complete Q and A exact in bits; GEMV partials P and output Y exact against
  the CPU, for each of R1–R8.
- MMA partials against the FP64 dot of the same FP16 operands, with a bound
  that depends on the absolute sum of products; propagation of that bound
  through the Hadamard, scales and bias, including cancellation.
- Padding/tails, 76 shifted row views, graphs with changing X and poisoned
  scratch/Y; the active outputs must be finite and change with X.

That is 27,000,832 Q values checked and 308 row/mode combinations, with 924
graph replays. The final run used 707,948,544 private bytes (0.659 GiB), well
below 20 GB, exit code **0**, no watchdog triggered. It was not a performance
measurement.

CPU: 5 independent tests PASS (exhaustive codebook hash, scalar reader against
the vectorized bit reader, ring/permutation, explicit H128 and the reduction
against exact integer products). The executable also checks 26 invalid
descriptors, the workspace formula and the R0 no-op without calling CUDA.
Native build: MSVC 19.44/CUDA 13.0, Release, sm_120, no fast math,
`--fmad=false -prec-div=true -prec-sqrt=true -ftz=false`.

## Bytes and limits of the prediction

K2 takes 2.01007 bpw with scales for the expert matrices: 8,891,904 bytes per
expert, against 9,953,280 for the three current payloads, a 10.66 % reduction.
Holding residency, overlap and compute constant, the design's sensitivity model
reduces 1.54 s of exposed wait to 1.376 s and 2.57 s total to 2.406 s. **This is
neither a measurement nor a TTFT promise.** K4 dense compression saves more
reads per tensor; its effect on tok/s depends on the fixed part of the TAIL and
on the costs of the complete operator. Decoding has not been shown to be free on
this GPU.

## Pending contract for integration

The implemented GEMV has an explicit CUDA quality identity. It is not shown to
reproduce HIP's packed-dot instruction, nor all of its partition decisions per
tensor and row count. `hip_bitwise_required` fails before launching. **The exact
FRONT cannot use this delivery as a replacement.** Exact emulation is not claimed
to be impossible in principle; it would require native conformance of the
instruction and a frozen HIP route/splits. The proposed alternative is a FRONT
with a separate quality identity, subject to approval by the results of its
gates, not an implicit fallback.

Still missing: the model's weight loader/plan, compressed expert transport
across the tiers with grouped scheduling, and dispatch of the projections in
the serving path/FRONT. This API alone does not remove the model's resident Q8
copies nor change the bytes of its real transfers. To qualify the integration:

1. Freeze the same trellis weights and the activation policy; compare the
   arithmetic across devices first, then the quantization quality.
2. Prefill: full-vocabulary KL mean/max at 64 positions at 2K/8K/32K and an
   AMD continuation of 64–256 tokens; relative budget ≤10 % on both KL and a
   top-1 loss ≤1 point against the frozen corpus/base.
3. Strict FRONT: the same tokens and complete-logit digests as the pair with
   the SAME weights, TP/EP and expert fold; include forced R2/R4/R6/R8 and
   rejected rows. The quality variant needs its own gate.
4. Measure TTFT/tok-s and the real DMA bytes of the complete workload on an
   otherwise idle machine.

## Reproduction

Use a native console with MSVC and CUDA 13 configured:

```powershell
cmake -S . -B build-cuda-trellis -G Ninja -DCMAKE_BUILD_TYPE=Release `
  -DSTRIX_CUDA_DS41_ONLY=ON -DSTRIX_ENABLE_CUDA_TRELLIS=ON `
  -DSTRIX_WARNINGS_AS_ERRORS=OFF -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-cuda-trellis --target strix-cuda-trellis-tests -j 2
python -m unittest discover -s tests -p test_trellis_reference.py -v
python tools/trellis_reference.py --out <fixtures-dir>
ctest --test-dir build-cuda-trellis -R '^strix-cuda-trellis-contract-tests$' --output-on-failure
```

The native tests are run explicitly so that CTest does not take a shared GPU.
The binary checks for an RTX PRO 6000 with sm_120 before allocating buffers.

```powershell
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
$env:CUDA_VISIBLE_DEVICES = '0'
$env:CUDA_MODULE_LOADING = 'EAGER'
$cases = @(Get-ChildItem <fixtures-dir>/*.trlcase | ForEach-Object FullName)
./build-cuda-trellis/strix-cuda-trellis-tests.exe `
  --book <fixtures-dir>/mul1.bin @cases
```

NumPy is required for the oracle/generator; the kernels depend neither on
Python nor on an external trellis library. `--source FILE --prefix TENSOR`
generates a case from a modern weight group, recording origin and hash in the
external manifest. `--inputs/--outputs` crops only complete H128 blocks.
