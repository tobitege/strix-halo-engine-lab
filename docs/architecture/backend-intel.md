# Intel GPU backend (SYCL / oneAPI, Level Zero)

Status: scaffold. Recorded host evidence is an MSVC Release run; other platforms are not qualified. The SYCL device side is written against SYCL
2020 and the Intel extensions but has not been compiled yet (no oneAPI toolchain on the build hosts). The shared
backend interface is described in `docs/architecture/backends.md` (separate branch, not edited here); this page is the
Intel section of it.

Layering (MODULAR BY DEVICE):

| Layer | Intel |
|---|---|
| 1. Shared, model-agnostic | unchanged: graph, serve, session, spec, xport, policies, identities, device declaration and gate |
| 2. Vendor backend | `src/intel/core`: queue, USM, the shared core interfaces (dense, grouped MoE, attention, norms, weight decode) |
| 3. Arch specialisation | `src/intel/arch/xe2` (Arc B `xe2-bmg`, Lunar Lake `xe2-lnl`; Panther Lake `xe3-ptl` until `arch/xe3` exists), chosen by `Capabilities::arch` |

No model names appear under `src/intel`. A model binds a shared core by filling its view (for example
`dense::Q8A8View`, corresponding data fields to `f16_dense_tall::q8a8::View`, without its launch-only `tile0`).
The portable view and oracle still need extraction into `src/core` and adoption by both launches.

## 1. Interface mapping

| Shared interface (HIP today) | SYCL form |
|---|---|
| `hipStream_t` | in-order `sycl::queue` behind the opaque `intel::Queue` (no SYCL type in `backend.hpp`) |
| `hipError_t` | `intel::Status`; SYCL exceptions are caught at the backend boundary |
| `hipMalloc` / `hipHostMalloc` / managed | `sycl::malloc_device` / `malloc_host` / `malloc_shared` (`MemoryKind`) |
| `hipMemcpyAsync`, `hipStreamSynchronize` | `queue.memcpy`, `queue.wait_and_throw` |
| HIP graphs (decode replay) | `sycl_ext_oneapi_graph` (record and replay a command graph) |
| `__global__` + grid/CTA | `parallel_for(nd_range)`; CTA = work-group, wave32 = sub-group (16 on Xe2: `reqd_sub_group_size(16)`) |
| LDS (`__shared__`) | SLM: `local_accessor` or `work_group_static` |
| `__builtin_amdgcn_s_barrier` + fences | `group_barrier(it.get_group())`, `atomic_fence` with `memory_scope::work_group` |
| `readfirstlane`, `permlanex16`, DPP | `group_broadcast`, `select_from_group`, `permute_group_by_xor`, `shift_group_left` |
| `v_wmma_f32_16x16x16_f16_w32` (`Half16`, `Float8`) | `joint_matrix<sub_group, half, use::a, 8, 16>` x `<half, use::b, 16, 16>` -> `<float, accumulator, 8, 16>`; or ESIMD `xmx::dpas<8, 8>` |
| `_Float16`, ext_vector_type | `sycl::half`, `sycl::vec` / `marray`; the reference kernels use the bit-exact `intel::fp16` |
| `__builtin_amdgcn_exp2f` | `sycl::native::exp2` (FAST class only; precise policies keep `sycl::exp2`) |
| device gate `bind_hip` / `bind_cuda` | `xpu-smi` listing (`query_xpu_smi`), then `ONEAPI_DEVICE_SELECTOR=level_zero:<n>`, confirmed by `SyclEnumerator` |

Shared cores and their Intel form:

| Core | Interface | Intel plan |
|---|---|---|
| `src/core/f16_dense_tall.hpp` + `f16_dense_q8a8.hpp` | Cols / Store policies, `View`, `supported()`, `enqueue()` | same policies as functors in a SYCL kernel; B tile staged in SLM once per work-group, A converted in registers; joint_matrix DPAS |
| `src/core/tiled_core.hpp`, `linear_plan.hpp` | plan + execution resolution | the plan is host code: reused; only the launch differs |
| `src/moe_core/grouped_core.hpp` | `GroupedCore<WeightDecode, ExpertLayout, InputTransform, Epilogue>` | same template parameters; `permlanex16` exchange becomes sub-group shuffles; decode policies are plain arithmetic and port as is |
| `src/attn_core/flash_core.hpp`, `online_softmax.hpp` | online softmax step, tile stage | softmax state is scalar per row: ports directly; QK and PV products on joint_matrix |
| `src/ops` (norms, rotary, activation quant) | elementwise / row reductions | `reduce_over_group`; no matrix engine |

Memory model. On an iGPU (Lunar Lake, Panther Lake) device, host and shared USM all live in the same LPDDR5X, as on
Strix Halo: weights go in `malloc_shared` (or `malloc_host` with zero copy) and the loader writes them in place, no
staging copy. On Arc B, `malloc_device` VRAM for weights and KV, `malloc_host` for upload staging and for host-visible
counters; `malloc_shared` only for small control blocks (page migration on first touch costs PCIe round trips).
Memory budgeting stays in the shared layer: the backend reports `global_bytes` and `integrated`.

## 2. Toolchain

| | Linux | Windows |
|---|---|---|
| Compiler | oneAPI DPC++ `icpx -fsycl` (Base Toolkit 2025.x), or the open-source intel/llvm `clang++ -fsycl` | `icx` / `icx-cl -fsycl` from the Base Toolkit, inside a VS 2022 developer prompt after `setvars.bat` |
| Runtime | Level Zero loader + `intel-compute-runtime` (xe or i915 kernel driver; Arc B needs a 6.12+ kernel with `xe`) | Arc / Core Ultra graphics driver (ships Level Zero) |
| Tools | `sycl-ls`, `xpu-smi`, `unitrace`, VTune | `sycl-ls`, VTune; `xpu-smi` is Linux first: the gate falls back to `SyclEnumerator` |
| AOT targets | `-fsycl-targets=intel_gpu_bmg_g21,intel_gpu_lnl_m,intel_gpu_ptl_h` | same |

Configure:

```
# host only (recorded evidence: MSVC Release):
cmake -S src/intel -B build-intel && cmake --build build-intel --config Release && ctest --test-dir build-intel -C Release
# full tree, opt-in:
cmake -B build -DSTRIX_BACKEND_INTEL=ON
# with SYCL (compiler must be oneAPI DPC++):
cmake -B build-sycl -G Ninja -DCMAKE_CXX_COMPILER=icpx -DSTRIX_BACKEND_INTEL=ON
```

`cmake/StrixIntel.cmake` turns SYCL on only when `CMAKE_CXX_COMPILER_ID` is `IntelLLVM` and a `-fsycl` program
compiles; otherwise it prints one status line and builds the host part. `STRIX_BACKEND_INTEL` is OFF by default and
nothing in the HIP or CUDA configuration changes.

Host regressions include independent FP32 bit patterns and injected SYCL boundary failures using a test double.
These do not compile against oneAPI or execute on a device. Qualification still requires a real SYCL build,
device/reference execution gates with an explicit arithmetic identity, and device-gate integration.

## 3. What ports first

Order by (value per line of new code) x (risk of a wrong result):

1. **Dense FP16 core with the q8a8 policy** (`f16_dense_q8a8`): one kernel shape covers every dense projection of
   every model; its arithmetic contract is already written, and the plain C++ reference kernel
   (`src/intel/core/dense_q8a8_ref.hpp`) is the oracle. Prefill is compute bound, so this is where XMX pays.
2. **Norms, rotary, activation quant (`src/ops`)**: small, no matrix engine, needed by every forward.
3. **Attention (`attn_core` dense flash prefill + decode)**: online softmax is portable as is; the matrix tiles reuse
   the dense core's joint_matrix layer.
4. **Grouped MoE core**: largest surface (decode policies, trellis, layouts); it waits for the dense tile to be measured
   because its tile is the dense tile per expert group.

Expected peaks and limits (vendor figures, to be measured with the engine's own probes before any claim):

| Device | Memory | Bytes peak | XMX FP16 dense | Decode bound (8 GB of weights per token) |
|---|---|---|---|---|
| Arc B580 (`xe2-bmg`, 20 Xe-cores) | 12 GB GDDR6, 192-bit 19 Gbps | 456 GB/s | about 117 TFLOPS (INT8 about 233 TOPS) | <= 57 tok/s |
| Lunar Lake Arc 140V (`xe2-lnl`, 8 Xe-cores) | LPDDR5X-8533, 128-bit, shared | 136.5 GB/s | about 32 TFLOPS (Intel quotes 67 INT8 TOPS) | <= 17 tok/s |
| Panther Lake Arc B390 (`xe3-ptl`, 12 Xe-cores) | LPDDR5X-9600, 128-bit, shared | 153.6 GB/s | about 60 TFLOPS (Intel quotes 120 INT8 TOPS) | <= 19 tok/s |
| Strix Halo (reference) | LPDDR5X-8000, 256-bit | 256 GB/s | | <= 32 tok/s |

Decode is bandwidth bound on all of them (bound = bytes peak / weight bytes per token); a 12 GB card also limits
which models fit without the host-tier path. Lunar Lake has 32 GB at most, so it is a small-model and correctness target.

## 4. Risks

- **No toolchain yet**: the SYCL branch of `backend.cpp` and `SyclEnumerator` are uncompiled. First step on a host with
  oneAPI is to build them; expect extension spelling fixes (`ext_intel_device_id`, `ext_intel_pci_address`).
- **joint_matrix maturity**: supported shapes and layouts differ per device and release. Query
  `ext::oneapi::experimental::info::device::matrix_combinations` at open and fall back to the reference path; ESIMD
  `dpas` is the fallback when joint_matrix code generation is poor (it ties the kernel to Intel, which is acceptable
  inside `arch/xe2`).
- **Sub-group 16 vs wave32**: the HIP cores assume 32 lanes per row tile. Every lane-index formula in a ported policy
  must be rewritten against `kSubgroup`, not copied.
- **Summation order**: DPAS accumulates in its own order; device vs reference comparisons use a tolerance, and the
  FAST/precise policy classes keep their meaning (precise policies stay on scalar FP32 chains).
- **Windows driver**: TDR resets long kernels (2 s default). Split prefill launches; never raise TdrDelay from the engine.
- **Device ids**: the arch table maps PCI id families (0xE2xx BMG, 0x64xx LNL, 0xB0xx PTL); confirm against the driver's
  table when new SKUs ship.
- **Shared USM on discrete**: page migration silently turns a fast path into a PCIe one; the backend chooses the kind,
  callers never pass `shared` for weights on Arc B.

## 5. Migration order

1. (done here) `STRIX_BACKEND_INTEL`, device listing (`src/device/xpu_smi.cpp`, `include/strix/device/intel_enumerator.hpp`),
   backend interface with no-SYCL stubs, reference dense kernel, host tests.
2. Build the SYCL branch with icpx; a `bind_intel` gate entry next to `bind_hip` / `bind_cuda` (pin
   `ONEAPI_DEVICE_SELECTOR`, confirm by UUID, print the receipt line).
3. Device test: reference q8a8 kernel on device vs host reference (bit-exact: same code, scalar order).
4. `arch/xe2` XMX dense tile behind `enqueue_dense_q8a8`; measure against the bytes and FLOPS peaks above with a
   predicted figure written before the change.
5. `src/ops` norms and rotary; then dense flash attention; then the grouped MoE core.
6. Serve: the shared serve and session layers select the backend by the bound device's vendor; no per-model Intel code.
