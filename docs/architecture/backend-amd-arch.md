# HIP backend: AMD architecture layer

Layer 3 of the MODULAR BY DEVICE rule: architecture specialisations inside the HIP backend, selected by capability,
never by model. gfx1151 (Strix Halo, RDNA 3.5) is the default and the only qualified target. This document lists where
the shared cores assume gfx1151 today, the capability layer that replaces those assumptions, the porting order to RDNA4
discrete (gfx1200 / gfx1201) and CDNA3/4 (gfx942 / gfx950), and the risks.

## 1. Inventory of gfx1151 assumptions (origin/main at the branch point)

### Build
- `cmake/StrixHipArchs.cmake:9` `STRIX_HIP_ARCHS` defaults to `gfx1151`; already a configurable list, consumed by every
  HIP target through `HIP_ARCHITECTURES` (`CMakeLists.txt:2389`, `2399`, `2458`, `2479`, `2486`, `2493`, `2499`, `2504`, ...).
  No raw `--offload-arch` flags in the shared-core build. Arch-list tests check configuration only; they provide no
  RDNA4/CDNA device compilation evidence. Compilation and native qualification require separate gates.
- `cmake/h3_10_gemm.cmake:4` the qualified device compile contract names wave32.
- `include/strix/hip_arch_caps.hpp` macro-level instruction availability; RDNA3 / RDNA3.5 only, unknown targets fail
  closed (gfx12xx and gfx9xx get no WMMA / DPP / sudot4 today).

### src/core
- `fma_rows.hpp:350` `kComputeUnits = 40` (gfx1151 CU count) and `kFillGroups = 80` drive form selection
  (`:364-369`); the timing table above it (`:320-349`) is measured on gfx1151. **Moved to the capability struct by this
  change.**
- `f16_dense_q8a8.hpp:106` `__launch_bounds__(32 * Waves)` wave32 literal; `:117` CTA-size rule measured on gfx1151.
- `f16_dense_tall.hpp:190`, `tiled_core.hpp:590-592`, `glu_pair.hpp` `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`
  (RDNA3 16-element half fragments, wave32 lane layout).
- `integer_operand.hpp:60-61` `wmma_i32_16x16x16_iu4_w32` / `iu8_w32` (IU4 does not exist on RDNA4; CDNA has no WMMA).
- `activation_quant.hpp`, `hadamard.hpp` DPP / cross-lane reductions sized for 32 lanes.
- Unified memory: `hip_host_ready.hpp:33` `hipHostMalloc(Mapped | Coherent)` host-visible flag polled by the device
  (cheap on LPDDR, a PCIe round trip on a discrete card); `hip_weight_uploader.hpp:37,65` pinned staging plus
  `hipMemcpyAsync`, which on Strix copies within the same DRAM; `direct_row_fetch.hpp:235` `mmap(MAP_POPULATE)` page-cache
  rows read in place (zero-copy on unified memory only).

### src/moe_core
- `grouped_core.hpp:145-146, 340-341, 500-501` and `grouped_f16.hpp`, `one_split_prefill.hpp`,
  `qwen_grouped_tiles.inc`, `hip_ds41_prefill_variants.inc` f16 WMMA w32 builtins (32 call sites in total).
- `dense_a8_small.inc:76-77`, `grouped_integer_decode.hpp` IU8 WMMA w32.
- Expert residency assumes host and device memory are the same pool (routed experts resident "on a gfx1151 node",
  `CMakeLists.txt:4416`).

### src/attn_core
- `wmma_types.hpp:40-45` builtin selection keyed on `__gfx1151__` / `__gfx1150__` / `__gfx110x__` macros.
- `wmma_frag.hpp:4,37-39` accumulator layout `row = 2e + lane / 16`, `__shfl` broadcast: RDNA3 layout, wave32.
  RDNA4 changes the WMMA operand/accumulator layout; MFMA is a different layout altogether.
- `index_score_wmma.hpp:70-72`, `sparse_flash_keysplit.hpp:59`, `sparse_flash_wmma.hpp:59` `kThreads = kWaves * 32`.
- `dense_flash_prefill.hpp:67` `static_assert(lds_bytes <= 65536)`; `:26` LDS sum; `:194, :264` VGPR / waves-per-EU
  budgets (192 VGPRs, `amdgpu_waves_per_eu(8)`) measured on gfx1151; `sparse_flash_wmma.hpp:24` 255 VGPR note.
- `online_softmax.hpp:32,50,66,86`, `select.hpp`, `index_score_band.hpp` `__shfl_xor` trees sized by `Lanes` up to 32.
- `index_score_wmma.hpp:10` 32 KiB LDS tile sized for a 64 KiB work-group budget.

### src/ops
- `norm/norm_device.hpp`: wave-level reductions; no WMMA, no tuning table. Lowest porting cost.

## 2. Capability layer

`include/strix/hip/arch.hpp` (pure constexpr, no HIP include, host-testable):

- `ArchCaps`: target, family, wave size, matrix family (`Wmma16Rdna3`, `Wmma16Rdna4`, `Mfma`), primary tile M/N/K,
  FP8 and IU4 support, LDS bytes per work-group, VGPRs per lane, reference CU count, memory kind (`Unified`, `Discrete`),
  `qualified`.
- Rows: gfx1151 (qualified), gfx1200, gfx1201, gfx942, gfx950 (declared, not qualified).
- `ArchTraits<Target>`: compile-time dispatch; kernels template on `Target` and branch with `if constexpr` on the caps.
- `kCompileTarget`: target of the current device pass; host passes resolve to gfx1151. Unsupported device passes
  resolve to `Unknown`, with zero budgets, no matrix support, unknown memory kind and no qualification.
  Host syntax tests cover synthetic target macros; they do not compile device instructions.
- `kTuningTarget` / `Tuning`: target of host-side selection rules. Fixed to gfx1151 so the host and every device pass
  agree; a host rule must never read `kCompileTarget`, since the host pass cannot see which device arch will run.

Instruction availability stays in `hip_arch_caps.hpp` (macros, fail closed). The two headers are complementary: macros
gate whether a builtin exists in this pass, `ArchCaps` gives budgets and the memory kind.

Runtime selection on a discrete card: the device declaration layer (`include/strix/device`) reports `gcnArchName`; a
future `caps_from_name()` maps it to a `Target`, and host rules then take `const ArchCaps&` instead of `Tuning`.

## 3. Porting order

1. (this change) Header, configurable arch list tested for RDNA4 / CDNA lists, `core/fma_rows` CU count from the caps.
   Bitwise identical on gfx1151 (`static_assert(kComputeUnits == 40)`).
2. Replace the remaining literals with caps reads, still gfx1151 values: `kThreads = kWaves * 32` ->
   `kWaves * caps.wave_size`, `dense_flash_prefill` LDS assert -> `caps.lds_bytes`, `f16_dense_q8a8` launch bounds.
3. `src/ops/norm`: make reductions generic in wave size (shuffle trees on `wave_size`); first code that builds and
   runs on gfx1201 / gfx942.
4. Fragment abstraction: one `Frag<Target>` per matrix family wrapping the builtin and the lane layout
   (`wmma_frag.hpp`, `wmma_types.hpp`, `integer_operand.hpp`). RDNA4 first (`*_w32_gfx12` builtins, new layout,
   FP8 inputs, no IU4 -> IU8 path), then MFMA (`v_mfma_f32_16x16x16f16`, wave64, 4 x 16 accumulators per lane).
5. `src/core` dense / tiled cores on `Frag<Target>`; per-target tuning tables keyed by `Target`, each measured.
6. `src/moe_core` grouped cores; `src/attn_core` flash kernels last (largest VGPR / LDS sensitivity, gfx950 160 KiB LDS
   allows larger K/V tiles).
7. Discrete memory path (section 4) before any end-to-end serve on a discrete card.

Every step after 1 that changes a kernel for a non-gfx1151 target must keep gfx1151 bits; CI compares gfx1151
outputs before and after, and the new target is qualified separately (`qualified = true` only with measurements).

## 4. What discrete VRAM changes

- **Residency.** On Strix weights live in one LPDDR pool; mmap'ed page-cache rows (`direct_row_fetch.hpp`) are read in
  place. On a discrete card every weight must be copied to VRAM; page-cache residency no longer means device residency.
  Expert offload becomes a real tiering decision (VRAM / pinned host / disk), with a byte budget per tier.
- **Transfers.** `hip_weight_uploader` becomes a true H2D copy over PCIe (~25-60 GB/s vs ~240 GB/s local on Strix);
  upload time grows, and double-buffered staging matters. Pinned staging is cheap to keep on a discrete host (it
  does not steal device memory) whereas on Strix it reduces reclaimability.
- **Host-visible flags.** `hip_host_ready` mapped coherent memory polled by the GPU costs a PCIe read per poll on a
  discrete card; replace with a device-memory flag plus an event or a host-side `hipStreamWaitValue` path.
- **Zero-copy.** Any kernel that dereferences host pointers must be found and either copied or explicitly admitted with
  a bandwidth cost; `Memory::Discrete` lets the host rules refuse such paths by capability.
- **Bandwidth model.** Roofline and selection rules use memory bandwidth (243 GB/s on gfx1151); a discrete card has
  ~640 GB/s (gfx1201) to ~5 TB/s (gfx942); tuning tables measured on Strix do not transfer.

## 5. Risks

- **Layout drift.** RDNA4 WMMA operand / accumulator layouts differ from RDNA3; reusing `acc_row()` silently gives
  wrong results. Mitigation: the layout lives in `Frag<Target>` only, with a host reference per target.
- **Wave64.** Many reductions assume `Lanes <= 32` and 32-lane masks; on CDNA they compile and drop half the wave.
  Mitigation: static_assert on `wave_size` in every reduction until it is generic.
- **Host/device mismatch.** Reading `kCompileTarget` in a constexpr used by host selection gives different decisions in
  host and device passes. Mitigation: host rules use `Tuning` / runtime caps only.
- **Unqualified numbers.** CU, LDS and VGPR values for non-default rows are declarations from public specs; tuning on them
  needs measurement before use.
- **Multi-arch fat binaries.** Building `gfx1151;gfx1201` doubles device compile time and code size; CI should build the
  extra archs in a separate job.
- **Bits.** Any refactor of a shared core must keep gfx1151 bits; a change of bits needs a contract version and its test.
