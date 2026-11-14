# Exact HC normalized recomputation

Status: implemented candidate, native qualification pending. No performance or
native equality result is recorded here. `STRIX_QPF2K_HC_RECOMPUTE_NORM=1` enables
the candidate at eligible sites; `=0` restores the FP32 image. The unset default
remains OFF until the compiler, byte comparisons and interleaved timing establish
that ON is exact and not slower. Enabling the production default requires that
evidence. Restart after changing this cached control. Numerical identity remains
`fp16-v1+hcout16` when that profile is selected; this control has no registry bit.

The existing inject+norm and combine+inject+norm kernels still store their BF16
streams and perform the same reduction, division, epsilon addition and `rsqrtf`.
ON stores that computed inverse once per row/stream instead of writing the FP32
normalized image. Plain HC norm does the same inside its existing launch. Each
consumer forms `qpf2k_mul(qpf2k_mul(inverse, bf16(stream)), weight)`, with pinned
operand order and two distinct FP32 instructions. Projection then performs its
existing FP16 conversion and ordered WMMA/injection chains; output keeps its
existing FP32 mix. There is no extra launch or stream reread to compute inverse.

The qpf2k prefetched projection, OUT16 and output3 (including band16/lowband) use a
separate recompute view, without setting the older `wide.inverse` dispatch field.
All existing folds, low16 controls, prefetch depth, weight packs/copies and output
schedules remain eligible. ON prints once per process:
`HC normalized recompute: ON (byte-identical)`. This is a route witness, not a
substitute for the pending native comparisons.

## Stream lifetime and site admission

`prepare_hyper_prefill_rows` checks every output, normalized+low workspace and
injection extent against the complete input-row range. Thus `rows.output` cannot
alias even another row's BF16 streams. The folded entry point additionally checks
that the inject output is exactly the HC input, with the same row stride. It
permits exact in-place replacement of the hyper input, and of a combine input,
only after the existing producer has read those values. Folded workspace/input
overlap continues to use the existing private-workspace handling.

After the producer, projection reads streams and writes only low/low16/low_t and
injection; output reads streams and writes the separately admitted activation.
These launches share the execution stream, and the next site's inject cannot
overwrite streams until the current output finishes. The inverse scratch has the
same stream-ordered lifetime. The test seam also rejects stream/destination
overlap, and fixtures exercise in-place producers and disjoint consumers.

Admission requires uniform, aligned BF16 rows, H=2560, four streams, rank=320,
FP32 norm weights, fast prefill, the qpf2k prefetched projection and either OUT16
or 32-row output3. Unsupported sites keep their existing image and all existing
folds. These include nonuniform rows, other activation/weight/shape profiles,
projection-mask bit 3 disabled, the older norm16/inverse modes, HC diagnostic
checks, and output configurations selecting the base/packed/16-row output instead
of OUT16/output3. No such site is redirected to a slower consumer. MTP and other
profiles retain their existing contracts.

## Counts at T=16384

There are 10,240 values per row and five 64-column projection CTAs. Each value is
recomputed once in **each** projection column block (five times); the same value
feeds that block's FP16 conversion and FP32 injection chains without another
recompute. Output recomputes it once. Projection prefetch holds raw BF16 and norm
weights, and recomputes only when consuming that stage. Output3 likewise prefetches
raw BF16 slots and normalizes them when consumed. This preserves overlap
without pulling a global-load wait forward into the prefetch. Its existing
unconditional terminal prefetch does not perform normalization multiplies. Tails
can perform additional clamped-row work, but T16384 has no projection/output tail.

- FP32 image stores avoided: **671,088,640 bytes (640 MiB)** per site.
- Inverse stores added: **262,144 bytes (0.25 MiB)**. Net materialized bytes
  avoided: **670,826,496 bytes (639.75 MiB)** per site.
- Useful consumer multiplies: `2 * 10240 * (5 + 1) = 122880` per row; producer
  multiplies removed: 20,480. Net **102,400 per row / 1,677,721,600 per site**.
- No normalization multiplies are added for the unused terminal prefetch.
- Logical image traffic (store plus five projection reads plus output read):
  4,480 MiB becomes 1,920 MiB of BF16 consumer reads, before scalar/weight traffic
  and terminal prefetches. The inverse is globally loaded once per row/stream
  **per CTA**: five projection blocks plus twenty output feature blocks; including
  its store, that is 6.5 MiB. These are logical counts, not measured DRAM traffic;
  repeated projection reads and norm weights can hit cache. Norm weights are
  40 KiB and loaded per stage/feature into registers for reuse. Timing must decide.

The arena and existing private workspace reservations are unchanged: dropping
stores does **not** claim a 640 MiB reduction in allocated capacity. Low values
still occupy their existing workspace offsets. No identity/cache key is changed.

## Resources and validation

Both arms are separate template instantiations. Each matrix kernel stores its
inverse in the eight unused halves after K32 in each Pitch40 LDS row: four FP32
values fit exactly, and no existing vector store or WMMA fragment accesses those
halves. This avoids crossing an LDS occupancy threshold in projection.

| Kernel | OFF / ON static staging bytes | VGPR / occupancy |
|---|---:|---|
| Plain norm | 1028 / 1028 | Pending compiler report |
| Inject+norm, combine+inject+norm | 0 / 0 | Pending compiler report |
| Prefetched projection | 30720 / 30720 | Pending compiler report |
| OUT16 | 23040 / 23040 | Pending compiler report |
| Output3 and band/lowband forms | 0 / 512 | Pending compiler report |

The table is source accounting, not compiler resource evidence. Offline checks
enumerated staging/inverse addresses, verified disjointness and FP32 alignment,
and checked projection lane/channel coverage. Offline CMake configuration could
not find a C++ compiler in the implementation environment; no HIP compilation or
GPU program was run. Compiler checks, native bit equality, model logits/state and
complete-workload checks remain pending. Build both template arms for gfx1151
with `-Rpass-analysis=kernel-resource-usage` and inspect VGPR, SGPR, scratch, LDS
and resulting occupancy for all the touched instantiations before promotion.

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRIX_ENABLE_HIP_RUNTIME=ON -DSTRIX_HIP_ARCHITECTURES=gfx1151 \
  -DCMAKE_HIP_FLAGS=-Rpass-analysis=kernel-resource-usage
cmake --build build --target strix-hip-hc-wide-rows-tests
```

`--recompute-norm` compares OFF/ON low, low16, injection, output activation,
streams, combined rows and guards bitwise, and checks that ON never writes the
image. It covers plain norm, bit-1 and bit-2 folds, in-place producers, partial
rows/CTAs, all lowband modes, BF16 NaN payloads/Inf/subnormal/-0 and epsilon edges.
`--time` first checks the selected T16384 transaction, then runs two warmups per
arm and seven interleaved A/B pairs, reversing pair order on odd pairs. Inputs
are immutable across timed calls; reset/copy cost is outside event timing. Existing
weight pack/copy launches are inside timing. `--site 0|1|2` chooses plain norm,
inject+norm or combine+inject+norm; `--output 14|7` chooses OUT16 or output3.
Explicit arm flags in the fixture select both instantiations independently of
the cached environment switch; production uses the switch and site admission.

The following deciding ticket is capped at eight minutes after building. Supply
`EVIDENCE` as a private directory outside Git. Use the same kernel/host settings
for both arms, and record the running kernel and `amdgpu.iommu_perfopt` privately.
The trace isolates a production-shaped T16384 site with no additional inverse
launch. Sum producer, projection and output duration by arm; also retain total
transaction timing and the existing pack/copy cost. Reject any byte mismatch or
repeatable slowdown. A timeout or missing evidence is not a pass.

```sh
export EVIDENCE  # operator-supplied private destination
timeout 480s bash -euc '
  : "${EVIDENCE:?set a private evidence directory}"
  build/strix-hip-hc-wide-rows-tests --recompute-norm
  build/strix-hip-hc-wide-rows-tests --time --site 1 --output 14
  STRIX_QPF2K_HC_RECOMPUTE_NORM=0 rocprofv3 --kernel-trace \
    --output-directory "$EVIDENCE/off" -- \
    build/strix-hip-hc-wide-rows-tests --trace --arm off --site 1 --output 14
  STRIX_QPF2K_HC_RECOMPUTE_NORM=1 rocprofv3 --kernel-trace \
    --output-directory "$EVIDENCE/on" -- \
    build/strix-hip-hc-wide-rows-tests --trace --arm on --site 1 --output 14
'
```

Repeat with `--site 2` or `--output 7` in a separate ticket if the first route
passes. Production resolver/fold coverage and model logits/state must also pass
with only `STRIX_QPF2K_HC_RECOMPUTE_NORM` changed before default promotion.

## Default, projection shape and stream mirror

The route is the default with the WMMA HC output. `STRIX_QPF2K_HC_RECOMPUTE_NORM=0` restores the image; `=1` also
admits the route with the lane-chain output forms.

- The projection runs as one block per 64 rows over all 320 low outputs (`STRIX_QPF2K_HC_PROJ_TILE`, 0 selects the
  five-column-block form). Accumulator order, injection chains and the epilogue are those of the base kernel.
- A stream row pitch that is a multiple of 4096 bytes makes the rows of a block alias, because the consumers read
  many rows at the same column. With the engine's 20480-byte pitch the recompute step measured 10.49 ms against
  6.12 ms at 20608 bytes (8,192 rows). The folded inject + norm therefore also stores the stream row into the first
  20480 bytes of the row in the normalized workspace (pitch 42240), and the consumers read that copy: 7.19 ms
  against 8.29 ms for the image. Norm-only sites with an aliasing pitch keep the image.
- Fixtures: `ctest -R "strix-hip-hc-recompute-(norm|mirror)-tests" --output-on-failure` (non-aliasing and aliasing
  pitch; every written byte, and the mirror against the stream rows).
