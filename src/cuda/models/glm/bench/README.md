# Native GLM FFN tools

These tools are standalone and do not link into or change the HIP node engine.
The candidate is not statistically qualified. No GPU results are checked into
Git. Keep exported assets, provenance manifests, histograms, logs, binaries and
disassembly in a private build directory.

Build offline from a Visual Studio 2022/CUDA 13 installation:

```powershell
./src/cuda/models/glm/bench/build.ps1 -BuildDir BUILD_DIR
& BUILD_DIR/host_tests.exe
python src/cuda/models/glm/bench/test_export.py --build-dir BUILD_DIR
```

`host_tests.exe` is built with MSVC and CUDA's header-only half conversion;
it has no CUDA runtime linkage or CUDA calls. `build.ps1` never runs the GPU
benchmark. It emits `glm_native_bench.exe`, logs, SASS and PTX, and checks for
HMMA and LDGSTS in both CUTLASS mainloops. Inspect the two `grouped_mma` specializations: HMMA must be inside the
increasing-K back edge, for both gate/up and down. The build pins sm_120,
FP contraction off, subnormals on, precise division/sqrt and host strict FP.

For real weights, export a complete full-width EXL3 layer before the GPU window:

```powershell
python src/cuda/models/glm/bench/export_model.py --model MODEL_DIR --layer 3 --kind experts --out BUILD_DIR/experts.native
```

The exporter validates each matrix's codebook, rate, dtype, shape, byte extent,
scale finiteness and unsupported conventions. It supports split-shard groups,
copies original packed/scaling bytes, and writes a separate SHA256/provenance
manifest. FP8-only, incomplete or non-mul1 packs fail; they cannot stand in for
real EXL3 fixtures. `--kind shared` or `--kind dense` exports one FFN. The binary
archive is `GLMNAT01`, little-endian E/H/I u32, then expert-major gate/up/down:
u32 bits, packed u16 words, suh u16 and svh u16. No padding or trailing bytes.
Archive loading copies the full tensors, never repeating one expert's weights.

An explicitly authorized GPU window uses:

```powershell
$env:CUDA_DEVICE_ORDER='PCI_BUS_ID'
$env:CUDA_VISIBLE_DEVICES='GPU_UUID'
./src/cuda/models/glm/bench/window.ps1 -BuildDir BUILD_DIR -Weights BUILD_DIR/experts.native -Path both
```

The launcher has a 570-second aggregate deadline including setup and CPU
reference work. It kills an overrun and reports incomplete, rather than
claiming a successful <=10-minute suite. It checks the real archive hash before
launch. Without `-Weights`, synthetic routed/shared/dense tests still run and
real data is explicitly `NO-REFERENCE`. Device access occurs only in the
benchmark with `--run`; it requires exactly one visible sm_120 RTX PRO 6000.
Use `-Path cutlass` (the launcher's default) or `-Path legacy` for one arm;
`-Path both` runs both arms with the same seeds, fixtures and shapes within the
same aggregate deadline. The executable defaults to `--path legacy`; select
`--path cutlass` explicitly for draft `glm-pf-rtx-native-v1-c1`. See its
[arithmetic, scheduling and qualification contract](../../../../../docs/models/glm/rtx-prefill-native.md#c1-grouped-gemm-candidate-draft).

For an initial GPU correctness diagnostic, run `glm_native_bench --run --path cutlass --smoke`:
1/17/257 rows, 33 experts, 128-wide matrices exercise empty experts, both tile
classes, row/group tails and parity-buffer reuse across three windows. This is
not a full-shape throughput result.

The full benchmark tests C=1024/8192/32768 with distinct seeded matrices,
uniform and lognormal(sigma=0.6) top-8 routing. `--histogram FILE` supplies E
little-endian float64 nonnegative popularity counts (at least eight positive),
resampled without replacement. This is labeled as a resampled histogram, not
actual routed rows. Optional `--routes FILE` supplies complete C*8 u32 IDs
followed by C*8 float32 route weights, preserving router slot order. Optional
`--input FILE` supplies C*H float32 activations. IDs must be unique per row.
Use `--rows 1024|8192|32768` with captured input/route files to select their
single shape; without it the default suite runs all three shapes.
Captured histogram/input/route provenance belongs alongside private results.

Full-MoE median milliseconds include histogram/prefix/scatter, all group
decodes, input transforms, gate/up/SwiGLU, down, and fixed-slot weighted merge.
Decode and gate/up/down substage times are arithmetic means of event intervals;
do not add them to a median or call them isolated GEMM times. Grouped useful
TMAC/s excludes decode/dispatch/merge, includes transforms/epilogues. Useful
MAC count is `C*topk*3*H*I` (one multiply-add = one MAC); it excludes padding,
decode, transforms and nonlinear operations. Dense/shared use topk=1.

Accuracy covers **all rows and channels** against a SIMT scalar ascending-K
FP32 FMA reference with the exact decoded FP16 weights. This reference reuses
dispatch/transforms/epilogue but replaces the tensor-core reduction completely;
the independent CPU reference additionally checks the complete first row,
all input/output channels and all eight selected experts. It reports max
absolute error, relative L2 and FP32 ULP histogram bins; a repeated-run all-row
equality diagnostic checks merge determinism separately. It does not claim
bitwise equivalence to node arithmetic or model/endpoint qualification.
Nonfinite reference or output values fail. GPU execution and real-tensor
accuracy remain mandatory subsequent gates.

The initial 16x128 implementation has no asynchronous pipelines. Decode
materialization (~14.50 GB written per complete 288-expert layer), shared-memory
barriers, small-expert tails and routing skew can substantially reduce
full-layer throughput. Real timing determines the next schedule/identity;
theoretical peaks cannot qualify this implementation. The selectable C1 path
uses pipelined CUTLASS GEMMs and overlapped double-buffered decode; its phase
report separately measures transforms, GEMMs, epilogues, merge, problem-list
construction and decode exposed/hidden estimates. Static pipeline evidence
does not establish execution correctness, achieved overlap or throughput.

The C1 decoder uses one warp per 256-weight tile, eight warps per CTA, and
strides over tiles with a grid budget derived during setup from SM count and
kernel residency (capped at two decoder CTAs per SM). The combined gate/up/down
window launch dispatches K3/K4/K5/K6 once per matrix, stages each packed tile
with 16-byte loads, and stores coalesced FP16 directly in row-major B. The
mul1 codebook and fragment permutation are unchanged; suh/svh still execute
at their original transform/epilogue rounding boundaries. Empty routed experts
skip materialization; the problem list already gives them zero-M GEMMs.

For each profiled iteration, `decode_window` reports the expert/rate counts,
empty experts, packed bytes read, FP16 bytes written, decode duration, logical
GB/s (input plus output divided by duration), and skipped bytes. Logical GB/s
is not a hardware DRAM counter. `GEMM_overlap_ms` is the sum of timestamp
intersections with gate/up and down GEMM intervals on the compute stream;
it measures concurrent intervals, not absence of resource contention.
`exposed_wait_ms` starts after the independent input transform and ends after
the compute stream's ready wait; it can include event/scheduling overhead.
The aggregate hidden estimate also includes overlap with non-GEMM work.

Dependency order is dispatch -> decode on a nonblocking stream -> ready[g]
-> compute GEMMs. Input transforms precede the ready wait. Decode g+1 uses the
other parity buffer while compute g executes; decode g+2 waits only for
retired[g], recorded after the down GEMM's final B read. The output epilogue
uses separate FP32 scratch. Descriptors and scratch remain compute-stream-only,
and the two decoded operand arenas do not alias. Each call completes all
consuming GEMMs before its merge/end event, permitting subsequent serial calls
to reuse the workspace and events. Concurrent calls require separate resources.

Host tests cover every 16-bit state at each bitrate, full tiles, the coalesced
inverse permutation, rectangular B layouts, uneven grid strides and scale
rounding. These CPU and disassembly checks do not qualify CUDA execution,
real-weight fidelity, overlap efficiency or latency targets.


## KDA sublayer (M3 partial candidate)

Build also emits `glm_kda_bench.exe`, `kda_host_tests.exe` and KDA SASS/PTX.
Run `kda_host_tests.exe` and the CPU archive tests explicitly:

```powershell
python -B src/cuda/models/glm/bench/test_attention_export.py --build-dir BUILD_DIR
python -B src/cuda/models/glm/bench/export_model.py --model MODEL_DIR --out BUILD_DIR/kda.native --layer L --kind kda
```

`config.json` selects the layer type. `--kind attention` also supports exporting
complete DSA tensor sets; a missing tensor is an error.
`GLMATT01` contains a layer/hidden/head/count header, then named records with
original dtype, shape, byte count and raw little-endian tensor bytes. The JSON
sidecar retains per-tensor source extents/hashes and the config/archive hashes.
No conversion, scale absorption or fabricated reference is allowed. Model
assets, manifests, receipts and measurement outputs belong outside Git.

The KDA benchmark requires `--weights FILE --rows C --iters N --allow-gpu`.
Its input is seeded synthetic normalized hidden rows and nonzero initial
state/conv, explicitly labeled in output. It runs the complete qkv, f/g/beta,
conv, blocked recurrence, output gate and output projection; it excludes input
normalization, residual/hyperconnections and FFN. Decode is included. Native
warmup is excluded; each timed repetition and the full FP32 reference start
from identical state. Every output and S/conv element is compared, with rel L2,
max abs and ULP diagnostics. Repeatability checks cover output and both state
buffers. Diagnostic accuracy has no invented passing tolerance.

Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES` to the authorized
single GPU UUID, then use:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File src/cuda/models/glm/bench/kda_window.ps1 -BuildDir BUILD_DIR -Weights BUILD_DIR/kda.native
```

The launcher verifies the archive hash, records executable/archive hashes, and
runs C=1024/8192/32768 with three timed repetitions plus warmup/full reference.
A single 570-second deadline covers all child execution, with a five-second
kill wait. A timed-out or incomplete suite is an error, not a pass; no claim
that all cases fit the window is made before execution. `-PlanOnly` validates
assets and emits the plan without launching CUDA. The build script never
runs this launcher or any benchmark.

`inspect_kda_sass.py` requires HMMA and cp.async in both projection K backedges,
HMMA in triangular/incoming-state/output block loops, two MMA tiles in the
unrolled state-carry product, and scalar FFMA with no HMMA in both references.
Compilation/disassembly and host proofs establish neither real execution
correctness nor statistical route qualification. See the [precision contract](../../../../../docs/models/glm/rtx-prefill-native.md#kda32-attention-sublayer-candidate-draft).


## DSA layer and MTP composition (draft)

The offline build also emits `glm_dsa_bench.exe`, `dsa_host_tests.exe` and
DSA/MTP SASS/PTX evidence. Run CPU checks explicitly:

```powershell
BUILD_DIR/dsa_host_tests.exe
python -B src/cuda/models/glm/bench/test_dsa_export.py --build-dir BUILD_DIR
powershell -NoProfile -ExecutionPolicy Bypass -File src/cuda/models/glm/bench/test_dsa_window.ps1 -BuildDir BUILD_DIR
python -B src/cuda/models/glm/bench/export_model.py --model MODEL_DIR --out BUILD_DIR/dsa.native --layer 7 --kind attention
```

The layer-7 export requires all 29 original tensors, even across shard
boundaries. The benchmark consumes real weights and seeded synthetic normalized
rows; it does not claim prompt fidelity. It times decode, state projections and
store, pooling, query projections, query absorption, index scores, selection,
sparse attention, value absorption and output projection. Useful MAC accounting
excludes padding, repeats, transform arithmetic and skipped index dots when all
pools fit. Device and wall times include decoding, with warmup excluded. Every
output and exported state element is compared to the full FP32 reference; native
repeatability covers output, state and selected token lists. BF16 state reports
also count differing raw elements. The stated near-tie policy and counts are
part of each diagnostic record. The default identity is
`glm-pf-rtx-native-v1-dsa64-f32idx`: FP32 index projection and dot chains,
including all dependencies of pool selection. `--mma-index` opts into the
separate `glm-pf-rtx-native-v1-dsa64-mmaidx` identity; the launcher accepts
`-MmaIndex` and records that identity. Attention uses 32-key tiles shared by
all 64 heads, fixed 256-key splits and an ascending-order FP32 combine.
The 40 useful TMAC/s objective at 8K remains unmeasured until a device window.
Numerical identity remains a draft.

Set the authorized device environment and execute the bounded window:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File src/cuda/models/glm/bench/dsa_window.ps1 -BuildDir BUILD_DIR -Weights BUILD_DIR/dsa.native
```

This runs 1024/8192/32768 rows, one timed native repetition each, plus warmup and
a complete reference. One aggregate 570-second deadline, followed by at most a
five-second kill wait, covers all cases and references. Timeout/incomplete means
failure. Completion within that budget is not promised before device execution.
`-PlanOnly` validates the archive hash and records the plan without launching a
process. Selection mismatches use exit code 3: each result is recorded in
`dsa-window-results.json`, all later lengths still run, and the launcher returns
3 when the suite completes with any mismatch. Other errors abort immediately.
Compile and export before the window. No build or CPU test launches
CUDA. The SASS checker requires MMA K backedges in projection, index and
absorption kernels, HMMA and cp.async in the sparse attention chunk loop, scalar FFMA
references/router and no local loads/stores in native MMA kernels.

`enqueue_mtp` in `mtp.cuh` reuses DSA plus the existing grouped expert path and
has a scalar reference arm. To prepare its full layer-45 assets separately:

```powershell
python -B src/cuda/models/glm/bench/export_model.py --model MODEL_DIR --out BUILD_DIR/mtp.native --layer 45 --kind mtp
python -B src/cuda/models/glm/bench/export_model.py --model MODEL_DIR --out BUILD_DIR/mtp-experts.native --layer 45 --kind experts
python -B src/cuda/models/glm/bench/export_model.py --model MODEL_DIR --out BUILD_DIR/mtp-shared.native --layer 45 --kind shared
```

The MTP attention/fusion archive has 40 tensors; self-attention records retain
`self_attn.` names and fusion/norm/router records retain their layer-relative
names. Expert/shared assets remain separate `GLMNAT01` archives. The DSA bench
accepts the 29-tensor layer-7 archive only. MTP composition has no real-model
benchmark receipt until all layer-45 assets and paired embedding/hidden inputs
are available and device tests run. These standalone APIs do not alter engine
routing or qualify the complete prefix hand-off.


### Index arithmetic sweep (draft v2)

See the [arithmetic identities and fused streaming contract](../../../../../docs/models/glm/rtx-prefill-native.md#dsa-index-arithmetic-ablation-v2-drafts).
The provisional near-FP32 candidate keeps attention projections on FP16
CUTLASS, uses FastF32 for index projections, and fuses BF16 three-product
causal scores with streaming top-512. It has no measured selection/latency
qualification. Every ablation compares against the full independent FP32
layer reference, with the same stated near-tie policy.

```powershell
# Offline plan; no CUDA APIs or device execution.
powershell -NoProfile -ExecutionPolicy Bypass -File src/cuda/models/glm/bench/dsa_window.ps1 -BuildDir BUILD_DIR -Weights BUILD_DIR/dsa.native -VariantSweep -PlanOnly
# Authorized GPU window only, after setting the device environment:
powershell -NoProfile -ExecutionPolicy Bypass -File src/cuda/models/glm/bench/dsa_window.ps1 -BuildDir BUILD_DIR -Weights BUILD_DIR/dsa.native -VariantSweep -Iterations 1
```

The sweep includes `fp32`, `mma`, `mma-proj-fp32-score`,
`fp32-proj-mma-score`, and `tf32x3` at 8K and 32K, plus `tf32x3` at 1K.
Use `-IndexVariant NAME` without `-VariantSweep` for one variant at all three
lengths, or `--index-variant NAME` directly on the executable. The legacy
`--mma-index` / `-MmaIndex` aliases remain supported. The aggregate deadline
includes archive hashing, all warmups, repeatability, FP32 references and
readback; a timeout is an incomplete failure. Sweep logs use distinct
`dsa-window-VARIANT-ROWS.log` names. The fused path includes selection in the
`index_scores` phase. Inspect full-layer time before assessing either target.


The DSA sweep now emits both `variant_vs_node_mimic` and
`node_mimic_vs_scalar_fp32` selection diagnostics for every variant/length.
The paired-half reference is an IEEE approximation, not a bitwise hardware
DOT2 claim. It runs only index dependencies and preserves the full scalar
arm for output/state metrics. The exit code tests the provisional model
baseline agreement threshold on both exact and outside-tolerance counts;
scalar disagreements still appear in the legacy `selection` record.
See the [arithmetic and qualification contract](../../../../../docs/models/glm/rtx-prefill-native.md#paired-half-index-diagnostic-draft).
The 570-second aggregate launcher deadline includes all model arms, copies
and checks; partial results or missing pair records cannot qualify a sweep.
