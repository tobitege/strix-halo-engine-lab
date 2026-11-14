# Complete output-row split deciding fixture

This is a separate experimental operator fixture. The ordinary Qwen commands,
model pack, numerical profiles, prefill paths, caches and defaults are unchanged.
It does not implement a full model split or distributed prefill. The native route
requires explicit build selection and `--enable-output-row-split`.

The fixture loads the actual GGUF Q8_0 `output.weight` tensor at the production
2560 by 248320 geometry. It calls the existing production
`launch_hip_q8_head_rows` BF16 prefetch8/wave8 operator. A rank owns complete output
rows with their entire original input dimension. Rank0 owns [0,124160), rank1
[124160,248320). Each rank holds 337715200 packed weight bytes during the split
phase. There is no partial-dot or rank-subtotal reduction. The ordinary dense
reference phase first holds the complete 675430400-byte tensor on each rank;
its device allocation is released before allocating the owned shard.

Every input is an independently captured, already normalized BF16 hidden row.
The sequence contains 1..32 rows; this is a sequence of head invocations, not
batched model prefill. Hidden input and complete FP32 vocabulary references must
come from the same unchanged target/profile at the declared context/positions.
The capture owner must retain that target's source/binary/identity and full-state
receipts. A fixture-generated reference alone cannot establish model fidelity.

Before split execution, both ranks independently verify the full GGUF SHA256,
packed tensor SHA256, hidden/reference extents and SHA256, binary, source revision,
recipe and environment. The dense native head must match every bit of the supplied
full-vocabulary reference. Seven scalar rows per input include both sides of the
shard boundary; the independent scalar Q8/BF16 leaf and wave32/local reduction
must match the native values bitwise. `--scalar-leaf fma|separate` is declared
before execution from the actual target compiler/ISA receipt. Never try both to
select a passing result. Scalar probes are not all-vocabulary scalar validation.

The stable project `usb4_host_auto` and `Communicator` transport are used unchanged.
Each head invocation exchanges 496640 bytes per direction, plus the existing
128-byte communicator trailer. Both ranks gather in original vocabulary order,
compare every FP32 bit and firstmax against the independent reference, and exchange
bound result receipts. Input, source, target, reference, world, shard, context,
capacity, epoch/generation, sequence, binary and effective environment are bound.
Fixed control payloads are 512 bytes. The sequence has one offer, one reference,
one payload and one result-control exchange per input, then one final commit.
Payload bytes/rank are `496640*inputs + 512*(inputs+3)`; trailer bytes/rank are
`128*(2*inputs+3)`, excluding transport setup/discovery and topology calibration.
These are byte counts, not latency, bandwidth or throughput measurements.

The real planner/control codec is shared with the host tests. Bounds, alias,
readiness, shape, identity, truncation, stale-generation and failure checks use
runtime exceptions and remain active under NDEBUG. A peer failure poisons the
sequence; no partial row or retry becomes publishable. Device and host copy
owners remain retained through communicator cancellation and local drain.
Publication follows both final control receipts; this is not recovery or a
consensus service after a successful peer subsequently disconnects.

A positive result is `EXACT_OPERATOR` for this recipe/reference only. It does not
qualify full target state, distributed model generation, memory fit at a model
capacity, MTP, cache reuse, prefix/session keys, chunked prefill or serving.
MTP/draft/sidecar, prefix/session cache, prefill-chunk and batch flags are named
`UNPROVED_COMPOSITION_REFUSED`. Ordinary engine behavior is unaffected.

## Host build

No HIP configuration or device compiler is needed:

```sh
cmake -S lab/qwen_output_split -B build-output-split-host -DCMAKE_BUILD_TYPE=Release
cmake --build build-output-split-host --parallel 12
ctest --test-dir build-output-split-host --output-on-failure
```

Run the host executable both with and without NDEBUG. It exercises actual shard
planning, gathering, wire codec, both-rank admission, control ordering, failure
poisoning, CLI refusal and scalar conversion/recipe checks. These tests do not
prove HIP compilation or native equality.

## Explicit native build and run contract

The root build is unchanged. Opt in through the separate root project hook:

```sh
cmake -S . -B build-output-split-native -DCMAKE_BUILD_TYPE=Release \
  -DSTRIX_ENABLE_HIP_RUNTIME=ON -DSTRIX_HIP_ARCHITECTURES=gfx1151 \
  -DBUILD_TESTING=OFF -DSTRIX_FAMILY_DEEPSEEK41=OFF \
  -DCMAKE_PROJECT_strix_engine_lab_INCLUDE=lab/qwen_output_split/register-native.cmake \
  -DSTRIX_SPLIT_SOURCE_SHA=FULL_CLEAN_SOURCE_COMMIT
cmake --build build-output-split-native --target strix-qwen-output-split-native --parallel 12
```

The SHA must be the actual full clean source packet revision. Configuration is
not a build receipt. Retain compiler/flags, evaluated source closure, binaries
and full SHA256 on both ranks. A physical device owner controls build and native
turns; each turn includes cleanup and must stay within its authorized deadline.

Run one process on each of two gfx1151 nodes, using identical arguments except
rank and the new output path. All digest arguments below are required sealed
values, not executable placeholders:

```sh
build-output-split-native/strix-qwen-output-split-native \
  --enable-output-row-split --rank R --world 2 --transport usb4-auto \
  --model MODEL.gguf --model-sha MODEL_SHA256 \
  --tensor output.weight --tensor-sha PACKED_TENSOR_SHA256 \
  --hidden HIDDEN.bf16 --hidden-sha HIDDEN_SHA256 \
  --reference REFERENCE.f32 --reference-sha REFERENCE_SHA256 \
  --target-identity TARGET_CAPTURE_IDENTITY_SHA256 --source-sha FULL_CLEAN_SOURCE_COMMIT \
  --inputs N --capacity CONTEXT --position FIRST_POSITION --epoch FRESH_NONZERO_EPOCH \
  --scalar-leaf DECLARED_LEAF --budget-bytes ADMITTED_BYTES --reserve-bytes RESERVED_BYTES \
  --deadline-ms 430000 --out NEW_RAW_LOGITS.f32
```

The explicit quota, HIP free observation and host MemAvailable must each admit
the declared conservative storage peak plus reserve on BOTH ranks. The peak
includes full host/device reference weights, input storage, full reference and
candidate rows and communication staging. It is not process-total accounting:
allocator, driver and host-object overhead need the separately admitted reserve.
No context, slots or lazy growth are reduced for admission. No cache is enabled.

Full raw rows are written exclusively only after complete bilateral commit;
existing output paths are refused. The absolute process watchdog is bounded by
430 seconds, leaving cleanup time in a 470-second native turn. Existing transport
waits/abort remain bounded. No native test or turn is launched by configuration.

## Remaining model work

The first-generation `HipDistributedStageResources`,
`HipDistributedKnownSequence` and `HipDistributedGreedyRequest` were test-only and
were removed in R3-B1; PP work starts from the current transport. This fixture supplies a
separate complete-output-row seam and a runnable native decision gate, without
another model PP executor or positive transport-qualification assertion.

A model route still needs actual boundary/bootstrap owners, both-rank setup
admission, complete causal chunk/preload ordering, all canonical state slices,
full raw logits and complete token-ID gates against the same target. Cache keys
and MTP state/history/pending/RNG/rollback/verify-window binding need separate
implementation and qualification before those compositions can be enabled.
Arithmetic-changing reductions require a separate numerical identity and owner
bands before quality work. No new timing, quality, native qualification, expected
tok/s, percentage gain or throughput floor is established here.
