# CUDA Qwen prefill scheduling and tracing

The resident service computes rows `[0,P-1)` under the A8 prefill identity. The
node executes the final prompt row. The repeating block sizes are
`1024,2048,4096,8192`. Scheduling controls never change numerical identity:
every schedule produces bit-identical state and greedy continuations.

The default block schedule is `tail-large`: a short final remainder block moves
to the front when it is smaller than its predecessor, so the last block (whose
transfer cannot overlap later compute) is a large one. Row order stays
chronological: only the sizes move. With 32767 rows the plan becomes
`1023,1024,2048,4096,8192,8192,8192`; with 8191 rows it becomes
`1023,1024,2048,4096`. The block count, total rows and maximum capacity are preserved.
An already larger final block stays in place. Measured on the RTX PRO with the
node import (same greedy IDs as node-only prefill in every request), TTFT drops
12.7 % at 1K, 7.9 % at 8K and 5.0 % at 32K against the original order.
`--block-schedule default` (or `"block_schedule":"default"` per request) turns it
off and restores the original order.

### Flat blocks on the serve entry

`strix-cuda-qwen-serve` (and the gate tool `qwen_tf_logits`, which resolves as the serve entry) runs an untyped
schedule as one repeating block of the schedule maximum: `8192`, so 8192 rows are one block instead of
`1024,1024,2048,4096`, and 43008 rows are `2048,8192 x5` instead of eight blocks. The allocation is unchanged (every
workspace is already sized from the maximum) and so are the bits. The reason is work per token: every block pays the
per-launch skeleton of each kernel once, and the routed MoE visits every expert once per (layer, block) whatever the
row count (a tile of 20 rows costs almost as much as a tile of 64: the measured fit of the FAST project kernel is
1.4 us per tile plus 15 ns per assignment). The ramp exists for the hand-off entries, where a small first block
releases state to the node early; a decode session exports nothing, so the resolver names `block_schedule` among the
serve differences with this reason. The prefill service and the CLI entry keep the ramp (not measured on the hand-off
path). Lever `qwen_cuda_flat_blocks` (bitwise, default on): `STRIX_QRTX_FLAT_BLOCKS=0` or a typed `--blocks` /
`--block-rows` keeps the caller's schedule. Larger blocks were measured and not adopted: `--block-rows 16384` is not
faster at 32K / 42K (the MoE route workspace of 16384 rows leaves the cache-friendly range) and peaks 5 GB higher.

Receipt (server, strict ABAB, fresh server per arm, two prefills per length and process: first = cold, last = warm;
medians; ramp typed as `--blocks 1024,2048,4096,8192`, flat = the default):

| rows | FAST ramp cold / warm | FAST flat cold / warm | flat tok/s (warm) | exact ramp -> flat (warm) |
|---|---|---|---|---|
| 8192 (3 pairs) | 1325 / 1289 ms | 1164 / 1130 ms | 7,250 | 1873 -> 1716 ms (1 pair) |
| 32768 (3 pairs) | 4848 / 4886 ms | 4690 / 4731 ms | 6,927 | 7322 -> 7191 ms (1 pair) |
| 43008 (3 pairs) | 6459 / 6532 ms | 6348 / 6412 ms | 6,707 | 9837 -> 9753 ms (1 pair) |

Identity: `qwen_tf_logits` logit dumps (64 teacher-forced rows, 8K and 32K prompts, 63,569,920 bytes each) are
byte-identical between the ramp and the flat schedule, in the exact profile and in the FAST profile, so the FAST gate
receipt is unchanged. Kit: `scripts/bench/cuda/qwen_fast/prefill_abab.sh` with the two arms above.
These dumps establish target-logit equality only at the measured lengths. Complete recurrent, cache and MTP
state equality across schedules remains an outstanding native qualification gate.

`--final-block-rows N` optionally reserves a specific tail size after that
transformation, moving rows between existing blocks without adding blocks. Zero
disables the reservation. The feasible tail range is
`max(1,position-(count-1)*capacity)` through
`min(capacity,position-count+1)`; the target is clamped to that range. A one-block
prompt stays one block. A target above the loaded maximum is rejected. For
example, `--blocks 1024,8192 --final-block-rows 7167` gives `1024,7167` at 8191
rows. Moving from the default four-block 8191-row plan to two blocks would change
the block count, so that example requires this explicit base schedule.

POST `/v1/prefill` accepts `"block_schedule":"default"|"tail-large"` and
`"final_block_rows":N`. Omitted fields inherit startup defaults; explicit
`"default"` plus `0` restores the baseline within the same loaded service.
Overrides cannot enlarge the loaded capacity. Invalid overrides return 400
before contacting the node. `/health` reports the startup policy and capacity.

Every block contains 1 through the configured maximum real rows, bounded by
65520 (the padded row grids must fit CUDA grid y, whose limit is 65535). Row-independent operations pad to at least
16 rows and a multiple of 8; stateful operations see only real rows. Activation,
GDN scratch, MoE CSR and MTP workspaces are allocated once from the padded maximum
of `--blocks`. QSA uses a bounded row sub-batch workspace with full position
capacity. Scheduling cannot grow these allocations.

Boundary invariance follows the A8 row-local projections and fixed reduction
order, chronological GDN recurrence, and persistent KV/index and convolution
state. GDN carries its FP32 matrices and four BF16 convolution history entries;
PLE carries two token history words and nine convolution entries. Neither is
reset between blocks. QSA stores and attends at absolute positions. MTP pairs
use token `i+1` even when it lies across a block boundary, store at absolute `i`,
and export exactly `position-1` pairs plus the final held wide row. Padding never
advances those histories. CPU scheduling tests establish coverage and capacity,
not device bit equality. Complete fixture comparisons and greedy workload gates
are required for each candidate schedule.

Set `"trace":true` to receive a `trace` object. All event `ms` values share a
request-local `steady_clock` epoch created at worker entry. No trace clock reads,
callbacks or event storage occur for requests without tracing; existing aggregate
service/driver timers are retained.

* `gpu_ready` and `d2h_done` use stream-ordered host callbacks. Per-layer readiness
  is observed after the existing layer release point, not the earliest possible
  internal operator release. MTP readiness follows its compute span; the held row
  is marked after main-model compute. Other readiness precedes D2H at the current
  block/fixed release point; host history words use equivalent host marks.
* `piece` supplies component, first row/byte offset, row count and byte size.
  Its `id` joins readiness, D2H, enqueue and chunk membership. Split sender pieces
  share the producer id and carry their own range/bytes.
* `chunk_piece` has a `chunk` sequence number; encode and POST events use that
  sequence as `id`. `encode_start/end` include packing, codec and SHA work.
  `post_start/ack` bracket the complete HTTP exchange.
* `commit_wait_start` precedes drain; `commit_start/end` bracket commit work after
  all chunk acknowledgements.
* `block_start/end` give absolute boundaries. `trace.blocks` contains `elapsed_ms`
  and `compute_ms`: the sum of main-model and MTP compute-stream callback spans,
  excluding subsequent D2H and the emitter join between those spans. These are
  observed stream wall intervals, including launch starvation and instrumentation,
  rather than a sum of kernel durations. `elapsed_ms` includes copies and waits.

Callbacks and serialization perturb traced requests. Use untraced interleaved
runs for TTFT comparisons and separate traced runs to explain overlap. A nonzero
`dropped_events` means the trace is incomplete: an allocation failed inside a
callback, or the request exceeded the 1,048,576-event cap. Timestamps from different machines must not be subtracted.

The isolated Windows build uses `STRIX_CUDA_QWEN_ONLY=ON`, CUDA architecture 120
(pass `-DCMAKE_CUDA_ARCHITECTURES=120`; the top-level default also lists 89 and 86),
and the exact-arithmetic flags `--fmad=false --prec-div=true --prec-sqrt=true
--ftz=false`. It builds `strix-cuda-qwen-prefill-service`, `qwen_prefill_e2e`, and
CPU-only `qwen_prefill_schedule_test` and `qwen_handoff_codec_test`. CTest runs
only those two CPU checks. The gate runs the tail-large default;
`STRIX_QWEN_TAIL_LARGE=0` restores the original order and
`STRIX_QWEN_FINAL_BLOCK_ROWS=N` sets a reservation, alongside its existing block
and asset settings. The `xsched:TOKENS:N[:R]` job runs one prompt under the
original order, tail-large and (R > 0) an R-row final-block reservation, and
requires every exported component to be byte-identical across them.

## Weight packs (shared admission)

The Qwen CUDA engine admits a GGUF through `include/strix/weight_pack.hpp` (shared, model-agnostic) with the packs
the family declares in `include/strix/qwen_weight_packs.hpp`: `qwen38-ud-q4kxl-a8-v1` (product file, exact A8 kernels),
`qwen38-gsq-rco-iq3s-v1` (ISTA-DASLab GSQ-RCO IQ3_S and siblings; FAST: dense tensors transcoded to Q8_0 at load,
routed experts native IQ) and `qwen38-gguf-native-v1` (any other decodable mix, unqualified). The model path selects
the pack; `strix-cuda-qwen-serve --print-effective-config --model M --tables T --rope-table R` prints the pack id,
arithmetic class, route, tensor fingerprint and type census without loading weights, and the engine load runs the
same admission (the `loaded` receipt carries `weight_pack`, `pack_arithmetic`, `pack_fingerprint`).
Admission uses original merged headers, before upload changes storage types. Gate/up tensors in each layer must
share a type. Inspection consumes device options but performs no device discovery or binding.
The fingerprint identifies the named type/dimension inventory, not weight contents or numerical behavior;
it hashes host-native integer bytes and is not a portable persistent-cache key. Admission borrows its pack
declaration, which must outlive the receipt. Host fixtures do not qualify device execution or model fidelity.
Served prefill by length for any OpenAI-compatible server: `scripts/bench/common/prefill_client.py` (a
`serve-decode.sh --client-cmd` client; finite positive client TTFT and measured positive usage.prompt_tokens are
required). Rows preserve input order without cold/warm labels or length grouping. `tools/serve_bench.py` reports
`peak_1s_tok_s` only with measured per-chunk token counts; ordinary SSE text plus final usage leaves it null.
