# MiniMax grouped host prefill

The grouped communicator backend uses the existing chunked host pipeline. Both
scheduling changes below default ON through the MiniMax resolver, gated per
grouped call: every member must have at least `STRIX_MM_PF_ASYNC_MIN_ROWS` rows
(default 2048). A group with any smaller member takes the ordinary `comm.start`
path, with byte-identical scheduling to `STRIX_MM_PF_ASYNC_START=0`. The decision
uses only the ordered member row counts and is identical on both ranks.
`STRIX_MM_PF_ASYNC_MIN_ROWS=0` removes the row gate (unconditional grouped async
start); the global off switch still wins. Invalid or overflowing thresholds
are rejected before device work. Inspect the threshold with `--print-effective-config`.

- `STRIX_MM_PF_ASYNC_START=0` disables early receive-destination publication and
  worker-side pinned receive-staging reuse. When enabled, MiniMax opts in on its
  own communicator, without exporting `STRIX_DIST_ASYNC_START` to other engines.
- `STRIX_MM_PF_LATE_SRC_FENCE=0` retains the immediate main-stream source-free
  wait after a successful async start (early destination only). With both switches
  enabled, the source-free wait precedes the next producer of that member's
  partial. The late switch has no effect when async start is disabled or falls back.

`STRIX_MM_BASELINE=1` switches both off unless explicitly overridden. Inspect them
with `--print-effective-config`; restart both ranks when changing startup settings.
The module backend, small/unchunked payloads and direct routes retain their existing
start/finish semantics. A false `start_async` result uses neither caller event.

## Exact scheduling contract and lifetime

Producer/post kernels and launch shapes, ordered group members (including
4096+443), payload bytes, rank-0/rank-1 summation, exchange sequence/tags, chunk
framing and CRC remain unchanged. All compute, including shared scratch, stays on
the main stream. Only TX/RX copies and host transport work can overlap that compute.

Timing-disabled `last_r_use` and `p_free` events are allocated once at startup for
each member of each activation bank. The destination event is initially recorded
before first use, then recorded after every admitted async-group post that reads
that member's peer partial. Ordinary groups record no destination events. If a
bank returns from an ordinary group to async, the admitted call records any
untracked ordinary readers before publishing destinations. RX waits on that
generation before overwriting the peer partial. TX records `p_free` after its last D2H read. A successful async start alone arms the
source wait; the next producer waits before overwriting the local partial. Local
post and TX may read the same partial concurrently.

Every resumable unit finishes its exchanges, enqueues every final post, and closes
all remaining source reads on main before returning. Thus intervening decode,
ungrouped tails and later groups cannot overwrite an outstanding source. Events
and grouped partial buffers persist across resumes. Park/unpark, cancellation,
retirement and bank reuse drain device work before changing ownership. Nested
children have a separate activation bank and separate event generations; current
admission limits them to one chunk. Shared partial/scratch buffers contain no live
parent data after retirement, so children may borrow them serially. Failure aborts
and joins submission workers before device synchronization. Destruction drains
outstanding work before destroying caller events. The communicator owns pinned
staging and protects its reuse on the worker before the next transport write.

## Measured regime

GPU measurements used the ungated build on two nodes,
TP2; complete token IDs were identical in every arm. Both changes were enabled
together versus `STRIX_MM_PF_ASYNC_START=0`:

| Workload | Both changes (s) | Off (s) | Result |
|---|---|---|---|
| Lone 7940-token prefill, members 4096+3844, scheduler off, ABBA | 17.44 / 17.49 | 18.00 / 17.82 | -2.5 % |
| Same, BAAB | 17.25 / 17.42 | 17.70 / 17.79 | -2.3 % |
| Lone 4539-token prefill, members 4096+443, 8+8 runs | mean 11.64 | mean 11.55 | Neutral |
| Mixed serving: 3 decoders + 4539-token long + 2 shorts, default policy, ABBA | long TTFT 16.00 | long TTFT 15.50 | Unresolved |
| Same, BAAB | long TTFT 15.29 | long TTFT 15.47 | Unresolved |

An earlier mixed-serving AB/BA pair showed long TTFT deltas +0.26 / +0.84 s.
Thus 4096+443 is neutral to slightly slower, with no reproducible mixed-serving
gain, and is gated off by default. The balanced 4096+3844 regime retains the
reproducible -2.3 / -2.5 % gain. These measurements qualify token agreement for
the measured arms; full prompt-output/state/logit and complete-workload lifetime
gates below remain pending. They do not independently attribute gains to either
switch, or establish a universal crossover at 2048 rows.

## Diagnostics and qualification

`STRIX_MM_PF_TIMELINE=1` reports pipeline receive-staging reuse wait separately
from ordinary host receive-staging reuse for grouped units. Pipeline time includes
the caller wait on the ordinary route or the worker wait on the async route; worker
time can overlap compute and must not be added to wall time. Existing GPU busy/gap
labels measure event intervals, not hardware utilization.

Host tests cover resolver defaults, independent off switches, baseline overrides
the per-call all-member threshold decision (including boundaries, zero and
global off), and source/route inventory. `strix-hip-dist-hostpath-tests` exercises fallback,
instance opt-in with the global switch off, immediate/late source fences,
member/bank event reuse, retirement, cancellation and corruption. These API tests
require a HIP GPU and do not qualify complete MiniMax execution.

Before timing, compare both ranks against the unchanged binary with the same model,
profile, capacity and prompt/chunk plan: complete IDs, all normalized prompt outputs,
KV/IK state and first/continuation logits. Include resumed units, parked children,
cancellation and bank reuse, then the complete workload. Isolate early destination
with the late fence off before testing both changes. Keep private fixtures and
receipts outside Git. See [communicator](engine/transport/communicator.md#minimax-grouped-async-start).

## Draft FAST_I4

`STRIX_MMX_MODEL_FORMAT=fast_i4` is an opt-in lossy prefill identity. INT4 is produced per layer
during prefill in two reusable buffers; memory +3.6 GiB; context never shrinks; decode unchanged.
Each slot includes its packed weights, FP16 group scales and reusable output-sign pointer tables;
read-only source tables and the existing prefill workspace are accounted separately at startup.
EXL3 weights remain resident and every decode/MTP regime uses them. Allocation failure refuses
FAST_I4 rather than reducing configured capacity or silently falling back to a partial INT4 model.

The single adopted library `convert_exl3_iu4` converter runs once per MoE layer per prefill group,
then every chunk uses slot `L & 1`. The same rule covers ungrouped tails and paused layer ranges.
`STRIX_MMX_FAST_I4_SCOPE=all` (default) runs gate/up/down on IU4; `gate_up` keeps the original EXL3
FP16 down path and skips down conversion. Scope remains a quality-gated choice.

Quality policy (default on). INT4 noise is flat per layer (2-5 % of the MoE output energy, half of it from the
4-bit gate/up input), but an early layer's error is amplified by every later router and sparse-attention
selection, so the same local error costs three times more KL in layers 3-32 than in layers 33-59. By default
FAST_I4 therefore keeps MoE layers 3-37 on the exact EXL3 operator and runs INT4 in layers 38-59
(`STRIX_MMX_FAST_I4_EXL3_LAYERS`, a list of layers and ranges such as `3-37`, `!38-59` for the complement, or
`none` for INT4 in every MoE layer, the pre-policy identity). EXL3 stays resident, so the policy costs no memory
and skips the conversion of the exact layers. Prompts shorter than `STRIX_MMX_FAST_I4_MIN_ROWS` (default 4096,
`0` = every length) prefill on EXL3 in every layer, bit-identical to the exact format: the per-layer conversion
is a fixed cost and INT4 only pays above about 3K rows. Measured at 2K on four chat-template prompts against the
exact format (pooled last 256 rows each, `MIN_ROWS=0`): KL 0.0139, top-1 agreement 96.5 %, dNLL +0.006 with the
default policy, against 0.065 / 90.9 % / +0.024 with `none`. The fused chain is bit-identical to the unfused
calls under the policy. `STRIX_MMX_FAST_I4_SHADOW=1` (measurement only) runs the exact MoE and the exact-weight
A4 MoE on the same input of every INT4 layer and prints the relative output error per layer at prompt
completion; the live result stays INT4. `STRIX_MMX_FAST_I4_A8=gate_up|down|all` with
`STRIX_MMX_FAST_I4_A8_LAYERS` is a study path (two A4 planes per GEMM on the unfused calls); it is slower than
the exact operator and is not part of the policy. The layer-list grammar, the shadow reducer and the two-plane
quantiser are shared (`src/core/layer_set.hpp`, `src/core/i4_shadow.hpp`, `src/core/activation_quant.hpp`).

Serial conversion on the engine stream is the default. `STRIX_MMX_FAST_I4_PREFETCH=1` converts
only the next layer within the current range on a side stream, ordered by slot-free and ready
events, without adding a host synchronization. Retain it only if 8K TTFT improves by at least
0.4 s over serial. `STRIX_MMX_FAST_I4_STATS=1` reports event-based converter and whole-group MoE
times per layer at the existing prompt completion fence; the latter includes inter-chunk gaps.
The report belongs to the prefill task that completes (a nested child neither prints nor clears
its parked parent's), gives IU4 and EXL3 launch counts for every MoE layer that launched (also
for a prompt under the row gate, which has no IU4 launch), and charges sampled time to the
arithmetic of the sampled group.
Kernel-chain timing should also use the prefill timeline markers.

With scope `all`, the INT4 chain is the default (`STRIX_MOE_I4_CHAIN=0` selects the unfused calls,
same bits). It quantises the gate/up input once, runs gate/up as one paired grouped GEMM whose
epilogue applies SwiGLU and the down input transform, then the down GEMM and the ordered combine.
It runs once per chunk, so chunk c's MoE overlaps the exchange of chunk c-1.
The paired gate/up keeps pass-0 FP32 gate values in global scratch (same thread reads them back in
pass 1, no barrier; `STRIX_MOE_I4_CHAIN_GSAVE=0` selects the register save). The register save needed
so many VGPRs that the pair was forced to plan 1 (BM64) while the unfused call ran plan 2 (BM128): at
32K that cost MoE GEMM time and slowed the overlapping attention kernels (+1.2 % TTFT). With the global
save the plan is chosen freely (VGPR 183-239, no spills; scratch 5·T·inter·4 bytes, ~126 MB at T=4096).
Measured on the TP2 pair: 8K 12.12 s versus 12.34 s unfused and 12.76 s versus 12.98 s (-1.7 %, both
orders); 32K 55.23 s versus 55.74 s (-0.9 %, mean of six pairs, three per order; single runs vary ±3 %).
Chain GEMMs run at 31 % (8K) and 32 % (32K) of the iu4 WMMA peak. State, final-row and token anchors are
equal and the ranks agree in every run; no regime gate is needed.

`STRIX_MMX_PF_IU4=0` and `--ab pfi4:PROMPT.u32:128` retain the same model load and buffers while
selecting the original EXL3 prefill arm; `STRIX_MMX_FAKEQ_A4` remains available for quality study.
There is no resident-layer budget, replace mode, or INT4 decode policy. Exact/default arithmetic
and anchors remain the reference. Native compiler, execution, memory, timing and model-quality
gates are pending; predicted throughput is not qualification evidence.
