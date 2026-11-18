# MiniMax layer-cut runtime kit

Experimental, explicitly selected route; native compiler, execution, fidelity and complete-workload gates are pending.
This kit contains commands and qualification procedure, not measured speed or exactness claims.
The existing device-only and full MMRX handoff routes remain the defaults for their existing commands.

## Build and host checks

Build the CUDA producer with `src/cuda/models/minimax/build_prefill.bat` on its CUDA host, and the usual
`strix-minimax-rank` HIP target on each pair node. The producer links `src/handoff.cpp` and `src/bp_codec.cpp`.
Host-only tests need neither runtime nor model assets:

```powershell
cmake -S tests/pair_layercut_host -B build/layercut-host
cmake --build build/layercut-host --config Release
ctest --test-dir build/layercut-host -C Release --output-on-failure
```

## Run

Supply the same complete prompt token IDs and model assets on all hosts. The producer needs the safetensors index,
embedding shard and shards containing layers `[0,K)`. It skips files containing only later layers, and does not read
layer K's input norm; the pair owns that seam norm. A shard containing both ranges is necessarily opened.
The rope table must cover **all N prompt positions**, including the last row. Unlike full MMRX offload,
layer-cut prefills **all N rows**; the pair computes the first token with its head on row N-1.

Start the node script on rank 0 (rank 1 is launched through the supplied SSH host). Ports in `--layercut-listen`
retain the existing base-port convention: rank 0 listens on BASE_PORT and rank 1 on BASE_PORT+1.
The relay listener is a separate port on rank 1. No script assumes a machine or account name.

```bash
bash scripts/bench/layercut/run_pair.sh \
  ./build/strix-minimax-rank ./model ./prompt.u32 \
  PEER_SSH_HOST PEER_IP 46000 46010 24 4096 64 2 ./layercut-out
```

Then on the CUDA PC:

```powershell
./scripts/bench/layercut/run_pc.ps1 -ModelDir ./model -Rope ./rope.bin -Prompt ./prompt.u32 `
  -Node0 NODE0_IP -Node1 NODE1_IP -Port0 46000 -Port1 46001 -K 24 -Repeat 2
```

`--layer-cut K` implies resident timing mode with exactly the prefix layers loaded. An explicitly supplied
`--resident` must equal K. `--repeat 2` / COUNT=2 use the same process and resident weights; the first run is labelled
cold and subsequent runs warm. Both ranks finish their requested decode steps before accepting the next session.
The node script uses relay. For direct mode launch both ranks with `--cut-act direct` and omit relay controls;
use `-Act direct` on the PC. `-Codec bp` selects lossless bp-v1 for every repetition of this route.

The default pair identity is `default/c4096`: production prefill bits depend on the pair's chunk boundaries (a row's
MoE K split class follows the rows its expert holds in that chunk), so the identity names the pair-alone chunk rows
(`--max-rows`, default 4096) and every block must be a multiple of them (identity grain; the resolver refuses other
blocks and `auto` only proposes multiples). For the FP32 pair profile, set `STRIX_MM_PREFILL_IDENTITY=fp32`
on both nodes (the node script forwards it to rank 1) and use `-PairIdentity fp32` on the PC. Additional active
prefill policies append `;` and the existing prefill policy identity. Copy that exact string into
`--cut-pair-identity`; a mismatch is refused before compute.
Omitting `-C` in the PC wrapper leaves block selection to the shared resolver, including environment overrides.

## Controls and planning

The shared resolver is `include/strix/pair/layercut.hpp`: CLI overrides environment, then defaults.
`--split` / `STRIX_SPLIT`: `device-only` (default), `handoff-full`, `layer-cut`, `auto`.
`--cut` / `STRIX_SPLIT_CUT`: 1..59 or `auto`; `--cut-block` / `STRIX_SPLIT_BLOCK`: positive multiple of 16,
or `auto` (default: the identity grain, 4096 for the default pair identity, else 2048); `--cut-act` / `STRIX_SPLIT_ACT`: `link` (default), `relay` or `direct`.
`link` sends the same producer bytes as `relay`, but rank 0's rows reach rank 1 over the pair's own link inside the
engine (one control exchange per prefill chunk, at the same schedule point on both ranks); no relay ports.
`--cut-xport` / `STRIX_SPLIT_XPORT`: `act-first` (default; contract v2) or `fifo` (v1, the previous order, off switch).
`--cut-kv-frag` / `STRIX_SPLIT_KV_FRAG`: KV rows per piece under act-first (multiple of 16, default 1024).
The rank follows the contract version in the producer's HELLO. Bench only: the producer's `--abab-xport` alternates
fifo / act-first per `--repeat` run (the resolved one first) for a strict in-process ABAB.
The rank route requires `--layercut-listen NAME:PROMPT.u32:BASE_PORT:STEPS:COUNT` and a cut policy on both ranks.
Relay adds rank 0 `--cut-relay HOST:PORT` and rank 1 `--cut-relay-listen PORT`.
`--print-effective-config` prints the resolved `split KEY VALUE (cli|env|default|planner)` lines without device work.

For an automatic field, callers supply the model shape, prompt length and declared whole-model rates
`STRIX_SPLIT_RTX_ROWS_S`, `STRIX_SPLIT_PAIR_ROWS_S`, plus `STRIX_SPLIT_LINK_MBPS` (decimal MB/s), or
`STRIX_HANDOFF_LINK=1gbe|2.5gbe|10gbe|usb4` (118/290/1180/4600 MB/s).
The shared `pair::chain::choose` performs the search; these rates seed a prediction, never qualification.
The grain-based cost scale is a planning estimate from whole-model 8K timings; it does not qualify other policy
suffixes, transports, hardware, layer ranges or tails. Qualification requires matching profiles and complete workloads;
repetitions within one process do not establish interleaved A/B gains.
For executing the cut experiment, keep `--split layer-cut` and use automatic cut/block as needed on **both** sides
with identical inputs. General `auto` may resolve to the pair's device-only route; it must not be interpreted as
an executable CUDA-only/decode route. The CUDA executable refuses that outcome. Planner predictions currently
represent one aggregate link; direct/relay duplicate-link contention is not a measured cost model.
The deployment adapter advertises protocol `minimax-m3.1/layercut-v1` with gate `pending` and unknown range footprints;
it cannot become the automatic production selection.

## Wire and ownership

`strix::handoff::Sender/Listener/Session` provide framing, identity/bp-v1, piece checksums, ordered consumption,
END manifests and REPLY. Hello.model is `minimax-m3.1/layercut-v1`, rank 0/1, world 2.
Hello.meta is little endian: u32 version=1, cut, block, hidden; u64 rows, token FNV-1a-64;
u32 identity byte length; identity bytes (no NUL). Version, cut, block, hidden, rows, token digest and identity must match.
The fixed identity is `mm-layercut-v1 rtx=cuda-fp32id L[0,K) pair=IDENTITY L[K,60)`.

Contract v1 (fifo): for each block, ACT (`0x4C434101`, index=block ordinal, row0, real rows) precedes KV for layers
0..K-1 (`0x4C434B02`, index=layer). Contract v2 (act-first, Meta.version 2): ACT blocks keep block order on their own
cursor; KV of each layer arrives as contiguous row fragments (row0 = that layer's next row) in any interleaving with
ACT and other layers; gaps, duplicates and incomplete coverage at END are rejected. The producer keeps [0,K) KV
resident on the GPU, sends ACT as urgent pieces over both producer links (`strix::handoff::Lanes`: KV is handed to a
Sender only while no ACT of any destination is unreleased) and exports KV fragments into a bounded pinned pool, so
the GPU never waits for KV. bp-v1 (`--codec bp`), if on, applies to KV only; ACT is always sent uncoded.
Per-rank KV pinned storage is `max(8, floor(32 MiB / fragment_bytes)) * fragment_bytes`, where
`fragment_bytes = min(cut-kv-frag, cut-block) * 1280`. The 32 MiB value is a sizing target, not a cap:
block=fragment=8192 uses 80 MiB per rank, in addition to the two ACT slots and other staging allocations.
Timeline lines: `layercut-tl pc` (per block: ring wait, GPU+D2H, ACT queued, ACT/KV released per rank) and
`layercut-xport pc` on the producer; `layercut-tl rR` (ACT stored, relayed, last KV stored, fetch start and wait),
`layercut-rank` and `layercut-act` (XXH64 of the staged ACT, exactness witness) on each rank. ACT is FP32 residual **h**, `[rows][6144]`, 24,576 bytes/row, never normalized hn.
Relay ACT goes only to rank 0 with `handoff::kRelay`; rank 0 submits it to rank 1 before publishing local rows.
Rank 1's producer session then carries KV only; its separate relay session carries ACT only with the same metadata/session ID.
Direct ACT goes to both ranks with flags=0. KV is rank-local BF16 K, V, IK in component-major planes
(512/512/256 bytes/row); both ranks total 2,560 bytes/row/layer. The 2,304-byte deduplicated-index form is a planner
binding constant, not a negotiated wire format. Both ranks receive their own IK in v1.
Unknown, duplicate, out-of-order, incomplete, wrong-size or wrong-range pieces fail the session before state import/reply.

The CUDA prefix caches persist across blocks. Two pinned slots bound outgoing plaintext; callbacks return each slot
only after all its pieces release. Copy-stream waits follow per-layer CUDA events; network backpressure never inserts
a compute-stream wait. Slot reuse can wait on the host. Receiver activations use a pinned full-prompt buffer and a
condition variable; KV stages all N rows per prefix layer in host memory and scatters component planes before import.
The pair recomputes the cut's input Gemma norm with the same reduction as post-exchange and supports grouped chunk units.
An existing group can await several ACT blocks before launching; it is not forced to serial chunks.
The block sets a unit-width cap, not an exact scheduling unit: group width may cap it further and grouping can be
disabled. Units can cross block boundaries (for example, three-chunk blocks with group width two).

## Receipts and gates still required

Producer TTFT ends when rank 0's u32 token reply arrives, after both-rank prefix KV import. `rtx compute` is the sum
of block host intervals including embedding upload and D2H completion; it excludes ring backpressure and the reply wait.
Per-block receipts show real rows and submission times. Identity-codec ACT/KV last-byte times use sender release callbacks;
bp-v1 releases after encoding, so its printed last-byte values are explicitly aggregate upper bounds from session open.
They are not independent measured ACT/KV wire times. Pair receipts give prefill (including ACT waits), prefix KV import,
ACT condition-variable wait and first/last consumed-piece times. Decode prints complete IDs and FNV, plus a u32 output file.

Before qualification: build CUDA and HIP; run K=1/24/59, N smaller than C, exact multiples and short tails, C=16/2048/4096,
relay/direct, identity/bp-v1 and two repetitions. Compare full generated IDs, final-row logits and all 60 layers of K/V/IK
against the declared mixed-stage identity reference, including the FP32 pair profile and grouped/serial chunk schedules.
A mixed production pair profile is not automatically equal to either all-CUDA or all-pair prefill.
Inject wrong metadata, repeated/missing pieces, connection close, relay failure and stalled peers; verify no token reply
or decode uses incomplete state and all readers/copies retire. Check memory under long prompts and bounded ring pressure.
Run complete workloads and interleaved medians only after fidelity passes; record kernel/IOMMU state with timing receipts.
No native execution or fidelity result is established by the host tests (native receipts: see the PR / docs).

## Measured recipe (MiniMax M3.1, 1 GbE)

- Ports: rank r listens on BASE_PORT + r, so the relay port of rank 1 must differ from BASE_PORT + 1; the producer sends
  to HOST0:BASE_PORT,HOST1:BASE_PORT+1.
- Pipelining: the listener runs each block as one prefill unit (`block / max-rows` chunks, bit-neutral) so the pair
  starts on the first block. Keep the pair-alone `--max-rows`; the old `STRIX_MM_PF_GROUP=2 --max-rows C/2` recipe
  changed the pair's chunk identity (logits differ from the pair alone) and is refused for the default identity.
- Unit gate (bit-neutral): a `--session` with `--unit-chunks 1` prints the same state / final-rows fnv as the default.
- Exactness gate: ranks with `STRIX_MM_PREFILL_IDENTITY=fp32` and the producer with `--cut-pair-identity fp32`; the
  layer-cut "state fnv" line must equal the node-only FP32 `--session` state fnv on both ranks.
- Quality: `STRIX_MM_QUALITY_OUT=PREFIX STRIX_MM_QUALITY_ROWS=N` on the ranks dumps the prefill logits of a `--session`
  (PREFIX-NAME-*) and of the layer-cut (PREFIX-cut-NAME-*); copy rank 1's files next to rank 0's and run
  `logits_compare.py REF_PREFIX TEST_PREFIX N` (top-1 agreement, KL, top-2 margins). Dumps sit before the KV import:
  do not time those runs.
