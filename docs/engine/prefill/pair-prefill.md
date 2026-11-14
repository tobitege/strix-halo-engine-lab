# Pair prefill and rank-0 continuation

Pair handoff is explicit: set `STRIX_QWEN_PF_PIPE=1` and `STRIX_QWEN_PF_PIPE_DECODE=1`,
then launch one process per rank with `--rank 0|1`. The transport default is
`usb4gpu` with the production size router; `--transport usb4auto` (or
`usb4-auto`) restores host-only. New direct seam bitwise/TTFT gates are pending;
the measurements below used the previous host path. See the
[transport decision](../../coord/transport-routing.md).
Use identical binaries, model assets, prompt IDs, numerical options and capacity.
The default pair chunk is 4,096 rows where admitted. Session save after a
two-node prefill requires a prompt length that is a multiple of the chunk size.
Measured on two nodes over USB4 (release check of the merged code, one fresh pair per run): tokens, full
logits and final state equal the single-node run at 32K, 128K and an odd 32,771-token prompt. Prefill
32K **12.1-12.3 s (~2,660-2,710 tok/s)** vs 19.2 s on one node; 128K **46.5-46.9 s (~2,800 tok/s)** vs 79.8 s.

The resolver runs short requests entirely on rank 0, without a connection or
handoff. Otherwise it pipelines a legal prefix, restores full-model execution
on rank 0, and consumes the suffix before generation. Plain prefixes require
full tiles or a >=16-row multiple-of-eight tail. MTP also respects its actual
owner ladder; its continuation uses the existing predictor/session frontier.
Rank 1 exits after its work. A loaded layer/transfer cost bucket compares serial
and pipeline estimates including handoff. Without one, the provisional crossover
is 28,672 rows: the available 32K/128K end-to-end savings fit crosses at 24,907,
rounded up on the measured 4,096-row grid. Remeasure this threshold with 4,096-row
chunks and the new handoff; it is an estimate, not a short-context speed claim.

Seam packing fences the D2D snapshot before source arenas can be reused by
independent streams. Diagnostic files use an owned copy of authenticated staging
bytes. Final-state wire v2 uses two pinned 16 MiB slots, a separate copy stream,
CRC32C per piece and one completion ACK after all H2D installs. Both ranks must
use this wire version. Memory overhead is 32 MiB pinned host per active handoff.

## One-way seam pieces on the device route

With `usb4gpu`, a seam piece above 64 KiB is eligible for a one-way transfer only when
the communicator admits a pure device route and the device backend supports it
(`dist::Communicator::transfer`). The sending rank posts no receive and the receiving rank
no transmit. An existing early receive is retired once through a symmetric scratch exchange;
subsequent one-way pieces do not prepost another receive. The receiving GPU checks the
in-band identity and word checksum of the frame as for a symmetric exchange, and the
sender's operation completes only against the receive its peer posted.

The seam CRC-32C of such a piece is computed on the device at both ends
(`xport::DeviceCrc32c`: one digest per 64 KiB span, folded on the host), so the piece is
neither copied to the host by staging on the sender nor read back by staging on the receiver.
This does not exclude host copies inside the transport's fallback. The value is the
host CRC-32C of the same bytes, bit for bit; the control stream (piece CRC, whole-frame CRC,
consumption ACK) is unchanged; native logit equality remains pending. Pieces of 64 KiB or less and the
host-only transport keep the host path.

Host or mixed communicator routes decline `transfer()` and use the symmetric fallback,
including reverse scratch traffic. With host streams available, the default route table
selects a mixed route above 1 MiB: a full 16 MiB seam piece uses that fallback, with host
staging for the mixed share. Selecting `usb4gpu` alone does not establish a copy-free or
one-way transfer; inspect the route table and actual one-way/byte counters.

`STRIX_PAIR_LINK_ONE_WAY=0` restores the symmetric exchange (a full-size scratch payload in
the unused direction); both ranks must set it the same way. `STRIX_PAIR_LINK_DEVICE_CRC=0`
restores the host CRC with its copies on the rank that sets it (the wire does not change, so
the ranks may differ). The run prints one `qwen-pair link:` line with the pieces and
the device bytes sent and received by the rank.

The link is shared (`pair::DeviceLink`, `src/xport/pair_device_link.hpp`). The Gemma 4 pair
uses it too: its default transport stays `usb4auto` (host path, unchanged), and
`--transport usb4gpu` uses the same admission, fallback and device CRC rules
(its receipt line is `g4-pair link:`).

Both switches default ON in the implementation; native execution qualification is pending.
Host fold arithmetic and vector-backed staging tests do not execute `DeviceCrc32c`,
`Communicator::transfer` or GPU queues. Qualification requires native CRC comparisons,
both directions, multi-chunk transfers, slot reuse, preposted-to-one-way transitions,
host/mixed fallback, independent CRC OFF settings and failure teardown, with actual
route/byte counters and complete token, logit and state equality for the workload.
Kernel compilation alone does not establish these gates.

## Stage balance, seam copies and the pair timeline

`STRIX_PAIR_TIMING=1` prints one `pair-timing` line per chunk and rank and a per-rank sum
(the shared `strix::dist::PipelineTimeline`): chunk span, seam acquire wait, producer fence,
seam pack/load, and the progress thread's D2H wait, CRC, wire, ACK wait, header wait and H2D
wait; the state handoff is the last row. Spans are host clocks of blocking calls; the
producer fence is where the host waits for the chunk's GPU work. Off, it adds no work.

Measured with it (4,096-row chunks, 32K, two nodes, host USB4 path): per chunk the
seam costs CRC 4-5 ms, wire ~18 ms for 105 MB and an ACK wait of ~4 ms; rank 0 never waits for
a frame slot. The seam is therefore not the limit. What is: stage A (layers before the
cut) also runs the embedding/PLE prelude, so with the even 24/24 split it is the slower stage.
The default cut is now **23** (`cut-source=measured-default`); `STRIX_QWEN_PF_PIPE_CUT=24`
restores the even split. Cut sweep (logits bitwise equal to the single-node prefill in every
arm): 32K 24: 12.44/12.51/12.50 s, 23: 12.08, 22: 12.31, 21: 12.63/12.65, 20: 13.00;
128K 24: 49.36/50.15 s, 23: 47.55/47.53.

Stage accounting. Stage B's host thread returns from a chunk ahead of the device, and the
device tail of each chunk (150-170 ms with 4,096-row chunks) used to drain inside the wait for
the next frame. Stage B therefore looked about 1.1 s cheaper per 32K prompt than it is, and the
pipeline looked unbalanced with an idle second stage. A kernel trace of stage B at cuts 22 and
23 shows the same time for every kernel, including the first layer after the seam: one more
layer costs stage B exactly one layer. Stage B now drains before it waits for a frame:
`exposed-seam-wait-ms` is true idle time, `gpu-drain-ms` on the same line (the `seam-fence`
column of `STRIX_PAIR_TIMING=1`) is the device tail, and the chunk time is the stage's real
work. The total is unchanged (both had to finish before the rows are scattered): 32K
11.61/11.68 s against 11.60/11.69 s, logits bitwise equal. With the honest numbers the stages
at cut 23 are 10.0-10.1 s (A) and 10.2 s (B) at 32K: the default cut is within 1 % of an
ideal split, so the remaining pair time is the pipeline latency of one chunk, not balance.

Half-layer cut. `STRIX_QWEN_PF_PIPE_CUT=N.5` cuts inside layer N, after its attention
sub-block: stage A runs the attention of layer N and stage B its expert half. The seam row is
the same object (four streams, the mix output, four coefficients; 25,616 bytes), taken at the
feed-forward hyperconnection mix instead of the attention mix, and the sequence state of
layer N stays with stage A in the handoff. Logits and the tokens generated after the handoff
are bitwise equal to the whole-layer cuts. Measured at 32K: 22.5: 11.82/11.85 s, 23:
11.61-11.69 s, 23.5: 11.84 s, 22: 12.08 s, so the default stays 23; the half step is for
prompts and models whose layer costs put the balance point inside a layer. A costs file
(`STRIX_QWEN_PF_PIPE_COSTS`) may add an `attention` object next to `layers` (per-layer cost
of the attention sub-block, same units); only then does the cost model also offer half-layer
cuts. The cut type and the half-layer balance are shared code (`strix/pair/pipe.hpp`,
`strix/pair/balance.hpp`) for any engine whose layers have two residual sub-blocks.

The seam pack (rank 0) and scatter (rank 1) moved each chunk's rows with 3 x 4,096 small
device copies (~27 ms per chunk on each rank). They now use one strided copy per component
(same bytes).

The remaining distance to 2x is structural for a two-stage layer pipeline: wall time is
about (single-node time with the pair's chunk size) / 2 x (1 + chunk / prompt), plus the
state handoff (0.11 s at 32K, 0.4 s at 128K). A different first or last chunk size does not
shorten it (the largest chunk sets the fill plus drain), and smaller uniform chunks lose
more per-node efficiency than they save (single node at 32K: 4,096-row tiles 20.6-21.1 s
vs 19.0-20.0 s with 16,384-row tiles, all bitwise equal).

## Prompt-bound MTP golden and replay

Keep the qualified asset/provenance/window argv in private
`assets/mtp-qualified.argv`, exactly one argument per line. It includes `--mtp`
and all required sidecar/target pins, policy and device budgets, but excludes the
model, prompt, pair flags, golden options and the two host-file budgets below.
Preserve its numerical options for every command. No private asset values are
published here. These Bash arrays spell the complete launch assembly:

```bash
mapfile -t MTP < assets/mtp-qualified.argv
COMMON=(MODEL.gguf --prompt-ids assets/prompt-32768.ids --context 131072 --generate 128
        "${MTP[@]}" --mtp-golden-max-bytes 34359738368 --mtp-replay-host-bytes 35433480192)
# Single-node capture; the destination must not already exist.
env -u STRIX_QWEN_PF_PIPE -u STRIX_QWEN_PF_PIPE_DECODE -u STRIX_QWEN_PF_LAYER_RANGE \
  strix-hip-decode-token "${COMMON[@]}" --mtp-golden-dump assets/prompt-32768.golden
SHA=$(sha256sum assets/prompt-32768.golden); SHA=${SHA%% *}
REPLAY=("${COMMON[@]}" --mtp-golden assets/prompt-32768.golden --mtp-golden-sha256 "$SHA")
env -u STRIX_QWEN_PF_PIPE -u STRIX_QWEN_PF_PIPE_DECODE -u STRIX_QWEN_PF_LAYER_RANGE \
  strix-hip-decode-token "${REPLAY[@]}"
# Copy exactly the same golden/argv/prompt/model assets to the other rank first.
# Run once per node, setting R to 0 and 1 respectively:
R=0
STRIX_QWEN_PF_PIPE=1 STRIX_QWEN_PF_PIPE_DECODE=1 STRIX_QWEN_PF_PIPE_CHUNK=4096 \
  strix-hip-decode-token "${REPLAY[@]}" --rank "$R" --transport usb4auto
```

The 32K golden contains 32,767 adjacent prompt pairs, about 33.89 GB for the
2560x4/248320 geometry. Reserve complete replay memory in addition to device and
model memory. Change the prompt to 1,193 tokens only after capturing and pinning
its own golden; a 32K golden cannot validate that prompt. Self-generated goldens
are calibration evidence, not independent model-fidelity qualification. A pair
golden dump is rejected; capture is single-node and replay is paired.

## Shared pair modules

The seam pipe (two slots, CRC32C pieces and whole-frame CRC, consumption ACK), its HIP staging, the stage-balance cut
chooser, the `--rank/--peer/--transport` handling with the link setup, and the per-request mode selection live in
`include/strix/pair/` (`pipe.hpp`, `balance.hpp`, `link.hpp`, `regime.hpp`) and `src/xport/hip_pipe_staging.hpp`.
Qwen binds them with its seam-header validator; the Nemotron (gemma4) pair route (`STRIX_G4_PAIR=1`, see the model
README) uses the same modules with its own seam layout.
