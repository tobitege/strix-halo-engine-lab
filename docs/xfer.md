# State hand-off transport and shared policy

`include/strix/handoff.hpp` and `src/handoff.cpp` implement framed TCP sessions,
1..64 streams, the project-owned bp-v1 codec, per-piece XXH64 and manifest checks.
`include/strix/handoff_policy.hpp` provides the shared resolver and an estimated cost model.

## Implementation and qualification

- `STRIX_HANDOFF_RECV_WAITALL=0` restores the previous receive organization. Both paths
  advance by actual bytes received, reject EOF/errors and preserve frame validation.
- `SenderOptions::encode_batch` defaults to 16. A batch also needs space in
  `max_queued_wire_bytes`: reservations cover conservative encoding workspace and unsent
  output buffers before encoding starts. A single oversized piece may run alone for
  progress. Worker stacks, allocator bookkeeping and caller input storage are separate;
  `max_pending_plain_bytes` controls input credits. Identity retains caller storage.
- The host tests compare complete decoded payloads and per-sequence encoded frame bytes,
  force a multi-piece batch, and check bounded/oversized reservations. These establish
  transport invariants, not inference fidelity or a throughput gain.
- The copy probe brackets the kernel on its own stream with device events, measures copy
  events separately, and reports the intersection of their intervals. An interval overlap
  can include gaps between copies and does not establish continuous hardware concurrency.
  Device execution must be remeasured; previous host-observed kernel times do not establish
  interference or its absence. `copy_wall_s` includes host submission and synchronization.

## Adoption sequence

The probe exercises the shared TCP Sender and receiver. Qwen currently adopts only the
resolver: its HTTP packer and reply-waiting POST workers remain in the family adapter.
Shared TCP probe results do not qualify end-to-end Qwen service behavior or other families.
Next, move reusable packing/posting orchestration into shared transport code, adapt Qwen,
then migrate the other family senders while retaining only layout/numerical hooks. Each
adoption needs complete state/token equality and complete-workload timing before promotion.
That migration is outside this local correction.

## Shared resolver (`strix/handoff_policy.hpp`)

`resolve(family_defaults, request)`: CLI > `STRIX_HANDOFF_*` > family defaults. Each read is recorded for
`--print-effective-config`, and each value gets a line `handoff <key> <value> (default|env|cli|auto: ...)`.

| Switch | Values |
|---|---|
| `STRIX_HANDOFF_LINK` / `--handoff-link` | `1gbe`, `2.5gbe`, `10gbe`, `usb4`, `oculink`, `loopback`, or goodput in MB/s |
| `STRIX_HANDOFF_CODEC` | `identity`, `bp-v1`, `auto` (needs a declared link) |
| `STRIX_HANDOFF_STREAMS` | `1..64`, `auto` |
| `STRIX_HANDOFF_ENCODE_THREADS` | `0..256`, `auto` |
| `STRIX_HANDOFF_ENCODE_BATCH` | `1..256`, `auto` |

`auto` runs `predict()` over candidates constrained by every explicit or default non-auto setting,
including batch policy. The diagnostic predicts the final applied choice. It picks the cheapest choice within 1 % of the
earliest end: identity before bp-v1, then fewer streams, then fewer threads. The defaults stay what each family ships.
The Qwen prefill service resolves `--handoff-codec/-connections/-encode-threads/-link` through it (connections
`auto` is refused after CLI/environment precedence: an HTTP chunk POST waits for its reply).
The HTTP adapter also refuses encode-batch overrides, which it does not execute.

## Model limits

The model estimates wire(piece) as per-message cost plus wire bytes divided by
min(link goodput, streams times per-stream cap). Encoding is plain bytes divided by a
thread-limited codec rate, followed by a wire stage in submission order and a fixed finish
allowance. The default codec constants originate from Qwen state samples; they are not
universal family or device properties. Batched estimates assume enough independent ready
pieces, omit receiver decoding, and include nominal/non-TCP link classes. Calibrate codec
ratios, encoder/decoder rates, link behavior and producer schedules on the intended host
and workload before considering production promotion. An undeclared workload uses a
steady 1 GiB BF16 estimate; predictions and theoretical throughput are not qualification.
Family defaults stay in force unless explicitly overridden.

Kit: `scripts/bench/xfer/`. Store comparative results and source inputs outside Git;
public records should describe project contracts and independently qualified local behavior.
