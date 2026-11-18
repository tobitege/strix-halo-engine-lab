# deepseek41 expert server and parts-host (strix/ds41-expert-rpc-v1)

Status: implemented (wire, server loop, CPU reference and HIP backends, probe and exactness check); the CUDA parts-host
side is `include/strix/cuda/ds41_parts_host*.hpp` / `src/cuda/models/ds41/ds41_parts_host*`. Numbers in this file are qualification evidence only where stated.

## Roles

- **parts-host**: the device that runs the dense parts of every layer (attention, mixing, norms, router, shared expert,
  head) and the routed experts it keeps resident ("hot" experts).
- **expert server**: one or two nodes that keep the other routed experts ("cold") and compute them on request. It holds
  no dense weights.

Per MoE layer of a window the parts-host:

1. runs the router; as soon as the selected ids and weights are known, sends one `EXPERTS` request with the rows that
   select a non-resident expert (their A8 MoE input, ids, weights and the mask of remote slots);
2. meanwhile computes the shared expert and its resident experts;
3. receives the answer and runs the combine.

## Exact combine for any residency

The one-device combine of a row is

```
y = (shared + low) + high
low  = fold over the row's slots with id <  split, ascending id, from 0: acc = acc + w * raw
high = fold over the row's slots with id >= split, likewise
```

with one FP32 product and one FP32 add per step. For every (half, row) the server answers:

- `FOLD`: the fold of the half's leading slots when they are all remote (the partial sum the parts-host continues;
  a prefix fold is exactly the first steps of the one-device fold);
- `RAW`: the unweighted raw vector of every other remote slot of the half (after the first resident one), in ascending
  id.

The parts-host seeds each half with its `FOLD` item (or 0) and folds the remaining slots with its own raw vectors or the
`RAW` items. The result is bit for bit the one-device combine, whatever experts are resident where; no new numeric
identity is needed. A fold of an interior run would not be exact (the parts-host's partial sum enters before it), which
is why later remote slots travel as raw vectors.

Both sides derive the item list from the request alone (`plan_response`), so answers carry no shape metadata.

Sizes (hidden 5120, top-k 6): a request is `48 + R * (52 + 5760)` bytes (R1: 5,860 B); an answer is
`72 + items * 20,480` bytes (one item: 20,552 B; two: 41,032 B).

## Wire

32-byte little-endian header: magic `DS4X`, version 1, type, request id, payload bytes, flags, FNV-32 check.

| type | direction | payload |
|---|---|---|
| HELLO | parts-host -> server | session token (32 B), model identity, geometry (layers, hidden, inter, experts, top-k, split), client and numeric identity strings |
| HELLO_ACK | server -> parts-host | status (0 ok, 1 token, 2 geometry, 3 model), geometry, residency (expert range, layer mask), world, max rows, engine, numeric identity |
| EXPERTS | parts-host -> server | layer, rows, top-k, flags; ids and weights [rows][top-k] (router order); remote mask [rows]; A8 codes [rows][hidden]; A8 deltas [rows][hidden/32] |
| RESULT | server -> parts-host | layer, items, hidden, timings (server, compute, relay ns); items x hidden FP32 |
| PING / PONG, CLOSE / ACK, ERROR | | |

Flags of `EXPERTS`: `1` null compute (zeros of the planned size: wire probes), `2` no fold (every remote slot as RAW).
Requests may be pipelined; answers come in order. TCP with `TCP_NODELAY`, gathered writes; an optional busy-poll mode
makes the socket non-blocking and spins on reads and writes.

## Server layouts

- **direct**: one endpoint per node; each serves its expert range (`--experts 0:192` and `--experts 192:384`) and the
  parts-host sends each node the rows and slots it keeps.
- **relay**: one endpoint for the pair. Rank 0 listens and forwards each request to rank 1 over the inter-node link
  (`--peer usb4auto`, or TCP for tests); rank 0 computes the low half's items, rank 1 the high half's. The answer is
  streamed: header and rank 0's items as soon as they are ready, then rank 1's.

```
strix-deepseek41-expert-server MODEL.gguf --listen HOST:PORT --token-file F [--experts A:B] [--layers LIST]
    [--backend hip|host] [--busy-poll] [--once] [--max-rows N]
strix-deepseek41-expert-server MODEL.gguf --relay-rank 0 --peer usb4auto --listen HOST:PORT --token-file F [--layers LIST]
strix-deepseek41-expert-server MODEL.gguf --relay-rank 1 --peer usb4auto [--layers LIST]
```

`READY {json}` on stdout when listening (rank 1: when following); `{"event":"STOPPED",...}` at the end;
`{"event":"POISONED","error":...}` on a fatal error.

The HIP backend loads only the routed-expert tensors of the listed layers for its range (192 experts x 40 layers =
76.4 GB per node), builds the routes of each request on the host (experts in ascending id, chunks of at most 8 entries),
runs the decode gate/up (IQ2_XXS + SwiGLU + A8) and down (Q2_K) kernels, packs the FOLD/RAW items on the device with
strict FP, and copies them out: one upload and one download per request.

## Latency notes (measured on the lab pair, 1 GbE to the parts-host)

- Run the server with `--busy-poll`: a blocking server pays the wake-up of its idle CPU (about 0.7 ms after a 12 ms
  gap); busy-poll keeps the R1 round trip (5.9 KB request, 20.5 KB answer) at 0.34-0.37 ms p50 for any gap.
- `--keep-warm-us 500`: the GPU drops to an idle state after a few ms without work (one expert: 77 us hot, ~300 us
  after 3-12 ms idle); the keep-warm kernel holds it at ~125 us.
- direct beats relay for decode (relay adds the inter-node hop and the follower's wake-up).

## Tools

- `strix-ds41-xs-check`: sends windows (recorded routes or random ids, random A8 inputs, random remote masks) to one or
  two endpoints and compares every answer item bit for bit with the CPU reference backend, and the parts-host combine
  of the answers with the one-device combine.
- `strix-ds41-xs-probe`: round trips with the decode sizes (R rows, m remote experts per row, FOLD or RAW answers),
  null or real compute, pauses between requests, optional busy-poll; prints and writes percentiles. Portable (Linux and
  Windows).
- `strix-ds41-expert-rpc-tests`: framing, messages, plans, the split combine against the one-device combine over random
  residencies, direct and relay over sockets; with `STRIX_DS41_EXPERTS_GGUF` and `STRIX_DS41_LAYER`, the CPU reference
  backend against `reference_window` on a real layer.
