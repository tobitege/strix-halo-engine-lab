# hetpf: DeepSeek V4.1 CUDA prefill -> pair decode hand-off

One measurement of the mixed route against both devices alone, with the same prompt and the same request shape.

| arm | route | TTFT |
|---|---|---|
| A | `strix-cuda-ds41-prefill-service` prefills, exports the attention state and POSTs it to rank 0's `/admin/import-state` (rank 0 relays it to rank 1, both import); a streamed `/v1/completions` on the pair continues it | service wall (prefill, export, sha256, transfer, import) + pair first nonempty text chunk |
| B | the pair prefills the same ids itself (`exact` or `fast` = `STRIX_DS41_FAST_PREFILL=1`, per serve process) | first nonempty streamed text chunk |
| R | the CUDA service alone (its `next_token` is the first token) | measured response availability (includes export) |

Every request asks for 65 tokens (the first + 64 decode steps). Each serve process runs at least two requests of an arm:
attempt 1 is reported as cold, the last valid later attempt as warm. Failed first attempts stay unavailable. Arms alternate inside a process (`A,B,A,B`, then `B,A,B,A`).

Files:
- `link_probe.py`: `serve` on the node, `probe` on the host: application-level RTT (1-byte TCP ping-pong) and useful
  upload/download throughput; `--state-bytes` predicts the state's transfer time.
- `pair_serve.sh OUT MODE HOLD`: one pair serve process inside a pair GPU ticket on rank 0 (rank 1 over ssh); writes
  `OUT/ready`, stops on `OUT/stop` or after HOLD s; samples GPU busy and sclk on both nodes.
- `het_ab.py`: the client (arms A/B/R), one JSON line per request; reuses the existing `Server` transport and ID loader, with strict terminal stream validation.
- `run-pc.sh OUT`: the whole run from the CUDA host (link probe, service, R, one ticket per `TICKETS` entry, summary).
  Hosts, paths, the ticket command and the device come from the environment (see its header); nothing is hard-coded.
- `summarize.py`: `OUT/summary.txt`: observed latency cold/warm per arm, A's phase breakdown, text chunks/s, and sampled
  busy-clock integrals per prompt token (prefill/to first text) or per text chunk (decode), continuation prefixes against the
  first exact B, state sha256 and next token.

Gate (declared before measuring): the pair checks the state's sha256 on import and rank 1 checks its digest; the A
continuation must start with the service's `next_token` and be taken without a pair prefill (`continued`); continuations
are compared with the pair's own exact and FAST ones. The CUDA exact prefill is not bitwise equal to the pair's, so A is
a mixed route: equality with the pair is never required, a KL/dNLL check against the pair exact reference is.

Verdict: **PROVISIONAL**, including when every implemented gate passes. KL/dNLL is not implemented;
continuation prefix agreement alone cannot qualify the mixed numerical route. Missing terminal
IDs/usage, a failed import receipt, wrong next token or nonzero computed prompt invalidates a
sample and fails the campaign. Import accounting is matched to the completion's response ID.
Text chunk latency/throughput is explicitly not token latency/throughput: UTF-8 buffering and
invisible tokens can change chunk boundaries. Generated IDs supply counts only.

Use a new OUT directory for every campaign (existing directories are rejected). A random run ID
names fresh remote ticket directories and is recorded in request/link records. Summaries reject
mixed campaign files. `.done` means successful implemented gates; `.failed` and `status` preserve
failure/cancellation. Cleanup stops owned process groups and signals an active or queued ticket;
monitors, ranks, the service and probe also have absolute timeout backstops. Unreachable remote
hosts can delay cleanup until those backstops; SIGKILL/power loss cannot execute shell traps.
The probe requires the intended interface via `--bind`; it permits four workers, 10,000 pings
or 256 MiB per operation, 30 seconds per connection and 180 seconds total by default.

Clocks are estimates requiring synchronized host/node wall clocks; offsets are not measured by
this kit. Each interval integrates busy times MHz from the preceding sample, clips at window
edges, and rejects missing coverage or gaps above 1.5 seconds. Both pair nodes are required.
"Cold" means the first attempt of that arm in the pair process; A's CUDA service has already run R.
The wrapper records running kernels, `amdgpu.iommu_perfopt` (or unavailable), both ranks' effective
configs and the service arguments alongside private receipts. Do not treat unavailable metadata
or unsynchronized sampling as qualification evidence. The wrapper is a DS41/Trellis experiment
adapter with explicit sampling/device assumptions; extract reusable lifecycle/sampling helpers
and add explicit device selection before extending it to another engine.

Offline tests of the kit: `python3 -B scripts/bench/hetpf/test_hetpf.py`.
