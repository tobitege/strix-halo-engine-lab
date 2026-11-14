# Qwen HIP serial HTTP request ownership

The serial `make_http_serve` caller constructs `serve::SerialSessionOwner` before
the HTTP service listens. Its entry is `SessionRuntime::run()`. Each control turn
reserves one FIFO request, acquires a logical generation, calls the existing
`MtpBridgeEngine::generate`, publishes through the runtime, and completes the
ticket with the bridge's complete native statistics. There is no additional
request loop or scalar model decode implementation in the adapter.

Logical leases never address or reset model slots. The adapter advertises only
owner control and one logical request. Generic prefill, decode, state access,
restore, fork and append are unsupported. Generic retention is disabled before
cache construction. Existing native prefix/session owners, identities, budgets,
lazy allocation, defaults and OFF controls remain responsible for model state.

The opt-in serial queue protocol retains cancelled FIFO entries until owner
completion. Idle control wakes yield back to the common entry and a subsequent
request wakes the same queue. Native exceptions remain request outcomes: request
errors retain their full statistics, standard errors retain their exact message
(including an empty message), and unknown errors use `engine failure`. The bridge
retains its failure latch; the existing watcher closes HTTP and queue admission.
Owner control does not add an early failure stop or drain policy.

Completion uses the common taken-job mechanics with an explicit serial timing
option. Queue time starts at FIFO removal, before logical activation. Only a
still-observed request adds queue time to the engine's entry-relative first-token
measurement. Expired periods retain no partial observation. Cancellation,
constraint cancellation receipts, totals and histograms use existing rules.
Escaping idle/observer callbacks preserve caller-owned queue lifetime. An observer
still runs after totals and before ticket completion, so its exception leaves
that ticket unfinished, as the legacy serial entry does.

The host replay compiles the unchanged bridge and serial binding extracted from
production source with pinned hashes and unique anchors. Its independent reference
was captured from the actual pre-adoption queue implementation. It exercises real
queue and ApiService delivery with explicit native ABI doubles. It checks complete
native receipts, cancellation/errors, timing attribution and observation expiry,
logical generation reuse, session control forwarding, probe control-only evidence,
HTTP completion, SSE and disconnect behavior. These tests do not qualify device
arithmetic, native capacity, scheduling or performance. See the
[host test contract](../../tests/serial_session_owner/README.md).

The batched adopter retains its existing binding and protocol. Qwen factory/CUDA
and cache-policy consolidation, other family migrations, and physical fork/COW
remain outstanding. Native qualification requires separate strict shared/legacy
ON/OFF builds, complete IDs/logits/state/frontier/RNG/receipts/capacity comparisons,
and interleaved complete-workload TTFT and throughput checks on every applicable
rank. Host doubles establish neither portability nor a no-slowdown result.
