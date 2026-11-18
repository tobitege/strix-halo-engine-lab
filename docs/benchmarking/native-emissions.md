# Native committed-token peak capture

`scripts/bench/common/native-emissions.py` transparently forwards a native program's binary stdin and
stdout. Supply full-line regular expressions for its documented committed-token protocol, each with
a named decimal `token` group and optional `slot` group. Draft, proposal, and diagnostic lines must
not match. The proxy records each matching ID and a same-host `perf_counter_ns` timestamp immediately
when its stdout line arrives. It does not tokenize text or infer counts from arbitrary output chunks.

Run the native command through `run --out TRACE --token-pattern REGEX -- COMMAND ...`. Point an owned
server's native command declaration at this proxy; keep executable and output paths in runtime config.
The output path may contain literal `{pid}` and `{time_ns}` placeholders, expanded by each proxy
process before opening its new trace (for example, `native-{pid}-{time_ns}.jsonl`). Other text is
unchanged. This lets an immutable server config start repeated native processes without reusing files.
After the client burst and owned cleanup, run `peak --trace TRACE --receipt CLIENT_JSON --out PEAK_JSON`.
The client receipt must contain the complete `events` from `tools/serve_bench.py`, including final usage
and `[DONE]` arrival (`finished`). Client and proxy must run on the same host and Python monotonic clock.

The collector pools only IDs in the burst's arrival-to-finished interval. Their count must exactly equal
summed completion usage, including any stop tokens counted there. A missing ID, extra drained token,
unmatched protocol, incomplete receipt, or span below one second yields an unavailable peak. Inspect
the recorded reason rather than substituting an estimate. Only declared matching lines contribute.

Trace writes are append-only and unbuffered so forced owned-process cleanup preserves the events.
On Windows the proxy owns its native child tree through a kill-on-close Job object; cleanup cannot
leave the native process behind. The extra stdout hop and per-token trace writes have host overhead:
use this only for a separately labelled instrumented peak run, never its throughput as the plain arm.

Home: shared (host shared measurement proxy and count/clock gate, independent of model and backend).
