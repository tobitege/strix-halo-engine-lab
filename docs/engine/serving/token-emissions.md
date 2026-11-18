# Committed-token publication trace

For a diagnostic request, set `"return_token_emissions": true` in the request body. With response
receipts enabled, `strix.token_emissions` contains `clock: "steady_ns"`, `includes_stop_tokens: true`,
and `events: [[timestamp_ns, committed_token_count], ...]`. The timestamp is taken when the shared
generation ticket publishes a nonempty group. Proposed and rejected draft tokens are absent. Stop
tokens are included when the engine publishes them, matching the completion usage convention.

The trace is off by default and requests neither logits nor scores. It preserves the existing token
publication, presentation and timed completion paths. Its clock describes server publications rather
than network arrivals; do not merge it with client arrival timestamps or with another host's clock.

`tools/serve_bench.py --body-extra '{"return_token_emissions":true}'` uses a complete trace for the
one-second peak, while retaining client arrivals for TTFT and elapsed request throughput. Counts must
equal final completion usage and timestamps must be monotonic. An invalid trace, or a burst mixing
traced and untraced requests, has an unavailable peak. Peaks require at least one second of emissions.
Receipts retain exact integer timestamps. Label trace-enabled peaks separately from uninstrumented
throughput measurements, because recording and returning the diagnostic has host overhead.

Home: shared (host shared: generation ticket, response receipt and benchmark client). Every engine
that publishes committed tokens through the shared generation queue uses the same implementation.
