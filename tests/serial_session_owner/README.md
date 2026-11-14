# Serial native-control owner replay

This host project runs the actual shared serial owner, SessionRuntime, queue and
ApiService. The production bridge, session route handlers and serial construction
binding are extracted as exact byte slices with unique anchors and pinned hashes.
Native ABI and profile types are explicit host doubles. No device backend or
unadopted generic resource backend is enabled.

```sh
cmake -S tests/serial_session_owner -B build/serial-host
cmake --build build/serial-host --config Release --target serial_session_owner_tests
ctest --test-dir build/serial-host -C Release -R '^serial-session-owner.*$' --no-tests=error --output-on-failure
```

The central session module registers the same targets in the root host build.
`STRIX_SERIAL_OWNER_REFERENCE` optionally selects an immutable pre-adoption capture;
its hash is checked before registering the comparison. The reference executable
uses archived actual `GenerationQueue::run` and bridge code, not a handwritten
legacy loop. Extraction and capture happen before production edits; neither is
regenerated to accommodate a mismatch.

Replay compares token IDs, exact errors, all applicable statistics, complete native
receipt fields, observer order, totals, histogram buckets/counts and decode sums.
Queue/busy times are volatile and checked for valid boundaries. First-token time
must equal the fixed native measurement plus queue time only while observation
remains complete; real observe expiry is tested separately. HTTP normalization
changes only top-level response `id`, `created`, and `strix.queue_ms`, with type
and range checks. It never recursively strips receipts or field names.

Independent checks cover stale logical generations, release/reuse, cache absence,
idle wakes, observer exceptions, native constraints/transforms/thinking receipts,
session control forwarding, probe failure/control-only evidence and API refusals.
Release assertions are always active; a separate sentinel must exit with the
specific failing assertion diagnostic. A host type double cannot establish native
numerical, physical state, capacity, scheduling or whole-workload speed equality.
