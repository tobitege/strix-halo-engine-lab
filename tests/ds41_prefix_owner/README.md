# DS41 pooled prefix owner fixture

This host fixture compiles the actual DS41 prefix facade and attention-state
source with the existing `strix_state_cache` owner/IO library. Root registration
uses the same minimal closure, without adding serve or accelerator dependencies.
Existing DS41 prefix and durable IO tests remain separate and unmodified.

```sh
cmake -S tests/ds41_prefix_owner -B build/ds41-prefix-owner-r15/standalone
cmake --build build/ds41-prefix-owner-r15/standalone --config Debug --target ds41_prefix_owner_tests
ctest --test-dir build/ds41-prefix-owner-r15/standalone -C Debug -R '^ds41_prefix_owner(_checks)?$' --no-tests=error --output-on-failure --timeout 90 --output-junit "$PWD/build/ds41-prefix-owner-r15/debug.xml"
```

Repeat with Release. Checks throw explicitly under `NDEBUG`; the sentinel test
requires exit 1, empty stdout and the exact deliberately failing diagnostic.
Root host builds use HIP/CUDA OFF. Explicit non-strict host builds remain
distinct from the product's default strict warning policy.

The trace covers branching ancestry, multiple snapshots, unknown-next/minimum
row boundaries, delta admission/refusal, duplicates, load/use versus lookup,
pressure, unlimited and tiny budgets, clear and cumulative RAM statistics.
Independent native-shaped row/index/ring/carry bytes cover different ratios and
ownership layouts. Allocation counts, shared ancestor addresses, the actual
common directory, outstanding read leases, charge release, partial allocation
failures and exact LRU victims prevent a silently bypassed shared owner from
passing. Two facades share a service store with separate policy scopes.
Paired traces use independent rank images and compare admission, frontiers and
RAM statistics while checking each rank's complete payload bytes. A failed delta
capture also exercises pin release after the selected ancestor snapshot evicts.

For entry equivalence, archive and compile unmodified sources before migration,
freeze the trace executable and byte captures, then compare the adopted trace
(`--trace`) without refreshing expectations. Durable regression uses the actual
existing IO fixture: compare all 16 filenames and raw bytes, and cross-read
copies with original and adopted executables. Readers mutate retain/forget
state, so frozen directories must remain read-only inputs to copying.
Generated source archives, scripts, manifests and reports belong only under
`build/ds41-prefix-owner-r15/`.

The guide describes [ownership and remaining consolidation](../../docs/engine/session-ds41-prefix-owner-adoption.md).
Native strict compile/link, both-rank complete state/frontier/RNG/receipt
comparisons, physical detach/rollback/fork isolation, unchanged capacity and
interleaved whole-workload latency/throughput remain operator gates. Host states
establish no native fidelity or performance claim.
