# DS41 prefix durable IO host tests

This fixture compiles the actual DS41 prefix source and attention-state code and
links the existing common IO library. A test-only callback enters the same Writer
thread before/after its ordinary jobs. Faults are installed on that worker's TLS;
caller-thread faults are separately checked for isolation. Ordinary production
builds have no callback or extra writer state. No codec, cache or IO double is
substituted.

Checks throw explicitly and remain active with `NDEBUG`. The second test executes
a deliberate failure and requires exit 1 plus the exact sentinel diagnostic.
The fixture covers complete synthetic KV/index/ring/carry bytes, consumed token
history and next-token frontiers, shared/branch blocks, known/unknown whole and
partial boundaries, restart, inventory, load/use/forget/retain, shared IO activity,
lazy/refused writes, budgets, active payload charging and write/delete ordering.
Publication faults cover allocation/create/write/sync/replace/directory-sync,
partial failures and successful short writes, last-good visibility and retry.
Cleanup observation closes enumeration handles and has a one-second bound for
transient host rename/delete aliases; a retained temporary must still fail.
Read/allocation failures and corrupt/truncated/missing files are safe misses.
Both DS41 page-cache settings execute; POSIX-only capabilities are named as
unavailable on other hosts, not passed by simulation.

Standalone (use an installed host compiler):

```sh
cmake -S tests/ds41_prefix_checkpoint_io -B build/ds41-prefix-io-r14/standalone
cmake --build build/ds41-prefix-io-r14/standalone --config Debug --target ds41_prefix_checkpoint_io_tests
ctest --test-dir build/ds41-prefix-io-r14/standalone -C Debug -R '^ds41_prefix_checkpoint_io(_checks)?$' --no-tests=error --output-on-failure --timeout 120
```

Repeat with Release. The root host build registers the same two tests and keeps
`strix-deepseek41-prefix-cache-tests` as its production runtime regression. Use
HIP/CUDA OFF for host-only builds. Root warning policy stays strict by default;
an explicitly non-strict host arm is separate evidence, not strict qualification.

The executable also accepts `write NEW_DIRECTORY` and `read COPIED_DIRECTORY`.
The reader's retain/forget checks deliberately mutate its input, so use copies.
For independent equality, freeze actual old sources, executable and captured
files before production edits. Compare every filename, raw byte and hash from
the adopted writer, then cross-read copied old/new fixtures. Running the adopted
writer twice does not establish entry equivalence. Generated evidence belongs
under `build/ds41-prefix-io-r14/`; frozen references are never refreshed.

Synthetic host states test storage semantics, not native arithmetic, direct-IO
residency, physical model restore, transport, capacity or speed. Those require
separate Linux and both-rank operator gates. See the
[adoption boundary](../../docs/engine/checkpoint-ds41-prefix-adoption.md).
