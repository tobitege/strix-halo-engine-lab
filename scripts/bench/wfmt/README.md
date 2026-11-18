# Weight-format exactness kit

This shared kit preserves distinct storage, image and arithmetic identities. It
does not select faster arithmetic or supply model assets. Build and receipt
directories are caller inputs and must remain outside version control.

HIP production kernels exercise the extracted storage helpers. `BoundCore`,
generic `load_fragment` entry points and the source admission registry remain
scaffolding covered by fixtures, not production launch policy. Existing row
construction adapters remain in place; consolidating them and connecting the
typed facade requires a separate migration and native qualification.

Invoke `build.sh` through the deployment's CPU build lane, passing `--source`,
`--build`, `--backend` and optionally `--toolchain` and repeatable `--target`.
Backends are `host`, `hip`, `cuda-qwen`, `cuda-ds41`, `cpu`, `intel` and `nvfp4`.
The script only configures/builds; it never acquires a GPU ticket. A target not
introduced yet fails explicitly. Configure/build output goes to log files, and
`wfmt-summary.txt` records status and source identity. Preserve compiler flags,
compiler version, architecture, build wall time and object size with receipts.

Run `compare.py --self-test` to exercise changed-byte, changed-length and stale
hash rejection. `compare.py --fixture BASELINE` writes a deterministic protocol
fixture. The host test executable's `--wfmt-capture CANDIDATE --wfmt-matrix MATRIX`
mode writes the same fixture. Comparing them validates the protocol, **not** a
production kernel. P01 changes no production operation; later migration PRs add
test-only adapters before replacing numerical bodies.

For actual adapters, run `capture.py --exe EXECUTABLE --matrix MATRIX --out DIR
--compiler VERSION --architecture ARCH --source SOURCE --flags FLAGS --device ID
--adapter reference|native|backend`. Use `--source SOURCE --source-ref COMMIT` with an archived source tree; COMMIT must be a full 40- or 64-hex revision declared by the archive producer (this does not verify archive contents).
Source commit and exact compiler flags are required. `--adapter protocol` is explicitly fixture-only.
The reference adapter uses `--wfmt-reference-capture DIR`; backend fixtures use
`--wfmt-backend-capture DIR`; native hooks use `--wfmt-capture DIR --wfmt-matrix MATRIX`.
A protocol fixture is rejected for numerical adapters. Each adapter must emit
nonempty `.wfmt` records matching the supplied matrix or capture fails. Capture stores executable/matrix SHA256,
source commit, flags, compiler, architecture and device. Frozen binaries and all
raw payloads stay outside Git. `--runner` is an optional shell-style argv prefix;
it is parsed without invoking a shell. Devices and asset paths are supplied by
the caller. Use an identical compiler, flags, architecture and device for both
arms, and the immediately preceding source commit as the legacy arm.

The matrix is a `wfmt.matrix@1` JSON object with a nonempty `cases` array. Each
case declares `file` (relative `.wfmt` path), `case`, `arithmetic`, `shape`, optional
`format`/`image`, and `required_sections` mapping section names to positive byte
lengths. `optional_sections` maps optional names to exact lengths (zero allowed).
Undeclared files/sections, missing cases/required sections, changed metadata and
wrong lengths fail. See [protocol-matrix.json](../../../tests/wfmt/protocol-matrix.json)
for a protocol-only example. A record with zero total payload bytes always fails.
Version 2 capture receipts retain `matrix.json`, every record's SHA256 and adapter
arguments. Failed adapters retain evidence but never receive a successful receipt.

`compare.py --baseline OLD --candidate NEW --require-bytes --out RECEIPTS` checks
the full record inventory, semantic metadata, section lengths and every byte.
FNV-1a64 is a receipt only, with offset 14695981039346656037 and prime
1099511628211 modulo 2^64. Equal hashes never replace direct equality.

Version 1 records contain eight magic bytes `WFMT 00 01 0d 0a`, a little-endian
64-bit JSON-header length, a UTF-8 JSON object, then section bytes in header order.
Required metadata are `case`, `arithmetic` (name@revision), `shape` and `sections`.
Each section has `name`, `bytes`, and a 16-digit hexadecimal `fnv1a64`. Names are
`source`, `sidecars`, `activation_codes`, `activation_scales`, `prepared`,
`partials`, `output`, `logits` and `state`. Exclude unwritten/padding bytes or
initialise them identically. Preserve signed zeros and defined NaN payloads.

`run.py --build BUILD --baseline OLD --candidate NEW --backend BACKEND --pr P01
--output RECEIPTS [--runner RUNNER --partition-tests]` runs the selected mandatory CTest group,
requires every named test exactly once with successful JUnit status, then compares
captures. P01 requires contracts, census, compare and tools; P02 requires reference,
geometry and iq3s; P03 readers and admission; P04 resolve, cache and effective-config;
P05 cpu (all names prefixed `wfmt.`). P02 additionally requires the DS41 formats,
native-fp8 and trellis-identity host tests for host/hip/cuda-ds41, and both NVFP4
host tests for host/nvfp4. P05 additionally requires `wfmt.intel` and the Intel
backend-host-tests, dense-boundaries and memory-boundaries tests for host/intel.
P06/P07/P08 require the HIP backend, their `wfmt.hip-dense`/`wfmt.hip-gsq`/
`wfmt.hip-trellis` cases, and every corresponding native suite listed in `run.py`.
Unimplemented or unregistered required tests fail; backend names are validated.
The runner requires successful version 2 receipts for both arms, verifies their
matrix/inventory/content hashes, and requires matching compiler, flags, architecture,
device, platform, matrix and adapter arguments. Executable/source identities may
differ between migration arms. `--protocol-fixtures-only` explicitly bypasses capture
receipts and labels the result `protocol-fixtures`, never qualification; use it for
P01 protocol examples. Direct `compare.py` results are labelled `protocol-comparison`.
Invoke this in a new shell
after the build lane exits. Partition cases so each GPU ticket fits 120 seconds;
the runner's timeout cannot be increased above that limit. Pass additional
family gates separately until they are registered in the binding census. No
missing fixture is silently treated as a passing gate.

Queued runners require `--partition-tests`: each mandatory CTest receives its own
sequential ticket, with CTest's timeout applied inside that ticket. FIFO queue time
does not consume the execution timeout. The deployment runner must also cap the
whole ticket at 120 seconds (for example, `GPU_TURN_MAX=120`); use its quiet timing
label convention. Each partition's JUnit inventory is checked before aggregating
the complete mandatory inventory into `ctest.xml`.

For native transfer captures, execute `capture.py --adapter hip-transfer` inside
an already-held bounded GPU ticket. Its timeout then covers execution only.
Compile the test with `STRIX_WFMT_CAPTURE=1` and include
`hip_transfer_capture.hpp` after HIP/test headers. Set each retained route's
arithmetic, logical shape and storage ID with `capture::set_context`; use
`capture::emit` for named activation, prepared and partial buffers. Successful
H2D/D2H copies provide additional source/output records without changing payloads.
Run the predecessor once to produce raw records, freeze their inventory using
`matrix_from_records.py --baseline RAW --out MATRIX`, then capture both arms
against that same immutable matrix. `capture.py --source-revision FULL_COMMIT_ID`
records frozen archive provenance when the source has no Git directory.

`isa.py --old OLD --new NEW --backend hip|cuda --symbols SYMBOLS.json --out DIR`
disassembles required symbols and retains raw and address-normalised streams.
`--tool` selects the installed disassembler. HIP also uses `--readelf` (llvm-readelf),
`--objcopy` (llvm-objcopy) and `--bundler` (clang-offload-bundler). Direct AMDGPU ELF64
images and hosts with `.hip_fatbin` are supported; other containers fail closed.
`--architecture` defaults to gfx1151. Exactly one matching bundled image is required.
Selected kernels must have AMDGPU notes for LDS, scratch, VGPRs, SGPRs, kernargs,
wavefront and workgroup limits, plus resource-bearing `.kd` descriptor bytes.
The descriptor's code-address offset is excluded; resource registers are compared.
Missing notes/descriptors fail. Native extraction still requires the target LLVM tools.
The manifest is a nonempty JSON array
of required symbol substrings. Any changed instruction/resource stream fails
for review; byte equality alone cannot whitelist codegen changes. Review spills,
registers, shared/LDS bytes and occupancy, and retain timing receipts plus a
written explanation. No kernel is changed by P01.

For a required source trait that changes mangled names, `--symbol-map PAIRS.json`
accepts explicit `[{"old":"FULL_OLD_SYMBOL","new":"FULL_NEW_SYMBOL"}]` pairs.
Both sides must be one-to-one and present in the selected functions. Only those
names and function placement are normalised: instruction order within each
function, every unmapped function, and every resource/descriptor value remain
mandatory. The receipt retains the pairs; missing, duplicate or extra functions
and changed instructions/resources cannot be excused by the map.

Whole-model performance qualification supplies model/prompt paths externally:
two prefills per process (first cold, last warm), ABAB at least three pairs,
clocks, busy fraction, time, tokens and clocks/token alongside throughput. Such
long cases must be partitioned into bounded tickets. Capture hooks are test-only;
they copy existing buffers rather than recomputing arithmetic.

The legacy wrapper gate injects one test-only adapter into both the preceding
source revision and the candidate. It captures accepted dtype, shape, byte range
and payload through the existing DS41, H3, Gemma, vision and (on POSIX) GLM and
MiniMax APIs. The same record writer and synthetic fixtures run in both binaries;
no old parser implementation is copied into the kit. Run inside the CPU build
lane with both source refs and revisions supplied:

```sh
python3 scripts/bench/wfmt/legacy_compare.py --baseline-source "$OLD_SOURCE" \
  --candidate-source "$NEW_SOURCE" --baseline-revision "$OLD_REF" \
  --candidate-revision "$NEW_REF" --build-root "$BUILD_ROOT" \
  --output "$RECEIPTS" --verify-session
```

The optional session check records the preceding revision's known aliased-shard
failure and requires the candidate to pass. Generated fixtures, binaries, logs,
provenance and byte receipts stay outside Git. The aliased-shard loader keeps all
asset-receipt path references while indexing each opened physical file once.

Both source arguments must be clean checkout roots whose HEAD is the requested
revision. The driver verifies both commits and tree IDs before configuring either
build, rejecting tracked edits, untracked files and dirty submodules. Keep build
and receipt directories outside the source checkouts (or ignored).
