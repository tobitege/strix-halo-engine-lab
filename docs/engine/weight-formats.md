# Weight formats and numerical contracts

The modular weight layer keeps a tensor's storage interpretation separate from
its operand image, backend and arithmetic identity. A container parser recognising
a type does not make that type executable. The initial implementation preserves
all existing production choices and numerical contracts.

The [binding census](../../tests/wfmt/bindings.json) records the audited source
revision, every recognised GGUF geometry, typed-only and parse-only entries,
legacy consumer sites and the owned migration work items P02–P14. Safetensors
recognition is recorded separately from model admission. Derived CB4/U5/U6,
Q6P32 and prepared operands are images; they are not additional GGUF wire types.
MXFP4 and native NVFP4 remain distinct storage contracts.

`wfmt.census` checks the reader's complete registry-backed geometry map and inventories
format-related operations in each declared source file. Adding or removing an
operation requires reviewing its binding and migration assignment. The keyword
scan is a closure guard for the declared files, not a substitute for reviewing
new consumer files. Include each new consumer in both the binding entries and
`closure_files`. Architecture scaffolds remain unavailable until implemented and
qualified; a source appearing in the census is not a qualification receipt.
DS41 decoder declarations and both CUDA Qwen geometry/decoder switches are also
checked against per-site storage lists. Derived representations use `images`,
separately from `formats`; Engram denotes custom E4M3 codes/E8M0 group-scale storage.
Other sites still require manual per-operation review: a name scan also sees calls
and comments and does not establish exhaustive numerical-variant coverage.

Arithmetic identities remain separate for CPU integer products/tree32, Intel
F16 operands/ascending reduction, HIP staged F16 products, CUDA four-chain IMMA,
native IQ FP32, DS41 group products and affine correction, transformed trellis,
Gemma EXL3 variants, H3 transformed operands and tolerance-qualified NVFP4 MMA.
An existing public identity is preserved; a new registry name does not change
its arithmetic or qualification status.
The current descriptors are an inventory, not complete arithmetic specifications.
Before migration, freeze unspecified rounding, reduction, tiny-input, accumulation
and transform boundaries and qualify them with operation-specific byte captures.
Registry/source entrypoints are declarations at this stage, not a usable weight API.
Borrowed strings/spans require stable backing storage that outlives every copy;
strings remain immutable, while spans permit mutation according to their types.
Dynamic registries must own stable storage for them.

Use the [weight-format comparison kit](../../scripts/bench/wfmt/README.md) to
build host cases and collect versioned byte records before and after each
migration. Receipts identify the exact source, compiler, flags, architecture,
shape, source bytes and arithmetic. Comparison requires equal lengths and byte
contents; hashes alone are insufficient. Intermediate hooks are test-only and
introduced with the corresponding migration. Store generated captures, binaries,
logs and private model inputs outside Git. Host-only census/contract changes do
not claim device execution or new numerical qualification.

Home: shared (host shared contracts and test infrastructure). The shared layer
serves all model families; vendor operand sources and instruction specialisations
are introduced in their assigned migration steps.

P03 introduces owning positioned sources and a common GGUF/safetensors index.
File sources retain their read handle, check size/change snapshots, and compute a
full SHA-256 content identity lazily. A tensor ID includes its pack-relative shard namespace, bounded header digest
and name. Identical node copies retain the same ID; same-basename shards in
different pack subdirectories remain distinct. Slices keep sources alive after an index or loader is destroyed.
Every indexed tensor carries its full stored extents (`RawTensor::extents`, any rank). A safetensors
tensor whose rank exceeds the WeightView rank (four), such as a Conv3d vision patch embedding
`[1280,3,2,14,14]`, is indexed opaquely: name, dtype, extents and byte range are kept, it is never
bound by a family schema, and an explicit view of it fails for that tensor only, not for the file.
Engines that need it read it by byte range; legacy consumer tables receive the full extents. GGUF
stays limited to rank 1..4 because the container itself stores at most four dimensions.

Admission validates exact format allowlists, role masks, logical shapes, expert
axes, payload bounds and semantic sidecars before publishing a map. GGUF native
axis order is normalized to logical row-major shape: `[N,K]` for matrices.
Safetensors dtypes remain scalar storage descriptions until a verified schema
binds FP8 scale grids, trellis transforms or H3 channel/group quantization.
Malformed ranges, duplicate names, overflow, unsupported transforms and missing
sidecars fail with a structured status. No allocation failure changes arithmetic.

Existing model shape checks and execution/upload paths remain authoritative.
Gemma manifests, MiniMax pack readers, GLM indices and H3 readers expose admitted
maps; Qwen CUDA also records each legacy conversion recipe while retaining the
source format. Qwen and MiniMax declaration-only model description files have no
loaded tensors, so their actual weight loaders perform admission. GLM's legacy
fixed mul1 codebook is explicit even when its old container omits a marker.
The reader tests verify source lifetime, content identities, mutation rejection,
mixed GGUF formats and the existing DS41 positioned-read byte contract.

`validate_binding` reuses the same allowlist/shape/schema policy without reading
numerical payloads. Admission still performs the bounded value checks once;
subsequent resolver calls do not re-scan weights. The counting-source fixture
checks this distinction and a malformed FP8-code fixture verifies that full
admission keeps its value gate. A Linux host fixture also instantiates GLM's
production index and compares its legacy byte ranges with the shared map.

For a non-circular compatibility gate, the kit's legacy wrapper adapter compiles
unchanged against the immediately preceding revision and the candidate. It
records each public API's dtype, shape, offset, length, acceptance and raw bytes,
then uses the same versioned writer and byte comparator. The root host session
fixture also covers aliases: asset identity retains all path references, while
one opened physical shard contributes one tensor table.

Migration remains partial: wrappers retain legacy execution representations, and
MiniMax still parses shard headers for both its execution table and shared index.
The legacy capture fixtures cover small plain tensors and selected refusal paths;
they do not qualify conversion or transformed-pack compatibility, model fidelity,
complete logits/state equality, or timing. The compatibility driver requires clean
checkouts at the requested commits and records their verified commit/tree IDs
before configuring either build. Source-derived logical shapes are not independent
graph-shape validation.
