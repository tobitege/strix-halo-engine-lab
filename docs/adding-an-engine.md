# Adding an engine

*Español: [adding-an-engine.es.md](adding-an-engine.es.md)*

A model family is added by writing what is specific to the model and adapting it to
the shared layers. Serving, sessions, caches and scheduling are not written again.
This page lists what a family implements, what it receives, and how backends are
kept apart. A complete, compiling example is in `examples/skeleton_engine/`.

Background: [engine/session-layer.md](engine/session-layer.md) (contracts),
[engine/backends.md](engine/backends.md) (Backend seam),
[models/model-families.md](models/model-families.md) (family layout rules).

## The example

`examples/skeleton_engine/skeleton_engine.hpp` is a family in one file. Its "model"
is a counter, so the file contains only the adapter code that every family needs.
It is built as `strix-skeleton-serve` and tested by `skeleton_engine`
(`tests/skeleton_engine_tests.cpp`), which sends a completion and a chat request
through the production `strix::serve::run()` path with the CPU reference backend.

```
cmake --build build --target strix-skeleton-serve skeleton_engine_tests
ctest --test-dir build -R skeleton_engine
build/strix-skeleton-serve --registry-root examples/skeleton_engine \
    --registry models/registry.toml --model skeleton --port 8080
curl -s localhost:8080/v1/completions -d '{"model":"skeleton","prompt":"hi","max_tokens":4}'
```

### Families already on the layer

Two production families are worked examples of the two ways to adopt it:

- **Gemma4** (single device): the family implements `session::Engine` and the
  shared `SessionRuntime` drives admission, steps and publication
  (`src/gemma4/engine/session_service.cpp`).
- **MiniMax** (two ranks over a pair transport): the family keeps its own
  planner, because every step is a lockstep exchange with the second rank, and
  uses the runtime's native control mode. The planner reserves, activates and
  completes requests through `SessionRuntime`; the queue, the slot store, the
  resident-prefix cache and the HTTP front end are the shared ones
  (`src/minimax/serve/minimax_lead.hpp`, `session_service.cpp`,
  `serve_runtime.cpp`). Rank 1 is the same service in the follower role.
  `strix-serve --model minimax-m3.1-flash MODEL_DIR --rank R --transport T ...`
  and the legacy `strix-minimax-serve` spelling start the same service.
  `docs/engine/minimax-frontend.md` and `docs/engine/minimax-paired-state.md`
  describe the family side.

## What a family implements

| Piece | Where | In the example |
|---|---|---|
| Model description | `models/<model>.toml` and a row in `models/registry.toml` (`name`, `family`, `description`; a production row also names its CMake option, profile tag, defaults and adapter files) | `examples/skeleton_engine/models/` |
| Model adapter | A class deriving from `strix::session::Engine` (`include/strix/session/engine.hpp`) | `skeleton::Engine` |
| State hooks | `strix::session::Hooks` given to `KvStore`: initialize, reset, release, capture, view, restore, truncate | `Engine::hooks()` |
| Tokenizer | `serve::BpeTokenizer`, or any `serve::ApiTokenizer` (`include/strix/serve/tokenizer.hpp`) | `ByteTokenizer` |
| Renderer and parser | `ApiConfig::chat_renderer`, optional `output_parser`, `stop_tokens` (`include/strix/serve/api.hpp`); shared templates live in `include/strix/serve/chat_template.hpp` | the lambda in `Service` |
| Service | A `serve::SessionService` that owns backend, engine, queue, `SessionRuntime` and `ApiService` (`include/strix/serve/run.hpp`) | `skeleton::Service` |
| Registration | `install_session_service(ServiceFactories&)` under the registry `family` value, referenced from `install_service_factories` in `src/serve/run.cpp` behind the family's CMake option | `install_session_service` |

### The Engine contract

Required:

- `prefill(tokens, state)`: consume the tokens into the slot. The call must consume
  all of its input, so that a snapshot taken afterwards sits exactly at that frontier.
- `decode(state, sampler)`: produce one or more tokens. The newest token stays
  *pending* (produced, not yet consumed as a row). The result reports the frontier
  before and after, and a finish reason. A step is terminal when it reaches a stop
  token (`stop`), uses up `sampler.max_tokens` (`length`), fills the context
  (`context`) or is cancelled; the runtime rejects a result that breaks these rules.
- `caps()`: context, slots, batch width and which optional features exist.
- `receipt()`: the execution receipt reported with every response.

Optional, each announced through `caps()`: `prefill_batch` and `decode_batch`
(several sessions in one forward), `advance` (mixed prefill and decode rows in one
cycle, with `RowHints`), `fork_lane`, `cancel_lane`, `resume_lane`,
`append_external` (lanes of one parallel job), `commit_pending`, and
`RecurrentStateOps` (`include/strix/session/recurrent_state.hpp`) for models whose
state is not only attention KV.

### Embedding rows (image input)

A prompt stays a sequence of 32-bit ids. A row that is not a vocabulary token, such as
one output row of a vision encoder, is a *virtual id* (`include/strix/session/embed_rows.hpp`):

```
bit 31 = 1 | handle (17 bits) | row (14 bits)
```

- `EmbedInterner` maps a digest (content and producer identity) to a handle, once per
  process. The same image gives the same ids, so prefix directories, turn caches and
  family prefix caches compare and hash image runs like text, and a prefix may end
  inside an image. A handle keeps its payload while a lease exists; afterwards the
  handle stays reserved for its digest until the handle space wraps (`epoch()`).
- `EmbedTable` is the immutable handle -> rows view of one request: bf16 rows of the
  LM width, a role per row, optional positions, the vocabulary token a row stands
  for, and optional rows added after later layers. It owns leases.
- `expand_embed_rows` replaces each placeholder token of the rendered prompt by the
  ids of one span and reports where the runs are.
- `GenerationJob::embeds` carries the table of the request; it is null for a prompt
  without virtual ids.

An engine opts in with `Capabilities::embed_rows` and implements
`bind_embeds(state, table)`. Its embed stage then resolves token rows by lookup as
before and virtual rows from the bound table, after `EmbedTable::validate` passed for
the whole input and before any write. An engine without the capability refuses any id
with bit 31 set in prevalidation (`reject_embed_ids`). Decode never produces a virtual
id. Handles are local to the process: what leaves it carries `EmbedTable::manifest()`
and the receiver rewrites the ids with `remap_embed_ids`.

A prompt without virtual ids takes the same path as before, with or without a bound
table: ids, state and results are unchanged.

#### Serving media through the API

The API layer needs one object per media kind, `ApiMediaAdapter`
(`include/strix/serve/api.hpp`), registered in `ApiConfig::vision.adapters`:

| Member | Meaning |
|---|---|
| `kind()` | the canonical part type it consumes (`"image"`) |
| `identity()` | weights digest and arithmetic id; shown in `/v1/models` and part of the capabilities hash |
| `placeholder()` | text that tokenizes to exactly one id; marks where the rows of one item go |
| `part_text(n)` | what the renderer receives for the n-th item (default: the placeholder alone) |
| `max_rows()` | upper bound of the rows of one item |
| `encode(item, row_budget)` | preprocess, encode or hit a cache, intern; returns the lease and the id layout |
| `epoch()` | the interner epoch; when it moves, `ApiConfig::vision.on_epoch` withdraws cached prefixes |

With an adapter loaded, the three front doors (`/v1/chat/completions`, `/v1/messages`,
`/v1/responses`) pass the body through `serve::media::normalize`, hand the renderer
`part_text` in place of each media part, replace each placeholder token by the ids of
its item and submit the job with `GenerationJob::embeds`. `SessionRuntime` checks the
ids against the table before a lane exists and calls `bind_embeds`. `usage` counts the
rows, the `strix` receipt gains a member per kind (`vision`), and `/tokenize` returns
real ids plus an `images` array. The capability `image_input` is true only then.

Without an adapter nothing above runs: media parts get the refusal they always got,
and text requests are unchanged with or without one. Remote image URLs are off unless
the service sets a fetch policy and a fetcher.

The service that loads an adapter also folds its identity into the prefix format id
(`embed_format_id`), so a change of tower or weights invalidates image-bearing
prefixes through the key. Adapters shared across services must synchronize their own
encoding and epoch state; kinds in one request must use a shared handle namespace.
The epoch callback synchronizes with the cache owner and completes withdrawal before
returning. The API mutex alone does not protect runtime caches.

Fake-encoder and skeleton tests qualify request plumbing only. They do not establish
real embedding consumption, decoder safety, model fidelity, epoch-cache invalidation
on a production runtime, or GPU qualification.

### Backend operations

A family asks a `session::Backend` (`include/strix/session/backend.hpp`) for:

- buffers (`allocate`), queues (`create_queue`), events (`record`, `wait`,
  `synchronize`);
- host and device copies (`upload`, `download`, `copy`) and `memory_budget()`;
- kernels through the operation registry: `supports(key)` and `query(key)` answer
  whether an operation exists for a data type and format, `dispatch` runs one, and
  an `ExecutionPlan` runs a prepared sequence. A missing operation is reported as
  `UnsupportedOperation`, never as a silent fallback.

Dispatch is per operation, on the host. There is no virtual call inside a kernel.

## What a family receives

Provided by the session and serve layers once the pieces above exist:

- **Serving**: `strix-serve --model <name>`; the OpenAI-compatible front end
  (`/v1/completions`, `/v1/chat/completions`, `/v1/messages`, `/v1/responses`,
  `/tokenize`, streaming, tools, structured output), health, status, metrics, drain
  and authentication.
- **Sessions and slots**: the request owner (`serve::SessionRuntime`) leases slots,
  schedules prefill and decode, batches sessions up to the reported width, handles
  cancellation and stop conditions, and checks every decode result.
- **Prefix cache**: longest-common-prefix restore and the end-of-turn snapshot that
  makes the next turn prefill only the new message. It is a facade over the single
  `PrefixStateCache`, so there is one byte budget and one durable writer
  (`include/strix/session/prefix_cache.hpp`, `checkpoint_io.hpp`).
- **Row budget**: `RowHints` and `EffectiveRows` carry the per-cycle row budget of a
  cohort to the engine and back; a family that implements `advance` gets mixed
  prefill and decode cohorts from the shared scheduler.
- **Parallel jobs**: fork, per-lane cancellation and batched lanes
  (`include/strix/session/parallel_job.hpp`).

Shared libraries that a family calls instead of copying (see "Shared code first" in
`AGENTS.md`):

- **Two-node pipelines and transports**: `include/strix/pair` (link, lockstep, pipe,
  balance) and `include/strix/dist` (transport selection, GPU-direct and host copy).
- **Kernels**: `src/core` (linear, integer and packed plans), `src/moe_core` (MoE
  routing and grouped GEMM), `src/attn_core` (flash attention tiles, masks, online
  softmax).
- **Drafting**: `include/strix/spec` and the speculation controller and verifier
  headers. A family that supplies a draft source reports it through `caps()` and the
  accepted and proposed counts of `StepResult`; drafting is on by default where the
  family qualifies it.

## Backends

| Backend | Files | Built when | Selected by |
|---|---|---|---|
| CPU reference | `include/strix/session/backend_cpu.hpp`, `src/session/backend_cpu.cpp` | always (part of `strix_session`) | the service constructs `session::CpuBackend` |
| HIP | `include/strix/session/backend_hip.hpp`, `src/session/backend_hip.cpp` | `STRIX_ENABLE_HIP_RUNTIME=ON` (`strix_session_hip`) | the service constructs the HIP backend |

The service is the only place that names a backend: in the example it is the
`ss::CpuBackend backend_;` member, and the engine takes a `session::Backend&`.
A family that follows this pattern needs no accelerator header in its adapter.
`backend_hip.hpp` itself exposes no HIP type; the HIP runtime is included only by
`src/session/backend_hip.cpp`.

### What still leaks

The seam covers buffers, queues, events, copies, the memory budget and operation
dispatch. Today it is the storage and scheduling boundary, not yet the kernel
boundary:

- The production kernels of every family are HIP translation units (`.hip`) that
  take HIP streams and device pointers directly. Their operations are not yet
  registered in the Backend operation registry, so a family cannot run its real
  forward pass on another backend by changing the constructed backend.
- The shared kernel cores (`src/core`, `src/moe_core`, `src/attn_core`) and the
  GPU-direct transports include the HIP runtime directly.
- The session adapters of the migrated families use `session::Backend` for session
  state, while their engines keep HIP types for model execution.
- The CUDA path is a separate set of sources (`src/cuda`, `cmake/QwenCuda.cmake`)
  that does not implement `session::Backend` yet; `engine/backends.md` maps it.
- Families that are not migrated own their request loop and caches and do not use
  the seam at all.

The CPU backend is a correctness reference for the contracts and for host tests. It
is not a production inference path.

## Levers and switches

A lever is an implementation choice with an off switch. Declare it as one row of a
`strix::lever::Decl` table (`include/strix/lever.hpp`): its `STRIX_*` switch, its
exactness class (`bitwise`: on by default; `bit_changing`: off until the engine's
gate opens), its default, and, when it wins only in some regimes, the crossover on
one axis (rows, context tokens or concurrent streams). `lever::select` is the
decision rule (explicit switch, then gate, then regime, then default) and records
the result for `--print-effective-config`; `lever::require_consistent` refuses two
levers that the table declares incompatible. `static_assert(lever::valid(table))`
checks the table at compile time.

The environment-reading overload is for startup configuration: it allocates and
records under the shared registry mutex. Capture an owned switch value at startup
and use the explicit-value overload for per-request or per-launch decisions.
Existing `profile::Lever` and `profile::DefaultItem` rows in `engine_profile.hpp`
remain authoritative for profile values, defaults and qualification. Before an
engine adopts `lever::Decl`, provide a shared adapter deriving the overlapping
fields from those rows; do not maintain a second set of names or metadata.

Read any other switch through `include/strix/env_flags.hpp`. The
`strix-env-switch-census` test (`tools/env_switch_census.py --check`) fails when a
new switch is read with a raw `getenv`, or when one boolean switch is read with two
different defaults. `tools/env_switch_baseline.txt` lists the switches that are
still read raw; after introduction it only shrinks (`--update` removes migrated names).
`tools/env_switch_census.py --report` prints the current counts.
`--seed` creates a baseline only when the destination does not exist. The census
tracks literal names, including adjacent ordinary string literals; runtime-built
names and wrapper reads are outside its scope. Escapes in raw literal names are
rejected because this scanner does not decode C++ escapes. Existing baseline names may gain
raw read sites, so a passing census is not proof that all configuration is governed.

### grandfathered at introduction — 2026-11-13

These switches reached main before the census ratchet landed. The explicit
`python tools/env_switch_census.py --grandfather NAME... --root .` route adds only
the named switches and refuses any name not currently read raw, without writing
on refusal. CMake continues to call `--check`, never `--grandfather`.

| Switch | File | Owner area | Status |
|---|---|---|---|
| `STRIX_BATCH_NEST_RATIO` | `apps/hip_mtp_decode_bridge.hpp` | Qwen HTTP scheduling | to be governed |
| `STRIX_GLM_PF_DENSE_FORM` | `src/glm/prefill/glm_prefill.hip` | GLM prefill | to be governed |
| `STRIX_HOST_WAIT` | `src/device/hip_gate.cpp`, `src/device/hip_gate_none.cpp` | Shared device host-wait policy | to be governed |
| `STRIX_MM_PF_MSA_FLASH` | `src/minimax/engine/engine.hip` | MiniMax prefill | to be governed |
| `STRIX_MM_PF_MSA_FLASH_LAYERS` | `src/minimax/engine/engine.hip` | MiniMax prefill | to be governed |
| `STRIX_MSA_TEST_FLASH` | `src/minimax/attn/msa_fused_tests.hip` | MiniMax attention qualification | to be governed |
| `STRIX_PREFILL_FINAL_CONE` | `src/known_prefix_prefill_runtime.cpp` | Shared known-prefix prefill | to be governed |
| `STRIX_PREFILL_FINAL_CONE_ROWS` | `src/known_prefix_prefill_runtime.cpp` | Shared known-prefix prefill | to be governed |
| `STRIX_PROMPT_BATCH_LOG` | `apps/hip_mtp_decode_bridge.hpp` | Qwen HTTP scheduling | to be governed |

`STRIX_GLM_ABA_FOLD` in `apps/glm_rank_main.cpp` instead uses `envcfg::flag`
with exact matching: only `"1"` enables it, as before. `--update` removed
`STRIX_CONSTRAINT_MASK_AHEAD`, `STRIX_CONSTRAINT_RANKED` and `STRIX_DS41_PF_CORE`
because they are no longer read raw.

## Checklist

1. `models/<model>.toml`, registry row, CMake option `STRIX_FAMILY_<NAME>`.
2. Model code under `src/<family>/`: tensors, numerics, kernels that are truly
   model-specific. Reuse `src/core`, `src/moe_core`, `src/attn_core`.
3. `session::Engine` adapter and `KvStore` hooks; report capabilities truthfully.
   A family with image input sets `embed_rows` and resolves virtual ids in its
   embed stage; every other family rejects them before a write.
4. `SessionService` and `install_session_service`; no `main()` and no request loop.
5. Host test with the CPU backend or a scripted engine, in the style of
   `tests/skeleton_engine_tests.cpp`; device exactness gate for the real kernels.
6. Documentation in `docs/models/` and a row in the family table.
