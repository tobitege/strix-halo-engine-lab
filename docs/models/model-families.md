# Model families

Model metadata, model-file admission, and serving support are separate contracts.
`models/registry.toml` and `models/*.toml` describe selectable models for the shared
serve shell; they do not add kernels or prove that a family can execute. The shared
shell resolves an explicit registry ID/name, or a family alias when it selects one
entry, through `--model`. Metadata `--inspect` reports the description and whether
a shared serve factory is linked. `--family-inspect` calls the linked factory's
deferred inspection after family validation, before service construction/device
work. Neither inspection mode executes a model.

The GGUF `ModelFamily` admission registry in
`include/strix/model_family_registry.hpp` is distinct: it validates architecture,
metadata and reader limits for GGUF admission. TOML selection does not imply that
every listed family is registered in that older registry, and shared serve
registration does not migrate that registry or a rank executable.

## Current migration status

- **Gemma4 / Nemotron** is the first real shared serving adapter. The shared
  `strix-serve` factory and the legacy Gemma4 serve executable both use the shared
  `SessionRuntime` owner. Its family planner preserves Gemma4's grouped execution.
- **MiniMax** has a real registry factory, extracted native Runner/resources and
  a thin compatibility entry using the common shell. The leader returns
  `Lead::runtime()`; rank 1 follows without an HTTP listener. Physical slot
  bindings and the native no-listener follower service are implemented.
  Explicit default-OFF [paired eager state](../engine/minimax-paired-state.md)
  connects generic storage hooks to the native Runner/follower for the bounded
  no-configured-drafter operator profile. Ordinary service construction remains
  OFF. A separate typed plain-greedy R1 library activation implements consumed
  input, retained-candidate decode, external append, grouped mixed cycles and
  lane cancellation/resume with complete paired rollback. Page COW, fork and
  completed-turn caching remain unsupported; native compiler/device gates are
  NOT RUN. The combined source/default selector audit
  includes shared session and runtime owners; stale source receipt rows remain
  unresolved for missing attributable provenance. Host fixtures do not
  establish model fidelity or native performance, and both-rank qualification
  remains pending.
- **Qwen, DeepSeek, GLM and other serve/rank entry points** retain their existing
  family paths. Rank-main unification is outside the completed serving slice.

`STRIX_LEGACY_MAINS` defaults ON for Gemma4 and MiniMax compatibility entries.
OFF retains family libraries, the new server and tests. Gemma4 and MiniMax factories
are explicitly linked by `strix_session_link_services`; a new family still needs
root build integration and an explicit factory reference. The future architecture
proposal is not the current build contract.

Preserve provenance for adopted code, dependencies, model assets and official
mathematical or ISA references. Keep external review correspondence and
comparative inspection artifacts outside public documentation; describe this
project's contracts and qualification evidence without implying independent
authorship of adopted material. See the repository's publication policy.

## Shared session family kit

Read the contracts in [session-layer](../engine/session-layer.md),
[Spanish session guide](../engine/session-layer.es.md),
[execution backends](../engine/backends.md), and
[recurrent state](../engine/recurrent-state.md).

- **Engine:** family adapters expose prefill/decode and optional mixed advance
  and lane hooks, along with effective `Capabilities` and an execution receipt.
  Unsupported policies fail closed. The family retains tokenization, chat
  semantics, arithmetic, stop handling, and compute planning. Gemma4's actual
  mixed route submits grouped native plans; this contract does not scalarize it.
- **Backend:** neutral buffers, queues, events, copies, budgets, operation lookup,
  and grouped-plan execution form the seam. The CPU implementation is an
  executable reference for host correctness, not an engine capable of loading
  and running arbitrary models. Gemma4 still uses its existing target resources
  and kernels through the grouped-plan adapter. The HIP resource adapter is thin;
  CUDA/Vulkan/SYCL/Level Zero model execution is not implemented by this API.
  Shared headers contain no accelerator SDK types.
- **KvStore:** generation-checked handles own slot lifecycles and rank views.
  Snapshots are sealed to payload provenance, schema, rank order, frontier and
  purpose. Prompt-reuse snapshots restore consumed state with a fresh sampler and
  drafter; full resume snapshots preserve pending tokens and complete sampler,
  draft, selection and policy cursors on every rank. Attention truncation alone
  cannot restore recurrent state.
- **PrefixCache:** longest-prefix lookup and admission use explicit eviction,
  duplicate, size and accounting policy. Identity includes model, numerical
  profile, state format and rank topology; matching token IDs alone never prove
  that opaque state can be restored. Gemma4 retains its legacy cache settings.
- **SessionHooks / SessionRuntime:** family hooks bridge typed request admission,
  engine advancement and physical rank state. The shared runtime owns queue
  admission, leases, prefix admission, accounting/publication and finish. Family
  planners keep native grouping, arithmetic and stop semantics.
- **SessionService / ServiceFactory:** a service owns assets, engine, store,
  queue, runtime and request handling with assets outliving their users. Factories
  validate and inspect before construction. `strix-serve` installs factories by
  explicit static-link references; registry descriptions alone do not install
  one. Adding a family currently requires build/factory integration edits.

## Reuse and qualification

Before adding a shape-specific kernel, inspect the shared blocks and backend
operation registry for an existing implementation that can express the required
operation. Reuse it through its contract where semantics and layout match. A
shared-kernel change still requires the affected families' complete exactness,
ISA and performance gates. Host references and synthetic replay establish only
their stated host contracts, never accelerator parity, model fidelity or
no-slowdown.

Gemma4's independent original-source frozen host replay supports migration claims
for its listed synthetic cases. Both current serve mains execute the migrated
implementation and are not independent old/new evidence. Standalone Debug and
Release host suites each passed 15 tests; root Debug passed 17 with two official
tokenizer skips. These results do not qualify device execution, real-model
fidelity, native speed, or universal backend portability. See the replay matrix
and remaining gates in [the English contract](../engine/session-layer.md).

For Gemma4 a local host replay can be built without accelerator SDKs:

```sh
cmake -S tests/gemma4_session -B build/gemma4-session-p5
cmake --build build/gemma4-session-p5 --config Debug
ctest --test-dir build/gemma4-session-p5 -C Debug --output-on-failure
```

An actual service run uses the shared shell and the family argument separator;
provide valid local assets for the selected checkpoint:

```sh
strix-serve --model nemotron_3_5_bolt_e4b -- MODEL.gguf --rope-tables TABLE_DIR --tokenizer tokenizer.json
```

Metadata inspection is `strix-serve --model gemma4 --inspect`; it reads registry
and description metadata only. Deferred family inspection is
`strix-serve --model gemma4 --family-inspect`; it validates and invokes the linked
family inspection callback without constructing the service or starting device
work. Host-only tests and the CPU reference do not execute the command's model.

## GGUF admission registry

`include/strix/model_family_registry.hpp` defines
`ModelFamily{name, architectures, validate, describe, reader_limits}`.
`register_model_family(family)` permits an identical repeat and rejects clashing
names or architectures. `find_model_family(architecture)` and
`registered_model_families()` query it. `admit_model_file(path)` reads the
architecture, applies the family's reader limits, validates the parsed metadata,
and rejects files without a matching registered family.

Registration remains explicit, for example
`strix::deepseek41::register_family()`. A static library must be explicitly
referenced by its program so its registration code is linked. Model description
registry rows are not substitutes for this admission call.

## Adding or changing a family

1. **Contract and admission.** Define geometry, tensor manifest and validation
   against actual metadata. Register only the architectures and reader limits the
   implementation supports.
2. **Execution plan.** Record ordered operations, tensor/storage expectations,
   activation and persistent-state ownership, and numerical identity. Keep
   family-specific semantics in the family adapter.
3. **Reuse before specialization.** Check shared operation blocks first. Add a
   kernel only when the existing contracts cannot express the required layout or
   arithmetic. Do not edit another family's hot path casually.
4. **Tests and gates.** Add host references and device comparisons for the exact
   operation sequence. A shared kernel change needs exactness and ISA review for
   every affected family, plus complete workload performance qualification.
   Host-only evidence must be labeled as such.
5. **Build and serving integration.** Wire targets into the actual family build,
   install a `ServiceFactory` through an explicit static reference if migrating
   serving, and keep rank-main work separately scoped. TOML metadata alone does
   not create runtime support.

## deepseek41 as the worked example

The family is registered through its explicit `register_family()` call. Its
existing GGUF reader contract retains numeric metadata arrays when configured
through `Limits::max_retained_array_elements`; the default remains zero. This
reader behavior is opt-in and leaves default parsing unchanged.

| Piece | Existing example | Library area |
|---|---|---|
| Contract, plan and registration | `deepseek41_model`, `deepseek41_plan`, `deepseek41_family` | `strix_family_deepseek41` |
| Formats, A8 and MoE windows | `ds41_formats`, `ds41_moe`, `hip_ds41_formats` | `strix_ds41_formats`, `strix_ds41_hip` |
| Layer sites and attention | `ds41_layer`, `ds41_attention`, HIP counterparts | `strix_ds41_layer`, `strix_ds41_attention` |
| Prefill and runners | `ds41_prefill`, host/device forwards | `strix_ds41_runtime`, `strix_ds41_device_forward` |

The MiniMax source audit includes the family source tree and the shared session
contracts, queue/storage/backend/prefix owners, runtime implementation, root
registration, and session CMake module. Its scope excludes shared HTTP,
tokenizer/registry, and transport implementations governed by separate gates;
Gemma4 family code and nested tests also remain separately gated. New files in
the MiniMax/shared-owner scope fail closed until reviewed. Seven older
device/default/shared-core receipt fingerprints have a tracked historical
version, but changed afterward and are not yet reconciled with selector gates.
They remain unresolved. This audit is source review only and grants no
numerical, model-fidelity, device, or performance qualification. Gemma4 host completed-turn integration is implemented; native qualification
remains separate.

Completed-turn reuse is a separate default-OFF shared library policy. Producers
need `turn_commit`, quiescent pending-anchor sealing and complete prompt-reuse
state that initializes new-request sampling. Ordinary `prefix_cache` support
is independent. Supply stable asset/arithmetic/schema/template/topology identity,
complete retained bytes and final-phase admission through the shared hooks; see
[conversation cache](../engine/conversation-cache.md). Shared host fakes do not
qualify a family producer or native/distributed adoption.

Final-phase admission (`turn_checkpoint`) precedes commit; optional
`turn_snapshot` prepares a sealed candidate after commit/frontier validation and
before prompt capture. Absence preserves commit-prepared producers. Zero byte
bounds refuse before work. Shared standalone tests pass 11/11 in Debug/Release,
legacy ON/OFF. The real Gemma continuation/turn-commit/turn-cache tests are centrally
registered against its production engine. Gemma4 host completed-turn producer integration is implemented. Its separate
family validation is not certified by the MiniMax paired-state tests. Native state, physical COW/fork,
distributed sessions and numerical/performance gates remain unqualified.


Serving consolidation ends with ONE request/slot owner, ONE `ApiService` with
messages/responses/tokenize for every family, and ONE pooled cache/checkpoint/
durable writer. Adoption order: Gemma4, MiniMax, Qwen, DS41, GLM. Extra current
implementations remain work; two launchers do not establish completion. The
[deletion inventory and compatibility gates](../engine/session-layer.md#consolidation-end-state-and-migration-order)
name the policy/IO removed in each slice. Generic resource/state/page implementations
remain held until a production adopter; frozen fanout contracts remain intact.
MiniMax resident prefill now uses the shared cache/slot owner, separate from its
default-OFF generic state/input profiles. RAM retention now delegates to the existing `PrefixStateCache`/pool/ledger for
Gemma4 snapshots/turns and MiniMax resident/named paths, with one injected owner
per service. The common host checkpoint IO now has actual Qwen named-session,
MiniMax named-session and pooled-prefix publication adapters, preserving legacy
formats and family manifest/rank policy. Gemma4 retains typed RAM snapshots.
The [checkpoint storage contract](../engine/checkpoint-storage.md) records the
remaining platform and adopter gates. Shared test registration alone does not
complete the single frontend/loop/cache/checkpoint/writer migration. Other family
policies and native qualification remain outstanding in the stated order. No
device owner is serialized by the RAM owner and no new fork/COW or portability
is advertised.
