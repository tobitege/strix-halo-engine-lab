# Deployment on different hardware

[Español](deployment.es.md) · [Individual switches](toggles.md)

Deployment is planned from measured device facts and model ownership, without named
hardware classes or installed-RAM lookup tables. Build for the actual GPU target;
an admitted memory plan does not add kernel support for a different architecture.

## Availability

Checked on the lab's two gfx1151 nodes (117.2 GiB device-usable memory each): strict
native builds of every engine app below, the host tests, `--deploy-check` on both
ranks (no device opened, no peer connection) and with simulated memory sizes. The
last column says what a served `--deploy auto` run on those nodes has shown. Complete
token/logit/state qualification and allocation peaks on other hardware remain
required; a successful inspection is not that qualification.

| Engine | Deployment choices | Launch requirements under `--deploy` | Checked on the lab nodes |
|---|---|---|---|
| Qwen | Native encoding; single node, two replicas, explicit two-node pipelined prefill | Registered eager tiled route; HTTP needs the tiled MTP serving arguments. Unpinned slots use the native per-stack state/scratch/metadata and bridge bounds; `--http-batch N` caps the search (`0` keeps the single runner). Default session RAM is fitted from the remainder after slots and mandatory owners; explicit `--session-ram-bytes` never shrinks. Pipeline is CLI prefill with rank-0 continuation, not HTTP. YaRN, copy and graph routes refuse. | Served (`--deploy auto`, HTTP): plan, `deployment_id`, capacity and drain as specified; 24 greedy tokens equal to the same launch without `--deploy`. The previous adapter forced 1 slot where the plain launch served 4; the new slot/session sizing has host coverage and awaits a node re-check. |
| Nemotron | Q4_K_M; single node or independent replicas | `--draft off`. EXL3/drafter sizing and pair execution are not bound. The prefill prepared-panel bound is conservative. | Served (`--deploy auto`, single node): 8 slots at 65,536 context, the same as the plain launch; plan, `deployment_id`, capacity and drain as specified; 24 greedy tokens equal to the launch without `--deploy`. |
| DeepSeek | Native encoding; TP2 | `--cache-profile off`, `--no-dspark`, no session RAM or heterogeneous stage, trellis off (`STRIX_DS41_TRELLIS=0`: its workspace is not sized). Unsupported prefill estimates refuse. | Served pair (`--deploy auto`): one agreed plan (84.6 GiB required per rank; the load reports 79.8 GiB of weights and 0.88 GiB of prefill-owned weights); capacity and drain as specified; 24 greedy tokens equal to the launch without `--deploy`. That build did not yet publish the plan on `GET /v1/deployment` or `/health`; fixed afterwards (strict build passes), served re-check pending. |
| GLM | EXL3; TP2 | Default profile only; `--prefix-cache-gib 0`. Default and explicit `STRIX_GLM_PF_PIPELINE=2` use the same bound (shared full-row scratch plus sequence-parallel rings and the coherent normalized-row buffer). Prefix import, external exchange and truncated stacks refuse. | Served pair (`--deploy auto`): one agreed plan (94.0 GiB required per rank; the engine's own load plan is 81.3 GiB physical); plan, `deployment_id`, capacity and drain as specified; 24 greedy tokens equal to the launch without `--deploy`. |
| MiniMax | EXL3; TP2 | `--dspark-mode off` (DSpark is found next to the model otherwise), `STRIX_SESSION_CACHE=0` (environment control) and no session save/restore; `--max-rows` at most 4096. | Served pair (`--deploy auto`): both ranks agreed on one plan (100.0 GiB required per rank against 109.2 GiB budget; the load reports 90.3 GiB device used); plan, `deployment_id`, capacity and drain as specified; 24 greedy tokens equal to the launch without `--deploy`. |

INT4 (FAST_I4) formats are not offered: their runtimes are not in this build, so no
plan can select them and a `format.prefill = "int4-stored"` or `"int4-jit"` pin
refuses with "no supported candidate". The providers keep a hook
(`deploy::int4_moe_formats`) that adds these candidates once an engine links its
INT4 runtime and controls. `profile.allow_pending` does not enable a missing runtime.
Disaggregated prefill is a schema vocabulary item, not an executable deployment
choice in these bindings. Existing launches without deployment retain their own
defaults and admission contracts.

## Inspect before loading

Append these options to the engine's normal command, preserving its required
model/state assets, profile and rank/transport arguments:

```sh
--deploy auto --deploy-check
--deploy deployment.toml --deploy-check
```

`--deploy auto` searches the provider's supported choices. `--deploy FILE` constrains
that search with a TOML file. `--deploy-check` prints the plan and exits before
weight payload loading, device-owner construction or pair connection; metadata
must still be readable. Pair inspection therefore needs measured peer inventory.
Inspect `refused`, `refusal`, candidate reasons, `requested`, `chosen` and each
rank's budget. A refused plan must not be launched. Remove `--deploy-check` to
execute an admitted configuration, keeping the same launch arguments.

The probe observes the gfx target, CU count, unified GTT versus discrete VRAM,
total usable bytes and currently free bytes. Unknown or ambiguous observations
remain unknown and cannot justify a fit. Nodes come from measured inventory;
link selection comes from the launch transport and spec, not an inferred peer or
a measured bandwidth benchmark. Live pairs exchange fresh rank-ordered facts and
agree on the plan hash before loading. Replicas remain separate launches managed
by the orchestrator; deployment does not start processes or configure networking.

For every candidate and rank the resolver reserves:

1. KV and other persistent request state for the requested context and slots.
2. Workspace for the supported prefill/decode envelope.
3. Weights, format conversions, staging and other named owners.

The available budget is measured free device-usable memory minus the safety
margin. Unified memory also remains subject to native host/GTT admission; system
RAM is not an extra pool to add to GTT. Every rank must fit. Context and pinned
slots never shrink. Unpinned slots are searched from the provider's maximum
downward before giving up; topology/format order is provider-defined, not a
universal throughput optimizer. Native allocator guards remain authoritative. Qwen also checks the planned HTTP slot count against native measured startup admission and refuses to serve if it no longer fits; re-inspect under current memory pressure. Optional default session RAM cannot take capacity from KV slots; it may resolve to zero. The resolved amount is included in the plan and applied to execution.

## TOML contract

The input schema is `strix-deploy-v1`; output JSON uses `strix-deployment-v1`.
These are constraints, not model converters or numerical qualification settings.

| Field | Effect and refusal conditions |
|---|---|
| `schema` | Required exact input schema string. Unknown keys, duplicates, malformed types and conflicting aliases refuse. |
| `model` | Optional provider model identifier; a different identifier refuses. Omit to use the active provider. It is not a model path. |
| `hardware.nodes`, `hardware.link` | Require the inventory count and launch-compatible transport. Link may be `auto` for a single node; multi-node choices require an explicit link. No peer facts are invented. |
| `topology.mode` | `auto`, `single`, `replica` (alias `replicas`), `tp2`, `pipeline`, `disaggregated`. Only provider-advertised choices can run. TP2 requires two measured nodes. |
| `topology.ranks` | Alias for node count; must agree with an explicit node count. Does not launch ranks. |
| `memory.context` | Positive requested token capacity; maps to the engine's context/capacity contract. Insufficient KV budget refuses rather than truncating context. |
| `memory.slots` | Positive explicit slot count, never reduced. Omit for provider search, subject to adapter and explicit CLI constraints. |
| `memory.policy` | Only `kv-first` is implemented; `formats-first` refuses. |
| `memory.margin_gib` | Nonnegative finite safety reserve, converted to bytes. A reserve larger than free memory cannot fit. |
| `format.decode`, `format.prefill` | `auto` or an exact provider encoding name. Unsupported pairs refuse; Qwen/DeepSeek currently use `native`, Nemotron `q4_k_m`, GLM/MiniMax `exl3`. |
| `format.resident_layers`, `format.total_layers`, `format.jit_slots` | Exact placement pins, including zero. Resident layers cannot exceed total layers; the complete tuple must match an advertised candidate. No arbitrary partial placement is synthesized. |
| `profile.name`, `profile.allow_pending` | Profile must match the active launch selection. `allow_pending` accepts only `false`; it cannot qualify a draft route. |
| `devices` | Optional array of independently measured node facts, in rank order for pairs. Missing, inconsistent or insufficient facts refuse inspection. Live pair execution replaces supplied facts with fresh observations. |
| `pins` | Optional string table repeating canonical explicit selections with identical values. It is not an override layer; mismatches refuse. Usually omit it. |

Root `nodes`, `link`, `context`, `slots`, string `topology` and string `profile`
are accepted aliases. Do not specify both a root field and its nested equivalent.
There are no `hardware.gfx`, `min_gtt_gib` or `format.verify_min_rows` keys in this
schema. Device facts describe observations, not hardware tiers.

### One file per engine

These contexts, slots and margins are illustrative policy choices, not measured
capacity or performance claims. Each file must pass inspection for its actual
assets and launch. Multi-node examples also require the inventory described below.

Qwen, single node (`qwen.toml`):

```toml
schema = "strix-deploy-v1"
model = "qwen3.8"
[hardware]
nodes = 1
link = "none"
[topology]
mode = "single"
[memory]
policy = "kv-first"
context = 8192
slots = 1
margin_gib = 8
[format]
decode = "native"
prefill = "native"
```

Nemotron replicas (`nemotron.toml`):

```toml
schema = "strix-deploy-v1"
model = "nemotron-bolt-e4b"
[hardware]
nodes = 2
link = "tcp"
[topology]
mode = "replica"
[memory]
context = 8192
slots = 1
[format]
decode = "q4_k_m"
prefill = "q4_k_m"
```

Launch each replica independently with `--deploy-node 0` or `--deploy-node 1`
and `--draft off`; the local indexed observation is remeasured at execution.
The link field does not turn replicas into a tensor-parallel pair.

DeepSeek pair (`deepseek.toml`; the active model identifier is intentionally inferred):

```toml
schema = "strix-deploy-v1"
[hardware]
nodes = 2
link = "usb4auto"
[topology]
mode = "tp2"
[memory]
context = 8192
slots = 1
[format]
decode = "native"
prefill = "native"
```

GLM pair (`glm.toml`):

```toml
schema = "strix-deploy-v1"
model = "glm-5.3"
[hardware]
nodes = 2
link = "usb4-auto"
[topology]
mode = "tp2"
[memory]
context = 8192
slots = 1
[format]
decode = "exl3"
prefill = "exl3"
```

MiniMax pair (`minimax.toml`):

```toml
schema = "strix-deploy-v1"
model = "minimax-m3.1-flash"
[hardware]
nodes = 2
link = "usb4-auto"
[topology]
mode = "tp2"
[memory]
context = 8192
slots = 1
[format]
decode = "exl3"
prefill = "exl3"
```

Use the transport spelling accepted by the selected engine. Keep one process per
rank for TP2 and retain the engine's normal peer/transport launch configuration.
For offline inspection, append one `[[devices]]` table per node. Supply all six
fields: `gfx_target` (string), `cu_count` (integer), `mem_total_bytes` and
`mem_free_bytes` (integers), `unified` (boolean), and `source` (nonempty string
describing measurement provenance). Copy actual observations; do not duplicate
the local device to stand in for an unmeasured peer. Free bytes must not exceed
total bytes. Keep inventories and resulting operational receipts private.

## Format by execution regime

The intended rule distinguishes row reuse, not hardware names. For decode at
`R = 1`, weight traffic commonly dominates: compact EXL3 can minimize bytes read.
For wide verification or batched work, INT4 can win when `rows >= R*`, the
engine/kernel crossover where reused matrix work outweighs additional storage
and conversion cost. Prefill uses INT4 where that runtime and profile are
available and qualified. `R*` is not a universal constant or a TOML pin.

After the INT4 runtimes, sizing and controls are integrated and qualified, the
remaining budget can hold all INT4 packs (`int4-stored`), per-layer just-in-time
packs (`int4-jit`), or a supported partial resident set plus JIT staging. Retained
EXL3 sources, scale/index data, conversion workspace and simultaneous load peaks
must also be charged. More memory can make all INT4 resident **after** the same
KV reservation; it does not automatically enlarge a pinned context or change
the decode identity. Today the GLM/MiniMax deployment bindings refuse these
INT4 execution pins instead of pretending EXL3 executes them.

| Hardware change | Planning consequence |
|---|---|
| More usable/free memory | May admit more unpinned slots or a supported resident format; every rank still has to fit. |
| More memory bandwidth | Changes decode cost and the INT4 crossover; requires local measurements, not scaled published throughput. CU count alone predicts neither. |
| Discrete GPU | Budget against free VRAM; charge host/device transfers and staging. Extra host RAM does not make weights VRAM-resident. Kernel support and native admission still apply. |
| Faster link | Can reduce collective or handoff cost and move the useful pipeline/disaggregation crossover. It does not enable an unsupported topology or merge node budgets. |

Single-node placement avoids communication. Replicas retain a complete model per
replica and serve independent requests. TP2 splits supported owners across two
ranks but still has replicated owners. Qwen pipeline retains the full model on
each rank and charges handoff staging. Disaggregated prefill would separate
prompt processing from decode and require supported state transfer; it remains
pending here. Automatic selection considers only the provider's implemented
choices, not all these designs on every engine.

## HTTP contract for orchestrators

Use the server's configured authentication for these routes.

| Surface | Contract |
|---|---|
| `GET /v1/deployment` | Immutable active plan JSON, including `deployment_id`; 404 when no active plan callback is installed. No fabricated default plan. |
| `GET /health` | Includes `deployment_id` when bound. Draining answers HTTP 503. |
| `GET /api/status` | `capacity` contains numeric `slots_total`, `slots_busy`, `queue_depth`, and nullable `kv_bytes_free`. DS41 with a shared KV pool publishes unassigned arena bytes across all free ranges (including pool slack), from a scheduler-thread snapshot refreshed on reserve/release; this is not the largest contiguous reservation or device headroom. DS41 without a pool, Qwen, GLM, MiniMax and Nemotron publish `null`: their KV is owned by fixed slots or individual streams, with no shared free-byte allocator figure. Nemotron's per-stream KV/overlay reuse is not a shared admission pool. |
| `POST /admin/drain` | Idempotently closes admission and returns `{"status":"draining","drained":false}` until accounted work completes, then `drained:true`. Status also exposes `drained` after drain begins. |

New generation admission after drain returns 503; ordinary queue saturation
remains 429. Running requests finish. The shared queue preserves accepted work;
DeepSeek retains its existing cancellation of queued/deferred work. Drain does
not terminate the process. Stop routing, request drain, wait for completion, then
let the process manager stop or replace the server.

`deployment_id` is the canonical plan hash (16 lowercase hexadecimal digits),
not a unique process/restart identifier or a model-fidelity certificate. Identical
plans can have identical IDs after restart; remeasurement can change the ID.
Use health and process lifecycle alongside it. `/v1/capabilities` continues to
describe executed API policies; deployment describes placement and budgets.

For individual controls and off switches, see [toggles](toggles.md). Keep model,
profile, context, slots and numerical identity fixed when validating a change;
compare complete token IDs and logits/state, then measure the complete workload.
