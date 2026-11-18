# Shared cores: a map of the code

*[Español](shared-cores.es.md)*

The engine builds each mechanism once, in a shared home, and a model family only binds it: it picks a policy,
supplies a small adapter or traits class, or reads a table row. This page maps the shared homes to the code and
says which engine binds each one today. It is a map, not a contract: the design rules are in
[docs/architecture](../architecture/README.md) and the switches in [toggles](../toggles.md).

Engines named below: Qwen (Qwen 3.8, `src/` top level and `include/strix/qwen_*`), DeepSeek V4.1 (`*ds41*`,
`*deepseek41*`, `src/deepseek41/`), GLM-5.3 (`src/glm/`), MiniMax M3.1 (`src/minimax/`), Gemma 4 / Nemotron
(`src/gemma4/`), and the vision tower (`src/vision/`). "Users" lists the files that include the shared header on
the main branch; it is a starting point for reading, not a full call graph.

## Summary

| Component | Home | Main files / types | What an engine binds | Users today | Deeper docs |
|---|---|---|---|---|---|
| FP16 tall dense core | `src/core` | `f16_dense_tall.hpp` (`ColumnTile`, `Plan`, `tail_policy`) | a named policy (below) | GLM KDA prefill, Qwen GDN launcher, vision encoder | [kernel design](../architecture/05-kernel-design.md) |
| FAST Q8_0 x A8 dense policy | `src/core` | `f16_dense_q8a8.hpp` (`View`, `supported`, `enqueue_parts`), policy `dense-q8a8-f16-v1` | a `View` of its weights and rows | DeepSeek V4.1 (lever `ds41-dense-f16-v1`) | [toggles: profiles](../toggles.md#profiles-exact-vs-fast) |
| Tiled WMMA prefill core | `src/core` | `tiled_core.hpp` (`GroupedArgs`, `NoAdapter`, `NoChain`), `wmma_tiled.hpp/.hip` | an `Adapter` and a `Chain` template argument | DeepSeek V4.1 (`src/ds41_integer_operand.hpp`), GLM and MiniMax prefill MoE (via `moe_i4_chain.hpp`, `wmma_tiled.hpp`) | [tiled prefill](prefill/tiled-prefill.md) |
| Integer mainloops | `src/core` | `integer_mainloop.inc`, `integer_grouped_mainloop.inc`, `integer_grouped_schedule.hpp`, `integer_plan.hpp`, `integer_operand.hpp` | an operand policy and plan flags (`Partial`, `SplitStage`) | DeepSeek V4.1 (`src/ds41_dense_adapter.hpp`, `src/hip_ds41_prefill.hip`) | [quantization and numerics](../architecture/06-quantization-and-numerics.md) |
| Small-row GEMV | `src/core` | `gemv_rows.hpp` with `gemv_rows_direct.inc` / `gemv_rows_staged.inc`; `gemv_policy.hpp` (`GemvKernel`, `GemvSelection`) | operand, lane sum and epilogue of the sweep | Qwen (`src/hip_q8_rows_gemv.hpp`), Gemma 4 (`src/gemma4/kernels/linear.hip`) | [kernel design](../architecture/05-kernel-design.md) |
| Row layout / pitch rule | `src/core` | `row_layout.hpp` (`RowLayoutPolicy`) | the row stride of its row-major buffers | shared execution arena; Qwen known-prefix window, MoE and shared-expert launchers | [row layout](prefill/row-layout.md) |
| EXL3 trellis codec | `src/core` | `trellis_format.hpp` (format, host reference), `trellis_codec.hpp` (device reader) | packed trellis weights read through the codec | Gemma 4 (`src/gemma4/exl3/`), MiniMax (`src/minimax/moe/`, `src/moe_core/exl3_trellis.hpp`), tiled core | [trellis codec](kernels/trellis-codec.md) |
| Engine profiles | `include/strix` | `engine_profile.hpp` (`Name`, `Lever`, `Engine`, `Status`) and one `*_profile_table.hpp` per engine | one table row per lever | Qwen (`profiles/qwen.hpp`, `qwen_profile_table.hpp`), DeepSeek V4.1 (`deepseek41_pair_profile_table.hpp`, `profiles/ds41_rtx.hpp`), GLM (`glm_pair_profile_table.hpp`), MiniMax (`src/minimax/engine/mm_profile_table.hpp`), Gemma 4 (`profiles/gemma4.hpp`) | [toggles: profiles](../toggles.md#profiles-exact-vs-fast) |
| FAST bundle helper | `include/strix` | `fast_profile.hpp` (`Member`, `Bundle`, `lever()`, `expand()`) | a bundle switch and its member levers | DeepSeek V4.1 (`ds41_fast_profile.hpp`), MiniMax (`mm_profile_table.hpp`) | [toggles](../toggles.md) |
| Lever declarations | `include/strix` | `lever.hpp` (`Decl`, `Regime`, `Choice`, `select`) | a `Decl` row: switch, exactness, default, regime | shared serve rules (`serve/cache_budget_rule.hpp`, `serve/queued_prefix_protection.hpp`) | [regime rule](../toggles.md#the-regime-rule) |
| Grouped MoE prefill core | `src/moe_core` | `grouped_core.hpp` (`GroupedCore`), `grouped_policy.hpp` (`GroupedPrefillPolicy<Xor64, PipeTail>`, `GroupedLdsRing`), `grouped_f16.hpp` | weight decode, expert layout, input transform, epilogue, traits, policy | GLM (`src/glm/prefill/glm_moe_traits.hpp`), MiniMax (`src/minimax/moe/mmx_prefill_core.hpp`), Qwen (`src/hip_moe_q4k_a8_known_rows.hip`, storage policy) | [MiniMax grouped prefill](../minimax-grouped-prefill.md), [routed MoE organizations](prefill/routed-moe-organizations.md) |
| MoE decode schedules | `src/moe_core` | `decode.hpp`, `decode_schedule.hpp`, `grouped_integer_decode.hpp` | traits: K chains, split materialization, epilogue | GLM (`src/glm/moe/glm_moe.hip`), MiniMax (`src/minimax/moe/mmx_moe.hip`, `mmx_moe_traits.hpp`), DeepSeek V4.1 (integer decode) | [kernel design](../architecture/05-kernel-design.md) |
| Router helpers | `src/moe_core` | `route.hpp` (serial k-pass selection), `route_prune.hpp` (FAST `RoutePrune`), `device_plan.hpp` (sync-free device route planner), `one_split_prefill.hpp` | a score / choice policy | `route.hpp`: Qwen, DeepSeek V4.1, MiniMax; `route_prune.hpp`: Qwen; `device_plan.hpp`: GLM, MiniMax, Qwen; `one_split_prefill.hpp`: GLM | [MoE route prune](kernels/moe-route-prune.md) |
| Sparse flash, key split | `src/attn_core` | `sparse_flash_keysplit.hpp` (`KeySplit<H, D>`, `Split = true` two-term operands) | key lists, scales, masks (policy) | MiniMax MSA (`src/minimax/attn/msa_flash_policy.hpp`, `msa_fused.hip`) | [MiniMax MSA](../minimax-msa.md) |
| Sparse flash | `src/attn_core` | `sparse_flash_wmma.hpp` | per-row key lists | GLM prefill, DeepSeek V4.1 attention | [kernel design](../architecture/05-kernel-design.md) |
| Dense flash prefill | `src/attn_core` | `dense_flash_prefill.hpp` (FAST causal GQA) | head geometry and operand type | Gemma 4 (`src/gemma4/prefill/attention.hip`), vision encoder | `src/gemma4/prefill/README.md` |
| Banded and selected flash | `src/attn_core` | `banded_flash_prefill.hpp` (paired-key schedule), `selected_flash_prefill.hpp` (selected-key schedule) | key decoding, row access, masks, value-row permutation | DeepSeek V4.1 (banded), Qwen (selected, `src/hip_indexed_attention_launcher.hip`) | [kernel design](../architecture/05-kernel-design.md) |
| Index scores and selection | `src/attn_core` | `index_score_wmma.hpp` (FAST), `index_score_band.hpp` (exact top-k through approximate scores), `index_score_mode.hpp`, `select.hpp` (exact radix selection) | indexer geometry, selection policy | GLM (`src/glm/prefill/glm_prefill.hip`, `glm_dsa.hip`), MiniMax (`msa_fused.hip`) | [index score WMMA](prefill/index-score-wmma.md) |
| Flash building blocks | `src/attn_core` | `flash_core.hpp`, `flash_tile.hpp`, `online_softmax.hpp`, `masks.hpp`, `wmma_frag.hpp` | (used by the cores above) | every flash core | - |
| Pair transport | `src/xport` | `channel.hpp` (TP2 staging and progress), `transport_backend.hpp`, `staging.hpp` (ticket / lease core), `framed_exchange.hpp`, `row_fold.hpp` | wire ABI in `Traits` | GLM (`src/glm/xfer/`), DeepSeek V4.1 (`src/hip_deepseek41_prefill_xport.hpp`), MiniMax (`src/minimax/xfer/`, `src/minimax/pfx/`) | [pair](../pair.md), [communicator](transport/communicator.md), [exchange row fold](decode/exchange-row-fold.md) |
| Pair route policies | `include/strix/pair` | `link.hpp`, `pipe.hpp`, `balance.hpp`, `regime.hpp`, `prefill_group.hpp`, `lockstep.hpp`, `producer_publish.hpp`, `embed_frame.hpp` | the engine's layer count, cut and frames | `link`/`pipe`: Qwen, Gemma 4; `balance`: Qwen; `regime`: Gemma 4; `prefill_group`: MiniMax; `lockstep`: GLM, DeepSeek V4.1, MiniMax; `producer_publish`: GLM; `embed_frame`: DeepSeek V4.1 | [pair prefill](prefill/pair-prefill.md), [multi-node](../architecture/12-multi-node-and-usb4.md) |
| Session layer | `src/session`, `include/strix/session` | `engine.hpp`, `kv.hpp`, `backend.hpp`, `prefix_cache.hpp`, `parallel_job.hpp`, `embed_rows.hpp`, `embed_splice.hpp` | identity fields of a prefix, embed rows of a prompt | prefix cache: MiniMax session cache, shared serve runtime; embed splice: DeepSeek V4.1 | [session layer](session-layer.md), [prefix state cache](serving/prefix-state-cache.md) |
| Serve front doors | `src/serve`, `include/strix/serve` | `api.hpp`, `http_server.hpp`, `media.hpp`, `run.hpp`, `session_runtime.hpp` | an adapter (assets, engine, queue, handler) | every served engine | [serving](../serving.md), [serving internals](serving/serving.md) |
| Serve scheduling helpers | `include/strix/serve` | `prefill_interleave.hpp`, `prefill_group.hpp`, `token_memo.hpp`, `cycle_receipt.hpp` (`RunReceipt`), `row_budget.hpp` | plan calls from the scheduler | interleave: GLM serve; prefill group: DeepSeek V4.1 serve; token memo: MiniMax API, shared API; receipt: DeepSeek V4.1 serve, Qwen MTP runtime, shared API | [token memo](serving/token-memo.md), [cycle receipt](serving/cycle-receipt.md), [row budget](serving/row-budget.md) |
| Server / CLI parity gate | `include/strix` | `serve_cli_parity.hpp`, test `tests/serve_cli_parity_tests.cpp` | the key=value config of both entry points and the named differences | GLM (`apps/glm_rank_defaults.hpp`) | - |
| Vision runtime | `src/vision`, `include/strix/vision` | `preprocess.hpp`, `encoder.hpp` / `encoder.hip`, `api_media.hpp`, `embed_cache.hpp` (`EmbedCache`), `adapter.hpp` (`VisionAdapter`) | one `VisionAdapter` | DeepSeek V4.1 (`src/deepseek41/vision_adapter.hpp`, `apps/deepseek41_serve_main.cpp`) | [image input](vision.md), [vision adapters](vision-adapters.md) |
| Speculative drafting | `src/spec`, `include/strix/spec` | `drafting.hpp` (`Source`, `Proposal`, `ContextProposer`, `ArbiterConfig`), `drafting.cpp` | draft sources (serial, lookup, model) | Gemma 4 (`src/gemma4/engine/lookup.hpp`) | [speculative decoding](../architecture/09-speculative-decoding.md), [prompt lookup](speculative/prompt-lookup.md) |
| Quality gate | `tools/strix-gate` | `gate.py` (scoped A/B runner), `fast_logit_gate.py` (FAST logit gate, `--dnll-rule floor`) | a job description per engine | every FAST lever; the floor rule is for DeepSeek V4.1 FAST prefill only | [gate README](../../tools/strix-gate/README.md), [methodology](../methodology.md) |

## The components

**FP16 tall dense core** ([`src/core/f16_dense_tall.hpp`](../../src/core/f16_dense_tall.hpp)). A dense FP16 GEMM for
prefill row batches on WMMA. It is FAST class: activations are rounded to FP16, one WMMA chain per tile from +0 in
ascending k. Named policies sit on top of it. The policy `dense-q8a8-f16-v1`
([`f16_dense_q8a8.hpp`](../../src/core/f16_dense_q8a8.hpp), namespace `strix::core::f16_dense_tall::q8a8`) runs
Q8_0 weights x A8 activations as one FP16 chain per output instead of the exact integer products; DeepSeek V4.1
binds it (worked example below).
Its stagers decode two weights per packed FP16 operation (`kWPacked`, `STRIX_F16_Q8A8_WPACK`, default on, same bits;
the kit `ds41/dense-forms` times both forms and gates the bits). The optional `kColumnFast` trait puts the column blocks on `grid.x`, so the CTAs
of one row block share x in L2 (same bits). MiniMax's FAST index projection `minimax-index-proj-wmma-v1`
([`src/minimax/attn/index_proj.hpp`](../../src/minimax/attn/index_proj.hpp)) binds the core with `stage()` reading its
packed weights and, from 2,048 rows, x pre-rounded into k-tile-major FP16 tiles. That x16 tile form is shared
(`X16Cols<Base>`, `enqueue_x16_tiles`, `kX16Rows` in the core; same bits as the FP32 x path): MiniMax binds it for
the index projections (`STRIX_MM_PF_IDX_WMMA=1`, layers `STRIX_MM_PF_IDX_WMMA_LAYERS`, default 40-59, opt-in until its
gate), GLM for the WMMA groups of x in `kda_wmma::dense_group` (KDA f_a | g_a | b; `STRIX_GLM_PF_X16=0` = direct form)
and its FAST indexer projections `glm-pf-idxproj-wmma-v1` (`STRIX_GLM_PF_IDXPROJ_WMMA=1`, opt-in until its gate). Kit:
`STRIX_MMATTN_BENCH=wmma` on `src/minimax/attn/index_proj_tests.hip` (strict ABAB, series `x16` / `diag` / `orders`
in `STRIX_MMATTN_WMMA_SERIES`; build line in the file header).

**Tiled core and integer mainloops** ([`src/core/tiled_core.hpp`](../../src/core/tiled_core.hpp)). The library
WMMA prefill core, header-only so that engines instantiate it with their own adapters instead of keeping private
copies. Its integer mainloops (`integer_mainloop.inc`, `integer_grouped_mainloop.inc`) and the register / LDS
schedule (`integer_grouped_schedule.hpp`) are shared; `integer_plan.hpp` holds the plan flags.

**Small-row GEMV** ([`src/core/gemv_rows.hpp`](../../src/core/gemv_rows.hpp)). The decode GEMV for 1 to 16
activation rows that share one sweep over the weights. Two sweeps are shared as textual bodies (`.inc`); a kernel
includes one and binds its operand, lane sum and epilogue. `gemv_policy.hpp` selects the kernel form.

**Row layout** ([`src/core/row_layout.hpp`](../../src/core/row_layout.hpp)). The physical row stride of row-major
device buffers. A kernel that walks many rows in lockstep touches addresses one stride apart, so the stride rule
is set once here ([row layout](prefill/row-layout.md)).

**EXL3 trellis** ([`trellis_format.hpp`](../../src/core/trellis_format.hpp),
[`trellis_codec.hpp`](../../src/core/trellis_codec.hpp)). The packed-weight format (pure C++, host readers) and the
one device operand source every engine reads trellis weights through ([trellis codec](kernels/trellis-codec.md)).

**Engine profiles and the FAST bundle** ([`include/strix/engine_profile.hpp`](../../include/strix/engine_profile.hpp),
[`include/strix/fast_profile.hpp`](../../include/strix/fast_profile.hpp)). One switch `--profile
default|fast|quality` selects every lever's value; each lever is one row of the engine's `*_profile_table.hpp`.
`fast_profile.hpp` is the shared FAST prefill bundle: one switch (for example `STRIX_DS41_FAST_PREFILL=1`,
`STRIX_MM_FAST_PREFILL=1`) selects named member levers, and a set member variable wins over the bundle. Its two
readers, `lever()` on the device side and `expand()` in the host resolver, share the same rules.
[`include/strix/lever.hpp`](../../include/strix/lever.hpp) states a lever as data (`Decl`: switch, exactness,
default, per-regime rule) with `select` as the one decision rule.

**Grouped MoE core** ([`src/moe_core/grouped_core.hpp`](../../src/moe_core/grouped_core.hpp)). The grouped
expert prefill core. No launch policy lives in it: adapters keep their kernel signatures and row policy.
`GroupedPrefillPolicy<Xor64, PipeTail>` ([`grouped_policy.hpp`](../../src/moe_core/grouped_policy.hpp)) chooses the
LDS storage (XOR64 or padded) and the tail-rows policy `kPipeTail`: partial row blocks of the pipelined item run on
`item_pipe` with a row guard on the store, bitwise, instead of the scalar core.

**MoE decode and routing** ([`src/moe_core/decode.hpp`](../../src/moe_core/decode.hpp),
[`route.hpp`](../../src/moe_core/route.hpp), [`route_prune.hpp`](../../src/moe_core/route_prune.hpp),
[`device_plan.hpp`](../../src/moe_core/device_plan.hpp)). Decode schedules share a family; traits keep the
engine-specific K chains, split materialization and epilogues. `route.hpp` is the serial k-pass selection (the
policy owns score transforms, ties and normalization). `route_prune.hpp` is the FAST route prune that zeroes
negligible assignments before the route plan is built. `device_plan.hpp` is the sync-free device route planner and
item queue.

**Attention cores** ([`src/attn_core`](../../src/attn_core)). `sparse_flash_keysplit.hpp` is the FAST sparse
attention core in key-split form (per-row key lists made of segments; with `Split = true` queries and
probabilities are carried as two terms, hi + lo; the four per-wave states are merged once, in wave order).
`sparse_flash_wmma.hpp` is the per-row key-list form, `dense_flash_prefill.hpp` the FAST dense causal GQA prefill,
`selected_flash_prefill.hpp` the selected-key schedule, `banded_flash_prefill.hpp` the paired-key schedule.
`index_score_wmma.hpp` is the FAST indexer score core and `index_score_band.hpp` keeps an exact top-k on top of
approximate scores. All of them share `flash_tile.hpp`, `flash_core.hpp` and `online_softmax.hpp`.

**Pair transport** ([`src/xport`](../../src/xport), [`include/strix/pair`](../../include/strix/pair)). `channel.hpp`
is the shared TP2 staging and progress engine; the engine-specific wire ABI lives in its `Traits`.
`staging.hpp` is the ticket / lease core, `framed_exchange.hpp` the framing of one checked exchange.
`include/strix/pair` holds the route policies: link setup (`link.hpp`), the two-stage pipeline seam (`pipe.hpp`),
stage balance (`balance.hpp`), per-request mode selection (`regime.hpp`), the prefill group (`prefill_group.hpp`),
the lockstep control channel (`lockstep.hpp`) and the media embedding frame (`embed_frame.hpp`, `pair::EmbedFrame`).

**Session layer** ([`include/strix/session`](../../include/strix/session), [`src/session`](../../src/session)).
Committed model input, KV snapshots, the prefix cache (adapters supply the identity of weights, numerical profile,
state format and rank topology; unknown fields are refused), embedding rows in the token stream (`embed_rows.hpp`)
and their splice into a prefill run (`embed_splice.hpp`, `embed_splice.hip`).

**Serve layer** ([`include/strix/serve`](../../include/strix/serve), [`src/serve`](../../src/serve)). The API front
doors (`api.hpp`, `http_server.hpp`), one normalizer for media parts (`media.hpp`), the adapter shape of a served
engine (`run.hpp`), prefill interleaving between decode rounds (`prefill_interleave.hpp`), which short prompts
share one prefill pass (`prefill_group.hpp`), the prompt-encoding memo (`token_memo.hpp`) and the per-cycle and
per-run receipts (`cycle_receipt.hpp`, `RunReceipt`). `include/strix/serve_cli_parity.hpp` is the parity gate: a
family resolves one request through the server and the CLI and `compare()` fails on any difference that is not
named.

**Vision runtime** ([`include/strix/vision`](../../include/strix/vision), [`src/vision`](../../src/vision)).
Preprocessing, one data-driven ViT encoder, the bridge to the API media layer and an LRU of projected image rows
(`EmbedCache`). A family supplies a `VisionAdapter` ([vision adapters](vision-adapters.md)).

**Speculative drafting** ([`include/strix/spec/drafting.hpp`](../../include/strix/spec/drafting.hpp)). Context
proposals, one arbiter and one stats line for every engine; sources are `serial`, `lookup` and `model`.

**Quality gate** ([`tools/strix-gate`](../../tools/strix-gate/README.md)). `gate.py` runs scoped A/B jobs and
compares evidence; `fast_logit_gate.py` is the FAST logit gate. Its `--dnll-rule floor` sets the dNLL limit
relative to null runs and is allowed for DeepSeek V4.1 FAST prefill only.

## How an engine binds a core

Example: DeepSeek V4.1 binds the FAST Q8_0 x A8 dense policy.

1. The policy is a row of the engine's FAST bundle:
   [`include/strix/ds41_fast_profile.hpp:18`](../../include/strix/ds41_fast_profile.hpp) declares member
   `STRIX_DS41_FAST_DENSE` = `1` as lever `ds41-dense-f16-v1`, so `STRIX_DS41_FAST_PREFILL=1` (or the member alone)
   turns it on through the shared rules of `fast_profile.hpp`.
2. The engine includes the shared policy:
   [`src/hip_ds41_layer.hip:2`](../../src/hip_ds41_layer.hip) `#include "core/f16_dense_q8a8.hpp"`.
3. [`src/hip_ds41_layer.hip:3227`](../../src/hip_ds41_layer.hip) documents the lever; `fast_dense_on()` (line 3234)
   reads it through `fast_profile::dense_on()`.
4. `enqueue_fast_dense` (line 3238) fills a `q8a8::View` (line 3241) from the engine's own
   `Q8A8DensePrefillView`: weight, codes, deltas, row bytes, rows, strides.
5. Line 3247: calls under the minimum row count, or shapes the policy does not take (`q8a8::supported`), return
   false and keep the exact kernels. Line 3248: `q8a8::enqueue_parts` launches the shared kernel.

The engine adds no kernel: it maps its buffers onto the policy's view and keeps the exact path as the fallback.
MiniMax binds the grouped core the same way:
[`src/minimax/moe/mmx_prefill_core.hpp:16`](../../src/minimax/moe/mmx_prefill_core.hpp) instantiates
`GroupedCore` with its own weight decode, layout, input transform, epilogue and traits, and
`GroupedPrefillPolicy<false, Tail>` for the tail-rows policy (off by default).

## Adding a new user of a core

Bind the existing core through its policy, adapter or traits; do not copy a kernel into an engine. A new model on
an existing family needs only its description and registry row
([adding a model](adding-a-model.md)); a new architecture is described in its
["New architecture"](adding-a-model.md#new-architecture) section. A default stays bitwise; anything that changes
bits ships under a named FAST policy and passes the [quality gate](../../tools/strix-gate/README.md)
([toggles: profiles](../toggles.md#profiles-exact-vs-fast)).
