# Serving over HTTP

[English](serving.md) · [Español](serving.es.md) · [Main README](../README.md)

Every engine ships a server. This page is the entry point: which server each engine has, what it does, and how to
expose it safely. The measured serving results stay on the model pages; the request contracts, options and gates are
in [docs/engine/serving/serving.md](engine/serving/serving.md).

| Engine | Server | Serving features | Measured results |
|---|---|---|---|
| Qwen3.8-Flash-Next | `strix-hip-decode-token --http-serve` | OpenAI/Anthropic-compatible chat, completions and stateless Responses routes; request slots sized from memory (`--http-batch N`); automatic tiled prefill (T8192 first, smaller tiles under memory pressure); decoding keeps streaming inside long prompt blocks and short requests nest into a paused long prompt; certified prefix state cache; constrained JSON, thinking budgets, logprobs, logit transforms, SSE progress; QSA MTP pool on the single runner; two replicas behind the TensorMeld router | [Qwen page](models/qwen3.8/README.md#how-fast) |
| DeepSeek V4.1 Flash pair | `strix-deepseek41-serve` (one per rank) | chat, completions and Responses routes; server warm-up and DSpark calibration; Engram lookahead between prefill pieces; bounded agent prefix retention; deferred decoder cone for long prompts; `--profile default\|fast` | [DeepSeek page](models/deepseek-v4.1/README.md#how-fast) |
| DeepSeek V4.1 Flash, standalone RTX | `strix-cuda-ds41-replica --serve SPOOL` (persistent file-queue worker) | each request bitwise identical to a fresh process; residency seed and graph pre-capture; session cache | [DeepSeek page](models/deepseek-v4.1/README.md#deepseek-standalone-rtx-results) |
| MiniMax M3.1 Flash | `strix-minimax-serve` | greedy decoding; structural DSpark auto admission; deadline resumes, short-first and nested admission; named sessions (`context_length_exceeded` on overflow) | [MiniMax page](models/minimax-m3.1/README.md#how-fast) |
| GLM-5.3 Flash pair | `strix-glm5-serve` (one per rank) | chat and completions routes with SSE; continuous batching of up to eight requests, every request's tokens equal the same request served alone; exact MTP at one request, batched MTP verify from two (`STRIX_GLM_SERVE_G2=0` off); decode-first chunked prefill (`--prefill-chunk`, 0 off); exact prompt-prefix cache, 4 GiB per rank by default (`--prefix-cache-gib 0` off) (measured rows: prefix cache off; 8 built-in short chat prompts in the GLM chat template, 22–59 prompt tokens (mean ~33), up to 128 greedy tokens each, barrier start) | [GLM page](models/glm-5.3/README.md#glm-serve) |
| Nemotron 3.5 Bolt E4B | `strix-gemma4-serve` | continuous batching of up to eight chat requests (`STRIX_G4_SERVE_SLOTS`), every request's tokens equal the same request served alone; MTP drafter and prompt lookup at one request; `--profile default\|fast\|quality` | [Nemotron page](models/nemotron-3.5-bolt-e4b/README.md#how-fast) |

Qwen on one node with an explicit MTP sidecar recipe (for example
`recipes/qwen-distill-g-yarn.mtp-bundle.json`): the recipe's target windows need a fixed target
organization (17 flags, including `--virtual-q8-lm-head --prefetch-q8-lm-head`) and a calibration
golden dumped with the tiled command. The server names any missing flag before loading the model.
The two commands, verified on a 124 GiB Strix Halo node with the 111.3 GB UD-Q4_K_XL target, are in
[engine/serving/serving.md](engine/serving/serving.md#one-node-explicit-sidecar-mtp-recipe).

## Exposing the servers

Qwen and DeepSeek publish executed features at `GET /v1/capabilities`; each response's `strix` object
states requested versus effective sampling policy. On every server (Qwen, DeepSeek, MiniMax) unsupported
generation fields return 400 naming the field. MiniMax decodes greedy: a positive temperature is accepted and
reported as ignored in `timings.sampling_note`. Qwen supports constrained JSON exact to its constrained target, thinking budgets, logprobs,
logit transforms and SSE progress. Budgets/transforms are declared token-changing policies;
logprobs/progress preserve tokens. Policy selection uses admitted serial-equivalent MTP windows or
serial fallback; `--http-selection-mtp off` forces the latter. See
[chat request contracts](engine/serving/serving.md#chat-requests) and
[constrained output](engine/serving/serving.md#constrained-json-output).

The HTTP servers (`strix-hip-decode-token --http-serve` and `strix-deepseek41-serve`) listen on loopback unless
`--http-bind` says otherwise. They have no TLS of their own. Authentication is one optional shared key:
with `--http-api-key-file FILE` every API and admin route (completions, `/admin/drain`, `/admin/import-state`)
needs `Authorization: Bearer <key>` or `x-api-key: <key>`; without it the server accepts any client. `/health`,
`/metrics` and `/api/status` answer without the key. Bodies, queued requests and open connections are bounded
(`--http-max-queue`, `--http-max-connections`). The HTTP servers close a request that is not fully received
within 60 seconds (408; plus one second per MiB of body) and treat a client that accepts no data for 60 seconds as
gone, cancelling its generation. Ambiguous request framing (conflicting Content-Length, any Transfer-Encoding,
malformed request line or header names) is refused with 400/501. Request bodies being received or handled share a
256 MiB budget (503 with Retry-After when full), and one request body may hold at most 2,097,152 JSON values. Keep the servers on loopback or a private network and put
remote clients behind an authenticating front end such as the [TensorMeld](https://github.com/hyukudan/tensormeld)
router or its control channel; the two-node DeepSeek ranks exchange tensors over the USB4 link, and the RTX split
(pipeline stage TCP port) and the expert servers talk plain TCP: none of these carry authentication, so they must
stay on a direct cable or a private network.

## Server and CLI resolve the same configuration

A server with one request must run what the engine's CLI runs. Each family has one resolver that both entry points
call; `strix-serve-cli-parity-tests` (host, no GPU) resolves the same plain request through both and fails on any
difference that is not listed in `tests/serve_cli_parity_tests.cpp` with the reason serving requires it, and on any
listed difference that no longer exists. `strix-serve-cli-parity-tests --print` prints both resolved configurations
side by side. Its ordinary output lists included and skipped families; GLM covers complete/missing MTP assets
and explicit opt-out. Gemma compares `parse_options` (CLI) with `prepare_serve` (server). The Qwen pair of entry
points is covered by `strix-qwen-serve-prefill-admission-tests` (the same words with and without `--http-serve`
through `resolve_effective_config`, comparing selected route fields and environment values with field/family
allowances; this does not establish complete configuration or numerical-identity parity).
Gemma's benchmark request contract is tested by `gemma4-session-shared-api`
using the same request fixture as the Python tool.

Removed differences: GLM's automatic max-rows (the server now runs the CLI's load-time admission,
`apps/glm_auto_rows.hpp`, and resolves the 8192-row candidate when both ranks admit it; the 4096-row
`auto-4096-serve` fallback is gone); DeepSeek V4.1's `--tp-attention` / `--engram-direct` (the resolver emits them for
the server too); MiniMax's two host setenv lists (one table, `host.NAME=` lines in `--print-effective-config`).
Differences kept and named for a decision: MiniMax transport (server device exchange, CLI host exchange) and its
prefill MoE table ring (CLI only: qualified with one engine owner); Gemma's stream count (server
`STRIX_G4_SERVE_SLOTS`, CLI 1); Qwen's tiled prefill admitted at serving startup (the CLI resolves it from its
prompt). Scheduling between concurrent requests (slots, per-step prefill limits, peer timeouts) has no CLI
counterpart.

GLM uses the shared prefill load callbacks for candidate admission and installation, including the first-forward
reserve and the engine's two-rank install votes/retries. The synthetic host gate in `tests/pr593_host` checks agreed
rows and installed layouts under reduced and asymmetric memory, including 4096-row fallback. These host checks
do not qualify serving row sizes for native token/logit/state equality; that requires the engine's native gates.

To measure the result, run the server and the CLI on the same build and prompt and report the ratio (exit 1 below
`--min-ratio`, default 0.99, when complete token sequences differ, or when any measured run lacks evidence).
Disable prefix/session caching for full-prefill measurements. The tool arms `/api/status?detail=1`, matches
per-request accounting by response ID, and rejects missing counts or any cached/imported tokens; its prefill
rate uses computed tokens only. Servers without that accounting cannot qualify full-prefill parity here.
Use raw prompt IDs on `/v1/completions`; for chat, render once externally and use those same IDs for both
entry points. The tool does not benchmark chat rendering. Use the actual served model name and select
`--family gemma` for its supported `strix.generated_ids` receipt (other families request `return_token_ids`).

```
python3 tools/serve_cli_ratio.py server --url http://127.0.0.1:PORT --prompt-ids IDS.txt --out server.json
python3 tools/serve_cli_ratio.py cli --family qwen --out cli.json cli-run1.log cli-run2.log cli-run3.log
python3 tools/serve_cli_ratio.py report server.json cli.json
```

For CLI logs without complete generated IDs, append `--output-ids RUN1.ids RUN2.ids RUN3.ids` to the `cli`
command, one whitespace/comma-separated generated-ID file per log. Missing IDs fail even when timings pass.
DeepSeek CLI logs provide decode timing only: explicitly use `report --decode-only`, which makes no prefill
parity claim. `--skip-decode` gates prefill only. Cache-reuse throughput requires a separate benchmark with
matched initial state and a defined reuse metric.

## Further reading

- [Serving over HTTP](engine/serving/serving.md): request contracts, batched decoding, default prompt prefill,
  constrained output, measured scheduling policies.
- [Conversation keep](engine/serving/conversation-keep.md): kept lane states for follow-up turns.
- [Prompt token memo](engine/serving/token-memo.md): a turn that resends its history encodes only the new tail.
- Queued prefix protection (Qwen server, `STRIX_PREFIX_QUEUE_PROTECT=1`, default off): a waiting request holds the
  longest compatible cached prefix state so eviction cannot drop it before the request is admitted.
- [Prefix state cache](engine/serving/prefix-state-cache.md) and
  [serving prefill and prefix-cache composition](engine/serving/serving-prefill-prefix-cache.md).
- [DeepSeek prefix cache qualification](engine/serving/ds41-prefix-cache-qualification.md),
  [DSpark serving auto admission](engine/serving/dspark-auto-qualification.md),
  [MiniMax session cache](minimax/session-cache.md), [DeepSeek RTX session cache](models/deepseek-v4.1/rtx-session-cache.md).
- [Several requests in one decode loop](engine/scheduling/multi-request-cohort.md),
  [sampled decoding](engine/decode/sampled-decoding.md), [release assets](operations/release-assets.md),
  [startup preflight](operations/startup-preflight.md), [service health](operations/service-health.md).
