# Several requests in one decode loop (cohort windows)

Opt-in CLI benchmark mode for aggregate throughput with several independent requests. Every request
produces exactly the tokens it produces alone; the receipt reports aggregate and per-request rates.

```
strix-hip-decode-token MODEL.gguf TOKEN_ID ... (request 0: positional prompt, --mtp-golden ...)
  --mtp-cohort-request TOKENS_FILE,GOLDEN_FILE,GOLDEN_SHA256   (requests 1..3, repeatable)
  [--mtp-cohort-mode lockstep|sequential]                      (default lockstep)
```

On the serial prompt route (the default) `TOKENS_FILE` holds decimal token IDs separated by commas or whitespace;
`GOLDEN_FILE` is that prompt's MTP
golden under the run's numerical profile (dumped with `--mtp-golden-dump` like any other golden). The rest of
the command line is the ordinary speculative configuration of the README; the cohort requires prefix windows
over the row set `2,4,8`, the draft chain and one numerical profile for prompt and decode (no
`--prefill-numerical-profile`). At most four requests in total on the serial prompt route (eight on the tiled
prompt, below); `--generate` applies to each request.

## Tiled prompt (batched route)

On the tiled MTP prompt (`--known-prefix-window --known-prefix-full-rows --known-prefix-prefill-rows T`, the serving
route) cohort requests take `--mtp-cohort-request TOKENS_FILE` without a golden: the predictor is qualified once by
the calibration replay of the positional prompt's first `--mtp-calibration-rows` pairs, exactly as for the server,
and every request then runs its whole prompt through the tiled prefill (T-row blocks, tail blocks, serial rows) like
a served request. Up to eight requests in total (seven besides the positional prompt); prompts are bounded by the
context only. `--mtp-sample-*` applies to every request (request i uses seed + i).

Each request has a slot (see [serving.md](../serving/serving.md#batched-decoding)): its own state, session, launch plan, PLE
backend, tiled prefill owners and backends (the prefill arenas are shared and used by one prompt at a time), prefix
cache view, predictor, target producer and window owners on one shared window stream. `lockstep` runs every
request's prompt (one after another) and then decodes them together; `sequential` admits each request after the
previous one ended, each exactly as alone. With more than four requests a cycle forms two or more cohort windows (two
rows per request, groups of four requests per 8-row window).

The receipt (`mode` `native-mtp-cohort-tiled-prompt-target-verified-greedy|sampled`) has the fields of the serial
cohort receipt plus each request's tiled prompt receipt, its prompt time and restored tokens; the replay errors are
those of the shared calibration replay. `STRIX_BATCH_INTERLEAVE=1` interleaves the prompts with the decode cycles
(then the cycle wall includes the prompts).

## Per-request state

Each request has its own serial scratch and persistent state (KV cache, GDN recurrent and convolution
state, PLE history), PLE lookup backend and prefetch state, launch plan, session, MTP target capture,
D1 record, MTP runtime with its own draft state and draft vocabulary (`auto:DIR` picks the pack from that
request's prompt), and its own 2/4/8 window owners with their deferred GDN record, PLE snapshots and greedy
replies. Weights, the MTP sidecar assets, the manifest and the state layout are shared. Every request runs
its own prompt, independent replay against its own golden (the declared error policy, 0 by default),
qualification and activation before generation, exactly as the single-request route does. In cohort mode
the window owners of all requests use one stream.

## Cycle

`lockstep` runs one cycle of every active request at a time (`hip_mtp_greedy_cohort_cycle`):

1. Each request drafts its chain as in an ordinary cycle, up to the first draft below `--mtp-spec-min-p`
   (the cost-model early stop is not used in a cohort cycle), and at most to the per-request row cap
   minus one: the cap is 8 / requests in the cycle (4 rows with two requests, 2 with three or four), so
   that one cycle fills one 8-row window (`STRIX_COHORT_MAX_ROWS=2|4|8` overrides it).
2. Each request gets the prefix owner the verifier would size for its drafts: 1 + the leading drafts at or
   above the minimum probability, at least 2, rounded up to 2, 4 or 8 and limited to the cap, if that
   owner fits its budget, context room and lineage history. Requests are packed into cohorts of 4 or 8
   rows, larger windows first; a cohort that does not add up is rounded up with padding rows of a smaller
   window (padding rows repeat the last draft and are never accepted), else its last request leaves it. A
   request alone, one that needs eight rows, or one with no fitting owner verifies with its own verifier
   as usual.
3. A cohort runs as one window (`HipKnownPrefixWindow::execute_mtp_cohort`): every request's own window is
   admitted exactly as on its own, then the steps advance in lockstep on the shared stream. Steps that read
   only weights and per-row activations run once for the rows of all requests on a primary owner (an idle
   owner of the cohort's row count): routed expert union, HC mixers, router, LM head (each request's
   argmax rows into its own replies), the shared expert projections (on the primary's side stream, joined
   before the MoE combine, as in one window) and the input and output projections of GDN and QSA. Inside
   the GDN and QSA steps the recurrence and the KV append/selection/attention run per request from that
   request's state (`gdn_window_cohort_enqueue`, `indexed_attention_known_window_cohort_launch`), and the
   shared-expert activation runs per request. Every other step (embedding, PLE, MoE combine, injections,
   the observed last MLP injection) is each request's own launch.
4. Each request then accepts its longest prefix, commits its deferred GDN rows and PLE snapshot, rewinds
   its frontier and publishes its rows through its own target capture, as after its own window.

`sequential` runs the requests one after another, each exactly as alone; it is the in-process reference
for the aggregate rate.

### Global row budget and wide steps

`STRIX_COHORT_ROWS=16|32` replaces the one 8-row cohort per cycle by a row budget. Each request keeps the
rows it has in an 8-row cohort (8 / requests, at least 2; `STRIX_COHORT_MAX_ROWS` overrides), so up to four
requests run as before, and five to eight requests (10 to 16 rows) fill one group instead of two cohorts.
More rows per request (4 rows each at four requests, 8 at two) were measured to cost more than the drafts
they accept. Requests fill groups of at most the budget; a group's requests fill chunks of at most 8 rows
(largest windows first). A chunk of two or more
requests whose rows add up to 4 or 8 runs on an idle owner of that row count that no request of the group
verifies with (its primary, as an ordinary cohort); any other request runs as a chunk of its own with its
own window. The chunks of a group advance step by step on the shared stream
(`HipKnownPrefixWindow::execute_mtp_cohort_chunks`): step i of every chunk before step i + 1 of any. No
padding rows are added. With chunks only, every chunk still reads the weights of its shared steps.

`STRIX_COHORT_WIDE=gdn,qsa,head,moe` (or `all`) additionally runs the listed steps once for the rows of all
chunks when they add up to 9..32 rows (the dense projections pad to 16 or 32 rows with rows that read the
first row's input and write only a sink buffer no row reads):

| Step | Wide form | Per-row arithmetic |
| --- | --- | --- |
| GDN layer (`gdn`) | Input and output projections as one 16- or 32-row dense A8 launch (16-row WMMA tiles from the same pitch-64 activation slot); prepare and recurrence per session | Dense A8 only (the FP32 Q8 wide forms do not keep the 1..8-row bits): the chain per (row, output) of the 1..8-row form |
| QSA layer (`qsa`) | Input and output projections as one launch; norm, KV append, selection and attention per session | Dense A8 with a Q8_0 output projection only |
| LM head (`head`) | Unstaged launches of at most 16 rows; each request's greedy rows into its own replies | The R1 leaf, lane-tree and local-tree DAG per row |
| Routed experts (`moe`) | One expert union over all rows on the lead request's stream, with a CSR workspace per row count; down in batches of at most 8 assignments | Per (row, slot) accumulators, merge in slot order (A8 profile only) |
| MoE combine (`combine`) | One launch over all rows, after every chunk joins its shared expert (every chunk has a primary) | Elementwise per row |
| Injections (`inject`) | One launch over all rows | Elementwise per row |

HC mixers, router and shared expert stay per chunk: their weights per layer (about 5-7 MB) are re-read
from the MALL by the second chunk.

Inside the cohort GDN and QSA launches the per-session parts are fused across sessions too: the GDN
prepare and recurrence of every session run as one launch each (`blockIdx.y` = session, each running the
per-session body for its 2, 4 or 8 rows; `STRIX_COHORT_GDN_SEGMENTS=0` keeps one pair per session). With
`STRIX_COHORT_QSA_SEGMENTS=1` the state-free QSA sub-parts (norms and rope, block selection, softmax) also run
once for every row while the KV append, block scores, attention scores and value/gate output stay per session
(measured neutral, so off by default). The shared expert activation and scalar gate, formerly one launch
per session and layer, run as one launch over per-row addresses. The draft chains of a cohort cycle advance
depth by depth: step k of every session (each on its own stream) is enqueued before any session waits for
its step (`STRIX_COHORT_DRAFT_INTERLEAVE=0`: one chain after another). A wide union's status covers all its rows: a failing row fails the
whole cohort window, as in an 8-row cohort.

### Automatic per-cycle policy (default)

With `--mtp-cohort-request` each cohort cycle picks its form from the total rows its requests verify:

| Rows in the cycle | Form |
| --- | --- |
| up to 8 | One 8-row cohort, as before |
| 9 to 15 | Wide steps; the routed experts run as one 16-row union |
| exactly 16 | Wide steps; the 16-row expert projection stages its activations 4 rows at a time |

Chunks hold up to 16 rows and a request is never split across chunks. Every form keeps each row's
arithmetic (see Exactness), so each request's token ids are identical to its single-request run. Measured on
Qwen with the MTP drafter, aggregate decode: 8 requests 69.3 tok/s (62.9 tok/s with the previous defaults,
+10 %); 1, 2 and 4 requests unchanged. `STRIX_COHORT_AUTO=0` restores the previous defaults; a switch that is
set explicitly (`STRIX_COHORT_ROWS`, `STRIX_COHORT_WIDE`, `STRIX_COHORT_STEP_ROWS`, `STRIX_MOE_R16_PROJECT_G`)
takes precedence over the policy.

## Exactness

The shared steps perform each row's arithmetic exactly as that row's own window does (per-row pointers,
the same kernels and reduction trees; the multi-row kernels are row-count independent per row). The dense
A8 output codes of QSA are written per request at the request's rows of one slot whose layout does not
depend on the row count. Fixtures compare cohorts byte for byte against each session's own window:
`strix-hip-gdn-deferred-window-tests` (sessions with their own state, pending rows and commits; cohorts of
2 to 4 sessions, 2 to 8 rows; dense A8 cohorts of up to 8 sessions and 16 or 32 rows),
`strix-hip-known-window-tests` (QSA cohorts of two to eight sessions at mixed positions, FP32 and dense A8,
16 and 32 rows with dense A8; LM head rows of 16 and 32 from several arenas against R1 heads) and
`strix-hip-moe-known-rows-tests 16|32 1 a8q51|a8q80|a8q5k` (routed unions over several 8-row arenas against
the serial A8 rows, including experts shared by every row). On the full model each request's generated
tokens are compared with its own single-request run.

## Switches

| Switch | Effect |
| --- | --- |
| `--mtp-cohort-request` | Enables the mode; absent, every route is unchanged. |
| `--mtp-cohort-mode sequential` | Requests one after another (reference). |
| `STRIX_COHORT_SHARED=0` | Lockstep cohorts with every step launched per request (no shared steps). |
| `STRIX_COHORT_SHARED_EXPERT=0` | Shared steps as above except the shared expert, which runs per request. |
| `STRIX_COHORT_MAX_ROWS=2\|4\|8` | Per-request row cap in a cohort cycle (default 8 / requests; with a budget, budget / requests). |
| `STRIX_COHORT_ROWS=16\|32` | Global row budget per cycle: chunks of up to 8 rows in one stream of steps. |
| `STRIX_COHORT_WIDE=gdn,qsa,head,moe,combine,inject\|all` | With a budget: the listed steps once for all chunks' rows (9..32). |
| `STRIX_COHORT_GDN_SEGMENTS=0`, `STRIX_COHORT_DRAFT_INTERLEAVE=0` | Per-session GDN prepare/recurrence; one draft chain after another. |
| `STRIX_COHORT_SHARED_ACT_MERGED=0` | Shared-expert activation and scalar gate as one launch per session instead of one launch for the rows of every session (default; bitwise equal per row, measured 5.4 ms per cycle at 8 sessions, 1 ms at 2). A/B name `sharedact`. |
| `STRIX_COHORT_QSA_SEGMENTS=1` | State-free QSA sub-parts once for all rows (neutral). |
| `STRIX_COHORT_AB=1`, `STRIX_COHORT_AB_OFF=gdnseg,qsaseg,combine,draft,gdn,...` | Measurement: odd cohort cycles run without the listed features (wide families by name); mean cycle ms, tokens and tok/s per parity at exit. |
| `STRIX_DENSE_A8_WMMA_SMALL=1` | Opt-in: dense A8 Q8_0 projections of 2..8 rows (shared expert, GDN, QSA and PLE producers) run a WMMA tile sized for the real rows instead of the dot4 form; bitwise equal per row. A/B name `densesmall`. |
| `STRIX_COHORT_QSA2_MERGED=0\|1` | QSA part 2 of every cohort segment in merged launches (any segment sizes; bitwise equal). Default: automatic, merged with three or more sessions (paired, per cycle: -0.7 ms at three, -2.2 at four, -4.0 at six, -8.0 at eight) and per session with two (merged costs 1.6 ms more there). `1` forces it, `0` turns it off. Off while a QSA pool check switch is set. A/B name `qsa2`. |
| `STRIX_COHORT_GDN_ANYSEG=1` | Opt-in: GDN part 2 (prepare and recurrence) fused for segments of 1..8 rows, not only 2, 4 or 8; bitwise equal. A/B name `anyseg`. |
| `STRIX_COHORT_AUTO=0` | Turns the automatic per-cycle policy off: one 8-row cohort per cycle unless the switches above are set. |
| `STRIX_COHORT_WIDE_LOG=1` | Per cohort call (first 4, then every 256th): chunks, rows and steps run wide, per chunk and per request; at exit, calls and mean wall ms per row total, chunk count and wide or not. |

Each of the three opt-in small-row switches prints one `lever active: <SWITCH> (first launch: ...)` line on stderr
the first time its path launches, so a serve log shows which of them are engaged.

The receipt (`mode` `native-mtp-cohort-...`) reports `mtp_cohort_wall_ms`, `mtp_cohort_generated_tokens`,
`mtp_cohort_aggregate_tokens_per_second`, cohort window counts and, per request, its replay errors, cycles,
acceptance, window histogram, finish time, rate and generated tokens.

## Limits

- Serial prompt route: CLI benchmark mode, at most four requests of up to 4097 prompt tokens each. The tiled
  prompt route serves through the same slots (`--http-batch`, [serving.md](../serving/serving.md)).
- PLE and the elementwise steps run per request (PLE weights are small; elementwise steps read no weights).
- Drafting is sequential per request.
