# Prompt groups: several requests' prompt blocks in one prefill run

On by default (`STRIX_PROMPT_BATCH=0` turns it off) for the batched tiled MTP route (`--http-batch N`, cohort requests on the tiled
prompt with `STRIX_BATCH_INTERLEAVE=1`). Without it every prompt unit runs alone, as before.

## What runs together

A request's prompt runs as a sequence of units: T-row blocks, tail blocks (`--known-prefix-tail-rows`), then serial
rows. The units of one request and their boundaries do not change. When the oldest running prompt's next unit is a
tiled block, the next blocks of the other running prompts join it, oldest first, while their rows fit the row
budget: the rows of the T-row block owner, or `STRIX_PROMPT_BATCH_ROWS` when it is smaller. With
`STRIX_PROMPT_BATCH_ROWS` above T, setup builds one more tiled owner of that many rows (its own arena) that only runs
groups. A group of two or more blocks runs as one segmented prefill; a lone block or serial rows run alone. The
merged ladder head of a prompt shorter than T (a partial block: the first rows of the prompt's T owner, a multiple
of 128) is a block like the others and joins a group as one segment (`STRIX_PROMPT_BATCH_PARTIAL=0`: it runs alone,
as it did before; the first and largest unit of every prompt below T rows was then never grouped). The
decode cycle of the requests past their prompt follows each prompt step as before (prefill and decode are not fused).

Group size by load (`STRIX_PROMPT_BATCH_AUTO`, default on; `=0` off): a group takes at most 4 blocks, so the first
prompts of a burst finish sooner, unless every slot is busy and at least `STRIX_PROMPT_BATCH_AUTO_QUEUE` requests
(default 1) wait for one (backlog): then a group fills the whole row budget as before, so throughput at saturation is
unchanged. Scheduling only: a block's arithmetic is the same in any group, so tokens are identical. An explicit
`STRIX_PROMPT_BATCH_ROWS` keeps a fixed row budget and turns the load rule off. Measured (One Strix Halo node, 8 slots, 4 output tokens, 24 paired bursts per cell (fresh server per arm, order alternating over six turns), on vs STRIX_PROMPT_BATCH_AUTO=0): 8 x 1024 burst: first-token p50 8.62 vs 10.25 s (-21 %), mean completion 9.19 vs 11.22 s (-23 %), makespan +1.2 % (95 % CI -1.3 to +6.0 %, not resolved); 16 x 1024 burst (8 queued): first-token p50 12.63 vs 14.58 s (-15 %), mean completion 14.52 vs 15.51 s (-7.7 %), makespan +0.2 % (95 % CI -1.5 to +1.8 %). Token ids equal in every pair. A higher threshold (3 or 9 queued requests) did not shorten 10/12/16 x 1024 bursts. With `STRIX_PROMPT_BATCH_LOG=1` each
group line ends in `idle cap 4` or `backlog`.

## Segmented prefill

`execute_known_prefix_prefill_segments_eager` runs the blocks of several sessions over consecutive rows of one tiled
arena. Segment s owns rows `[offset_s, offset_s + rows_s)` (multiples of 8) with its own session, state memory,
positions, tokens, PLE lookup and history, terminal head and lineage. Every step runs in the step-major order of a
tiled block:

- Steps whose arithmetic per row depends only on that row's input and the weights run once for the rows of every
  segment through their full-T entry, inside a cohort scope (per-segment session and position rules): HC mixers,
  router, routed expert union (one CSR over all rows; each (row, output) sums in slot order whatever batch it lands
  in), inject and MoE combine. `STRIX_PROMPT_BATCH_SHARED` lists them (`hc`, `router`, `moe`, `elementwise`,
  `shared` for the shared expert's dense projections, `all`, or `none`; default `hc,router,moe,elementwise`).
- Every other step runs per segment exactly as that segment's block alone: the same entry, row count, tile
  boundaries and order (GDN projections and recurrence, QSA projections, KV, selection and attention, PLE, the
  shared expert, row loops and the terminal head of the segment's last row).

Each segment is admitted as its block alone before anything is enqueued; a failure after that poisons every
segment's session. The lineage of a segment records the producer of the session's own block of that size (the
per-row arithmetic it repeats). The routed union of a group of R < T rows uses its own CSR workspace of R rows.

Exactness: every request's state after each block, its terminal logits, the predictor rows and every generated
token are those of the request alone (`STRIX_PREFIX_CACHE_DIGEST=1` adds a SHA-256 of the final prompt row's logits
to the prompt receipt next to the state digest).

## Switches

| Switch | Effect |
| --- | --- |
| `STRIX_PROMPT_BATCH=0` | Run every prompt unit alone (default: group the running prompts' blocks) |
| `STRIX_PROMPT_BATCH_AUTO=0` | Always fill the row budget (default: at most 4 blocks per group unless requests wait for a slot) |
| `STRIX_PROMPT_BATCH_AUTO_QUEUE=N` | Queued requests that count as backlog for the load rule (default 1) |
| `STRIX_PROMPT_BATCH_ROWS=N` | Row budget of a group; above T, a dedicated group owner of N rows (multiple of 8, up to 8192) |
| `STRIX_PROMPT_BATCH_PARTIAL=0` | Partial blocks of the T owner run alone (default: they join groups as segments) |
| `STRIX_PROMPT_BATCH_LOG=1` / `2` | One stderr line per group; `2` also one line per prompt unit that found no group, with the reason |
| `STRIX_PROMPT_BATCH_SHARED=list` | Families run once for all segments (default `hc,router,moe,elementwise`) |

Receipts: each request's tiled prompt receipt counts `grouped_blocks`; the cohort receipt adds `prompt_batch`
(groups, grouped blocks and rows, group and prompt time).

## Serial suffix as known-row windows

After its tiled blocks a prompt has (n-1) mod 16 rows left (fewer than the smallest tail block). They used to run
one serial row per step (~47 ms each on Qwen). With `STRIX_PROMPT_WINDOW` (default on, `=0` restores the serial
rows) they run as known-row windows of 8, 4 and 2 rows on the request's verify-window owners (an odd row first runs
serially): each window row's arithmetic is the serial row's (VERIFY_WINDOW_PLAN), every row is committed, and the
predictor rows, held row and pending ticket are those the serial rows leave. Greedy requests only; a sampled request,
or a window no owner admits, keeps the serial rows. With `STRIX_PROMPT_WINDOW_COHORT` (default on, `=0` runs each
prompt's windows alone) the running prompts' next windows run in one stream of steps
(`HipMtpWindowVerifier::prompt_rows_cohort`, chunks of up to eight rows sharing the weight-only steps as decode
cohorts do); each prompt's window is the one it runs alone.

Measured (Qwen one node, one replica `--http-batch 8`, tokens == each request alone):
wall tok/s chat C1 67 -> 71, C4 77 -> 118, C8 76 -> 125; 8-category set C4 61 -> 101.
