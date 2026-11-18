# Row budget for the batched MTP decode

The batched Qwen MTP server chooses how many verify rows each decoding request gets in a cycle from a cost
model. `STRIX_SERVE_ROW_BUDGET=0` (or the baseline mode) restores the fixed table of 8, 4 or 2 rows by number
of requests.

## What it does

The cohort cycle runs through the joint window controller with the rule `budget`
(`STRIX_MTP_JOINT_PLAN=budget` selects the same rule by name). The rule hands the shared planner
`strix::serve::plan_rows` (`include/strix/serve/row_budget.hpp`) one demand per request:

- at least two physical rows, at most the request's window owner (2, 4 or 8 rows);
- the chance that each further row yields a token: before drafting the 0.8 prior, after drafting the draft's
  own probability, both blended with the request's observed acceptance by depth;
- a cycle cost of `fixed + row x rows + lane x requests` milliseconds.

The planner buys rows while they add more expected tokens per millisecond than the cycle already has, inside a
budget of 32 rows per cycle. Each request then keeps the largest owner inside its planned rows and moves to the
next owner while the budget holds it and the added rows still pay. Confident drafts get deep windows, doubtful
drafts stay at two rows, and rows per request fall as the requests of a cycle grow.

## Cost model

The priors are 28 ms fixed, 6 ms per row and 6 ms per request (one request: 34 + 6 x rows). Once the measured
cycles cover topologies in which rows and requests moved independently, their least-squares fit replaces the
priors. `STRIX_SERVE_ROW_COST=fixed,row[,lane]` pins the model.

The budget plans cycles of up to `STRIX_SERVE_ROW_BUDGET_MAX_LANES` decoding requests (default 6, range 0 to 8).
Above the bound every request gets the fixed table's two rows. The bound is measured: the budget is faster at
two to six requests and slower at eight, where its deeper rows cost more than they yield.

Above the bound the joint cycle groups requests by the row budget instead of a fixed count of eight: by default one
group (one logical target execution, one weight read) takes up to 16 requests (32 rows, the widest shared step), so
9 to 16 requests run one forward per cycle and 17 to 24 two. `STRIX_SERVE_JOINT_GROUP_LANES=8..16` sets the cap
(8 = the old form, one forward per eight requests). Each wide cohort step spans up to `kCohortWideMaxSessions` (16)
sessions; every wide launcher's session table holds that many (the GDN prepare/recurrence table holds eight, so it runs
once per eight sessions, each block on its own session's state). An earlier nine-request stall was a launcher that
accepted only eight sessions. Measured on one node, nine requests: 171.8 against 211.8 ms per cycle, the same token ids
per request as the old form. `ab` alternates 8 and 16 by cycle (in-process A/B: the receipt's `forwards` field tells
the arms apart). Scheduling only: the 9..32-row wide steps keep every row's arithmetic. Coverage: groups of 10 to 16
live requests were not reached on the measured node (nine slots at 8K context); their token ids at 16 submitted
requests match the old form, and the session split rests on the per-session state ownership above.

Measured topology pricing (`STRIX_SERVE_ROW_MEASURED`, default `auto`): at the lane counts of the decision table
(`kMtpRowMeasuredAutoLanes`, now 7 requests) the budget prices each owner topology (how many requests hold 2, 4 or 8 rows)
by its own measured cycle time (three samples or more) and its delivered tokens, and takes the assignment with the most
expected tokens per millisecond (an exact search over owner counts). It also runs the bounded probes above the lane bound,
so 7 requests can leave the fixed two rows when four rows each measure faster per token. `1` uses it at every count 1-8, a list such as `7,8` at those counts,
`0` keeps the affine fit everywhere. Joint rule texts for in-process A/B: `budget-measured` (all counts), `budget-fit`
(none). `--print-effective-config` / the serve lever list show `row_measured` with the lane mask. The delivered tokens are
compared with the prediction of the complete candidates the plan priced (context rows included), aged over about the
last eight cycles like the timing. Qualification: the table was measured for the served Qwen joint cycle on one device
class; another binding of this controller starts from `0` until its own ABAB receipt.

## Exactness

Scheduling only. Every emitted token comes from the request's own target rows, so token ids equal those of a
run with `STRIX_SERVE_ROW_BUDGET=0`.

## Observing it

With `STRIX_SERVE_RECEIPT=1` the log carries `serve lever name=row_budget ... active=1` and the per-cycle rows
(see [cycle-receipt.md](cycle-receipt.md)).

`scripts/bench/qwen/serve_cycles.py` turns a served log (`STRIX_MTP_BATCH_LOG=2`) and serve_bench results into a per-burst table: decode-window and wall tok/s, emissions per lane, cycle / draft / target ms, the host gap between cycles and each owner topology's measured target ms and emissions; `--pairs A,B` compares in-process A/B arms (ids per request, both rates). Each burst takes the result burst whose live lanes match its `c<N>` name (or `RES.JSON#K`, validated); the id verdict is IDENTICAL only with both arms, no failed request, equal request counts and a full non-null hash per request (else UNAVAILABLE, exit status 1).
