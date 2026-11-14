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

## Exactness

Scheduling only. Every emitted token comes from the request's own target rows, so token ids equal those of a
run with `STRIX_SERVE_ROW_BUDGET=0`.

## Observing it

With `STRIX_SERVE_RECEIPT=1` the log carries `serve lever name=row_budget ... active=1` and the per-cycle rows
(see [cycle-receipt.md](cycle-receipt.md)).
