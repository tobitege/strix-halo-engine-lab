# Serve cycle receipt

`STRIX_SERVE_RECEIPT=1` makes the batched Qwen MTP server account for every decode round in the shared
receipt of `include/strix/serve/cycle_receipt.hpp`. The switch is off by default; unset, the server reads no
extra clock and prints no extra line.

## What is emitted

One line per decode round on the server log:

```
serve cycle seq=N lanes=L rows=R useful=U pad=P emitted=E draft_rows=a/b/c/d/e/f/g/h draft_ms=.. verify_ms=.. host_ms=.. ms=.. prompt=0|1 begin_ns=.. end_ns=.. admitted=A forwards=F solo=s/n/w/p/a/o
```

| Field | Meaning |
|---|---|
| `lanes` | Requests that ran target rows in the round. |
| `admitted` | Requests holding a slot when the round ran (decoding or still reading their prompt). |
| `forwards` | Target execution counter (0 = not reported by the engine). The current Qwen adapter counts shared windows plus solo lanes, so serial multi-step verification and rejected-window replay can be undercounted. Do not use those cycles as dispatch evidence. This counter does not measure weight traffic; operations can use different dispatch and fusion routes. |
| `solo` | Lanes that verified in a forward of their own, by reason: `single` (only decoding lane), `no_rows` (no window rows, e.g. no draft), `wide`, `padding`, `alone` (only lane of its group or chunk), `no_owner`. |
| `rows` | Physical verify rows: a lane's rows of a shared window, its own window owner, or the serial rows it consumed. |
| `useful` / `pad` | Rows that carried the pending row or a drafted candidate, and the rest. `useful + pad = rows`. |
| `emitted` | Tokens committed to the lanes by the round. |
| `draft_rows` | Draft rows by depth: entry `d` counts the lanes whose draft reached depth `d + 1`. |
| `draft_ms` | Wall time spent proposing drafts. |
| `verify_ms` | The rest of the runtime's cycle: target verification and selection. |
| `host_ms` | The rest of the round: applying results and delivering tokens. |
| `ms` | Wall time of the round. A prompt unit of the same scheduler step is not included. |
| `prompt` | 1 when a prompt unit ran in the same scheduler step: the round is outside the steady-state window. |

At the first round the server prints one activation line per serve lever, with the resolved state rather than
the raw environment:

```
serve lever name=cohort_rows env=STRIX_COHORT_ROWS active=0 value=0
```

When the batch server returns it prints `serve ledger <json>`. The same object is served by the status route
under `cycle_ledger` while the switch is on and at least one round was recorded. It carries the cumulative
totals, the per-lane-count table of rounds without a prompt unit, and the least-squares fit
`fixed_ms + row_ms x rows + lane_ms x lanes`.

## Reading it

`tools/serve_bench.py` computes the steady-state aggregate rate from these lines (`--log`) or from two ledger
snapshots (`--status-path`), never from client timestamps:

```
python3 tools/serve_bench.py --url http://127.0.0.1:PORT --chat --prompt-file prompt.txt \
    --concurrency 1,4,8,16 --max-tokens 128 --receipt --log server.log --out result.json
```

The receipt is read-only: it does not change scheduling, drafting or selection, so token ids are those of a
run without it.
