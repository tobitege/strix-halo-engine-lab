# Exchange row fold (CPU fold of the one-row all-reduce)

In pair decode every sublayer ends with a small all-reduce: each rank publishes its partial row, the exchange
module's progress thread moves it over the link, and a post-exchange kernel adds the two partials before the
norm / expand. With the row fold the progress thread itself adds the two partials (rank 0's first) right after the
peer payload arrives and before it publishes the receive flag; the kernel then reads one folded slot.

- Shared code: `src/xport/row_fold.hpp` (`RowFold::sum_f32`, `fold_rows`), `Channel::fold_next()`,
  `Channel::can_fold()`, `Channel::last_folded()`. Engines only declare the operation before the send.
- Exactness: one binary32 addition per element on both sides, so results are bit-identical. Gates: the CPU test
  `strix-xport-row-fold-tests` and the device test `strix-hip-glm5-post-exchange-rows-tests` (folded kernel forms,
  and host sum == device sum on 64,000,000 raw bit-pattern pairs).
- Only the pair sum can move to the CPU: the norm (MiniMax) and the expand (GLM) need device-resident state, so the
  kernel launch stays.

## Switches

| Variable | Engine | Default | Meaning |
|---|---|---|---|
| `STRIX_MMX_FOLD=N` | MiniMax | 0 (off) | fold exchanges of at most N rows |
| `STRIX_GLMX_FOLD=N` | GLM | 0 (off) | fold boundaries of at most N rows |
| `STRIX_MMX_RX_TRACE=path`, `STRIX_GLMX_RX_TRACE=path` | both | unset | diagnostic: per exchange, when the progress thread saw the producer, finished the link call and published the receive flag (monotonic ns), written to `path.<rank>` at exit |

Both ranks must use the same setting only for comparable timing; the wire format does not change, so mixed settings
remain correct. Rollback: unset the variable (or set it to 0).

## Measurement (two-node pair, one row per step)

Work against wait of the post-exchange kernel (kernel trace joined with the receive trace, rank 0):

| Engine | kernel span | waiting for the peer | after the payload arrived |
|---|---|---|---|
| MiniMax, fold off | 48.3 us | 28.1 us | 20.2 us |
| MiniMax, fold on | 41.1 us | 21.8 us | 19.3 us |
| GLM (both arms) | 24.5 us | 21.2 us | 3.3 us |

The peer payload never arrived before the kernel started, so most of the span is link latency, not work. The fold
adds about 0.4 us to the progress thread and removes about 0.9 us after arrival.

In-process A/B, same tokens, logits and state in every run:

| Engine | off (ms/token) | on (ms/token) | result |
|---|---|---|---|
| MiniMax, 8K context | 44.06 / 44.15 / 44.17 / 44.14 | 44.19 / 44.19 / 44.11 | neutral (+0.03) |
| GLM, short context | 38.16 / 38.54 / 38.16 / 38.20 | 38.17 / 38.12 / 38.13 | -0.03 to -0.05, at the noise edge |

The default therefore stays off; the switch is kept for platforms with a faster link, where the after-arrival part
is a larger share of the token.
