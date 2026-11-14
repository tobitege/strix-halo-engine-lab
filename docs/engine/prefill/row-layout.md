# Row layout policy: physical row stride off page multiples

Row buffers hold one logical row per token. A kernel that walks many rows in lockstep (the same column of 16 or
more rows at once) touches addresses one row stride apart. When the stride is a multiple of 4,096 bytes all of
those addresses share a page offset and the rows of a tile compete for the same cache sets and translation
entries. The values are unaffected; only time is lost.

`src/core/row_layout.hpp` supplies the shared physical stride policy for the known-rows arena and detached
shared-expert workspace. Other row-buffer owners have not adopted it:

- `physical_row_stride(policy, tight, rows, alignment)` keeps the logical width and the buffer's row alignment and
  adds a step to a stride that would repeat a page offset sooner than `min_period` rows. The step is rounded up to
  the buffer's alignment, so 64-byte rows stay 64-byte aligned and the known-rows arena keeps its 256-byte rows.
- Buffers with fewer than 16 rows (closed windows of two, four or eight rows) keep the tight stride.
- Alignments that cannot use de-aliasing keep the tight layout, including page-aligned arenas.
- Consumers take the stride from the row views (pointer + stride). Code that used to recompute the 256-byte rounded
  width now reads the views or calls the policy.

The known-rows arena (`build_known_prefill_arena`) takes every stride from the policy; a caller may pass its own
`RowLayoutPolicy`, otherwise the process policy applies to engine, server and fixtures alike.

## Controls

| Variable | Default | Meaning |
|---|---|---|
| `STRIX_ROW_DEALIAS_BYTES` | 256 | Step added to an aliasing stride (multiple of 64). `0` restores the tight strides. |
| `STRIX_ROW_DEALIAS_PERIOD` | 2 | Rows before a page offset may repeat. 2 moves exact page multiples only; 16 also moves strides such as 5,120 bytes. |
| `STRIX_ROW_LAYOUT_VERBOSE` | off | Prints payload, stride and page period of known-rows arena buffers. |

## Reported measurements (combined code)

These author-reported timings used this layout change together with another PR; the exact combined revision
was not recorded here. They do not establish a speedup for this isolated change and were not independently
reproduced in the host review. The GDN fixture hash covers final buffers after several timed variants, not an
independent equality verdict for each variant or the convolution-history payload. Native token, logit and
complete-state equality and GPU tail behavior remain qualification requirements.

Qwen row buffers at 8,192 rows: two stream slots of 20,480 bytes and the MLP intermediate of 176,128 bytes were
exact page multiples and move to 20,736 and 176,384 bytes. Cost: 256 bytes per arena row (324,352 to 324,608, +0.08 %).

Fixture sweeps, milliseconds per call, identical hashes:

| Kernel | page-multiple stride | +64 | +128 | +192 | +256 |
|---|---|---|---|---|---|
| HC step, 8,192 rows (20,480 base; mirrored copy at the page multiple) | 7.17 | 6.11 | 6.13 | n/a | 6.04 |
| GDN chunk, 8,192 rows (94,208 base; 6.56 at its real 90,496) | 6.97 | 6.83 | 6.57 | 6.68 | 6.54 |
| Index score, 16,384 rows (20,480 / 36,864 base) | 3.64 / 10.94 | 3.66 / 11.12 at +64 | | | |

The index-score writer is insensitive to the stride. In the application the A8 routed-expert runtime requires
256-byte rows, so 256 is the smallest legal step there; it is also the best measured.

| Prompt | tight (ms) | policy (ms) | change | tokens/s | GPU clocks per token |
|---|---|---|---|---|---|
| 8,192 tokens (4 pairs) | 4,565.8 | 4,447.4 | -2.6 % | 1,794 to 1,842 | 1.243 M to 1.199 M |
| 32,768 tokens (2 pairs) | 18,356.2 | 17,904.2 | -2.5 % | 1,785 to 1,830 | 1.421 M to 1.367 M |

`STRIX_ROW_DEALIAS_PERIOD=16` (also moves the 5,120-byte hidden rows) measured the same as the default within
run-to-run spread (4,520 and 4,452 ms), so the default moves exact page multiples only.
