# Vision tower adapters

Image input is built once in shared code. A model family contributes one *adapter*: a small class that states
what is specific to its tower and runs nothing itself.

| Layer | Shared home | What the adapter supplies |
|---|---|---|
| Decode, resize, normalise, patchify | `include/strix/vision/{image,preprocess}.hpp` | resize rule, resampler, mean and standard deviation, patch size and row order |
| Weights | `include/strix/vision/gguf_weights.hpp` (`GgufWeights`, any GGUF tower) | tensor names (`bind`) |
| Tower | `include/strix/vision/encoder.hpp` (`ReferenceEncoder` on the host, `Encoder` on the device) | `EncoderConfig` |
| Projector | `DenseOps::linear` of the backend in use | which rows are merged, which linears run, the special rows |
| Rows in the prompt | `include/strix/session/embed_rows.hpp` | `SpanLayout`: one role and position per row, the token id the rows count as |
| Cache | `include/strix/vision/embed_cache.hpp` | `identity()` |

The interface is `strix::vision::VisionAdapter` (`include/strix/vision/adapter.hpp`). Shared code names no family.

## Writing an adapter

1. **Validate the file.** A strict validator in the family's model source (metadata keys and types, value ranges,
   exact tensor inventory), as `deepseek41::validate_vision` does. Admission by digest stays where the family
   already pins its files.
2. **`config()`**: fill `EncoderConfig` from the validated metadata. Check `vision::hip_supported(config)`.
3. **`bind()`**: map tensor names to `EncoderWeights` through a `WeightSource`. `GgufWeights` returns a tensor
   with GGUF dimensions `(d0, ..., dn)` as the matrix `[dn][d0 * ... * d(n-1)]`, which is the source framework's
   `[out][in...]` weight; a flattened convolution weight therefore has features in (channel, frame, y, x) order,
   the order `vision::patchify` emits. Open the file with a `vision::HostBudget`: the file length is checked
   against `max_bytes` (and against `exact_bytes`, the pinned length, when the family pins its file) before a byte
   is allocated, then the bounded header is validated, then the bytes are read.
4. **`patches()`**: `ResizePolicy::target`, `vision::resize`, `vision::normalize`, `vision::patchify`, then the
   patch coordinates. Refuse an image the rule cannot represent with `std::invalid_argument`. Call
   `vision::check_grid_axes` before the resample: a patch coordinate is 16 bits, so an axis is at most 65,536
   patches whatever the token limit allows.
5. **`project()`**: receives the tower output and the grid it came from. Build the projector input (merge,
   unshuffle, pool), call `dense.linear` for every linear so the host reference and the device share the graph,
   and write the rows in span order, special rows included. The grid is a caller's argument: start with
   `vision::checked_merged_grid(grid, merge, max_tokens)` (64-bit products, positive axes, `rows` equal to the
   product, the token limit) and size every buffer from its result. `layout()` does the same.
6. **`layout()`**: the same span as roles and positions. Row count and order must equal what `project()` writes.
7. **`device_options()`** (optional): a policy bit of the device arithmetic the tower needs, found by the gate.
8. **`identity()`**: the weights digest plus everything that changes the rows. Cached rows and prefix entries
   are keyed by it, so a changed parameter can never reuse old rows. State a float parameter by its bit
   pattern, not by a rounded decimal, and refuse a non-finite one.
9. **Tests**: a host test on a small synthetic tower with the real tensor names (projector against a direct
   computation, layout, resize table, cache key, refusal of malformed grids and files), and a gate on the real
   file. Gates compare tensors with `vision::abs_diff` (`include/strix/vision/compare.hpp`): a NaN or an infinity
   on either side is a failure, it never disappears in a maximum.

A family that needs something the interface does not carry extends the shared interface; it does not work around
it inside the adapter.

Open before a tower is served: the cache key ends with `EncoderBackend::arithmetic()`, one constant string per
backend class. It does not state the options a device backend was built with (`EncoderOptions`, for example split
activations), so two device backends with different options share keys today. The effective options have to become
part of that string, and the backend has to be built from `device_options()` in one place, before any served path
shares a cache between differently built backends.

## DeepSeek V4.1

`src/deepseek41/vision_adapter.{hpp,cpp}`, validator `deepseek41::validate_vision`, host test
`deepseek41-vision`, gate tool `strix-deepseek41-vision-gate` with the reference dumps of
`tools/vision/ds41_ref.py`.

Read from the file: 32 blocks, width 1024, 16 heads, RMSNorm (weight only, epsilon 1e-6), fused `wqkv` and `wo`
with bias, gated SwiGLU 2816 without bias, 2D rotary positions with base 10000, patch 14, a final norm, a 3x3
merge into `aligner.w1` (9216 to 5120) and `aligner.w2` (5120 to 5120), three learned rows (`image_start`,
`image_newline`, `image_end`), at most 1024 image rows, at least 295936 pixels, image token id 129264.

Measured on the tensors: the patch weight is laid out (channel, y, x) (neighbouring weights correlate along
those axes and not along the alternative); the aligner input is channel-major, index `c * 9 + slot` (the column
norms follow a 3 x 3 pattern in the slot and are flat otherwise).

**Not stated by the file.** These come from the reference implementation that ships inside the checkpoint
(`inference/image_processor.py`, `inference/vision.py`, at the revision named by the file's
`general.source.url` and `general.source.revision`); `tools/vision/ds41_ref.py upstream` runs that code on the
vision tensors and `compare` measures our rows against it. Each is a field of `deepseek41::VisionRecipe` or a
rule of the adapter, and part of the adapter identity:

| Parameter | Value | Status against the reference code |
|---|---|---|
| Canvas size | an image under 295936 pixels is scaled up (`int()` of each side); each side is rounded up to a multiple of 14, not of 42; a canvas whose span would exceed 1024 rows is shrunk to the largest one that fits; an extreme aspect ratio collapses to one column or one row of cells | was wrong (both sides to multiples of 42, aspect not kept), corrected |
| Placement | the image keeps its aspect ratio inside the canvas (`ImageOps.pad`): bicubic with antialias, centred, margin grey 127, sizes and offsets rounded half to even | was wrong (stretched to the canvas), corrected |
| Mean, standard deviation | 0.5, 0.5 per channel, as `(v / 255 - 0.5) / 0.5` with a true division | confirmed; the reciprocal form differed in the last place |
| Pixel precision | the normalised pixels are rounded to BF16 before the patch embedding | was missing, corrected |
| Row limit | the whole span (image rows, row ends, start, end) is at most 1024 rows | was wrong (1024 image rows), corrected |
| Partial cells | a grid that is not a multiple of 3 is completed with zero rows on the right and at the bottom before the aligner | was missing, corrected |
| SwiGLU halves | first half of `mlp.w1` is the gate | confirmed |
| Rotary layout | row angles then column angles, 16 frequencies per axis, pairs `(i, i + 32)` | confirmed |
| Aligner activation | GELU (exact, erf) | confirmed |
| Slot order in a 3 x 3 cell | `c * 9 + dy * 3 + dx` (`unfold` on channel, y, x) | confirmed |
| Span | start, each grid row followed by the row end (also the last), end; 1D positions | confirmed |

The reference rounds every linear to BF16; the adapter's FP32 reference rows are compared with the reference code
run in FP32, and its own BF16 run measures what that arithmetic alone moves (see the status table). The
transformers package has no model code for this family: the checkpoint's own files are the reference.

Image input for this family is still not advertised by any service: the judge of the device arithmetic on images
with flat areas (per-row minimum or a language-model gate) is an open decision, see the status table.

### Gate

```
python3 tools/vision/ds41_ref.py images --out REF
python3 tools/vision/ds41_ref.py preprocess --out REF
python3 tools/vision/ds41_ref.py encode --out REF --model VISION.gguf
python3 tools/vision/ds41_ref.py upstream --out REF --model-dir CHECKPOINT [--dtype bf16] [--patches ours]
python3 tools/vision/ds41_ref.py compare --out REF --a "" --b upstream
strix-deepseek41-vision-gate --model VISION.gguf --ref REF --backend host --only rings_graph
strix-deepseek41-vision-gate-hip --model VISION.gguf --ref REF --backend hip
```

Dump comparison retains the recorded span count, including recipes without a final newline. New dumps
write per-case `.layout.json` files with merge factor, recipe and feature widths. Legacy untagged dumps
without that metadata retain the default merge factor of three; regenerate non-default dumps before comparison.
`compare` reports compared/skipped tensor counts. `--strict --max-rel-l2 LIMIT` requires nonempty,
complete coverage, recorded feature widths, finite numerical results within that limit and exact special rows
(the default limit is zero). This checks dump agreement; source and weight provenance still need independent
verification, particularly when assembling a dump set with `--only`.

`encode` is an independent FP32 implementation in numpy, not the upstream code: it proves the wiring of the
shared tower and of the adapter, and it is the FP32 reference the device arithmetic (`vision-f16-v1`, class
FAST) is judged against. Thresholds: preprocessing max difference 1e-6 on the patch tensor; host backend
cosine at least 0.99999 per row; device backend mean cosine at least 0.9995, minimum 0.998, relative L2 at most
2 %; the special rows bit-equal to the file.

### Gate status on the pinned file

| Check | Result |
|---|---|
| Admission and strict validation | pass (266 tensors) |
| Preprocessing against Pillow, 6 images | patch tensor bit-equal |
| Host FP32 tower + adapter against the numpy implementation | cosine 1.000000, relative L2 0.00000 (rows differ only by their BF16 storage) |
| Device arithmetic `vision-f16-v1`, 7 images | single FP16 operands: 4 pass, 3 fail the row minimum; split activations: 1 of the 5 run fails it |
| Reference processor (checkpoint code), 6 images | patch tensor bit-equal (maximum difference 0), same grids, including grids with partial cells |
| Reference model in FP32, 7 images, same patches | tower and rows cosine 1.000000, row minimum at least 0.999994, relative L2 at most 0.0002; special rows bit-equal |
| Reference processor + model in FP32 against `preprocess` + `encode`, 6 images | rows cosine 1.000000, row minimum at least 0.999999, relative L2 at most 0.0001 |
| Reference model in its own BF16 arithmetic against its FP32 run, 7 images | row minimum 0.69 to 0.997, relative L2 2.4 % to 18 %: no image meets the device thresholds |

The device failures are on images with large flat areas (a smooth gradient, an upscaled 16 x 16 image, smooth
rings): the mean cosine stays at or above 0.998 while single rows fall far below it (row minimum 0.925, 0.996 and
0.978). `--diagnose` of the gate tool compares the residual stream after selected blocks with the host reference:

- Through block 11 the error is small and uniform, but already 30 times larger without split activations
  (relative L2 0.013 against 0.0003): on a flat patch the linears subtract large, nearly equal terms, so the
  rounding of a single FP16 operand is a large fraction of the result.
- Between blocks 12 and 15 the tower forms a few rows of very large norm (about 1000 against a median of 7).
  Which rows those become is decided by small differences between nearly identical patches, so the rounding
  error flips the choice: 69 of 1521 rows leave the reference at block 15 without split activations, 1 with them.

The error is therefore clustered in rows, input dependent and a precision effect, not a wiring fault. The adapter
requests split activations (`device_options()`); with them the gradient and the upscaled image pass and the rings
image misses the minimum on one row (0.9969; mean 0.99999, relative L2 0.3 %). The device arithmetic of this
tower is **not accepted yet**: the remaining single-row flips need either more precision in the first blocks
and in the attention operands, or a judgement on the language model's output instead of a per-row minimum.

Measured tower time on one node (warm, no per-layer synchronisation): 111 ms for 1,521 patches (the minimum
image), 388 ms for 3,888, 1,907 ms for 9,216 (the token cap); about 2.1 x, 1.8 x and 1.4 x that with split
activations. The aligner adds 22 to 83 ms.

## Image rows in the language model prefill

An encoded image reaches the language model as rows of its hidden width that replace the embedding-table rows of
part of a prefill run. The mechanism is shared; a family only states its per-row extras.

| Piece | Where | What it does |
|---|---|---|
| `vision::span_embed_rows` | `include/strix/vision/embed_span.hpp` | `layout()` + the rows of `encode_cached` become one `session::EmbedRows` (roles, positions, the token id the rows stand for) |
| `session::resolve_embed_run` | `include/strix/session/embed_splice.hpp` | the ids of one run become vocabulary ids plus an `EmbedSplice`: segments of consecutive run rows that read consecutive provided rows, checked against the family's rules (width, row order, positions, deep rows) |
| `session::enqueue_embed_splice` | `src/session/embed_splice.hip` | device copy of one segment into the layer-0 input, BF16 or binary16 to FP32, into every residual stream copy of a row; no index arrays |

A run without a virtual id resolves to the same ids and an empty splice, and the embed stage then launches and
uploads nothing: text prompts are unchanged bit for bit.

### DeepSeek V4.1

`DevicePrefill::set_embed_rows(splice)` names the provided rows of the next run. The run looks every token up as
before, overwrites the spliced rows, and for segments marked `extras` (every row of an image span: start, image,
row ends, end)

- sets the per-row image flag, so the routers add their image bias for those rows;
- treats the rows as dead n-gram positions: no Engram n-gram reads through one, and its own n-grams are padding.
  A token history kept for the decode engines marks such rows with `ds41::engram_dead_token`.

Positions are the prompt index, as for text; `resolve_embed_run` refuses a span that is not in row order. A
segment without `extras` is handled exactly as the token rows it replaces, which is what the identity gate uses:

```
strix-deepseek41-prefill MODEL.gguf --prompt IDS --rows N --layers L --embed-identity A:B
```

runs the prompt once from the table and then with rows `[A, B)` provided: the table's own rows must reproduce
every bit (logits, next token, one stream hash per layer, attention state). Shifted rows must change the logits;
one-row or identical-row controls fail as inconclusive. Rows cut to eight significant bits must give the same
bits as binary16 and as BF16. The image arm requires finite logits and an observable change, plus independent
checks: actual dead-position hashes equal a full-padding oracle and differ from plain history; a device router
fixture using an uploaded image flag selects expert 1 with image bias and expert 0 with text bias. These are
internal wiring checks, not model-fidelity qualification. Long text after an image run and a shorter text run
must equal the first run, also when the shorter run has an extras-disabled splice. Choose an interval extending
beyond the prompt midpoint to exercise the stale-tail case. Stream hashes are enabled automatically.
The gate rejects `--ttft-timing` and worlds other than 1 before loading assets.

Image serving remains unqualified: span-role and dead-position semantics need model-contract evidence.
Extend boundaries, cancellation, two-rank splitting and old-binary text equality need separate qualification.

Memory: `deepseek41::vision_memory` (`src/deepseek41/vision_splice.hpp`) lists the device bytes image input adds
once per rank (tower panels and workspace from `vision::plan`, the aligner panels, the splice staging at the
longest run plus a conservative second staging allocation during growth) and the host bytes of one largest image; `add_vision_capacity` adds them to the pools of a slot
capacity report. No served route enables image input yet.
