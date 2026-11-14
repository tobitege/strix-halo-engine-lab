# Drafter for prefill compression (CPU reference)

Prefill compression (class LOSSY, opt-in per request, OFF by default) lets a small model read a long prompt
first and keep only the parts that matter for the request. The small model is the **drafter**. It never
generates text: it runs its first 15 blocks over the prompt, and the query/key projections of its next block
score every token against the request ("QK mass", see [pflash-scorer.md](pflash-scorer.md)).

This page covers the CPU reference of the drafter: `include/strix/pflash/drafter.hpp`,
`src/pflash/drafter_cpu.cpp`. It defines the result and is the oracle for a device implementation. It is far
too slow to serve requests, and nothing calls it yet: every request is unchanged.

## Weights: the public drafter only

The scorer needs **only the drafter's own weights**. The drafter is the public Qwen3.5-0.8B model, published
under Apache-2.0. Trained add-ons exist elsewhere (replacement scoring projections, a unit-boundary probe, a
request probe); their terms of use are not stated, so this path does not need them. They are optional inputs
of `ScoreOptions`, null by default:

| Without add-ons (default) | With the optional add-on |
|---|---|
| Scoring projections: block 15's own `attn_q` / `attn_k` | replaced by the scoring-head file |
| Request rows: the tokens of the request window the caller passes | the request probe's runs inside that window |
| Unit boundaries: `cut_prob` is all zeros; the selection layer cuts units by length and line starts | the segment probe's output |

This repository ships no weights. `load_drafter` reads a GGUF with F32, F16 or BF16 tensors and fails closed:
a missing or mistyped key or tensor, or a shape that disagrees with the metadata, is an error. Only blocks
0-14 and five tensors of block 15 are read; `load_embeddings` reads just the token rows of the prompt.

## What it computes

Residual stream `x`, one row per token. `rms(x) = x / sqrt(mean(x^2) + eps)`. Every block is
`x += mix(rms(x) * attn_norm)` then `x += down(silu(gate(h)) * up(h))` with `h = rms(x) * post_attention_norm`.
Block `b` is a full-attention block when `(b + 1) % full_attention_interval == 0` (blocks 3, 7, 11 of the 15
that run), otherwise a delta-rule block.

**Delta-rule block** (H heads of width D; q, k, v, gate z, write strength and decay input are projections of
the normed input):

1. q, k, v pass a causal depthwise convolution over the last `conv_kernel` tokens, then SiLU.
2. Per head: `q <- q / sqrt(sum(q^2) + eps) / sqrt(D)`, `k <- k / sqrt(sum(k^2) + eps)`.
3. Per head a state `S` [D][D], zero at the first token. For each token:
   `S <- S * exp(a * softplus(alpha + dt_bias))`; `S <- S + k (x) ((v - S^T k) * sigmoid(beta))`; `o = S^T q`.
4. `o <- rms(o) * norm * silu(z)` per head, then the output projection.

**Full-attention block** (8 query heads share 2 key/value heads in groups of 4; the query projection carries
a second half per head, the output gate): queries and keys are RMS-normed per head, the first `rope_dims`
dimensions of each head rotate by `position * base^(-i / (rope_dims / 2))` with pairs `(i, i + rope_dims / 2)`,
attention is causal with scale `1 / sqrt(head_dim)`, and the mixed values are multiplied by
`sigmoid(gate)` before the output projection.

**Scoring** (`score_residual`): the residual after block 14 goes to `qk_mass` with block 15's projections and
no position term. The query rows are the request window (its last `request_cap` tokens); the keys are the
tokens before the window. Earlier request windows can be mixed in at weights 0.5, 0.25, ...

Arithmetic: scalar loops in a fixed order, float storage, double accumulators; dot products use four
interleaved partial sums. The same inputs give the same outputs on one build, and a prefix of a prompt gives
bit-identical rows (the model is causal).

## Python reference and goldens

`tools/pflash_ref.py` is an independent implementation (numpy only, float64, reads the GGUF itself).

| Mode | Purpose |
|---|---|
| `synthetic --out FILE` | golden from generated weights; the C++ test repeats the generator, so it needs no file |
| `golden --gguf DRAFTER --out FILE` | golden from the public drafter |
| `next --gguf DRAFTER --text T` | runs all blocks and the output head and prints the likeliest next tokens |
| `rank --gguf DRAFTER` | six notes and a question about one of them: where the right note ranks |

`next` is the check that the block math above is the model's: with the public drafter it continues
"The capital of France is" with " Paris" and "One, two, three, four, five," with " six". The two details
that a wrong reading would get wrong were tried both ways: gating the delta-rule output with `sigmoid(z)`
instead of `silu(z)` gives unrelated tokens, and rotating adjacent pairs instead of `(i, i + rope_dims / 2)`
raises the mean next-token loss on a 96-token text from 1.40 to 1.56.

The tool has no tokenizer: it maps ASCII text to tokens by greedy longest match over the vocabulary, which
is enough for goldens (they are defined on token ids) but is not the model's tokenization in general.

Goldens in the repository: `tests/pflash_drafter_synthetic_golden.inc` and
`tests/pflash_drafter_public_golden.inc`. The second holds token ids and numbers computed from
`Qwen3.5-0.8B-BF16.gguf` (1,516,744,736 bytes, sha256
`cedf89af31c9041b601fa58303285bc46d99c51baee1b13f5e919626ca526ee5`, Apache-2.0), not the weights.

## Test

```
cmake --build <build> --target strix-pflash-drafter-tests
ctest -R strix-pflash-drafter-tests --output-on-failure
```

- Synthetic golden: residual norms, residual samples and QK mass against the Python reference.
- Forward: repeatable, causal bit for bit, later tokens depend on earlier ones, malformed weights rejected.
- Scoring: mass sums to 1 over the keys before the request, request cap, history mixing, suffix candidates,
  optional scoring head, bad windows rejected, the `PromptScorer` wrapper.
- Files: round trip in F32, BF16 and F16, the synthetic golden through the loader, 17 rejected files.
- Public golden: runs only when `STRIX_PFLASH_DRAFTER_GGUF` names the drafter file; otherwise it prints
  `public: skipped`.

## Limits

- Reference speed: single-threaded scalar arithmetic, about 0.6 GFLOP per token; the 126-token public golden
  takes about 8 s on one desktop core, and the cost of the attention blocks grows with the square of the length.
- No state is kept between calls (no sessions); the device implementation owns that.
- The caller supplies the tokenizer, the byte offsets and the request window (chat adapter).
- Quality is not measured here. See the note below before relying on raw scores.

**First-token share.** With the drafter's own projections, the first token of a prompt receives a large part
of the mass whatever the question is. In the `rank` smoke (six one-line notes, six questions) it took 22-31 %
of the mass. Ranked by mass per token, the note that answers the question came first in 2 of 6 questions with
that token counted and in 6 of 6 without it. A caller that ranks units by raw mass should leave the first
token out; in a chat prompt it belongs to the template envelope, which is always kept. This is a smoke
observation on one tiny prompt, not a quality measurement.
