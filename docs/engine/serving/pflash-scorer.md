# Prompt scorer for prefill compression (CPU reference)

Status: host code and tests only. Nothing in the serving path calls it yet, so every request is unchanged.

Prefill compression is a **LOSSY** feature: a small drafter model reads the whole prompt, a scorer ranks the
parts of the prompt by relevance to the request, and the target model prefills only the parts that are kept.
It changes what the model sees, so it is opt-in per request and OFF by default. This page covers one piece:
the scorer math and the loaders of its weight files. The drafter forward, the selection of units and the
serving hook are separate pieces.

Code: `include/strix/serve/prompt_scorer.hpp`, `src/pflash/scorer_cpu.cpp`, `src/pflash/probe_gguf.cpp`.
Test: `strix-prompt-scorer-tests` (`tests/prompt_scorer_tests.cpp`).

## What it computes

All three parts read the same drafter activations: the residual stream after block 14 (the input of block 15),
one vector per prompt token.

| Part | Output per token | Definition |
|---|---|---|
| QK mass | relevance score | Attention mass that the request tokens put on each context token, using the query and key projections of one attention block without a position term |
| Segment probe | probability that a unit starts here | layer norm, linear, GELU (tanh form), linear, a short smoothing filter, sigmoid |
| Request probe | raw logit; request text where logit >= tau | layer norm, linear, GELU (erf form), linear |

**QK mass.** With `x` the residual of a token, `heads` query heads and `kv_heads` key heads:

```
q_in   = rms_norm(x) * attn_norm
Q[h]   = rms_norm(Wq[h] q_in) * q_norm          (for request tokens)
K[g]   = rms_norm(Wk[g] q_in) * k_norm          (for every token; cacheable per conversation)
p[h,t] = softmax over allowed keys j of  Q[h][t] . K[h / (heads / kv_heads)][j] / sqrt(head_dim)
mass[j] = sum over h and t of p[h,t][j] / (heads * request tokens)
```

Allowed keys: a query window scores everything before it and never itself; a scattered request set scores
every token that is not a request token. Masses of several query windows are mixed by weight (current
query 1, i-th earlier query 0.5^(i+1)). A unit's score is the sum of its token masses, or that sum divided by
its length (density). `Wq` may carry a second half per head (a gate); only the first `head_dim` rows of each
head are used.

**Segment probe smoothing.** `cut_prob[t] = sigmoid(raw[t] + conv_bias + sum_k conv[k] * raw[t + k - radius])`
with zeros outside the prompt. The raw logit is added on top of the filter taps.

**Request runs.** Request text is the set of maximal runs of `raw >= tau` that stay inside one message and
are at least 2 tokens long.

Also provided: a box smoother, and a flatness measure (`exp(entropy) / selected units`) for the "nothing
stands out" fallback of the selection layer.

## Weight files

Three small GGUF files, all F32, in the format published with the Apache-2.0 lucebox project ("PFlash v2"):

| File | `general.architecture` | Schema | Tensors |
|---|---|---|---|
| Scoring head | `pflash_scoring_head` | `qwen3_5_0_8b_nope_qk_mass_v1` | `scoringhead.attn_q.weight`, `scoringhead.attn_k.weight` |
| Segment probe | `segmentprobe` | `qwen3_5_0_8b_segment_probe_v1` or `_v2` | `segmentprobe.{norm,fc1,fc2,conv}.{weight,bias}` |
| Request probe | `requestprobe` | `qwen3_5_0_8b_request_probe_v1` | `requestprobe.{norm,fc1,fc2}.{weight,bias}` |

The loaders fail closed. A file is rejected when a key or tensor is missing or has another type or shape,
when the schema, base model or feature tap is not the expected one, or when the drafter hash stored in the
file (`<prefix>.runtime_gguf_sha256`) differs from the hash of the drafter that was loaded. That applies to a
file that is given; none of the three has to be.

**Licence of the weights.** This repository ships no weights, and its tests use synthetic weights only. The
drafter base model (Qwen3.5-0.8B) is published under Apache-2.0. The three trained files above are published
with the licence field `other` and no licence text, so their terms of use are not stated; whoever downloads
them is responsible for clearing that. The default path uses none of them: all three are optional inputs,
absent by default. The QK mass then comes from the drafter's own block-15 projections, the request rows from
the request window the caller passes, and unit boundaries from length and line starts
(see [pflash-drafter.md](pflash-drafter.md)).

## Limits

- Reference implementation: scalar loops with double accumulators, cost O(request tokens x heads x prompt
  tokens x head_dim). It exists to define the result and to check a device implementation, not to serve.
- The GELU of the segment probe is evaluated in closed form; an implementation that uses a lookup table for
  it differs in the last bits.
- The second output of schema v2 segment probes (`subunit.*`) is not read.
- Choosing which request runs become query rows by chat role (latest user turn first, caps per role) needs
  the chat regions and is not part of this piece; `request_rows` only applies a tail cap.

## Test

```
cmake --build <build> --target strix-prompt-scorer-tests
ctest -R strix-prompt-scorer-tests --output-on-failure
```

Closed-form cases for each formula, a seeded random QK case against an independent evaluation, a round trip
of each weight file, and 23 rejected-file cases.
