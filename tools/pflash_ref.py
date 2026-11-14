#!/usr/bin/env python3
"""Reference forward of the prompt-compression drafter, numpy only.

The drafter is a small hybrid model (gated delta-rule blocks with a causal full-attention block every
`full_attention_interval`). Prompt compression runs blocks 0-14 over the prompt and scores every token with
the query/key projections of block 15, used without a position term ("QK mass"). This file is the
independent reference the C++ CPU implementation (src/pflash/drafter_cpu.cpp) is checked against. It reads
a GGUF file directly; all arithmetic is float64.

  pflash_ref.py synthetic --out tests/pflash_drafter_synthetic_golden.inc
  pflash_ref.py golden    --gguf DRAFTER.gguf --out tests/pflash_drafter_public_golden.inc
  pflash_ref.py next      --gguf DRAFTER.gguf --text "The capital of France is"
  pflash_ref.py rank      --gguf DRAFTER.gguf

`synthetic` needs no file: the weights come from a small generator that the C++ test repeats.
`next` runs every block and the output head: a sanity check that the block math matches the model.
`rank` prints, for a few note/question prompts, where the note that answers the question ranks by QK mass.
"""
import argparse
import math
import mmap
import struct
import sys

import numpy as np

# ---- GGUF ------------------------------------------------------------------------------------------

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_F32, _F16, _BF16 = 0, 1, 30


class Gguf:
    def __init__(self, path, keep_strings=False):
        self.file = open(path, "rb")
        self.map = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.pos = 0
        if self._take(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        self._scalar("<I")
        tensor_count, key_count = self._scalar("<Q"), self._scalar("<Q")
        self.meta = {}
        for _ in range(key_count):
            key = self._string()
            self.meta[key] = self._value(self._scalar("<I"), keep_strings)
        self.tensors = {}
        for _ in range(tensor_count):
            name = self._string()
            dims = [self._scalar("<Q") for _ in range(self._scalar("<I"))]
            self.tensors[name] = (dims, self._scalar("<I"), self._scalar("<Q"))
        align = int(self.meta.get("general.alignment", 32))
        self.data = (self.pos + align - 1) // align * align

    def _take(self, n):
        out = self.map[self.pos:self.pos + n]
        self.pos += n
        return out

    def _scalar(self, fmt):
        return struct.unpack(fmt, self._take(struct.calcsize(fmt)))[0]

    def _string(self):
        return self._take(self._scalar("<Q")).decode("utf-8", "replace")

    def _value(self, kind, keep_strings):
        if kind == 8:
            return self._string()
        if kind == 9:
            elem, count = self._scalar("<I"), self._scalar("<Q")
            if elem == 8:
                items = [self._string() for _ in range(count)]
                return items if keep_strings else None
            fmt = _SCALAR[elem]
            return list(struct.unpack("<%d%s" % (count, fmt[1]), self._take(struct.calcsize(fmt) * count)))
        return self._scalar(_SCALAR[kind])

    def tensor(self, name):
        """float64 array in numpy order (GGUF dimensions reversed): a matrix is [out][in]."""
        dims, kind, offset = self.tensors[name]
        count = int(np.prod(dims))
        begin = self.data + offset
        if kind == _F32:
            flat = np.frombuffer(self.map, dtype="<f4", count=count, offset=begin)
        elif kind == _F16:
            flat = np.frombuffer(self.map, dtype="<f2", count=count, offset=begin)
        elif kind == _BF16:
            bits = np.frombuffer(self.map, dtype="<u2", count=count, offset=begin).astype(np.uint32) << 16
            flat = bits.view(np.float32)
        else:
            raise ValueError("%s: unsupported tensor type %d" % (name, kind))
        return flat.astype(np.float64).reshape(dims[::-1])


# ---- model -----------------------------------------------------------------------------------------

SCORE_BLOCK = 15  # blocks 0..14 run; block 15 gives the scoring projections


def config_from_gguf(g):
    arch = g.meta["general.architecture"]
    m = lambda key: g.meta[arch + "." + key]
    return {
        "hidden": m("embedding_length"), "blocks": m("block_count"),
        "heads": m("attention.head_count"), "kv_heads": m("attention.head_count_kv"),
        "head_dim": m("attention.key_length"), "rope_dims": m("rope.dimension_count"),
        "rope_base": float(m("rope.freq_base")), "eps": float(m("attention.layer_norm_rms_epsilon")),
        "gdn_heads": m("ssm.group_count"), "gdn_head_dim": m("ssm.state_size"),
        "conv_kernel": m("ssm.conv_kernel"), "interval": m("full_attention_interval"),
    }


def is_full(cfg, block):
    return (block + 1) % cfg["interval"] == 0


def rms_norm(x, weight, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def silu(x):
    return x * sigmoid(x)


def softplus(x):
    return np.logaddexp(0.0, x)


def delta_block(cfg, w, x):
    """Gated delta rule: per head a [key][value] state, decayed, corrected towards v along k, read with q."""
    heads, dim, taps, eps = cfg["gdn_heads"], cfg["gdn_head_dim"], cfg["conv_kernel"], cfg["eps"]
    width = heads * dim
    tokens = x.shape[0]
    h = rms_norm(x, w("attn_norm.weight"), eps)
    qkv = h @ w("attn_qkv.weight").T
    z = h @ w("attn_gate.weight").T
    beta = sigmoid(h @ w("ssm_beta.weight").T)
    decay = np.exp(w("ssm_a") * softplus(h @ w("ssm_alpha.weight").T + w("ssm_dt.bias")))
    conv = w("ssm_conv1d.weight")  # [channels][taps], last tap on the current token
    padded = np.concatenate([np.zeros((taps - 1, qkv.shape[1])), qkv])
    mixed = silu(sum(padded[i:i + tokens] * conv[:, i] for i in range(taps)))
    q = mixed[:, :width].reshape(tokens, heads, dim)
    k = mixed[:, width:2 * width].reshape(tokens, heads, dim)
    v = mixed[:, 2 * width:].reshape(tokens, heads, dim)
    q = q / np.sqrt(np.sum(q * q, axis=-1, keepdims=True) + eps) / math.sqrt(dim)
    k = k / np.sqrt(np.sum(k * k, axis=-1, keepdims=True) + eps)
    state = np.zeros((heads, dim, dim))
    out = np.zeros((tokens, heads, dim))
    for t in range(tokens):
        state *= decay[t][:, None, None]
        delta = (v[t] - np.einsum("hkv,hk->hv", state, k[t])) * beta[t][:, None]
        state += k[t][:, :, None] * delta[:, None, :]
        out[t] = np.einsum("hkv,hk->hv", state, q[t])
    out = rms_norm(out, w("ssm_norm.weight"), eps).reshape(tokens, width) * silu(z)
    return x + out @ w("ssm_out.weight").T


def rope(cfg, x):
    """x [tokens][heads][head_dim]; the first rope_dims of each head rotate, pair (i, i + rope_dims / 2)."""
    half = cfg["rope_dims"] // 2
    freq = cfg["rope_base"] ** (-np.arange(half) / half)
    angle = np.arange(x.shape[0])[:, None] * freq[None, :]
    cos, sin = np.cos(angle)[:, None, :], np.sin(angle)[:, None, :]
    a, b = x[..., :half], x[..., half:2 * half]
    out = x.copy()
    out[..., :half] = a * cos - b * sin
    out[..., half:2 * half] = b * cos + a * sin
    return out


def attention_block(cfg, w, x):
    heads, kv, dim, eps = cfg["heads"], cfg["kv_heads"], cfg["head_dim"], cfg["eps"]
    tokens = x.shape[0]
    h = rms_norm(x, w("attn_norm.weight"), eps)
    q_gate = (h @ w("attn_q.weight").T).reshape(tokens, heads, 2 * dim)  # per head: query rows, then gate rows
    q = rope(cfg, rms_norm(q_gate[..., :dim], w("attn_q_norm.weight"), eps))
    k = rope(cfg, rms_norm((h @ w("attn_k.weight").T).reshape(tokens, kv, dim), w("attn_k_norm.weight"), eps))
    v = (h @ w("attn_v.weight").T).reshape(tokens, kv, dim)
    future = np.triu(np.ones((tokens, tokens), dtype=bool), 1)
    out = np.zeros((tokens, heads, dim))
    for head in range(heads):
        group = head // (heads // kv)
        logits = q[:, head] @ k[:, group].T / math.sqrt(dim)
        logits[future] = -np.inf
        p = np.exp(logits - logits.max(axis=-1, keepdims=True))
        out[:, head] = (p / p.sum(axis=-1, keepdims=True)) @ v[:, group]
    out = out * sigmoid(q_gate[..., dim:])
    return x + out.reshape(tokens, heads * dim) @ w("attn_output.weight").T


def feed_forward(cfg, w, x):
    h = rms_norm(x, w("post_attention_norm.weight"), cfg["eps"])
    return x + (silu(h @ w("ffn_gate.weight").T) * (h @ w("ffn_up.weight").T)) @ w("ffn_down.weight").T


def forward(cfg, get, tokens, blocks=SCORE_BLOCK):
    """Residual after `blocks` blocks, [tokens][hidden]. `get(name)` returns a tensor."""
    x = get("token_embd.weight")[np.asarray(tokens)].copy()
    for block in range(blocks):
        w = lambda name, b=block: get("blk.%d.%s" % (b, name))
        x = attention_block(cfg, w, x) if is_full(cfg, block) else delta_block(cfg, w, x)
        x = feed_forward(cfg, w, x)
    return x


def qk_mass(cfg, get, residual, query_begin, query_end):
    """Mean over heads and query rows of the softmax over the keys before the query window; no position
    term. Keys at or after query_begin are masked, their mass is 0."""
    heads, kv, dim, eps = cfg["heads"], cfg["kv_heads"], cfg["head_dim"], cfg["eps"]
    w = lambda name: get("blk.%d.%s" % (SCORE_BLOCK, name))
    tokens = residual.shape[0]
    h = rms_norm(residual, w("attn_norm.weight"), eps)
    q = (h @ w("attn_q.weight").T).reshape(tokens, heads, 2 * dim)[..., :dim]
    q = rms_norm(q, w("attn_q_norm.weight"), eps)
    k = rms_norm((h @ w("attn_k.weight").T).reshape(tokens, kv, dim), w("attn_k_norm.weight"), eps)
    mass = np.zeros(tokens)
    for head in range(heads):
        group = head // (heads // kv)
        logits = q[query_begin:query_end, head] @ k[:query_begin, group].T / math.sqrt(dim)
        p = np.exp(logits - logits.max(axis=-1, keepdims=True))
        mass[:query_begin] += (p / p.sum(axis=-1, keepdims=True)).sum(axis=0)
    return mass / (heads * (query_end - query_begin))


# ---- synthetic weights (repeated by tests/pflash_drafter_tests.cpp) --------------------------------

SYNTHETIC = {"hidden": 32, "blocks": 16, "heads": 4, "kv_heads": 2, "head_dim": 8, "rope_dims": 4,
             "rope_base": 1.0e7, "eps": 1.0e-6, "gdn_heads": 4, "gdn_head_dim": 8, "conv_kernel": 4,
             "interval": 4, "ffn": 64, "vocab": 64}


def synthetic_weights(cfg):
    """Tensor k of the fixed order below is filled from the generator s <- 1664525 s + 1013904223 (mod 2^32),
    seeded with 2654435769 (k + 1) (mod 2^32); u = (s >> 8) / 2^24; value = base + (u - 0.5) * span."""
    out, counter = {}, [0]

    def fill(name, shape, base, span):
        counter[0] += 1
        state = (2654435769 * counter[0]) & 0xFFFFFFFF
        values = np.empty(int(np.prod(shape)))
        for i in range(values.size):
            state = (1664525 * state + 1013904223) & 0xFFFFFFFF
            values[i] = base + ((state >> 8) / 16777216.0 - 0.5) * span
        out[name] = values.astype(np.float32).astype(np.float64).reshape(shape)

    hidden, ffn = cfg["hidden"], cfg["ffn"]
    width = cfg["gdn_heads"] * cfg["gdn_head_dim"]
    attn = cfg["heads"] * cfg["head_dim"]
    attn_kv = cfg["kv_heads"] * cfg["head_dim"]
    fill("token_embd.weight", (cfg["vocab"], hidden), 0.0, 2.0)
    for block in range(SCORE_BLOCK + 1):
        p = "blk.%d." % block
        fill(p + "attn_norm.weight", (hidden,), 1.0, 0.5)
        if block == SCORE_BLOCK or is_full(cfg, block):
            fill(p + "attn_q.weight", (2 * attn, hidden), 0.0, 0.5)
            fill(p + "attn_k.weight", (attn_kv, hidden), 0.0, 0.5)
            if block != SCORE_BLOCK:
                fill(p + "attn_v.weight", (attn_kv, hidden), 0.0, 0.25)
                fill(p + "attn_output.weight", (hidden, attn), 0.0, 0.25)
            fill(p + "attn_q_norm.weight", (cfg["head_dim"],), 1.0, 0.5)
            fill(p + "attn_k_norm.weight", (cfg["head_dim"],), 1.0, 0.5)
        else:
            fill(p + "attn_qkv.weight", (3 * width, hidden), 0.0, 0.5)
            fill(p + "attn_gate.weight", (width, hidden), 0.0, 0.5)
            fill(p + "ssm_beta.weight", (cfg["gdn_heads"], hidden), 0.0, 0.5)
            fill(p + "ssm_alpha.weight", (cfg["gdn_heads"], hidden), 0.0, 0.5)
            fill(p + "ssm_conv1d.weight", (3 * width, cfg["conv_kernel"]), 0.0, 1.0)
            fill(p + "ssm_dt.bias", (cfg["gdn_heads"],), 0.0, 2.0)
            fill(p + "ssm_a", (cfg["gdn_heads"],), -0.75, 1.0)
            fill(p + "ssm_norm.weight", (cfg["gdn_head_dim"],), 1.0, 0.5)
            fill(p + "ssm_out.weight", (hidden, width), 0.0, 0.25)
        if block != SCORE_BLOCK:
            fill(p + "post_attention_norm.weight", (hidden,), 1.0, 0.5)
            fill(p + "ffn_gate.weight", (ffn, hidden), 0.0, 0.25)
            fill(p + "ffn_up.weight", (ffn, hidden), 0.0, 0.25)
            fill(p + "ffn_down.weight", (hidden, ffn), 0.0, 0.25)
    return out


# ---- tokens (stand-in for the real tokenizer: greedy longest match, ASCII text only) ---------------

def greedy_tokens(vocab, text):
    table = {token: index for index, token in enumerate(vocab)}
    coded = text.replace(" ", "\u0120").replace("\n", "\u010a")
    out, pos = [], 0
    while pos < len(coded):
        for length in range(min(24, len(coded) - pos), 0, -1):
            index = table.get(coded[pos:pos + length])
            if index is not None:
                out.append(index)
                pos += length
                break
        else:
            raise ValueError("no token for %r" % coded[pos])
    return out


NOTES = [
    "Note 1: The harbour office opens at seven and closes at noon on Fridays.",
    "Note 2: Marta keeps the spare key for the boat shed under the blue flower pot.",
    "Note 3: The bakery on Mill Street sells rye bread only on Tuesday mornings.",
    "Note 4: The old lighthouse was painted green after the storm of last winter.",
    "Note 5: Piano lessons for the children moved to the library on Thursday evenings.",
    "Note 6: The mountain road stays closed until the snow plough arrives in April.",
]
QUESTIONS = [
    (1, "Question: Where is the spare key for the boat shed?"),
    (2, "Question: On which day can I buy rye bread?"),
    (3, "Question: What colour is the lighthouse now?"),
    (4, "Question: Where do the piano lessons take place?"),
    (5, "Question: When does the mountain road open again?"),
    (0, "Question: At what time does the harbour office close on Fridays?"),
]


def note_prompt(vocab, question):
    tokens, spans = [], []
    for note in NOTES:
        piece = greedy_tokens(vocab, note + "\n")
        spans.append((len(tokens), len(tokens) + len(piece)))
        tokens += piece
    query_begin = len(tokens)
    tokens += greedy_tokens(vocab, question + "\n")
    return tokens, spans, query_begin


# ---- golden files ----------------------------------------------------------------------------------

def write_golden(path, namespace, source, tokens, query_begin, residual, mass):
    def rows(values, fmt, per_line):
        items = [fmt % v for v in values]
        return "\n".join("    " + ", ".join(items[i:i + per_line]) + "," for i in range(0, len(items), per_line))

    count = len(tokens)
    sample_tokens = [0, count // 2, count - 1]
    samples = [residual[t, d] for t in sample_tokens for d in range(8)]
    norms = np.sqrt(np.sum(residual * residual, axis=-1))
    with open(path, "w", newline="\n", encoding="ascii") as f:
        f.write("// Generated by tools/pflash_ref.py (%s); do not edit.\n" % source)
        f.write("// tokens; query window [query_begin, token count); QK mass per token; L2 norm of the block-14\n")
        f.write("// residual per token; residual[t][0..8) for t = first, middle, last token.\n")
        f.write("namespace %s {\n" % namespace)
        f.write("inline constexpr std::uint32_t tokens[] = {\n%s\n};\n" % rows(tokens, "%d", 16))
        f.write("inline constexpr std::uint32_t query_begin = %d;\n" % query_begin)
        f.write("inline constexpr double mass[] = {\n%s\n};\n" % rows(mass, "%.9e", 6))
        f.write("inline constexpr double residual_norm[] = {\n%s\n};\n" % rows(norms, "%.9e", 6))
        f.write("inline constexpr double residual_sample[] = {\n%s\n};\n" % rows(samples, "%.9e", 6))
        f.write("}  // namespace %s\n" % namespace)


def cached(g):
    store = {}

    def get(name):
        if name not in store:
            store[name] = g.tensor(name)
        return store[name]
    return get


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("mode", choices=["synthetic", "golden", "next", "rank"])
    parser.add_argument("--gguf")
    parser.add_argument("--out")
    parser.add_argument("--text", default="The capital of France is")
    args = parser.parse_args()

    if args.mode == "synthetic":
        cfg = SYNTHETIC
        weights = synthetic_weights(cfg)
        tokens = [(7 * i * i + 3 * i + 5) % cfg["vocab"] for i in range(40)]
        query_begin = 31
        residual = forward(cfg, weights.__getitem__, tokens)
        mass = qk_mass(cfg, weights.__getitem__, residual, query_begin, len(tokens))
        write_golden(args.out, "golden_synthetic", "synthetic weights", tokens, query_begin, residual, mass)
        return 0

    g = Gguf(args.gguf, keep_strings=True)
    cfg = config_from_gguf(g)
    vocab = g.meta["tokenizer.ggml.tokens"]
    get = cached(g)
    if args.mode == "next":
        tokens = greedy_tokens(vocab, args.text)
        x = forward(cfg, get, tokens, blocks=cfg["blocks"])
        last = rms_norm(x[-1], get("output_norm.weight"), cfg["eps"])
        logits = get("token_embd.weight") @ last
        best = np.argsort(-logits)[:5]
        print("tokens", tokens)
        print("next", [(int(i), vocab[i], round(float(logits[i]), 2)) for i in best])
        return 0
    if args.mode == "rank":
        # The first token of a prompt collects a large share of the mass whatever is asked, so the ranking
        # is printed with and without it.
        hits = [0, 0]
        for target, question in QUESTIONS:
            tokens, spans, query_begin = note_prompt(vocab, question)
            mass = qk_mass(cfg, get, forward(cfg, get, tokens), query_begin, len(tokens))
            ranks = []
            for skip in (0, 1):
                density = [float(mass[max(b, skip):e].sum()) / (e - max(b, skip)) for b, e in spans]
                order = sorted(range(len(spans)), key=lambda i: -density[i])
                ranks.append(order.index(target) + 1)
                hits[skip] += order[0] == target
            print("target note %d: rank %d, without the first token %d; first token mass %.3f; note mass %s" % (
                target + 1, ranks[0], ranks[1], mass[0], " ".join("%.3f" % mass[b:e].sum() for b, e in spans)))
        print("top-1 %d/%d, without the first token %d/%d" % (hits[0], len(QUESTIONS), hits[1], len(QUESTIONS)))
        return 0
    tokens, _, query_begin = note_prompt(vocab, QUESTIONS[0][1])
    residual = forward(cfg, get, tokens)
    mass = qk_mass(cfg, get, residual, query_begin, len(tokens))
    source = "%s, %s" % (g.meta.get("general.name", "drafter"), g.meta.get("general.license", "licence not recorded"))
    write_golden(args.out, "golden_public", source, tokens, query_begin, residual, mass)
    return 0


if __name__ == "__main__":
    sys.exit(main())
