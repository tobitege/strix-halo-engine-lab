#!/usr/bin/env python3
"""Bytes per tensor class of a GGUF (header only; no tensor data is read).

Used by the tiered expert residency analysis (docs/engine/memory/expert-residency.md) to split a model into
always-resident bytes (attention, dense/shared, router, norms, embeddings) and routed-expert bytes, per layer and
per expert. Sizes come from the tensor offsets, so every quant format is covered without a block-size table.

    python -I scripts/bench/residency/gguf_class_bytes.py MODEL.gguf [--json out.json]
"""
import argparse
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "tools"))
from pflash_ref import Gguf  # noqa: E402

GGML_TYPE = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K",
             12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
             20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 29: "IQ1_M", 30: "BF16"}


def tensor_class(name):
    if "_exps" in name:
        return "routed_experts"
    if "_shexp" in name:
        return "shared_expert"
    if "ffn_gate_inp" in name or "exp_probs" in name:
        return "router"
    if name.startswith("blk."):
        if "norm" in name:
            return "norms"
        if re.search(r"\.(attn_|ssm_|q_|k_|v_|o_)", name):
            return "attention_gdn"
        if re.search(r"\.ffn_(gate|up|down)\.", name):
            return "dense_ffn"
        return "layer_other"
    if "token_embd" in name and "per_layer" not in name:
        return "token_embd"
    if name.startswith("output"):
        return "lm_head_norm"
    if "per_layer" in name:
        return "ple"
    return "global_other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--json")
    a = ap.parse_args()
    g = Gguf(a.gguf)
    size = os.path.getsize(a.gguf)
    items = sorted(g.tensors.items(), key=lambda kv: kv[1][2])
    rows = []
    for i, (name, (dims, kind, off)) in enumerate(items):
        end = items[i + 1][1][2] if i + 1 < len(items) else size - g.data
        rows.append((name, dims, kind, end - off))
    by_class = collections.Counter()
    fmt_by_class = collections.defaultdict(collections.Counter)
    layer_exp = collections.Counter()
    exp_shape = {}
    for name, dims, kind, nbytes in rows:
        c = tensor_class(name)
        by_class[c] += nbytes
        fmt_by_class[c][GGML_TYPE.get(kind, str(kind))] += nbytes
        if c == "routed_experts":
            layer = int(name.split(".")[1])
            layer_exp[layer] += nbytes
            exp_shape[name] = (dims, GGML_TYPE.get(kind, str(kind)), nbytes)
    total = sum(by_class.values())
    arch = g.meta.get("general.architecture")
    keys = {k: v for k, v in g.meta.items() if arch and k.startswith(arch + ".") and not isinstance(v, list)}
    print("file %.3f GB, tensors %d, arch %s" % (size / 1e9, len(rows), arch))
    for k in sorted(keys):
        if re.search(r"(block_count|expert|embedding_length|feed_forward|head_count|context_length|full_attention)", k):
            print("  %s = %s" % (k, keys[k]))
    for c, b in by_class.most_common():
        fm = ", ".join("%s %.2f" % (f, v / 1e9) for f, v in fmt_by_class[c].most_common(4))
        print("%-16s %9.3f GB %5.1f %%  [%s]" % (c, b / 1e9, 100.0 * b / total, fm))
    n_exp = int(keys.get(arch + ".expert_count", 0) or 0)
    if layer_exp and n_exp:
        per = sorted(layer_exp.values())
        print("routed layers %d, experts %d, per-layer GB min %.3f max %.3f, per-expert MB min %.3f max %.3f" % (
            len(per), n_exp, per[0] / 1e9, per[-1] / 1e9, per[0] / n_exp / 1e6, per[-1] / n_exp / 1e6))
        fm = collections.Counter()
        for name, (dims, f, nb) in exp_shape.items():
            fm[(name.split(".")[2], f)] += 1
        print("expert tensor formats: " + ", ".join("%s:%s x%d" % (k[0], k[1], v) for k, v in sorted(fm.items())))
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"file_bytes": size, "classes": dict(by_class),
                       "layer_expert_bytes": {str(k): v for k, v in sorted(layer_exp.items())},
                       "expert_count": n_exp,
                       "expert_tensors": {k: [v[0], v[1], v[2]] for k, v in exp_shape.items()},
                       "meta": {k: v for k, v in keys.items()}}, f, indent=1)


if __name__ == "__main__":
    main()
