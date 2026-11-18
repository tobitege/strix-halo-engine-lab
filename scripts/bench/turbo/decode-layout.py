#!/usr/bin/env python3
"""Header-only GGUF decode weight census. Logical traffic is not a DRAM counter.

Usage: decode-layout.py MODEL [--emitted-per-pass E] [--verify-rows R]
Embedding lookup is excluded; output head is included. MoE traffic is bounded
between one top-k union and disjoint per-row top-k unions. Alignment padding is
included for non-Q8 tensors; Q8 and onfly copies use their exact format sizes.
"""
import argparse
import json
import math
import os
import struct


def census(path, rows, emitted):
    if rows < 1 or emitted <= 0:
        raise ValueError("rows and emitted-per-pass must be positive")
    with open(path, "rb") as f:
        def scalar(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

        def string():
            return f.read(scalar("Q")).decode("utf-8")

        formats = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i",
                   6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

        def value(kind, keep=True):
            if kind == 8:
                n = scalar("Q")
                if keep:
                    return f.read(n).decode("utf-8")
                f.seek(n, 1)
            elif kind == 9:
                elem, n = scalar("I"), scalar("Q")
                if elem in formats:
                    f.seek(struct.calcsize(formats[elem]) * n, 1)
                else:
                    for _ in range(n):
                        value(elem, False)
            elif kind in formats:
                return scalar(formats[kind])
            else:
                raise ValueError(f"unknown GGUF metadata type {kind}")

        if f.read(4) != b"GGUF" or scalar("I") not in (2, 3):
            raise ValueError("expected GGUF v2/v3")
        nt, nk = scalar("Q"), scalar("Q")
        meta = {}
        for _ in range(nk):
            key, kind = string(), scalar("I")
            keep = any(key.endswith(s) for s in (
                "architecture", "alignment", "expert_count", "expert_used_count", "block_count"))
            v = value(kind, keep)
            if keep:
                meta[key] = v
        tensors = []
        for _ in range(nt):
            name, nd = string(), scalar("I")
            dims = [scalar("Q") for _ in range(nd)]
            tensors.append((name, dims, scalar("I"), scalar("Q")))
        alignment = meta.get("general.alignment", 32)
        data = math.ceil(f.tell() / alignment) * alignment
    tensors.sort(key=lambda t: t[3])
    experts = next((v for k, v in meta.items() if k.endswith(".expert_count")), 0)
    topk = next((v for k, v in meta.items() if k.endswith(".expert_used_count")), 0)
    groups = {k: {"tensors": 0, "q8_bytes": 0, "int4_bytes": 0, "shapes": {}}
              for k in ("gdn_in", "gdn_out", "qsa_io", "shared_expert")}
    dense = moe = embed = saving = unused = fallback_copy = 0
    fallback_tensors = 0
    for i, (name, dims, kind, offset) in enumerate(tensors):
        end = tensors[i + 1][3] if i + 1 < nt else os.path.getsize(path) - data
        size = math.prod(dims) // 32 * 34 if kind == 8 else end - offset
        if "token_embd.weight" in name:
            embed += size
            continue
        if "_exps.weight" in name:
            moe += size
        else:
            dense += size
        group = None
        if name.endswith(("attn_qkv.weight", "attn_gate.weight")):
            group = "gdn_in"
        elif name.endswith("ssm_out.weight"):
            group = "gdn_out"
        elif name.endswith(("attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight")):
            group = "qsa_io"
        elif name.endswith(("ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight")):
            group = "shared_expert"
        if group and kind == 8 and len(dims) == 2:
            k, n = dims
            # Mirror qwen_onfly_i4_candidate and launch_iu4_rows. Allocation
            # alone does not mean the projection dispatch consumes the copy.
            convertible = (0 < k <= 0xFFFFFFFF and 0 < n <= 0xFFFFFFFF
                           and k % 64 == 0)
            copy = math.ceil(n / 64) * 64 * k * 17 // 32 if convertible else 0
            eligible = convertible and rows <= 8 and k // 64 in (32, 40, 64, 96)
            if eligible:
                saving += size - copy
            else:
                fallback_tensors += 1
                fallback_copy += copy
                if name.endswith("ffn_down_shexp.weight"):
                    unused += copy
            g = groups[group]
            g["tensors"] += 1
            g["q8_bytes"] += size
            g["int4_bytes"] += copy
            shape = "x".join(map(str, dims))
            g["shapes"][shape] = g["shapes"].get(shape, 0) + 1
    if not experts or not topk:
        raise ValueError("expert counts unavailable; cannot infer active MoE bytes")
    # Storage eligibility does not establish a consumer, numerical qualification,
    # or a full-matrix read: embeddings are sparse lookups, so keep them separate.
    known = ("attn_qkv.weight", "attn_gate.weight", "ssm_out.weight",
             "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
             "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight")
    remaining = {}
    for name, dims, kind, _ in tensors:
        if kind != 8 or len(dims) != 2 or dims[0] % 64 or name.endswith(known):
            continue
        # Strip the block number, retaining the tensor's semantic suffix.
        role = name.split(".", 2)[2] if name.startswith("blk.") else name
        r = remaining.setdefault(role, {"tensors": 0, "q8_bytes": 0, "int4_bytes": 0,
                                       "shapes": {}, "read_kind": "lookup" if "token_embd" in name else "projection"})
        r["tensors"] += 1
        r["q8_bytes"] += math.prod(dims) // 32 * 34
        r["int4_bytes"] += math.ceil(dims[1] / 64) * 64 * dims[0] * 17 // 32
        shape = "x".join(map(str, dims))
        r["shapes"][shape] = r["shapes"].get(shape, 0) + 1
    remaining_saving = sum(r["q8_bytes"] - r["int4_bytes"] for r in remaining.values()
                           if r["read_kind"] == "projection")
    exact = [dense + moe * min(experts, topk * u) / experts for u in (1, rows)]
    return {"schema": "decode-layout/v1", "metadata": meta, "groups": groups,
            "dense_bytes_per_pass": dense, "all_expert_bytes": moe,
            "embedding_bytes_excluded": embed, "verify_rows": rows,
            "emitted_per_pass": emitted, "onfly_copy_bytes": sum(g["int4_bytes"] for g in groups.values()),
            "unconsumed_shared_down_copy_bytes": unused,
            "unconverted_storage_eligible_q8_2d": remaining,
            "unconverted_projection_logical_saved_bytes": remaining_saving,
            "unconverted_projection_byte_saving_at_243GBs_ms": remaining_saving / 243e6,
            "unconsumed_copy_bytes": fallback_copy,
            "q8_fallback_tensors": fallback_tensors,
            "logical_saved_bytes_per_pass": saving,
            "onfly_row_route": ("Q8 fallback for a single wide projection call" if rows > 8 else
                                "contains Q8 fallback shapes" if fallback_tensors else "eligible"),
            "exact_logical_weight_bytes_per_pass_bounds": exact,
            "onfly_logical_weight_bytes_per_pass_bounds": [b - saving for b in exact],
            "exact_logical_weight_bytes_per_emitted_token_bounds": [b / emitted for b in exact],
            "onfly_logical_weight_bytes_per_emitted_token_bounds": [(b - saving) / emitted for b in exact],
            "warning": "layout bounds only: exclude drafter, activations, KV, cache effects and repeated physical loads; mixed-width cohorts require a call histogram"}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("--verify-rows", type=int, default=8)
    p.add_argument("--emitted-per-pass", type=float, default=1)
    a = p.parse_args()
    print(json.dumps(census(a.model, a.verify_rows, a.emitted_per_pass), indent=2))
