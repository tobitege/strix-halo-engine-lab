#!/usr/bin/env python3
"""kind_excess.py SHORT.csv LONG.csv [--mult M] [--top N]: long-context excess per kernel kind and per phase.

Reads two rocprofv3 kernel traces (`rocprofv3 --kernel-trace --output-format csv`) of one prefill process each, a
short prompt and a long one that is M times longer (default 4, e.g. 32K and 128K), and prints per phase and per
kernel kind: total ms in each trace and LONG - M x SHORT, the part of the long prompt that does not scale with its
length (attention-like work grows faster). Kind = kernel name up to its argument list, anonymous namespaces and
the leading namespace dropped. Phases are regex groups on the kind (--phase NAME=REGEX, first match wins; the
defaults are the Qwen QSA/GDN/MoE/HC prefill kernels, other families pass their own). CPU only, stdlib only.
"""
import argparse, csv, re

PHASES = [("indexer", r"qsa_flash::(score|select|key_range|row_maxima)|block_pool|select_rows"),
          ("attend", r"qsa_fast::|attention_"), ("qsa-proj", r"QsaInput|QsaOutput|qsa_index::|norm_rope"),
          ("gdn", r"Gdn|gdn"), ("moe", r"moe|router|Shared|known_moe|wave_rows_route"), ("hc", r"qpf2k_hc|inject_norm")]


def kind(name):
    n = re.sub(r"\(anonymous namespace\)::", "", name).replace("strix::", "")
    return re.sub(r"^void ", "", n.split("(")[0].strip())


def kinds(path):
    out = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f, strict=True)
        required = {"Kernel_Name", "Start_Timestamp", "End_Timestamp"}
        try:
            headers = reader.fieldnames
        except csv.Error as exc:
            raise ValueError(f"{path}:1: {exc}") from exc
        if not headers or not required.issubset(headers):
            raise ValueError(f"{path}:1: required headers: {', '.join(sorted(required))}")
        if len(headers) != len(set(headers)):
            raise ValueError(f"{path}:1: duplicate headers")
        while True:
            try:
                r = next(reader)
            except StopIteration:
                break
            except csv.Error as exc:
                raise ValueError(f"{path}:{reader.reader.line_num}: {exc}") from exc
            try:
                if None in r or any(value is None for value in r.values()):
                    raise ValueError("row does not match header width")
                k = kind(r["Kernel_Name"])
                if not k:
                    raise ValueError("empty kernel name")
                start, end = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
                if start < 0 or end < start:
                    raise ValueError("timestamps must satisfy 0 <= start <= end")
                ms = (end - start) / 1e6
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError(f"{path}:{reader.line_num}: {exc}") from exc
            out[k] = out.get(k, 0.0) + ms
        if not out:
            raise ValueError(f"{path}:{reader.line_num or 1}: no kernel rows")
    return out


ap = argparse.ArgumentParser()
ap.add_argument("short"); ap.add_argument("long")
ap.add_argument("--mult", type=float, default=4.0); ap.add_argument("--top", type=int, default=12)
ap.add_argument("--phase", action="append", default=[], help="NAME=REGEX (replaces the default phases)")
a = ap.parse_args()
phases = [tuple(p.split("=", 1)) for p in a.phase] or PHASES
try:
    s, l = kinds(a.short), kinds(a.long)
except (ValueError, OSError, csv.Error) as exc:
    ap.error(str(exc))
def phase(k):
    return next((n for n, rx in phases if re.search(rx, k)), "other")
print(f"phase | short ms | long ms | long - {a.mult:g} x short")
names = [n for n, _ in phases] + ["other"]
for n in names + ["total"]:
    xs = sum(v for k, v in s.items() if n == "total" or phase(k) == n)
    xl = sum(v for k, v in l.items() if n == "total" or phase(k) == n)
    print(f"{n} | {xs:.1f} | {xl:.1f} | {xl - a.mult * xs:+.1f}")
rows = sorted(((l.get(k, 0.0) - a.mult * s.get(k, 0.0), k) for k in set(s) | set(l)), reverse=True)
# Use full keys for arithmetic and classification. Expand colliding display labels.
labels = {k: k if len(k) <= 90 else k[:87] + "..." for _, k in rows}
counts = {}
for label in labels.values():
    counts[label] = counts.get(label, 0) + 1
labels = {k: k if counts[label] > 1 else label for k, label in labels.items()}
print(f"kinds with the largest excess (top {a.top}):")
for e, k in rows[:a.top]:
    print(f"  {e:+9.1f} | {s.get(k, 0.0):9.1f} | {l.get(k, 0.0):9.1f} | {phase(k)} | {labels[k]}")
