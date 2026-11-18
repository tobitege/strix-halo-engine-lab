"""Pilot corpus builder: 48 parents (12 each of code / math / prose / tools), nested lengths per parent.

  python -m tools.qualification.turbo_bar.corpus --out DIR --salt SALT --tokenize-cmd "TOKENIZER MODEL encode" \
      --code-files 'src/**/*.py' --prose-files 'docs/**/*.txt' [--lengths 8192,32768] [--search-per-category 3]

The tokenize command gets the path of a UTF-8 text file as its LAST argument and prints token ids.
For each length L of a parent stream ``ids``: prompt = ids[:L], continuation = ids[L:L+256], and 128 spread
positions: [64, L-1) is split into 128 equal bins and bin i holds offset sha256(salt|parent|L|i) mod width.
A position p scores the prediction of prompt[p+1]. Splits SEARCH / CONFIRM: sha256(salt, parent) order per
category. The corpus is frozen by the sha256 recorded in corpus.sha256 (manifest field corpus_sha256).
"""
import argparse
import glob
import hashlib
import json
import random
import sys
from pathlib import Path

from tools.qualification.turbo_bar import tasks
from tools.qualification.turbo_bar.common import canon_sha, sha256_bytes, sha256_file, tokenize, write_json

CATEGORIES = ("code", "math", "prose", "tools")
CONT = 256
N_POS = 128


def spread_positions(salt, parent, length, n=N_POS, lo=64):
    hi = length - 1
    if hi - lo < n:
        raise ValueError("length too short for the spread positions")
    out = []
    for i in range(n):
        a, b = lo + (i * (hi - lo)) // n, lo + ((i + 1) * (hi - lo)) // n
        off = int(hashlib.sha256(f"{salt}|{parent}|{length}|{i}".encode()).hexdigest(), 16) % (b - a)
        out.append(a + off)
    return out


_seed = tasks._seed


def math_pieces(salt, parent):
    rng = _seed("math", salt, parent)
    n = 0
    while True:
        block = []
        for _ in range(8):
            n += 1
            q, ans, steps = tasks.gen_math(rng)
            block.append(f"Problem {n}: {q}\nSolution:\n" + "\n".join(f"- {s}" for s in steps) + f"\nAnswer: {ans}\n")
        yield "\n".join(block) + "\n"


def tools_pieces(salt, parent):
    rng = _seed("tools", salt, parent)
    tl = tasks._json_tools()
    names = sorted(tl)
    n = 0
    while True:
        n += 1
        chosen = rng.sample(names, rng.randint(1, 3))
        lines = [f"### Session {n}", "Available tools:"] + [json.dumps(tl[c][0]) for c in chosen]
        for _ in range(rng.randint(1, 3)):
            c = rng.choice(chosen)
            args = tl[c][1](rng)
            lines.append(f"User: {tl[c][2](args)}")
            lines.append("Assistant: " + json.dumps({"name": c, "arguments": args}))
            lines.append("Tool result: " + json.dumps({"status": rng.choice(["ok", "ok", "queued", "partial"]),
                                                       "id": rng.randint(1000, 99999),
                                                       "echo": args}))
            lines.append(f"Assistant: Done, the {c.replace('_', ' ')} request finished with id "
                         f"{rng.randint(1000, 99999)}.")
        yield "\n".join(lines) + "\n\n"


def build_stream(pieces, cmd, need):
    ids = []
    for text in pieces:
        ids += tokenize(cmd, text)
        if len(ids) >= need:
            return ids[:need]
    raise ValueError(f"sources exhausted before {need} tokens ({len(ids)})")


def ordered_files(patterns, salt):
    files = {}
    for pat in patterns:
        for p in glob.glob(pat, recursive=True):
            if Path(p).is_file():
                files[sha256_file(p)] = p       # identical content counts once
    return [(sha, files[sha]) for sha in sorted(files, key=lambda s: sha256_bytes(f"{salt}|{s}".encode()))]


def build(out, salt, cmd, code_files, prose_files, lengths=(8192, 32768), per_category=12, search_per_category=3):
    out = Path(out)
    for d in ("ids", "positions"):
        (out / d).mkdir(parents=True, exist_ok=True)
    need = max(lengths) + CONT
    pools = {"code": ordered_files(code_files, salt), "prose": ordered_files(prose_files, salt)}
    parents, cases = [], []
    for cat in CATEGORIES:
        names = [f"{cat}-{i:02d}" for i in range(per_category)]
        order = sorted(names, key=lambda n: sha256_bytes(f"{salt}|{n}".encode()))
        split = {n: ("SEARCH" if order.index(n) < search_per_category else "CONFIRM") for n in names}
        for name in names:
            sources = []
            if cat in pools:
                pool = pools[cat]

                def pieces():
                    while pool:
                        sha, path = pool.pop(0)
                        sources.append({"name": Path(path).name, "sha256": sha})
                        yield Path(path).read_text(encoding="utf-8", errors="replace") + "\n\n"
                ids = build_stream(pieces(), cmd, need)
            else:
                ids = build_stream((math_pieces if cat == "math" else tools_pieces)(salt, name), cmd, need)
            parents.append({"parent": name, "category": cat, "split": split[name], "sources": sources})
            for L in lengths:
                case = f"{name}-L{L}"
                files = {
                    "prompt_ids": (f"ids/{case}.prompt.txt", ids[:L]),
                    "cont_ids": (f"ids/{case}.cont.txt", ids[L:L + CONT]),
                    "positions": (f"positions/{case}.positions.txt", spread_positions(salt, name, L)),
                }
                rec = {"case": case, "parent": name, "category": cat, "split": split[name], "length": L}
                for key, (rel, vals) in files.items():
                    sep = "\n" if key == "positions" else " "
                    data = (sep.join(map(str, vals)) + "\n").encode("utf-8")
                    (out / rel).write_bytes(data)
                    rec[key] = rel
                    rec[key + "_sha256"] = sha256_bytes(data)
                cases.append(rec)
    doc = {"schema": 1, "salt": salt, "lengths": list(lengths), "continuation_tokens": CONT, "n_positions": N_POS,
           "tokenize_cmd_sha256": sha256_bytes(cmd.encode("utf-8")), "parents": parents, "cases": cases}
    write_json(out / "corpus.json", doc)
    sha = canon_sha(doc)
    (out / "corpus.sha256").write_text(sha + "\n", encoding="ascii")
    return doc, sha


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--salt", required=True)
    ap.add_argument("--tokenize-cmd", required=True)
    ap.add_argument("--code-files", action="append", default=[])
    ap.add_argument("--prose-files", action="append", default=[])
    ap.add_argument("--lengths", default="8192,32768")
    ap.add_argument("--per-category", type=int, default=12)
    ap.add_argument("--search-per-category", type=int, default=3)
    a = ap.parse_args(argv)
    doc, sha = build(a.out, a.salt, a.tokenize_cmd, a.code_files, a.prose_files,
                     [int(x) for x in a.lengths.split(",")], a.per_category, a.search_per_category)
    print(len(doc["parents"]), "parents,", len(doc["cases"]), "cases; corpus sha256", sha)
    return 0


if __name__ == "__main__":
    sys.exit(main())
