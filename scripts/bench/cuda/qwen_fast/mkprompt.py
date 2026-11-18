#!/usr/bin/env python3
"""Build prompt token files (little-endian uint32) for qwen_tf_logits / gate.sh. Nothing is downloaded.

  mkprompt.py --out DIR NAME=SOURCE [NAME=SOURCE ...] [--tokenize BIN --model GGUF] [--repeat-to N]

SOURCE is a .u32 file (copied), a comma/space separated list of token ids, or, with --tokenize (the strix-tokenize
binary) and --model (a GGUF carrying the tokenizer), a plain text file tokenised by the engine's own tokenizer
("encode" mode). --repeat-to N cycles the tokens until the prompt has N of them (a stress prompt for long contexts;
real text prompts are preferable for quality numbers).
"""
import argparse
import array
import pathlib
import re
import subprocess
import sys


def load(src, tokenize, model):
    p = pathlib.Path(src)
    if p.suffix == ".u32":
        a = array.array("I")
        a.frombytes(p.read_bytes())
        if sys.byteorder == "big":
            a.byteswap()
        return list(a)
    text = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix == ".ids" or re.fullmatch(r"[\d,\s]+", text.strip() or "x"):
        return [int(t) for t in re.split(r"[,\s]+", text.strip()) if t]
    if not tokenize or not model:
        sys.exit(f"{src}: text needs --tokenize BIN --model GGUF")
    res = subprocess.run([tokenize, model, "encode", str(p)], capture_output=True, text=True, check=True)
    return [int(t) for t in re.split(r"[,\s]+", res.stdout.strip()) if t]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenize")
    ap.add_argument("--model")
    ap.add_argument("--repeat-to", type=int)
    ap.add_argument("items", nargs="+", metavar="NAME=SOURCE")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for item in a.items:
        name, _, src = item.partition("=")
        if not name or not src:
            ap.error(f"{item}: expected NAME=SOURCE")
        ids = load(src, a.tokenize, a.model)
        if a.repeat_to and ids:
            ids = (ids * (a.repeat_to // len(ids) + 1))[: a.repeat_to]
        arr = array.array("I", ids)
        if sys.byteorder == "big":
            arr.byteswap()
        (out / f"{name}.u32").write_bytes(arr.tobytes())
        print(f"{name}: {len(ids)} tokens -> {out / (name + '.u32')}")


main()
