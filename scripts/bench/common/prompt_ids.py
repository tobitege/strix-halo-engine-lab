#!/usr/bin/env python3
"""prompt_ids.py --out DIR --sizes 8192,32768 (--tokenize-cmd 'BIN MODEL encode' | --tokenizer-json FILE) [--seed FILE]

Writes DIR/p<N>.ids (one id per line: the Qwen --prompt-ids format) for every size (--format csv: DIR/p<N>.csv,
comma-separated, the DeepSeek prefill --prompt format; --format u32: little-endian uint32, GLM / MiniMax): the seed text is
tokenized once (raw text, no template) and repeated then truncated to N tokens. Also writes DIR/prompts.json with
the sha256 of every id list so two runs can prove they read the same prompt. Default seed:
scripts/bench/prompts/seed.txt (the public kit's seed).
"""
import argparse
import hashlib
import json
import os
import shlex
import struct
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_ids(text):
    return [int(t) for t in text.replace(",", " ").split()]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--sizes", required=True, help="comma list; suffix K = 1024 (8K,32K,128K,256K)")
    p.add_argument("--seed", default=os.path.join(HERE, "..", "prompts", "seed.txt"))
    p.add_argument("--tokenize-cmd", help="command that prints ids for a text file appended as last argument")
    p.add_argument("--tokenizer-json", help="HF tokenizer.json (needs the tokenizers package)")
    p.add_argument("--format", choices=("ids", "csv", "u32"), default="ids")
    args = p.parse_args(argv)
    if args.tokenize_cmd:
        seed = parse_ids(subprocess.check_output(shlex.split(args.tokenize_cmd) + [args.seed], timeout=600).decode())
    elif args.tokenizer_json:
        from tokenizers import Tokenizer
        with open(args.seed, encoding="utf-8") as stream:
            seed = Tokenizer.from_file(args.tokenizer_json).encode(stream.read(), add_special_tokens=False).ids
    else:
        raise SystemExit("--tokenize-cmd or --tokenizer-json is required")
    if not seed:
        raise SystemExit("seed tokenized to zero ids")
    os.makedirs(args.out, exist_ok=True)
    meta = []
    for item in args.sizes.split(","):
        item = item.strip().upper()
        n = int(item[:-1]) * 1024 if item.endswith("K") else int(item)
        ids = (seed * (n // len(seed) + 1))[:n]
        if args.format == "u32":
            data = struct.pack("<%dI" % len(ids), *ids)
        else:
            data = (("," if args.format == "csv" else "\n").join(map(str, ids)) + "\n").encode()
        name = "p%d.%s" % (n, args.format)
        with open(os.path.join(args.out, name), "wb") as stream:
            stream.write(data)
        # ids_sha256 is format independent: the same prompt in .ids / .csv / .u32 has the same value.
        meta.append(dict(tokens=n, file=name, sha256=hashlib.sha256(data).hexdigest(),
                         ids_sha256=hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()))
    with open(os.path.join(args.out, "prompts.json"), "w") as stream:
        json.dump(dict(seed=os.path.basename(args.seed), seed_tokens=len(seed), prompts=meta), stream, indent=1)
    print(" ".join("%s:%s" % (m["file"], m["sha256"][:12]) for m in meta))
    return 0


if __name__ == "__main__":
    sys.exit(main())
