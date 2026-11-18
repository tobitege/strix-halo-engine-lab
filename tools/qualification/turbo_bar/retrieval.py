"""Retrieval matrix builder (needle in a haystack, scored by the generated answer, never by a next-token logit).

  python -m tools.qualification.turbo_bar.retrieval --out retrieval.jsonl --chunk 4096 \
      [--tokenize-cmd "TOKENIZER MODEL encode"] [--lengths 8192:16,32768:16,131072:4] [--template-offset 0]

Factors per case: prefix mode new | restored (identical haystack sent once before, so the server can restore a
cached prefix; the runner records the receipt fields that show a hit), concurrency C1 | C8 (8 different keys on
the same haystack sent concurrently), chunk cut: the needle record straddles a prefill chunk boundary | mid-chunk
control. Token positions are measured with --tokenize-cmd when given (else estimated from chars-per-token).
``--template-offset`` is the number of chat-template tokens the server puts before the user text.
"""
import argparse
import itertools
import sys

from tools.qualification.turbo_bar import tasks
from tools.qualification.turbo_bar.common import canon, sha256_bytes, tokenize, write_jsonl

COMBOS = [(m, c, k) for m in ("new", "restored") for c in (1, 8) for k in ("straddle", "mid")]
PROBE_COMBOS = [("new", 1, "straddle"), ("new", 1, "mid"), ("restored", 1, "straddle"), ("new", 8, "straddle")]
PAD_WORDS = ["note", "entry", "page", "folio", "memo", "item", "slip", "tag"]


def _prefix(parts, idx):
    text = tasks.render(parts[:idx])
    if idx > 0:
        text += "\n" if (idx - 1) % 7 == 6 else " "
    return tasks.HEADER + text


def make_case(seed, length, mode, conc, cut, chunk, cmd=None, cpt=4.0, template_offset=0):
    count = (lambda t: len(tokenize(cmd, t))) if cmd else (lambda t: int(len(t) / cpt))
    for _ in range(3 if cmd else 1):
        rng = tasks._seed("retrieval", seed, length)
        g = tasks.gen_needle(rng, length, cpt, 0.5)
        if not cmd:
            break
        total = count(tasks.wrap_prompt(tasks.render(g["parts"]), g["key"]))
        if abs(total - length) <= 0.02 * length:
            break
        cpt = len(tasks.render(g["parts"])) / max(1, total - 60)
    rec = tasks.record_sentence(g["key"], g["value"])
    rec_tokens = max(1, count(rec))
    others = [(n, v) for i, (n, v) in sorted(g["records"].items()) if i != g["needle_idx"]]
    base = list(g["parts"])
    base[g["needle_idx"]] = tasks._filler(rng, [n for n, _ in others[:20]] or ["Rune Falk"])
    nb = max(2, length // chunk)
    # Interior boundaries are k * chunk for k = 1 .. nb - 1 (8K with 4096-token chunks has exactly one);
    # the mid-chunk control at boundary + chunk / 2 then stays inside the context.
    k = tasks._seed("retrieval-boundary", seed, length).randint(1, max(1, nb - 1))
    boundary = k * chunk
    desired = boundary - rec_tokens // 2 if cut == "straddle" else boundary + chunk // 2 - rec_tokens // 2
    desired -= template_offset
    cum, j = 0, 0
    while j < len(base) and cum < (desired - 25) * cpt:
        cum += len(base[j]) + 1
        j += 1
    pad, parts, start = 0, base, 0
    for _ in range(10):
        pad_s = [] if not pad else [" ".join(PAD_WORDS[i % len(PAD_WORDS)] for i in range(pad)) + "."]
        parts = base[:j] + pad_s + [rec] + base[j:]
        idx = j + len(pad_s)
        start = count(_prefix(parts, idx))
        err = desired - start
        if abs(err) <= 3 or not cmd:
            break
        if err < 0 and pad == 0 and j > 1:
            j -= 1 + int(-err * cpt // 120)
            j = max(1, j)
            continue
        pad = max(0, pad + int(round(err / 1.2)))
    end = start + rec_tokens
    text = tasks.render(parts)
    s_abs, e_abs = start + template_offset, end + template_offset
    crosses = any(s_abs < b * chunk < e_abs for b in range(1, length // chunk + 2))
    meta = {"boundary_token": boundary, "needle_start": s_abs, "needle_end": e_abs, "needle_tokens": rec_tokens,
            "crosses_boundary": crosses, "measured": bool(cmd), "chunk": chunk, "cut_ok": crosses == (cut == "straddle"),
            "prompt_tokens": count(tasks.wrap_prompt(text, g["key"])) if cmd else None}
    keys = [(g["key"], g["value"])]
    if conc == 8:
        keys += tasks._seed("retrieval-keys", seed, length).sample(others, 7)
    cid = f"ret-{length}-{seed:02d}-{mode}-C{conc}-{cut}"
    warm = tasks.msgs(tasks.wrap_prompt(text, others[0][0])) if mode == "restored" else None
    items = []
    for n, (name, value) in enumerate(keys):
        messages = tasks.msgs(tasks.wrap_prompt(text, name))
        items.append({"id": f"{cid}-k{n}", "kind": "retrieval", "split": "retrieval", "group": cid, "primary": n == 0,
                      "length": length, "mode": mode, "concurrency": conc, "cut": cut, "messages": messages,
                      "max_tokens": 16, "check": {"answer": value}, "warmup": warm if n == 0 else None,
                      "meta": meta, "sha": sha256_bytes(canon(messages).encode("utf-8"))})
    return items


def build(lengths, chunk, cmd=None, cpt=4.0, template_offset=0):
    items = []
    for length, n in lengths:
        combos = PROBE_COMBOS if length >= 131072 else COMBOS
        for i in range(n):
            mode, conc, cut = combos[i % len(combos)]
            items += make_case(i, length, mode, conc, cut, chunk, cmd, cpt, template_offset)
    return items


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--tokenize-cmd")
    ap.add_argument("--lengths", default="8192:16,32768:16,131072:4")
    ap.add_argument("--cpt", type=float, default=4.0)
    ap.add_argument("--template-offset", type=int, default=0)
    a = ap.parse_args(argv)
    lengths = [tuple(int(y) for y in x.split(":")) for x in a.lengths.split(",")]
    items = build(lengths, a.chunk, a.tokenize_cmd, a.cpt, a.template_offset)
    write_jsonl(a.out, items)
    bad = sum(1 for x in items if x["primary"] and not x["meta"]["cut_ok"])
    print(len(items), "items;", bad, "cases whose measured cut placement missed its target")
    return 0


if __name__ == "__main__":
    sys.exit(main())
