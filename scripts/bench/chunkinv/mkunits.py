#!/usr/bin/env python3
"""mkunits.py: independent long-context gate units for the MiniMax invariant-order gate (gate32b).

Six units of different content (prose, docs, code, json, chat, mixed), each built only from files of this repository
at the given checkout (disjoint file sets, so the units are independent texts), chat formatted with thinking disabled.
Every unit carries seven checkable questions: five planted access codes at depths ~8/30/52/74/92 % (needle retrieval),
one two-hop fact pair at ~20 % / ~65 % (long QA) and one question over the five codes (largest code). The document
fills the unit to exactly --tokens tokens, so the gate's last 256 rows are mostly document text plus a short answer cue.

usage: mkunits.py --src REPO_CHECKOUT --tokenizer MODEL/tokenizer.json --template MODEL/chat_template.jinja --out DIR
                   [--tokens 32768] [--warm 256]
The head (system + developer defaults of the model's template, the user turn up to the document) and the tail (end of the
user turn, generation prompt with thinking_mode=disabled) are rendered by the model's own chat template (jinja2).
writes DIR/<unit>.u32 (u32 ids), DIR/<unit>.answers.json and DIR/units.txt; --warm also writes DIR/w<N>.u32.
Deterministic: same checkout + same tokenizer -> same ids (the needles come from a seeded RNG per unit name).
"""
import argparse, glob, json, os, random, re, sys

import numpy as np
import tokenizers

BOS, EOS, THINK_END = "]~b]", "[e~[", "</mm:think>"
NAMES = ["amber", "cobalt", "crimson", "jade", "ivory", "onyx", "saffron", "teal", "umber", "violet", "slate", "coral"]
ITEMS = ["the blue ledger", "the brass compass", "the silver key ring", "the oak seal", "the green atlas",
         "the copper lamp", "the glass prism", "the iron stamp"]
WINGS = ["Halden", "Merrow", "Tarsk", "Quillon", "Brevik", "Ostara", "Vellin", "Corwen", "Ysolde", "Pargo", "Linmar",
         "Desh", "Farrow", "Galt", "Hesper", "Iver", "Kestrel", "Morrow", "Norhold", "Rivet"]
DEPTHS = [0.08, 0.30, 0.52, 0.74, 0.92]
HOP_DEPTHS = [0.20, 0.65]


def files(src, pats, excl=(), max_bytes=None):
    out = set()
    for p in pats:
        out.update(glob.glob(os.path.join(src, p), recursive=True))
    keep = []
    for f in sorted(out):
        rel = os.path.relpath(f, src).replace(os.sep, "/")
        if os.path.isfile(f) and not any(re.search(e, rel) for e in excl) and (max_bytes is None or os.path.getsize(f) <= max_bytes):
            keep.append((rel, f))
    return keep


def read(f):
    return open(f, encoding="utf-8", errors="replace").read()


def concat(fs, limit=400_000):
    parts, n = [], 0
    for rel, f in fs:
        t = "## " + rel + "\n\n" + read(f).strip() + "\n\n"
        parts.append(t)
        n += len(t)
        if n > limit:
            break
    return "".join(parts)


def chat_text(fs, limit=400_000):
    """A multi-turn transcript: the user asks about the next section, the assistant answers with its text."""
    turns, n = [], 0
    for rel, f in fs:
        for sec in re.split(r"\n(?=#{1,3} )", read(f)):
            sec = sec.strip()
            if len(sec) < 200:
                continue
            title = sec.splitlines()[0].lstrip("# ").strip()[:120]
            t = f"{BOS}user\nPlease walk me through \"{title}\" from {rel}.{EOS}\n{BOS}ai\n{sec}{EOS}\n"
            turns.append(t)
            n += len(t)
            if n > limit:
                return "".join(turns)
    return "".join(turns)


def mixed_text(lists, limit=400_000, piece=1500):
    """Round-robin pieces (line aligned, ~piece chars) of several file lists."""
    texts = [concat(l, limit) for l in lists]
    pos, out, n = [0] * len(texts), [], 0
    while n < limit and any(p < len(t) for p, t in zip(pos, texts)):
        for k, t in enumerate(texts):
            if pos[k] >= len(t):
                continue
            e = t.find("\n", pos[k] + piece)
            e = len(t) if e < 0 else e + 1
            out.append(t[pos[k]:e])
            n += e - pos[k]
            pos[k] = e
    return "".join(out)


def sources(src):
    es = files(src, ["docs/**/*.es.md", "*.es.md"])
    return {
        "prose": concat(files(src, ["docs/*.md"], excl=[r"\.es\.md$", r"^docs/minimax", r"README"])),
        "docs": concat(files(src, ["docs/reference/**/*.md", "docs/getting-started/**/*.md", "docs/operations/**/*.md",
                                   "docs/models/**/*.md"], excl=[r"\.es\.md$"])),
        "code": concat(files(src, ["src/deepseek41*.cpp"])),
        "json": concat(files(src, ["**/*.json"], excl=[r"^docs/optimizations\.json$", r"node_modules"], max_bytes=64_000)),
        "chat": chat_text(files(src, ["docs/minimax*.md", "docs/minimax/**/*.md", "docs/engine/**/*.md"],
                                excl=[r"\.es\.md$"])),
        "mixed": mixed_text([es, files(src, ["tools/**/*.py"]), files(src, ["scripts/**/*.sh"])]),
    }


def needle_line(unit, sentence):
    if unit == "code":
        return "// " + sentence
    if unit == "json":
        return '"note": "' + sentence + '",'
    return sentence


def render(template, content):
    """Render one user message through the model's template; return (head, tail) around the DOC placeholder."""
    import jinja2.sandbox

    def fail(msg):
        raise SystemExit("template: " + msg)

    env = jinja2.sandbox.ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = fail
    out = env.from_string(open(template, encoding="utf-8").read()).render(
        messages=[{"role": "user", "content": content}], add_generation_prompt=True, thinking_mode="disabled")
    head, tail = out.split("@@GATE32B-DOC@@")
    return head, tail


def build_unit(tk, template, unit, text, total):
    rng = random.Random("gate32b-" + unit)
    names = rng.sample(NAMES, 5)
    codes = rng.sample(range(10000, 100000), 5)
    item, wing = rng.choice(ITEMS), rng.choice(WINGS)
    cab = rng.choice("BCDFGHJKLMNPRSTVWXZ") + str(rng.randint(2, 9))
    qs = [f"Q{k + 1}: What is the access code of the {names[k]} vault?" for k in range(5)]
    qs.append(f"Q6: In which wing is the cabinet that holds {item}?")
    qs.append("Q7: Which of the five vault access codes is the largest?")
    answers = [str(c) for c in codes] + [wing, str(max(codes))]
    user = ("Read the document below. Notes for the record are planted at different places in it. "
            "After the document, answer these questions using only the document:\n" + "\n".join(qs) +
            "\nAnswer with exactly one line per question in the form A1=<answer>, and no other text.\n\n<document>\n"
            "@@GATE32B-DOC@@\n</document>\n\nNow answer Q1 to Q7, one line each, A1=... to A7=...")
    head, tail = render(template, user)
    assert tail.endswith(THINK_END), "template: generation prompt without the thinking switch"
    hid, tid = tk.encode(head).ids, tk.encode(tail).ids
    n = total - len(hid) - len(tid)
    enc = tk.encode(text)
    if len(enc.ids) < n + 2000:
        raise SystemExit(f"{unit}: only {len(enc.ids)} tokens of source text for {n}")
    facts = [(DEPTHS[k], f"Note for the record: the access code of the {names[k]} vault is {codes[k]}.") for k in range(5)]
    facts += [(HOP_DEPTHS[0], f"Note for the record: {item} is kept in cabinet {cab}."),
              (HOP_DEPTHS[1], f"Note for the record: cabinet {cab} stands in the {wing} wing.")]
    for frac, sent in sorted(facts, reverse=True):  # insert from the back so earlier offsets stay valid
        at = enc.offsets[int(frac * n)][0]
        nl = text.find("\n", at)
        nl = len(text) if nl < 0 else nl
        if nl - at > 400:  # long lines (minified JSON): insert at the token boundary itself
            nl = at
        text = text[:nl] + "\n" + needle_line(unit, sent) + "\n" + text[nl:]
    enc = tk.encode(text)
    doc = enc.ids[:n]
    ids = np.array(hid + doc + tid, dtype="<u4")
    assert ids.size == total, (unit, ids.size)
    depth = {}
    for _, sent in facts:
        c = text.find(sent)
        k = next(i for i, o in enumerate(enc.offsets) if o[0] >= c)
        assert k < n, (unit, sent, k, n)
        depth[sent] = round((len(hid) + k) / total, 3)
    dec = tk.decode(ids.tolist(), skip_special_tokens=False)
    for _, sent in facts:
        assert sent in dec, (unit, "needle lost", sent)
    return ids, {"unit": unit, "tokens": total, "questions": qs, "answers": answers,
                 "depths": [depth[s] for _, s in facts], "head_tokens": len(hid), "tail_tokens": len(tid)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokens", type=int, default=32768)
    ap.add_argument("--warm", type=int, default=0)
    a = ap.parse_args()
    tk = tokenizers.Tokenizer.from_file(a.tokenizer)
    for t in (BOS, EOS, THINK_END):
        if len(tk.encode(t).ids) != 1:
            raise SystemExit(f"tokenizer: {t!r} is not one token")
    os.makedirs(a.out, exist_ok=True)
    lines = []
    for unit, text in sources(a.src).items():
        ids, meta = build_unit(tk, a.template, unit, text, a.tokens)
        ids.tofile(os.path.join(a.out, unit + ".u32"))
        json.dump(meta, open(os.path.join(a.out, unit + ".answers.json"), "w"), indent=1)
        lines.append(f"{unit} {ids.size} tokens, bos {ids[0]}, depths {meta['depths']}, head {meta['head_tokens']}, "
                     f"tail {meta['tail_tokens']}, source {len(text)} chars")
    if a.warm:
        w = tk.encode(f"{BOS}user\n" + concat(files(a.src, ["README.md"]))[:4000]).ids[:a.warm]
        np.array(w, dtype="<u4").tofile(os.path.join(a.out, f"w{a.warm}.u32"))
        lines.append(f"w{a.warm} {len(w)} tokens (warm-up session)")
    open(os.path.join(a.out, "units.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
