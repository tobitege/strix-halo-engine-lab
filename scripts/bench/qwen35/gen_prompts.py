#!/usr/bin/env python3
"""Deterministic prompt set for qwen35 block-drafting measurements (no model, no network).
Classes: code (multi-turn code edit: the answer re-emits a modified file), prose, json (CSV -> JSON).
Writes prompts/<class>-<n>.json ({"class", "messages"}) and prompts/manifest.json. Run: python gen_prompts.py [outdir]
Tokenize with tokenize_prompts.py once the model's tokenizer exists."""
import json, random, sys, pathlib

out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).parent / "prompts")
out.mkdir(parents=True, exist_ok=True)


def make_module(rng, n_funcs):
    names = ["load", "parse", "merge", "filter", "score", "rank", "dedupe", "flatten", "window", "bucket", "scale", "clip"]
    rng.shuffle(names)
    lines = ["import csv", "import json", "import math", "from dataclasses import dataclass", "", "", "@dataclass", "class Record:",
             "    key: str", "    value: float", "    weight: float = 1.0", "", ""]
    for name in names[:n_funcs]:
        a, b = rng.choice(["items", "rows", "records"]), rng.choice(["limit", "factor", "offset"])
        lines += [f"def {name}_{a}({a}, {b}=1):", f'    """{name.capitalize()} the {a} using {b}."""', "    result = []",
                  f"    for index, item in enumerate({a}):", f"        if item.weight * {b} > {rng.randint(1, 9)}:",
                  f"            result.append(Record(item.key, item.value * {b}, item.weight))",
                  "        elif index % 2 == 0:", "            result.append(item)", "    return result", "", ""]
    return "\n".join(lines).rstrip() + "\n"


def edit_turns(src, rng, k):
    """k edit requests with the exactly-applied answers (synthetic assistant turns)."""
    msgs = [{"role": "user", "content": "Here is utils.py.\n\n```python\n" + src + "```\n\nRemember it; I will ask for edits."},
            {"role": "assistant", "content": "Understood. Send the first edit."}]
    cur = src
    for i in range(k):
        old = f"def {cur.split('def ')[1 + i % 3].split('(')[0]}("
        new = old.replace("def ", "def edited_")
        cur2 = cur.replace(old, new)
        msgs += [{"role": "user", "content": f"Rename `{old[4:-1]}` to `{new[4:-1]}` everywhere and return the complete updated file in one code block."},
                 {"role": "assistant", "content": "```python\n" + cur2 + "```"}]
        cur = cur2
    last = "Now add type hints (`float`, `int`, `list[Record]`) to every function signature, keep everything else, and return the complete file in one code block."
    msgs.append({"role": "user", "content": last})
    return msgs


def csv_json(rng, rows):
    hdr = "id,name,city,age,score,active"
    cities = ["Lyon", "Porto", "Kyoto", "Leeds", "Turin", "Graz", "Cork", "Bern"]
    names = ["Ada", "Bruno", "Chen", "Dalia", "Emil", "Fatou", "Goran", "Hana", "Ivo", "Jun", "Kai", "Lena"]
    lines = [hdr] + [f"{i},{rng.choice(names)}{i},{rng.choice(cities)},{rng.randint(18, 70)},{rng.randint(0, 1000) / 10:.1f},{rng.choice(['true', 'false'])}" for i in range(1, rows + 1)]
    return "\n".join(lines)


prose = ["Write a 600-word essay on why the sea is salty, aimed at curious teenagers.",
         "Tell a short story about a lighthouse keeper who finds a message in a bottle. About 600 words.",
         "Explain how a transformer language model produces its next token, in plain prose without bullet points, about 500 words."]

manifest = []
rng = random.Random(27)
for n, (funcs, k) in enumerate([(6, 2), (9, 2), (12, 1)]):
    src = make_module(rng, funcs)
    msgs = edit_turns(src, rng, k)
    p = out / f"code-{n}.json"
    p.write_text(json.dumps({"class": "code", "messages": msgs}, indent=1))
    manifest.append(p.name)
for n, rows in enumerate([40, 80]):
    csv_text = csv_json(rng, rows)
    msgs = [{"role": "user", "content": "Convert this CSV to a JSON array of objects (numbers as numbers, booleans as booleans). Output only the JSON.\n\n" + csv_text}]
    p = out / f"json-{n}.json"
    p.write_text(json.dumps({"class": "json", "messages": msgs}, indent=1))
    manifest.append(p.name)
for n, t in enumerate(prose):
    p = out / f"prose-{n}.json"
    p.write_text(json.dumps({"class": "prose", "messages": [{"role": "user", "content": t}]}, indent=1))
    manifest.append(p.name)
(out / "manifest.json").write_text(json.dumps(manifest, indent=1))
print("wrote", len(manifest), "prompts to", out)
