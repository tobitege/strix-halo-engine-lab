#!/usr/bin/env python3
"""Chat-template + tokenize the prompt set (thinking off) into <name>.ids (comma-separated ids) for strix-cuda-qwen35-spec --prompt-file.
Usage: python tokenize_prompts.py MODEL_DIR [prompts_dir]   (MODEL_DIR holds tokenizer.json + chat_template.jinja)"""
import json, sys, pathlib
from transformers import AutoTokenizer

model = sys.argv[1]
d = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else pathlib.Path(__file__).parent / "prompts")
tok = AutoTokenizer.from_pretrained(model)
for name in json.loads((d / "manifest.json").read_text()):
    msgs = json.loads((d / name).read_text())["messages"]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True)
    ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids
    (d / name.replace(".json", ".ids")).write_text(",".join(map(str, ids)))
    print(name, len(ids), "tokens")
