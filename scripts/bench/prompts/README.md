# Kit prompts

`kit.json` contains eight original project-owned prompts: English and Spanish chat,
code, JSON, math, prose, summary and SQL. They use the repository license. They are
not the unpublished README benchmark inputs.

`--prompts FILE` accepts the same JSON array of objects with `id`, `category`, `text`.
IDs must be unique ASCII letters/digits/underscores/hyphens. A JSON object with a
`prompts` array is also accepted. A UTF-8 text file supplies one nonempty prompt per
line, assigned `prompt-1`, etc. JSON supports multiline text via escaped newlines.
No file content is evaluated as shell code. SHA256 covers the exact source bytes.

GGUF prompts use `strix-tokenize ... encode`. EXL3 prompts use the local model's
`tokenizer.json` through Python `tokenizers`, without automatic special tokens.
These are raw completion prompts, not an inferred chat template. Long-context
inputs repeat token IDs and truncate to the requested count; this is recorded in
results. Use the eventual README prompt manifest to compare identical workloads.

`classes/` holds the three decode-class prompts used by `scripts/bench/common/serve-decode.sh`
(prose essay, code edit that returns most of a file, CSV to JSON). Drafting acceptance differs by
an order of magnitude between them, so served decode is always reported per class.
