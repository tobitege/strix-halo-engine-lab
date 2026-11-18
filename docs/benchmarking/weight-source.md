# Weight-source timing and quality kit

Paired ABAB timing and teacher-forced full-vocabulary quality between arms of this engine that differ only in
their weight source or expert arithmetic: [`scripts/bench/weight-source/README.md`](../../scripts/bench/weight-source/README.md).
Timing: `run-fair.py` runs the ABAB matrix and `summarize.py` reports clocks per token; size distinct inputs with
`prepare-prompts.py` and a local tokenizer. Quality: `strix-cuda-qwen-logits` exports selected rows and
`quality.py compare` / `receipt` report KL, top-1 agreement and dNLL against a reference arm (diagnostics only;
a FAST qualification still needs its frozen gate).
