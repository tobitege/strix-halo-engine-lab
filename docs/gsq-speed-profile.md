# Qwen3.8-Flash-Next speed profile (GSQ Q2_0 experts), opt-in

The speed profile is for operators who prefer decode speed and memory over quality. It is **off by default**, and the
production model and command lines are unchanged by it.

## What it is

- **Model:** `hybrid-q2.gguf` = the production UD-Q4_K_XL file with its 48 x 512 routed experts replaced by the GSQ-RCO
  Q2_0 experts (sha256 `106603af0bc020222b90ed73ebfa04fd812280f53ec69135202a6cc71fc98b1e`, 68.29 GB). Dense
  layers, attention, GDN, HC, shared experts, PLE table and head are production's bytes.
- **Flag:** `STRIX_GSQ_Q20_A8=1`. Without it the engine refuses the file's content identity
  (`checkpoint-content-not-qualified`) and falls back. Production files run as before with or without the flag.
- **Command line:** any production command line (decode, MTP, fast prefill) with the model path changed. The expert
  prefill then runs the GSQ FP16 WMMA kernels under `--prefill-numerical-profile fast-prefill-f16wmma-v1`.
- **Prefill form switch:** `STRIX_GSQ_PREFILL_F16` (default on). `=0` keeps the A8 expert prefill forms instead (Q2_0:
  iu8 WMMA, equal to the dot4 reference in every finite word, with NaN sign/payload differences allowed; IQ types:
  dot4). The choice is part of the prefill arithmetic identity
  (`prefill_gsq_moe_arithmetic` in the receipt), so prefix caches of the two forms never mix.

Recipe format: strix-stream/recipe-v1 (tierstream).

## Cost and gains (one Strix Halo node, C1, production MTP)

| | production UD-Q4_K_XL | speed profile |
|---|---|---|
| top-1 agreement vs the Q8 reference (QC64) | 88.1 % | **76.6 % (-11.5 points)**, worst on agent and code prompts |
| KL vs the Q8 reference | 0.238 | **0.716 (3.0x)** |
| decode code1k / prose1k / prose8k, tok/s | 59.8 / 55.4 / 49.2 | 72.8-73.3 / 61.1-64.4 / 61.5 |
| TTFT 1K / 8K / 32K | 1.13 / 8.3 / 29.3 s | 1.02 / 7.97 (median of 5; min 7.87) / 28.5-29.0 s (parity, n = 2) |
| resident weights | 111.3 GB | 68.3 GB |

Quality and production numbers come from the GSQ quality report (llama.cpp QC64 scoring of the exact file; the engine's FP16
expert prefill arithmetic was not rescored on QC64). The profile
fails the current quality criterion; that is the user's trade to make.

## Expert prefill arithmetic

`gsq-moe-prefill:f16wmma-v2`: each weight is formed in FP32 from its block and rounded once to FP16 (exact for Q2_0,
since `d * (q - 1)` is an FP16 value; the IQ types round `block_element` once), activations are FP16, and the products
accumulate in FP32 on WMMA. This is the same arithmetic class as production's fast expert prefill (fast-prefill-f16wmma-v1),
not the A8 dot4 identity. Measured against a float64 oracle (kernel suite): relative RMS 1.6e-4 for FP16 vs 1.8e-2 for
A8 dot4. Generated tokens therefore differ from the A8 path after a few dozen to a few hundred tokens, as production's
fast prefill differs from its A8 prefill.

Kernels: `src/moe_core/hip_moe_gsq_f16_prefill.inc` (gate/up + down, every GSQ gate type x {Q2_0, IQ4_NL} down),
`src/moe_core/hip_moe_gsq_wmma_prefill.inc` (Q2_0 iu8 WMMA, exact to dot4), decoders in `src/moe_core/hip_moe_gsq_weights.inc`. Tests:
`ctest -R strix-hip-moe-prefill-gsq-tests`.

## Receipts

- `prefill_moe_gsq_rows {f16wmma, f16wmma_batch128, iu8wmma, dot4}`: prompt rows x MoE layers per expert prefill form
  (main and MTP receipts). On the speed profile, `f16wmma` = 48 x prompt rows.
- `prefill_gsq_moe_arithmetic`: the effective form name.
