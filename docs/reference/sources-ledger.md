# Source and provenance ledger

Status: P0 partially resolved. A reference becomes a canonical source for this
engine only when it is pinned here with an immutable revision, its applicable
license and a verifiable contract.

| Component | Pinned candidate source | Verified license | Independent contract | Status |
| --- | --- | --- | --- | --- |
| Flash-Next architecture | `QwenLM/Qwen3.8-Flash-Next@69885871a64393807d988b27b1b5e380e8f28526`, paper `arXiv:2608.30320` and official checkpoint `Qwen/Qwen3.8-Flash-Next@de4b8e4d43b917e7706784d8bb445c9af86a3540` | Qwen Community License 1.0; use case/distribution terms to be reviewed | QSA/GDN/Gated Residual/n-gram contracts still to be extracted | **Primary source pinned** |
| Flash-Next Q4 configuration and weights | `unsloth/Qwen3.8-Flash-Next-GGUF@38bb39ee97821de2c9009abb7e93950eec396e66` | Derivative; mapping of the official model's conditions pending | Hash/tensor map against the official checkpoint | Candidate; not approved |
| MTP Q8_0 predictor | same snapshot, `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | Pending | Pending | Candidate; not approved |
| Cross-check MTP implementation | `ggml-org/llama.cpp#27836@1d8de7c1b0c7d2febf8f983174d8e6a711e2b1af` | Project MIT license; open/non-canonical PR | Structural and input-combiner cross-check; its attention contains a QSA TODO, so it is NOT an oracle for the full forward | **Secondary; input fusion corroborated, full forward blocked** |
| MoE routing and experts | Official Qwen paper and checkpoint/config pinned above | Yes, subject to the usage policy | Tie-breaking, ordering and layout pending | Partial |
| MTP acceptance algorithm | `huggingface/transformers@ebd5de00b43e94cae2da6d7aeb5a76ecdda51dae`, `src/transformers/generation/utils.py` and `candidate_generator.py` | Apache-2.0 | Greedy acceptance/crop fixture; sampled mode still has RNG pending | **Greedy pinned; sampled RNG open** |
| Tokenizer | Must be extracted from the canonical snapshot and pinned by hash | Pending | Exact IDs in fixtures | Open |
| Chat template/reasoning controls | Must be extracted from the canonical snapshot and pinned by hash | Pending | Golden prompts and tokens | Open |
| N-gram/PLE | Pinned official config (`ngram_size=3`, base 20M, 128 partitions, PLE at layer 2) and Qwen paper/code | Qwen Community License 1.0 | Hashing, rows, conv and lookup timing pending | **P0 open** |
| Hyper-connections/output gate | Official config (`hc_count=4`, `hc_lowrank=320`, sigmoid) and Qwen paper/code | Qwen Community License 1.0 | Residual mixing, scratch and numerical order pending | **P0 open** |
| Partial/interleaved mRoPE | Pinned official config (`partial_rotary_factor=0.25`) and Qwen code | Qwen Community License 1.0 | Multimodal positions and text oracle pending | Open |
| Transformers `qwen4_exp` semantics | `huggingface/transformers@ebd5de00b43e94cae2da6d7aeb5a76ecdda51dae`, `src/transformers/models/qwen4_exp/modeling_qwen4_exp.py` | Apache-2.0 | Independent tiny fixtures for RMSNorm, partial RoPE, recurrent GDN and PLE n-gram hashing | **Pinned for A0.5b; greedy MTP acceptance fixed, sampled RNG/QSA ties remain open** |
| IQ4_XS storage/dequant contract | `ggml-org/llama.cpp@8019dc563b1ecbae6b161a70c3a1359f1b206c1e`, `ggml/src/ggml-common.h` and `ggml/src/ggml-quants.c` | MIT | Independent byte-layout parser, FP16 conversion and full 256-value dequant fixture | **Pinned for A1.1 compact-format experiments; not a canonical checkpoint decision** |
| IQ2_XXS, Q2_K and MXFP4 storage contracts (deepseek41 experts) | `ggml-org/llama.cpp@552f18f912a32ea86edf82e2b76431cb7131538d`, `ggml/src/ggml-common.h` (block layouts, `iq2xxs_grid`, `ksigns_iq2xs`, `kvalues_fp4`), `ggml/src/ggml-quants.c` (`dequantize_row_iq2_xxs`, `dequantize_row_q2_K`, `dequantize_row_mxfp4`) and `ggml/src/ggml-impl.h` (E8M0 conversion) | MIT | The 256-entry IQ2_XXS grid is copied as format data (`include/strix/ds41_iq2_xxs_grid.inc`); sign bytes are recomputed from their parity rule; host decoders and declared A8 products are written from the layouts and compared bitwise with the device on real blocks of the pinned deepseek41 files (`tests/hip_ds41_formats_tests.hip`) | **Pinned for the deepseek41 format work (P2)** |
| GGUF v3 structure | `ggml-org/ggml@353b63b439f27ab2cc19dac97ab1681ba6d2d084`, `docs/gguf.md`; geometries in `ggml-org/llama.cpp@6c7a87f7e5e5cd75b8a641c3471f2dee84a6ed17` | MIT for both projects; public specification/catalog | Parsing, bounds, block and range fixtures | Primary source pinned for A0.2/A0.3 |
| Experimental USB4 DMA-BUF extension | [Linux USB4STREAM source](https://github.com/torvalds/linux/blob/72d3fcf802c45d00b300f25b848a93c3a2bd7c7e/drivers/thunderbolt/stream.c), pinned commit `72d3fcf802c45d00b300f25b848a93c3a2bd7c7e` | GPL-2.0; upstream Intel notices preserved in the integration patch. Project kernel code GPL-2.0-only; public UAPI GPL-2.0 WITH Linux-syscall-note. Exact upstream license/exception texts retained in `lab/direct_gpu_stream/kernel/` | Project-owned experimental ABI, attachment/SG/controller-drain contracts and shared property tests | Enabled real kernel driver object compiled with strict warnings; no module load, DMA execution or GPU/cache qualification |
| deepseek41 decode attention semantics (window ring, KV compression, indexer, candidate pool, sink attention) | `deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170`, `inference/model.py` (`Compressor`, `Indexer`, `select_candidate_blocks`, `Attention`, `precompute_freqs_cis`) and `inference/kernel.py` (`act_quant`, `fp4_act_quant`, `sparse_attn` scale and clamp rules) | MIT | Semantics only, no code copied: `docs/models/deepseek-v4.1/attention.md` declares the engine's own float32 order; host references equal the device bitwise (`tests/hip_ds41_attention_tests.hip`), and an exported small case of the reference code is compared with tolerances (`tests/ds41_attention_tests.cpp`, `STRIX_DS41_ATTN_CASE`) | **Pinned for the deepseek41 attention work (P3)** |
| Vector CRC32C folding | Intel ISA-L `crc/crc32_iscsi_by16_10.asm`, Git blob `6ba8ee3ee6d142e818dfde75b051ed7f01e275a1`, and `crc/crc_const.asm`, Git blob `417ff3e8ba7b4cf2db989a576643bd18fca63e67` ([immutable assembly](https://api.github.com/repos/intel/isa-l/git/blobs/6ba8ee3ee6d142e818dfde75b051ed7f01e275a1), [immutable constants](https://api.github.com/repos/intel/isa-l/git/blobs/417ff3e8ba7b4cf2db989a576643bd18fca63e67)) | BSD-3-Clause; Intel 2011-2020 and 2026 notices retained in the adapted header and `LICENSES/ISA-L-CRC32C.txt` | Reflected Castagnoli CRC32C, init/final xor `0xffffffff`, every payload byte; ISA dispatch and scalar/interleaved fallback preserve the same 32 bits | Opt-in backend; independent CPU oracle corpora and native byte/integrity/lifetime tests passed. Original assembly authors credited; folding constants/sequence adapted to intrinsics |
| Image decoder (PNG, JPEG) for vision input | `nothings/stb` `stb_image.h` v2.30, vendored unmodified at `third_party/stb/stb_image.h`, SHA-256 `594c2fe35d49488b4382dbfaec8f98366defca819d916ac95becf3e75f4200b3` | Public domain (Unlicense) or MIT at the user's choice; text in `LICENSES/stb_image.txt` | Compiled into one translation unit, memory input only, PNG and JPEG only; header-first size limits and refusals in `tests/vision/preprocess_tests.cpp` | In use (host preprocessing) |
| Image resampling (bilinear, bicubic, Lanczos, 8-bit fixed point) | Independent implementation of the separable resampling algorithm published in Pillow (`src/libImaging/Resample.c`); no code copied | Not applicable (no source reused) | Output pixels bit-equal to Pillow 12.3.0 on the generated cases of `tools/vision/make_images.py` | In use (host preprocessing) |

## Approval rule

The experimental selected dense projection adapts wave-transform and ordered
split-reduction organization from the related project-owned `strix_trellis`
tree at `9c28680b46e379abe0ad92a14c4a7611fd7bfd74`, specifically
`src/transforms.hpp` and `src/backend.hpp`. That pinned tree contains no
license or notice file; do not infer an upstream license or independent
authorship for the adapted organization. Distribution terms remain unresolved.
The new FP32 transform/FMA recipe, asset pins and qualification boundary are
documented in [the dense projection contract](../research/dense-trellis-g1.md).
The source model's existing provenance and asset terms continue to apply.

Each row needs a URL/repo, a SHA or immutable revision, the applicable license,
artifact hashes and a verifiable contract. Secondary references, such as
open, non-canonical pull requests, may guide tests and structural checks, but
none of them is the sole oracle for an implementation.
