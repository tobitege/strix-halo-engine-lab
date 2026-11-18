# Dense (non-hybrid) Qwen-style host model declaration

Scope: the checkpoint-format and dense-decoder host layer. It rejects hybrid attention
(`layer_types` other than `full_attention`), so it does NOT admit the released
Qwen3.8-27B, which interleaves 48 gated-delta-net and 16 gated full-attention layers;
hybrid-family integration is planned and has not landed in this checkout.
The shared mapped storage, safetensors/GGUF archive, NVFP4 import view and
RoPE/geometry parser are available for that future integration.

Home: shared (host shared). File ownership, GGUF decoding, safetensors indexing,
NVFP4 views, dense config parsing and dense graph construction are shared code.
The family adapter contains its tensor names and admission rules. Qwen dense and
Gemma bind the shared mapped safetensors reader. Device uploaders consume the
same owning views; this change adds no device kernels or production defaults.

Link `strix_qwen27_model` and include `strix/qwen27_model.hpp`. The entry points
are `parse_hf_config`, `parse_gguf_config`, `load_gguf`, `load_safetensors`,
`required_tensors`, `graph` and explicit `register_family`. Model geometry comes
from assets, without an assumed 27B preset. The dense architecture aliases are
`qwen3`, `qwen38`, `qwen3_8` and `qwen3_8_dense`; admission rejects MoE and hybrid
attention configurations. This is a host declaration, not an inference executable.

The HF reader accepts a root decoder config or nested `text_config`. It retains
hidden/layer/head/KV/intermediate/vocabulary/context sizes, explicit head dimension,
RMS epsilon, bias/tied-head controls, full or partial rotary dimensions, and
default, linear or YaRN rotary parameters. YaRN requires original context length.
The GGUF reader uses architecture-prefixed metadata, with vocabulary from metadata,
token count or embedding shape, checked for agreement. GGUF extents become row-major
`[output,input]` exactly once. Complete required weights and shapes are checked.

GGUF Q4_0, Q4_K, IQ4_NL and IQ4_XS retain their packed bytes and use the existing
shared CPU decoder for row inspection. Mixed scalar F32/F16/BF16 tensors and other
formats supported by that decoder are admitted with per-row block alignment.
Mapped source owners outlive the model through copied tensor views.

NVFP4 supports the following contiguous, unswizzled export layout per linear base:

| Tensor | Type | Shape |
|---|---|---|
| `.weight` | U8, adjacent low/high E2M1 nibbles | `[O,I/2]` |
| `.weight_scale` | F8_E4M3 or F8_E4M3FN | `[O,I/16]` |
| `.weight_scale_2` | positive finite F32 multiplier | scalar or `[1]` |

The decode contract is `(E2M1 * E4M3) * global_scale`, with 16 columns per block.
Zero block scales are valid; negative and NaN block scales are refused. Input
`.pre_quant_scale` vectors and `.input_scale`/`.input_scale_2` scalars are retained
as explicit backend obligations. Row decoding returns stored weights; it does not
apply input smoothing. Inverse global scales, transposed/swizzled scale layouts
and alternative packed tensor names require an explicit format policy first.
Directories use `model.safetensors` or `model.safetensors.index.json`; scales may
reside in different shards from weights. Index membership, ownership, duplicate
names, dtype sizes, overflow, overlap, holes and truncated ranges are checked.

An omitted tied output head aliases the embedding, including its scales. Optional
MTP uses an explicit next-token layer count and `mtp.layers.N` canonical names:
dense transformer weights plus `enorm`, `hnorm` and `[H,2H]` `eh_proj`. GGUF stores
these as appended `blk.L` layers and `nextn.*` roles. The trunk count excludes MTP.
This declaration covers normalized embedding/hidden concatenation and a shared
head. Other auxiliary topologies require their own family semantics and assets.

The graph declares dense projections, causal GQA, per-head Q/K RMS norms, SwiGLU,
pre-norm residuals, KV state and optional MTP to the shared core categories. Backend
kernel selection and architecture specialization remain outside the family.

Host qualification generates all tiny assets locally:

```sh
cmake -S tests/qwen27_host -B build-qwen27-host
cmake --build build-qwen27-host --config Release
ctest --test-dir build-qwen27-host -C Release --output-on-failure
```

The compile-only CUDA header fixture can be checked from a compiler environment:

```sh
nvcc -c -arch=sm_120 -std=c++20 -Iinclude -Isrc tests/qwen27_host/cuda_header_check.cu -o build-qwen27-host/cuda_header_check.obj
```

Synthetic files establish host format and declaration behavior. Real exported model
coverage, device binding, native execution, model fidelity and complete-workload
qualification remain open. No throughput or production exactness claim is made.
