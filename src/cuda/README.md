# src/cuda: the CUDA backend

One tree, the same layering as the HIP side (`docs/engine/cuda-backend.md`, which also lists the phase 2 plan):

| Directory | Holds |
|---|---|
| `core/` | model-agnostic device helpers and kernels: PTX wrappers (`ptx.cuh`), reductions and the top-k order (`reduce.cuh`), trellis decode (`trellis.cu`), the gfx1151-identical math tables (`amd_math_tables.cu`, header `include/strix/cuda/amd_math.cuh`) |
| `core/dense/` | shared tiled Q8_0 x A8 IMMA, repack and optional hc_expand / SwiGLU-down paths; caller-owned plan, stream and scratch |
| `arch/` | capability queries (`capability.hpp`); `arch/<sm>/` the architecture specialisations, selected by capability, never by model |
| `models/ds41`, `models/qwen`, `models/glm`, `models/minimax` | what is truly model-specific: the bindings, drivers and kernels not yet lifted into `core/` |

Public headers: `include/strix/cuda/`. CMake: `cmake/StrixCudaBackend.cmake` (target `strix_cuda_core`, core source
lists), used by the DeepSeek configuration and the Qwen-only one. The host layers (serve, session, spec, xport) are
shared with HIP and are not duplicated here.

New CUDA work goes into `core/` behind the interface the HIP core exposes (`src/core`, `src/moe_core`,
`src/attn_core`); a model adds only what is truly its own.
