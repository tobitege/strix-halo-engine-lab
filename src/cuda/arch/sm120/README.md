# src/cuda/arch/sm120: Blackwell (compute capability 12.0) specialisations

Code here uses what only sm_120 has (block-scaled FP4/FP8 `mma.sync ... kind::mxf4 / kind::f8f6f4`, built for
`sm_120a`) and is chosen at run time through `src/cuda/arch/capability.hpp`,
never by model: a shared core kernel keeps its portable sm_80+ path and the sm_120 path is a policy of that core.

State on main: empty. Every kernel on main uses instructions that sm_86, sm_89 and sm_120 share (INT8 / FP16 /
BF16 `mma.sync`, `ldmatrix`, `cp.async`; FP8 E4M3 `mma.sync` from sm_89), so nothing is Blackwell-only yet. The
first entries are expected from the NVFP4 replica pilot (DeepSeek routed experts) once it is qualified.
