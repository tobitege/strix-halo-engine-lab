"""Integration checks for host paths whose enclosing engines require GPU kernels."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[2]
hip = (root / "src/qwen35/hip/engine.hip").read_text()
assert "g_wide" not in hip
impl = hip[hip.index("struct HipEngine::Impl"):]
assert "WideScratch wide;" in impl
destructor = impl[impl.index("~Impl()"):impl.index("template <class T>")]
assert destructor.index("hipStreamSynchronize") < destructor.index("hipFree")
for member in ("codes", "deltas", "g", "u"):
    assert f"fr(wide.{member})" in destructor
for line in impl.splitlines():
    if "linear(" in line or "glu(" in line:
        assert line.rstrip().endswith(", stream, wide, iu4);")  # every dense call carries both engine-owned scratches

cuda = (root / "src/qwen35/cuda/engine.cu").read_text()
assert cuda.index("pf.sl.clear();") < cuda.index("allocations.clear();")
for name in ("CudaEngine::from_gguf(", "CudaEngine::synthetic("):
    factory = cuda[cuda.index(name):]
    assert factory.index("opt.prefill_slots < 1 || opt.prefill_slots > 2") < factory.index("P.opt = opt;")
assert "prefill effective slots:" in cuda and "prefill-admission:" in cuda
assert "dense_big_fast8_supported()" in cuda
big = (root / "src/cuda_core/dense/dense_big.cu").read_text()
assert "get_table(S, mode," in big
# Track conditional scopes: an else/elif must not inherit the positive guard.
scopes = []
conversions = 0
for number, line in enumerate(big.splitlines(), 1):
    directive = re.match(r"\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)", line)
    if directive:
        kind, condition = directive.groups()
        guarded = condition.strip() == "defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890"
        if kind in ("if", "ifdef", "ifndef"):
            scopes.append(guarded if kind == "if" else False)
        elif kind in ("else", "elif"):
            scopes[-1] = guarded if kind == "elif" else False
        else:
            scopes.pop()
    uses = re.findall(r"\bcvt(?:\.[A-Za-z0-9_]+)*\.e4m3[A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*\b", line.split("//", 1)[0])
    if uses:
        assert any(scopes), f"unguarded E4M3 conversion at dense_big.cu:{number}"
        conversions += len(uses)
assert conversions >= 2, "expected E4M3 conversion instructions"
assert not scopes, "unbalanced preprocessor scopes"
for marker, guard in (("cudaError_t dense_big_split(", "f8"), ("cudaError_t dense_big_gemm(", "a.f8")):
    function = big[big.index(marker):]
    assert function.index(f"if ({guard} && !dense_big_fast8_supported())") < function.index("Scratch& S")
print("engine ownership, dispatch wiring, slot bounds and FAST8 guards: PASS")
