"""GLM RMSNorm extraction identity against the fixed phase-1 tip; no GPU."""
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = "2f0452097"
DEVICE = "src/glm/kernels/glm_tier_a.hip"
ENGINE = "src/glm/engine/engine.hip"


def old(path):
    return subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT, text=True)


def read(path):
    return (ROOT / path).read_text()


def canon(text):
    text = re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"\s+", "", text)


def body(source, name):
    match = re.search(r"\b" + re.escape(name) + r"\([^;{}]*\)\s*\{", source)
    assert match, name
    start = match.end() - 1
    depth, end = 1, start + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


def main():
    before, after = old(DEVICE), read(DEVICE)
    shared = read("src/ops/norm/norm_device.hpp")
    assert "template <class Traits, unsigned L>" in shared
    signature = shared[shared.index("__global__ void rmsnorm_kernel"):shared.index(" {", shared.index("__global__ void rmsnorm_kernel"))]
    for typed, concrete in (("typename Traits::Input", "float"), ("typename Traits::Output", "float"),
                            ("typename Traits::ScaleFormat", "WType"), ("typename Traits::Scale", "void")):
        signature = signature.replace(typed, concrete)
    original_signature = re.search(r"__global__ void rmsnorm_kernel[^{}]+(?=\{)", before)[0]
    assert canon(signature) == canon(original_signature), "kernel signature/attributes"
    normalized = body(shared, "rmsnorm_kernel")
    # Remove only these compile-time type/order checks, then undo the three
    # adapter qualifications. Every arithmetic token must match the baseline.
    for check in (
        "static_assert(Traits::reduction_order == ReductionOrder::lane_strided_fma_lds_halving);",
        "static_assert(std::is_same_v<typename Traits::Input, float>);",
        "static_assert(std::is_same_v<typename Traits::Output, float>);",
    ):
        assert normalized.count(check) == 1
        normalized = normalized.replace(check, "")
    for new, previous in (("Traits::template reduce<L>", "lds_tree<L>"),
                          ("Traits::inv_sqrt", "inv_sqrt"), ("Traits::load", "wload")):
        assert normalized.count(new) == 1
        normalized = normalized.replace(new, previous)
    assert canon(normalized) == canon(body(before, "rmsnorm_kernel")), "device arithmetic"
    for helper in ("wload",):
        assert canon(body(before, helper)) == canon(body(after, helper)), helper
    # lds_tree now runs levels 16..1 in wave 0 registers with the same pairing (core/ordered_reduce.hpp); its bits
    # against the LDS loop are checked on device by tests/hip_exact_reduce_tests.hip.
    assert "ordered_reduce::tree_broadcast<L>(s, part, threadIdx.x)" in body(after, "lds_tree"), "lds_tree"
    traits = read("src/glm/kernels/glm_norm.hpp")
    for alias in ("using Input = float;", "using Output = float;", "using Scale = void;",
                  "using ScaleFormat = WType;", "static constexpr float epsilon = kEps;"):
        assert alias in traits, alias
    assert canon(body(traits, "lanes")) == "{returnnorm_lanes(n);}"
    assert canon(body(traits, "launch")) == "{rmsnorm(x,w,wt,n,eps,y,rows,stream);}"
    assert canon(body(after, "reduce")) == "{returnlds_tree<L>(s,part);}"
    assert canon(body(after, "inv_sqrt")) == "{returnstrix::glm::inv_sqrt(x);}"
    assert canon(body(after, "load")) == "{returnwload(w,wt,i);}"
    launch = body(after, "rmsnorm").replace("GlmNorm::lanes(n)", "norm_lanes(n)")
    launch = launch.replace("ops::norm::rmsnorm_kernel<GlmNormDeviceTraits, ", "rmsnorm_kernel<")
    assert canon(launch) == canon(body(before, "rmsnorm")), "launch geometry/arguments"
    engine = read(ENGINE).replace("D::GlmNorm::run(", "D::rmsnorm(")
    engine = engine.replace('"../kernels/glm_norm.hpp"', '"glm/kernels/glm_tier_a.hpp"')
    assert canon(engine) == canon(old(ENGINE)), "engine call order/arguments"
    assert read(ENGINE).count("D::GlmNorm::run(") == old(ENGINE).count("D::rmsnorm(") == 14
    # These helpers carry sqrt/partition/scale conversion semantics.
    for path in ("src/glm/common/glm_math.hpp", "src/glm/rtxpf/import/glm_dims.hpp"):
        assert read(path) == old(path), path
    for engine_path in ("src/hip_ds41_layer.hip", "src/hip_gdn_launcher.hip",
                        "src/hip_hyper_mix_launcher.hip", "src/minimax/engine/engine.hip",
                        "src/minimax/engine/mm_kernels.hpp"):
        assert read(engine_path) == old(engine_path), engine_path
    changed = subprocess.check_output(["git", "diff", "--name-only", BASE, "--", "src"], cwd=ROOT, text=True).splitlines()
    new_files = {"src/glm/kernels/glm_norm.hpp", "src/ops/norm/norm.hpp", "src/ops/norm/norm_device.hpp", "src/ops/norm/reference.hpp"}
    assert set(changed) <= {DEVICE, ENGINE, "src/glm/kernels/glm_tier_a.hpp"} | new_files, changed
    print("PASS: GLM norm body, helpers, 128/256 launches and 14 calls unchanged after trait normalization")
    print("PASS: GLM math identity and other engine adoption points unchanged (source check only)")


if __name__ == "__main__":
    main()
