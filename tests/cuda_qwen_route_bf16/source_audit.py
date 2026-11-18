"""Source/OFF restoration audit. Does not compile or execute CUDA."""
from pathlib import Path
import argparse
import json
import re
import subprocess
import sys

FILES = [
    "include/strix/moe_route_bf16.hpp", "src/hip_moe_route_bf16.inc",
    "src/cuda/models/qwen/qwen_moe.cu", "src/cuda/models/qwen/qwen_prefill_driver.cu",
    "include/strix/cuda/qwen_route_bf16.hpp", "include/strix/cuda/qwen_moe.hpp",
    "cmake/QwenCuda.cmake", "include/strix/prefill_identity.hpp",
    "include/strix/qwen_profile_table.hpp", "src/cuda/models/qwen/qwen_handoff.hpp",
    "src/cuda/models/qwen/qwen_handoff.cpp", "src/hip_moe_q4k_a8_known_rows.hip",
    "src/hip_moe_experts_launcher.hip", "src/moe_core/hip_moe_da16_down_body.inc",
    "src/hip_moe_prefill_commit.inc", "apps/cuda_qwen_prefix_cache.hpp",
    "src/cuda/models/qwen/qwen_prefix_identity.cpp", "tests/cuda/qwen_moe_tests.cu",
    "tests/cuda/qwen_route_bf16_e2e.cu",
    "tests/cuda_qwen_route_bf16/host.cpp",
    "tests/cuda_qwen_route_bf16/annotation.cpp",
    "tests/cuda_qwen_route_bf16/CMakeLists.txt",
    "tests/cuda_qwen_route_bf16/native-ticket.sh",
    "tests/cuda_qwen_route_bf16/source_audit.py",
    "docs/models/qwen3.8/rtx-route-bf16.md",
]
NEW_FILES = set(FILES[-6:]) | {"include/strix/cuda/qwen_route_bf16.hpp",
                             "tests/cuda/qwen_route_bf16_e2e.cu"}
TOKEN = re.compile(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|'
                   r"'(?:\\.|[^'\\])*'|[A-Za-z_][A-Za-z_0-9]*|"
                   r'\d+(?:\.\d*)?(?:[eEpP][+-]?\d+)?[A-Za-z_0-9]*|[^\s]', re.S)

def tokens(source):
    return [m.group() for m in TOKEN.finditer(source)
            if not m.group().startswith(("//", "/*"))]

def close_brace(source, opening):
    depth = 0
    for m in TOKEN.finditer(source, opening):
        if m.group() == "{":
            depth += 1
        elif m.group() == "}":
            depth -= 1
            if depth == 0:
                return m.start()
    raise ValueError("unclosed C++ body")

def body(source, name):
    start = source.index("void " + name + "(")
    opening = source.index("{", start)
    return source[opening + 1:close_brace(source, opening)]

def select_false(source, condition):
    while condition in source:
        start = source.index(condition)
        opening = source.index("{", start)
        end = close_brace(source, opening)
        following = source[end+1:]
        match = re.match(r"\s*else\s*\{", following)
        if not match:
            raise ValueError("OFF source branch requires an explicit else")
        else_open = end + 1 + match.end() - 1
        else_end = close_brace(source, else_open)
        source = source[:start] + source[else_open+1:else_end] + source[else_end+1:]
    return source

# The CUDA backend moved under src/cuda/{core,models/<model>} and include/strix/cuda/ (bytes unchanged apart
# from path names): historical parents are read at their old paths, with the same renames applied.
CUDA_MOVES = [("src/cuda/core/amd_math_tables.cu", "src/cuda/qwen_amd_math_tables.cu"),
              ("src/cuda/core/trellis.cu", "src/cuda/trellis.cu"),
              ("include/strix/cuda/amd_math.cuh", "src/cuda/qwen_amd_math.cuh"),
              ("include/strix/cuda/amd_math_tables.hpp", "src/cuda/qwen_amd_math_tables.hpp"),
              ("src/cuda/models/qwen/", "src/cuda/"), ("src/cuda/models/ds41/", "src/cuda/"),
              ("include/strix/cuda/", "include/strix/cuda_")]
# Literal renames of the move, in order (also applied to comments that name the old paths).
CUDA_PATH_MOVES = [("../../src/cuda/qwen_amd_math", "strix/cuda/amd_math"),
                   ("src/cuda/qwen_amd_math_tables.cu", "src/cuda/core/amd_math_tables.cu"),
                   ("src/cuda/qwen_amd_math_tables.hpp", "include/strix/cuda/amd_math_tables.hpp"),
                   ("src/cuda/qwen_amd_math.cuh", "include/strix/cuda/amd_math.cuh"),
                   ('"qwen_amd_math.cuh"', '"strix/cuda/amd_math.cuh"'),
                   ('"qwen_amd_math_tables.hpp"', '"strix/cuda/amd_math_tables.hpp"'),
                   ("qwen_amd_math_tables.hpp", "amd_math_tables.hpp"), ("qwen_amd_math.cuh", "amd_math.cuh"),
                   ("src/cuda/trellis.cu", "src/cuda/core/trellis.cu"),
                   ("src/cuda/ds41_", "src/cuda/models/ds41/ds41_"), ("src/cuda/qwen_", "src/cuda/models/qwen/qwen_"),
                   ("strix/cuda_", "strix/cuda/"),
                   ('"../hip_moe_route_bf16.inc"', '"../../../hip_moe_route_bf16.inc"')]

def legacy_path(path):
    for new, old in CUDA_MOVES:
        if path.startswith(new):
            return old + path[len(new):]
    return path

def moved_paths(source):
    for old, new in CUDA_PATH_MOVES:
        source = source.replace(old, new)
    return source

def shared_include_paths(source):
    # The historical parent may still include the two-line forwarding shims.
    # Normalize only those includes; every implementation byte is still compared.
    shims = {"hip_moe_" + name + ".inc" for name in (
        "da16_down_body", "f16x_down_body", "f16x_project_body", "gsq_f16_prefill",
        "gsq_weights", "gsq_wmma_prefill", "prefill_core2", "prefill_da16",
        "prefill_g3", "prefill_p1", "prefill_p1da16")}
    source = moved_paths(source)
    return re.sub(r'(?m)^(#include ")([^"\n]+)(")',
                  lambda m: m[1] + ("moe_core/" if m[2] in shims else "") + m[2] + m[3], source)


def checks(source, parent):
    out = {}
    def check(name, ok):
        out[name] = bool(ok)

    helper = source["include/strix/moe_route_bf16.hpp"]
    expected = parent["include/strix/moe_route_bf16.hpp"].replace(
        "#if defined(__HIPCC__)\n__host__ __device__",
        "#if defined(__HIPCC__) || defined(__CUDACC__)\n__host__ __device__")
    check("helper_only_cuda_annotation_no_math_change", helper == expected)
    for name in ["src/hip_moe_route_bf16.inc", "src/hip_moe_q4k_a8_known_rows.hip",
                 "src/hip_moe_experts_launcher.hip", "src/moe_core/hip_moe_da16_down_body.inc",
                 "src/hip_moe_prefill_commit.inc", "include/strix/prefill_identity.hpp",
                 "include/strix/qwen_profile_table.hpp", "src/cuda/models/qwen/qwen_handoff.hpp",
                 "src/cuda/models/qwen/qwen_handoff.cpp"]:
        check("unchanged:" + name, shared_include_paths(source[name]) == shared_include_paths(parent[name]))
    moe = source["src/cuda/models/qwen/qwen_moe.cu"]
    old_moe = parent["src/cuda/models/qwen/qwen_moe.cu"]
    for name in ["down_kernel", "down_rows_kernel", "merge_kernel"]:
        off = select_false(body(moe, name), "if constexpr (RouteBF16)")
        check("off_math_and_guards:" + name, tokens(off) == tokens(body(old_moe, name)))
    for name in ["prologue_kernel", "csr_kernel", "project_kernel", "project_rows_kernel",
                 "native_project_kernel", "native_down_kernel", "commit_kernel"]:
        check("unchanged_kernel:" + name, body(moe, name) == body(old_moe, name))
    check("production_reuses_project_storage_helpers", '#include "../../../hip_moe_route_bf16.inc"' in moe)
    check("both_a8_writers_and_reader_instantiated",
          all(x in moe for x in ["down_rows_kernel<true, RouteBF16>", "down_rows_kernel<false, RouteBF16>",
                                "down_kernel<true, RouteBF16>", "down_kernel<false, RouteBF16>",
                                "enqueue_down_storage<true>", "enqueue_down_storage<false>",
                                "merge_kernel<true>", "merge_kernel<false>"]))
    check("capacity_and_other_pack_checked_before_any_enqueue",
          body(moe, "enqueue").index("validate_route_storage") < body(moe, "enqueue").index("enqueue_prologue")
          and body(moe, "enqueue").index("other_weight_pack") < body(moe, "enqueue").index("enqueue_prologue"))
    check("individual_down_and_merge_capacity_guards",
          "validate_route_storage(v, rows)" in body(moe, "enqueue_down") and
          "validate_route_storage(v, rows)" in body(moe, "enqueue_merge"))
    driver = source["src/cuda/models/qwen/qwen_prefill_driver.cu"]
    old_driver = parent["src/cuda/models/qwen/qwen_prefill_driver.cu"]
    off_alloc = select_false(body(driver, "Prefill::Impl::allocate_block"), "if (route_policy.enabled)")
    check("off_allocations_all_preserved", tokens(off_alloc) == tokens(body(old_driver, "Prefill::Impl::allocate_block")))
    a = old_driver.index("    char buf[384];", old_driver.index("Prefill::Prefill"))
    b = old_driver.index("    m.describe = buf;", a)
    check("off_receipt_format_and_fields_unchanged", old_driver[a:b] in driver)
    check("candidate_receipt_only_when_enabled",
          "if (m.route_policy.enabled) {\n        const auto id = numerical_identity();" in driver)
    check("policy_frozen_before_device",
          driver.index("m.cfg = qwen_options::resolve_route(config);")
          < driver.index("m.route_policy = *m.cfg.route_policy;") < driver.index('strix::device::bind_cuda()'))
    check("all_target_calls_bind_storage",
          "mr.route_bf16 = route_policy.enabled;" in driver and
          "mr.rawdots = {moe_raw, qwen_route::row_bytes(route_policy.enabled)};" in driver)
    check("candidate_capacity_only_no_new_off_admission",
          "mr.rawdot_capacity = route_policy.enabled ? qwen_route::layout(cap, true).bytes : 0U;" in driver)
    check("session_target_bound", "s->target_identity = qwen_route::identity" in driver and
          "seg[i].s->target_identity != qwen_route::identity" in driver)
    check("named_engine_composition_guards", all(x in driver for x in [
        "Composition::node_handoff", "Composition::mtp_sidecar", "Composition::graphs",
        "Composition::cohort", "Composition::overlap", "Composition::untagged_state_import",
        "Composition::prefix_cache", "Composition::other_weight_pack", "Composition::multisegment"]))
    check("prefix_cache_declines_candidate",
          "qwen_route::require_composition(qwen_route::environment(config.moe_route_bf16).enabled" in
          source["apps/cuda_qwen_prefix_cache.hpp"])
    check("cache_pair_explicit_not_hip_fp16",
          all('put("' + x + '"' in source["src/cuda/models/qwen/qwen_prefix_identity.cpp"]
              for x in ["target.prefill", "target.decode", "target.storage", "target.pair"]))
    check("no_silent_global_feature_disable",
          "m.use_graphs = false" not in driver and "setenv(" not in driver and "_putenv" not in driver)
    cmake = source["cmake/QwenCuda.cmake"]
    check("real_cuda_operator_build_closure",
          "add_executable(qwen_route_bf16_operator tests/cuda/qwen_moe_tests.cu)" in cmake and
          "target_link_libraries(qwen_route_bf16_operator PRIVATE strix_qwen_cuda)" in cmake and
          "src/cuda/models/qwen/qwen_moe.cu" in cmake)
    fixture = source["tests/cuda/qwen_moe_tests.cu"]
    check("real_gpu_before_host_operator_gold",
          fixture.index("mo::enqueue(d.rows_view") < fixture.index("ref_prologue(s, clear_words); ref_csr(s);"))
    check("fixture_storage_capacity_real",
          "v.rawdot_capacity = s.view[V_raw].size();" in fixture)
    check("full_same_target_diagnostic",
          all(x in source["tests/cuda/qwen_route_bf16_e2e.cu"] for x in [
              "p.restore_canonical(initial)", "p.commit_cohort({0})", "p.commit(keep)",
              'dump(output/"serial-logits.f32",logits)', "same_state(final,state",
              "p.prefill_chunk(0,prompt", "std::memcmp(result.logits.data()"]))
    payload = source["tests/cuda_qwen_route_bf16/native-ticket.sh"]
    check("native_deadline_includes_configure_and_j12",
          "timeout --signal=TERM --kill-after=5s 460s" in payload and
          "cmake -S . -B" in payload and "-j12 --verbose" in payload and
          payload.index("cmake -S . -B") < payload.index("cmake --build"))
    check("native_source_and_branch_are_pinned",
          'git rev-parse HEAD' in payload and "EXPECTED_SOURCE_BRANCH" in payload and
          "git status --porcelain --untracked-files=all" in payload)
    check("no_ready_or_quality_promotion", "qualification\\\":\\\"unqualified" in driver and
          source["include/strix/prefill_identity.hpp"] == parent["include/strix/prefill_identity.hpp"])
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--parent", required=True)
    ap.add_argument("--mutants", action="store_true")
    args = ap.parse_args()
    source = {p: (args.source / p).read_text(encoding="utf-8") for p in FILES}
    parent = {}
    for p in FILES:
        if p in NEW_FILES:
            continue
        proc = subprocess.run(["git", "show", args.parent + ":" + legacy_path(p)], cwd=args.source,
                              capture_output=True, text=True, encoding="utf-8")
        if proc.returncode == 0:
            parent[p] = proc.stdout
        else:
            print("NO-REFERENCE " + p, file=sys.stderr)
            return 2
    results = checks(source, parent)
    if args.mutants:
        mutations = [
            ("src/cuda/models/qwen/qwen_moe.cu", "current = 0.0F", "current = 1.0F", "off_math_and_guards:merge_kernel"),
            ("src/cuda/models/qwen/qwen_moe.cu", "make_float2(out[0], out[1])", "make_float2(out[1], out[0])", "off_math_and_guards:down_kernel"),
            ("src/cuda/models/qwen/qwen_moe.cu", '#include "../../../hip_moe_route_bf16.inc"', "", "production_reuses_project_storage_helpers"),
            ("src/cuda/models/qwen/qwen_prefill_driver.cu", 'R * kTopK * kHidden, "moe raw dots"', 'R * kHidden, "moe raw dots"', "off_allocations_all_preserved"),
            ("include/strix/moe_route_bf16.hpp", "0x7fffU", "0x8000U", "helper_only_cuda_annotation_no_math_change"),
            ("apps/cuda_qwen_prefix_cache.hpp", "qwen_route::require_composition(qwen_route::environment(config.moe_route_bf16).enabled", "qwen_route::require_composition(false", "prefix_cache_declines_candidate"),
            ("cmake/QwenCuda.cmake", "target_link_libraries(qwen_route_bf16_operator PRIVATE strix_qwen_cuda)", "", "real_cuda_operator_build_closure"),
        ]
        for path, old, new, label in mutations:
            changed = dict(source)
            assert old in changed[path]
            changed[path] = changed[path].replace(old, new)
            results["reject_mutant:" + label] = not checks(changed, parent)[label]
    for p in FILES:
        results["LF:" + p] = b"\r\n" not in (args.source / p).read_bytes()
    print(json.dumps({"checks": len(results), "passed": sum(results.values()),
                      "results": results}, indent=2))
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())
