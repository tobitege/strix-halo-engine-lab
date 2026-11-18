"""Compile the actual engine predicates and dense workspace planner without HIP.

Only resident buffers and device item metadata are doubles. No route decision or
workspace calculation is reimplemented. ENGINE_SOURCE can select the unfixed head
to demonstrate failing resource and decode regressions with the same tests.
"""
from pathlib import Path
import sys

root, out, engine_path = map(Path, sys.argv[1:])
engine = engine_path.read_text(encoding="utf-8")
prefill = (root / "src/minimax/moe/mmx_prefill.hip").read_text(encoding="utf-8")
header = (root / "src/minimax/moe/mmx_prefill.hpp").read_text(encoding="utf-8")


def between(text, start, end):
    begin = text.index(start)
    return text[begin:text.index(end, begin)]


methods = between(engine, "    static constexpr std::size_t kDenseQkvMinRows", "    void stage_attn(")
has_startup = "void require_invariant_dense() const" in methods
if engine_path.resolve() == (root / "src/minimax/engine/engine.hip").resolve():
    # Check that tested methods are wired into startup and both actual projection sites.
    assert has_startup
    assert engine.index("alloc();") < engine.index("require_invariant_dense();") < engine.index("route_levers();")
    assert "; dense_qkv(n, qkv))" in engine and "; dense_o(n, o1))" in engine
out.joinpath("engine.inc").write_text(methods, encoding="utf-8")
out.joinpath("startup.inc").write_text(
    "e.require_invariant_dense();" if has_startup else
    'throw std::runtime_error("missing invariant startup validation");', encoding="utf-8")
parts = [
    between(header, "inline constexpr unsigned kMaxDenseJobs", "\n"),
    between(header, "inline constexpr int kDenseFixedSplits", "\n"),
    between(prefill, "unsigned prefill_dense_splits(", "\nnamespace {"),
    between(prefill, "constexpr std::size_t align256(", "\n"),
    between(prefill, "constexpr std::size_t kDenseItems", "// Pinned host staging"),
    between(prefill, "bool prefill_dense_supported(", "\nvoid enqueue_prefill_dense("),
]
out.joinpath("planner.inc").write_text("\n".join(parts), encoding="utf-8")
