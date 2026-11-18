"""Compile unchanged production host regions with a deterministic engine double."""
from pathlib import Path
import sys

root, out = map(Path, sys.argv[1:])
app = (root / "apps/minimax_rank_main.cpp").read_text(encoding="utf-8")
bench = (root / "tests/hip_mmx_moe_bench.hip").read_text(encoding="utf-8")


def block(source, marker):
    start = source.index(marker)
    opening = source.index("{", start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[start:end]


start = app.index("    first.resize(static_cast<std::size_t>(prefills));")
end = app.index("    // xb flags:", start)
(out / "prefill.inc").write_text(app[start:end], encoding="utf-8")
# Empty on the unfixed head: the behavioral tests must reject missing qualification.
receipt = 'if (flag == "pfdensemlp") {'
(out / "receipt.inc").write_text(block(app, receipt) if receipt in app else "", encoding="utf-8")
(out / "groups.inc").write_text(block(bench, "std::vector<std::vector<DenseSite*>> dense_v4_groups()"), encoding="utf-8")
engine = (root / "src/minimax/engine/engine.hip").read_text(encoding="utf-8")
accessor = "std::uint64_t Engine::prefill_dense_mlp_count() const"
(out / "accessor.inc").write_text(
    block(engine, accessor) if accessor in engine else accessor + " { return 0; }", encoding="utf-8")
