"""Adapter wiring checks complement executable host policy tests; no native claims."""
from pathlib import Path
import sys


def check(serve, hip, case):
    if case == "capacity":
        admission = serve.split("std::vector<Running*> group;", 1)[1].split("if (group.size() >= 2U) device_prefill_group(group)", 1)[0]
        assert "budget, group_capacity)" in admission
        capacity = hip.split("DevicePrefill::group_capacity() const noexcept", 1)[1].split("DevicePrefill::DraftSeed", 1)[0]
        for bound in ("o.chunk_rows", "o.max_rows", "o.max_positions", "m.candidates.n / m.cand_window_bytes", "m.streams[0].n"):
            assert bound in capacity, bound
    elif case == "routes":
        capacity = hip.split("DevicePrefill::group_capacity() const noexcept", 1)[1].split("DevicePrefill::DraftSeed", 1)[0]
        assert "group_decline_reason(o)" in capacity
        assert "cap.rejected_class_mask = kSoloRouteClass" in capacity
        assert "if (!group_options_supported(o))" in hip
        assert "options.fast_dense = options.fast_dense || ds41::fast_profile::dense_on()" in hip
        assert "options.fast_moe = options.fast_moe || ds41::fast_profile::moe_on()" in hip
        assert "ds41::fast_profile::attention_on()" in hip
        assert 'g.set("reason", serve::Json(group_reason))' in serve
    elif case == "storage":
        assert "o.group_rows = gp.storage_rows(a.context)" in serve
        assert "gp.storage_rows(a.context) >= kMinRows" in serve
    elif case == "fairness":
        admission = serve.split("std::vector<Running*> group;", 1)[1].split("if (group.size() >= 2U) device_prefill_group(group)", 1)[0]
        assert "serve::limit_prefill_overtakes(ids, waiting," in admission
        assert "i = pick; i < waiting.size()" in admission
        assert "cands.push_back({i, rows, i," in admission
    elif case == "inspection":
        main = serve.split("int main(int argc, char** argv) try {", 1)[1]
        preflight = main.split("pair_defaults::Effective production_defaults", 1)[0]
        assert "prefill_group_policy()" in preflight
        assert 'std::string_view(argv[i]) == "--print-effective-config"' in preflight
        assert "resolved_group_policy.describe()" in preflight
        policy = serve.split("const serve::PrefillGroupPolicy& prefill_group_policy()", 1)[1].split("// The row-split prefill", 1)[0]
        assert "static const auto policy" in policy and "envcfg::text" in policy
        assert "PrefillGroupPolicy::from_env" in policy
    else:
        raise ValueError(case)


if __name__ == "__main__":
    root = Path(sys.argv[1])
    check((root / "apps/deepseek41_serve_main.cpp").read_text(),
          (root / "src/hip_deepseek41_prefill.hip").read_text(), sys.argv[2])
    print("prefill group " + sys.argv[2] + " adapter contract passed")
