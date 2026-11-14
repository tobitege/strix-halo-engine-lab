"""Fresh-process master-control matrix and actual receipt JSON parsing."""
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

env = {k: v for k, v in os.environ.items()
       if not k.startswith(("STRIX_PREFILL_", "STRIX_QPF2K_"))}
cases = 0
for spec, tag, mode in [("", "", 0), ("off", "", 0), ("0", "", 0),
                        ("cap9", "cap9", 3), ("cap09", "cap9", 3),
                        ("rel0.5000n", "rel0.5n", 1), ("topp5e-1", "topp0.5", 2)]:
    for cli, config, master in itertools.product((0, 1), repeat=3):
        run_env = dict(env, STRIX_QPF2K_MOE_ROUTE_PRUNE=spec, STRIX_PREFILL_EXACT_FP16=str(master))
        result = subprocess.run([sys.argv[1], "identity", str(cli), str(config)],
                                env=run_env, capture_output=True, text=True, check=True)
        receipt = json.loads(result.stdout)
        effective = "" if cli or config or master else tag
        name = "fp16-v1" + ("+rp" + effective if effective else "")
        assert receipt["policy"] == (mode if effective else 0), (spec, cli, config, master)
        assert receipt["prefill_identity"] == name, receipt
        identity_hash = 14695981039346656037
        for byte in name.encode("ascii"):
            identity_hash = ((identity_hash ^ byte) * 1099511628211) & ((1 << 64) - 1)
        assert receipt["prefill_identity_hash"] == f"{identity_hash:016x}", receipt
        cases += 1
for spec in ("cap\t9", "cap\n9", "cap\r9", 'cap9"', "cap9\\"):
    result = subprocess.run([sys.argv[1], "identity", "0", "0"],
                            env=dict(env, STRIX_QPF2K_MOE_ROUTE_PRUNE=spec),
                            capture_output=True, text=True)
    assert result.returncode == 1 and not result.stdout, spec
print(f"{cases} master-control/JSON cases and 5 malformed-input rejections passed")

# Host tests cannot submit HIP work, but must catch a stale binding to the raw request.
root = Path(__file__).resolve().parents[1]
window = (root / "src/hip_known_prefix_window.hip").read_text(encoding="utf-8")
assert "routed_config.prefill_moe_route_prune=prefill_route_prune_active();" in window
assert "const bool pruned=prefill_route_prune_active().active();" in window
assert "moe_core::route_prune_active()" not in window
launcher = (root / "src/hip_moe_experts_launcher.hip").read_text(encoding="utf-8")
assert "static moe_core::RouteCensusWriter<> writer(path);" in launcher
assert "writer.append(count, k," in launcher
