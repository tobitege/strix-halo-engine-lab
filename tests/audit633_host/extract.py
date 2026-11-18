"""Exercise production admission regions with scalar parser inputs, without HIP.

The full runtime geometry and its host dependencies compile directly. Only the
large application's parser shell is replaced; the rotary admission block and
descriptor validation compile verbatim. An optional local revision proves that
the regression assertions reject the original PR. No network or checkout changes.
"""
from pathlib import Path
import subprocess
import sys

root, out = map(Path, sys.argv[1:3])
revision = sys.argv[3] if len(sys.argv) > 3 else ""


def source(path):
    if revision:
        return subprocess.check_output(
            ["git", "show", f"{revision}:{path}"], cwd=root, text=True, encoding="utf-8"
        )
    return (root / path).read_text(encoding="utf-8")


options = source("apps/hip_decode_options.hpp")
start = options.index("        options.deployment_spec = std::make_shared<const strix::deploy::Spec>")
start = options.index("\n    }", start) + len("\n    }")
end = options.index("    options.mtp_native_bundle =", start)
out.joinpath("admission.inc").write_text(options[start:end], encoding="utf-8")
rotary = source("src/rotary_descriptor.cpp")
factory = rotary[rotary.index("RotaryDescriptor RotaryDescriptor::yarn("):rotary.index("double RotaryDescriptor::attention_factor(")]
validation = rotary[rotary.index("std::uint32_t parse_qwen_rope_yarn("):]
out.joinpath("rotary.inc").write_text("namespace strix {\n" + factory + validation, encoding="utf-8")
out.joinpath("strix").mkdir(exist_ok=True)
for name in ("rotary_descriptor.hpp", "mtp_payload_geometry.hpp"):
    out.joinpath("strix", name).write_text(source("include/strix/" + name), encoding="utf-8")
