"""Compile production host regions without HIP or the full engine build.

The device and model are test doubles. Gate decisions, options, flag replacement,
and Engram hashing are copied verbatim at configure time, never reimplemented.
"""
from pathlib import Path
import sys

root, out = map(Path, sys.argv[1:])
app = (root / "apps/deepseek41_prefill_main.cpp").read_text(encoding="utf-8")
hip = (root / "src/hip_deepseek41_prefill.hip").read_text(encoding="utf-8")
engram = (root / "src/ds41_engram.cpp").read_text(encoding="utf-8")


def between(text, start, end):
    begin = text.index(start)
    return text[begin:text.index(end, begin)]


parts = [between(app, "struct Args {", "// --ttft-timing: wall time"),
         between(app, "strix::deepseek41::PrefillOptions options(", "// --ttft-timing: the optional"),
         between(app, "int embed_identity_gate(", "int run_world1(")]
out.joinpath("gate.inc").write_text("\n".join(parts), encoding="utf-8")
flags = between(hip, "    if (!embeds.empty() || m.image_rows_set) {", "        std::size_t embed_at = 0;")
# Exclude the device-only probe; include the entire replacement and clean/dirty transition.
flags = flags.split("        if (m.options.verify_embed_extras")[0]
out.joinpath("flags.inc").write_text(flags + "        (void)embed_width;\n    }\n", encoding="utf-8")
out.joinpath("engram.inc").write_text(
    between(engram, "EngramHasher::EngramHasher(", "EngramRowReader::EngramRowReader("), encoding="utf-8")
# Ensure validation is wired before file reads and world/timing dispatch, not merely defined.
main = app[app.index("int main("):]
assert main.index("validate_embed_identity_args(a);") < main.index("read_prompt(a.prompt_path, a.rows)")
