"""Compile the production host replay resolution without HIP or device work."""
import pathlib
import re
import sys

root, out = map(pathlib.Path, sys.argv[1:])
source = (root / "src/hip_deepseek41_prefill.hip").read_text(encoding="utf-8")
setter = source.split("void DevicePrefill::set_precision_for_new_sequence(const PrefillOptions& p)", 1)[1].split("\n}", 1)[0]
copies = "\n".join(re.findall(r"    o\.decoder_replay_\w+ = p\.decoder_replay_\w+;", setter))
resolution = re.search(r"    const std::uint32_t bounded = .*;", source).group(0)
resolution = resolution.replace("m.options", "o").replace("prefill_decoder_replay_rows()", "inherited")
report = "\n".join(re.findall(r"    result\.decoder_replay_\w+ = \w+;", source))
(out / "replay_host.hpp").write_text(
    "inline void apply_replay(PrefillOptions& o, const PrefillOptions& p) {\n" + copies + "\n}\n"
    "inline PrefillResult resolve_replay(const PrefillOptions& o, std::uint32_t inherited, std::uint32_t P0 = 0U) {\n"
    + resolution + "\n    const bool bounded_oracle = false;\n    PrefillResult result;\n"
    + report + "\n    return result;\n}\n", encoding="utf-8")
