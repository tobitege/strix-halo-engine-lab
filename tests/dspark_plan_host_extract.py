"""Compile production plan blocks without a HIP runtime; never duplicate their logic.

The harness stubs GPU/controller boundaries, but executes admission, park/resume,
and observation collection/dispatch copied verbatim from the selected source.
--source also permits a pre-fix source for regression-sensitivity checks.
"""
import argparse
from pathlib import Path


def generate(source, template, output):
    text = source.read_text(encoding="utf-8")

    def region(start, end, include_end=False):
        begin = text.index(start)
        finish = text.index(end, begin) + (len(end) if include_end else 0)
        return text[begin:finish]

    blocks = {
        "PARKED": region("struct BatchSpeculator::ParkedPlan {", "\n};", True),
        "PARK": region("std::shared_ptr<BatchSpeculator::ParkedPlan> BatchSpeculator::park()", "\n}", True),
        "RESUME": region("void BatchSpeculator::resume(", "\n}", True),
        "ADMISSION": region("    std::vector<bool> drafting(nd, false);", "    std::vector<ds::Draft> drafts(nd);"),
        "COLLECT": region("            if (i < arbiter_used_.size()", "\n"),
        "OBSERVE": region("    for (const auto& a : arbitrated)", "    stats_.finish_seconds += fs;"),
    }
    result = template.read_text(encoding="utf-8")
    for name, body in blocks.items():
        result = result.replace("@" + name + "@", body)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate(args.source, args.template, args.output)
