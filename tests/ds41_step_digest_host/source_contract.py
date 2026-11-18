"""Guard rank-local ownership/wiring; extract the production checksum for a HIP-free test."""
import re
import sys
from pathlib import Path


def check(source):
    digest = re.search(r"struct StepTokenDigest \{.*?\n\};", source, re.S)
    assert digest, "missing production checksum"
    profile = re.search(r"struct StepProfile \{.*?\n\};", source, re.S)
    assert profile, "missing production step profile"
    rank = source.split("void run_rank(", 1)[1]
    rank = rank[:rank.index('\n}')]
    assert re.search(r"^    StepTokenDigest step_tokens;", rank, re.M), "run_rank must own its checksum"
    assert len(re.findall(r"\bStepTokenDigest\s+\w+\s*[;{=]", source)) == 1, "no shared checksum storage"
    assert "g_step_tokens" not in source, "global checksum mixes emulated ranks"
    assert re.search(r"^    StepProfile step_profile;", rank, re.M), "run_rank must own its profile"
    assert len(re.findall(r"\bStepProfile\s+\w+\s*[;{=]", source)) == 1, "no shared profile storage"
    assert "g_step_profile" not in source, "global profile races between emulated ranks"
    for name, call in (
        ("lead", "lead(a, fwd, comm, dog, make_prefill, stage.get(), step_tokens, step_profile)"),
        ("follow", "follow(a, fwd, dog, make_prefill, stage.get(), step_tokens, step_profile)"),
    ):
        body = source.split(f"void {name}(", 1)[1].split('\n}', 1)[0]
        signature, body = body.split('{', 1)
        assert "StepTokenDigest& step_tokens" in signature, f"{name} must borrow the rank checksum"
        assert "step_tokens.add(tokens);" in body, f"{name} must record its tokens"
        assert "StepProfile& step_profile" in signature, f"{name} must borrow the rank profile"
        assert "step_profile.add(fwd.last_stats());" in body, f"{name} must record its stats"
        assert call in rank, f"{name} must receive the rank checksum"
    assert "static_cast<unsigned long long>(step_tokens.tokens)" in rank
    assert "static_cast<unsigned long long>(step_tokens.hash)" in rank
    assert "step_profile.print(rank);" in rank
    return digest.group(0) + "\n" + profile.group(0)


if __name__ == "__main__":
    try:
        implementation = check(Path(sys.argv[1]).read_text(encoding="utf-8"))
        if len(sys.argv) == 3:
            Path(sys.argv[2]).write_text(
                '#pragma once\n#include "strix/deepseek41_batch_forward.hpp"\n#include <cstdio>\n'
                'using strix::deepseek41::BatchForward;\n' + implementation + "\n", encoding="utf-8")
    except AssertionError as error:
        sys.exit(str(error))
