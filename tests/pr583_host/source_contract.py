"""Extract production host decisions; guard their wiring into the HIP orchestration."""
import re
import sys
from pathlib import Path


def extract(comm, forward):
    methods = []
    for name in ("agree_exchange", "staged_ready_chunk_rows"):
        match = re.search(r"    \[\[nodiscard\]\] [^\n]+ " + name + r"\(.*?\n    }", comm, re.S)
        assert match, f"missing shared {name} decision"
        methods.append(match.group(0))
    constructor = forward.split("BatchForward::BatchForward(", 1)[1].split("\n}", 1)[0]
    async_policy = re.search(r"    const bool engram_async = .*?m.engram_async = [^;]+;", constructor, re.S)
    assert async_policy, "async admission must use the governed controls"
    assert async_policy.group(0).count("strix::envcfg::flag(") == 2
    startup = re.search(r"    if \(m.distributed\(\)\) \{\n        const bool requested = .*?\n    }", constructor, re.S)
    assert startup, "both ranks must agree before stepping, including local OFF"
    policy = startup.group(0)
    assert "requested && m.engram_async && m.engram_reader != nullptr" in policy
    assert "W * m.model.engram.rows_per_token() * 264U, m.half_other.bytes()" in policy
    assert "options.communicator->agree_exchange(" in policy
    assert forward.count("m.engram_from_lead =") == 1, "no local override of pair agreement"
    assert "const bool engram_peer_rows = engram_async && m.engram_from_lead;" in forward
    preflight = forward.index("staged_ready_chunk_rows(R, std::size_t{half} * 4U)")
    assert preflight < forward.index("product(0U, chunk);") < forward.index("ready_detail::gather_publish(")
    assert "engram ready view refused mid exchange" not in forward
    assert "if (r0 + rn < R) product(r0 + rn, R - r0 - rn);" in forward
    return "\n".join(methods), async_policy.group(0) + "\n" + policy


if __name__ == "__main__":
    try:
        methods, startup = extract(Path(sys.argv[1]).read_text(encoding="utf-8"),
                                   Path(sys.argv[2]).read_text(encoding="utf-8"))
        if len(sys.argv) == 4:
            out = Path(sys.argv[3])
            (out / "admission.inc").write_text(methods + "\n", encoding="utf-8")
            (out / "startup.inc").write_text(startup + "\n", encoding="utf-8")
    except (AssertionError, ValueError) as error:
        sys.exit(str(error))
