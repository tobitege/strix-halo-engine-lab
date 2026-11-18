"""Count the full k-tile loop in cuobjdump --dump-sass output (no GPU needed).

Static instruction counts include predicated instructions, NOPs and barriers,
but exclude never-executed @!PT instructions.
They are an issue-mix estimate, not dynamic profiler counts or qualification.
The counted interval is the largest backwards BRA loop containing FFMA.
"""
import argparse
import collections
import json
import re
from pathlib import Path


def inspect(sass):
    records = []
    for section in re.split(r"Function : ", sass)[1:]:
        name = section.splitlines()[0].strip()
        if "core_kernelI" not in name:
            continue
        instructions = []
        for line in section.splitlines():
            match = re.search(r"/\*([0-9a-fA-F]+)\*/\s+(.*?)\s*;", line)
            if match:
                instructions.append((int(match[1], 16), match[2]))
        loops = []
        for address, operation in instructions:
            branch = re.search(r"\bBRA\b.*?(0x[0-9a-fA-F]+)", operation)
            if not branch or int(branch[1], 16) >= address:
                continue
            start = int(branch[1], 16)
            body = [op for pc, op in instructions if start <= pc <= address
                    and not re.match(r"^@!PT\s", op)]
            counts = collections.Counter(re.sub(r"^@\S+\s+", "", op).split()[0] for op in body)
            if counts["FFMA"]:
                loops.append((start, address, body, counts))
        if not loops:
            raise ValueError("no FFMA loop in " + name)
        start, end, body, counts = max(loops, key=lambda loop: len(loop[2]))
        params = [int(v) for v in re.findall(r"Lj(\d+)E", name)]
        rows = params[0]
        columns = 128
        useful = rows * columns * 16 // 256
        record = dict(symbol=name, version=params[1],
                      rows=rows, columns=columns,
                      loop_start=hex(start), loop_end=hex(end), instructions=len(body),
                      ffma=counts["FFMA"], useful_ffma=useful,
                      ffma_fraction=counts["FFMA"] / len(body),
                      useful_ffma_fraction=useful / len(body),
                      opcodes=dict(sorted(counts.items())))
        records.append(record)
    if not records:
        raise ValueError("no core kernels found")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sass", type=Path)
    args = parser.parse_args()
    print(json.dumps(inspect(args.sass.read_text()), indent=2))
