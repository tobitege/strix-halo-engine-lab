"""Shared fixtures for the turbo_bar tests."""
import sys
from pathlib import Path

import numpy as np

from tools.qualification.option_qual import records
from tools.qualification.turbo_bar import manifest as mf
from tools.qualification.turbo_bar import spool
from tools.qualification.turbo_bar.common import safe_name

V = 8
INPUTS = [3, 1, 4, 1, 5, 2, 6, 5, 3, 5, 0, 2]
POSITIONS = [2, 5, 8, 10]
FAKE_TOK = (
    "import sys, zlib\n"
    "t = open(sys.argv[-1], encoding='utf-8').read()\n"
    "print(' '.join(str(zlib.crc32(w.encode()) % 9973) for w in t.split()))\n"
)


def fake_tokenizer(tmp):
    p = Path(tmp) / "faketok.py"
    p.write_text(FAKE_TOK, encoding="utf-8")
    return f'"{Path(sys.executable).as_posix()}" "{p.as_posix()}"'


def make_layout(tmp, ts=("T:a",), parents=4, noise_r=0.5, noise_t=0.1, erep="exact", t_spread_equals_e=False,
                control=False, seed=0):
    tmp = Path(tmp)
    rng = np.random.default_rng(seed)
    cases = []
    arms = ["E", "R", "Erep"] + list(ts) + (["C"] if control else [])
    for p in range(parents):
        for source in ("spread", "continuation"):
            positions = POSITIONS if source == "spread" else list(range(8191, 8447))
            base = rng.normal(size=(len(positions), V)).astype(np.float32) * 2
            z = {"E": base, "R": base + rng.normal(size=base.shape).astype(np.float32) * noise_r,
                 "Erep": base.copy() if erep == "exact" else base + np.float32(1e-3)}
            for t in ts:
                z[t] = base.copy() if (t_spread_equals_e and source == "spread") else \
                    base + rng.normal(size=base.shape).astype(np.float32) * noise_t
            if control:
                z["C"] = base + np.float32(1e-4) * rng.normal(size=base.shape).astype(np.float32)
            files = {}
            for a in arms:
                f = tmp / f"p{p}.{source}.{safe_name(a)}.bin"
                if source == "spread":
                    spool.write(f, INPUTS, POSITIONS, z[a])
                else:
                    records.write_steps(f, [(i, pos, int(np.argmax(z[a][i])), INPUTS[i % len(INPUTS)], z[a][i])
                                            for i, pos in enumerate(positions)])
                files[a] = str(f)
            cases.append({"case": f"p{p}/{source}", "parent": f"p{p}", "category": ["code", "math"][p % 2], "length": 8192,
                          "source": source, "format": "spool" if source == "spread" else "steps", "files": files,
                          "split": "CONFIRM", "expected_positions": list(positions)})
    return {"E": "E", "R": "R", "T": list(ts), "Erep": "Erep", "control": "C" if control else None, "cases": cases,
            "margins": {}, "bootstrap": {"B": 200, "seed": 1}, "manifest": {"version": 1, "core_sha256": None}}


def frozen_manifest(tmp, name="m.json"):
    """A frozen manifest with E and R over small fake weight/binary files."""
    tmp = Path(tmp)
    (tmp / "w_e.gguf").write_bytes(b"exact weights")
    (tmp / "w_r.gguf").write_bytes(b"reference pack")
    (tmp / "bin").write_bytes(b"binary")
    path = tmp / name
    m = mf.new("toy", corpus_sha256="c" * 64, tasks_sha256="d" * 64, tokenizer_receipt_sha256="e" * 64,
               chat_template_sha256="f" * 64, checkpoint_sha256="a" * 64)
    mf.save(path, m)
    for role, w in (("E", "w_e.gguf"), ("R", "w_r.gguf")):
        mf.add_arm(path, mf.make_arm(role, role, f"id-{role}", [mf.file_record(tmp / w)], mf.file_record(tmp / "bin"),
                                     arithmetic="exact" if role == "E" else "controlled"))
    mf.freeze(path)
    return path
