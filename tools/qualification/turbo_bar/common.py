"""Small shared helpers: hashing, canonical JSON, id text files."""
import hashlib
import json
import re
from pathlib import Path


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canon_sha(obj):
    return sha256_bytes(canon(obj).encode("utf-8"))


ITEM_FIELDS = ("kind", "split", "group", "primary", "length", "mode", "concurrency", "cut", "meta")


def unique_rows(rows):
    out = {}
    for r in rows:
        if r["id"] in out:
            raise ValueError(f"duplicate item id: {r['id']}")
        out[r["id"]] = r
    return out


def suite_record(items):
    by = unique_rows(items)
    if not by:
        raise ValueError("empty suite")
    return {"sha256": canon_sha([by[k] for k in sorted(by)]),
            "items": {k: dict({f: v[f] for f in ITEM_FIELDS if f in v}, item_sha256=canon_sha(v))
                      for k, v in sorted(by.items())}}


def experiment_binding(m, arm):
    if not m.get("frozen") or arm not in m["arms"]:
        raise ValueError("frozen manifest and current arm required")
    return {"manifest_core_sha256": m["frozen"]["core_sha256"], "manifest_version": m["version"],
            "arm": arm, "arm_sha256": canon_sha(m["arms"][arm])}


def server_identity(m, arm):
    spec = m["arms"][arm]
    route = spec.get("effective_route")
    if not isinstance(route, dict) or not route:
        raise ValueError(f"{arm}: declare effective_route before collecting HTTP evidence")
    return {"identity": spec["identity"], "binary_sha256": spec["binary"]["sha256"],
            "weights_sha256": [w["sha256"] for w in spec["weights"]], "effective_route": route}


def http_binding(m, arm, suite):
    if not m:
        raise ValueError("frozen manifest required")
    rec = m["protocol"].get("suites", {}).get(suite)
    if not rec:
        raise ValueError(f"missing frozen {suite} suite")
    return dict(experiment_binding(m, arm), suite=suite, suite_sha256=rec["sha256"],
                server_identity=server_identity(m, arm))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            out.append(json.loads(line))
    return out


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def parse_ids(text):
    return [int(x) for x in re.split(r"[,\s]+", text.strip()) if x]


def read_ids(path):
    return parse_ids(Path(path).read_text(encoding="utf-8"))


def safe_name(arm):
    return re.sub(r"[^A-Za-z0-9._-]", "_", arm)


def tokenize(cmd, text):
    """Run an external tokenizer: shlex-split CMD plus the path of a temp UTF-8 text file as the LAST argument
    (same convention as strix-tokenize MODEL encode FILE); parse whitespace/comma separated ids from stdout."""
    import os
    import shlex
    import subprocess
    import tempfile
    argv = shlex.split(cmd) if isinstance(cmd, str) else list(cmd)
    fd, path = tempfile.mkstemp(suffix=".txt", prefix="turbo-tok-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(text.encode("utf-8"))
        r = subprocess.run(argv + [path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    finally:
        os.unlink(path)
    return parse_ids(r.stdout.decode("utf-8"))
