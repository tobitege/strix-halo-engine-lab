#!/usr/bin/env python3
"""Lossless ratio and single-thread throughput of zstd level 1 on a real hand-off state, per region.

Usage: state_ratio.py STATE.rtxs [--out DIR] [--level 1]
  STATE.rtxs: a strix-qwen-rtx-state-v1 file ("STRXRTX1", u64 json bytes, JSON header, payload).
  Groups the row components by id prefix (kv, index, mtp, ...) and the fixed components by id, and prints one JSON
  line per group: plain bytes, zstd ratio and MB/s on the plain bytes and on the byte-plane split (plane j = byte j of
  every element, width 2 for row groups, 4 for the FP32 fixed matrices). With --out it writes each group to
  DIR/<group>.bin so strix-xfer-probe codec --data file:DIR/<group>.bin --width W measures bp-v1 on the same bytes.
Needs Python >= 3.14 (compression.zstd). No GPU.
"""
import argparse
import json
import hashlib
from pathlib import Path
import re
import struct
import sys
import time

try:
    from compression import zstd
except ImportError:  # pragma: no cover
    zstd = None


def planes(b, w):
    n = len(b) // w
    return [b[j : n * w : w] for j in range(w)] + ([b[n * w :]] if len(b) % w else [])


def measure(name, b, w, level):
    t = time.perf_counter()
    c = zstd.compress(b, level=level)
    t_plain = time.perf_counter() - t
    t = time.perf_counter()
    cp = sum(len(zstd.compress(p, level=level)) for p in planes(b, w))
    t_planes = time.perf_counter() - t
    mb = len(b) / 1e6
    return {
        "group": name, "width": w, "plain_bytes": len(b),
        "zstd_ratio": round(len(b) / max(1, len(c)), 4), "zstd_MBps_1t": round(mb / t_plain, 1),
        "zstd_planes_ratio": round(len(b) / max(1, cp), 4), "zstd_planes_MBps_1t": round(mb / t_planes, 1),
    }


def validated_groups(data):
    """Validate all untrusted metadata and payload before exposing any output."""
    if len(data) < 16 or data[:8] != b"STRXRTX1":
        raise ValueError("not a STRXRTX1 file")
    (jlen,) = struct.unpack("<Q", data[8:16])
    if jlen > len(data) - 16:
        raise ValueError("truncated JSON header")
    head = json.loads(data[16:16 + jlen])

    def uint(obj, key):
        value = obj[key]
        if type(value) is not int or not 0 <= value < 2**64:
            raise ValueError("invalid unsigned metadata: " + key)
        return value

    ids = set()
    def identifier(c):
        name = c["id"]
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*", name):
            raise ValueError("invalid component id")
        # Windows devices are special even with an extension; reject them on every host.
        if re.fullmatch(r"(?i:CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])", name.split(".")[0]):
            raise ValueError("reserved component id")
        if name.casefold() in ids:
            raise ValueError("duplicate component id")
        ids.add(name.casefold())
        return name

    position = uint(head, "position")
    payload = data[16 + jlen:]
    groups, off, row_bytes = {}, 0, 0
    if not isinstance(head["rows"], list) or not head["rows"] or not isinstance(head["fixed"], list):
        raise ValueError("invalid component lists")
    for c in head["rows"]:
        name = identifier(c).split(".")[0]
        width, shift = uint(c, "bytes_per_row"), uint(c, "shift")
        if not width or shift > 1:
            raise ValueError("invalid row component")
        row_bytes += width
        size = max(0, position - shift) * width
        if size > len(payload) - off:
            raise ValueError("truncated rows")
        groups.setdefault(name, []).append(payload[off:off + size])
        off += size
    if row_bytes != uint(head, "row_bytes"):
        raise ValueError("row bytes do not add up")
    fixed_bytes, fixed0, at = uint(head, "fixed_bytes"), off, 0
    if fixed_bytes != len(payload) - fixed0:
        raise ValueError("invalid payload length")
    for c in head["fixed"]:
        name = "fixed." + identifier(c)
        start, size = uint(c, "offset"), uint(c, "bytes")
        if not size or start != at or start % 256 or size > fixed_bytes - start:
            raise ValueError("invalid fixed component")
        groups.setdefault(name, []).append(payload[fixed0 + start:fixed0 + start + size])
        at = start + ((size + 255) // 256) * 256
    if at != fixed_bytes or uint(head, "payload_bytes") != len(payload):
        raise ValueError("fixed/payload bytes do not add up")
    if head["sha256"] != hashlib.sha256(payload).hexdigest():
        raise ValueError("payload checksum mismatch")
    return head, {name: b"".join(parts) for name, parts in groups.items()}


def write_groups(out, groups):
    root = Path(out).resolve()
    destinations = []
    seen = set()
    for name, data in groups.items():
        path = root / (name + ".bin")
        # Validate every destination before opening any file. Refuse existing files
        # as well: this avoids following symlinks or overwriting hard-linked victims.
        if path.resolve().parent != root or path.exists() or path.is_symlink() or name.casefold() in seen:
            raise ValueError("unsafe or existing output destination")
        seen.add(name.casefold())
        destinations.append((path, data))
    root.mkdir(parents=True, exist_ok=True)
    for path, data in destinations:
        with path.open("xb") as f:
            f.write(data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("state")
    ap.add_argument("--out")
    ap.add_argument("--level", type=int, default=1)
    a = ap.parse_args()
    head, groups = validated_groups(Path(a.state).read_bytes())
    groups = {name: data for name, data in groups.items() if len(data) >= 4096}
    if zstd is None:
        sys.exit("state_ratio.py needs Python 3.14 (compression.zstd)")
    if a.out:
        write_groups(a.out, groups)
    print(json.dumps({k: head[k] for k in ("position", "row_bytes", "fixed_bytes")}))
    for name, data in groups.items():
        print(json.dumps(measure(name, data, 4 if name.startswith("fixed.") else 2, a.level)))


if __name__ == "__main__":
    main()
