"""Version 1 portable capture records. Hashes are receipts, never equality proofs."""
import hashlib
import json
import pathlib
import struct

MAGIC = b"WFMT\x00\x01\r\n"
SECTIONS = {"source", "sidecars", "activation_codes", "activation_scales",
            "prepared", "partials", "output", "logits", "state"}


def fnv1a64(data):
    value = 14695981039346656037
    for byte in data:
        value = ((value ^ byte) * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def write_record(path, metadata, sections):
    if not any(sections.values()) or set(sections) - SECTIONS:
        raise ValueError("unknown or empty byte sections")
    header = dict(metadata)
    header["sections"] = [{"name": key, "bytes": len(value), "fnv1a64": fnv1a64(value)}
                          for key, value in sorted(sections.items())]
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    with pathlib.Path(path).open("wb") as stream:
        stream.write(MAGIC + struct.pack("<Q", len(encoded)) + encoded)
        for key in sorted(sections):
            stream.write(sections[key])


def read_record(path):
    with pathlib.Path(path).open("rb") as stream:
        if stream.read(8) != MAGIC:
            raise ValueError(f"invalid record version: {path}")
        size_bytes = stream.read(8)
        if len(size_bytes) != 8:
            raise ValueError("truncated header size")
        size, = struct.unpack("<Q", size_bytes)
        if size > 1024 * 1024:
            raise ValueError("oversized header")
        header = json.loads(stream.read(size))
        for key in ("case", "arithmetic", "shape"):
            if key not in header:
                raise ValueError(f"missing metadata: {key}")
        sections = {}
        for section in header.pop("sections"):
            name, count = section["name"], section["bytes"]
            if name not in SECTIONS or name in sections or not isinstance(count, int) or count < 0:
                raise ValueError("invalid section")
            data = stream.read(count)
            if len(data) != count or fnv1a64(data) != section["fnv1a64"]:
                raise ValueError(f"truncated or invalid receipt: {name}")
            sections[name] = data
        if not any(sections.values()) or stream.read(1):
            raise ValueError("empty sections or trailing bytes")
    return header, sections


def read_matrix(path):
    """A matrix freezes record names, semantic metadata and exact section sizes."""
    matrix = json.loads(pathlib.Path(path).read_text())
    if matrix.get("schema") != "wfmt.matrix@1" or not matrix.get("cases"):
        raise ValueError("nonempty wfmt.matrix@1 required")
    names = set()
    for case in matrix["cases"]:
        name = case["file"]
        parts = pathlib.PurePosixPath(name)
        if (not name.endswith(".wfmt") or parts.is_absolute() or ".." in parts.parts
                or "\\" in name or ":" in name or str(parts) != name or name in names):
            raise ValueError("invalid or duplicate matrix record name")
        names.add(name)
        for key in ("case", "arithmetic"):
            if not isinstance(case[key], str) or not case[key]:
                raise ValueError("missing matrix identity")
        if not isinstance(case["shape"], list) or any(type(n) is not int or n < 0 for n in case["shape"]):
            raise ValueError("invalid matrix shape")
        required, optional = case["required_sections"], case.get("optional_sections", {})
        if not required or set(required) & set(optional):
            raise ValueError("required evidence or disjoint optional sections missing")
        for sizes, minimum in ((required, 1), (optional, 0)):
            if set(sizes) - SECTIONS or any(type(n) is not int or n < minimum for n in sizes.values()):
                raise ValueError("invalid matrix section lengths")
    return matrix


def validate_inventory(directory, matrix):
    directory = pathlib.Path(directory)
    files = {p.relative_to(directory).as_posix(): p for p in directory.rglob("*.wfmt")}
    expected = {case["file"]: case for case in matrix["cases"]}
    if files.keys() != expected.keys():
        raise ValueError("capture inventory differs from matrix")
    hashes = {}
    for name, path in sorted(files.items()):
        meta, sections = read_record(path)
        case = expected[name]
        if any(meta.get(key) != case.get(key) for key in ("case", "arithmetic", "shape", "format", "image")):
            raise ValueError(f"capture metadata differs from matrix: {name}")
        required = case["required_sections"]
        sizes = {**required, **case.get("optional_sections", {})}
        if not set(required) <= sections.keys() or sections.keys() - sizes.keys():
            raise ValueError(f"capture sections differ from matrix: {name}")
        if any(len(data) != sizes[key] for key, data in sections.items()):
            raise ValueError(f"capture lengths differ from matrix: {name}")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def compare_record(old, new):
    a, ab = read_record(old)
    b, bb = read_record(new)
    # Build/source identity is provenance; arithmetic, shape and case are contracts.
    for key in ("case", "arithmetic", "shape", "format", "image"):
        if a.get(key) != b.get(key):
            raise ValueError(f"metadata differs: {key}")
    if set(ab) != set(bb):
        raise ValueError("section inventory differs")
    for key in ab:
        if len(ab[key]) != len(bb[key]) or ab[key] != bb[key]:
            raise ValueError(f"bytes differ: {key}")
    return {key: fnv1a64(value) for key, value in ab.items()}
