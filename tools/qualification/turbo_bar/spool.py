"""Reader/writer of the engine's teacher-logits spool (TMLLOG01 / TMLLOG02), little-endian.

TMLLOG01: magic, u64 vocab, u64 rows, u64 input_count, input_count x u32 input IDs,
          rows x (u32 input, u64 position, vocab x f32), footer "TMLEOF01" + u64 rows.
TMLLOG02: as 01 but rows x u64 selected positions (strictly ascending) follow the input IDs;
          ``rows`` is the selected count.
The row at position p holds the logits that predict ``inputs[p + 1]`` (the true next token).
"""
import os
import struct

import numpy as np

MAGIC1, MAGIC2, FOOTER = b"TMLLOG01", b"TMLLOG02", b"TMLEOF01"
ROW_HEAD = struct.Struct("<IQ")


class SpoolError(ValueError):
    pass


def _exact(f, n, what):
    data = f.read(n)
    if len(data) != n:
        raise SpoolError(f"truncated {what}")
    return data


def header(path):
    """Validate size and footer up front; returns dict(version, vocab, rows, inputs, positions|None, data_offset)."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        magic = _exact(f, 8, "magic")
        if magic not in (MAGIC1, MAGIC2):
            raise SpoolError("bad spool magic")
        version = 1 if magic == MAGIC1 else 2
        vocab, rows, count = struct.unpack("<QQQ", _exact(f, 24, "header"))
        if not vocab or count < 2 or not rows or (version == 1 and rows != count - 1):
            raise SpoolError("inconsistent spool dimensions")
        expected = 32 + 4 * count + (8 * rows if version == 2 else 0) + rows * (ROW_HEAD.size + 4 * vocab) + 16
        if size < expected:
            raise SpoolError("spool footer missing or file truncated (incomplete file)")
        if size > expected:
            raise SpoolError("trailing bytes after spool footer or row count mismatch")
        inputs = np.frombuffer(_exact(f, 4 * count, "inputs"), dtype="<u4")
        if int(inputs.max()) >= vocab:
            raise SpoolError("spool input outside vocabulary")
        positions = None
        if version == 2:
            positions = np.frombuffer(_exact(f, 8 * rows, "selected positions"), dtype="<u8")
            if rows and (int(positions[-1]) >= count - 1
                         or (rows > 1 and not (np.diff(positions.astype(np.int64)) > 0).all())):
                raise SpoolError("selected positions not strictly ascending or out of range")
        off = f.tell()
        f.seek(off + rows * (ROW_HEAD.size + 4 * vocab))
        if _exact(f, 8, "footer") != FOOTER:
            raise SpoolError("spool footer missing")
        if struct.unpack("<Q", _exact(f, 8, "footer count"))[0] != rows:
            raise SpoolError("footer row count mismatch")
    return dict(version=version, vocab=int(vocab), rows=int(rows), inputs=inputs, positions=positions,
                data_offset=off)


def complete(path):
    try:
        for _ in read(path):
            pass
        return True
    except (OSError, SpoolError):
        return False


def read(path):
    """Yield (position, input, label, logits float32 ndarray) per row; rejects any inconsistency."""
    h = header(path)
    vocab, inputs, sel = h["vocab"], h["inputs"], h["positions"]
    prev = -1
    with open(path, "rb") as f:
        f.seek(h["data_offset"])
        for r in range(h["rows"]):
            token, pos = ROW_HEAD.unpack(_exact(f, ROW_HEAD.size, "row header"))
            if pos >= len(inputs) - 1:
                raise SpoolError("row position has no next token")
            if sel is None and pos != r:
                raise SpoolError("v1 row position differs from row index")
            if sel is not None and pos != int(sel[r]):
                raise SpoolError("row position differs from selected position")
            if pos <= prev:
                raise SpoolError("row positions not strictly ascending")
            prev = pos
            if token != int(inputs[pos]):
                raise SpoolError("row input differs from inputs[position]")
            logits = np.frombuffer(_exact(f, 4 * vocab, "row logits"), dtype="<f4")
            if not np.isfinite(logits).all():
                raise SpoolError("non-finite logits")
            yield pos, token, int(inputs[pos + 1]), logits


def write(path, inputs, positions, rows_logits, version=2):
    """Fixture writer (version 1 callers pass ascending positions; no check here)."""
    inputs = [int(x) for x in inputs]
    rows_logits = [np.asarray(z, dtype="<f4") for z in rows_logits]
    vocab = rows_logits[0].size
    with open(path, "wb") as f:
        f.write((MAGIC2 if version == 2 else MAGIC1) + struct.pack("<QQQ", vocab, len(rows_logits), len(inputs)))
        f.write(np.asarray(inputs, dtype="<u4").tobytes())
        if version == 2:
            f.write(np.asarray(positions, dtype="<u8").tobytes())
        for p, z in zip(positions, rows_logits):
            f.write(ROW_HEAD.pack(inputs[p], p) + z.tobytes())
        f.write(FOOTER + struct.pack("<Q", len(rows_logits)))
