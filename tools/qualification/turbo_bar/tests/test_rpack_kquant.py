"""Unit tests for rpack_kquant (synthetic data only)."""
import json
import math
import os
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import rpack_kquant as rp  # noqa: E402

BITS = 3
L = 1 << BITS


def random_blocks(qt, nb, rng):
    """Random valid blocks of a type, gaussian-shaped codes. Returns raw uint8 (nb, bsize)."""
    mid = (qt.hi + qt.lo) / 2
    sd = (qt.hi - qt.lo) / 6
    codes = np.clip(np.rint(rng.normal(mid, sd, (nb, qt.nsub, qt.sublen))), qt.lo, qt.hi).astype(np.int32)
    p = {"d": rng.uniform(0.002, 0.02, nb).astype(np.float16),
         "dmin": rng.uniform(0.001, 0.01, nb).astype(np.float16),
         "codes": codes, "sc": np.ones((nb, qt.nsub), np.int32),
         "mn": np.zeros((nb, qt.nsub), np.int32)}
    if qt.name in ("Q4_K", "Q5_K"):
        p["sc"] = rng.integers(20, 64, (nb, 8)).astype(np.int32)
        p["mn"] = rng.integers(0, 64, (nb, 8)).astype(np.int32)
    elif qt.name == "Q6_K":
        p["sc"] = rng.integers(-100, 101, (nb, 16)).astype(np.int32)
        p["dmin"] = np.zeros(nb, np.float16)
    elif qt.name != "Q5_1":
        p["dmin"] = np.zeros(nb, np.float16)
    else:
        p["dmin"] = rng.uniform(-0.1, 0.0, nb).astype(np.float16)
    return qt.pack(p)


def on_grid(codes_sub, bits):
    """True if the used codes lie on c0 + step*k with k < 2^bits."""
    u = np.unique(codes_sub)
    if len(u) == 1:
        return True
    g = int(np.gcd.reduce(np.diff(u)))
    return (u[-1] - u[0]) // g <= (1 << bits) - 1


def naive_se(qt, raw, bits):
    """Naive baseline: keep scales, snap codes to the nearest multiple of the coarse step."""
    blk = raw.reshape(-1, qt.bsize)
    p = qt.unpack(blk)
    x = qt.decode(p)
    ns = max(1, (qt.hi - qt.lo) // ((1 << bits) - 1))  # Q4_K: 2 (nearest even code)
    c = p["codes"]
    q = qt.lo + np.rint((c - qt.lo) / ns) * ns
    p["codes"] = np.clip(q, qt.lo, qt.hi).astype(np.int32)
    return float(((qt.decode(p) - x).astype(np.float64) ** 2).sum())


class TestRequant(unittest.TestCase):
    def test_each_type(self):
        rng = np.random.default_rng(1234)
        for qt in rp.QTYPES.values():
            with self.subTest(type=qt.name):
                nb = 64
                raw = random_blocks(qt, nb, rng)
                blk = raw.reshape(nb, qt.bsize)
                # pack/unpack round trip is exact
                self.assertTrue(np.array_equal(qt.pack(qt.unpack(blk)), blk))
                new, se, nrm = rp.requantize_bytes(qt, raw.tobytes(), BITS)
                # (a) byte size unchanged
                self.assertEqual(len(new), raw.nbytes)
                nblk = np.frombuffer(new, np.uint8).reshape(nb, qt.bsize)
                # (b) on a 2^bits-level grid per sub-block
                codes = qt.unpack(nblk)["codes"]
                for i in range(nb):
                    for s in range(qt.nsub):
                        self.assertTrue(on_grid(codes[i, s], BITS), (i, s))
                x = qt.decode(qt.unpack(blk))
                y = qt.decode(qt.unpack(nblk))
                self.assertLessEqual(len(np.unique(np.round(y[0, 0], 7))), L)
                # (c) rel mse finite, below naive rounding
                rel = float(((y - x).astype(np.float64) ** 2).sum() / (x.astype(np.float64) ** 2).sum())
                self.assertTrue(math.isfinite(rel))
                self.assertAlmostEqual(rel, se / nrm, delta=1e-4 * max(rel, 1e-9) + 1e-9)
                naive = naive_se(qt, raw, BITS) / float((x.astype(np.float64) ** 2).sum())
                self.assertLess(rel, naive)

    def test_bits2(self):
        rng = np.random.default_rng(5)
        qt = rp.QTYPES[12]
        raw = random_blocks(qt, 8, rng)
        new, _, _ = rp.requantize_bytes(qt, raw.tobytes(), 2)
        codes = qt.unpack(np.frombuffer(new, np.uint8).reshape(8, 144))["codes"]
        for i in range(8):
            for s in range(8):
                self.assertTrue(on_grid(codes[i, s], 2))


def _s(b):
    return struct.pack("<Q", len(b)) + b


def write_gguf(path, tensors, alignment=32):
    """tensors: list of (name, dims, type_id, bytes)."""
    h = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 3)
    h += _s(b"general.architecture") + struct.pack("<I", 8) + _s(b"test")
    h += _s(b"general.alignment") + struct.pack("<II", 4, alignment)
    h += _s(b"tokenizer.tokens") + struct.pack("<IIQ", 9, 8, 2) + _s(b"a") + _s(b"bc")
    off = 0
    for name, dims, t, data in tensors:
        h += _s(name.encode()) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        h += struct.pack("<IQ", t, off)
        off += (len(data) + alignment - 1) // alignment * alignment
    h += b"\0" * (-len(h) % alignment)
    body = b""
    for _, _, _, data in tensors:
        body += data + b"\0" * (-len(data) % alignment)
    with open(path, "wb") as f:
        f.write(h + body)
    return len(h)


class TestGguf(unittest.TestCase):
    def test_roundtrip_and_skip(self):
        rng = np.random.default_rng(7)
        q4 = rp.QTYPES[12]
        exps = random_blocks(q4, 6, rng).tobytes()          # matches pattern
        other = random_blocks(q4, 3, rng).tobytes()         # does not match
        q3k = rng.integers(0, 256, 110 * 2, dtype=np.uint8).tobytes()  # matches, unsupported
        with tempfile.TemporaryDirectory() as td:
            src, out = os.path.join(td, "s.gguf"), os.path.join(td, "o.gguf")
            hlen = write_gguf(src, [
                ("blk.0.ffn_gate_exps.weight", [256, 6], 12, exps),
                ("blk.0.attn_q.weight", [256, 3], 12, other),
                ("blk.1.ffn_up_exps.weight", [256, 2], 11, q3k)])
            info = rp.read_gguf(src)
            self.assertEqual(info["data_start"], hlen)
            self.assertEqual(len(info["tensors"]), 3)
            rep = rp.build(src, out, bits=3, identity="T", jobs=1, chunk_blocks=4, quiet=True)
            a, b = open(src, "rb").read(), open(out, "rb").read()
            self.assertEqual(len(a), len(b))
            self.assertEqual(a[:hlen], b[:hlen])                         # header identical
            o = {t["name"]: t for t in rp.read_gguf(out)["tensors"]}
            s = lambda t: slice(t["abs_offset"], t["abs_offset"] + t["nbytes"])  # noqa: E731
            t0, t1, t2 = (o[n] for n in ("blk.0.ffn_gate_exps.weight", "blk.0.attn_q.weight",
                                          "blk.1.ffn_up_exps.weight"))
            self.assertEqual(a[s(t1)], b[s(t1)])                         # non-matching untouched
            self.assertEqual(a[s(t2)], b[s(t2)])                         # unsupported untouched
            self.assertNotEqual(a[s(t0)], b[s(t0)])                      # matching changed
            y = rp.dequantize(q4, b[s(t0)])
            self.assertTrue(np.isfinite(y).all())
            self.assertEqual(len(y), 256 * 6)
            # (e) skip reported
            self.assertEqual([k["name"] for k in rep["skipped"]], ["blk.1.ffn_up_exps.weight"])
            self.assertEqual(rep["skipped"][0]["type"], "Q3_K")
            self.assertEqual(len(rep["tensors"]), 1)
            self.assertTrue(math.isfinite(rep["tensors"][0]["rel_mse"]))
            js = json.load(open(out + ".json", encoding="utf-8"))
            self.assertEqual(js["identity"], "T")
            self.assertEqual(len(js["tool_sha256"]), 64)
            self.assertIsNone(js["output_sha256"])
            self.assertFalse(os.path.exists(out + ".partial"))
            with self.assertRaises(FileExistsError):
                rp.build(src, out, identity="T", quiet=True)


if __name__ == "__main__":
    unittest.main()
