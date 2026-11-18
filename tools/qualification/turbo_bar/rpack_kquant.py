#!/usr/bin/env python3
"""Reference pack ("R") builder for the TURBO quality bar.

Purpose: take our own GGUF model and produce a smaller-precision twin of it that
the engine runs with its UNCHANGED exact kernels ("controlled arithmetic"). The
source file is copied byte for byte, then only the data of selected tensors
(default: the MoE expert FFN weights) is rewritten in place. Tensor type, block
geometry and byte size are kept, so each rewritten block is still a valid block
of its original ggml type, but its weights lie on a coarser grid of 2^bits
levels (codes restricted to an arithmetic progression c0 + step*k, with the
sub-block scales/mins and super-block d/dmin re-chosen to minimise squared
error). This is a reference for measuring quality only; never a shipped format.

Supported ggml types: Q4_K, Q5_K, Q6_K, Q8_0, Q5_1. Other matching types are skipped
and reported (never touched). The pattern is matched with re.fullmatch.

Usage:
  python3 rpack_kquant.py census SRC.gguf [--pattern RE] [--names]
  python3 rpack_kquant.py build SRC.gguf OUT.gguf --bits 3 --identity ID
        [--pattern RE] [--jobs N] [--limit-tensors N]
        [--source-sha256 HEX] [--hash-output]
build refuses an existing OUT, writes OUT.partial, renames it when done and
writes OUT.json (identity, source, bits, per-tensor rel_mse, totals, tool sha).
Known limitation: a super-block whose 8 sub-blocks are all exactly constant and
nonzero may not be representable exactly (K types); real weights never are.
"""
import argparse
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import struct
import sys
import time

import numpy as np

DEFAULT_PATTERN = r"blk\.\d+\.ffn_(gate|up|down)_exps\.weight"

# ggml type id -> (name, elements per block, bytes per block)
TYPE_INFO = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22), 7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2),
    26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
}


# ----------------------------------------------------------------------------
# GGUF header reader
# ----------------------------------------------------------------------------
_SCALAR = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
           5: ("<i", 4), 6: ("<f", 4), 7: ("<B", 1), 10: ("<Q", 8), 11: ("<q", 8),
           12: ("<d", 8)}


def _u32(f):
    return struct.unpack("<I", f.read(4))[0]


def _u64(f):
    return struct.unpack("<Q", f.read(8))[0]


def _read_value(f, t):
    if t in _SCALAR:
        fmt, n = _SCALAR[t]
        return struct.unpack(fmt, f.read(n))[0]
    if t == 8:
        return f.read(_u64(f)).decode("utf-8", "replace")
    if t == 9:
        et = _u32(f)
        n = _u64(f)
        if et in _SCALAR:
            f.seek(n * _SCALAR[et][1], 1)
        elif et == 8:
            for _ in range(n):
                f.seek(_u64(f), 1)
        else:
            for _ in range(n):
                _read_value(f, et)
        return None
    raise ValueError("bad GGUF value type %d" % t)


def read_gguf(path):
    """Minimal GGUF v3 reader. Returns dict with alignment, data_start, tensors."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version = _u32(f)
        if version != 3:
            raise ValueError("unsupported GGUF version %d" % version)
        n_tensors = _u64(f)
        n_kv = _u64(f)
        alignment = 32
        for _ in range(n_kv):
            key = f.read(_u64(f)).decode("utf-8", "replace")
            t = _u32(f)
            v = _read_value(f, t)
            if key == "general.alignment" and isinstance(v, int):
                alignment = v
        tensors = []
        for _ in range(n_tensors):
            name = f.read(_u64(f)).decode("utf-8", "replace")
            nd = _u32(f)
            dims = [_u64(f) for _ in range(nd)]
            ttype = _u32(f)
            off = _u64(f)
            tensors.append({"name": name, "dims": dims, "type": ttype, "offset": off})
        pos = f.tell()
    data_start = (pos + alignment - 1) // alignment * alignment
    for t in tensors:
        t["n_elements"] = int(np.prod(t["dims"], dtype=object)) if t["dims"] else 1
        info = TYPE_INFO.get(t["type"])
        t["type_name"] = info[0] if info else "type%d" % t["type"]
        if info and t["n_elements"] % info[1] == 0:
            t["nbytes"] = t["n_elements"] // info[1] * info[2]
        else:
            t["nbytes"] = None
        t["abs_offset"] = data_start + t["offset"]
    return {"version": version, "alignment": alignment, "data_start": data_start,
            "header_end": pos, "tensors": tensors}


# ----------------------------------------------------------------------------
# Block types: unpack / decode / requantize / pack
# Parts: d, dmin (float16 (nb,)), sc, mn (int32 (nb,nsub)), codes (int32 (nb,nsub,len))
# ----------------------------------------------------------------------------
def _f16(blocks, a):
    return np.ascontiguousarray(blocks[:, a:a + 2]).view("<f2")[:, 0].copy()


def _f16b(v):
    return np.asarray(v).astype("<f2").view(np.uint8).reshape(-1, 2)


def _fit_ab(x, L, iters=4):
    """Per sub-block fit x ~ a*k + b, k in 0..L-1 (float64, a >= 0)."""
    x = x.astype(np.float64)
    n = x.shape[-1]
    mn, mx = x.min(-1), x.max(-1)
    a, b = (mx - mn) / (L - 1), mn
    for _ in range(iters):
        sa = np.where(a > 0, a, 1.0)
        k = np.clip(np.rint((x - b[..., None]) / sa[..., None]), 0, L - 1)
        sk, skk = k.sum(-1), (k * k).sum(-1)
        sx, skx = x.sum(-1), (k * x).sum(-1)
        den = n * skk - sk * sk
        ok = den > 1e-9
        a2 = (n * skx - sk * sx) / np.where(ok, den, 1.0)
        b2 = (sx - a2 * sk) / n
        use = ok & (a2 > 0)
        a, b = np.where(use, a2, a), np.where(use, b2, b)
    return a, b


def _assign(x, a_eff, b_eff, L):
    sa = np.where(a_eff > 0, a_eff, 1.0)
    return np.clip(np.rint((x - b_eff[..., None]) / sa[..., None]), 0, L - 1).astype(np.int64)


def _unpack_scales(s):
    s = s.astype(np.int32)
    sc = np.empty((s.shape[0], 8), np.int32)
    mn = np.empty_like(sc)
    sc[:, :4], mn[:, :4] = s[:, 0:4] & 63, s[:, 4:8] & 63
    sc[:, 4:] = (s[:, 8:12] & 15) | ((s[:, 0:4] >> 6) << 4)
    mn[:, 4:] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    return sc, mn


def _pack_scales(sc, mn):
    out = np.empty((sc.shape[0], 12), np.uint8)
    out[:, 0:4] = (sc[:, :4] & 63) | ((sc[:, 4:] >> 4) << 6)
    out[:, 4:8] = (mn[:, :4] & 63) | ((mn[:, 4:] >> 4) << 6)
    out[:, 8:12] = (sc[:, 4:] & 15) | ((mn[:, 4:] & 15) << 4)
    return out


class QType:
    name = ""
    type_id = 0
    bsize = 0
    nsub = 0
    sublen = 0
    lo = 0
    hi = 0

    @property
    def elems(self):
        return self.nsub * self.sublen

    def steps(self, L):
        smax = max(1, (self.hi - self.lo) // (L - 1))
        if smax <= 12:
            return list(range(1, smax + 1))
        return sorted(set(int(v) for v in np.linspace(max(1, smax // 3), smax, 12)))

    def decode(self, p):
        raise NotImplementedError

    def requant(self, x, bits):
        """x: float32 (nb,nsub,len). Returns (parts, per-block squared error)."""
        L = 1 << bits
        a, b = _fit_ab(x, L)
        best, best_se = None, None
        for step in self.steps(L):
            if (self.hi - self.lo) < step * (L - 1):
                continue
            p = self._cand(x, a, b, step, L)
            se = ((self.decode(p) - x).astype(np.float64) ** 2).sum((1, 2))
            if best is None:
                best, best_se = p, se
            else:
                imp = se < best_se
                for k in best:
                    best[k][imp] = p[k][imp]
                best_se = np.where(imp, se, best_se)
        return best, best_se


class _KType(QType):
    nsub, sublen, lo = 8, 32, 0
    has_qh = False

    def decode(self, p):
        dsc = p["d"].astype(np.float32)[:, None] * p["sc"].astype(np.float32)
        mm = p["dmin"].astype(np.float32)[:, None] * p["mn"].astype(np.float32)
        return dsc[..., None] * p["codes"].astype(np.float32) - mm[..., None]

    def _cand(self, x, a, b, step, L):
        c0max = self.hi - step * (L - 1)
        d16 = (a.max(1) / (step * 63)).astype(np.float16)
        d16 = np.where(d16 > 0, d16, np.float16(1))
        dd = d16.astype(np.float64)
        sc = np.clip(np.rint(a / (dd[:, None] * step)), 1, 63)
        u = dd[:, None] * sc
        c0 = np.clip(np.ceil(b / u), 0, c0max)
        r = np.maximum(u * c0 - b, 0)
        dm16 = (r.max(1) / 63).astype(np.float16)
        dm = dm16.astype(np.float64)
        mn = np.clip(np.rint(r / np.where(dm > 0, dm, 1.0)[:, None]), 0, 63)
        mn[dm == 0] = 0
        k = _assign(x, u * step, u * c0 - dm[:, None] * mn, L)
        codes = c0[..., None].astype(np.int64) + step * k
        return {"d": d16, "dmin": dm16, "sc": sc.astype(np.int32), "mn": mn.astype(np.int32),
                "codes": codes.astype(np.int32)}


class Q4K(_KType):
    name, type_id, bsize, hi = "Q4_K", 12, 144, 15

    def unpack(self, blk):
        sc, mn = _unpack_scales(blk[:, 4:16])
        qs = blk[:, 16:144].reshape(-1, 4, 32).astype(np.int32)
        codes = np.empty((blk.shape[0], 8, 32), np.int32)
        codes[:, 0::2], codes[:, 1::2] = qs & 15, qs >> 4
        return {"d": _f16(blk, 0), "dmin": _f16(blk, 2), "sc": sc, "mn": mn, "codes": codes}

    def pack(self, p):
        nb = p["d"].shape[0]
        out = np.empty((nb, 144), np.uint8)
        out[:, 0:2], out[:, 2:4] = _f16b(p["d"]), _f16b(p["dmin"])
        out[:, 4:16] = _pack_scales(p["sc"], p["mn"])
        c = p["codes"]
        out[:, 16:] = ((c[:, 0::2] & 15) | ((c[:, 1::2] & 15) << 4)).reshape(nb, 128)
        return out


class Q5K(_KType):
    name, type_id, bsize, hi = "Q5_K", 13, 176, 31

    def unpack(self, blk):
        sc, mn = _unpack_scales(blk[:, 4:16])
        qh = blk[:, 16:48].astype(np.int32)[:, None, :]
        qs = blk[:, 48:176].reshape(-1, 4, 32).astype(np.int32)
        sh = (2 * np.arange(4))[None, :, None]
        codes = np.empty((blk.shape[0], 8, 32), np.int32)
        codes[:, 0::2] = (qs & 15) | (((qh >> sh) & 1) << 4)
        codes[:, 1::2] = (qs >> 4) | (((qh >> (sh + 1)) & 1) << 4)
        return {"d": _f16(blk, 0), "dmin": _f16(blk, 2), "sc": sc, "mn": mn, "codes": codes}

    def pack(self, p):
        nb = p["d"].shape[0]
        out = np.empty((nb, 176), np.uint8)
        out[:, 0:2], out[:, 2:4] = _f16b(p["d"]), _f16b(p["dmin"])
        out[:, 4:16] = _pack_scales(p["sc"], p["mn"])
        lo, hi = p["codes"][:, 0::2], p["codes"][:, 1::2]
        sh = (2 * np.arange(4))[None, :, None]
        qh = (((lo >> 4) & 1) << sh).sum(1) | (((hi >> 4) & 1) << (sh + 1)).sum(1)
        out[:, 16:48] = qh
        out[:, 48:] = ((lo & 15) | ((hi & 15) << 4)).reshape(nb, 128)
        return out


class Q6K(QType):
    name, type_id, bsize, nsub, sublen, lo, hi = "Q6_K", 14, 210, 16, 16, 0, 63

    def unpack(self, blk):
        nb = blk.shape[0]
        ql = blk[:, 0:128].reshape(nb, 2, 64).astype(np.int32)
        qh = blk[:, 128:192].reshape(nb, 2, 32).astype(np.int32)
        c = np.empty((nb, 2, 4, 32), np.int32)
        c[:, :, 0] = (ql[..., :32] & 15) | (((qh >> 0) & 3) << 4)
        c[:, :, 1] = (ql[..., 32:] & 15) | (((qh >> 2) & 3) << 4)
        c[:, :, 2] = (ql[..., :32] >> 4) | (((qh >> 4) & 3) << 4)
        c[:, :, 3] = (ql[..., 32:] >> 4) | (((qh >> 6) & 3) << 4)
        sc = np.ascontiguousarray(blk[:, 192:208]).view(np.int8).astype(np.int32)
        return {"d": _f16(blk, 208), "dmin": np.zeros(nb, np.float16), "sc": sc,
                "mn": np.zeros((nb, 16), np.int32), "codes": c.reshape(nb, 16, 16)}

    def pack(self, p):
        nb = p["d"].shape[0]
        c = p["codes"].reshape(nb, 2, 4, 32)
        out = np.empty((nb, 210), np.uint8)
        ql = np.empty((nb, 2, 64), np.int32)
        ql[..., :32] = (c[:, :, 0] & 15) | ((c[:, :, 2] & 15) << 4)
        ql[..., 32:] = (c[:, :, 1] & 15) | ((c[:, :, 3] & 15) << 4)
        qh = (c[:, :, 0] >> 4) | ((c[:, :, 1] >> 4) << 2) | ((c[:, :, 2] >> 4) << 4) | ((c[:, :, 3] >> 4) << 6)
        out[:, 0:128] = ql.reshape(nb, 128)
        out[:, 128:192] = qh.reshape(nb, 64)
        out[:, 192:208] = p["sc"].astype(np.int8).view(np.uint8)
        out[:, 208:210] = _f16b(p["d"])
        return out

    def decode(self, p):
        dsc = p["d"].astype(np.float32)[:, None] * p["sc"].astype(np.float32)
        return dsc[..., None] * (p["codes"] - 32).astype(np.float32)

    def _cand(self, x, a, b, step, L):
        c0max = self.hi - step * (L - 1)
        d16 = (a.max(1) / (step * 127)).astype(np.float16)
        d16 = np.where(d16 > 0, d16, np.float16(1))
        dd = d16.astype(np.float64)
        sc = np.clip(np.rint(a / (dd[:, None] * step)), 1, 127)
        u = dd[:, None] * sc
        c0 = np.clip(np.rint(32 + b / u), 0, c0max)
        k = _assign(x, u * step, u * (c0 - 32), L)
        codes = c0[..., None].astype(np.int64) + step * k
        nb = x.shape[0]
        return {"d": d16, "dmin": np.zeros(nb, np.float16), "sc": sc.astype(np.int32),
                "mn": np.zeros((nb, 16), np.int32), "codes": codes.astype(np.int32)}


class Q80(QType):
    name, type_id, bsize, nsub, sublen, lo, hi = "Q8_0", 8, 34, 1, 32, -127, 127

    def unpack(self, blk):
        nb = blk.shape[0]
        q = np.ascontiguousarray(blk[:, 2:34]).view(np.int8).astype(np.int32)
        return {"d": _f16(blk, 0), "dmin": np.zeros(nb, np.float16),
                "sc": np.ones((nb, 1), np.int32), "mn": np.zeros((nb, 1), np.int32),
                "codes": q.reshape(nb, 1, 32)}

    def pack(self, p):
        nb = p["d"].shape[0]
        out = np.empty((nb, 34), np.uint8)
        out[:, 0:2] = _f16b(p["d"])
        out[:, 2:] = p["codes"].reshape(nb, 32).astype(np.int8).view(np.uint8)
        return out

    def decode(self, p):
        return p["d"].astype(np.float32)[:, None, None] * p["codes"].astype(np.float32)

    def _cand(self, x, a, b, step, L):
        a, b = a[:, 0], b[:, 0]
        d = np.where(a > 0, a / step, np.abs(b) / 127)
        d16 = d.astype(np.float16)
        d16 = np.where(d16 > 0, d16, np.float16(1))
        dd = d16.astype(np.float64)
        c0 = np.clip(np.rint(b / dd), self.lo, self.hi - step * (L - 1))
        k = _assign(x, (dd * step)[:, None], (dd * c0)[:, None], L)
        codes = c0[:, None, None].astype(np.int64) + step * k
        nb = x.shape[0]
        return {"d": d16, "dmin": np.zeros(nb, np.float16), "sc": np.ones((nb, 1), np.int32),
                "mn": np.zeros((nb, 1), np.int32), "codes": codes.astype(np.int32)}


class Q51(QType):
    name, type_id, bsize, nsub, sublen, lo, hi = "Q5_1", 7, 24, 1, 32, 0, 31

    def unpack(self, blk):
        nb = blk.shape[0]
        qh = np.ascontiguousarray(blk[:, 4:8]).view("<u4")[:, 0].astype(np.int64)
        qs = blk[:, 8:24].astype(np.int32)
        hb = ((qh[:, None] >> np.arange(32)[None, :]) & 1).astype(np.int32)
        c = np.empty((nb, 32), np.int32)
        c[:, :16], c[:, 16:] = qs & 15, qs >> 4
        c |= hb << 4
        return {"d": _f16(blk, 0), "dmin": _f16(blk, 2), "sc": np.ones((nb, 1), np.int32),
                "mn": np.zeros((nb, 1), np.int32), "codes": c.reshape(nb, 1, 32)}

    def pack(self, p):
        nb = p["d"].shape[0]
        c = p["codes"].reshape(nb, 32).astype(np.int64)
        out = np.empty((nb, 24), np.uint8)
        out[:, 0:2], out[:, 2:4] = _f16b(p["d"]), _f16b(p["dmin"])
        qh = (((c >> 4) & 1) << np.arange(32)[None, :]).sum(1).astype("<u4")
        out[:, 4:8] = qh.view(np.uint8).reshape(nb, 4)
        out[:, 8:24] = (c[:, :16] & 15) | ((c[:, 16:] & 15) << 4)
        return out

    def decode(self, p):
        return (p["codes"].astype(np.float32) * p["d"].astype(np.float32)[:, None, None]
                + p["dmin"].astype(np.float32)[:, None, None])

    def _cand(self, x, a, b, step, L):
        a, b = a[:, 0], b[:, 0]
        d16 = (a / step).astype(np.float16)
        d16 = np.where(d16 > 0, d16, np.float16(1))
        m16 = b.astype(np.float16)
        dd, mm = d16.astype(np.float64), m16.astype(np.float64)
        k = _assign(x, (dd * step)[:, None], mm[:, None], L)
        return {"d": d16, "dmin": m16, "sc": np.ones((x.shape[0], 1), np.int32),
                "mn": np.zeros((x.shape[0], 1), np.int32), "codes": (step * k).astype(np.int32)}


QTYPES = {q.type_id: q for q in (Q4K(), Q5K(), Q6K(), Q80(), Q51())}


def dequantize(qt, raw):
    """Flat float32 weights for raw block bytes (ggml reference dequantizer)."""
    blk = np.frombuffer(raw, np.uint8).reshape(-1, qt.bsize)
    return qt.decode(qt.unpack(blk)).reshape(-1)


def requantize_bytes(qt, raw, bits):
    """Requantize raw block bytes. Returns (new bytes, sum sq err, sum sq orig)."""
    blk = np.frombuffer(raw, np.uint8).reshape(-1, qt.bsize)
    x = qt.decode(qt.unpack(blk))
    parts, se = qt.requant(x, bits)
    return qt.pack(parts).tobytes(), float(se.sum()), float((x.astype(np.float64) ** 2).sum())


# ----------------------------------------------------------------------------
# Selection, workers, commands
# ----------------------------------------------------------------------------
def select(info, pattern):
    rx = re.compile(pattern)
    sel, skipped = [], []
    for t in info["tensors"]:
        if not rx.fullmatch(t["name"]):
            continue
        qt = QTYPES.get(t["type"])
        if qt is None or t["nbytes"] is None:
            skipped.append({"name": t["name"], "type": t["type_name"],
                            "reason": "unsupported type" if qt is None else "bad geometry"})
        else:
            sel.append(t)
    return sel, skipped


def _work(task):
    ti, src, out, type_id, off, nblocks, bits = task
    qt = QTYPES[type_id]
    with open(src, "rb") as f:
        f.seek(off)
        raw = f.read(nblocks * qt.bsize)
    if len(raw) != nblocks * qt.bsize:
        raise IOError("short read in tensor %d" % ti)
    new, se, nrm = requantize_bytes(qt, raw, bits)
    if len(new) != len(raw):
        raise AssertionError("byte size changed")
    with open(out, "r+b") as f:
        f.seek(off)
        f.write(new)
    return ti, se, nrm


def _tasks(sel, src, out, bits, chunk_blocks):
    for ti, t in enumerate(sel):
        qt = QTYPES[t["type"]]
        nb = t["nbytes"] // qt.bsize
        cb = chunk_blocks or max(1, 262144 // qt.elems)
        for s in range(0, nb, cb):
            n = min(cb, nb - s)
            yield (ti, src, out, t["type"], t["abs_offset"] + s * qt.bsize, n, bits)


def census(src, pattern, names=False):
    info = read_gguf(src)
    sel, skipped = select(info, pattern)
    rx = re.compile(pattern)
    agg = {}
    for t in info["tensors"]:
        if rx.fullmatch(t["name"]):
            c = agg.setdefault(t["type_name"], [0, 0, t["type"] in QTYPES])
            c[0] += 1
            c[1] += t["nbytes"] or 0
    print("file: %s" % src)
    print("tensors total: %d  alignment: %d  data_start: %d" %
          (len(info["tensors"]), info["alignment"], info["data_start"]))
    print("pattern: %s" % pattern)
    for k, (n, b, sup) in sorted(agg.items()):
        print("  %-8s count=%-5d bytes=%-14d %s" % (k, n, b, "supported" if sup else "SKIPPED"))
    print("supported matching: %d tensors, %d bytes; skipped: %d" %
          (len(sel), sum(t["nbytes"] for t in sel), len(skipped)))
    if names:
        for t in info["tensors"]:
            if rx.fullmatch(t["name"]):
                print("  %s  %s" % (t["name"], t["type_name"]))
    return agg


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def build(src, out, bits=3, identity="", pattern=DEFAULT_PATTERN, jobs=1, limit=None,
          source_sha256=None, hash_output=False, chunk_blocks=None, quiet=False):
    if not 2 <= bits <= 4:
        raise ValueError("bits must be 2..4")
    partial = out + ".partial"
    if os.path.exists(out) or os.path.exists(partial):
        raise FileExistsError("refusing to overwrite %s (or %s)" % (out, partial))
    t0 = time.time()
    info = read_gguf(src)
    sel, skipped = select(info, pattern)
    if limit is not None:
        for t in sel[limit:]:
            skipped.append({"name": t["name"], "type": t["type_name"], "reason": "limit-tensors"})
        sel = sel[:limit]
    shutil.copyfile(src, partial)
    se = [0.0] * len(sel)
    nrm = [0.0] * len(sel)
    left = [t["nbytes"] // QTYPES[t["type"]].bsize for t in sel]
    left = [-(-n // (chunk_blocks or max(1, 262144 // QTYPES[t["type"]].elems)))
            for n, t in zip(left, sel)]
    tasks = _tasks(sel, src, partial, bits, chunk_blocks)
    pool = multiprocessing.Pool(jobs) if jobs > 1 else None
    try:
        for ti, s, n in (pool.imap_unordered(_work, tasks) if pool else map(_work, tasks)):
            se[ti] += s
            nrm[ti] += n
            left[ti] -= 1
            if left[ti] == 0 and not quiet:
                print("done %s rel_mse=%.4g" % (sel[ti]["name"], se[ti] / nrm[ti] if nrm[ti] else 0.0),
                      file=sys.stderr, flush=True)
    finally:
        if pool:
            pool.close()
            pool.join()
    os.replace(partial, out)
    tens = []
    for t, s, n in zip(sel, se, nrm):
        qt = QTYPES[t["type"]]
        tens.append({"name": t["name"], "type": qt.name, "elements": t["n_elements"],
                     "rel_mse": s / n if n > 0 else 0.0})
    tot_se, tot_n = sum(se), sum(nrm)
    nbytes = sum(t["nbytes"] for t in sel)
    nel = sum(t["n_elements"] for t in sel)
    with open(os.path.abspath(__file__), "rb") as f:
        tool_sha = hashlib.sha256(f.read()).hexdigest()
    rep = {
        "identity": identity,
        "source": {"path": os.path.abspath(src), "size": os.path.getsize(src),
                   "sha256": source_sha256},
        "bits": bits, "grid_levels": 1 << bits, "pattern": pattern,
        "tensors": tens, "skipped": skipped,
        "totals": {"tensors_rewritten": len(sel), "tensors_skipped": len(skipped),
                   "bytes_rewritten": nbytes, "elements_rewritten": nel,
                   "rel_mse": tot_se / tot_n if tot_n > 0 else 0.0,
                   "effective_grid_bits": bits,
                   "storage_bits_per_weight": 8.0 * nbytes / nel if nel else 0.0,
                   "seconds": round(time.time() - t0, 2)},
        "tool_sha256": tool_sha,
        "output_sha256": _sha256_file(out) if hash_output else None,
    }
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1)
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("census")
    c.add_argument("src")
    c.add_argument("--pattern", default=DEFAULT_PATTERN)
    c.add_argument("--names", action="store_true")
    b = sub.add_parser("build")
    b.add_argument("src")
    b.add_argument("out")
    b.add_argument("--bits", type=int, default=3)
    b.add_argument("--identity", required=True)
    b.add_argument("--pattern", default=DEFAULT_PATTERN)
    b.add_argument("--jobs", type=int, default=1)
    b.add_argument("--limit-tensors", type=int, default=None)
    b.add_argument("--source-sha256", default=None)
    b.add_argument("--hash-output", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "census":
        census(a.src, a.pattern, a.names)
    else:
        rep = build(a.src, a.out, a.bits, a.identity, a.pattern, a.jobs, a.limit_tensors,
                    a.source_sha256, a.hash_output)
        t = rep["totals"]
        print("built %s: %d tensors, rel_mse=%.4g, skipped=%d" %
              (a.out, t["tensors_rewritten"], t["rel_mse"], t["tensors_skipped"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
