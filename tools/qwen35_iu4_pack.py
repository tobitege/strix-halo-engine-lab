#!/usr/bin/env python3
"""HF BF16 safetensors -> GGUF pack with the dense projections in the shared IU4 g64 layout (src/core/iu4.hpp).

Every tensor of the reference GGUF (e.g. the Q8_0 file) keeps its name, shape and metadata; the dense projections
listed in PROJECTIONS are re-quantised from the BF16 source with the `mse` scale rule of src/core/iu4.hpp
(per K64 group clip search 1.00..0.60 step .02 on [-8, 7], FP16 scale, RNE nibbles) and stored as ggml type 200
(strix::gguf::type_strix_iu4_g64): tiles [K/16][N/16][column within tile][K within tile / 2] signed nibbles
(low nibble = even k) followed by FP16 scales [N][K/64]. Everything else is copied byte for byte from the reference.

  qwen35_iu4_pack.py --hf DIR --reference REF.gguf --out OUT.gguf [--jobs 8] [--only REGEX] [--check NAME]
Needs numpy only. N (outputs) must be a multiple of 64 for a converted tensor, K a multiple of 64.
"""
import argparse, json, os, re, struct, sys
import numpy as np

TYPE_IU4 = 200
GEOM = {0: (1, 4), 1: (1, 2), 30: (1, 2), 8: (32, 34), TYPE_IU4: (64, 34)}
SCALAR = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: 'B', 10: 'Q', 11: 'q', 12: 'd'}
CLIPS = [np.float32(1.0) - np.float32(0.02) * np.float32(i) for i in range(21)]

# GGUF tensor -> HF tensor (per layer N). Output head: lm_head.
PROJECTIONS = {
    r'blk\.(\d+)\.attn_qkv\.weight': 'model.language_model.layers.{0}.linear_attn.in_proj_qkv.weight',
    r'blk\.(\d+)\.attn_gate\.weight': 'model.language_model.layers.{0}.linear_attn.in_proj_z.weight',
    r'blk\.(\d+)\.ssm_out\.weight': 'model.language_model.layers.{0}.linear_attn.out_proj.weight',
    r'blk\.(\d+)\.attn_q\.weight': 'model.language_model.layers.{0}.self_attn.q_proj.weight',
    r'blk\.(\d+)\.attn_k\.weight': 'model.language_model.layers.{0}.self_attn.k_proj.weight',
    r'blk\.(\d+)\.attn_v\.weight': 'model.language_model.layers.{0}.self_attn.v_proj.weight',
    r'blk\.(\d+)\.attn_output\.weight': 'model.language_model.layers.{0}.self_attn.o_proj.weight',
    r'blk\.(\d+)\.ffn_gate\.weight': 'model.language_model.layers.{0}.mlp.gate_proj.weight',
    r'blk\.(\d+)\.ffn_up\.weight': 'model.language_model.layers.{0}.mlp.up_proj.weight',
    r'blk\.(\d+)\.ffn_down\.weight': 'model.language_model.layers.{0}.mlp.down_proj.weight',
    r'output\.weight': 'lm_head.weight',
}


class SafeTensors:
    def __init__(self, directory):
        self.dir = directory
        index = os.path.join(directory, 'model.safetensors.index.json')
        files = sorted(set(json.load(open(index))['weight_map'].values())) if os.path.exists(index) else \
            sorted(f for f in os.listdir(directory) if f.endswith('.safetensors'))
        self.meta = {}
        for fn in files:
            with open(os.path.join(directory, fn), 'rb') as f:
                n, = struct.unpack('<Q', f.read(8))
                header = json.loads(f.read(n))
            for k, v in header.items():
                if k != '__metadata__':
                    self.meta[k] = (fn, 8 + n + v['data_offsets'][0], v['data_offsets'][1] - v['data_offsets'][0], v['dtype'], v['shape'])

    def get(self, name, rows=None):
        fn, off, size, dtype, shape = self.meta[name]
        if dtype != 'BF16':
            raise ValueError(name + ': only BF16 sources are supported, got ' + dtype)
        count = int(np.prod(shape))
        row_elems = count // shape[0]
        n = count if rows is None else rows * row_elems
        raw = np.fromfile(os.path.join(self.dir, fn), dtype=np.uint16, count=n, offset=off)
        out = (raw.astype(np.uint32) << 16).view(np.float32)
        return out.reshape((-1,) + tuple(shape[1:]))


def _rd(f, fmt):
    return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))


def _rstr(f):
    n, = _rd(f, 'Q')
    return f.read(n)


def _skip(f, t):
    if t == 8:
        _rstr(f)
    elif t == 9:
        et, n = _rd(f, 'IQ')
        for _ in range(n):
            _skip(f, et)
    else:
        f.read(struct.calcsize(SCALAR[t]))


class Gguf:
    def __init__(self, path):
        self.path = path
        with open(path, 'rb') as f:
            assert f.read(4) == b'GGUF'
            self.version, nt, nkv = _rd(f, 'IQQ')
            start = f.tell()
            self.align = 32
            for _ in range(nkv):
                k = _rstr(f)
                t, = _rd(f, 'I')
                if k == b'general.alignment':
                    self.align, = _rd(f, 'I')
                else:
                    _skip(f, t)
            end = f.tell()
            f.seek(start)
            self.kv_raw = f.read(end - start)
            self.nkv = nkv
            self.tensors = []
            for _ in range(nt):
                n = _rstr(f).decode()
                nd, = _rd(f, 'I')
                dims = _rd(f, 'Q' * nd)
                ty, off = _rd(f, 'IQ')
                self.tensors.append((n, dims, ty, off))
            self.base = (f.tell() + self.align - 1) // self.align * self.align
        self.by_name = {t[0]: t for t in self.tensors}

    @staticmethod
    def nbytes(dims, ty):
        b, s = GEOM[ty]
        return int(np.prod(dims)) // b * s

    def raw(self, name, size=None, skip=0):
        n, dims, ty, off = self.by_name[name]
        with open(self.path, 'rb') as f:
            f.seek(self.base + off + skip)
            return f.read(self.nbytes(dims, ty) - skip if size is None else size)

    def dequant(self, name, rows=None):
        n, dims, ty, off = self.by_name[name]
        k = dims[0]
        nrows = int(np.prod(dims[1:])) if len(dims) > 1 else 1
        nrows = nrows if rows is None else rows
        if ty == 8:
            buf = np.frombuffer(self.raw(name, nrows * k // 32 * 34), dtype=np.uint8).reshape(nrows, k // 32, 34)
            d = np.ascontiguousarray(buf[:, :, :2]).view(np.float16).astype(np.float32)
            q = np.ascontiguousarray(buf[:, :, 2:]).view(np.int8).astype(np.float32)
            return (q * d).reshape(nrows, k)
        if ty == TYPE_IU4:
            return dequant_iu4(self.raw(name), nrows, k)
        dt = {0: np.float32, 1: np.float16}[ty]
        return np.frombuffer(self.raw(name, nrows * k * np.dtype(dt).itemsize), dtype=dt).astype(np.float32).reshape(nrows, k)


def quantise_mse(w, chunk=128):
    """w: float32 [N][K] (row = output). Returns (packed uint8 [N*K/2] tile layout, scales fp16 [N][K/64])."""
    n_out, k = w.shape
    assert n_out % 64 == 0 and k % 64 == 0
    groups = k // 64
    q_all = np.empty((n_out, k), dtype=np.int8)
    s_all = np.empty((n_out, groups), dtype=np.float16)
    for r0 in range(0, n_out, chunk):
        x = np.ascontiguousarray(w[r0:r0 + chunk]).reshape(-1, groups, 64)
        pos = np.maximum(x.max(axis=2), 0)
        neg = np.maximum((-x).max(axis=2), 0)
        base = np.maximum(pos / np.float32(7), neg / np.float32(8))
        best = np.full(base.shape, np.inf, dtype=np.float32)
        best_s = np.ones(base.shape, dtype=np.float16)
        for c in CLIPS:
            raw = np.maximum(c * base, np.float32(2.0 ** -24))
            cb = raw.astype(np.float16)
            cs = cb.astype(np.float32)
            inv = np.float32(1) / cs
            q = np.clip(np.rint(x * inv[:, :, None]), -8, 7)
            d = x - cs[:, :, None] * q
            err = np.einsum('ngk,ngk->ng', d, d)
            better = err < best
            best = np.where(better, err, best)
            best_s = np.where(better, cb, best_s)
        zero = base == 0
        best_s = np.where(zero, np.float16(1.0), best_s)
        cs = best_s.astype(np.float32)
        q = np.clip(np.rint(x * (np.float32(1) / cs)[:, :, None]), -8, 7).astype(np.int8)
        q[zero] = 0
        q_all[r0:r0 + chunk] = q.reshape(-1, k)
        s_all[r0:r0 + chunk] = best_s
    return pack_tiles(q_all), s_all


def pack_tiles(q):
    """int8 [N][K] in [-8, 7] -> bytes [K/16][N/16][16 columns][8] (core/iu4.hpp iu4_weight_byte)."""
    n_out, k = q.shape
    u = (q.astype(np.int16) & 15).astype(np.uint8).reshape(n_out // 16, 16, k // 16, 8, 2)
    byte = u[..., 0] | (u[..., 1] << 4)  # [nt][c][kt][p]
    return np.ascontiguousarray(byte.transpose(2, 0, 1, 3)).reshape(-1)


def unpack_tiles(packed, n_out, k):
    b = np.asarray(packed, dtype=np.uint8).reshape(k // 16, n_out // 16, 16, 8).transpose(1, 2, 0, 3)  # [nt][c][kt][p]
    lo = (b & 15).astype(np.int8)
    hi = (b >> 4).astype(np.int8)
    q = np.stack([lo, hi], axis=-1).reshape(n_out, k)
    return np.where(q >= 8, q - 16, q)


def dequant_iu4(blob, n_out, k):
    packed = np.frombuffer(blob, dtype=np.uint8, count=n_out * k // 2)
    scales = np.frombuffer(blob, dtype=np.float16, count=n_out * (k // 64), offset=n_out * k // 2).reshape(n_out, k // 64)
    q = unpack_tiles(packed, n_out, k).astype(np.float32).reshape(n_out, k // 64, 64)
    return (q * scales.astype(np.float32)[:, :, None]).reshape(n_out, k)


def hf_name_for(name):
    for pat, tmpl in PROJECTIONS.items():
        m = re.fullmatch(pat, name)
        if m:
            return tmpl.format(*m.groups())
    return None


def transform(name, w):
    """HF layout -> GGUF layout for projection `name` (identity unless a reordering was derived, see TRANSFORMS)."""
    fn = TRANSFORMS.get(re.sub(r'blk\.\d+\.', '', name))
    return w if fn is None else fn(w)


def _v_rows(w, head_dim, first=0):
    # HF groups the 48 value heads per key head (16 x 3); the GGUF stores them tiled (3 x 16).
    head = w[first:].reshape((16, 3, head_dim) + w.shape[1:])
    out = w.copy()
    out[first:] = head.transpose((1, 0, 2) + tuple(range(3, head.ndim))).reshape(w[first:].shape)
    return out


# Derived by comparing HF BF16 with the dequantised Q8_0 reference (max |diff| = Q8 rounding): the GDN value-head
# reordering. Rows of in_proj_qkv after q and k (4096), in_proj_z, in_proj_a/b; columns of out_proj.
TRANSFORMS = {
    'attn_qkv.weight': lambda w: _v_rows(w, 128, 4096),
    'attn_gate.weight': lambda w: _v_rows(w, 128),
    'ssm_out.weight': lambda w: w.reshape(w.shape[0], 16, 3, 128).transpose(0, 2, 1, 3).reshape(w.shape),
}


def convert_one(args):
    hf_dir, name, dims, nrows = args
    st = SafeTensors(hf_dir)
    w = transform(name, st.get(hf_name_for(name)))
    assert w.shape == (nrows, dims[0]), (name, w.shape, dims)
    packed, scales = quantise_mse(w)
    return packed.tobytes() + scales.tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hf', required=True)
    ap.add_argument('--reference', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=4)
    ap.add_argument('--only', default=None, help='convert only tensors matching this regex (the rest are copied)')
    a = ap.parse_args()
    ref = Gguf(a.reference)
    have = SafeTensors(a.hf).meta
    plan = []  # (name, dims, ty, blob_size, convert)
    for name, dims, ty, off in ref.tensors:
        # tensors without a BF16 source (the next-token-prediction layer, blk.<block_count>) are copied as in the reference
        conv = hf_name_for(name) in have and (a.only is None or re.search(a.only, name))
        if conv:
            assert ty == 8 and dims[1] % 64 == 0 and dims[0] % 64 == 0, name
            plan.append((name, dims, TYPE_IU4, True))
        else:
            plan.append((name, dims, ty, False))
    with open(a.out, 'wb') as out:
        out.write(b'GGUF' + struct.pack('<IQQ', ref.version, len(plan), ref.nkv))
        out.write(ref.kv_raw)
        off = 0
        offsets = []
        for name, dims, ty, conv in plan:
            nb = Gguf.nbytes(dims, ty)
            nm = name.encode()
            out.write(struct.pack('<Q', len(nm)) + nm + struct.pack('<I', len(dims)) + struct.pack('<' + 'Q' * len(dims), *dims) +
                      struct.pack('<IQ', ty, off))
            offsets.append(off)
            off = (off + nb + ref.align - 1) // ref.align * ref.align
        pos = out.tell()
        out.write(b'\0' * ((pos + ref.align - 1) // ref.align * ref.align - pos))
        base = out.tell()
        import multiprocessing as mp
        jobs = [(a.hf, n, d, d[1]) for n, d, t, c in plan if c]
        pool = mp.Pool(a.jobs)
        results = pool.imap(convert_one, jobs)
        for (name, dims, ty, conv), o in zip(plan, offsets):
            out.seek(base + o)
            blob = next(results) if conv else ref.raw(name)
            assert len(blob) == Gguf.nbytes(dims, ty), name
            out.write(blob)
            print(name, 'iu4' if conv else 'copy', len(blob), flush=True)
        out.truncate(base + off if offsets else base)
    print('size', os.path.getsize(a.out))


if __name__ == '__main__':
    main()
