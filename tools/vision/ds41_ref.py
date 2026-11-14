#!/usr/bin/env python3
"""Reference dumps for the DeepSeek V4.1 image tower gate (docs: the vision adapter gate).

Stages, each with the smallest dependency set, so they run where the packages are:

  images      numpy + Pillow     writes the fixed synthetic image set (PNG) and cases.txt
  preprocess  numpy + Pillow     resize plan + contain + grey canvas + normalise + patchify -> NAME.pixels.u8, NAME.patches.f32
  encode      numpy              tower + aligner + special rows from the vision GGUF -> NAME.hidden.f32, NAME.rows.f32
  upstream    torch + safetensors + Pillow   the same dumps from the reference code inside the checkpoint
  compare     numpy              cosine, relative L2 and per-row cosine of two dump sets

`encode` is an independent FP32 implementation of the graph in numpy (its own GGUF parser, its own formulas): it
checks the engine's wiring. `upstream` loads `inference/vision.py` and `inference/image_processor.py` of the
checkpoint directory and runs them on the vision tensors only (the shards that the index lists for `vision.*`,
`aligner.*` and the three learned rows; the language model is never loaded). The directory must be the revision
named by the GGUF (`general.source.url`, `general.source.revision`): `compare` reports the special rows, which
are bit-equal only for the same weights. Its dumps carry a tag: NAME.upstream[-ours][-bf16].{patches,hidden,rows}.f32
(`-ours`: fed with the patches of `preprocess`; `-bf16`: the reference's own BF16 arithmetic).

All dumps are little-endian raw arrays; cases.txt lists one image per line:
  name width height resized_w resized_h grid_h grid_w span_rows graph_only
"""
import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# ---- options the checkpoint does not state (defaults = the adapter's recipe) -------------------------------------


def add_recipe(p):
    p.add_argument("--mean", default="0.5,0.5,0.5")
    p.add_argument("--std", default="0.5,0.5,0.5")
    p.add_argument("--filter", default="bicubic", choices=["bilinear", "bicubic", "lanczos"])
    p.add_argument("--gate-second", action="store_true", help="second half of mlp.w1 outputs is the SiLU gate")
    p.add_argument("--rope-adjacent", action="store_true", help="rotate pairs (2i, 2i+1) instead of (i, i+D/2)")
    p.add_argument("--aligner-act", default="gelu", choices=["gelu", "gelu_tanh", "silu"])
    p.add_argument("--unshuffle-col-major", action="store_true", help="aligner slot = dx*r+dy instead of dy*r+dx")
    p.add_argument("--no-last-newline", action="store_true", help="no row-end embedding after the last grid row")
    p.add_argument("--pad-fill", type=int, default=127, help="grey level of the canvas around an image that does not fill it")
    p.add_argument("--pixels-f32", action="store_true", help="keep the normalised pixels in FP32 (no BF16 rounding)")


# ---- GGUF (v3) ---------------------------------------------------------------------------------------------------

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


class Gguf:
    def __init__(self, path):
        self.data = np.memmap(path, dtype=np.uint8, mode="r")
        self.pos = 0
        if bytes(self.data[:4]) != b"GGUF":
            raise ValueError("not a GGUF file")
        self.pos = 4
        version, = self._unpack("<I")
        if version != 3:
            raise ValueError("unsupported GGUF version %d" % version)
        n_tensors, n_kv = self._unpack("<QQ")
        self.meta = {}
        for _ in range(n_kv):
            key = self._text()
            kind, = self._unpack("<I")
            self.meta[key] = self._value(kind)
        infos = []
        for _ in range(n_tensors):
            name = self._text()
            n_dims, = self._unpack("<I")
            dims = self._unpack("<%dQ" % n_dims)
            kind, offset = self._unpack("<IQ")
            infos.append((name, dims, kind, offset))
        align = int(self.meta.get("general.alignment", 32))
        base = (self.pos + align - 1) // align * align
        self.tensors = {name: (dims, kind, base + offset) for name, dims, kind, offset in infos}

    def _unpack(self, fmt):
        size = struct.calcsize(fmt)
        out = struct.unpack(fmt, bytes(self.data[self.pos:self.pos + size]))
        self.pos += size
        return out

    def _text(self):
        n, = self._unpack("<Q")
        s = bytes(self.data[self.pos:self.pos + n]).decode("utf-8")
        self.pos += n
        return s

    def _value(self, kind):
        if kind == 8:
            return self._text()
        if kind == 9:
            item, n = self._unpack("<IQ")
            return [self._value(item) for _ in range(n)]
        return self._unpack(_SCALAR[kind])[0]

    def digest(self):
        """SHA-256 of the whole file (computed once)."""
        if getattr(self, "_digest", None) is None:
            h = hashlib.sha256()
            for start in range(0, self.data.shape[0], 64 << 20):
                h.update(self.data[start:start + (64 << 20)])
            self._digest = h.hexdigest()
        return self._digest

    def f32(self, name):
        """Tensor as FP32 in the source framework's shape (GGUF dimensions reversed)."""
        dims, kind, offset = self.tensors[name]
        shape = tuple(int(d) for d in reversed(dims))
        count = int(np.prod(shape))
        if kind == 0:
            a = np.frombuffer(self.data, dtype="<f4", count=count, offset=offset)
        elif kind == 1:
            a = np.frombuffer(self.data, dtype="<f2", count=count, offset=offset).astype(np.float32)
        elif kind == 30:  # BF16: the high half of the FP32 pattern
            a = (np.frombuffer(self.data, dtype="<u2", count=count, offset=offset).astype(np.uint32) << 16).view(np.float32)
        else:
            raise ValueError("%s: unsupported tensor type %d" % (name, kind))
        return np.array(a, dtype=np.float32).reshape(shape)


def model_config(g):
    a = "deepseek4-vision."
    if g.meta.get("general.architecture") != "deepseek4-vision":
        raise ValueError("general.architecture is not deepseek4-vision")
    keys = {"layers": "block_count", "width": "embedding_length", "mlp": "feed_forward_length",
            "heads": "attention.head_count", "proj": "projection_length", "patch": "patch_size",
            "down": "downsample_ratio", "max_tokens": "image.max_tokens", "min_pixels": "image.min_pixels",
            "theta": "rope.freq_base", "eps": "attention.layer_norm_rms_epsilon"}
    return {k: g.meta[a + v] for k, v in keys.items()}


# ---- preprocessing -----------------------------------------------------------------------------------------------


def span_rows_of(grid_h, grid_w, down):
    """Rows of the image span: start, each merged row + its row end, end. Partial 3x3 cells count (ceil)."""
    mh, mw = -(-grid_h // down), -(-grid_w // down)
    return mh * (mw + 1) + 2


def plan_grid(width, height, patch, down, min_pixels, max_tokens):
    """Canvas (w, h) of the upstream resize plan: sides are multiples of the patch (not of the 3x3 cell); small images
    are scaled up to min_pixels; the whole span (special rows included) stays within max_tokens."""
    if 0 < width * height < min_pixels:
        ratio = (min_pixels / (width * height)) ** 0.5
        width, height = int(width * ratio), int(height * ratio)
    best_w, best_h = math.ceil(width / patch) * patch, math.ceil(height / patch) * patch
    if span_rows_of(best_h // patch, best_w // patch, down) > max_tokens:
        r = height / width
        max_w = math.sqrt((max_tokens - 2) / r + 0.25) - 0.5
        max_h = max_w * r
        cell = patch * down
        if max_w < 1.0:  # very tall: one column
            best_h, best_w = (max_tokens - 2) // 2 * cell, cell
        elif max_h < 1.0:  # very wide: one row
            best_h, best_w = cell, (max_tokens - 3) * cell
        else:
            beta = min(math.floor(max_w) * cell / width, math.floor(max_h) * cell / height)
            best_h, best_w = math.floor(height * beta / patch) * patch, math.floor(width * beta / patch) * patch
    return best_w, best_h


def contain_size(width, height, canvas_w, canvas_h):
    """Largest size of the image's aspect ratio inside the canvas (one side equals the canvas side)."""
    image_ratio, canvas_ratio = width / height, canvas_w / canvas_h
    if image_ratio > canvas_ratio:
        return canvas_w, round(height / width * canvas_w)
    if image_ratio < canvas_ratio:
        return round(width / height * canvas_h), canvas_h
    return canvas_w, canvas_h


def bf16_round(x):
    """float32 values rounded to the nearest BF16 value (ties to even), still stored as float32."""
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    u = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return u.view(np.float32)


def read_cases(folder):
    cases = []
    with open(os.path.join(folder, "cases.txt")) as f:
        for line in f:
            t = line.split()
            if t and not t[0].startswith("#"):
                cases.append([t[0]] + [int(x) for x in t[1:]])
    return cases


def write_cases(folder, cases):
    with open(os.path.join(folder, "cases.txt"), "w", newline="\n") as f:
        f.write("# name width height resized_w resized_h grid_h grid_w span_rows graph_only\n")
        for c in cases:
            f.write(" ".join(str(x) for x in c) + "\n")


def synthetic(name, w, h):
    """Deterministic integer images (no file or network source)."""
    y, x = np.mgrid[0:h, 0:w].astype(np.int64)
    if name.startswith("grad"):
        r, g, b = (x * 255) // max(w - 1, 1), (y * 255) // max(h - 1, 1), ((x + y) * 255) // max(w + h - 2, 1)
    elif name.startswith("checker"):
        k = ((x // 17) + (y // 23)) % 2
        r, g, b = k * 230 + 12, (1 - k) * 200 + (x % 29), (x * 7 + y * 13) % 256
    elif name.startswith("text"):  # dark strokes on a light page: thin horizontal runs with gaps
        line = (y % 24 < 9) & (y % 24 > 2)
        stroke = line & (((x * 2654435761 + (y // 24) * 40503) >> 7) % 5 < 3) & (x % 11 != 0)
        v = np.where(stroke, 24, 238)
        r, g, b = v, v, v + (x % 3)
    elif name.startswith("noise"):
        v = (x * 73856093) ^ (y * 19349663) ^ ((x * y) * 83492791)
        r, g, b = v % 256, (v >> 8) % 256, (v >> 16) % 256
    else:  # rings
        d = (x - w // 2) ** 2 + (y - h // 2) ** 2
        r, g, b = (d // 97) % 256, (d // 41) % 256, (x * 3 + y * 5) % 256
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


IMAGE_SET = [  # name, width, height: minimum, a 4:3 photo size, the token cap, tiny, 8:1, odd
    ("grad_min", 546, 546), ("rings_4x3", 1008, 756), ("checker_tiny", 16, 16), ("text_wide", 1600, 200),
    ("noise_odd", 517, 389), ("checker_cap", 1344, 1344),
]


def cmd_images(args):
    from PIL import Image
    os.makedirs(args.out, exist_ok=True)
    cases = []
    for name, w, h in IMAGE_SET:
        Image.fromarray(synthetic(name, w, h), "RGB").save(os.path.join(args.out, name + ".png"), optimize=False)
        cases.append([name, w, h, 0, 0, 0, 0, 0, 0])
    # Graph-only case: a grid below the minimum size, so the slow host reference can run the real weights quickly.
    Image.fromarray(synthetic("rings_graph", 168, 126), "RGB").save(os.path.join(args.out, "rings_graph.png"))
    cases.append(["rings_graph", 168, 126, 0, 0, 0, 0, 0, 1])
    write_cases(args.out, cases)
    print("images: %d written to %s" % (len(cases), args.out))


def cmd_preprocess(args):
    from PIL import Image
    mean = np.array([float(v) for v in args.mean.split(",")], dtype=np.float32)
    std = np.array([float(v) for v in args.std.split(",")], dtype=np.float32)
    resample = {"bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC, "lanczos": Image.LANCZOS}[args.filter]
    cases = read_cases(args.out)
    for c in cases:
        name, graph_only = c[0], c[8]
        im = Image.open(os.path.join(args.out, name + ".png")).convert("RGB")
        w, h = im.size
        tw, th = (w, h) if graph_only else plan_grid(w, h, args.patch, args.downsample, args.min_pixels, args.max_tokens)
        if tw <= 0 or th <= 0 or tw % args.patch or th % args.patch or \
                span_rows_of(th // args.patch, tw // args.patch, args.downsample) > args.max_tokens:
            raise ValueError("%s: %dx%d does not fit the token limit" % (name, tw, th))
        cw, ch = contain_size(w, h, tw, th)
        if min(cw, ch) > 0 and (cw, ch) != (w, h):
            im = im.resize((cw, ch), resample)
        if (cw, ch) != (tw, th):  # centred on a grey canvas; only one axis has a margin
            canvas = Image.new("RGB", (tw, th), (args.pad_fill,) * 3)
            if min(cw, ch) > 0:  # an extreme aspect ratio can leave less than half a pixel: the canvas stays grey
                canvas.paste(im, (round((tw - cw) * 0.5), 0) if cw != tw else (0, round((th - ch) * 0.5)))
            im = canvas
        px = np.asarray(im, dtype=np.uint8)
        # the upstream operation order, in FP32; the patch tensor is then rounded to BF16 before the patch embedding
        norm = (px.astype(np.float32) / np.float32(255.0) - mean) / std
        if not args.pixels_f32:
            norm = bf16_round(norm)
        gh, gw = th // args.patch, tw // args.patch
        # [gh][py][gw][px][c] -> rows in raster order, features (channel, y, x)
        patches = norm.reshape(gh, args.patch, gw, args.patch, 3).transpose(0, 2, 4, 1, 3).reshape(gh * gw, -1)
        px.tofile(os.path.join(args.out, name + ".pixels.u8"))
        np.ascontiguousarray(patches, dtype="<f4").tofile(os.path.join(args.out, name + ".patches.f32"))
        c[1:8] = [w, h, tw, th, gh, gw, span_rows_of(gh, gw, args.downsample) - (1 if args.no_last_newline else 0)]
        write_layout(args.out, name, "", args.downsample, {"patches": patches.shape[1]}, args)
        print("preprocess %-14s %4dx%-4d -> %4dx%-4d grid %3dx%-3d rows %4d" % (name, w, h, tw, th, gh, gw, c[7]))
    write_cases(args.out, cases)
    import PIL
    print("pillow", PIL.__version__, "numpy", np.__version__)


# ---- encoder -----------------------------------------------------------------------------------------------------


def rms_norm(x, weight, eps):
    return x / np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True) + eps).astype(np.float32) * weight


def silu(x):
    return x / (1.0 + np.exp(-x))


def erf(x):  # Abramowitz-Stegun 7.1.26 is too coarse for a gate; use math.erf elementwise
    return np.vectorize(math.erf, otypes=[np.float64])(x)


def act(kind, x):
    x = x.astype(np.float64)
    if kind == "gelu":
        y = 0.5 * x * (1.0 + erf(x * 0.7071067811865476))
    elif kind == "gelu_tanh":
        y = 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))
    else:
        y = x / (1.0 + np.exp(-x))
    return y.astype(np.float32)


def rope(v, cos, sin, adjacent):
    """v [N][H][D]; cos/sin [N][D/2]; pair i is (i, i+D/2) or (2i, 2i+1)."""
    out = np.empty_like(v)
    half = v.shape[-1] // 2
    a, b = (v[..., 0::2], v[..., 1::2]) if adjacent else (v[..., :half], v[..., half:])
    ra, rb = a * cos[:, None, :] - b * sin[:, None, :], b * cos[:, None, :] + a * sin[:, None, :]
    if adjacent:
        out[..., 0::2], out[..., 1::2] = ra, rb
    else:
        out[..., :half], out[..., half:] = ra, rb
    return out


# A saved residual stream is only as good as the computation that made it. The key names everything the stream
# depends on; a state with another key (or none) is refused, never mixed into a new run.
STATE_VERSION = "ds41-ref-state-1"
STATE_SECONDS = 10.0  # least time between two saves


def state_key(g, cfg, patches, gh, gw, args):
    h = hashlib.sha256()
    tower = ",".join("%s=%r" % (k, cfg[k]) for k in ("layers", "width", "heads", "mlp", "eps", "theta"))
    text = "%s|%s|%s|grid=%dx%d|gate_second=%d|rope_adjacent=%d|" % (
        STATE_VERSION, g.digest(), tower, gh, gw, bool(args.gate_second), bool(args.rope_adjacent))
    h.update(text.encode("utf-8"))
    h.update(np.ascontiguousarray(patches, dtype="<f4").tobytes())
    return h.hexdigest()


def save_state(path, x, layer, key):
    np.savez(path + ".tmp.npz", x=x, layer=layer, key=np.array(key))
    os.replace(path + ".tmp.npz", path + ".npz")


def load_state(path, key, shape, layers):
    """(x, first layer) of a state saved by the same computation; ValueError for anything else."""
    with np.load(path + ".npz", allow_pickle=False) as state:
        names = set(state.files)
        if names != {"x", "layer", "key"}:
            raise ValueError("%s.npz is not a state of this tool version: remove it to start over" % path)
        saved, x, layer = str(state["key"]), state["x"], state["layer"]
    if saved != key:
        raise ValueError("%s.npz is from another computation (weights, patches, grid or tower options differ): "
                         "remove it to start over" % path)
    if layer.shape != () or layer.dtype.kind not in "iu" or not 0 < int(layer) < layers:
        raise ValueError("%s.npz: layer outside 1..%d" % (path, layers - 1))
    if x.shape != tuple(shape) or x.dtype != np.float32 or not np.isfinite(x).all():
        raise ValueError("%s.npz: the saved stream is not a finite float32 %s array" % (path, tuple(shape)))
    return x, int(layer)


def encode_one(g, cfg, patches, gh, gw, args, checkpoint=None):
    L, W, H, M, R = cfg["layers"], cfg["width"], cfg["heads"], cfg["mlp"], cfg["down"]
    D, N, eps = W // H, gh * gw, float(cfg["eps"])
    x = patches @ g.f32("vision.patch_embed.proj.weight").T + g.f32("vision.patch_embed.proj.bias")
    # 2D rotary angles: D/4 frequencies per axis, row angles then column angles
    nf = D // 4
    inv = (1.0 / np.power(np.float64(cfg["theta"]), (2.0 * np.arange(nf, dtype=np.float32) / np.float32(D // 2)).astype(np.float64))).astype(np.float32)
    rows = (np.arange(N) // gw).astype(np.float32)
    cols = (np.arange(N) % gw).astype(np.float32)
    ang = np.concatenate([rows[:, None] * inv[None, :], cols[:, None] * inv[None, :]], axis=1).astype(np.float64)
    cos, sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)
    scale = np.float32(1.0 / math.sqrt(D))
    # A long image may not fit one time-limited run: the residual stream is saved between layers and resumed.
    first, saved = 0, time.time()
    key = state_key(g, cfg, patches, gh, gw, args) if checkpoint else None
    if checkpoint and os.path.exists(checkpoint + ".npz"):
        x, first = load_state(checkpoint, key, x.shape, L)
        print("  resumed at layer %d" % first)
    pool = ThreadPoolExecutor(max_workers=min(H, os.cpu_count() or 1))
    for l in range(first, L):
        p = "vision.blocks.%d." % l
        h = rms_norm(x, g.f32(p + "norm1.weight"), eps)
        qkv = h @ g.f32(p + "attn.wqkv.weight").T + g.f32(p + "attn.wqkv.bias")
        q = rope(qkv[:, :W].reshape(N, H, D), cos, sin, args.rope_adjacent)
        k = rope(qkv[:, W:2 * W].reshape(N, H, D), cos, sin, args.rope_adjacent)
        v = qkv[:, 2 * W:].reshape(N, H, D)
        o = np.empty((N, H, D), dtype=np.float32)

        def attend(head):  # numpy releases the interpreter lock inside these calls: one head per thread
            s = (q[:, head, :] @ k[:, head, :].T) * scale
            s -= s.max(axis=1, keepdims=True)
            np.exp(s, out=s)
            s /= s.sum(axis=1, keepdims=True)
            o[:, head, :] = s @ v[:, head, :]

        list(pool.map(attend, range(H)))
        x = x + o.reshape(N, W) @ g.f32(p + "attn.wo.weight").T + g.f32(p + "attn.wo.bias")
        h = rms_norm(x, g.f32(p + "norm2.weight"), eps)
        m = h @ g.f32(p + "mlp.w1.weight").T
        gate, up = (m[:, M:], m[:, :M]) if args.gate_second else (m[:, :M], m[:, M:])
        x = x + (silu(gate) * up) @ g.f32(p + "mlp.w2.weight").T
        if checkpoint and l + 1 < L and time.time() - saved >= STATE_SECONDS:
            save_state(checkpoint, x.astype(np.float32, copy=False), l + 1, key)
            saved = time.time()
    pool.shutdown()
    if checkpoint and os.path.exists(checkpoint + ".npz"):
        os.remove(checkpoint + ".npz")
    hidden = rms_norm(x, g.f32("vision.norm.weight"), eps).astype(np.float32)

    # 3x3 pixel unshuffle in the layout of torch pixel_unshuffle on [C][H][W]: feature c*R*R + dy*R + dx
    # A grid that is not a multiple of R is completed with zero rows on the right and at the bottom.
    mh, mw = -(-gh // R), -(-gw // R)
    padded = np.zeros((mh * R, mw * R, W), dtype=np.float32)
    padded[:gh, :gw] = hidden.reshape(gh, gw, W)
    t = padded.reshape(mh, R, mw, R, W)  # [my][dy][mx][dx][c]
    t = t.transpose(0, 2, 4, 3, 1) if args.unshuffle_col_major else t.transpose(0, 2, 4, 1, 3)
    merged = np.ascontiguousarray(t).reshape(mh * mw, W * R * R)
    y = merged @ g.f32("aligner.w1.weight").T + g.f32("aligner.w1.bias")
    y = act(args.aligner_act, y) @ g.f32("aligner.w2.weight").T + g.f32("aligner.w2.bias")
    y = y.astype(np.float32).reshape(mh, mw, -1)
    out = [g.f32("image_start")[None, :]]
    for r in range(mh):
        out.append(y[r])
        if r + 1 < mh or not args.no_last_newline:
            out.append(g.f32("image_newline")[None, :])
    out.append(g.f32("image_end")[None, :])
    return hidden, np.concatenate(out, axis=0).astype(np.float32)


def cmd_encode(args):
    g = Gguf(args.model)
    cfg = model_config(g)
    print("tower", cfg)
    for c in read_cases(args.out):
        name, gh, gw = c[0], c[5], c[6]
        if args.only and name not in args.only.split(","):
            continue
        metadata = read_layout(args.out, name, "")
        if metadata and (metadata["downsample"] != cfg["down"] or
                         metadata["recipe"]["no_last_newline"] != args.no_last_newline):
            raise ValueError("%s: preprocessing layout differs from encoder recipe" % name)
        patches = np.fromfile(os.path.join(args.out, name + ".patches.f32"), dtype="<f4").reshape(gh * gw, -1)
        t0 = time.time()
        hidden, rows = encode_one(g, cfg, patches, gh, gw, args, os.path.join(args.out, name + ".resume") if args.resume else None)
        if rows.shape[0] != c[7]:
            raise ValueError("%s: %d rows, cases.txt says %d" % (name, rows.shape[0], c[7]))
        if not args.no_write:
            np.ascontiguousarray(hidden, dtype="<f4").tofile(os.path.join(args.out, name + ".hidden.f32"))
            np.ascontiguousarray(rows, dtype="<f4").tofile(os.path.join(args.out, name + ".rows.f32"))
            write_layout(args.out, name, "", cfg["down"],
                         {"patches": patches.shape[1], "hidden": hidden.shape[1], "rows": rows.shape[1]}, args,
                         metadata.get("recipe"))
        # Row norms next to the learned rows' norms: a plausibility figure for the recipe options, not a gate.
        norms = np.linalg.norm(rows.astype(np.float64), axis=1)
        image = np.sort(norms[1:-1])
        print("encode %-14s patches %5d rows %5d  %.1f s  hidden rms %.4f  row L2 median %.3f max %.3f (start %.3f end %.3f)" % (
            name, gh * gw, rows.shape[0], time.time() - t0, float(np.sqrt(np.mean(hidden.astype(np.float64) ** 2))),
            float(image[len(image) // 2]), float(image[-1]), float(norms[0]), float(norms[-1])))
        sys.stdout.flush()


# ---- upstream: the checkpoint's own reference code ---------------------------------------------------------------

UPSTREAM_PREFIXES = ("vision.", "aligner.", "image_start", "image_end", "image_newline")


def tag_suffix(tag):
    return "." + tag if tag else ""


def read_layout(folder, name, tag):
    path = os.path.join(folder, name + tag_suffix(tag) + ".layout.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def write_layout(folder, name, tag, downsample, widths, args, preprocessing=None):
    keys = ("mean", "std", "filter", "pad_fill", "pixels_f32", "gate_second", "rope_adjacent",
            "aligner_act", "unshuffle_col_major", "no_last_newline")
    recipe = {k: getattr(args, k) for k in keys if hasattr(args, k)}
    metadata = {"downsample": downsample, "widths": widths, "recipe": recipe}
    if preprocessing is not None:
        metadata["preprocessing"] = preprocessing
    with open(os.path.join(folder, name + tag_suffix(tag) + ".layout.json"), "w") as f:
        json.dump(metadata, f, sort_keys=True)


def read_grids(folder, tag):
    """Keep recorded span rows and per-case recipe/width metadata; legacy untagged dumps used merge=3."""
    grids = {}
    if not tag:
        for c in read_cases(folder):
            metadata = read_layout(folder, c[0], tag)
            down = metadata.get("downsample", 3)
            if down <= 0:
                raise ValueError("invalid dump merge factor")
            grids[c[0]] = ((c[5], c[6], -(-c[5] // down), -(-c[6] // down), c[7]), metadata)
        return grids
    with open(os.path.join(folder, tag + ".cases.txt")) as f:
        for line in f:
            t = line.split()
            if t and not t[0].startswith("#"):
                grids[t[0]] = (tuple(int(x) for x in t[1:6]), read_layout(folder, t[0], tag))
    return grids


def load_upstream_module(model_dir, name):
    import importlib.util
    path = os.path.join(model_dir, "inference", name + ".py")
    spec = importlib.util.spec_from_file_location("ds41_upstream_" + name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


def cmd_upstream(args):
    """Runs the reference implementation that ships inside the checkpoint (inference/vision.py and
    inference/image_processor.py) on the vision tensors only; the language model is never loaded."""
    try:
        import json
        import types
        import torch
        from safetensors import safe_open
    except ImportError as e:
        sys.exit("upstream: needs torch, safetensors and Pillow: %s" % e)
    with open(os.path.join(args.model_dir, "config.json"), encoding="utf-8") as f:
        config = json.load(f)
    vc = config["vision_config"]
    ua = types.SimpleNamespace(
        dim=0, vision_n_layers=vc["num_hidden_layers"], vision_dim=vc["hidden_size"],
        vision_n_heads=vc["num_attention_heads"], vision_inter_dim=vc["intermediate_size"],
        vision_patch_size=vc["patch_size"], vision_rope_theta=float(vc["rope_theta"]),
        vision_downsample_ratio=vc["downsample_ratio"], vision_max_n_token=vc["max_image_tokens"],
        vision_min_pixels=vc["min_pixels"], vision_max_wh_ratio=vc.get("max_wh_ratio"))
    with open(os.path.join(args.model_dir, "model.safetensors.index.json"), encoding="utf-8") as f:
        weight_map = json.load(f)["weight_map"]
    tensors, by_file = {}, {}
    for key, shard in weight_map.items():
        if key.startswith(UPSTREAM_PREFIXES):
            by_file.setdefault(shard, []).append(key)
    for shard, keys in sorted(by_file.items()):
        with safe_open(os.path.join(args.model_dir, shard), framework="pt") as st:
            for key in keys:
                tensors[key] = st.get_tensor(key)
    ua.dim = tensors["image_start"].numel()
    vision = load_upstream_module(args.model_dir, "vision")
    processor = load_upstream_module(args.model_dir, "image_processor")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    vit, aligner = vision.ViT(ua), vision.Aligner(ua)
    vit.load_state_dict({k[len("vision."):]: v for k, v in tensors.items() if k.startswith("vision.")})
    aligner.load_state_dict({k[len("aligner."):]: v for k, v in tensors.items() if k.startswith("aligner.")})
    vit, aligner = vit.to(dtype).eval(), aligner.to(dtype).eval()
    tag = "upstream" + ("-ours" if args.patches == "ours" else "") + ("-bf16" if args.dtype == "bf16" else "")
    only = set(filter(None, args.only.split(",")))
    p, r = ua.vision_patch_size, ua.vision_downsample_ratio
    cases_file = os.path.join(args.out, tag + ".cases.txt")
    done = {}
    if only and os.path.exists(cases_file):
        with open(cases_file) as f:
            done = {line.split()[0]: line.strip() for line in f if line.strip()}
    print("upstream: torch %s, %s, patches from %s, %d tensors" % (torch.__version__, args.dtype, args.patches, len(tensors)))
    for c in read_cases(args.out):
        name = c[0]
        if only and name not in only:
            continue
        t0 = time.time()
        if args.patches == "ours":
            gh, gw = c[5], c[6]
            raw = np.fromfile(os.path.join(args.out, name + ".patches.f32"), dtype="<f4").reshape(gh * gw, 3, p, p)
            patches = torch.from_numpy(raw.copy())
            mh, mw = -(-gh // r), -(-gw // r)
        else:
            if c[8]:
                continue  # graph-only cases have no resize step
            with open(os.path.join(args.out, name + ".png"), "rb") as f:
                patches, gh, gw, mh, mw = processor.load_image({"data": f.read()}, ua)
            np.ascontiguousarray(patches.float().reshape(gh * gw, -1).numpy(), dtype="<f4").tofile(
                os.path.join(args.out, "%s.%s.patches.f32" % (name, tag)))
        with torch.no_grad():
            hidden = vit(patches.to(dtype), gh, gw)
            image_rows = aligner(hidden, gh, gw)
        kinds = processor.image_token_types(mh, mw)
        rows = torch.empty(kinds.numel(), ua.dim, dtype=torch.float32)
        rows[kinds == processor.IMAGE] = image_rows.float()
        rows[kinds == processor.IMAGE_START] = tensors["image_start"].float()
        rows[kinds == processor.IMAGE_END] = tensors["image_end"].float()
        rows[kinds == processor.IMAGE_NEW_LINE] = tensors["image_newline"].float()
        np.ascontiguousarray(hidden.float().numpy(), dtype="<f4").tofile(os.path.join(args.out, "%s.%s.hidden.f32" % (name, tag)))
        np.ascontiguousarray(rows.numpy(), dtype="<f4").tofile(os.path.join(args.out, "%s.%s.rows.f32" % (name, tag)))
        write_layout(args.out, name, tag, r,
                     {"patches": 3 * p * p, "hidden": hidden.shape[1], "rows": rows.shape[1]}, args)
        done[name] = "%s %d %d %d %d %d" % (name, gh, gw, mh, mw, kinds.numel())
        with open(cases_file, "w", newline="\n") as f:
            f.write("\n".join(done.values()) + "\n")
        print("upstream %-14s grid %3dx%-3d merged %2dx%-2d span %4d  %.1f s" % (name, gh, gw, mh, mw, kinds.numel(), time.time() - t0), flush=True)


def row_stats(a, b):
    """cosine of the whole arrays, relative L2, mean and minimum of the per-row cosines, index of the worst row."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    cos = float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))
    rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))
    rc = (a * b).sum(axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-300)
    return cos, rel, float(rc.mean()), float(rc.min()), int(rc.argmin())


def cmd_compare(args):
    """Compares two dump sets (tags; '' = the adapter's own dumps). b is the reference of the relative figures."""
    ga, gb = read_grids(args.out, args.a), read_grids(args.out, args.b)
    print("# compare a='%s' against b='%s': cosine, relative L2, per-row cosine mean / min (worst row)" % (args.a, args.b))
    compared = skipped = failed = 0
    strict = getattr(args, "strict", False)
    threshold = getattr(args, "max_rel_l2", 0.0)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("max-rel-l2 must be finite and nonnegative")
    for name in sorted(ga.keys() | gb.keys()):
        if name not in ga or name not in gb:
            skipped += 3
            continue
        grid_a, meta_a = ga[name]
        grid_b, meta_b = gb[name]
        if grid_a != grid_b:
            print("%-13s GRID/LAYOUT DIFFERS: %s vs %s" % (name, grid_a, grid_b))
            skipped += 3
            continue
        gh, gw, mh, mw, span = grid_a
        last_newline = span == mh * (mw + 1) + 2
        if min(gh, gw, mh, mw) <= 0 or span not in (mh * (mw + 1) + 1, mh * (mw + 1) + 2):
            raise ValueError("%s: invalid recorded span layout" % name)
        for metadata in (meta_a, meta_b):
            no_last = metadata.get("recipe", {}).get("no_last_newline")
            if no_last is not None and no_last == last_newline:
                raise ValueError("%s: span disagrees with recorded recipe" % name)
        for kind in ("patches", "hidden", "rows"):
            fa = os.path.join(args.out, name + tag_suffix(args.a) + "." + kind + ".f32")
            fb = os.path.join(args.out, name + tag_suffix(args.b) + "." + kind + ".f32")
            if not (os.path.exists(fa) and os.path.exists(fb)):
                skipped += 1
                continue
            a, b = np.fromfile(fa, dtype="<f4"), np.fromfile(fb, dtype="<f4")
            if a.size != b.size:
                print("%-13s %-7s SIZE DIFFERS %d vs %d" % (name, kind, a.size, b.size))
                skipped += 1
                continue
            n = gh * gw if kind != "rows" else span
            for values, metadata in ((a, meta_a), (b, meta_b)):
                width = metadata.get("widths", {}).get(kind)
                if values.size == 0 or values.size % n or (width is not None and values.size != n * width):
                    raise ValueError("%s %s: invalid feature width" % (name, kind))
                if strict and width is None:
                    raise ValueError("%s %s: strict comparison requires recorded feature widths" % (name, kind))
            a, b = a.reshape(n, -1), b.reshape(n, -1)
            if kind == "rows":
                image = np.ones(n, dtype=bool)
                image[0] = image[-1] = False
                image[np.arange(mh if last_newline else mh - 1) * (mw + 1) + mw + 1] = False
                special = float(np.abs(a[~image] - b[~image]).max())
                extra = "  special rows max abs diff %.3g" % special
                if special != 0.0:
                    extra += "  (special rows differ; check weights and layout)"
                a, b = a[image], b[image]
            else:
                extra = "  max abs diff %.3g" % float(np.abs(a - b).max())
            cos, rel, mean, low, worst = row_stats(a, b)
            compared += 1
            if not all(math.isfinite(v) for v in (cos, rel, mean, low)) or rel > threshold or (kind == "rows" and special != 0):
                failed += 1
            print("%-13s %-7s cos %.6f relL2 %.5f rows mean %.6f min %.6f (row %d)%s" % (name, kind, cos, rel, mean, low, worst, extra))
    print("compared %d, skipped %d, outside threshold %d" % (compared, skipped, failed))
    if strict and (compared == 0 or skipped or failed):
        raise ValueError("strict comparison failed: missing coverage or numerical mismatch")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("images")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_images)
    p = sub.add_parser("preprocess")
    p.add_argument("--out", required=True)
    p.add_argument("--patch", type=int, default=14)
    p.add_argument("--downsample", type=int, default=3)
    p.add_argument("--min-pixels", type=int, default=295936)
    p.add_argument("--max-tokens", type=int, default=1024)
    add_recipe(p)
    p.set_defaults(fn=cmd_preprocess)
    p = sub.add_parser("encode")
    p.add_argument("--out", required=True)
    p.add_argument("--model", required=True, help="the vision GGUF")
    p.add_argument("--only", default="", help="comma-separated case names")
    p.add_argument("--no-write", action="store_true", help="print the figures only")
    p.add_argument("--resume", action="store_true", help="save the state between layers and continue a stopped run")
    add_recipe(p)
    p.set_defaults(fn=cmd_encode)
    p = sub.add_parser("upstream")
    p.add_argument("--out", required=True)
    p.add_argument("--model-dir", required=True, help="upstream checkpoint directory: config, index, inference/, vision shards")
    p.add_argument("--dtype", default="f32", choices=["f32", "bf16"], help="arithmetic of the upstream modules")
    p.add_argument("--patches", default="upstream", choices=["upstream", "ours"],
                   help="upstream: its own image processor; ours: NAME.patches.f32 of `preprocess` (graph check)")
    p.add_argument("--only", default="", help="comma-separated case names")
    p.set_defaults(fn=cmd_upstream)
    p = sub.add_parser("compare")
    p.add_argument("--out", required=True)
    p.add_argument("--a", default="", help="tag of the first dump set ('' = preprocess/encode)")
    p.add_argument("--b", default="upstream", help="tag of the reference dump set")
    p.add_argument("--strict", action="store_true", help="require nonempty complete coverage and numerical agreement")
    p.add_argument("--max-rel-l2", type=float, default=0.0, help="strict relative L2 limit; special rows must match exactly")
    p.set_defaults(fn=cmd_compare)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
