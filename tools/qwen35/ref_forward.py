#!/usr/bin/env python3
"""Transformers-free numpy reference for the GGUF "qwen35" family (float64 accumulation, one layer of weights at a time).

  ref_forward.py MODEL.gguf --ids 1,2,3 | --ids-file F  [--n 32] [--out DIR] [--mtp-hidden post|pre]
Outputs DIR/ref.npz (logits [T,V] f32, greedy ids, decode_logits [N-1,V], mtp_logits [T+N-1,V],
layer norms) and DIR/summary.txt. Reads bf16 and q8_0 packs. Semantics: design.md section 1.
MTP: position i feeds emb(token i+1) and the trunk hidden at i (post final norm by default;
--mtp-hidden pre uses the residual stream before output_norm); RoPE position i; own KV; predicts token i+2.
Home: specific tool (model arithmetic).
"""
import argparse, os, struct, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np

BF16, F32, F16, Q8_0 = 30, 0, 1, 8


class GGUF:
    def __init__(self, path):
        self.f = open(path, "rb")
        f = self.f
        assert f.read(4) == b"GGUF"
        ver, nt, nkv = struct.unpack("<IQQ", f.read(20))
        assert ver == 3
        self.kv = {}
        for _ in range(nkv):
            k = self._str()
            self.kv[k] = self._val(struct.unpack("<I", f.read(4))[0])
        self.t = {}
        for _ in range(nt):
            name = self._str()
            nd = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack("<%dQ" % nd, f.read(8 * nd))
            gt, off = struct.unpack("<IQ", f.read(12))
            self.t[name] = (tuple(reversed(dims)), gt, off)  # numpy order
        al = self.kv.get("general.alignment", 32)
        self.base = (f.tell() + al - 1) // al * al
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")

    def _str(self):
        n = struct.unpack("<Q", self.f.read(8))[0]
        return self.f.read(n).decode("utf-8")

    def _val(self, t):
        f = self.f
        if t == 8:
            return self._str()
        if t == 9:
            et, n = struct.unpack("<IQ", f.read(12))
            return [self._val(et) for _ in range(n)]
        fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}[t]
        return struct.unpack("<" + fmt, f.read(struct.calcsize(fmt)))[0]

    def shape(self, name):
        return self.t[name][0]

    def rows(self, name, r0, r1):
        """Rows r0:r1 of a 2-D tensor (or whole 1-D) dequantised to float64."""
        shape, gt, off = self.t[name]
        n_in = shape[-1]
        if len(shape) == 1:
            r0, r1, n_in = 0, 1, shape[0]
        isz = {BF16: 2, F32: 4}.get(gt)
        o = self.base + off
        if gt == Q8_0:
            rb = n_in // 32 * 34
            raw = np.asarray(self.mm[o + r0 * rb:o + r1 * rb]).reshape(r1 - r0, n_in // 32, 34)
            d = raw[:, :, :2].copy().view(np.float16).astype(np.float64)  # [R,nb,1]
            q = raw[:, :, 2:].view(np.int8).astype(np.float64)
            out = (q * d).reshape(r1 - r0, n_in)
        else:
            rb = n_in * isz
            raw = np.asarray(self.mm[o + r0 * rb:o + r1 * rb])
            if gt == BF16:
                out = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
            else:
                out = raw.view(np.float32).astype(np.float64)
            out = out.reshape(r1 - r0, n_in)
        return out[0] if len(shape) == 1 else out

    def full(self, name):
        s = self.shape(name)
        return self.rows(name, 0, s[0]) if len(s) > 1 else self.rows(name, 0, 1)


POOL = ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4))


def linear(g, x, name, chunk_elems=16 << 20):
    """x [T,K] f64 -> x @ W.T, W streamed in row chunks."""
    n, k = g.shape(name)
    step = max(1, chunk_elems // k)
    out = np.empty((x.shape[0], n))

    def work(r0):
        r1 = min(n, r0 + step)
        out[:, r0:r1] = x @ g.rows(name, r0, r1).T
    list(POOL.map(work, range(0, n, step)))
    return out


def rms(x, gain, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * gain


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Model:
    def __init__(self, path):
        self.g = g = GGUF(path)
        k = g.kv
        p = "qwen35."
        self.L = k[p + "block_count"]
        self.H = k[p + "embedding_length"]
        self.nh, self.nkv = k[p + "attention.head_count"], k[p + "attention.head_count_kv"]
        self.hd = k[p + "attention.key_length"]
        self.eps = k[p + "attention.layer_norm_rms_epsilon"]
        self.theta, self.rd = k[p + "rope.freq_base"], k[p + "rope.dimension_count"]
        self.K = k[p + "ssm.conv_kernel"]
        self.dk = k[p + "ssm.state_size"]
        self.nk, self.nv = k[p + "ssm.group_count"], k[p + "ssm.time_step_rank"]
        self.interval = k[p + "full_attention_interval"]
        self.nmtp = k[p + "nextn_predict_layers"]
        self.V = g.shape("token_embd.weight")[0]
        self.caches = {}

    def is_attn(self, l):
        return (l + 1) % self.interval == 0

    def embed(self, ids):
        return np.stack([self.g.rows("token_embd.weight", int(i), int(i) + 1)[0] for i in ids])

    def mlp(self, b, x):
        g = self.g
        h = rms(x, g.full(f"{b}.post_attention_norm.weight"), self.eps)
        a = silu(linear(g, h, f"{b}.ffn_gate.weight")) * linear(g, h, f"{b}.ffn_up.weight")
        return x + linear(g, a, f"{b}.ffn_down.weight")

    def rope(self, x, pos):
        """x [T,heads,hd]; NeoX rotate-half over first rd dims."""
        rd = self.rd
        half = rd // 2
        inv = self.theta ** (-2.0 * np.arange(half) / rd)
        ang = pos[:, None] * inv[None, :]
        c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
        x1, x2 = x[..., :half], x[..., half:rd]
        out = x.copy()
        out[..., :half] = x1 * c - x2 * s
        out[..., half:rd] = x2 * c + x1 * s
        return out

    def attn(self, b, x, pos, cache):
        g, nh, nkv, hd = self.g, self.nh, self.nkv, self.hd
        T = x.shape[0]
        h = rms(x, g.full(f"{b}.attn_norm.weight"), self.eps)
        qg = linear(g, h, f"{b}.attn_q.weight").reshape(T, nh, 2 * hd)
        q, gate = qg[..., :hd], qg[..., hd:]
        k = linear(g, h, f"{b}.attn_k.weight").reshape(T, nkv, hd)
        v = linear(g, h, f"{b}.attn_v.weight").reshape(T, nkv, hd)
        q = self.rope(rms(q, g.full(f"{b}.attn_q_norm.weight"), self.eps), pos)
        k = self.rope(rms(k, g.full(f"{b}.attn_k_norm.weight"), self.eps), pos)
        if "k" in cache:
            k_all, v_all = np.concatenate([cache["k"], k]), np.concatenate([cache["v"], v])
        else:
            k_all, v_all = k, v
        cache["k"], cache["v"] = k_all, v_all
        S = k_all.shape[0]
        kpos = np.arange(S)
        grp = nh // nkv
        o = np.empty((T, nh, hd))
        for hh in range(nh):
            sc = (q[:, hh, :] @ k_all[:, hh // grp, :].T) / np.sqrt(hd)
            sc = np.where(kpos[None, :] <= pos[:, None], sc, -np.inf)
            sc = np.exp(sc - sc.max(axis=1, keepdims=True))
            o[:, hh, :] = (sc / sc.sum(axis=1, keepdims=True)) @ v_all[:, hh // grp, :]
        o = (o * sigmoid(gate)).reshape(T, nh * hd)
        return x + linear(g, o, f"{b}.attn_output.weight")

    def gdn(self, b, x, cache):
        g, nk, nv, dk = self.g, self.nk, self.nv, self.dk
        T = x.shape[0]
        h = rms(x, g.full(f"{b}.attn_norm.weight"), self.eps)
        mixed = linear(g, h, f"{b}.attn_qkv.weight")
        z = linear(g, h, f"{b}.attn_gate.weight").reshape(T, nv, dk)
        beta = sigmoid(linear(g, h, f"{b}.ssm_beta.weight"))
        a = linear(g, h, f"{b}.ssm_alpha.weight")
        gg = g.full(f"{b}.ssm_a") * np.logaddexp(0.0, a + g.full(f"{b}.ssm_dt.bias"))  # <= 0
        C = mixed.shape[1]
        w = g.full(f"{b}.ssm_conv1d.weight")  # [C,K]
        st = cache.get("conv", np.zeros((self.K - 1, C)))
        ext = np.concatenate([st, mixed])
        cache["conv"] = ext[-(self.K - 1):]
        conv = sum(w[:, j][None, :] * ext[j:j + T] for j in range(self.K))
        conv = silu(conv)
        q = conv[:, :nk * dk].reshape(T, nk, dk)
        k = conv[:, nk * dk:2 * nk * dk].reshape(T, nk, dk)
        v = conv[:, 2 * nk * dk:].reshape(T, nv, dk)
        q = q / np.sqrt(np.sum(q * q, -1, keepdims=True) + 1e-6) / np.sqrt(dk)
        k = k / np.sqrt(np.sum(k * k, -1, keepdims=True) + 1e-6)
        idx = np.arange(nv) % nk  # tiled: GGUF v head t uses key head t % nk
        q, k = q[:, idx], k[:, idx]
        S = cache.get("S", np.zeros((nv, dk, dk)))
        o = np.empty((T, nv, dk))
        for t in range(T):
            S = S * np.exp(gg[t])[:, None, None]
            mem = np.einsum("hk,hkv->hv", k[t], S)
            d = (v[t] - mem) * beta[t][:, None]
            S = S + np.einsum("hk,hv->hkv", k[t], d)
            o[t] = np.einsum("hk,hkv->hv", q[t], S)
        cache["S"] = S
        nw = g.full(f"{b}.ssm_norm.weight")
        o = o / np.sqrt(np.mean(o * o, -1, keepdims=True) + self.eps) * nw * silu(z)
        return x + linear(g, o.reshape(T, nv * dk), f"{b}.ssm_out.weight")

    def layer(self, l, x, pos):
        b = f"blk.{l}"
        c = self.caches.setdefault(l, {})
        x = self.attn(b, x, pos, c) if self.is_attn(l) else self.gdn(b, x, c)
        return self.mlp(b, x)

    def trunk(self, ids, pos0, norms=None):
        """Feeds ids at positions pos0.. (all layers, layer-major); returns pre-norm residual [T,H]."""
        x = self.embed(ids)
        pos = np.arange(pos0, pos0 + len(ids))
        for l in range(self.L):
            x = self.layer(l, x, pos)
            if norms is not None:
                norms[l].append(np.linalg.norm(x, axis=-1))
        return x

    def final(self, x):
        return rms(x, self.g.full("output_norm.weight"), self.eps)

    def logits(self, hn):
        return linear(self.g, hn, "output.weight").astype(np.float32)

    def mtp(self, ids_next, hidden):
        """ids_next [T] = token i+1 for position i; hidden [T,H]; returns logits [T,V] float32."""
        g, b = self.g, f"blk.{self.L}"
        e = rms(self.embed(ids_next), g.full(f"{b}.nextn.enorm.weight"), self.eps)
        hh = rms(hidden, g.full(f"{b}.nextn.hnorm.weight"), self.eps)
        x = linear(g, np.concatenate([e, hh], axis=1), f"{b}.nextn.eh_proj.weight")
        x = self.attn(b, x, np.arange(len(ids_next)), {})
        x = self.mlp(b, x)
        return self.logits(rms(x, g.full(f"{b}.nextn.shared_head_norm.weight"), self.eps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--ids")
    ap.add_argument("--ids-file")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default=".")
    ap.add_argument("--mtp-hidden", choices=["post", "pre"], default="post")
    a = ap.parse_args()
    ids = [int(t) for t in (a.ids or open(a.ids_file).read()).replace(",", " ").split()]
    os.makedirs(a.out, exist_ok=True)
    m = Model(a.gguf)
    T, N = len(ids), a.n
    t0 = time.time()
    norms = [[] for _ in range(m.L)]
    x = m.trunk(ids, 0, norms)
    pre = [x]
    hn = m.final(x)
    logits = m.logits(hn)
    post = [hn]
    greedy = [int(np.argmax(logits[-1]))]
    dec_logits = []
    seq = list(ids) + [greedy[0]]
    print("prefill %.1fs" % (time.time() - t0), flush=True)
    for s in range(N - 1):
        x = m.trunk([greedy[-1]], T + s, norms)
        hn = m.final(x)
        lg = m.logits(hn)
        dec_logits.append(lg[0])
        pre.append(x)
        post.append(hn)
        greedy.append(int(np.argmax(lg[0])))
        seq.append(greedy[-1])
        print("decode %d/%d %.1fs tok %d" % (s + 1, N - 1, time.time() - t0, greedy[-1]), flush=True)
    res = {"prompt_ids": np.array(ids), "logits": logits, "greedy": np.array(greedy),
           "decode_logits": np.array(dec_logits, dtype=np.float32).reshape(-1, m.V),
           "layer_norms": np.stack([np.concatenate(n) for n in norms])}
    summ = []
    if m.nmtp:
        hid = np.concatenate(post if a.mtp_hidden == "post" else pre)
        nxt = np.array(seq[1:1 + len(hid)])
        res["mtp_logits"] = m.mtp(nxt, hid)  # row i predicts token i+2 of seq
        mt = res["mtp_logits"].argmax(-1)
        full = res["logits"].argmax(-1).tolist() + [int(np.argmax(l)) for l in dec_logits]
        ok = sum(1 for i in range(len(mt)) if i + 2 < len(seq) and int(mt[i]) == seq[i + 2])
        cnt = sum(1 for i in range(len(mt)) if i + 2 < len(seq))
        summ.append("mtp (%s hidden) top1 vs greedy sequence: %d/%d" % (a.mtp_hidden, ok, cnt))
    np.savez(os.path.join(a.out, "ref.npz"), **res)
    lines = ["model %s layers %d hidden %d vocab %d" % (os.path.basename(a.gguf), m.L, m.H, m.V),
             "prompt ids %s" % ids, "greedy ids %s" % greedy,
             "logits[-1] top5 %s" % np.argsort(-logits[-1])[:5].tolist(),
             "logits finite %s max|l| %.3f" % (bool(np.isfinite(logits).all()), float(np.abs(logits).max())),
             "final-layer hidden norm per prompt pos %s" % np.round(res["layer_norms"][-1][:T], 3).tolist()[:16],
             "layer0..L-1 last-pos norm %s" % np.round(res["layer_norms"][:, T - 1], 2).tolist()[:70]] + summ
    lines.append("elapsed %.1fs" % (time.time() - t0))
    open(os.path.join(a.out, "summary.txt"), "w").write("\n".join(lines[:30]) + "\n")
    print("\n".join(lines[:30]))


if __name__ == "__main__":
    main()
