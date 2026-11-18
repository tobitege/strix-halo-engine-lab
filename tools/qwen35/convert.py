#!/usr/bin/env python3
"""HF qwen3_5 (Qwen3.8-27B dense) safetensors -> GGUF "qwen35" (contract: docs/models/qwen3.8-27b/README.md).

Stdlib + numpy only; own safetensors reader and GGUF v3 writer. Streams tensor by tensor.
  convert.py HF_DIR OUT.gguf --pack bf16|q8_0
  convert.py --synthetic-hf OUTDIR            (tiny random HF-layout model, per-head dims of the real one)
Home: specific tool (model arithmetic = the qwen35 family contract).
"""
import argparse, json, os, struct, sys
import numpy as np

ALIGN = 32
GGML_F32, GGML_F16, GGML_Q8_0, GGML_BF16 = 0, 1, 8, 30
U32, I32, F32, BOOL, STR, ARR, U64 = 4, 5, 6, 7, 8, 9, 10


# ----------------------------------------------------------------------------- safetensors reader
class SafeTensors:
    def __init__(self, directory):
        self.index = {}  # name -> (file, dtype, shape, start, end)
        for fn in sorted(os.listdir(directory)):
            if not fn.endswith(".safetensors"):
                continue
            path = os.path.join(directory, fn)
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            base = 8 + n
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                s, e = v["data_offsets"]
                self.index[k] = (path, v["dtype"], tuple(v["shape"]), base + s, base + e)

    def shape(self, name):
        return self.index[name][2]

    def read(self, name, r0=None, r1=None):
        """Returns uint16 (BF16 raw) or float32. Rows r0:r1 of a >=2-D tensor."""
        path, dt, shape, s, e = self.index[name]
        itemsize = {"BF16": 2, "F16": 2, "F32": 4}[dt]
        if r0 is not None:
            rowbytes = int(np.prod(shape[1:], dtype=np.int64)) * itemsize
            s, e = s + r0 * rowbytes, s + r1 * rowbytes
            shape = (r1 - r0,) + shape[1:]
        with open(path, "rb") as f:
            f.seek(s)
            buf = f.read(e - s)
        if dt == "BF16":
            return np.frombuffer(buf, dtype=np.uint16).reshape(shape)
        if dt == "F16":
            return np.frombuffer(buf, dtype=np.float16).astype(np.float32).reshape(shape)
        return np.frombuffer(buf, dtype=np.float32).reshape(shape)


def write_safetensors(path, tensors):
    """tensors: dict name -> uint16 (BF16 raw) or float32 array."""
    hdr, off = {}, 0
    for k, a in tensors.items():
        dt = "BF16" if a.dtype == np.uint16 else "F32"
        n = a.nbytes
        hdr[k] = {"dtype": dt, "shape": list(a.shape), "data_offsets": [off, off + n]}
        off += n
    h = json.dumps(hdr).encode()
    h += b" " * ((8 - len(h) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for a in tensors.values():
            f.write(np.ascontiguousarray(a).tobytes())


# ----------------------------------------------------------------------------- dtype helpers
def as_f32(a):
    if a.dtype == np.uint16:
        return (a.astype(np.uint32) << 16).view(np.float32)
    return a.astype(np.float32, copy=False)


def to_bf16(a):
    if a.dtype == np.uint16:
        return a
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    u = u + np.uint32(0x7FFF) + ((u >> 16) & np.uint32(1))
    return (u >> 16).astype(np.uint16)


def quant_q8_0(x):
    """x float32 [rows, K] (K % 32 == 0) -> bytes; ggml quantize_row_q8_0_ref (round half away from zero, d as f16)."""
    rows, k = x.shape
    b = np.ascontiguousarray(x, dtype=np.float32).reshape(rows, k // 32, 32)
    amax = np.abs(b).max(axis=2)
    d = (amax / np.float32(127)).astype(np.float32)
    with np.errstate(divide="ignore"):
        idv = np.where(d != 0, np.float32(1) / d, np.float32(0)).astype(np.float32)
    v = b * idv[:, :, None]
    q = (np.sign(v) * np.floor(np.abs(v) + np.float32(0.5))).astype(np.int8)
    out = np.empty((rows, k // 32, 34), dtype=np.uint8)
    out[:, :, :2] = d.astype(np.float16).view(np.uint8).reshape(rows, k // 32, 2)
    out[:, :, 2:] = q.view(np.uint8)
    return out.tobytes()


# ----------------------------------------------------------------------------- GGUF writer
def _s(x):
    b = x.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv(key, typ, val):
    out = _s(key) + struct.pack("<I", typ)
    if typ == STR:
        out += _s(val)
    elif typ == ARR:
        et, items = val
        out += struct.pack("<IQ", et, len(items))
        if et == STR:
            out += b"".join(_s(i) for i in items)
        else:
            out += struct.pack("<%d%s" % (len(items), {I32: "i", U32: "I", F32: "f"}[et]), *items)
    else:
        out += struct.pack("<" + {U32: "I", I32: "i", F32: "f", BOOL: "?", U64: "Q"}[typ], val)
    return out


class GGUFWriter:
    def __init__(self, path):
        self.path, self.kvs, self.tensors = path, [], []  # (name, shape_np, ggml_type, nbytes, producer)

    def kv(self, key, typ, val):
        self.kvs.append(_kv(key, typ, val))

    def add(self, name, shape, gtype, nbytes, producer):
        self.tensors.append((name, shape, gtype, nbytes, producer))

    def write(self):
        offs, off = [], 0
        for t in self.tensors:
            offs.append(off)
            off += (t[3] + ALIGN - 1) // ALIGN * ALIGN
        head = b"GGUF" + struct.pack("<IQQ", 3, len(self.tensors), len(self.kvs)) + b"".join(self.kvs)
        for (name, shape, gt, nb, _), o in zip(self.tensors, offs):
            dims = list(reversed(shape))  # ggml ne[0] = fastest axis
            head += _s(name) + struct.pack("<I", len(dims)) + struct.pack("<%dQ" % len(dims), *dims)
            head += struct.pack("<IQ", gt, o)
        head += b"\0" * ((-len(head)) % ALIGN)
        with open(self.path, "wb") as f:
            f.write(head)
            for (name, shape, gt, nb, prod), o in zip(self.tensors, offs):
                n = 0
                for chunk in prod():
                    if gt == GGML_Q8_0:
                        b = quant_q8_0(as_f32(chunk))
                    elif gt == GGML_BF16:
                        b = to_bf16(chunk).tobytes()
                    else:
                        b = as_f32(chunk).astype(np.float32).tobytes()
                    f.write(b)
                    n += len(b)
                if n != nb:
                    raise RuntimeError("%s: wrote %d bytes, planned %d" % (name, n, nb))
                f.write(b"\0" * (-nb % ALIGN))


# ----------------------------------------------------------------------------- conversion
def tiled_perm(vheads, kheads):
    """GGUF v head t holds HF head (t % kheads) * ratio + t // kheads."""
    r = vheads // kheads
    t = np.arange(vheads)
    return (t % kheads) * r + t // kheads


def block_rows(perm, block, base=0):
    return (base + (perm[:, None] * block + np.arange(block)[None, :])).reshape(-1)


def build(hf_dir, out_path, pack):
    cfg = json.load(open(os.path.join(hf_dir, "config.json")))
    tc = cfg.get("text_config", cfg)
    st = SafeTensors(hf_dir)
    L, H = tc["num_hidden_layers"], tc["hidden_size"]
    nk, nv, dk, dv = tc["linear_num_key_heads"], tc["linear_num_value_heads"], tc["linear_key_head_dim"], tc["linear_value_head_dim"]
    assert dk == dv, "engine GDN assumes key dim == value dim"
    interval = tc["full_attention_interval"]
    for i, t in enumerate(tc["layer_types"]):
        assert (t == "full_attention") == ((i + 1) % interval == 0), "layer_types vs full_attention_interval"
    hd = tc["head_dim"]
    rp = tc.get("rope_parameters", {})
    prf = rp.get("partial_rotary_factor", tc.get("partial_rotary_factor", 0.25))
    theta = rp.get("rope_theta", tc.get("rope_theta", 1e7))
    nmtp = tc.get("mtp_num_hidden_layers", 0)
    assert nmtp <= 1
    P = tiled_perm(nv, nk)
    qk = nk * dk

    g = GGUFWriter(out_path)
    g.kv("general.architecture", STR, "qwen35")
    g.kv("general.name", STR, "Qwen3.8-27B")
    g.kv("general.file_type", U32, 7 if pack == "q8_0" else 32)
    p = "qwen35."
    g.kv(p + "block_count", U32, L)
    g.kv(p + "context_length", U32, tc["max_position_embeddings"])
    g.kv(p + "embedding_length", U32, H)
    g.kv(p + "feed_forward_length", U32, tc["intermediate_size"])
    g.kv(p + "attention.head_count", U32, tc["num_attention_heads"])
    g.kv(p + "attention.head_count_kv", U32, tc["num_key_value_heads"])
    g.kv(p + "attention.key_length", U32, hd)
    g.kv(p + "attention.value_length", U32, hd)
    g.kv(p + "attention.layer_norm_rms_epsilon", F32, float(tc["rms_norm_eps"]))
    g.kv(p + "rope.freq_base", F32, float(theta))
    g.kv(p + "rope.dimension_count", U32, int(round(hd * prf)))
    g.kv(p + "ssm.conv_kernel", U32, tc["linear_conv_kernel_dim"])
    g.kv(p + "ssm.state_size", U32, dk)
    g.kv(p + "ssm.group_count", U32, nk)
    g.kv(p + "ssm.time_step_rank", U32, nv)
    g.kv(p + "ssm.inner_size", U32, nv * dv)
    g.kv(p + "full_attention_interval", U32, interval)
    g.kv(p + "nextn_predict_layers", U32, nmtp)
    add_tokenizer(g, hf_dir, tc, tc["vocab_size"])

    def matrix(name, src, rows=None, cols_blocks=None, block=0):
        n_out, n_in = st.shape(src)
        out_rows = n_out if rows is None else len(rows)
        gt = GGML_Q8_0 if pack == "q8_0" else GGML_BF16
        if gt == GGML_Q8_0:
            assert n_in % 32 == 0, name
            nb = out_rows * (n_in // 32) * 34
        else:
            nb = out_rows * n_in * 2

        def prod():
            step = max(1, (128 << 20) // (n_in * 2))
            if rows is None and cols_blocks is None and n_out * n_in * 2 > (256 << 20):
                for r0 in range(0, n_out, step):
                    yield st.read(src, r0, min(n_out, r0 + step))
                return
            a = st.read(src)
            if rows is not None:
                a = a[rows]
            if cols_blocks is not None:
                a = a.reshape(n_out, -1, block)[:, cols_blocks, :].reshape(n_out, -1)
            for r0 in range(0, a.shape[0], step):
                yield a[r0:r0 + step]
        g.add(name, (out_rows, n_in), gt, nb, prod)

    def vector(name, src, fn=None, perm=None):
        n = st.shape(src)

        def prod():
            a = as_f32(st.read(src))
            if perm is not None:
                a = a[perm]
            if fn is not None:
                a = fn(a.astype(np.float64)).astype(np.float32)
            yield a
        g.add(name, tuple(n), GGML_F32, int(np.prod(n)) * 4, prod)

    onep = lambda a: 1.0 + a
    lp = "model.language_model."
    matrix("token_embd.weight", lp + "embed_tokens.weight")
    vector("output_norm.weight", lp + "norm.weight", onep)
    matrix("output.weight", "lm_head.weight")

    def common(b, s):
        vector(f"{b}.attn_norm.weight", s + "input_layernorm.weight", onep)
        vector(f"{b}.post_attention_norm.weight", s + "post_attention_layernorm.weight", onep)
        matrix(f"{b}.ffn_gate.weight", s + "mlp.gate_proj.weight")
        matrix(f"{b}.ffn_up.weight", s + "mlp.up_proj.weight")
        matrix(f"{b}.ffn_down.weight", s + "mlp.down_proj.weight")

    def full_attn(b, s):
        matrix(f"{b}.attn_q.weight", s + "self_attn.q_proj.weight")
        matrix(f"{b}.attn_k.weight", s + "self_attn.k_proj.weight")
        matrix(f"{b}.attn_v.weight", s + "self_attn.v_proj.weight")
        matrix(f"{b}.attn_output.weight", s + "self_attn.o_proj.weight")
        vector(f"{b}.attn_q_norm.weight", s + "self_attn.q_norm.weight", onep)
        vector(f"{b}.attn_k_norm.weight", s + "self_attn.k_norm.weight", onep)

    rows_qkv = np.concatenate([np.arange(2 * qk), block_rows(P, dv, 2 * qk)])
    for l in range(L):
        s, b = f"{lp}layers.{l}.", f"blk.{l}"
        common(b, s)
        if (l + 1) % interval == 0:
            full_attn(b, s)
            continue
        a = s + "linear_attn."
        matrix(f"{b}.attn_qkv.weight", a + "in_proj_qkv.weight", rows=rows_qkv)
        matrix(f"{b}.attn_gate.weight", a + "in_proj_z.weight", rows=block_rows(P, dv))
        matrix(f"{b}.ssm_beta.weight", a + "in_proj_b.weight", rows=P)
        matrix(f"{b}.ssm_alpha.weight", a + "in_proj_a.weight", rows=P)
        cshape = st.shape(a + "conv1d.weight")  # [C,1,K] -> [C,K]; v channels permuted like attn_qkv rows
        K = cshape[-1]

        def conv_prod(a=a, K=K):
            yield as_f32(st.read(a + "conv1d.weight")).reshape(-1, K)[rows_qkv]
        g.add(f"{b}.ssm_conv1d.weight", (cshape[0], K), GGML_F32, cshape[0] * K * 4, conv_prod)
        vector(f"{b}.ssm_dt.bias", a + "dt_bias", perm=P)
        vector(f"{b}.ssm_a", a + "A_log", fn=lambda x: -np.exp(x), perm=P)
        vector(f"{b}.ssm_norm.weight", a + "norm.weight")
        matrix(f"{b}.ssm_out.weight", a + "out_proj.weight", cols_blocks=P, block=dv)

    if nmtp:
        b, s = f"blk.{L}", "mtp.layers.0."
        common(b, s)
        full_attn(b, s)
        vector(f"{b}.nextn.enorm.weight", "mtp.pre_fc_norm_embedding.weight", onep)
        vector(f"{b}.nextn.hnorm.weight", "mtp.pre_fc_norm_hidden.weight", onep)
        matrix(f"{b}.nextn.eh_proj.weight", "mtp.fc.weight")
        vector(f"{b}.nextn.shared_head_norm.weight", "mtp.norm.weight", onep)
    g.write()
    return len(g.tensors)


def add_tokenizer(g, hf_dir, tc, vocab):
    tj_path = os.path.join(hf_dir, "tokenizer.json")
    if not os.path.exists(tj_path):
        print("warning: no tokenizer.json, tokenizer.ggml.* omitted", file=sys.stderr)
        return
    tj = json.load(open(tj_path, encoding="utf-8"))
    tcfg = {}
    cp = os.path.join(hf_dir, "tokenizer_config.json")
    if os.path.exists(cp):
        tcfg = json.load(open(cp, encoding="utf-8"))
    toks = [None] * vocab
    ttype = [1] * vocab
    for t, i in tj["model"]["vocab"].items():
        toks[i] = t
    for at in tj.get("added_tokens", []):
        toks[at["id"]] = at["content"]
        ttype[at["id"]] = 3 if at.get("special") else 4
    for i in range(vocab):
        if toks[i] is None:
            toks[i] = "[PAD%d]" % i
            ttype[i] = 5
    merges = [m if isinstance(m, str) else " ".join(m) for m in tj["model"]["merges"]]
    g.kv("tokenizer.ggml.model", STR, "gpt2")
    g.kv("tokenizer.ggml.pre", STR, "qwen35")
    g.kv("tokenizer.ggml.tokens", ARR, (STR, toks))
    g.kv("tokenizer.ggml.token_type", ARR, (I32, ttype))
    g.kv("tokenizer.ggml.merges", ARR, (STR, merges))
    lookup = {t: i for i, t in enumerate(toks)}

    def tok_id(x):
        if isinstance(x, dict):
            x = x.get("content")
        return lookup.get(x) if x is not None else None
    eos = tok_id(tcfg.get("eos_token"))
    if eos is None:
        eos = tc.get("eos_token_id")
    pad = tok_id(tcfg.get("pad_token"))
    if pad is None:
        pad = tc.get("pad_token_id")
    if eos is not None:
        g.kv("tokenizer.ggml.eos_token_id", U32, int(eos))
    if pad is not None:
        g.kv("tokenizer.ggml.padding_token_id", U32, int(pad))
    if tc.get("bos_token_id") is not None:
        g.kv("tokenizer.ggml.bos_token_id", U32, int(tc["bos_token_id"]))
    g.kv("tokenizer.ggml.add_bos_token", BOOL, False)
    if tcfg.get("chat_template"):
        g.kv("tokenizer.chat_template", STR, tcfg["chat_template"])


# ----------------------------------------------------------------------------- synthetic HF model
def synthetic_hf(out):
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(20261115)
    H, I, V, L = 512, 1024, 1024, 4
    nk, nv, dk, nh, nkv, hd = 2, 6, 128, 6, 1, 256
    layer_types = ["full_attention" if (i + 1) % 4 == 0 else "linear_attention" for i in range(L)]
    tc = {"attention_bias": False, "attn_output_gate": True, "bos_token_id": 1020, "eos_token_id": 1020,
          "full_attention_interval": 4, "head_dim": hd, "hidden_act": "silu", "hidden_size": H, "intermediate_size": I,
          "layer_types": layer_types, "linear_conv_kernel_dim": 4, "linear_key_head_dim": dk, "linear_num_key_heads": nk,
          "linear_num_value_heads": nv, "linear_value_head_dim": dk, "max_position_embeddings": 4096,
          "model_type": "qwen3_5_text", "mtp_num_hidden_layers": 1, "num_attention_heads": nh, "num_hidden_layers": L,
          "num_key_value_heads": nkv, "partial_rotary_factor": 0.25, "rms_norm_eps": 1e-6, "tie_word_embeddings": False,
          "vocab_size": V, "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000, "rope_type": "default"}}
    json.dump({"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "text_config": tc,
               "tie_word_embeddings": False}, open(os.path.join(out, "config.json"), "w"), indent=1)

    def W(o, i, scale=1.0):
        return to_bf16((rng.standard_normal((o, i)) * scale / np.sqrt(i)).astype(np.float32))

    def nw(n, sd=0.1):
        return to_bf16((rng.standard_normal(n) * sd).astype(np.float32))
    T = {}
    lp = "model.language_model."
    T[lp + "embed_tokens.weight"] = to_bf16((rng.standard_normal((V, H)) * 0.5).astype(np.float32))
    T[lp + "norm.weight"] = nw(H)
    T["lm_head.weight"] = W(V, H, 2.0)

    def mlp(s):
        T[s + "input_layernorm.weight"] = nw(H)
        T[s + "post_attention_layernorm.weight"] = nw(H)
        T[s + "mlp.gate_proj.weight"] = W(I, H)
        T[s + "mlp.up_proj.weight"] = W(I, H)
        T[s + "mlp.down_proj.weight"] = W(H, I)

    def attn(s):
        T[s + "self_attn.q_proj.weight"] = W(nh * hd * 2, H)
        T[s + "self_attn.k_proj.weight"] = W(nkv * hd, H)
        T[s + "self_attn.v_proj.weight"] = W(nkv * hd, H)
        T[s + "self_attn.o_proj.weight"] = W(H, nh * hd)
        T[s + "self_attn.q_norm.weight"] = nw(hd)
        T[s + "self_attn.k_norm.weight"] = nw(hd)
    C = 2 * nk * dk + nv * dk
    for l in range(L):
        s = f"{lp}layers.{l}."
        mlp(s)
        if layer_types[l] == "full_attention":
            attn(s)
            continue
        a = s + "linear_attn."
        T[a + "in_proj_qkv.weight"] = W(C, H)
        T[a + "in_proj_z.weight"] = W(nv * dk, H)
        T[a + "in_proj_b.weight"] = W(nv, H)
        T[a + "in_proj_a.weight"] = W(nv, H)
        T[a + "conv1d.weight"] = to_bf16((rng.standard_normal((C, 1, 4)) * 0.5).astype(np.float32))
        T[a + "dt_bias"] = to_bf16((rng.standard_normal(nv) * 0.5 + 1.0).astype(np.float32))
        T[a + "A_log"] = to_bf16(np.log(rng.uniform(1.0, 8.0, nv)).astype(np.float32))
        T[a + "norm.weight"] = to_bf16((1.0 + rng.standard_normal(dk) * 0.1).astype(np.float32))
        T[a + "out_proj.weight"] = W(H, nv * dk)
    s = "mtp.layers.0."
    mlp(s)
    attn(s)
    T["mtp.fc.weight"] = W(H, 2 * H)
    T["mtp.norm.weight"] = nw(H)
    T["mtp.pre_fc_norm_embedding.weight"] = nw(H)
    T["mtp.pre_fc_norm_hidden.weight"] = nw(H)
    T["model.visual.patch_embed.proj.weight"] = W(16, 16)  # must be skipped by the converter
    write_safetensors(os.path.join(out, "model.safetensors"), T)
    # tiny byte-level BPE tokenizer.json
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    chars = [chr(c) for _, c in sorted(zip(bs, cs))]
    vocab = {c: i for i, c in enumerate(chars)}
    merges, lrng = [], np.random.default_rng(7)
    letters = [chr(c) for c in range(97, 123)]
    while len(vocab) < V - 4:
        keys = list(vocab)
        if len(vocab) < 300:
            a, b = letters[int(lrng.integers(0, 26))], letters[int(lrng.integers(0, 26))]
        else:
            a, b = keys[int(lrng.integers(0, len(keys)))], keys[int(lrng.integers(0, 256))]
        if a + b in vocab:
            continue
        merges.append(a + " " + b)
        vocab[a + b] = len(vocab)
    added = [{"id": len(vocab) + i, "content": t, "special": True}
             for i, t in enumerate(["<|im_start|>", "<|im_end|>", "<think>", "<|endoftext|>"])]
    json.dump({"version": "1.0", "model": {"type": "BPE", "vocab": vocab, "merges": merges}, "added_tokens": added},
              open(os.path.join(out, "tokenizer.json"), "w", encoding="utf-8"))
    json.dump({"eos_token": "<|im_end|>", "pad_token": "<|endoftext|>"}, open(os.path.join(out, "tokenizer_config.json"), "w"))
    print("synthetic HF model written to", out, "tensors", len(T))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("hf_dir", nargs="?")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--pack", choices=["bf16", "q8_0"], default="bf16")
    ap.add_argument("--synthetic-hf", metavar="OUTDIR")
    a = ap.parse_args()
    if a.synthetic_hf:
        synthetic_hf(a.synthetic_hf)
        return
    if not a.hf_dir or not a.out:
        ap.error("HF_DIR and OUT.gguf required")
    n = build(a.hf_dir, a.out, a.pack)
    print("wrote %s: %d tensors, %.2f GB" % (a.out, n, os.path.getsize(a.out) / 1e9))


if __name__ == "__main__":
    main()
