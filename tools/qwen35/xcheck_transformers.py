#!/usr/bin/env python3
"""One-shot semantic cross-check: remap a (synthetic) qwen3_5 HF dir into transformers Qwen3NextForCausalLM layout
(interleaved in_proj_qkvz / in_proj_ba per key-head group, all layers dense) and compare its logits with ref.npz made by
ref_forward.py on the converted GGUF.   xcheck_transformers.py HF_DIR REF.npz   (needs torch + transformers; tool only)"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import types, importlib.machinery
for _n in ("sklearn", "sklearn.metrics"):  # env sklearn binary is broken; transformers only imports roc_curve
    sys.modules[_n] = types.ModuleType(_n); sys.modules[_n].__spec__ = importlib.machinery.ModuleSpec(_n, None)
sys.modules["sklearn.metrics"].roc_curve = None
from convert import SafeTensors, as_f32
from transformers import Qwen3NextConfig, Qwen3NextForCausalLM

hf, refp = sys.argv[1], sys.argv[2]
tc = json.load(open(os.path.join(hf, "config.json")))["text_config"]
st = SafeTensors(hf)
G = lambda n: torch.from_numpy(as_f32(st.read(n)).copy())
L, nk, nv, dk = tc["num_hidden_layers"], tc["linear_num_key_heads"], tc["linear_num_value_heads"], tc["linear_key_head_dim"]
r = nv // nk
cfg = Qwen3NextConfig(vocab_size=tc["vocab_size"], hidden_size=tc["hidden_size"], intermediate_size=tc["intermediate_size"],
    num_hidden_layers=L, num_attention_heads=tc["num_attention_heads"], num_key_value_heads=tc["num_key_value_heads"],
    head_dim=tc["head_dim"], linear_conv_kernel_dim=4, linear_key_head_dim=dk, linear_value_head_dim=dk,
    linear_num_key_heads=nk, linear_num_value_heads=nv, mlp_only_layers=list(range(L)), layer_types=tc["layer_types"],
    rope_parameters={"rope_type": "default", "rope_theta": 1e7, "partial_rotary_factor": 0.25},
    max_position_embeddings=4096, num_experts=0, tie_word_embeddings=False, rms_norm_eps=1e-6)
m = Qwen3NextForCausalLM(cfg).eval()
sd = {"model.embed_tokens.weight": G("model.language_model.embed_tokens.weight"),
      "model.norm.weight": G("model.language_model.norm.weight"), "lm_head.weight": G("lm_head.weight")}
for l in range(L):
    s, d = f"model.language_model.layers.{l}.", f"model.layers.{l}."
    for a, b in [("input_layernorm", "input_layernorm"), ("post_attention_layernorm", "post_attention_layernorm")]:
        sd[d + b + ".weight"] = G(s + a + ".weight")
    for p in ("gate_proj", "up_proj", "down_proj"):
        sd[d + "mlp." + p + ".weight"] = G(s + "mlp." + p + ".weight")
    if tc["layer_types"][l] == "full_attention":
        for k in ("q_proj", "k_proj", "v_proj", "o_proj"):
            sd[d + "self_attn." + k + ".weight"] = G(s + "self_attn." + k + ".weight")
        for k in ("q_norm", "k_norm"):
            sd[d + "self_attn." + k + ".weight"] = G(s + "self_attn." + k + ".weight")
        continue
    a, o = s + "linear_attn.", d + "linear_attn."
    qkv, z = G(a + "in_proj_qkv.weight"), G(a + "in_proj_z.weight")
    H = qkv.shape[1]
    q, k, v = qkv[:nk * dk].view(nk, dk, H), qkv[nk * dk:2 * nk * dk].view(nk, dk, H), qkv[2 * nk * dk:].view(nk, r * dk, H)
    sd[o + "in_proj_qkvz.weight"] = torch.cat([q, k, v, z.view(nk, r * dk, H)], 1).reshape(-1, H)
    bb, aa = G(a + "in_proj_b.weight").view(nk, r, H), G(a + "in_proj_a.weight").view(nk, r, H)
    sd[o + "in_proj_ba.weight"] = torch.cat([bb, aa], 1).reshape(-1, H)
    sd[o + "conv1d.weight"] = G(a + "conv1d.weight")
    for k in ("dt_bias", "A_log"):
        sd[o + k] = G(a + k)
    sd[o + "norm.weight"] = G(a + "norm.weight")
    sd[o + "out_proj.weight"] = G(a + "out_proj.weight")
miss = m.load_state_dict(sd, strict=False)
print("missing", miss.missing_keys[:5], "unexpected", miss.unexpected_keys[:5])
ref = np.load(refp)
ids = torch.tensor(ref["prompt_ids"])[None]
with torch.no_grad():
    lg = m(input_ids=ids).logits[0].numpy()
d = np.abs(lg - ref["logits"])
print("max|dlogit| %.3e  mean %.3e  max|logit| %.2f" % (d.max(), d.mean(), np.abs(lg).max()))
print("top1 agreement %d/%d" % ((lg.argmax(-1) == ref["logits"].argmax(-1)).sum(), lg.shape[0]))
