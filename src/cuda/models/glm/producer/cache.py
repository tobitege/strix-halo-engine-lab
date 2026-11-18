"""CPU-only full-model compressed cache builder. No device or network imports.

The native decoder consumes the original packed K16/N16 layout directly. Cache
packing only aligns tensors; it never decodes, requantizes or permutes weights.
An atomic manifest publishes complete blobs. A restart validates those blobs
and does not reread or repack the original model.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct

VERSION = dict(format='glm-native-cache-v1', layout='k16-n16-msb-mul1',
               decoder='mul1-8tiles-v2', tile='cutlass-c1-16experts-v1', transform='h128-rn16-v1')
ROOT = 'model.language_model.'
# Decode-capable caches add the root-level vocabulary head (K6) in head.bin.
# The field enters the cache key; headless prefill caches are unchanged.
HEAD = 'lm_head-root-k6-v1'
HEAD_BYTES = {'lm_head.mul1': 4, 'lm_head.trellis': 475791360, 'lm_head.suh': 8192, 'lm_head.svh': 309760}
SIZES = {'F16': 2, 'BF16': 2, 'I16': 2, 'I32': 4, 'F32': 4}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def inventory(model):
    tensors, files = {}, []
    index = model / 'model.safetensors.index.json'
    paths = ([model / p for p in sorted(set(json.loads(index.read_text())['weight_map'].values()))]
             if index.exists() else sorted(model.glob('*.safetensors')))
    # MTP is distributed separately and is absent from the trunk shard index.
    mtp = model / 'mtp.safetensors'
    if mtp.exists() and mtp not in paths:
        paths.append(mtp)
        paths.sort(key=lambda p: p.name)
    if not paths:
        raise ValueError('no safetensors shards')
    for p in paths:
        with p.open('rb') as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError('short shard header')
            n, = struct.unpack('<Q', raw)
            if not 2 <= n <= min(64 << 20, p.stat().st_size - 8):
                raise ValueError('header length')
            header = json.loads(f.read(n))
        ranges = []
        for name, d in header.items():
            if name == '__metadata__':
                continue
            lo, hi = d['data_offsets']
            dtype, shape = d['dtype'], d['shape']
            if (dtype not in SIZES or any(type(x) is not int or x <= 0 for x in shape)
                    or not 0 <= lo <= hi <= p.stat().st_size - 8 - n
                    or hi-lo != math.prod(shape)*SIZES[dtype] or name in tensors):
                raise ValueError('invalid tensor: ' + name)
            tensors[name] = dict(path=p, offset=8+n+lo, bytes=hi-lo, dtype=dtype, shape=shape)
            ranges.append((lo, hi))
        ranges.sort()
        if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
            raise ValueError('overlapping tensors')
        files.append(p)
    return tensors, files


def key_contract(model_sha256, head=False):
    contract = dict(VERSION, model_sha256=model_sha256)
    if head:
        contract['head'] = HEAD
    return contract


def blob_of(name):
    if name.startswith('lm_head.'):
        return 'head.bin'
    match = re.search(r'\.layers\.(\d+)\.mlp\.experts\.', name)
    return f'experts-{int(match[1]):02}.bin' if match else 'base.bin'


def required_schema(head=False):
    out = {}
    def add(name, dtype, shape):
        out[name if name.startswith('lm_head.') else ROOT+name] = (dtype, shape)
    def matrix(name, i, o):
        add(name+'.mul1', 'I32', [])
        # Rate is validated separately; shape is [i/16,o/16,16*rate].
        add(name+'.trellis', 'I16', [i//16, o//16, None])
        add(name+'.suh', 'F16', [i])
        add(name+'.svh', 'F16', [o])
    add('embed_tokens.weight', 'BF16', [154880, 4096])
    add('norm.weight', 'BF16', [4096])
    for layer in range(46):
        p = f'layers.{layer}.'
        add(p+'input_layernorm.weight', 'BF16', [4096])
        add(p+'post_attention_layernorm.weight', 'BF16', [4096])
        if layer < 45:
            for site in ('hc_attn_', 'hc_ffn_'):
                for role, shape in [('fn', [24, 16384]), ('base', [24]), ('scale', [3])]:
                    add(p+site+role, 'F32', shape)
        a = p+'self_attn.'
        if layer < 45 and layer % 4 != 3:
            matrix(a+'qkv_proj', 4096, 24576)
            matrix(a+'o_proj', 8192, 4096)
            for role, shape in [('b_proj.weight', [64, 4096]), ('f_a_proj.weight', [128, 4096]),
                                ('g_a_proj.weight', [128, 4096]), ('f_b_proj.weight', [8192, 128]),
                                ('g_b_proj.weight', [8192, 128])]:
                add(a+role, 'F16', shape)
            add(a+'conv1d.weight', 'BF16', [24576, 1, 4])
            add(a+'o_norm.weight', 'BF16', [128])
            add(a+'A_log', 'F32', [64])
            add(a+'dt_bias', 'F32', [8192])
        else:
            for role, i, o in [('q_a_proj', 4096, 1536), ('q_b_proj', 1536, 16384),
                               ('kv_a_proj_with_mqa', 4096, 512), ('o_proj', 16384, 4096),
                               ('indexer.wq_b', 1536, 4096)]:
                matrix(a+role, i, o)
            add(a+'q_a_layernorm.weight', 'BF16', [1536])
            add(a+'kv_a_layernorm.weight', 'BF16', [512])
            for role, shape in [('kv_b_proj.weight', [32768, 512]), ('indexer.wk.weight', [128, 4096]),
                                ('indexer.index_kpool_compress_gate', [128, 4096]),
                                ('indexer.weights_proj.weight', [32, 4096]),
                                ('indexer.k_norm.weight', [128]), ('indexer.k_norm.bias', [128])]:
                add(a+role, 'F16', shape)
            add(a+'indexer.index_kpool_compress_ape', 'F32', [4, 128])
        m = p+'mlp.'
        if layer < 3:
            bases, inter = [m], 12288
        else:
            bases, inter = [m+f'experts.{e}.' for e in range(288)]+[m+'shared_experts.'], 2048
            add(m+'gate.weight', 'F16', [288, 4096])
            add(m+'gate.e_score_correction_bias', 'F32', [288])
        for b in bases:
            for role, i, o in [('gate_proj', 4096, inter), ('up_proj', 4096, inter), ('down_proj', inter, 4096)]:
                matrix(b+role, i, o)
        if layer == 45:
            matrix(p+'eh_proj', 8192, 4096)
            for role in ('enorm', 'hnorm', 'shared_head.norm'):
                add(p+role+'.weight', 'BF16', [4096])
    if head:
        matrix('lm_head', 4096, 154880)
        out['lm_head.trellis'] = ('I16', [256, 9680, 96])  # K6 only
    return out


def validate_schema(tensors, head=False):
    schema = required_schema(head)
    missing = sorted(set(schema)-set(tensors))
    if missing:
        raise ValueError(f'incomplete full model: {len(missing)} missing required tensors; first: {missing[0]}')
    for name, (dtype, shape) in schema.items():
        t = tensors[name]
        if t['dtype'] != dtype or len(t['shape']) != len(shape) or any(a != b for a, b in zip(t['shape'], shape) if b is not None):
            raise ValueError('dtype/shape: '+name)
        if name.endswith('.trellis') and t['shape'][-1] not in (48, 64, 80, 96):
            raise ValueError('rate: '+name)
        with t['path'].open('rb') as f:
            f.seek(t['offset'])
            if name.endswith('.mul1') and f.read(4) != struct.pack('<I', 0x83dcd12d):
                raise ValueError('mul1 marker: '+name)
        if name.endswith('.trellis'):
            base = name[:-8]
            if any(base+'.'+bad in tensors for bad in ('bias', 'mcg', 'su', 'sv', 'out_scales', '3inst')):
                raise ValueError('unsupported transform: '+name)
    if head and any(tensors[n]['bytes'] != b for n, b in HEAD_BYTES.items()):
        raise ValueError('lm_head K6 byte contract')
    return schema


def verify(directory, require_head=False):
    manifest = json.loads((directory/'manifest.json').read_text())
    if any(manifest.get(k) != v for k, v in VERSION.items()):
        raise ValueError('cache version')
    head = 'head' in manifest
    if head and manifest['head'] != HEAD:
        raise ValueError('cache head contract')
    if require_head and not head:
        raise ValueError('decode requires a head-capable cache (lm_head); this cache is prefill-only')
    key = hashlib.sha256(canonical(key_contract(manifest['model_sha256'], head))).hexdigest()
    if manifest['cache_key'] != key:
        raise ValueError('cache key')
    if hashlib.sha256(canonical(manifest['sources'])).hexdigest() != manifest['model_sha256']:
        raise ValueError('source manifest identity')
    for name, b in manifest['blobs'].items():
        if Path(name).name != name or (directory/name).stat().st_size != b['bytes'] or sha(directory/name) != b['sha256']:
            raise ValueError('cache blob integrity: '+name)
    schema = required_schema(head)
    if set(manifest['tensors']) != set(schema):
        raise ValueError('cache full-model tensor inventory')
    extents = {}
    for name, t in manifest['tensors'].items():
        dtype, shape = schema[name]
        if (t['dtype'] != dtype or len(t['shape']) != len(shape)
                or any(a != b for a, b in zip(t['shape'], shape) if b is not None)
                or t['offset'] < 0 or t['offset'] % 256
                or t['bytes'] != math.prod(t['shape'])*SIZES[dtype]
                or t['blob'] not in manifest['blobs']
                or t['offset']+t['bytes'] > manifest['blobs'][t['blob']]['bytes']):
            raise ValueError('cache tensor metadata: '+name)
        if name.endswith('.trellis') and t['shape'][-1] not in (48, 64, 80, 96):
            raise ValueError('cache tensor rate')
        if t['blob'] != blob_of(name):
            raise ValueError('cache tensor blob placement: '+name)
        extents.setdefault(t['blob'], []).append((t['offset'], t['offset']+t['bytes']))
    for ranges in extents.values():
        ranges.sort()
        if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
            raise ValueError('cache tensor overlap')
    return manifest


def validate_config(config):
    cfg = config.get('text_config', config)
    expected = ['linear_attention' if i % 4 != 3 else 'deepseek_sparse_attention' for i in range(45)]
    linear = cfg.get('linear_attn_config', {})
    geometry = dict(hidden_size=4096, vocab_size=154880, num_hidden_layers=45, hc_mult=4,
                    q_lora_rank=1536, kv_lora_rank=512, num_attention_heads=64,
                    qk_nope_head_dim=256, qk_rope_head_dim=0, v_head_dim=256,
                    index_n_heads=32, index_head_dim=128, index_kpool=4, index_topk=2048,
                    n_routed_experts=288, moe_intermediate_size=2048, intermediate_size=12288,
                    num_nextn_predict_layers=1)
    if (any(cfg.get(k) != v for k, v in geometry.items()) or cfg.get('layer_types') != expected
            or cfg.get('mlp_layer_types') != ['dense']*3+['sparse']*42
            or linear.get('num_heads') != 64 or linear.get('head_dim') != 128
            or linear.get('gate_lower_bound') != -5.0 or linear.get('short_conv_kernel_size') != 4
            or sorted(linear.get('kda_layers', [])) != [i for i in range(45) if i % 4 != 3]
            or any(cfg.get(k, v) != v for k, v in dict(routed_scaling_factor=2.5, num_experts_per_tok=8,
                                                      rms_norm_eps=1e-5).items())):
        raise ValueError('unsupported full-model configuration')


def packed_chunks(tensors, names, validate_finite=True):
    """Yield (name, virtual offset, data) with <=8 MiB transient storage.

    Cache publication uses finite validation. Byte equivalence tests may skip
    that validation; it never changes the emitted bytes.
    """
    offset = 0
    for name in names:
        t = tensors[name]
        padding = -offset % 256
        if padding:
            yield None, offset, b'\0' * padding
            offset += padding
        with t['path'].open('rb') as src:
            src.seek(t['offset'])
            left = t['bytes']
            while left:
                b = src.read(min(left, 8 << 20))
                if not b:
                    raise ValueError('source truncated')
                if validate_finite:
                    if t['dtype'] in ('F16', 'BF16'):
                        mask = 0x7c00 if t['dtype'] == 'F16' else 0x7f80
                        if any(v & mask == mask for v, in struct.iter_unpack('<H', b)):
                            raise ValueError('nonfinite tensor: ' + name)
                    elif t['dtype'] == 'F32' and any(not math.isfinite(v) for v, in struct.iter_unpack('<f', b)):
                        raise ValueError('nonfinite tensor: ' + name)
                yield name, offset, b
                offset += len(b)
                left -= len(b)


def build(model, directory, head=False):
    if (directory/'manifest.json').exists():
        manifest = verify(directory, require_head=head)
        if head != ('head' in manifest):
            raise ValueError('existing cache has a different head inventory; use another directory')
        return manifest
    tensors, files = inventory(model)
    schema = validate_schema(tensors, head)  # Fail before copying any incomplete model.
    config = json.loads((model/'config.json').read_text())
    validate_config(config)
    sources = [{'name': p.name, 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in files]
    source_manifest = dict(config=config, files=sources)
    model_sha = hashlib.sha256(canonical(source_manifest)).hexdigest()
    manifest = dict(key_contract(model_sha, head), sources=source_manifest,
                    cache_key=hashlib.sha256(canonical(key_contract(model_sha, head))).hexdigest(),
                    blobs={}, tensors={})
    groups = {}
    for name in sorted(schema):
        groups.setdefault(blob_of(name), []).append(name)
    directory.mkdir(parents=True, exist_ok=True)
    for blob, names in groups.items():
        tmp = directory/(blob+'.partial')
        h = hashlib.sha256()
        with tmp.open('wb') as dst:
            def write(b):
                dst.write(b)
                h.update(b)
            for name, offset, data in packed_chunks(tensors, names):
                if name is not None and name not in manifest['tensors']:
                    t = tensors[name]
                    manifest['tensors'][name] = dict(blob=blob, offset=offset, bytes=t['bytes'],
                                                     dtype=t['dtype'], shape=t['shape'])
                write(data)
            dst.flush()
            os.fsync(dst.fileno())
            manifest['blobs'][blob] = dict(bytes=dst.tell(), sha256=h.hexdigest())
        tmp.replace(directory/blob)
    # Verify source stability across initial packing; do not publish mixed revisions.
    if any(p.stat().st_size != s['bytes'] or sha(p) != s['sha256'] for p, s in zip(files, sources)):
        raise ValueError('source changed while building cache')
    tmp = directory/'manifest.json.partial'
    tmp.write_bytes(canonical(manifest))
    tmp.replace(directory/'manifest.json')
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--verify', action='store_true')
    p.add_argument('--with-head', action='store_true', help='decode-capable cache: add root lm_head (K6) in head.bin')
    a = p.parse_args()
    if not a.verify and a.model is None:
        p.error('--model required to build')
    m = verify(a.cache, require_head=a.with_head) if a.verify else build(a.model, a.cache, a.with_head)
    print(json.dumps({k: m[k] for k in ('model_sha256', 'cache_key')}))
