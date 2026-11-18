"""CPU-only EXL3 -> bounded native benchmark archive. No runtime dependencies.

Reads modern mul1 trellis groups, including groups split across shards. Does not
requantize FP8/BF16 models or invent trellis weights. Assets/provenance stay out
of Git; pass an output path in a private build directory.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def export(model, out, layer, kind):
    tensors = {}
    for path in sorted(model.glob('*.safetensors')):
        with path.open('rb') as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError('short header')
            n, = struct.unpack('<Q', raw)
            if not 2 <= n <= min(64*1024**2, path.stat().st_size-8):
                raise ValueError('invalid header extent')
            header = json.loads(f.read(n))
        for key, d in header.items():
            if key == '__metadata__':
                continue
            if key in tensors:
                raise ValueError('duplicate tensor: '+key)
            lo, hi = d['data_offsets']
            if not 0 <= lo <= hi <= path.stat().st_size-8-n:
                raise ValueError('invalid tensor extent')
            tensors[key] = (path, 8+n+lo, hi-lo, d)
    if kind in ('kda', 'attention', 'mtp'):
        return export_attention(model, out, layer, kind, tensors)
    root = f'model.language_model.layers.{layer}.mlp.'
    if kind == 'experts':
        names = [root+f'experts.{e}.' for e in range(288)]
    elif kind == 'shared':
        names = [root+'shared_experts.']
    else:
        names = [root]
    def get(key, dtype, shape):
        if key not in tensors:
            raise ValueError('missing EXL3 tensor '+key+'; an FP8 pack is not a trellis fixture')
        p, off, size, d = tensors[key]
        if d['dtype'] != dtype or d['shape'] != shape:
            raise ValueError('dtype/shape mismatch: '+key)
        count = 1
        for x in shape:
            count *= x
        if size != count * {'I16': 2, 'F16': 2, 'I32': 4}[dtype]:
            raise ValueError('byte count: '+key)
        return p, off, size
    hidden, inter = 4096, (12288 if kind == 'dense' else 2048)
    plan = []
    for base in names:
        for role, i, o in [('gate',hidden,inter), ('up',hidden,inter), ('down',inter,hidden)]:
            prefix = base+role+'_proj'
            for bad in ['bias','mcg','su','sv','out_scales','3inst']:
                if prefix+'.'+bad in tensors:
                    raise ValueError('unsupported convention: '+prefix+'.'+bad)
            marker = get(prefix+'.mul1','I32',[])
            with marker[0].open('rb') as f:
                f.seek(marker[1])
                if f.read(4) != struct.pack('<I',0x83dcd12d):
                    raise ValueError('mul1 marker')
            desc = tensors.get(prefix+'.trellis')
            if not desc:
                raise ValueError('missing EXL3 tensor '+prefix+'.trellis')
            sh = desc[3]['shape']
            bits = sh[2]//16 if len(sh)==3 else 0
            if bits not in (3,4,5,6):
                raise ValueError('unsupported rate')
            parts = [get(prefix+'.trellis','I16',[i//16,o//16,16*bits]),
                     get(prefix+'.suh','F16',[i]),get(prefix+'.svh','F16',[o])]
            # Check finite scale values without changing even signed zero bits.
            for p, off, size in parts[1:]:
                with p.open('rb') as f:
                    f.seek(off)
                    if any((v[0]&0x7c00)==0x7c00 for v in struct.iter_unpack('<H',f.read(size))):
                        raise ValueError('nonfinite scale')
            plan.append((prefix,bits,parts))
    # All groups are validated before creating the archive.
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp = out.with_suffix(out.suffix+'.partial')
    digest = hashlib.sha256()
    records = []
    with tmp.open('wb') as dst:
        def write(b):
            digest.update(b)
            dst.write(b)
        write(b'GLMNAT01'+struct.pack('<3I',len(names),hidden,inter))
        for prefix,bits,parts in plan:
            write(struct.pack('<I',bits))
            for p,off,size in parts:
                sha=hashlib.sha256()
                with p.open('rb') as f:
                    f.seek(off)
                    left=size
                    while left:
                        b=f.read(min(left,4*1024**2))
                        if not b:
                            raise ValueError('short tensor')
                        left-=len(b); sha.update(b); write(b)
                records.append(dict(prefix=prefix,source=str(p.resolve()),offset=off,bytes=size,sha256=sha.hexdigest()))
    tmp.replace(out)
    out.with_suffix(out.suffix+'.json').write_text(json.dumps(dict(format='GLMNAT01',layer=layer,kind=kind,
        sha256=digest.hexdigest(),tensors=records),indent=2),encoding='utf-8')
    print(json.dumps(dict(archive=str(out),sha256=digest.hexdigest(),matrices=len(plan))))


# Self-describing attention archive, consumed by standalone KDA/DSA benches.
# MTP adds layer-45 fusion/norm/router tensors; expert/shared archives are
# exported separately with --layer 45 --kind experts/shared. Original dtype/bytes and per-tensor hashes survive.
def export_attention(model, out, layer, kind, tensors):
    config_path = model/'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    cfg = config.get('text_config', config)
    types = cfg['layer_types']
    mtp = kind == 'mtp'
    if mtp and layer != 45:
        raise ValueError('MTP must be layer 45')
    if not (mtp or 0 <= layer < len(types)):
        raise ValueError('layer outside config')
    kda = not mtp and types[layer] == 'linear_attention'
    if kind == 'kda' and not kda:
        raise ValueError('requested KDA layer is DSA in config')
    root = f'model.language_model.layers.{layer}.self_attn.'
    h = cfg['hidden_size']
    heads = cfg['linear_attn_config']['num_heads']
    c = heads*128
    if (h != 4096 or heads != 64 or cfg['linear_attn_config']['head_dim'] != 128
            or cfg['linear_attn_config'].get('gate_lower_bound') != -5.0
            or cfg['linear_attn_config'].get('short_conv_kernel_size') != 4):
        raise ValueError('unsupported attention geometry')
    records, plan = [], []
    sizes = {'I16':2, 'I32':4, 'F16':2, 'BF16':2, 'F32':4}
    def add(name, dtype=None, shape=None):
        key = root+name
        if key not in tensors:
            raise ValueError('missing attention tensor '+key)
        p,off,size,d = tensors[key]
        if dtype is not None and (d['dtype'] != dtype or d['shape'] != shape):
            raise ValueError('dtype/shape mismatch: '+key)
        count = 1
        for x in d['shape']:
            count *= x
        if d['dtype'] not in sizes or size != count*sizes[d['dtype']]:
            raise ValueError('byte count/dtype: '+key)
        plan.append((key,p,off,size,d))
    def matrix(role, i, o):
        for bad in ['bias','mcg','su','sv','out_scales','3inst']:
            if root+role+'.'+bad in tensors:
                raise ValueError('unsupported convention: '+root+role+'.'+bad)
        add(role+'.mul1','I32',[])
        p,off,size,d=tensors[root+role+'.mul1']
        with p.open('rb') as f:
            f.seek(off)
            if f.read(4) != struct.pack('<I',0x83dcd12d):
                raise ValueError('mul1 marker')
        desc=tensors.get(root+role+'.trellis')
        sh=desc[3]['shape'] if desc else []
        bits=sh[2]//16 if len(sh)==3 else 0
        if bits not in (3,4,5,6):
            raise ValueError('unsupported attention rate')
        add(role+'.trellis','I16',[i//16,o//16,16*bits])
        add(role+'.suh','F16',[i]);add(role+'.svh','F16',[o])
    if kda:
        matrix('qkv_proj',h,3*c);matrix('o_proj',c,h)
        plain = [
            ('b_proj.weight','F16',[heads,h]),('f_a_proj.weight','F16',[128,h]),
            ('f_b_proj.weight','F16',[c,128]),('g_a_proj.weight','F16',[128,h]),
            ('g_b_proj.weight','F16',[c,128]),('conv1d.weight','BF16',[3*c,1,4]),
            ('o_norm.weight','BF16',[128]),('A_log','F32',[heads]),('dt_bias','F32',[c])]
    else:
        # Refuse partial/unsupported DSA layers; never silently export a subset.
        if (not mtp and types[layer] != 'deepseek_sparse_attention') or any(cfg[k] != v for k,v in
                [('q_lora_rank',1536),('kv_lora_rank',512),('num_attention_heads',64),
                 ('qk_nope_head_dim',256),('qk_rope_head_dim',0),('v_head_dim',256),
                 ('index_n_heads',32),('index_head_dim',128),('index_kpool',4),('index_topk',2048)]):
            raise ValueError('unsupported DSA geometry')
        for role,i,o in [('q_a_proj',h,1536),('q_b_proj',1536,16384),
                         ('kv_a_proj_with_mqa',h,512),('o_proj',16384,h),
                         ('indexer.wq_b',1536,4096)]:
            matrix(role,i,o)
        plain = [('q_a_layernorm.weight','BF16',[1536]),('kv_a_layernorm.weight','BF16',[512]),
                 ('kv_b_proj.weight','F16',[32768,512]),('indexer.wk.weight','F16',[128,h]),
                 ('indexer.weights_proj.weight','F16',[32,h]),('indexer.k_norm.weight','F16',[128]),
                 ('indexer.k_norm.bias','F16',[128]),('indexer.index_kpool_compress_gate','F16',[128,h]),
                 ('indexer.index_kpool_compress_ape','F32',[4,128])]
    for name,dtype,shape in plain:
        add(name,dtype,shape)
    if mtp:
        root = f'model.language_model.layers.{layer}.'
        matrix('eh_proj',8192,4096)
        for name in ('enorm.weight','hnorm.weight','input_layernorm.weight',
                     'post_attention_layernorm.weight','shared_head.norm.weight'):
            add(name,'BF16',[4096])
        add('mlp.gate.weight','F16',[288,4096])
        add('mlp.gate.e_score_correction_bias','F32',[288])
    # Validate all floating tensors before creating an output, retaining bits.
    for key,p,off,size,d in plan:
        if d['dtype'] not in ('F16','BF16','F32'):
            continue
        with p.open('rb') as f:
            f.seek(off);data=f.read(size)
        if len(data) != size:
            raise ValueError('short tensor')
        if d['dtype']=='F32':
            bad=any(not math.isfinite(x[0]) for x in struct.iter_unpack('<f',data))
        else:
            mask=0x7c00 if d['dtype']=='F16' else 0x7f80
            bad=any(x[0]&mask==mask for x in struct.iter_unpack('<H',data))
        if bad:
            raise ValueError('nonfinite attention tensor: '+key)
    out.parent.mkdir(parents=True,exist_ok=True)
    tmp=out.with_suffix(out.suffix+'.partial')
    digest=hashlib.sha256()
    with tmp.open('wb') as dst:
        def write(b):
            digest.update(b);dst.write(b)
        write(b'GLMATT01'+struct.pack('<4I',layer,h,heads,len(plan)))
        for key,p,off,size,d in plan:
            name=key[len(f'model.language_model.layers.{layer}.'):].encode('ascii') if mtp else key[len(root):].encode('ascii')
            dtype=d['dtype'].encode('ascii')
            write(struct.pack('<I',len(name))+name+dtype.ljust(4,b'\0'))
            write(struct.pack('<I',len(d['shape'])))
            for extent in d['shape']:
                write(struct.pack('<I',extent))
            write(struct.pack('<Q',size))
            sha=hashlib.sha256()
            with p.open('rb') as f:
                f.seek(off);left=size
                while left:
                    b=f.read(min(left,4*1024**2))
                    if not b: raise ValueError('short tensor')
                    left-=len(b);sha.update(b);write(b)
            records.append(dict(name=key,source=str(p.resolve()),offset=off,bytes=size,
                                dtype=d['dtype'],shape=d['shape'],sha256=sha.hexdigest()))
    tmp.replace(out)
    metadata=dict(format='GLMATT01',layer=layer,layer_type='mtp' if mtp else types[layer],
                  config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                  sha256=digest.hexdigest(),tensors=records)
    out.with_suffix(out.suffix+'.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(dict(archive=str(out),sha256=digest.hexdigest(),layer_type='mtp' if mtp else types[layer],tensors=len(plan))))


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--layer',type=int,default=3)
    ap.add_argument('--kind',choices=['experts','dense','shared','kda','attention','mtp'],default='experts')
    a=ap.parse_args()
    try:
        export(a.model,a.out,a.layer,a.kind)
    except (ValueError,KeyError,OSError) as e:
        ap.exit(2,str(e)+'\n')
