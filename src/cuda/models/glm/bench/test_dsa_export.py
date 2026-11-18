"""CPU-only DSA/MTP archive integrity tests; all fixtures stay in BuildDir."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from export_model import export

class DsaExportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=BUILD)
        self.model=Path(self.temp.name);self.out=self.model/'out.native'
    def tearDown(self):
        self.temp.cleanup()
    def fixture(self,mtp=False,bad=None):
        layer=45 if mtp else 7
        cfg=dict(hidden_size=4096,layer_types=['linear_attention']*7+['deepseek_sparse_attention'],
                 linear_attn_config=dict(num_heads=64,head_dim=128,gate_lower_bound=-5.0,short_conv_kernel_size=4),
                 q_lora_rank=1536,kv_lora_rank=512,num_attention_heads=64,qk_nope_head_dim=256,qk_rope_head_dim=0,
                 v_head_dim=256,index_n_heads=32,index_head_dim=128,index_kpool=4,index_topk=2048)
        (self.model/'config.json').write_text(json.dumps({'text_config':cfg}))
        root=f'model.language_model.layers.{layer}.';tensors=[]
        def add(name,dt,shape,data=None):
            if bad=='missing' and name=='self_attn.indexer.k_norm.bias':return
            if bad=='dtype' and name=='self_attn.kv_b_proj.weight':dt='BF16'
            if bad=='nonfinite' and name=='self_attn.indexer.index_kpool_compress_ape':data=struct.pack('<f',float('nan'))*512
            tensors.append((root+name,dt,shape,data))
        shapes=[('self_attn.q_a_proj',4096,1536),('self_attn.q_b_proj',1536,16384),
                ('self_attn.kv_a_proj_with_mqa',4096,512),('self_attn.o_proj',16384,4096),('self_attn.indexer.wq_b',1536,4096)]
        if mtp:shapes.append(('eh_proj',8192,4096))
        for role,i,o in shapes:
            add(role+'.mul1','I32',[],struct.pack('<I',0x83dcd12d));add(role+'.trellis','I16',[i//16,o//16,48]);add(role+'.suh','F16',[i]);add(role+'.svh','F16',[o])
        for name,dt,shape in [('q_a_layernorm.weight','BF16',[1536]),('kv_a_layernorm.weight','BF16',[512]),
                 ('kv_b_proj.weight','F16',[32768,512]),('indexer.wk.weight','F16',[128,4096]),
                 ('indexer.weights_proj.weight','F16',[32,4096]),('indexer.k_norm.weight','F16',[128]),
                 ('indexer.k_norm.bias','F16',[128]),('indexer.index_kpool_compress_gate','F16',[128,4096]),
                 ('indexer.index_kpool_compress_ape','F32',[4,128])]:add('self_attn.'+name,dt,shape)
        if mtp:
            for name in ['enorm.weight','hnorm.weight','input_layernorm.weight','post_attention_layernorm.weight','shared_head.norm.weight']:add(name,'BF16',[4096])
            add('mlp.gate.weight','F16',[288,4096]);add('mlp.gate.e_score_correction_bias','F32',[288])
        # Deliberately split q_a trellis from scales across shards.
        for shard in range(2):
            header={};parts=[];offset=0
            for name,dt,shape,data in tensors[shard::2]:
                n=1
                for x in shape:n*=x
                size=n*(4 if dt in ('I32','F32') else 2)
                header[name]=dict(dtype=dt,shape=shape,data_offsets=[offset,offset+size]);parts.append((size,data));offset+=size
            raw=json.dumps(header).encode()
            with (self.model/f'{shard}.safetensors').open('wb') as f:
                f.write(struct.pack('<Q',len(raw))+raw)
                for size,data in parts:
                    if data is None:f.seek(size,1)
                    else:f.write(data)
                f.truncate()
    def verify(self,mtp=False):
        export(self.model,self.out,45 if mtp else 7,'mtp' if mtp else 'attention')
        m=json.loads(Path(str(self.out)+'.json').read_text())
        self.assertEqual(m['layer_type'],'mtp' if mtp else 'deepseek_sparse_attention')
        self.assertEqual(len(m['tensors']),40 if mtp else 29)
        self.assertEqual(m['sha256'],hashlib.sha256(self.out.read_bytes()).hexdigest())
        with self.out.open('rb') as f:
            self.assertEqual(f.read(24),b'GLMATT01'+struct.pack('<4I',45 if mtp else 7,4096,64,40 if mtp else 29))
            for rec in m['tensors']:
                n,=struct.unpack('<I',f.read(4));name=f.read(n).decode();self.assertTrue(rec['name'].endswith('.'+name))
                self.assertEqual(f.read(4).rstrip(b'\0').decode(),rec['dtype'])
                nd,=struct.unpack('<I',f.read(4));self.assertEqual(list(struct.unpack('<'+'I'*nd,f.read(nd*4))),rec['shape'])
                size,=struct.unpack('<Q',f.read(8));data=f.read(size);self.assertEqual(hashlib.sha256(data).hexdigest(),rec['sha256'])
                with Path(rec['source']).open('rb') as src:
                    src.seek(rec['offset']);self.assertEqual(data,src.read(size))
            self.assertEqual(f.read(),b'')
    def test_dsa_roundtrip(self):self.fixture();self.verify()
    def test_mtp_roundtrip(self):self.fixture(mtp=True);self.verify(mtp=True)
    def test_dtype(self):
        self.fixture(bad='dtype')
        with self.assertRaisesRegex(ValueError,'dtype/shape'):export(self.model,self.out,7,'attention')
        self.assertFalse(self.out.exists())
    def test_missing(self):
        self.fixture(bad='missing')
        with self.assertRaisesRegex(ValueError,'missing attention tensor'):export(self.model,self.out,7,'attention')
        self.assertFalse(self.out.exists())
    def test_nonfinite(self):
        self.fixture(bad='nonfinite')
        with self.assertRaisesRegex(ValueError,'nonfinite attention'):export(self.model,self.out,7,'attention')
        self.assertFalse(self.out.exists())
    def test_mtp_layer(self):
        self.fixture()
        with self.assertRaisesRegex(ValueError,'MTP must be layer 45'):export(self.model,self.out,7,'mtp')
        self.assertFalse(self.out.exists())
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--build-dir',type=Path,required=True)
    a,rest=ap.parse_known_args();BUILD=a.build_dir.resolve();BUILD.mkdir(parents=True,exist_ok=True)
    unittest.main(argv=[__file__]+rest)
