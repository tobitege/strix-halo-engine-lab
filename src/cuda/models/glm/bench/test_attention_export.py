"""CPU attention archive tests; fixtures and exports remain in the build dir."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from export_model import export

class AttentionExportTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=BUILD)
        self.model=Path(self.temp.name)
        self.out=self.model/'out.native'
        self.config={'hidden_size':4096,'layer_types':['linear_attention'],
            'linear_attn_config':{'num_heads':64,'head_dim':128,'gate_lower_bound':-5.0,'short_conv_kernel_size':4}}
        self.write_config()
    def write_config(self):
        (self.model/'config.json').write_text(json.dumps({'text_config':self.config}))
    def tearDown(self):
        self.temp.cleanup()
    def fixture(self, bad_shape=False, nonfinite=False):
        root='model.language_model.layers.0.self_attn.'
        h,c=4096,8192
        tensors=[]
        for role,i,o in [('qkv_proj',h,3*c),('o_proj',c,h)]:
            tensors += [(role+'.mul1','I32',[],struct.pack('<I',0x83dcd12d)),
                (role+'.trellis','I16',[i//16,o//16,48],None),
                (role+'.suh','F16',[i],struct.pack('<H',0x8000)*i),
                (role+'.svh','F16',[o],struct.pack('<H',0x3c00)*o)]
        for name,dt,shape in [('b_proj.weight','F16',[64,h]),('f_a_proj.weight','F16',[128,h]),
                ('f_b_proj.weight','F16',[c,128]),('g_a_proj.weight','F16',[128,h]),
                ('g_b_proj.weight','F16',[c,128]),('conv1d.weight','BF16',[3*c,1,4]),
                ('o_norm.weight','BF16',[128]),('A_log','F32',[64]),('dt_bias','F32',[c])]:
            data=None
            if name=='o_norm.weight' and nonfinite:data=struct.pack('<H',0x7f80)*128
            tensors.append((name,dt,([127] if bad_shape and name=='o_norm.weight' else shape),data))
        # Split matrices/scales across two independent shards.
        for shard in range(2):
            header={};offset=0;parts=[]
            for name,dt,shape,data in tensors[shard::2]:
                n=1
                for x in shape:n*=x
                size=n*(4 if dt in ('I32','F32') else 2)
                header[root+name]=dict(dtype=dt,shape=shape,data_offsets=[offset,offset+size])
                offset+=size;parts.append((size,data))
            raw=json.dumps(header).encode()
            with (self.model/f'{shard}.safetensors').open('wb') as f:
                f.write(struct.pack('<Q',len(raw))+raw)
                for size,data in parts:
                    if data is None:f.seek(size,1)
                    else:f.write(data)
                f.truncate()
    def test_roundtrip(self):
        self.fixture();export(self.model,self.out,0,'kda')
        manifest=json.loads(Path(str(self.out)+'.json').read_text())
        self.assertEqual(manifest['sha256'],hashlib.sha256(self.out.read_bytes()).hexdigest())
        with self.out.open('rb') as f:
            self.assertEqual(f.read(24),b'GLMATT01'+struct.pack('<4I',0,4096,64,17))
            for rec in manifest['tensors']:
                n,=struct.unpack('<I',f.read(4));name=f.read(n).decode()
                self.assertTrue(rec['name'].endswith('.'+name))
                self.assertEqual(f.read(4).rstrip(b'\0').decode(),rec['dtype'])
                nd,=struct.unpack('<I',f.read(4));shape=list(struct.unpack('<'+'I'*nd,f.read(4*nd)))
                self.assertEqual(shape,rec['shape'])
                size,=struct.unpack('<Q',f.read(8));data=f.read(size)
                self.assertEqual(size,rec['bytes'])
                self.assertEqual(hashlib.sha256(data).hexdigest(),rec['sha256'])
                with Path(rec['source']).open('rb') as src:
                    src.seek(rec['offset']);self.assertEqual(data,src.read(size))
            self.assertEqual(f.read(),b'')
    def test_wrong_layer_type(self):
        self.config['layer_types']=['deepseek_sparse_attention'];self.write_config()
        with self.assertRaisesRegex(ValueError,'requested KDA layer is DSA'):export(self.model,self.out,0,'kda')
        self.assertFalse(self.out.exists())
    def test_wrong_gate_contract(self):
        self.config['linear_attn_config']['gate_lower_bound']=-3.0;self.write_config()
        with self.assertRaisesRegex(ValueError,'unsupported attention geometry'):export(self.model,self.out,0,'kda')
        self.assertFalse(self.out.exists())
    def test_wrong_shape(self):
        self.fixture(bad_shape=True)
        with self.assertRaisesRegex(ValueError,'dtype/shape'):export(self.model,self.out,0,'kda')
        self.assertFalse(self.out.exists())
    def test_nonfinite_bf16(self):
        self.fixture(nonfinite=True)
        with self.assertRaisesRegex(ValueError,'nonfinite attention'):export(self.model,self.out,0,'kda')
        self.assertFalse(self.out.exists())
    def test_missing_layer(self):
        with self.assertRaisesRegex(ValueError,'missing attention tensor'):export(self.model,self.out,0,'kda')
        self.assertFalse(self.out.exists())

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--build-dir',type=Path,required=True)
    a,rest=ap.parse_known_args();BUILD=a.build_dir.resolve();BUILD.mkdir(parents=True,exist_ok=True)
    unittest.main(argv=[__file__]+rest)
