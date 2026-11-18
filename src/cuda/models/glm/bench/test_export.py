"""CPU-only archive validation tests. All temporary assets stay in --build-dir."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from export_model import export


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=BUILD)
        self.root = Path(self.temp.name)
        self.model = self.root/'model'
        self.model.mkdir()
        self.out = self.root/'weights.native'

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, marker=0x83dcd12d, nonfinite=False, bias=False):
        # Full-width shared FFN; components deliberately split across shards.
        root = 'model.language_model.layers.3.mlp.shared_experts.'
        packed, scales = {}, {}
        for role, i, o, bits in [('gate',4096,2048,3),('up',4096,2048,5),('down',2048,4096,6)]:
            prefix = root+role+'_proj'
            packed[prefix+'.trellis'] = ('I16',[i//16,o//16,16*bits],bytes(i*o*bits//8))
            scales[prefix+'.suh'] = ('F16',[i],struct.pack('<H',0x7c00 if nonfinite else 0x3c00)*i)
            scales[prefix+'.svh'] = ('F16',[o],struct.pack('<H',0xbc00)*o)
            scales[prefix+'.mul1'] = ('I32',[],struct.pack('<I',marker))
        if bias:
            scales[root+'gate_proj.bias'] = ('F32',[2048],bytes(2048*4))
        for name, tensors in [('a.safetensors',packed),('b.safetensors',scales)]:
            header, data = {}, bytearray()
            for key,(dtype,shape,raw) in tensors.items():
                off=len(data); data.extend(raw)
                header[key]=dict(dtype=dtype,shape=shape,data_offsets=[off,len(data)])
            raw=json.dumps(header).encode()
            (self.model/name).write_bytes(struct.pack('<Q',len(raw))+raw+data)

    def test_split_shard_roundtrip_and_hash(self):
        self.fixture()
        export(self.model,self.out,3,'shared')
        raw=self.out.read_bytes()
        self.assertEqual(raw[:20],b'GLMNAT01'+struct.pack('<3I',1,4096,2048))
        pos=20
        for i,o,bits in [(4096,2048,3),(4096,2048,5),(2048,4096,6)]:
            self.assertEqual(struct.unpack_from('<I',raw,pos)[0],bits); pos+=4
            n=i*o*bits//8
            self.assertEqual(raw[pos:pos+n],bytes(n)); pos+=n
            self.assertEqual(raw[pos:pos+2*i],struct.pack('<H',0x3c00)*i); pos+=2*i
            self.assertEqual(raw[pos:pos+2*o],struct.pack('<H',0xbc00)*o); pos+=2*o
        self.assertEqual(pos,len(raw))
        manifest=json.loads(self.out.with_suffix('.native.json').read_text())
        self.assertEqual(manifest['sha256'],hashlib.sha256(raw).hexdigest())
        self.assertEqual(len(manifest['tensors']),9)

    def test_non_mul1_rejected_before_output(self):
        self.fixture(marker=1)
        with self.assertRaisesRegex(ValueError,'mul1 marker'):
            export(self.model,self.out,3,'shared')
        self.assertFalse(self.out.exists())

    def test_nonfinite_rejected(self):
        self.fixture(nonfinite=True)
        with self.assertRaisesRegex(ValueError,'nonfinite scale'):
            export(self.model,self.out,3,'shared')

    def test_unsupported_convention_rejected(self):
        self.fixture(bias=True)
        with self.assertRaisesRegex(ValueError,'unsupported convention'):
            export(self.model,self.out,3,'shared')

    def test_fp8_missing_trellis_is_not_reference(self):
        with self.assertRaisesRegex(ValueError,'missing EXL3 tensor'):
            export(self.model,self.out,3,'experts')
        self.assertFalse(self.out.exists())

    def test_header_extent_rejected(self):
        (self.model/'a.safetensors').write_bytes(struct.pack('<Q',10000)+b'{}')
        with self.assertRaisesRegex(ValueError,'header extent'):
            export(self.model,self.out,3,'shared')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--build-dir',type=Path,required=True)
    args,rest=ap.parse_known_args()
    BUILD=args.build_dir.resolve()
    BUILD.mkdir(parents=True,exist_ok=True)
    unittest.main(argv=[__file__]+rest)
