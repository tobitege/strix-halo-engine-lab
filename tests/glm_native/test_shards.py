"""CPU synthetic hash-cache tests by default; --model opts into real-shard packing.

Synthetic fixtures and authenticated records are confined to --build and removed.
With --model, only small JSON receipts are written; no cache is materialized.
base.bin covers every nonexpert tensor: dense KDA layers 0/1, sparse KDA 4/5,
DSA 3/7 and the unique MTP layer 45. Complete routed blobs cover both sparse
attention types twice, plus MTP (there is only one MTP layer in the model).
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('cache', ROOT/'src/cuda/models/glm/producer/cache.py')
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


class SourceHashTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=BUILD)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model, self.hashes = self.root/'model', self.root/'hashes'
        self.model.mkdir()
        (self.model/'config.json').write_text('{"z":1,"nested":{"b":2,"a":3}}')
        # Multiple files, >2 IO chunks and a non-sector-aligned tail overall.
        # Each artifact stays below 100 MB. These are valid small safetensors.
        for name, n in [('b', 2*(8 << 20)+124), ('a', 65540), ('c', 4)]:
            header = json.dumps({name: dict(dtype='I32', shape=[n//4], data_offsets=[0,n])}).encode()
            (self.model/f'{name}.safetensors').write_bytes(struct.pack('<Q', len(header))+header+(b'\x19\x27\x31\x45'*(n//4)))
        self.paths = sorted(self.model.glob('*.safetensors'))
        self.size = sum(p.stat().st_size for p in self.paths)

    def probe(self, *flags):
        return json.loads(subprocess.check_output([str(BUILD/'glm_native_source_inventory.exe'),
            str(self.model), str(self.hashes), *flags], text=True))

    def legacy_manifest(self):
        # Independent old full-read source SHA + canonical model/cache contract.
        source = dict(config=json.loads((self.model/'config.json').read_text()),
            files=[dict(name=p.name, bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in self.paths])
        contract = dict(cache.VERSION, model_sha256=hashlib.sha256(cache.canonical(source)).hexdigest())
        return dict(contract, sources=source, cache_key=hashlib.sha256(cache.canonical(contract)).hexdigest())

    def test_cold_restart_and_legacy_identity(self):
        first, second = self.probe(), self.probe()
        self.assertEqual(first['bytes_read'], self.size)
        self.assertEqual(first['verified_hash_hits'], 0)
        self.assertEqual(first['manifest'], self.legacy_manifest())
        self.assertEqual(second['manifest'], first['manifest'])
        if second['journal_unavailable_files']:
            self.assertEqual(second['bytes_read'], self.size)
            self.skipTest('real journal unavailable: cold/fallback identity checked; persistent hits require NTFS journal')
        self.assertEqual(second['verified_hash_hits'], len(self.paths))
        self.assertEqual(second['bytes_read'], 0, 'warm start must not read shard payloads')

    def test_authority_failure_and_no_journal_fully_hash(self):
        first = self.probe()
        saved = {p.name: p.read_bytes() for p in self.hashes.glob('*.sha256')}
        for flag in ('--authority-failure', '--no-journal'):
            result = self.probe(flag)
            self.assertEqual(result['manifest'], first['manifest'])
            self.assertEqual(result['verified_hash_hits'], 0)
            self.assertEqual(result['bytes_read'], self.size)
            self.assertEqual({p.name: p.read_bytes() for p in self.hashes.glob('*.sha256')}, saved)

    def test_authenticated_record_required(self):
        first = self.probe()
        for path in self.hashes.glob('*.sha256'):
            lines = path.read_text().splitlines()
            lines[1] = '0'*64  # Preserve metadata and the old MAC, forge digest.
            path.write_text('\n'.join(lines)+'\n')
        result = self.probe()
        self.assertEqual(result['manifest'], first['manifest'])
        self.assertEqual(result['verified_hash_hits'], 0)
        self.assertEqual(result['bytes_read'], self.size)

    def test_changed_bytes_all_timestamps_restored(self):
        import ctypes as c
        from ctypes import wintypes as w
        first = self.probe()
        if first['journal_unavailable_files']:
            self.skipTest('USN replay test requires active NTFS journal')
        class Basic(c.Structure):
            _fields_ = [(n, c.c_longlong) for n in ('creation', 'access', 'write', 'change')]+[('attributes', w.DWORD)]
        k = c.WinDLL('kernel32', use_last_error=True)
        k.CreateFileW.argtypes = [w.LPCWSTR,w.DWORD,w.DWORD,c.c_void_p,w.DWORD,w.DWORD,w.HANDLE]
        k.CreateFileW.restype = w.HANDLE
        k.GetFileInformationByHandleEx.argtypes = [w.HANDLE,c.c_int,c.c_void_p,w.DWORD]
        k.SetFileInformationByHandle.argtypes = [w.HANDLE,c.c_int,c.c_void_p,w.DWORD]
        k.WriteFile.argtypes = [w.HANDLE,c.c_void_p,w.DWORD,c.POINTER(w.DWORD),c.c_void_p]
        k.SetFilePointerEx.argtypes = [w.HANDLE,c.c_longlong,c.c_void_p,w.DWORD]
        k.FlushFileBuffers.argtypes = [w.HANDLE]
        k.CloseHandle.argtypes = [w.HANDLE]
        path = self.paths[0]
        # Never read bytes before checking restored LastAccessTime: all four
        # FILE_BASIC_INFO times must match.
        before = Basic()
        h = k.CreateFileW(str(path), 0x40000180, 1, None, 3, 0, None)
        self.assertNotEqual(h, c.c_void_p(-1).value)
        try:
            self.assertTrue(k.GetFileInformationByHandleEx(h, 0, c.byref(before), c.sizeof(before)))
            self.assertTrue(k.SetFilePointerEx(h, -1, None, 2))
            wrote = w.DWORD()
            self.assertTrue(k.WriteFile(h, b'X', 1, c.byref(wrote), None))
            self.assertEqual(wrote.value, 1)
            self.assertTrue(k.FlushFileBuffers(h))
            self.assertTrue(k.SetFileInformationByHandle(h, 0, c.byref(before), c.sizeof(before)))
        finally:
            k.CloseHandle(h)
        h = k.CreateFileW(str(path), 0x80, 1, None, 3, 0, None)
        try:
            after = Basic()
            self.assertTrue(k.GetFileInformationByHandleEx(h, 0, c.byref(after), c.sizeof(after)))
            self.assertEqual([getattr(before,n) for n in ('creation','access','write','change')],
                             [getattr(after,n) for n in ('creation','access','write','change')])
        finally:
            k.CloseHandle(h)
        result = self.probe()
        self.assertEqual(result['verified_hash_hits'], len(self.paths)-1)
        self.assertEqual(result['bytes_read'], path.stat().st_size)
        self.assertNotEqual(result['manifest']['model_sha256'], first['manifest']['model_sha256'])
        self.assertEqual(result['manifest'], self.legacy_manifest())
        self.assertEqual(self.probe()['bytes_read'], 0)


def run(model, build):
    started = time.perf_counter()
    tensors, files = cache.inventory(model)
    schema = cache.validate_schema(tensors)
    cache.validate_config(json.loads((model/'config.json').read_text()))
    blobs = ['base.bin'] + [f'experts-{l:02}.bin' for l in (3, 4, 5, 7, 45)]
    native = subprocess.run([str(build/'glm_native_shards.exe'), str(model), *blobs],
                            capture_output=True, text=True, check=True)
    records = [json.loads(line) for line in native.stdout.splitlines()]
    assert records[0]['tensors'] == len(schema)
    source_manifest = dict(config=json.loads((model/'config.json').read_text()),
                           files=[dict(name=p.name, bytes=p.stat().st_size, sha256='UNHASHED-TEST-ONLY') for p in files])
    model_sha = hashlib.sha256(cache.canonical(source_manifest)).hexdigest()
    assert records[0]['model_sha256'] == model_sha, 'source model identity canonicalization'
    assert records[0]['cache_key'] == hashlib.sha256(cache.canonical(dict(cache.VERSION, model_sha256=model_sha))).hexdigest()
    results = {r['blob']: r for r in records[1:]}
    groups = {b: [] for b in blobs}
    for name in sorted(schema):
        match = re.search(r'\.layers\.(\d+)\.mlp\.experts\.', name)
        blob = f'experts-{int(match[1]):02}.bin' if match else 'base.bin'
        if blob in groups:
            groups[blob].append(name)
    for blob, names in groups.items():
        begin = time.perf_counter()
        h, count = hashlib.sha256(), 0
        for _, offset, data in cache.packed_chunks(tensors, names, validate_finite=False):
            assert offset == count
            h.update(data)
            count += len(data)
        assert count == results[blob]['bytes'] == records[0]['blobs'][blob]['bytes']
        assert h.hexdigest() == results[blob]['sha256'], blob
        results[blob]['cache_pack_hash_ms'] = (time.perf_counter()-begin)*1000
        print('PASS real-shard/cache byte identity', blob, count, h.hexdigest(), flush=True)
    receipt = dict(verdict='EXACT', scope='compressed placement only; no GPU/model fidelity claim',
                   source_files=len(files), tensors=len(schema), staging_bytes=8 << 20,
                   layers=dict(dense_kda=[0, 1], sparse_kda=[4, 5], sparse_dsa=[3, 7], mtp=[45]),
                   blobs=list(results.values()), elapsed_ms=(time.perf_counter()-started)*1000)
    (build/'producer-shards-equivalence.json').write_text(json.dumps(receipt, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, help='Explicit real-shard packing test; omitted for synthetic CPU tests')
    parser.add_argument('--build', type=Path, required=True)
    args = parser.parse_args()
    BUILD = args.build.resolve()
    if args.model:
        run(args.model, BUILD)
    else:
        result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SourceHashTests))
        raise SystemExit(0 if result.wasSuccessful() else 1)
