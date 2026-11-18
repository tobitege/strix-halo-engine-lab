import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.qualification.turbo_bar import spool
from tools.qualification.turbo_bar.tests.helpers import INPUTS, POSITIONS, V


class SpoolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.z = np.random.default_rng(1).normal(size=(len(POSITIONS), V)).astype(np.float32)

    def write(self, version=2, positions=POSITIONS, name="s.bin", z=None):
        p = self.tmp / name
        spool.write(p, INPUTS, positions, self.z if z is None else z, version)
        return p

    def test_roundtrip_v2_and_v1(self):
        rows = list(spool.read(self.write(2)))
        self.assertEqual([r[0] for r in rows], POSITIONS)
        for i, (pos, inp, label, z) in enumerate(rows):
            self.assertEqual((inp, label), (INPUTS[pos], INPUTS[pos + 1]))
            np.testing.assert_array_equal(z, self.z[i])
        p1 = self.write(1, positions=list(range(len(INPUTS) - 1)), name="s1.bin",
                        z=np.zeros((len(INPUTS) - 1, V), dtype=np.float32))
        self.assertEqual([r[0] for r in spool.read(p1)], list(range(len(INPUTS) - 1)))
        self.assertTrue(spool.complete(p1))

    def test_missing_footer_and_count_mismatch(self):
        data = self.write().read_bytes()
        (self.tmp / "t.bin").write_bytes(data[:-16])
        with self.assertRaises(spool.SpoolError):
            list(spool.read(self.tmp / "t.bin"))
        self.assertFalse(spool.complete(self.tmp / "t.bin"))
        (self.tmp / "c.bin").write_bytes(data[:-8] + struct.pack("<Q", 3))
        with self.assertRaises(spool.SpoolError):
            list(spool.read(self.tmp / "c.bin"))
        (self.tmp / "x.bin").write_bytes(data + b"\0")
        with self.assertRaises(spool.SpoolError):
            list(spool.read(self.tmp / "x.bin"))

    def test_non_ascending_positions(self):
        with self.assertRaises(spool.SpoolError):
            list(spool.read(self.write(positions=[2, 5, 5, 10])))

    def test_input_mismatch_and_nonfinite(self):
        p = self.write()
        h = spool.header(p)
        data = bytearray(p.read_bytes())
        data[h["data_offset"]:h["data_offset"] + 4] = struct.pack("<I", 7)
        (self.tmp / "m.bin").write_bytes(bytes(data))
        with self.assertRaises(spool.SpoolError):
            list(spool.read(self.tmp / "m.bin"))
        z = self.z.copy()
        z[1, 2] = np.nan
        with self.assertRaises(spool.SpoolError):
            list(spool.read(self.write(z=z, name="n.bin")))

    def test_bad_magic(self):
        (self.tmp / "b.bin").write_bytes(b"NOTASPOOL" + b"\0" * 64)
        with self.assertRaises(spool.SpoolError):
            spool.header(self.tmp / "b.bin")


if __name__ == "__main__":
    unittest.main()
