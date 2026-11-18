"""Exercise the probe's actual send/report path without sockets or timing noise."""
import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location(
    "owners_probe", Path(__file__).resolve().parents[1] / "scripts/bench/handoff/owners_probe.py"
)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ProbeMedianTests(unittest.TestCase):
    def test_default_pairs_excludes_cold_and_averages_even_warm_samples(self):
        samples = [100, 9, 10, 11, 12, 13]  # cold A; warm A median 11, warm B median 11
        indices = [0, 0]

        def fake_send(host, port, frames, out, rank, gate):
            del host, port, gate
            out[rank] = (1000 + samples[indices[rank]], sum(frames))
            indices[rank] += 1

        output = io.StringIO()
        args = SimpleNamespace(to="rank0:1,rank1:2", family="minimax", tokens=2, pairs=3, gap=0)
        with mock.patch.object(probe, "send_one", fake_send), \
                mock.patch.object(probe.time, "sleep"), \
                mock.patch.object(probe.time, "perf_counter", return_value=1000), \
                contextlib.redirect_stdout(output):
            probe.send(args)
        self.assertIn("warm arm A: median 11.000 s over 2 (min 10.000 max 12.000)", output.getvalue())
        self.assertIn("warm arm B: median 11.000 s over 3 (min 9.000 max 13.000)", output.getvalue())


if __name__ == "__main__":
    unittest.main()
