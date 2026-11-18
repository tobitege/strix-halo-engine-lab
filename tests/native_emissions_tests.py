#!/usr/bin/env python3
"""CPU checks for transparent native emission capture and count-gated peaks."""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/bench/common/native-emissions.py"
SPEC = importlib.util.spec_from_file_location("native_emissions", SCRIPT)
native = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native)


class NativeEmissions(unittest.TestCase):
    def test_repeated_invocations_expand_unique_trace_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            template = pathlib.Path(folder) / "native-{pid}-{time_ns}.jsonl"
            code = "import sys; sys.stdin.buffer.read(); print('T 42',flush=True)"
            for _ in range(2):
                proc = subprocess.run([sys.executable, str(SCRIPT), "run", "--out", str(template),
                                       "--token-pattern", r"T (?P<token>\d+)", "--", sys.executable, "-c", code],
                                      input=b"", capture_output=True, timeout=10)
                self.assertEqual(proc.returncode, 0, proc.stderr.decode(errors="replace"))
            traces = sorted(pathlib.Path(folder).glob("native-*.jsonl"))
            self.assertEqual(len(traces), 2)
            self.assertNotEqual(traces[0].name, traces[1].name)
            for path in traces:
                self.assertNotIn("{", path.name)
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual([row["token"] for row in rows if "token" in row], [42])

    @unittest.skipUnless(os.name == "nt", "Windows owned-child job lifecycle")
    def test_forced_proxy_cleanup_terminates_owned_native_child(self):
        from ctypes import wintypes as w
        kernel = native.ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        kernel.OpenProcess.restype = w.HANDLE
        kernel.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
        kernel.CloseHandle.argtypes = [w.HANDLE]
        with tempfile.TemporaryDirectory() as folder:
            trace = pathlib.Path(folder) / "trace.jsonl"
            code = "import os,time; print('PID',os.getpid(),flush=True); print('T 42',flush=True); time.sleep(60)"
            proc = subprocess.Popen([sys.executable, str(SCRIPT), "run", "--out", str(trace),
                                     "--token-pattern", r"T (?P<token>\d+)", "--", sys.executable, "-c", code],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            owned_pid, handle = None, None
            try:
                owned_pid = int(proc.stdout.readline().split()[1])
                handle = kernel.OpenProcess(0x00100000, False, owned_pid)  # SYNCHRONIZE
                self.assertTrue(handle)
                self.assertEqual(proc.stdout.readline(), b"T 42\r\n" if os.name == "nt" else b"T 42\n")
                proc.kill()
                proc.wait(timeout=5)
                self.assertEqual(kernel.WaitForSingleObject(handle, 5000), 0, "native child survived forced proxy termination")
                lines = [json.loads(line) for line in trace.read_text().splitlines()]
                self.assertEqual([line["token"] for line in lines if "token" in line], [42])
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=5)
                if handle:
                    alive = kernel.WaitForSingleObject(handle, 0) == 258
                    kernel.CloseHandle(handle)
                    if alive:
                        subprocess.run(["taskkill", "/PID", str(owned_pid), "/T", "/F"], capture_output=True)
                for stream in (proc.stdin, proc.stdout, proc.stderr):
                    stream.close()

    def test_protocol_is_explicit_and_unambiguous(self):
        expressions = native.patterns([r"T (?P<token>\d+)", r"BT (?P<slot>\d+) (?P<token>\d+)"])
        self.assertEqual(native.token_event("T 42", expressions), dict(token=42, slot=None))
        self.assertEqual(native.token_event("BT 3 99", expressions), dict(token=99, slot="3"))
        for line in ("DRAFT 42", "T 42 extra", "INFO T 42", "BDONE 3 8 stop"):
            self.assertIsNone(native.token_event(line, expressions))
        with self.assertRaises(ValueError):
            native.patterns([r"T (\d+)"])
        with self.assertRaises(ValueError):
            native.token_event("T 42", [expressions[0], expressions[0]])

    def test_proxy_preserves_binary_stdio_and_records_committed_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            trace = pathlib.Path(folder) / "trace.jsonl"
            script = "import sys; data=sys.stdin.buffer.read(); sys.stdout.buffer.write(data+b'\\xff\\nT 42\\nDRAFT 9\\nBT 3 99\\n'); sys.stdout.buffer.flush()"
            proc = subprocess.run([sys.executable, str(SCRIPT), "run", "--out", str(trace),
                                   "--token-pattern", r"T (?P<token>\d+)", "--token-pattern", r"BT (?P<slot>\d+) (?P<token>\d+)",
                                   "--", sys.executable, "-c", script], input=b"input\r\n", capture_output=True, timeout=10)
            self.assertEqual(proc.returncode, 0, proc.stderr.decode(errors="replace"))
            self.assertEqual(proc.stdout, b"input\r\n\xff\nT 42\nDRAFT 9\nBT 3 99\n")
            lines = [json.loads(line) for line in trace.read_text().splitlines()]
            self.assertEqual([line["token"] for line in lines if "token" in line], [42, 99])
            self.assertTrue(lines[-1]["end"])

    def test_peak_requires_exact_usage_and_one_second(self):
        with tempfile.TemporaryDirectory() as folder:
            trace, receipt = pathlib.Path(folder) / "trace.jsonl", pathlib.Path(folder) / "client.json"
            lines = [dict(schema="native-emissions-v1", clock="perf_counter_ns")]
            lines += [dict(steady_ns=t, token=i) for i, t in enumerate([1000000000, 1100000000, 2200000000])]
            trace.write_text("\n".join(json.dumps(line) for line in lines))
            receipt.write_text(json.dumps(dict(events=[dict(arrival=0.5, finished=2.3, tokens=3, error=None)])))
            result = native.peak(trace, receipt)
            self.assertTrue(result["valid"])
            self.assertEqual(result["peak_1s_tok_s"], 2)
            receipt.write_text(json.dumps(dict(events=[dict(arrival=0.5, finished=2.3, tokens=4, error=None)])))
            result = native.peak(trace, receipt)
            self.assertFalse(result["valid"])
            self.assertIsNone(result["peak_1s_tok_s"])


if __name__ == "__main__":
    unittest.main()
