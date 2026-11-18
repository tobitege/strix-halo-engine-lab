#!/usr/bin/env python3
"""CPU regressions for owned server teardown and immutable paired-plan resume."""
import contextlib
import ctypes
import importlib.util
import io
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

SPEC = importlib.util.spec_from_file_location("fair_runner", pathlib.Path(__file__).with_name("run-fair.py"))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class Sampler:
    def __init__(self):
        self.samples, self.errors = [], []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class RunnerLifecycle(unittest.TestCase):
    def test_grouped_classes_have_one_process_cold_request(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.ExitStack() as patches:
            root = pathlib.Path(folder)
            arm = dict(name="a", cmd=["fixture"], url="http://127.0.0.1:1")
            config = dict(pack_identity="test", decode_prompts={k: [k] for k in ("prose", "edit", "json")})
            args = types.SimpleNamespace(release_wait_s=0, idle_memory_mib=2048, ready_s=1, timeout=1, tokens=3)
            def client(arm, prompts, concurrency, tokens, out, timeout):
                out.write_text("{}")
                return dict(file=out.name, start=0, end=1)
            for name, value in (("free_ram", 32 * 1024**3), ("gpu_sample", dict(memory_mib=0)),
                                ("wait_ready", None), ("stop_owned", None)):
                patches.enter_context(mock.patch.object(runner, name, return_value=value))
            patches.enter_context(mock.patch.object(runner, "Sampler", Sampler))
            patches.enter_context(mock.patch.object(runner.subprocess, "run", return_value=types.SimpleNamespace(
                stdout="NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 00000000:01:00.0, GPU-test\n")))
            patches.enter_context(mock.patch.object(runner.subprocess, "Popen", return_value=types.SimpleNamespace(pid=123)))
            patches.enter_context(mock.patch.object(runner.owner_api, "windows_job", return_value=None))
            patches.enter_context(mock.patch.object(runner, "client", side_effect=client))
            patches.enter_context(contextlib.redirect_stdout(io.StringIO()))
            runner.run_one(config, arm, "decode-all", 1, root / "entry", args)
            records = json.loads((root / "entry/run.json").read_text())["records"]
            self.assertEqual(len({row["file"] for row in records}), 12)
            self.assertEqual([(row["kind"], row["state"]) for row in records if row["state"] != "warm"],
                             [("prose", "cold"), ("edit", "intermediate"), ("json", "intermediate")])
            for kind in config["decode_prompts"]:
                self.assertEqual([row["concurrency"] for row in records if row["kind"] == kind and row["state"] == "warm"], [1, 4, 8])

    @unittest.skipUnless(os.name == "nt", "real Windows Job ownership")
    def test_server_parent_exit_does_not_leave_owned_child(self):
        from ctypes import wintypes as w
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        kernel.OpenProcess.restype = w.HANDLE
        kernel.WaitForSingleObject.argtypes = [w.HANDLE, w.DWORD]
        kernel.TerminateProcess.argtypes = [w.HANDLE, w.UINT]
        kernel.CloseHandle.argtypes = [w.HANDLE]
        owned = {}
        with tempfile.TemporaryDirectory() as folder:
            root = pathlib.Path(folder)
            gate, pidfile = root / "spawn", root / "child.pid"
            launcher = "\n".join([
                "import pathlib,subprocess,sys,time",
                "gate=pathlib.Path(sys.argv[1])",
                "while not gate.exists(): time.sleep(0.005)",
                "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])",
                "pathlib.Path(sys.argv[2]).write_text(str(child.pid))",
            ])
            arm = dict(name="a", cmd=[sys.executable, "-c", launcher, str(gate), str(pidfile)], url="http://127.0.0.1:1")
            config = dict(pack_identity="test-frozen-pack", decode_prompts=dict(prose=["unused-fixture"]))
            args = types.SimpleNamespace(release_wait_s=0, idle_memory_mib=2048, ready_s=5, timeout=5, tokens=3)

            def ready(proc, arm, timeout):
                gate.touch()  # Parent must be assigned to the Job before its child is spawned.
                proc.wait(timeout=timeout)
                self.assertEqual(proc.returncode, 0)
                owned["pid"] = int(pidfile.read_text())
                owned["handle"] = kernel.OpenProcess(0x00100001, False, owned["pid"])  # SYNCHRONIZE | TERMINATE
                self.assertTrue(owned["handle"])
                self.assertEqual(kernel.WaitForSingleObject(owned["handle"], 0), 258, "fixture child must outlive its parent")

            def client(arm, prompts, concurrency, tokens, out, timeout):
                out.write_text('{"sentinel":"completed receipt"}')
                return dict(file=out.name, start=0, end=1)

            try:
                with mock.patch.object(runner, "free_ram", return_value=32 * 1024**3), \
                     mock.patch.object(runner, "gpu_sample", return_value=dict(memory_mib=0)), \
                     mock.patch.object(runner, "Sampler", Sampler), mock.patch.object(runner, "wait_ready", ready), \
                     mock.patch.object(runner, "client", side_effect=client) as calls, contextlib.redirect_stdout(io.StringIO()):
                    runner.run_one(config, arm, "decode-prose", 1, root / "entry", args)
                self.assertEqual(calls.call_count, 4)
                self.assertEqual(kernel.WaitForSingleObject(owned["handle"], 5000), 0,
                                 "the server Job must terminate a child after its parent already exited")
                saved = json.loads((root / "entry/run.json").read_text())
                self.assertEqual(saved["status"], "complete")
                self.assertEqual(len(saved["records"]), 4)
            finally:
                handle = owned.get("handle")
                if handle:
                    if kernel.WaitForSingleObject(handle, 0) == 258:
                        kernel.TerminateProcess(handle, 99)  # Only this controlled fixture's retained process handle.
                        kernel.WaitForSingleObject(handle, 5000)
                    kernel.CloseHandle(handle)

    def fixture(self, root):
        a, b = root / "a.txt", root / "b.txt"
        a.write_text("A" * 80)
        b.write_text("B" * 80)
        config = dict(pack_identity="test-frozen-pack", arms=[
            dict(name=name, cmd=[sys.executable, "-c", "raise RuntimeError('must not launch')"],
                 url=f"http://127.0.0.1:{index + 1}", cache_policy="fixture caches disabled")
            for index, name in enumerate(("a", "b"))],
            decode_prompts={kind: [str(a)] for kind in ("prose", "edit", "json")},
            prefill_prompts={target: [str(a), str(b)] for target in ("8192", "32768", "43008")})
        configfile, out = root / "input.json", root / "out"
        configfile.write_text(json.dumps(config))
        out.mkdir()
        runner.write(out / "config.json", config)
        runner.write(out / "prompt-sha256.json", runner.validate(config))
        entry = out / "00-a-decode-prose"
        entry.mkdir()
        runner.write(entry / "run.json", dict(status="complete", arm="a", workload="decode-prose", pair=1,
                                              pack_identity=config["pack_identity"], records=[dict(file="sentinel.json")],
                                              sentinel="must not overwrite manifest"))
        (entry / "sentinel.json").write_bytes(b'{"sentinel":"must not overwrite receipt"}\n')
        return config, configfile, out, entry

    def resume(self, configfile, out, resume=True):
        with mock.patch.object(sys, "argv", ["run-fair.py", "--config", str(configfile), "--out", str(out),
                                              "--run-index", "0"] + (["--resume"] if resume else [])), \
             mock.patch.object(runner, "run_one", side_effect=AssertionError("resume must not launch this entry")) as launch, \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            try:
                result = runner.main()
            finally:
                launch.assert_not_called()
        return result

    def test_resume_keeps_completed_entry_without_launch_or_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            _, configfile, out, entry = self.fixture(pathlib.Path(folder))
            before = {path: path.read_bytes() for path in entry.iterdir()}
            self.assertEqual(self.resume(configfile, out), 0)
            self.assertEqual({path: path.read_bytes() for path in entry.iterdir()}, before)
            self.assertFalse((out / ".done").exists(), "one kept entry does not complete the 36-entry plan")

    def test_resume_refuses_changed_frozen_config_and_prompt_bytes(self):
        for drift in ("config", "prompt"):
            with self.subTest(drift=drift), tempfile.TemporaryDirectory() as folder:
                root = pathlib.Path(folder)
                config, configfile, out, entry = self.fixture(root)
                if drift == "config":
                    config["arms"][0]["cmd"].append("changed executable declaration")
                    configfile.write_text(json.dumps(config))
                else:
                    (root / "a.txt").write_text("changed prompt bytes")
                before = {path: path.read_bytes() for path in entry.iterdir()}
                with self.assertRaises(SystemExit) as error:
                    self.resume(configfile, out)
                self.assertEqual(error.exception.code, 2)
                self.assertEqual({path: path.read_bytes() for path in entry.iterdir()}, before)

    def test_resume_refuses_incomplete_identity_drift_and_missing_receipt(self):
        for damage in ("failed", "identity", "receipt"):
            with self.subTest(damage=damage), tempfile.TemporaryDirectory() as folder:
                _, configfile, out, entry = self.fixture(pathlib.Path(folder))
                saved = json.loads((entry / "run.json").read_text())
                if damage == "failed":
                    saved["status"] = "failed"
                elif damage == "identity":
                    saved["pair"] = 2
                else:
                    saved["records"][0]["file"] = "missing.json"
                runner.write(entry / "run.json", saved)
                before = {path: path.read_bytes() for path in entry.iterdir()}
                with self.assertRaises(SystemExit) as error:
                    self.resume(configfile, out)
                self.assertEqual(error.exception.code, 2)
                self.assertEqual({path: path.read_bytes() for path in entry.iterdir()}, before)

    def test_resume_and_index_refuse_missing_frozen_metadata(self):
        for resume in (False, True):
            for missing in ("config.json", "prompt-sha256.json"):
                with self.subTest(resume=resume, missing=missing), tempfile.TemporaryDirectory() as folder:
                    _, configfile, out, _ = self.fixture(pathlib.Path(folder))
                    (out / missing).unlink()
                    before = {path: path.read_bytes() for path in out.rglob("*") if path.is_file()}
                    with self.assertRaises(SystemExit) as error:
                        self.resume(configfile, out, resume=resume)
                    self.assertEqual(error.exception.code, 2)
                    self.assertFalse((out / missing).exists(), "missing frozen metadata must never be reconstructed")
                    self.assertEqual({path: path.read_bytes() for path in out.rglob("*") if path.is_file()}, before)


if __name__ == "__main__":
    unittest.main()
