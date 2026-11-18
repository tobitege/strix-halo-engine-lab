"""Offline PR 646 regressions: real producer formats and mocked process/peer execution."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
COMMON = HERE.parent / "common"
sys.path.insert(0, str(HERE))
from test_runner import make_run, sample, load
import receipt
import serve_receipts
import abab


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8", newline="\n")


def quiet(fn, *args):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


class ReceiptRegressions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)

    def summarize(self, *args):
        quiet(receipt.main, ["summarize", str(self.d), "--tag", "test", *args])
        return load(self.d)

    def test_native_stdout_formats_and_per_prefill_tokens(self):
        # apps/deepseek41_prefill_main.cpp:109,281; glm_rank_main.cpp:1749;
        # minimax_rank_main.cpp:215,2159. Stage time is deliberately different.
        fixtures = {
            "ds41": ["ttft run 0: 999.000 ms (at 999.000 ms)",
                     "ttft run 0: 8192 rows, wall 1000.000 ms (run 980.000 ms, engram read 1.000 ms, head 1.000 ms), next token 7, logits_fnv 0123456789abcdef",
                     "ttft run 1: 32768 rows, wall 2000.000 ms (run 1990.000 ms, engram read 1.000 ms, head 1.000 ms), next token 7, logits_fnv fedcba9876543210"],
            "glm": ["rank 1 pf gemv-ttft: 8192 tokens 1.000 s (0.122 ms/token), token 7, logits fnv 0123456789abcdef, state fnv fedcba9876543210",
                    "rank 1 pf gemv-ttft: 32768 tokens 2.000 s (0.061 ms/token), token 7, logits fnv 0123456789abcdef, state fnv fedcba9876543210"],
            "minimax": ["rank 1 s8: prompt 8192 tokens, TTFT 1.000 s, prefill 8192.0 tok/s (exchange 0.100 s, 10 exchanges)",
                        "rank 1 s32: prompt 32768 tokens, TTFT 2.000 s, prefill 16384.0 tok/s (exchange 0.100 s, 10 exchanges)",
                        "rank 1: result hash 12345678, peer 12345678 -> ranks AGREE"],
        }
        for family, lines in fixtures.items():
            with self.subTest(family=family):
                make_run(str(self.d), [], [sample(99 + i * .2) for i in range(40)], stdout="\n".join(lines) + "\n")
                write(self.d / "stdout.ts", "\n".join("%s %s" % (102 + i, line) for i, line in enumerate(lines)))
                args = ["--family", family, "--need-prefill"]
                if family == "minimax":
                    args += ["--require", "ranks.AGREE"]
                r = self.summarize(*args)
                self.assertTrue(r["valid"], r["invalid_reasons"])
                self.assertEqual(r["prefill_tokens"], [8192, 32768])
                self.assertEqual((r["prefill_ms_cold"], r["prefill_ms_warm"]), (1000, 2000))
                self.assertEqual((r["tok_s_cold"], r["tok_s_warm"]), (8192, 16384))

    def test_request_latency_never_becomes_prefill(self):
        make_run(str(self.d), [(102, "qpf2k sched block rows 8192")], [sample(99 + i * .2) for i in range(30)],
                 stdout=json.dumps({"generation_request_elapsed_ms": 3000}))
        r = self.summarize("--need-prefill", "--anchor-regex", "qpf2k")
        self.assertFalse(r["valid"])
        self.assertEqual(r["prefills"], 0)
        self.assertEqual(r["metrics"]["request_latency_ms"], 3000)

    def test_boundary_eviction_and_missing_coverage(self):
        for rows in ([sample(100), sample(100.2, ev=50), sample(101.2, ev=50)],
                     [sample(100.2), sample(101.2)]):
            make_run(str(self.d), [(101.1, "prefill-wall-ms=1000")], rows)
            self.assertFalse(self.summarize("--need-prefill")["valid"])

    def test_pair_full_width_counts_and_peer_validity(self):
        paths = [self.d / "r0.json", self.d / "r1.json"]
        evidence = [{"key": "0", "tokens": "8192", "logits": "12345678aaaaaaaa"}]
        for other, valid, expected in [([], True, 1),
                                       ([dict(evidence[0], logits="12345678bbbbbbbb")], True, 1),
                                       (evidence, False, 1), (evidence, True, 2)]:
            for p, ev, ok in [(paths[0], evidence, True), (paths[1], other, valid)]:
                write(p, dict(valid=ok, invalid_reasons=[] if ok else ["rc=1"], rank_evidence=ev))
            self.assertTrue(receipt.pair_receipts([str(p) for p in paths], expected))
            self.assertTrue(all(not json.loads(p.read_text())["valid"] for p in paths))

    def test_abab_matches_rounds_not_hash_sets(self):
        a = {i: {"logits_sha": h} for i, h in enumerate(["x", "y", "x"])}
        b = {i: {"logits_sha": h} for i, h in enumerate(["y", "x", "y"])}
        self.assertIn("DIFFER", abab.logits_verdict(a, b))
        self.assertIn("nondeterminism", abab.logits_verdict(a, b))
        b[0]["logits_sha"] = "-"
        self.assertIn("missing", abab.logits_verdict(a, b))
        self.assertIn("missing", abab.logits_verdict(a, {}))

    def test_abab_summary_rejects_per_round_mismatch(self):
        for arm, hashes in (("A", ["x", "y", "x"]), ("B", ["y", "x", "y"])):
            for rnd, h in enumerate(hashes):
                write(self.d / (arm + str(rnd)) / "receipt.json", dict(
                    arm=arm, round=rnd, valid=True, prefill_ms_warm=1, prefill_ms_cold=1,
                    tok_s_warm=1, sclk_mhz=1, busy_pct=1, logits_sha=h, gen_sha="-"))
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            abab.main(["summary", str(self.d), "--base", "A"])
        self.assertNotIn("bitwise-same logits", output.getvalue())
        self.assertIn("DIFFER", output.getvalue())

    def test_stdout_prefill_and_required_agreement(self):
        make_run(str(self.d), [], [sample(99), sample(100), sample(101), sample(102)],
                 stdout="ttft run 0: 8192 rows, wall 1000.000 ms\nranks AGREE\n")
        write(self.d / "stdout.ts", "101 ttft run 0\n102 ranks AGREE\n")
        r = self.summarize("--need-prefill", "--prefill-regex", r"^ttft run \d+: \d+ rows, wall ([\d.]+) ms",
                           "--require", "ranks.AGREE")
        self.assertTrue(r["valid"], r["invalid_reasons"])

    def test_empty_served_receipt_never_valid(self):
        make_run(str(self.d / "server"), [(100, "READY")], [])
        quiet(serve_receipts.main, [str(self.d), "--tag", "t", "--arm", "a", "--client", "1"])
        self.assertFalse(load(self.d / "client")["valid"])

    def test_served_failures_missing_work_and_peer_propagation(self):
        for fault in ("client", "samples", "empty", "missing", "peer", "server", "sampler"):
            with self.subTest(fault=fault):
                s = self.d / "server"
                make_run(str(s), [(100, "READY")], [] if fault == "samples" else [sample(100), sample(101)])
                meta = json.loads((s / "meta.json").read_text())
                meta["intentional_shutdown"] = fault != "server"
                write(s / "meta.json", meta)
                write(s / "sampler.error", "failed" if fault == "sampler" else "")
                result = dict(completed=0 if fault == "empty" else 1, failed=0)
                write(self.d / "client-result.json", result if fault != "missing" else {})
                peer = self.d / "peer.json"
                write(peer, dict(valid=fault != "peer", invalid_reasons=["rc=1"] if fault == "peer" else []))
                quiet(serve_receipts.main, [str(self.d), "--tag", "t", "--arm", "a", "--client", "1",
                      "--client-rc", "1" if fault == "client" else "0", "--peer", str(peer)])
                self.assertFalse(load(self.d / "client")["valid"])
                self.assertFalse(json.loads(peer.read_text())["valid"])

    def test_missing_bursts_have_invalid_receipts(self):
        s = self.d / "server"
        make_run(str(s), [(100, "READY")], [sample(100), sample(101)])
        quiet(serve_receipts.main, [str(self.d), "--tag", "t", "--arm", "a", "--client-rc", "0",
                                  "--expected", "prose-0,json-0"])
        self.assertFalse(load(self.d / "prose-0")["valid"])
        self.assertFalse(load(self.d / "json-0")["valid"])


class VisionRegressions(unittest.TestCase):
    def test_manifest_http_failure_truncated_stream_and_scoring(self):
        module = runpy.run_path(str(HERE.parent / "vision/battery.py"))
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            imgs = [d / (name + ".png") for name in ("red", "blue", "green")]
            for img in imgs:
                write(img, "mock image")
            responses = []
            for status, lines in [(200, [b'data: {"choices":[{"delta":{"content":"red"}}]}\n', b'data: [DONE]\n']),
                                  (500, []), (200, [b'data: {"choices":[]}\n'])]:
                response = mock.MagicMock(status=status)
                response.__iter__.return_value = iter(lines)
                response.read.return_value = b"failed"
                responses.append(response)
            conn = mock.MagicMock()
            conn.getresponse.side_effect = responses
            out = d / "battery.json"
            with mock.patch("http.client.HTTPConnection", return_value=conn):
                self.assertEqual(quiet(module["run"], 1, str(out), "T", [str(p) for p in imgs]), 1)
            results = json.loads(out.read_text())
            self.assertEqual(len(results), 3)
            self.assertEqual([r["status"] for r in results], [200, "error", "error"])
            # Simulate an interrupted/old writer dropping the final result; manifest retains it.
            write(out, results[:2])
            score = runpy.run_path(str(HERE.parent / "vision/score.py"))
            with mock.patch.dict(score["main"].__globals__, classes=lambda _: (["red"], [])), \
                 mock.patch.object(sys, "argv", ["score", str(d), str(out)]):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(score["main"](), 1)
                self.assertIn("colours 1/3", output.getvalue())


@unittest.skipUnless(shutil.which("bash"), "Bash required")
class ShellRegressions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)
        self.bin = self.d / "bin"
        self.bin.mkdir()
        self.env = dict(os.environ, KIT_PYTHON=Path(sys.executable).as_posix(), KIT_OUT=(self.d / "out").as_posix(),
                        KIT_SAMPLE_S="0.1", KIT_SAMPLER_KIND="nvidia", KIT_PEER="mock", KIT_PEER_ROOT=ROOT.as_posix(),
                        KIT_PEER_OUT=(self.d / "peer").as_posix(), KIT_BUDGET="30")
        self.env["PATH"] = str(self.bin) + os.pathsep + self.env["PATH"]
        self.shim("nvidia-smi", "printf '50, 1200, 40, 90, 10\\n'")
        self.shim("journalctl", "exit 0")
        self.shim("ssh", 'bash -c "${@: -1}"')  # no network; execute the peer in a separate local output directory

    def shim(self, name, body):
        p = self.bin / name
        write(p, "#!/usr/bin/env bash\n" + body + "\n")
        p.chmod(0o755)
        return p

    def bash(self, script, timeout=35):
        return subprocess.run([shutil.which("bash"), "-c", script], cwd=ROOT, env=self.env,
                              text=True, capture_output=True, timeout=timeout)

    def test_nvidia_sampler_rows_and_failures(self):
        dest = self.d / "sample"
        dest.mkdir()
        q = shlex.quote(dest.as_posix())
        script = 'source scripts/bench/common/runlib.sh; kit_sampler_start %s; sleep 0.8; kit_sampler_stop' % q
        r = self.bash(script)
        self.assertEqual(r.returncode, 0, r.stderr[-1000:])
        rows = receipt.samples(dest / "samples.tsv")
        self.assertGreaterEqual(len(rows), 2)
        self.assertEqual(rows[0][1:6], [50e6, 1200e6, 40000, 90, 10 * 1048576])
        self.shim("nvidia-smi", "exit 7")
        self.bash(script)
        self.assertIn("failed", (dest / "sampler.error").read_text())

    def test_shell_run_timestamps_stdout(self):
        dest = self.d / "run"
        r = self.bash('source scripts/bench/common/runlib.sh; kit_run %s 5 bash -c %s' % (
            shlex.quote(dest.as_posix()), shlex.quote("printf 'rank 1 pf gemv-ttft: 8192 tokens 1.000 s\\n'; sleep 0.3")))
        self.assertEqual(r.returncode, 0, r.stderr[-1000:])
        self.assertIn("rank 1 pf", (dest / "stdout.ts").read_text())

    def serve(self, tag, extra="", server="echo READY; sleep 12"):
        return self.bash('bash scripts/bench/common/serve-decode.sh --tag %s --port 1 --ready-log READY %s -- bash -c %s' % (
            tag, extra, shlex.quote(server)))

    def test_failed_client_and_term_resistant_server(self):
        started = time.monotonic()
        r = self.serve("failed", "--client-cmd 'sleep 0.3; exit 1'", "trap '' TERM; echo READY; sleep 12")
        self.assertLess(time.monotonic() - started, 15)
        self.assertNotEqual(r.returncode, 0)
        rec = load(self.d / "out/failed/client")
        self.assertFalse(rec["valid"])
        self.assertEqual(rec["client_rc"], 1)

    def test_readiness_and_warmup_share_deadline(self):
        self.env["KIT_BUDGET"] = "19"
        started = time.monotonic()
        r = self.serve("never", server="sleep 12")
        self.assertLess(time.monotonic() - started, 12)
        self.assertNotEqual(r.returncode, 0)
        # Replace only serve_bench execution with a sleeping Python fixture.
        wrapper = self.shim("kit-python", 'if [[ "$*" == *tools/serve_bench.py* ]]; then sleep 12; else exec ' +
                            shlex.quote(Path(sys.executable).as_posix()) + ' "$@"; fi')
        self.env["KIT_PYTHON"] = wrapper.as_posix()
        started = time.monotonic()
        r = self.serve("warm", "--classes prose --concurrency 1")
        self.assertLess(time.monotonic() - started, 12)
        self.assertNotEqual(r.returncode, 0)

    def test_peer_preserves_prompt_directory(self):
        # A mock serve_bench verifies the real prompt path and writes completed burst evidence.
        client = self.d / "client.py"
        write(client, '''import sys,json,pathlib
a=sys.argv
p=pathlib.Path(a[a.index('--prompt-file')+1]); assert p.read_text()=='fixture prompt'
if '--out' in a:
 pathlib.Path(a[a.index('--out')+1]).write_text(json.dumps({'bursts':[{'concurrency':1,'client':{'failed':0,'requests':1,'output_tokens':1}}]}))
''')
        wrapper = self.shim("kit-python", 'if [[ "$*" == *tools/serve_bench.py* ]]; then shift 2; exec ' +
                            shlex.quote(Path(sys.executable).as_posix()) + ' ' + shlex.quote(client.as_posix()) + ' "$@"; else exec ' +
                            shlex.quote(Path(sys.executable).as_posix()) + ' "$@"; fi')
        self.env["KIT_PYTHON"] = wrapper.as_posix()
        prompts = self.d / "prompts"
        write(prompts / "prose.txt", "fixture prompt")
        peer = "bash -c 'echo READY; sleep 12'"
        r = self.serve("pair", '--classes prose --concurrency 1 --prompt-dir %s --peer-cmd %s' % (
            shlex.quote(prompts.as_posix()), shlex.quote(peer)))
        self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-1500:])
        self.assertTrue((self.d / "out/pair/prose-0.json").is_file())
        self.assertEqual(load(self.d / "out/pair/prose-0")["client_rc"], 0)

    def test_peer_inputs_fail_fast(self):
        # The mock peer sees this rank's input root under a second root: an input present only on this rank must
        # stop the run before launch and name the path; inputs present on both ranks pass.
        local, peerfs = self.d / "in", self.d / "peerfs"
        for p in (local / "a.u32", local / "b.u32", peerfs / "a.u32"):
            write(p, "x")
        lb = self.bash("cd %s && pwd" % shlex.quote(local.as_posix())).stdout.strip()
        pb = self.bash("cd %s && pwd" % shlex.quote(peerfs.as_posix())).stdout.strip()
        self.shim("ssh", 'bash -c "$(printf %%s "${@: -1}" | sed "s#%s#%s#g")"' % (lb, pb))

        def check(*args):
            return self.bash("source scripts/bench/common/runlib.sh && kit_peer_inputs %s && echo PASSED" %
                             " ".join(shlex.quote(a) for a in args))
        r = check("a=%s/a.u32" % lb, "d:%s/a.u32:128" % lb, "--batch", "/nonexistent/x.u32:64:8")
        self.assertEqual(r.returncode, 0, r.stderr[-800:])
        self.assertIn("PASSED", r.stdout)
        self.assertIn("peer inputs ok (2 paths)", r.stderr)
        r = check("a=%s/a.u32" % lb, "--batch", "%s/b.u32:64:8" % lb)
        self.assertNotEqual(r.returncode, 0)
        self.assertNotIn("PASSED", r.stdout)
        self.assertIn("missing inputs: ", r.stderr)
        self.assertIn("/b.u32", r.stderr)
        self.assertNotIn("/a.u32", r.stderr)

    def test_minimax_seal_both_slices_and_absolute_rule(self):
        # The host installs NumPy in the user site. Only this test shim relaxes -I;
        # production launchers still require dependencies in their isolated interpreter.
        wrapper = self.shim("gate-python", '[[ $1 == -I ]] && shift; exec ' +
                            shlex.quote(Path(sys.executable).as_posix()) + ' "$@"')
        self.env["KIT_PYTHON"] = wrapper.as_posix()
        out = self.d / "out"
        prompt = self.d / "p.u32"
        prompt.write_bytes(struct.pack("<3I", 0, 0, 0))
        for arm in ("ref", "cand"):
            for rank in (0, 1):
                folder = (out if rank == 0 else self.d / "peer") / (arm + "-r%d" % rank)
                folder.mkdir(parents=True)
                z = [10., 0.] if rank == 0 else ([0., 30.] if arm == "cand" else [0., 0.])
                (folder / ("q-s-tf-r%d.f32" % rank)).write_bytes(struct.pack("<4f", *(z * 2)))
            write(out / (arm + ".env"), "")
            cmd = 'bash scripts/bench/minimax/fast-gate.sh seal --out %s --arm %s --bin %s --rows 2 --sessions %s' % (
                shlex.quote(out.as_posix()), arm, shlex.quote(Path(sys.executable).as_posix()), shlex.quote("s=" + prompt.as_posix()))
            r = self.bash(cmd)
            self.assertEqual(r.returncode, 0, (r.stdout + r.stderr)[-1500:])
            self.assertTrue((out / (arm + "-sealed/manifest-r1.json")).is_file())
        cmd = 'bash scripts/bench/minimax/fast-gate.sh compare --out %s --ref ref --cand cand --sessions %s' % (
            shlex.quote(out.as_posix()), shlex.quote("s=" + prompt.as_posix()))
        r = self.bash(cmd)
        self.assertEqual(r.returncode, 1, (r.stdout + r.stderr)[-1500:])  # rank 1 alone changed
        r = self.bash(cmd + " --dnll-rule floor")
        self.assertEqual(r.returncode, 2)
        self.assertIn("abs only", r.stderr)
        r = self.bash(cmd + " --null missing")
        self.assertEqual(r.returncode, 2)  # unsupported null arms cannot silently omit a slice

    def test_summary_done_publishes_complete_bounded_summary(self):
        out = self.d / "out"
        out.mkdir()
        write(out / "ticket.log", "ticket fixture\n" + "== SKIP fixture\n" * 80 + "kit rc=1\n")
        # Observe the marker's publication point, not merely the state after the call.
        r = self.bash('source scripts/bench/common/runlib.sh; touch() { cp "$KIT_OUT/summary.txt" "$KIT_OUT/at-done"; command touch "$@"; }; kit_summary_done "$KIT_OUT"')
        self.assertEqual(r.returncode, 0, r.stderr[-1000:])
        summary = (out / "summary.txt").read_text()
        self.assertEqual(summary, (out / "at-done").read_text())
        self.assertIn("SKIP", summary)
        self.assertIn("kit rc=1", summary)
        self.assertLessEqual(len(summary.splitlines()), 60)


if __name__ == "__main__":
    unittest.main()
