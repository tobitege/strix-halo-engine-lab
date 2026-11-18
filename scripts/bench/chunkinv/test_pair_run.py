"""Host integration regressions: SSH is a local shell, never a network connection."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
BASH = shutil.which("bash")
SCRIPT = Path(os.environ.get("PAIR_RUN_SCRIPT", ROOT / "scripts/bench/chunkinv/pair_run.sh"))


@unittest.skipUnless(BASH, "bash is required")
class PairRunTests(unittest.TestCase):
    def setUp(self):
        scratch = ROOT / "build-pair-run-tests" / "fixtures"
        scratch.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.local = self.work / "local"
        self.peer = self.work / "peer"
        self.stubs = self.work / "bin"
        for directory in (self.local, self.peer, self.stubs):
            directory.mkdir()
        self.script(self.stubs / "ssh", '''
[[ ${FAIL_SSH:-0} == 0 ]] || exit 255
command=${!#}
if [[ ${FAIL_COLLECTION:-0} == 1 && $command == *'cat --'* ]]; then exit 9; fi
cd -- "$PEER_CWD" || exit 1
exec /usr/bin/sh -c "$command"
''')
        self.script(self.stubs / "setsid", 'exec "$@"\n')
        self.script(self.stubs / "awk", '''
if [[ ${!#} == /proc/meminfo ]]; then printf '12.0'; else exec /usr/bin/awk "$@"; fi
''')
        self.script(self.stubs / "timeout", '''
shift 2
secs=$1; shift
[[ $secs != 0.01 ]] || exit 124
exec "$@"
''')
        self.script(self.stubs / "mktemp", '''
if [[ ${FIXED_RUN:-0} == 1 ]]; then
    path=${2%run.XXXXXXXX}run.fixed
    mkdir -- "$path" && printf '%s\\n' "$path"
else
    exec /usr/bin/mktemp "$@"
fi
''')
        for directory in (self.local, self.peer):
            self.script(directory / "rank ' binary", '''
printf '%s\\0' "$@" "${PAYLOAD:-}" "$PWD" > "$4.args"
sleep 0.2
echo 'prompt 8 tokens, TTFT 1.0 s, prefill 8.0 tok/s'
echo 'state fnv abc123'
[[ ${MISSING_RANK:-none} == "$2" ]] || echo 'tokens fnv def456'
if [[ $2 == 0 ]]; then exit "${EXIT0:-0}"; else exit "${EXIT1:-0}"; fi
''')
            (directory / "model ' dir").mkdir()
        self.env = os.environ.copy()
        # Windows paths are converted once before entering the test shell.
        self.env["PATH"] = str(self.stubs) + os.pathsep + self.env["PATH"]
        self.env["PEER_CWD"] = self.peer.as_posix()

    @staticmethod
    def script(path, body):
        path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8", newline="\n")
        path.chmod(0o755)

    def run_pair(self, *, out="results", extra=(), env=None, secs="5"):
        command = [BASH, SCRIPT.as_posix(),
                   "--bin", "./rank ' binary", "--model", "model ' dir",
                   "--peer", "local-test-only", "--out", out, "--secs", secs]
        command += ["PAYLOAD=space ' quote; $(touch injected-env) `touch injected-tick`\nline",
                    "--", "prompt ' file", "$(touch injected-arg)", "", *extra]
        result = subprocess.run(command, cwd=self.local, env=self.env | (env or {}),
                                capture_output=True, text=True, timeout=30)
        self.detail = result.stdout[-2000:] + result.stderr[-2000:]
        return result.returncode

    def test_existing_roots_survive_and_runs_are_fresh(self):
        for directory in (self.local, self.peer):
            (directory / "results").mkdir()
            (directory / "results/keep").write_text("unrelated", encoding="utf-8")
        for _ in range(2):
            self.assertEqual(self.run_pair(), 0, self.detail)
        for directory in (self.local, self.peer):
            self.assertEqual((directory / "results/keep").read_text(), "unrelated")
            self.assertEqual(len(list((directory / "results").glob("run.*"))), 2)

    def test_peer_collision_is_refused(self):
        collision = self.peer / "results/run.fixed"
        collision.mkdir(parents=True)
        (collision / "keep").write_text("unrelated", encoding="utf-8")
        self.assertNotEqual(self.run_pair(env={"FIXED_RUN": "1"}), 0)
        self.assertEqual((collision / "keep").read_text(), "unrelated")
        self.assertFalse(list(self.local.glob("results/run.*/*.args")))

    def test_relative_paths_and_remote_argument_bytes(self):
        out = "results ' $(touch injected-out)"
        self.assertEqual(self.run_pair(out=out), 0, self.detail)
        args = []
        for rank, directory in enumerate((self.local, self.peer)):
            run = next((directory / out).glob("run.*"))
            values = (run / f"r{rank}.args").read_bytes().split(b"\0")
            self.assertEqual(values[0], b"model ' dir")
            self.assertEqual(values[1], str(rank).encode())
            self.assertEqual(values[4:7], [b"prompt ' file", b"$(touch injected-arg)", b""])
            self.assertTrue(values[3].startswith(b"/"))
            self.assertFalse(list(directory.glob("injected-*")))
            args.append(values[4:-2])
        self.assertEqual(args[0], args[1])

    def test_rank_failures_and_timeout_are_failures(self):
        for env, status in (({"EXIT0": "7"}, "rc 7/0"), ({"EXIT1": "8"}, "rc 0/8")):
            with self.subTest(env=env):
                self.assertNotEqual(self.run_pair(env=env), 0, self.detail)
                self.assertIn(status, self.detail)
        self.assertNotEqual(self.run_pair(env={"FAIL_SSH": "1"}), 0, self.detail)
        self.assertNotEqual(self.run_pair(secs="0.01"), 0, self.detail)
        self.assertIn("rc 124/124", self.detail)

    def test_missing_measurements_or_collection_are_failures(self):
        for env in ({"MISSING_RANK": "0"}, {"MISSING_RANK": "1"}, {"FAIL_COLLECTION": "1"}):
            with self.subTest(env=env):
                self.assertNotEqual(self.run_pair(env=env), 0, self.detail)


if __name__ == "__main__":
    unittest.main()
