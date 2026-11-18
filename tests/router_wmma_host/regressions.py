"""Host regressions; subprocesses give startup-cached policy readers fresh state."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def hello(executable):
    def fingerprint(entry, rank, policy):
        env = {k: v for k, v in os.environ.items() if not k.startswith("STRIX_GLM_PF_")}
        env.update(STRIX_GLM_PF_WMMA="1", STRIX_GLM_PF_KDA_WMMA="0", STRIX_ROUTE_RECEIPT="0")
        if policy is not None:
            env["STRIX_GLM_PF_ROUTER_WMMA"] = policy
        return subprocess.check_output([executable, "fingerprint", entry, str(rank)], env=env, text=True).strip()

    off, on = fingerprint("http", 0, "0"), fingerprint("http", 1, "1")
    assert off == fingerprint("http", 1, None), "unset must resolve to OFF"
    assert off == fingerprint("cli", 1, "0"), "CLI/HTTP must share the OFF route"
    assert on == fingerprint("cli", 0, "1"), "CLI/HTTP must share the ON route"
    subprocess.run([executable, "hello", off, on], check=True)


def wrapper(source, work):
    bash = shutil.which("bash")
    assert bash, "Bash is required for the wrapper regression"
    script = (Path(source) / "scripts/bench/router-wmma/unit.sh").resolve().as_posix()
    names = ("strix-hip-glm5-pf-router-wmma-tests", "strix-hip-mm-router-wmma-tests")
    scenarios = [("pass", (0, 0), (0, 0), False), ("missing-all", None, None, False),
                 ("missing-one", (0, 0), None, False), ("glm-gate", (1, 0), (0, 0), False),
                 ("glm-time", (0, 1), (0, 0), False), ("mm-gate", (0, 0), (1, 0), False),
                 ("mm-time", (0, 0), (0, 1), False), ("trace-failure", (0, 0), (0, 0), True)]
    errors = []
    with tempfile.TemporaryDirectory(prefix="wrapper-", dir=work) as temporary:
        for label, glm, mm, trace in scenarios:
            root = Path(temporary) / label
            build, out = root / "build", root / "out"
            build.mkdir(parents=True)
            out.mkdir()
            marker = out / "unit.done"
            marker.touch()  # every failing case must remove stale success
            for name, statuses in zip(names, (glm, mm)):
                if statuses is None:
                    continue
                binary = build / name
                binary.write_text('#!/usr/bin/env bash\nif [ "${1:-}" = --timing ]; then\n'
                                  f'  echo TIME; exit {statuses[1]}\nfi\necho PASS; exit {statuses[0]}\n', encoding="utf-8")
                binary.chmod(0o755)
            env = os.environ.copy()
            if trace:
                profiler = build / "rocprofv3"
                profiler.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
                profiler.chmod(0o755)
                env["PATH"] = str(build.resolve()) + os.pathsep + env["PATH"]
            result = subprocess.run([bash, script, build.resolve().as_posix(), out.resolve().as_posix()] +
                                    (["--trace"] if trace else []), env=env, capture_output=True, text=True)
            success = label == "pass"
            if (result.returncode == 0) != success or marker.exists() != success:
                errors.append(f"{label}: exit={result.returncode}, marker={marker.exists()}")
    assert not errors, "; ".join(errors)


if __name__ == "__main__":
    if sys.argv[1] == "hello":
        hello(sys.argv[2])
    else:
        wrapper(*sys.argv[2:])
