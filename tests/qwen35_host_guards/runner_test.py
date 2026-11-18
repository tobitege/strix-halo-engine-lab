from pathlib import Path
import os
import shutil
import subprocess
import sys
root = Path(__file__).resolve().parents[2]
work = Path(sys.argv[1]).resolve()
work.mkdir(parents=True, exist_ok=True)
bash = shutil.which("bash")
if not bash:
    raise RuntimeError("bash required for runner regression")
sampler = work / "nvidia-smi"
sampler.write_text("#!/usr/bin/env bash\necho 1000\n")
sampler.chmod(0o755)
runner = root / "scripts/bench/qwen35/run_classes.sh"
for name, code, verdict, expected in [("mismatch", 2, "EXACT: FAIL", False), ("crash", 1, "", False), ("missing", 0, "", False), ("pass", 0, "EXACT: PASS (4 tokens compared, 0 mismatches)", True), ("early-failure", 0, "EXACT: PASS", False)]:
    fake = work / (name + ".sh")
    fake.write_text(f"#!/usr/bin/env bash\necho '{verdict}'\nexit {code}\n")
    if name == "early-failure":
        fake.write_text('#!/usr/bin/env bash\ncase "$*" in *code-0*) echo "EXACT: FAIL"; exit 2;; esac\necho "EXACT: PASS"\n')
    fake.chmod(0o755)
    # PATH conversion belongs to Git Bash on Windows, not to Python's native PATH syntax.
    command = 'PATH="$(cygpath -u "$1"):$PATH"; export PATH; bash "$2" "$3" unused.gguf "$4" 4' if os.name == "nt" else 'PATH="$1:$PATH"; export PATH; bash "$2" "$3" unused.gguf "$4" 4'
    result = subprocess.run([bash, "-c", command, "runner-test", str(work), str(runner), str(fake), str(work / name)], capture_output=True, text=True)
    if (result.returncode == 0) != expected:
        raise RuntimeError(f"{name}: exit {result.returncode}; {result.stderr[-500:]}")
print("runner: mismatch/crash/missing verdict fail; valid verdict passes")
