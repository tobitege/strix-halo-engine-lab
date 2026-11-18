#!/usr/bin/env python3
"""Build the identical legacy API capture adapter against two supplied source refs.
Run only inside the deployment's authorized CPU build lane. No device work.
"""
import argparse
import hashlib
import json
import pathlib
import platform
import re
import subprocess
import sys


def logged(command, path, required=True):
    with path.open("w") as log:
        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False).returncode
    if required and code:
        raise ValueError(f"command exited {code}; see {path.name}")
    return code


def verify_source(source, revision):
    """Require a clean checkout of the requested commit, not a receipt label."""
    def git(*args):
        result = subprocess.run(["git", "-C", str(source), *args],
                                capture_output=True, text=True, check=False)
        if result.returncode:
            raise ValueError("cannot verify source: " + result.stderr.strip())
        return result.stdout.strip()

    if pathlib.Path(git("rev-parse", "--show-toplevel")).resolve() != source.resolve():
        raise ValueError("source must be a checkout root")
    commit = git("rev-parse", "--verify", "--end-of-options", revision + "^{commit}")
    if git("rev-parse", "HEAD") != commit:
        raise ValueError("source HEAD does not match requested revision: " + revision)
    # Also reject staged changes, untracked inputs and dirty submodules. Requiring
    # the entire checkout clean avoids guessing the build's transitive inputs.
    if git("status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none"):
        raise ValueError("source checkout has uncommitted changes")
    return {"revision": revision, "commit": commit, "tree": git("rev-parse", commit + "^{tree}")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-source", type=pathlib.Path, required=True)
    parser.add_argument("--candidate-source", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--build-root", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--baseline-build", type=pathlib.Path)
    parser.add_argument("--candidate-build", type=pathlib.Path)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--verify-session", action="store_true")
    args = parser.parse_args()
    kit = pathlib.Path(__file__).resolve().parent
    matrix = args.candidate_source / "tests/wfmt" / ("legacy-matrix-windows.json" if platform.system() == "Windows" else "legacy-matrix.json")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "wfmt.legacy-wrappers@1", "pass": False,
               "baseline_revision": args.baseline_revision, "candidate_revision": args.candidate_revision,
               "architecture": platform.machine(), "flags": "Release; strict reference FP", "sources": {}}
    try:
        sources = (("baseline", args.baseline_source, args.baseline_revision),
                   ("candidate", args.candidate_source, args.candidate_revision))
        # Verify both trees before either configure can run.
        for mode, source, revision in sources:
            summary["sources"][mode] = verify_source(source, revision)
        for mode, source, _ in sources:
            build = (args.baseline_build if mode == "baseline" else args.candidate_build) or args.build_root / mode
            logged(["cmake", "-S", str(source), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                    "-DBUILD_TESTING=ON", "-DSTRIX_ENABLE_HIP_RUNTIME=OFF", "-DSTRIX_FAMILY_MINIMAX=ON",
                    "-DSTRIX_FAMILY_GLM5=ON", "-DSTRIX_FAMILY_GEMMA4=ON", "-DSTRIX_FAMILY_H3=ON", "-DSTRIX_WFMT_TESTS=ON",
                    "-DCMAKE_PROJECT_INCLUDE=" + str(kit / "legacy_capture.cmake")], args.output / f"{mode}-configure.log")
            targets = ["strix-wfmt-legacy-capture"]
            if args.verify_session:
                targets.append("strix-minimax-session-tests")
            logged(["cmake", "--build", str(build), "--config", "Release", "-j", str(args.jobs), "--target", *targets],
                   args.output / f"{mode}-build.log")
            candidates = [build / "strix-wfmt-legacy-capture", build / "Release/strix-wfmt-legacy-capture.exe"]
            executable = next((p for p in candidates if p.exists()), None)
            if executable is None:
                raise ValueError("legacy capture executable missing")
            cache = (build / "CMakeCache.txt").read_text()
            compiler_line = next(line for line in cache.splitlines() if line.startswith("CMAKE_CXX_COMPILER:"))
            compiler_info = next(build.glob("CMakeFiles/*/CMakeCXXCompiler.cmake")).read_text()
            compiler_version = re.search(r'set\(CMAKE_CXX_COMPILER_VERSION "([^"]+)"\)', compiler_info).group(1)
            flags = {line.split("=", 1)[0]: line.split("=", 1)[1] for line in cache.splitlines()
                     if line.startswith(("CMAKE_CXX_FLAGS:STRING=", "CMAKE_CXX_FLAGS_RELEASE:STRING="))}
            summary["sources"][mode].update({"compiler": compiler_line.split("=", 1)[1],
                                        "compiler_version": compiler_version, "flags": flags,
                                        "binary_sha256": hashlib.sha256(executable.read_bytes()).hexdigest()})
            logged([sys.executable, str(kit / "capture.py"), "--exe", str(executable), "--matrix", str(matrix),
                    "--out", str(args.output / mode), "--compiler", compiler_line.split("=", 1)[1],
                    "--architecture", platform.machine(), "--flags", "Release; strict reference FP"],
                   args.output / f"{mode}-capture.log")
            if args.verify_session:
                code = logged(["ctest", "--test-dir", str(build), "-C", "Release", "--output-on-failure",
                               "-R", "^strix-minimax-session-tests$"], args.output / f"{mode}-session.log", False)
                summary[mode + "_session_exit"] = code
                if mode == "candidate" and code:
                    raise ValueError("candidate session regression test failed")
                if mode == "baseline" and code and "duplicate tensor across safetensors shards: local" not in (args.output / "baseline-session.log").read_text():
                    raise ValueError("baseline session test has an unexpected failure")
        for key in ("compiler", "compiler_version", "flags"):
            if summary["sources"]["baseline"][key] != summary["sources"]["candidate"][key]:
                raise ValueError("baseline and candidate compiler qualification differ: " + key)
        logged([sys.executable, str(kit / "compare.py"), "--baseline", str(args.output / "baseline"),
                "--candidate", str(args.output / "candidate"), "--require-bytes", "--out", str(args.output / "comparison")],
               args.output / "compare.log")
        comparison = json.loads((args.output / "comparison/summary.json").read_text())
        summary.update({"pass": True, "records": comparison["records"]})
    except (ValueError, OSError, StopIteration) as error:
        summary["error"] = str(error)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "sources"}, sort_keys=True))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
