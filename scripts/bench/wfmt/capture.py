#!/usr/bin/env python3
"""Run a test-only capture adapter, retaining provenance and its raw records."""
import argparse
import hashlib
import json
import os
import pathlib
import platform
import re
import shlex
import subprocess
import sys
from records import read_matrix, validate_inventory, read_record


def source_revision(value):
    if not re.fullmatch(r"[0-9a-fA-F]{40}", value):
        raise argparse.ArgumentTypeError("source revision must be a full 40-hex commit ID")
    return value.lower()


def source_identity(source, source_ref):
    if source_ref is not None:
        if (len(source_ref) not in (40, 64) or
                any(c not in "0123456789abcdefABCDEF" for c in source_ref)):
            raise ValueError("source-ref must be a full 40- or 64-hex commit")
        if source is None or not source.is_dir():
            raise ValueError("source-ref requires an existing --source directory")
        return source_ref.lower()
    if source is not None:
        return subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", required=True, type=pathlib.Path)
    parser.add_argument("--matrix", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--source", type=pathlib.Path)
    parser.add_argument("--source-ref", help="full commit identity for an archived --source tree")
    parser.add_argument("--source-revision", type=source_revision,
                        help="commit ID for an externally frozen source archive (no --source)")
    parser.add_argument("--adapter", choices=("protocol", "reference", "native", "backend", "hip-transfer"), default="protocol")
    parser.add_argument("--compiler", required=True)
    parser.add_argument("--architecture", required=True)
    parser.add_argument("--flags", default="")
    parser.add_argument("--device", default="host")
    parser.add_argument("--runner", default="")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if list(args.out.rglob("*.wfmt")):
        parser.error("output already contains records; use a fresh capture directory")
    summary = {"schema": "wfmt.capture@2", "pass": False,
               "compiler": args.compiler, "architecture": args.architecture,
               "adapter": args.adapter, "fixture_only": args.adapter == "protocol",
               "flags": args.flags, "device": args.device, "platform": platform.platform()}
    try:
        exe = args.exe.resolve(strict=True)
        matrix = args.matrix.resolve(strict=True)
        contract = read_matrix(matrix)
        matrix_bytes = matrix.read_bytes()
        summary["executable_sha256"] = hashlib.sha256(exe.read_bytes()).hexdigest()
        summary["matrix_sha256"] = hashlib.sha256(matrix_bytes).hexdigest()
        if args.source_revision and (args.source or args.source_ref):
            raise ValueError("--source-revision excludes --source/--source-ref")
        source = args.source_revision or source_identity(args.source, args.source_ref)
        if source is not None:
            summary["source"] = source
        extra = args.args[1:] if args.args[:1] == ["--"] else args.args
        summary["adapter_args"] = extra
        adapter_args = ["--wfmt-capture", str(args.out.resolve()), "--wfmt-matrix", str(matrix)]
        if args.adapter == "reference":
            adapter_args = ["--wfmt-reference-capture", str(args.out.resolve())]
        elif args.adapter == "backend":
            adapter_args = ["--wfmt-backend-capture", str(args.out.resolve())]
        environment = os.environ.copy()
        if args.adapter == "hip-transfer":
            adapter_args = []
            environment["STRIX_WFMT_CAPTURE_DIR"] = str(args.out.resolve())
        command = shlex.split(args.runner) + [str(exe)] + adapter_args + extra
        with (args.out / "capture.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=args.timeout, check=False, env=environment)
        if result.returncode:
            raise ValueError(f"capture adapter exit {result.returncode}")
        if args.adapter != "protocol":
            for record in args.out.rglob("*.wfmt"):
                header, _ = read_record(record)
                if header["case"] == "protocol-fixture":
                    raise ValueError("protocol fixture cannot qualify a numerical capture")
        (args.out / "matrix.json").write_bytes(matrix_bytes)
        records = validate_inventory(args.out, contract)
        summary.update({"pass": True, "records": len(records), "record_sha256": records})
    except (ValueError, OSError, KeyError, TypeError, AttributeError, subprocess.SubprocessError) as error:
        summary["error"] = str(error)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
