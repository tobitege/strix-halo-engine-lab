#!/usr/bin/env python3
"""Run one bounded mandatory CTest group, then compare supplied byte captures."""
import argparse
from collections import Counter
import hashlib
import json
import pathlib
import re
import shlex
import subprocess
import sys
import xml.etree.ElementTree as ET
from records import read_matrix, validate_inventory

BACKENDS = ("host", "hip", "cuda-qwen", "cuda-ds41", "cpu", "intel", "nvfp4")
COMMON = {
    "P01": ("wfmt.contracts", "wfmt.census", "wfmt.compare", "wfmt.tools"),
    "P02": ("wfmt.reference", "wfmt.geometry", "wfmt.iq3s"),
    "P03": ("wfmt.readers", "wfmt.admission"),
    "P04": ("wfmt.resolve", "wfmt.cache", "wfmt.effective-config"),
    "P05": ("wfmt.cpu",),
    "P06": ("wfmt.hip-dense",),
    "P07": ("wfmt.hip-gsq",),
    "P08": ("wfmt.hip-trellis",),
}
HIP_REQUIRED = {
    "P06": tuple("strix-hip-" + part + "-tests" for part in
                 ("dense-a8-wmma", "gsq-dense-a8", "gsq-typed-sites", "dense-f16-tiled", "ggml-weight-formats")),
    "P07": tuple("strix-hip-" + part + "-tests" for part in
                 ("moe-prefill-gsq", "q20-a8-probe", "gsq-dense-a8", "gsq-typed-sites")),
    "P08": ("strix-trellis-codec-tests", "strix-hip-trellis-codec-tests",
            "strix-ds41-trellis-identity-tests", "strix-glm-trellis-identity-tests",
            "strix-moe-grouped-prefill-host-tests"),
}


def required_tests(pr, backend):
    if backend not in BACKENDS:
        raise ValueError("unknown backend")
    if pr in HIP_REQUIRED and backend != "hip":
        raise ValueError(pr + " requires the HIP backend")
    names = set(COMMON[pr])
    names.update(HIP_REQUIRED.get(pr, ()))
    if pr == "P02" and backend in ("host", "hip", "cuda-ds41"):
        names.update("strix-ds41-" + part + "-tests" for part in ("formats", "native-fp8", "trellis-identity"))
    if pr == "P02" and backend in ("host", "nvfp4"):
        names.update(("nvfp4_host", "nvfp4_tolerance_host"))
    if pr == "P05" and backend in ("host", "intel"):
        names.update(("wfmt.intel", "strix-intel-backend-host-tests",
                      "strix-intel-dense-boundaries", "strix-intel-memory-boundaries"))
    return names


def validate_tests(root, required):
    cases = list(root.iter("testcase"))
    if Counter(case.get("name") for case in cases) != Counter(required):
        raise ValueError("missing, duplicate or unexpected mandatory cases")
    if any(case.get("status") != "run" or any(case.find(tag) is not None for tag in
           ("skipped", "failure", "error")) for case in cases):
        raise ValueError("unsuccessful mandatory cases")
    return len(cases)


def capture_receipt(directory):
    receipt = json.loads((directory / "summary.json").read_text())
    if receipt.get("schema") != "wfmt.capture@2" or receipt.get("pass") is not True:
        raise ValueError("successful version 2 capture receipt required")
    for field in ("compiler", "architecture", "device", "platform"):
        if not isinstance(receipt.get(field), str) or not receipt[field].strip():
            raise ValueError(f"missing capture environment: {field}")
    if not isinstance(receipt.get("flags"), str) or not isinstance(receipt.get("adapter_args"), list):
        raise ValueError("missing capture flags/arguments")
    if not all(isinstance(arg, str) for arg in receipt["adapter_args"]):
        raise ValueError("invalid capture arguments")
    if not re.fullmatch(r"[0-9a-f]{64}", receipt.get("executable_sha256", "")):
        raise ValueError("missing executable identity")
    matrix_path = directory / "matrix.json"
    if hashlib.sha256(matrix_path.read_bytes()).hexdigest() != receipt.get("matrix_sha256"):
        raise ValueError("capture matrix receipt changed")
    matrix = read_matrix(matrix_path)
    if any(case["case"] == "protocol-fixture" for case in matrix["cases"]):
        raise ValueError("protocol fixtures require --protocol-fixtures-only")
    records = validate_inventory(directory, matrix)
    if records != receipt.get("record_sha256") or receipt.get("records") != len(records):
        raise ValueError("capture record receipt changed")
    return receipt


def validate_captures(baseline, candidate):
    old, new = capture_receipt(baseline), capture_receipt(candidate)
    for field in ("matrix_sha256", "compiler", "flags", "architecture", "device", "platform", "adapter_args"):
        if old[field] != new[field]:
            raise ValueError(f"capture workload/environment differs: {field}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", required=True, type=pathlib.Path)
    parser.add_argument("--baseline", required=True, type=pathlib.Path)
    parser.add_argument("--candidate", required=True, type=pathlib.Path)
    parser.add_argument("--backend", required=True, choices=BACKENDS)
    parser.add_argument("--pr", required=True, choices=COMMON)
    parser.add_argument("--protocol-fixtures-only", action="store_true",
                        help="compare protocol files without qualification or capture receipts")
    parser.add_argument("--runner", default="")
    parser.add_argument("--partition-tests", action="store_true",
                        help="run each mandatory test in its own bounded runner ticket")
    parser.add_argument("--output", required=True, type=pathlib.Path)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    if args.timeout < 1 or args.timeout > 120:
        parser.error("partition cases into tickets of at most 120 seconds")
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "wfmt.run@2", "pr": args.pr, "backend": args.backend, "pass": False,
               "mode": "protocol-fixtures" if args.protocol_fixtures_only else "qualification"}
    junit = args.output.resolve() / "ctest.xml"
    try:
        required = required_tests(args.pr, args.backend)
        summary["required_tests"] = sorted(required)
        summary["partition_tests"] = args.partition_tests
        if args.runner and not args.partition_tests:
            raise ValueError("a queued runner requires --partition-tests for bounded tickets")
        junit.unlink(missing_ok=True)
        partitions = [{name} for name in sorted(required)] if args.partition_tests else [required]
        root = ET.Element("testsuite")
        for index, partition in enumerate(partitions, 1):
            stem = f"ctest-{index:03d}" if args.partition_tests else "ctest"
            part_junit = args.output.resolve() / (stem + ".xml")
            part_junit.unlink(missing_ok=True)
            command = shlex.split(args.runner) + ["ctest", "--test-dir", str(args.build.resolve()),
                "-C", "Release", "--output-on-failure", "--no-tests=error", "--timeout", str(args.timeout),
                "--output-junit", str(part_junit),
                "-R", "^(" + "|".join(re.escape(name) for name in sorted(partition)) + ")$"]
            with (args.output / (stem + ".log")).open("w") as log:
                # The runner owns the FIFO wait and the ticket's process budget.
                # CTest bounds its one test inside the ticket, not while queued.
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=None if args.runner else args.timeout)
            if result.returncode:
                raise ValueError(f"mandatory CTest partition {index} failed: {result.returncode}")
            part_root = ET.parse(part_junit).getroot()
            validate_tests(part_root, partition)
            root.extend(part_root.iter("testcase"))
        count = validate_tests(root, required)
        ET.ElementTree(root).write(junit, encoding="utf-8", xml_declaration=True)
        if not args.protocol_fixtures_only:
            validate_captures(args.baseline, args.candidate)
        command = [sys.executable, str(pathlib.Path(__file__).with_name("compare.py")),
                   "--baseline", str(args.baseline), "--candidate", str(args.candidate),
                   "--require-bytes", "--out", str(args.output / "compare")]
        with (args.output / "compare.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        if result.returncode:
            raise ValueError("byte comparison failed")
        summary.update({"pass": True, "tests": count})
    except (ValueError, OSError, KeyError, TypeError, AttributeError, ET.ParseError, subprocess.SubprocessError) as error:
        summary["error"] = str(error)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
