#!/usr/bin/env python3
import argparse
import json
import pathlib
import tempfile
from records import compare_record, write_record


def self_test():
    with tempfile.TemporaryDirectory() as directory:
        a, b = pathlib.Path(directory) / "a", pathlib.Path(directory) / "b"
        meta = {"case": "self", "shape": [4], "arithmetic": "legacy@1"}
        write_record(a, meta, {"output": b"\x00\x01\x80\xff"})
        write_record(b, meta, {"output": b"\x00\x01\x80\xff"})
        compare_record(a, b)
        for data in (b"\x00\x00\x80\xff", b"\x00\x01\x80\xff\x00"):
            write_record(b, meta, {"output": data})
            try:
                compare_record(a, b)
            except ValueError:
                pass
            else:
                raise ValueError("comparison accepted changed bytes or length")
        # A corrupt payload carrying the original hash must also be rejected.
        b.write_bytes(a.read_bytes()[:-1] + b"\x00")
        try:
            compare_record(a, b)
        except ValueError:
            return
        raise ValueError("comparison trusted stale hash metadata")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=pathlib.Path)
    parser.add_argument("--candidate", type=pathlib.Path)
    parser.add_argument("--require-bytes", action="store_true", help="bytes are always required")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--fixture", type=pathlib.Path, help="write protocol fixture, never a kernel receipt")
    parser.add_argument("--out", type=pathlib.Path)
    args = parser.parse_args()
    summary = {"schema": "wfmt.compare@1", "pass": False, "mode": "protocol-comparison"}
    try:
        if args.fixture:
            args.fixture.mkdir(parents=True, exist_ok=True)
            write_record(args.fixture / "protocol.wfmt",
                         {"case": "protocol-fixture", "arithmetic": "legacy-fixture@1", "shape": [4]},
                         {"activation_codes": bytes([127, 0, 255, 128]), "output": bytes([1, 2, 3, 4]),
                          "source": bytes([0, 1, 128, 255])})
            summary.update({"pass": True, "fixture_only": True})
        elif args.self_test:
            self_test()
            summary.update({"pass": True, "self_test": True})
        else:
            if args.baseline is None or args.candidate is None:
                parser.error("--baseline and --candidate are required")
            old = {p.relative_to(args.baseline): p for p in args.baseline.rglob("*.wfmt")}
            new = {p.relative_to(args.candidate): p for p in args.candidate.rglob("*.wfmt")}
            if not old or old.keys() != new.keys():
                raise ValueError("empty or different record inventories")
            receipts = {str(key): compare_record(old[key], new[key]) for key in sorted(old)}
            summary.update({"pass": True, "records": len(old), "receipts": receipts})
    except (ValueError, OSError, KeyError, TypeError) as error:
        summary["error"] = str(error)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "receipts"}, sort_keys=True))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
