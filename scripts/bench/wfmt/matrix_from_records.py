#!/usr/bin/env python3
"""Freeze the old executable's observed record inventory before migration."""
import argparse
import json
import pathlib
import sys
from records import read_record, read_matrix, validate_inventory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=pathlib.Path)
    parser.add_argument("--out", required=True, type=pathlib.Path)
    args = parser.parse_args()
    try:
        if args.out.exists():
            raise ValueError("matrix already exists; baseline inventory is immutable")
        cases = []
        for path in sorted(args.baseline.rglob("*.wfmt")):
            metadata, sections = read_record(path)
            case = {key: metadata[key] for key in
                    ("case", "arithmetic", "shape", "format", "image") if key in metadata}
            case["file"] = path.relative_to(args.baseline).as_posix()
            case["required_sections"] = {name: len(data) for name, data in sections.items() if data}
            case["optional_sections"] = {name: 0 for name, data in sections.items() if not data}
            cases.append(case)
        matrix = {"schema": "wfmt.matrix@1", "cases": cases}
        # Validate before publishing the immutable inventory.
        if not cases:
            raise ValueError("baseline contains no capture records")
        validate_inventory(args.baseline, matrix)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as stream:
            stream.write(json.dumps(matrix, indent=2) + "\n")
        read_matrix(args.out)
        print(json.dumps({"pass": True, "records": len(cases)}))
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(json.dumps({"pass": False, "error": str(error)}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
