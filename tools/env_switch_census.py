#!/usr/bin/env python3
"""Census and ratchet of STRIX_* environment switches.

Every switch read from the environment is one of:
  governed  read through include/strix/env_flags.hpp (envcfg::flag/integer/text/choice, env_flag_default_on/off,
            env_bytes, env_choice, ...) or declared in a lever/profile table: validated, and listed by
            --print-effective-config;
  raw       read with getenv at its use site: unvalidated, unlisted, default known only to that line.

--check (the ctest) fails when
  * a switch is read raw that is not in the baseline (tools/env_switch_baseline.txt): a new switch must be read
    through env_flags.hpp or declared in a table, so the raw set can only shrink;
  * one boolean switch is read through the governed helpers with two different defaults.
Baseline names that are no longer read raw are reported and removed with --update (never added by hand).
--grandfather NAME... explicitly admits pre-introduction raw reads; never use it in the CMake test.

--self-test runs the scanner on its fixtures (positive and negative).
--report prints the counts; --tsv FILE writes one row per switch (name, class, read sites, documented).
The scan is lexical: comments are removed, white space and new lines between the tokens of a call are allowed.
A switch whose name is built at run time or passed through a wrapper is not seen. The baseline governs names,
not read sites: a baselined switch may gain another raw read.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

SOURCE_DIRS = ("src", "include", "apps")
SOURCE_EXT = {".cpp", ".hpp", ".h", ".hip", ".inc", ".inl", ".cu", ".cuh"}
DOC_PATHS = ("docs", "README.md", "AGENTS.md", "models", "recipes")
DOC_EXT = {".md", ".toml", ".txt", ".json", ".sh"}
HELPER_FILE = "include/strix/env_flags.hpp"

# Lexical pass: comments become one space, string and character literals are kept. The patterns below then run on
# code only, and allow any white space (including new lines) between the tokens of a call.
LEXEME = re.compile(
    r"""//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|'(?:\\[^\n]{1,8}?|[^'\\\n])'""", re.S)
STRING = r'"(?:\\.|[^"\\\n])*"'
# Consume the entire literal argument: a baselined prefix must not hide a new suffix.
RAW = re.compile(r'\bgetenv\s*\(\s*((?:' + STRING + r'\s*)+)\)')
GOVERNED = re.compile(
    r'\b(envcfg\s*::\s*[a-z_]+|env_[a-z_0-9]+)\s*(?:<[^;{}()]*>\s*)?'
    r'\(\s*"(STRIX_[A-Z0-9_]+)"')
TABLE = re.compile(r'\.env\s*=\s*"(STRIX_[A-Z0-9_]+)"')
FLAG_DEFAULT = re.compile(
    r'\benvcfg\s*::\s*flag\s*\(\s*"(STRIX_[A-Z0-9_]+)"\s*,\s*(true|false)\b'
    r'|\benv_flag_default_(on|off)\s*\(\s*"(STRIX_[A-Z0-9_]+)"')
NAME = re.compile(r'STRIX_[A-Z0-9_]+')


def source_files(root: Path):
    for top in SOURCE_DIRS:
        base = root / top
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.suffix in SOURCE_EXT and path.is_file():
                yield path


def documented_names(root: Path) -> set[str]:
    names: set[str] = set()
    for entry in DOC_PATHS:
        path = root / entry
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.suffix in DOC_EXT) if path.is_dir() else []
        for file in files:
            try:
                names.update(NAME.findall(file.read_text(encoding="utf-8", errors="ignore")))
            except OSError:
                pass
    return names


def strip_comments(text: str) -> str:
    return LEXEME.sub(lambda m: " " if m.group(0).startswith(("//", "/*")) else m.group(0), text)


def scan_text(text: str):
    """Switches of one source text: (raw names, governed names, [(name, 'on'|'off')] of governed flag reads)."""
    code = strip_comments(text)
    raw = set()
    for argument in RAW.findall(code):
        # C++ escape decoding is outside this lexical scanner. Fail closed instead
        # of accepting a baselined prefix of an unsupported constant expression.
        if "\\" in argument:
            raise ValueError("unsupported escape in raw getenv literal; use an ordinary literal name")
        name = "".join(token[1:-1] for token in re.findall(STRING, argument))
        if NAME.fullmatch(name):
            raw.add(name)
    governed = {name for helper, name in GOVERNED.findall(code) if helper != "env_reject"}
    governed.update(TABLE.findall(code))
    defaults = []
    for a_name, a_default, b_default, b_name in FLAG_DEFAULT.findall(code):
        defaults.append((a_name or b_name, "on" if (a_default == "true" or b_default == "on") else "off"))
    return raw, governed, defaults


def scan(root: Path):
    raw: dict[str, set[str]] = defaultdict(set)
    governed: dict[str, set[str]] = defaultdict(set)
    defaults: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    inspected = 0
    for path in source_files(root):
        rel = path.relative_to(root).as_posix()
        inspected += 1
        if rel == HELPER_FILE:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "STRIX_" not in text:
            continue
        try:
            file_raw, file_governed, file_defaults = scan_text(text)
        except ValueError as error:
            raise ValueError(f"{rel}: {error}") from error
        for name in file_raw:
            raw[name].add(rel)
        for name in file_governed:
            governed[name].add(rel)
        for name, default in file_defaults:
            defaults[name][default].add(rel)
    return raw, governed, defaults, inspected


# Fixtures for --self-test: (source text, raw names, governed names, flag defaults).
FIXTURES = [
    ('const char* v = std::getenv("STRIX_A");', {"STRIX_A"}, set(), []),
    ('auto v = std::getenv ("STRIX_A");', {"STRIX_A"}, set(), []),
    ('auto v = getenv\n    (\n  "STRIX_A"\n);', {"STRIX_A"}, set(), []),
    ('auto v = std::getenv /* why */ ( /* name */ "STRIX_A");', {"STRIX_A"}, set(), []),
    ('auto v = ::getenv(// name\n "STRIX_A");', {"STRIX_A"}, set(), []),
    ('auto v = std::getenv("STRIX_BASELINE" "_NEW");', {"STRIX_BASELINE_NEW"}, set(), []),
    ('auto v = getenv("STRIX_" /* join */ "A" "");', {"STRIX_A"}, set(), []),
    ('// old: getenv("STRIX_A")\nint x = 0;', set(), set(), []),
    ('/* getenv("STRIX_A")\n   getenv("STRIX_B") */ int x = 0;', set(), set(), []),
    ('const char* doc = "see // getenv"; auto v = getenv("STRIX_A");', {"STRIX_A"}, set(), []),
    ('const char* doc = "/* open"; auto v = getenv("STRIX_A"); /* getenv("STRIX_B") */', {"STRIX_A"}, set(), []),
    ('char q = \'"\'; auto v = getenv("STRIX_A"); // getenv("STRIX_B")', {"STRIX_A"}, set(), []),
    ('auto v = my_getenv("STRIX_A"); auto w = getenv_or("STRIX_B");', set(), set(), []),
    ('auto v = getenv(name); auto w = getenv("OTHER_A"); auto x = getenv("strix_a");', set(), set(), []),
    ('const char* s = "getenv(\\"STRIX_A\\")";', set(), set(), []),
    ('bool on = envcfg::flag("STRIX_A", true);', set(), {"STRIX_A"}, [("STRIX_A", "on")]),
    ('bool on = strix::envcfg :: flag (\n "STRIX_A" ,\n false );', set(), {"STRIX_A"}, [("STRIX_A", "off")]),
    ('bool on = envcfg::flag /* x */ ("STRIX_A", /* y */ true);', set(), {"STRIX_A"}, [("STRIX_A", "on")]),
    ('bool on = env_flag_default_off ("STRIX_A");', set(), {"STRIX_A"}, [("STRIX_A", "off")]),
    ('strix::env_number<std::uint32_t>("STRIX_PROBE_DEPTH", 4U);', set(), {"STRIX_PROBE_DEPTH"}, []),
    ('strix::env_number<std::uint32_t>("STRIX_PROBE_GPU_LINKS", 8U);', set(), {"STRIX_PROBE_GPU_LINKS"}, []),
    ('strix::env_number<unsigned>("STRIX_PROBE_PAUSE_US", 0U);', set(), {"STRIX_PROBE_PAUSE_US"}, []),
    ('strix::env_number<double>("STRIX_PROBE_OUTLIER_US", 10000.0);', set(), {"STRIX_PROBE_OUTLIER_US"}, []),
    ('strix::env_number /* type */ < unsigned >\n ("STRIX_A", 0U);', set(), {"STRIX_A"}, []),
    ('// envcfg::flag("STRIX_A", true)\nint n = envcfg::integer("STRIX_B", 4);', set(), {"STRIX_B"}, []),
    ('{.name = "x", .env\n = "STRIX_A", .default_on = true}', set(), {"STRIX_A"}, []),
    ('detail::env_reject("STRIX_A", value, "why");', set(), set(), []),
    ('auto a = getenv("STRIX_A"); bool b = envcfg::flag("STRIX_A", true);', {"STRIX_A"}, {"STRIX_A"},
     [("STRIX_A", "on")]),
]


def self_test() -> int:
    failed = 0
    for index, (text, raw, governed, defaults) in enumerate(FIXTURES):
        got = scan_text(text)
        if got != (raw, governed, defaults):
            failed += 1
            print(f"env switch census self-test: fixture {index} {text!r}: got {got}, "
                  f"expected {(raw, governed, defaults)}", file=sys.stderr)
    print(f"env switch census self-test: {len(FIXTURES) - failed}/{len(FIXTURES)} fixtures ok")
    return 1 if failed else 0


def read_baseline(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")}


def write_baseline(path: Path, names: set[str], *, exclusive: bool = False) -> None:
    header = ("# Switches read with a raw getenv today (generated: tools/env_switch_census.py --update).\n"
              "# After introduction this list only shrinks. New switches use include/strix/env_flags.hpp or a\n"
              "# lever table (include/strix/lever.hpp). Explicit introduction exceptions use --grandfather.\n")
    with path.open("x" if exclusive else "w", encoding="utf-8", newline="\n") as output:
        output.write(header + "".join(name + "\n" for name in sorted(names)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--baseline", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--update", action="store_true", help="drop baseline names that are no longer read raw")
    mode.add_argument("--grandfather", nargs="+", metavar="NAME",
                      help="explicitly admit named pre-introduction switches currently read raw (not for CI)")
    mode.add_argument("--seed", action="store_true", help="write the baseline from the current tree (first use only)")
    mode.add_argument("--self-test", action="store_true", help="run the scanner on its fixtures")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--tsv", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    baseline_path = args.baseline or root / "tools" / "env_switch_baseline.txt"

    if args.self_test:
        return self_test()
    try:
        raw, governed, defaults, inspected = scan(root)
    except ValueError as error:
        print(f"env switch census: {error}", file=sys.stderr)
        return 2
    # A scan that saw nothing proves nothing: a wrong --root or a missing baseline is an error, not a pass.
    if inspected == 0:
        print(f"env switch census: no source files under {root} ({', '.join(SOURCE_DIRS)})", file=sys.stderr)
        return 2
    if not args.seed and not baseline_path.is_file():
        print(f"env switch census: baseline {baseline_path} not found", file=sys.stderr)
        return 2
    baseline = read_baseline(baseline_path)
    conflicts = {name: sides for name, sides in defaults.items() if len(sides) > 1}
    new_raw = sorted(set(raw) - baseline)
    stale = sorted(baseline - set(raw))

    if args.seed:
        try:
            write_baseline(baseline_path, set(raw), exclusive=True)
        except FileExistsError:
            print(f"env switch census: baseline {baseline_path} already exists; use --update to remove names",
                  file=sys.stderr)
            return 2
        print(f"env switch census: baseline seeded with {len(raw)} raw switches")
        return 0
    if args.grandfather:
        requested = set(args.grandfather)
        invalid = sorted(requested - set(raw))
        if invalid:
            print(f"env switch census: cannot grandfather names not currently read raw: {', '.join(invalid)}",
                  file=sys.stderr)
            return 2
        write_baseline(baseline_path, baseline | requested)
        for name in sorted(requested - baseline):
            print(f"env switch census: grandfathered {name} ({', '.join(sorted(raw[name]))})")
        return 0
    if args.update:
        write_baseline(baseline_path, baseline & set(raw))
        print(f"env switch census: baseline {len(baseline)} -> {len(baseline & set(raw))} (removed {len(stale)})")
        return 0

    if args.report or args.tsv:
        documented = documented_names(root)
        names = sorted(set(raw) | set(governed))
        only_governed = [n for n in names if n not in raw]
        multi = [n for n in names if len(raw.get(n, set()) | governed.get(n, set())) > 1]
        if args.report:
            print(f"switches read: {len(names)}")
            print(f"  governed only: {len(only_governed)}")
            print(f"  raw getenv: {len(raw)} (of which also governed somewhere: {len(set(raw) & set(governed))})")
            print(f"  read in more than one file: {len(multi)}")
            print(f"  documented (docs/, README, AGENTS, models/, recipes/): {len([n for n in names if n in documented])}")
            print(f"  boolean switches with two governed defaults: {len(conflicts)}")
        if args.tsv:
            rows = ["name\tclass\tfiles\tdocumented\n"]
            for n in names:
                files = sorted(raw.get(n, set()) | governed.get(n, set()))
                rows.append(f"{n}\t{'raw' if n in raw else 'governed'}\t{len(files)}\t{'yes' if n in documented else 'no'}\n")
            args.tsv.write_text("".join(rows), encoding="utf-8", newline="\n")

    if not args.check:
        return 0
    failed = False
    for name in new_raw:
        failed = True
        where = ", ".join(sorted(raw[name])[:3])
        print(f"env switch census: {name} is read with a raw getenv ({where}). Read a new switch through "
              f"include/strix/env_flags.hpp (envcfg::flag/integer/text/choice) or declare it in a lever table.",
              file=sys.stderr)
    for name, sides in sorted(conflicts.items()):
        failed = True
        on = ", ".join(sorted(sides["on"])[:2])
        off = ", ".join(sorted(sides["off"])[:2])
        print(f"env switch census: {name} is read default-on ({on}) and default-off ({off}).", file=sys.stderr)
    if stale:
        print(f"env switch census: {len(stale)} baseline switches are no longer read raw; "
              f"run tools/env_switch_census.py --update", file=sys.stderr)
    if failed:
        return 1
    print(f"env switch census: ok ({len(raw)} raw switches, none new; {len(set(governed) - set(raw))} governed)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
