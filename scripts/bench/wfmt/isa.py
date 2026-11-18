#!/usr/bin/env python3
"""Conservative ISA gate. Any changed operation/resource stream requires review."""
import argparse
import json
import pathlib
import re
import struct
import subprocess


def normalise(text, symbols):
    lines, selected = [], not symbols
    for line in text.splitlines():
        # Keep resource lines and opcode/modifier text; strip only addresses/encodings.
        symbol = re.search(r"<([^>]+)>:", line)
        cuda_symbol = re.search(r"Function\s*:\s*(\S+)", line)
        if symbol or cuda_symbol:
            name = (symbol or cuda_symbol).group(1)
            selected = not symbols or any(item in name for item in symbols)
            if selected:
                lines.append("FUNCTION " + name)
            continue
        if not selected:
            continue
        line = re.sub(r"^\s*[0-9a-f]+:\s*(?:[0-9a-f]{2}\s+)*", "", line)
        line = re.sub(r"/\*[0-9a-f]+\*/", "", line)
        line = re.sub(r"//\s*[0-9A-Fa-f]+:.*$", "", line)
        line = re.sub(r"\s+", " ", line).strip()
        if line and not re.search(r"file format|Fatbin|^code for|^arch =|^identifier =", line):
            lines.append(line)
    return lines


def symbol_maps(pairs):
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("nonempty explicit symbol-pair array required")
    old, new = {}, {}
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict) or set(pair) != {"old", "new"}:
            raise ValueError("symbol pairs require exactly old and new names")
        a, b = pair["old"], pair["new"]
        if any(not isinstance(name, str) or not name or any(c.isspace() for c in name) for name in (a, b)):
            raise ValueError("full symbol names required")
        if a in old or b in new:
            raise ValueError("symbol map must be bijective")
        old[a] = new[b] = f"WFMT_PAIR_{index:06d}"
    return old, new


def mapped_stream(stream, mapping):
    groups, current = {}, None
    for line in stream:
        if line.startswith("FUNCTION "):
            name = line[len("FUNCTION "):]
            if name in groups:
                raise ValueError("duplicate selected function")
            current = groups[name] = []
        elif current is not None:
            current.append(line)
        else:
            raise ValueError("instruction outside selected function")
    if mapping.keys() - groups.keys():
        raise ValueError("mapped symbol missing from selected functions")
    translated = {}
    replacements = {**mapping, **{key + ".kd": value + ".kd" for key, value in mapping.items()}}
    pattern = re.compile(r"(?<![A-Za-z0-9_.$])(?:" +
                         "|".join(re.escape(key) for key in sorted(replacements, key=len, reverse=True)) +
                         r")(?![A-Za-z0-9_.$])")
    for name, body in groups.items():
        canonical = mapping.get(name, name)
        if canonical in translated:
            raise ValueError("canonical symbol collision")
        translated[canonical] = [pattern.sub(lambda match: replacements[match[0]], line) for line in body]
    # Symbol-only changes may reorder functions, never operations within one.
    return [line for name, body in sorted(translated.items()) for line in ["FUNCTION " + name, *body]]


def mapped_resources(resources, mapping):
    result = {}
    for name, values in resources.items():
        canonical = mapping.get(name, name)
        if canonical in result:
            raise ValueError("canonical resource symbol collision")
        result[canonical] = values
    return result


def hip_image(binary, directory, architecture, objcopy, bundler):
    data = binary.read_bytes()
    if data[:6] == b"\x7fELF\x02\x01" and len(data) >= 20 and struct.unpack_from("<H", data, 18)[0] == 224:
        return binary
    # Linked HIP hosts carry bundled images in .hip_fatbin. Unsupported containers fail closed.
    directory.mkdir(parents=True, exist_ok=True)
    fatbin, device = directory / "fatbin", directory / "device.co"
    subprocess.run([objcopy, "--dump-section", f".hip_fatbin={fatbin}", str(binary)],
                   check=True, capture_output=True)
    listing = subprocess.check_output([bundler, "--list", "--type=o", f"--input={fatbin}"], text=True)
    targets = [line.strip() for line in listing.splitlines()
               if line.strip().startswith("hip-") and architecture in line.split("--")[-1].split(":")]
    if len(targets) != 1:
        raise ValueError("expected exactly one requested HIP device image")
    subprocess.run([bundler, "--unbundle", "--type=o", f"--targets={targets[0]}",
                    f"--input={fatbin}", f"--output={device}"], check=True, capture_output=True)
    return device


def kernel_descriptors(data):
    """Read resource-bearing AMDHSA descriptor bytes; omit the code-address offset."""
    if data[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<H", data, 18)[0] != 224:
        raise ValueError("expected AMDGPU little-endian ELF64")
    offset = struct.unpack_from("<Q", data, 40)[0]
    size, count = struct.unpack_from("<HH", data, 58)
    if size != 64 or not count:
        raise ValueError("missing ELF section table")
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, offset + i * size) for i in range(count)]
    descriptors = {}
    for section in sections:
        if section[1] != 2:  # SHT_SYMTAB
            continue
        if section[9] != 24 or section[5] % 24:
            raise ValueError("invalid ELF symbol table")
        strings = sections[section[6]]
        names = data[strings[4]:strings[4] + strings[5]]
        for pos in range(section[4], section[4] + section[5], 24):
            name, _, _, index, value, length = struct.unpack_from("<IBBHQQ", data, pos)
            symbol = names[name:].split(b"\0", 1)[0].decode()
            if not symbol.endswith(".kd"):
                continue
            owner = sections[index]
            start = owner[4] + value - owner[3]
            if length != 64 or start < owner[4] or start + 64 > owner[4] + owner[5] or start + 64 > len(data):
                raise ValueError("invalid kernel descriptor")
            if symbol in descriptors:
                raise ValueError("duplicate kernel descriptor")
            descriptors[symbol] = (data[start:start + 12] + data[start + 44:start + 64]).hex()
    return descriptors


def hip_resources(notes, data, symbols):
    starts = list(re.finditer(r"(?m)^( *)- \.[a-z_]+:", notes))
    if not starts:
        raise ValueError("missing AMDGPU kernel metadata")
    indent = min(len(match[1]) for match in starts)
    starts = [match for match in starts if len(match[1]) == indent]
    descriptors = kernel_descriptors(data)
    resources = {}
    keys = ("group_segment_fixed_size", "private_segment_fixed_size", "vgpr_count", "sgpr_count",
            "kernarg_segment_size", "kernarg_segment_align", "wavefront_size", "max_flat_workgroup_size")
    for index, start in enumerate(starts):
        entry = notes[start.start():starts[index + 1].start() if index + 1 < len(starts) else len(notes)]
        def field(key):
            found = re.findall(r"(?m)^ {" + str(indent + 2) + r"}\." + re.escape(key) + r":\s*([^\n]+)", entry)
            if len(found) != 1:
                raise ValueError(f"missing or duplicate HIP metadata: {key}")
            return found[0].strip().strip("\"'")
        # The first key of a YAML entry may be .name instead of .args.
        entry = re.sub(r"(?m)^( *)- ", r"\1  ", entry, count=1)
        name = field("name")
        if not any(symbol in name for symbol in symbols):
            continue
        if name in resources:
            raise ValueError("duplicate kernel metadata")
        values = {key: int(field(key)) for key in keys}
        descriptor = field("symbol")
        if descriptor not in descriptors:
            raise ValueError(f"missing resource descriptor: {descriptor}")
        values["descriptor"] = descriptors[descriptor]
        resources[name] = values
    if any(not any(symbol in name for name in resources) for symbol in symbols):
        raise ValueError("required kernel resource metadata missing")
    return resources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", required=True, type=pathlib.Path)
    parser.add_argument("--new", required=True, type=pathlib.Path)
    parser.add_argument("--backend", required=True, choices=("hip", "cuda"))
    parser.add_argument("--symbols", required=True, type=pathlib.Path,
                        help="JSON array of required symbol substrings")
    parser.add_argument("--symbol-map", type=pathlib.Path,
                        help="explicit bijective [{old: full_symbol, new: full_symbol}] name pairs")
    parser.add_argument("--out", required=True, type=pathlib.Path)
    parser.add_argument("--tool", help="llvm-objdump or cuobjdump path")
    parser.add_argument("--readelf", default="llvm-readelf")
    parser.add_argument("--objcopy", default="llvm-objcopy")
    parser.add_argument("--bundler", default="clang-offload-bundler")
    parser.add_argument("--architecture", default="gfx1151")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"schema": "wfmt.isa@1", "pass": False, "backend": args.backend}
    try:
        symbols = json.loads(args.symbols.read_text())
        if not isinstance(symbols, list) or not symbols or not all(isinstance(s, str) and s for s in symbols):
            raise ValueError("nonempty required symbol array expected")
        mappings = None
        if args.symbol_map:
            pairs = json.loads(args.symbol_map.read_text())
            mappings = symbol_maps(pairs)
            summary["symbol_pairs"] = pairs
        tool = args.tool or ("llvm-objdump" if args.backend == "hip" else "cuobjdump")
        options = (["--disassemble", "--mcpu=" + args.architecture] if args.backend == "hip" else
                   ["--dump-sass", "--dump-resource-usage"])
        streams = []
        for index, (name, binary) in enumerate((("old", args.old), ("new", args.new))):
            resources = None
            if args.backend == "hip":
                binary = hip_image(binary.resolve(), args.out.resolve() / (name + ".device"),
                                   args.architecture, args.objcopy, args.bundler)
                notes = subprocess.check_output([args.readelf, "--notes", str(binary)], text=True,
                                                stderr=subprocess.STDOUT)
                (args.out / (name + ".metadata.txt")).write_text(notes)
                resources = hip_resources(notes, binary.read_bytes(), symbols)
            output = subprocess.check_output([tool] + options + [str(binary)], text=True,
                                             stderr=subprocess.STDOUT)
            (args.out / (name + ".isa.log")).write_text(output)
            if any(symbol not in output for symbol in symbols):
                raise ValueError(f"required symbol missing in {name} disassembly")
            stream = normalise(output, symbols)
            if not stream or any(not any(line.startswith("FUNCTION ") and symbol in line for line in stream)
                                 for symbol in symbols):
                raise ValueError(f"empty {name} operation stream")
            if mappings is not None:
                stream = mapped_stream(stream, mappings[index])
                if resources is not None:
                    resources = mapped_resources(resources, mappings[index])
            if resources is not None:
                stream.append("RESOURCES " + json.dumps(resources, sort_keys=True))
            (args.out / (name + ".normalised.txt")).write_text("\n".join(stream) + "\n")
            streams.append(stream)
        summary["old_bytes"], summary["new_bytes"] = args.old.stat().st_size, args.new.stat().st_size
        summary["pass"] = streams[0] == streams[1]
        if not summary["pass"]:
            summary["error"] = "instruction/resource stream changed: explanation, byte gate and performance receipts required"
    except (ValueError, OSError, IndexError, struct.error, subprocess.SubprocessError) as error:
        summary["error"] = str(error)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
