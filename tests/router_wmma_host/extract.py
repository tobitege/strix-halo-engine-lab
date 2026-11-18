"""Compile actual ownership declarations and device-gate host loops with HIP spies.

-- ROUTER_SOURCE may be a saved pre-fix tree. Kernel code is never compiled or
simulated: the spies exercise skipped stores, exact-only and omitted fallback.
"""
import re
import sys
from pathlib import Path

source, output = map(Path, sys.argv[1:])


def read(path):
    return (source / path).read_text(encoding="utf-8")


def block(text, marker):
    start = text.index(marker)
    end = text.index("{", start) + 1
    depth = 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


engine = read("src/minimax/engine/engine.hip")
core = read("src/core/f16_dense_tall.hpp")
control = read("src/glm/serve/glm_serve_control.hpp")
pieces = {
    "OWNER": (block(core, "struct DeviceStorageDeleter") + ";\n" +
              re.search(r"template <class T> using DeviceStorage = .*;", core)[0])
             if "struct DeviceStorageDeleter" in core else "",
    "WEIGHT_FIELD": re.search(r"^    [^\n]*\bwr16(?:\s*=.*)?;[^\n]*", engine, re.M)[0],
    "FLAG_FIELD": re.search(r"^    [^\n]*\brouter_flags(?:\s*=.*)?;[^\n]*", engine, re.M)[0],
    "WEIGHT_ALLOC": re.search(r"^\s*y\.wr16[^\n]*dalloc[^\n]*;", engine, re.M)[0],
    "FLAG_ALLOC": re.search(r"^\s*router_flags[^\n]*dalloc[^\n]*;", engine, re.M)[0],
    "HELLO": block(control, "struct Hello") + ";\n" + block(control, "inline std::string hello_mismatch"),
}
for family in ("glm", "mm"):
    filename = f"tests/hip_{'glm_pf' if family == 'glm' else 'mm'}_router_wmma_tests.hip"
    pieces[family.upper() + "_LOOP"] = block(read(filename), "    for (unsigned f = 0; f < 3; ++f)")
template = Path(__file__).with_name("host.cpp.in").read_text(encoding="utf-8")
for name, value in pieces.items():
    template = template.replace("@" + name + "@", value)
output.mkdir(parents=True, exist_ok=True)
(output / "host.cpp").write_text(template, encoding="utf-8")
stub = output / "stubs/glm/prefill/glm_pf_exact.hpp"
stub.parent.mkdir(parents=True, exist_ok=True)
stub.write_text("#pragma once\nnamespace strix::glm::pf::exact {\n"
                "enum class Route { tm, x, x2, x3, x3r8 };\n"
                "inline Route route(unsigned) { return Route::x3r8; }\n"
                "inline unsigned x_auto_rows() { return 64; }\n}\n", encoding="utf-8")
