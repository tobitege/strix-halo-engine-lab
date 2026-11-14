#!/usr/bin/env python3
"""Exercise the production MiniMax source census against disposable scoped trees."""
import pathlib
import shutil
import subprocess
import sys
import tempfile

root = pathlib.Path(__file__).resolve().parents[1]
scratch = pathlib.Path(tempfile.gettempdir())
binary = pathlib.Path(sys.argv[1]).resolve()

def copy_scope(dest):
    for rel in ("src/minimax", "src/moe_core", "src/dspark_common", "src/xport",
                "include/strix/minimax", "include/strix/session", "src/session"):
        shutil.copytree(root / rel, dest / rel)
    (dest / "include/strix/serve").mkdir(parents=True)
    shutil.copy2(root / "include/strix/serve/session_runtime.hpp", dest / "include/strix/serve/session_runtime.hpp")
    for rel in ("CMakeLists.txt", "cmake/StrixSession.cmake", "src/serve/engine.cpp",
                "src/serve/session_runtime.cpp", "include/strix/serve/engine.hpp"):
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / rel, target)
    for base, prefix in (("apps", "minimax_"), ("src", "minimax_"), ("src", "hip_minimax_"),
                         ("include/strix", "minimax_"), ("include/strix", "hip_minimax_"),
                         ("src", "bp_codec."), ("include/strix", "bp_codec.")):
        target_dir = dest / base
        target_dir.mkdir(parents=True, exist_ok=True)
        for path in (root / base).iterdir():
            if path.is_file() and path.name.startswith(prefix):
                shutil.copy2(path, target_dir / path.name)

def audit(tree):
    return subprocess.run([str(binary), "--audit-sources", str(tree)], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

with tempfile.TemporaryDirectory(prefix="reviewed-", dir=scratch) as temp:
    baseline = pathlib.Path(temp) / "tree"
    copy_scope(baseline)
    good = audit(baseline)
    assert good.returncode == 0, good.stdout
    # The historical source snapshot is the reviewed receipt baseline. CRLF
    # must not create a mismatch because the production audit normalizes it.
    crlf = baseline / "src/minimax/engine/mm_pack.cpp"
    crlf.write_bytes(crlf.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    normalized = audit(baseline)
    assert normalized.returncode == 0, normalized.stdout
    victim = baseline / "src/minimax/engine/mm_pack.cpp"
    victim.write_bytes(victim.read_bytes() + b"\n// disposable audit drift\n")
    edited = audit(baseline)
    assert edited.returncode != 0 and "src/minimax/engine/mm_pack.cpp" in edited.stdout, edited.stdout

with tempfile.TemporaryDirectory(prefix="added-", dir=scratch) as temp:
    tree = pathlib.Path(temp) / "tree"
    copy_scope(tree)
    (tree / "src/minimax/serve/new_owner.cpp").write_text("// new in-scope source\n", encoding="utf-8")
    added = audit(tree)
    assert added.returncode != 0 and "src/minimax/serve/new_owner.cpp" in added.stdout, added.stdout

with tempfile.TemporaryDirectory(prefix="removed-", dir=scratch) as temp:
    tree = pathlib.Path(temp) / "tree"
    copy_scope(tree)
    owner = tree / "src/session/kv.cpp"
    owner.unlink()
    removed = audit(tree)
    assert removed.returncode != 0 and "src/session/kv.cpp" in removed.stdout, removed.stdout

with tempfile.TemporaryDirectory(prefix="shared-owner-", dir=scratch) as temp:
    tree = pathlib.Path(temp) / "tree"
    copy_scope(tree)
    owner = tree / "src/session/kv.cpp"
    owner.write_bytes(owner.read_bytes() + b"\n// shared owner drift\n")
    result = audit(tree)
    assert result.returncode != 0 and "src/session/kv.cpp" in result.stdout, result.stdout

with tempfile.TemporaryDirectory(prefix="queue-owner-", dir=scratch) as temp:
    tree = pathlib.Path(temp) / "tree"
    copy_scope(tree)
    owner = tree / "src/serve/engine.cpp"
    owner.write_bytes(owner.read_bytes() + b"\n// shared queue owner drift\n")
    result = audit(tree)
    assert result.returncode != 0 and "src/serve/engine.cpp" in result.stdout, result.stdout

print("PASS receipted Git snapshot and source audit negatives; current source gate retains seven unresolved rows")
