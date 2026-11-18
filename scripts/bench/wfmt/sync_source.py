#!/usr/bin/env python3
"""Synchronize a Git source archive while preserving mtimes of unchanged files."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile


def destination(root, name):
    relative = PurePosixPath(name)
    if not relative.parts or relative.is_absolute() or any(p == ".." for p in relative.parts) or "\\" in name or ":" in name:
        raise ValueError("source archive contains an unsafe path")
    target = root.joinpath(*relative.parts)
    # Do not follow the old leaf: a tracked symlink may become a directory.
    return target


def parents(names):
    return {str(p) for name in names for p in PurePosixPath(name).parents if str(p) != "."}


def synchronize(root, archive_path, revision):
    inventory = root / ".wfmt-source-inventory.json"
    if inventory.is_symlink():
        raise ValueError("source inventory must not be a symlink")
    previous = json.loads(inventory.read_text()) if inventory.exists() else {"files": []}
    old = set(previous["files"])
    entries, directories = {}, set()
    # Read and validate the entire archive before changing the old tree.
    with tarfile.open(archive_path) as archive:
        for member in archive:
            target = destination(root, member.name)
            name = target.relative_to(root).as_posix()
            if name == inventory.name or name.startswith(inventory.name + "/"):
                raise ValueError("source archive conflicts with inventory")
            if member.isdir():
                directories.add(name)
                continue
            if name in entries:
                raise ValueError("source archive contains duplicate paths")
            if member.issym():
                (target.parent / member.linkname).resolve().relative_to(root)
                data = member.linkname
            elif member.isfile():
                data = archive.extractfile(member).read()
            else:
                raise ValueError("source archive contains an unsupported member")
            entries[name] = (member, data)
    directories |= parents(set(entries) | directories)
    if directories & entries.keys():
        raise ValueError("source archive uses a file as a directory")
    old_dirs = parents(old)
    for name in old | old_dirs | set(entries) | directories:
        target = destination(root, name)
        for parent in target.parents:
            if parent == root:
                break
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                relative = parent.relative_to(root).as_posix()
                if relative not in old or relative not in directories:
                    raise ValueError("source path has a conflicting ancestor: " + name)
                # Descendants of a tracked leaf do not exist in the old tree.
                break
        else:
            raise ValueError("source path escapes root")
        # Never inspect through a symlink scheduled to become a directory.
        if any(p.is_symlink() or p.is_file() for p in target.parents if p != root and root in p.parents):
            continue
        exists = target.exists() or target.is_symlink()
        if name in directories and exists and (target.is_symlink() or not target.is_dir()) and name not in old:
            raise ValueError("untracked directory conflict: " + name)
        if name in entries and exists:
            if target.is_dir() and not target.is_symlink():
                # Refuse even untracked empty directories; never follow links.
                for base, dirs, files in os.walk(target, followlinks=False):
                    for child in dirs + files:
                        path = Path(base) / child
                        rel = path.relative_to(root).as_posix()
                        owned = rel in old if path.is_symlink() or not path.is_dir() else rel in old_dirs
                        if not owned:
                            raise ValueError("untracked descendant conflict: " + rel)
                if name not in old_dirs:
                    raise ValueError("untracked directory conflict: " + name)
            elif name not in old:
                raise ValueError("untracked file conflict: " + name)
    removed = changed = 0
    for name in sorted(old, key=lambda n: (-len(PurePosixPath(n).parts), n)):
        target = destination(root, name)
        if name not in entries and (target.is_symlink() or target.is_file()):
            target.unlink()
            removed += 1
    for name in sorted(old_dirs, key=lambda n: (-len(PurePosixPath(n).parts), n)):
        target = destination(root, name)
        if name not in directories and target.is_dir() and not target.is_symlink():
            if not any(target.iterdir()):
                target.rmdir()
    for name in sorted(directories, key=lambda n: (len(PurePosixPath(n).parts), n)):
        destination(root, name).mkdir(parents=True, exist_ok=True)
    for name, (member, data) in entries.items():
        target = destination(root, name)
        if member.issym():
            if not target.is_symlink() or os.readlink(target) != data:
                if target.exists() or target.is_symlink():
                    target.unlink()
                target.symlink_to(data)
                changed += 1
        else:
            if target.is_symlink() or not target.exists() or target.read_bytes() != data:
                if target.is_symlink():
                    target.unlink()
                target.write_bytes(data)
                changed += 1
            target.chmod(member.mode)
    receipt = {"schema": "wfmt.source@1", "revision": revision,
               "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
               "files": sorted(entries)}
    # Leave the previous inventory intact if writing the new one is interrupted.
    with tempfile.NamedTemporaryFile(mode="w", dir=root, delete=False, encoding="utf-8") as pending:
        pending.write(json.dumps(receipt, indent=2) + "\n")
    try:
        os.replace(pending.name, inventory)
    finally:
        Path(pending.name).unlink(missing_ok=True)
    print(json.dumps({"schema": receipt["schema"], "revision": revision,
                      "files": len(entries), "changed": changed, "removed": removed}, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    args = parser.parse_args()
    if len(args.revision) != 40 or any(c not in "0123456789abcdef" for c in args.revision):
        parser.error("revision must be a full lowercase Git object ID")
    root = args.source.resolve()
    root.mkdir(parents=True, exist_ok=True)
    synchronize(root, args.archive, args.revision)


if __name__ == "__main__":
    main()
