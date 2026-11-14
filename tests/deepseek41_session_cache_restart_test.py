"""Fresh processes/ranks with growing operational directories; no GPU."""
import json
import struct
import os
import pathlib
import subprocess
import sys
import tempfile
import contextlib
import shutil
import time

exe = str(pathlib.Path(sys.argv[1]).resolve())
with tempfile.TemporaryDirectory(prefix="ds41-compat-") as tmp:
    root = pathlib.Path(tmp)
    model = root / "model.bin"
    model.write_bytes(b"immutable numerical model" * 4096)
    env = dict(os.environ)
    for name in ("STRIX_HASH_CACHE", "STRIX_PREFIX_DIR", "STRIX_OUTPUT_DIR", "STRIX_LOG_DIR"):
        path = root / name
        path.mkdir()
        env[name] = str(path)
    numerical = root / "trellis"
    numerical.mkdir()
    pack = numerical / "weights.safetensors"
    def write_pack(payload):
        header=json.dumps({'x':{'dtype':'U8','shape':[len(payload)],'data_offsets':[0,len(payload)]}}).encode()
        pack.write_bytes(struct.pack('<Q',len(header))+header+payload)
    write_pack(b"numerical pack")
    env["STRIX_DS41_TRELLIS"] = str(numerical)
    # A coincidental path-valued selector must remain a scalar, not become an asset.
    env["STRIX_TEST_PATH_SELECTOR"] = str(root / "STRIX_LOG_DIR")
    def probe(rank):
        return subprocess.check_output([exe, "--compat-probe", str(model), str(rank)], env=env, text=True).splitlines()
    first = [probe(rank) for rank in range(2)]
    for name in ("STRIX_HASH_CACHE", "STRIX_PREFIX_DIR", "STRIX_OUTPUT_DIR", "STRIX_LOG_DIR"):
        (pathlib.Path(env[name]) / "new-entry").write_text("mutable run output")
    second = [probe(rank) for rank in range(2)]
    assert first == second, "compat key changed across fresh processes/restarts"
    assert first[0][0] != first[1][0], "rank must remain bound"
    assert first[0][1] == first[1][1], "shared compatibility differs across ranks"
    write_pack(b"different numerical pack")
    assert probe(0) != first[0], "immutable numerical directory content not bound"
    write_pack(b"numerical pack")
    model.write_bytes(b"changed numerical model" * 4096)
    assert probe(0) != first[0], "immutable model content not bound"
print("PASS explicit hash-cache directory: two ranks, two fresh runs, mutable outputs excluded, model bound")

# Independent server pairs contend on one set of rank-local disk directories.
# Holding rank 0's OS lock must block publication in another process.
with tempfile.TemporaryDirectory(prefix="ds41-disk-lock-") as tmp:
    root = pathlib.Path(tmp)
    (root / "rank0").mkdir()
    (root / "rank1").mkdir()
    lock = open(root / "rank0" / ".session-rank0.lock", "w+b")
    lock.write(b"0"); lock.flush()
    if os.name == "nt":
        import msvcrt
        lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
    else:
        import fcntl
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    workers = [subprocess.Popen([exe, "--disk-worker", tmp, name]) for name in ("aa", "bb")]
    try:
        import time
        time.sleep(0.5)
        assert all(p.poll() is None for p in workers), "publication escaped interprocess lock"
        assert not list(root.rglob("*.state")), "rank payload published without reservation lock"
    finally:
        lock.close()
    for p in workers:
        assert p.wait(timeout=60) == 0, "shared-directory transaction failed"
    assert not list(root.rglob("*.state")) and not list(root.rglob("*.group")), "delete left published generations"
print("PASS interprocess lock, concurrent quota/publication/replacement/delete")

# Fresh processes exercise the actual durable Sessions path (plain and DSpark),
# with entry-recorded byte hashes checked by the executable in each process.
with tempfile.TemporaryDirectory(prefix="ds41-process-restart-") as tmp:
    subprocess.run([exe, "--codec-write", tmp], check=True, timeout=30)
    subprocess.run([exe, "--codec-read", tmp], check=True, timeout=30)
print("PASS production Sessions write/read across fresh processes, both-rank payload/group bytes")

@contextlib.contextmanager
def lock_scratch():
    scratch = tempfile.mkdtemp(prefix="ds41-ordered-lock-")
    try:
        yield scratch
    except BaseException:
        print(f"Retained failed lock fixture: {scratch}")
        raise
    else:
        # All child processes and external handles are closed before teardown.
        # Windows can briefly retain a pending directory deletion after exit.
        for attempt in range(5):
            try:
                shutil.rmtree(scratch)
                break
            except OSError as error:
                if os.name != "nt" or error.winerror not in (32, 145) or attempt == 4:
                    raise
                print(f"Retry Windows lock-fixture teardown: {error.winerror}")
                time.sleep(0.05)

# Both ranks keep independent persistent lock inodes, even when their payload
# directories coincide. Exercise each old lock name as an external lock holder.
for shared in (False, True):
    for blocked_rank in range(2):
        with lock_scratch() as tmp:
            root = pathlib.Path(tmp)
            paths = [root / ("shared" if shared else f"rank{r}") for r in range(2)]
            for path in paths:
                path.mkdir(exist_ok=True)
            lock_path = paths[blocked_rank] / f".session-rank{blocked_rank}.lock"
            with lock_path.open("a+b") as lock:
                lock.write(b"0"); lock.flush(); lock.seek(0)
                inode = lock_path.stat().st_ino
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
                else:
                    import fcntl
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                args = ["--disk-worker-shared" if shared else "--disk-worker", tmp]
                processes = [subprocess.Popen([exe, *args, name]) for name in ("aa", "bb")]
                try:
                    try:
                        time.sleep(0.25)
                        assert all(p.poll() is None for p in processes), "ordered lock did not block paired writer"
                        assert not list(root.rglob("*.state")), "payload published before both ordered locks"
                    finally:
                        if os.name == "nt":
                            lock.seek(0); msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
                        else:
                            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
                    for process in processes:
                        assert process.wait(timeout=30) == 0, "concurrent paired writer refused/hung"
                finally:
                    for process in processes:
                        if process.poll() is None:
                            process.kill(); process.wait(timeout=5)
                assert lock_path.stat().st_ino == inode, "cleanup replaced external lock inode"
                deadline = time.monotonic() + 1.0
                observed = []
                while True:
                    leftovers = [str(p.relative_to(root)) for p in root.rglob("*")
                                 if p.suffix in (".state", ".group") or ".tmp" in p.name]
                    if not leftovers:
                        break
                    observed.extend(leftovers)
                    assert time.monotonic() < deadline, f"cleanup left publications/temporaries: {leftovers}"
                    time.sleep(0.01)
                if observed:
                    print(f"Transient directory entries cleared within one second: {observed}")
                for r in range(2):
                    assert (paths[r] / f".session-rank{r}.lock").exists(), "cleanup unlinked a rank lock"
print("PASS external legacy lock names on each rank, shared/separate directories, simultaneous paired writers, stable inodes")
