#!/usr/bin/env python3
"""Run a parameterized, sequential ABAB comparison inside an externally reserved GPU window.

Config: {"pack_identity": "...", "arms": [{"name":"ours", "cmd":[...],
"url":"http://127.0.0.1:8097", "cwd":"...", "env":{}, "body_extra":{},
"cache_policy":"prefix and conversation caches disabled by these cmd/env settings"},
{...}], "decode_prompts":{"prose":["..."],"edit":["..."],"json":["..."]},
"prefill_prompts":{"8192":["cold.txt","warm.txt"],"32768":[...],"43008":[...]}}.
Paths are supplied by the caller. No models are fetched. --plan validates without loading.
Each decode class and prefill target gets a fresh server process. Decode is cold C1,
then warm C1/C4/C8, with cache policy documented in each arm. Cold means process-first request;
Warm prefills use the process-last request with a distinct prefix. Decode warm
rows follow the first cold request. The same files feed both arms.
"""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import pathlib
import signal
import subprocess
import sys
import threading
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[3]
BENCH = ROOT / "tools/serve_bench.py"
SPEC = importlib.util.spec_from_file_location("serve_bench", BENCH)
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)
OWNER_SPEC = importlib.util.spec_from_file_location("native_emissions", ROOT / "scripts/bench/common/native-emissions.py")
owner_api = importlib.util.module_from_spec(OWNER_SPEC)
OWNER_SPEC.loader.exec_module(owner_api)


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def free_ram():
    if os.name == "nt":
        class Memory(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (n, ctypes.c_ulonglong) for n in ("total", "available", "page", "freepage", "virt", "freevirt", "extended")]
        mem = Memory()
        mem.length = ctypes.sizeof(mem)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem)):
            raise OSError("GlobalMemoryStatusEx failed")
        return mem.available
    return int(next(s.split()[1] for s in pathlib.Path("/proc/meminfo").read_text().splitlines()
                    if s.startswith("MemAvailable:"))) * 1024


def gpu_sample():
    # Physical index 0 only, including before CUDA_VISIBLE_DEVICES is applied.
    result = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=clocks.sm,utilization.gpu,memory.used",
                             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    clock, busy, memory = map(float, result.stdout.strip().split(","))
    return dict(time=time.time(), clock_hz=clock * 1e6, busy=busy / 100, memory_mib=memory)


class Sampler:
    def __init__(self):
        self.samples, self.errors, self.stop = [], [], threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop.is_set():
            try:
                self.samples.append(gpu_sample())
            except Exception as error:
                self.errors.append(str(error))
            self.stop.wait(0.25)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=5)


def wait_ready(proc, arm, timeout):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited before ready: {proc.returncode}")
        try:
            with urllib.request.urlopen(arm["url"].rstrip("/") + arm.get("ready_path", "/v1/models"), timeout=1):
                return
        except Exception:
            time.sleep(0.25)
    raise TimeoutError("server readiness timeout")


def stop_owned(proc):
    if os.name == "nt":
        if proc.poll() is not None:
            return
        # Only this Popen process and its descendants; never match names globally.
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        # Popen owns a new session: descendants retain this group after its leader exits.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=20)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            proc.kill()
    finally:
        if os.name != "nt":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    proc.wait(timeout=5)


def client(arm, prompts, concurrency, tokens, out, timeout):
    offset = time.time() - time.perf_counter()
    events = bench.run_burst(arm["url"].rstrip("/") + "/v1/chat/completions",
                             [pathlib.Path(f).read_text(encoding="utf-8") for f in prompts],
                             concurrency, tokens, True, arm.get("body_extra", {}), timeout)
    burst = bench.burst_report(concurrency, tokens, events, [], None)
    write(out, dict(schema=bench.SCHEMA, bursts=[burst], events=events))
    if burst["client"]["failed"]:
        raise RuntimeError(f"client failed: {out.name}")
    start = min(e["arrival"] for e in events) + offset
    end = max(max(e["token_times"]) for e in events) + offset
    return dict(file=out.name, start=start, end=end)


def run_one(config, arm, workload, pair, out, args):
    until = time.monotonic() + args.release_wait_s
    while True:
        ram, memory = free_ram(), gpu_sample()["memory_mib"]
        if ram >= 20 * 1024**3 and memory < args.idle_memory_mib:
            break
        if time.monotonic() >= until:
            raise RuntimeError(f"resources not released before load: RAM {ram / 1024**3:.1f} GiB, GPU 0 {memory} MiB")
        time.sleep(2)
    out.mkdir(parents=True, exist_ok=False)
    identity = subprocess.run(["nvidia-smi", "-i", "0", "--query-gpu=name,pci.bus_id,uuid",
                               "--format=csv,noheader"], check=True, capture_output=True, text=True)
    device_name, device_bus, device_uuid = [part.strip() for part in identity.stdout.strip().split(",")]
    if ("RTX PRO 6000" not in device_name or device_bus.lower().lstrip("0") != ":01:00.0"
            or not device_uuid.startswith("GPU-")):
        raise RuntimeError("physical index 0 is not the expected RTX PRO at 01:00.0; no model loaded")
    env = dict(os.environ)
    env.update(arm.get("env", {}))
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=device_uuid)
    manifest = dict(arm=arm["name"], pair=pair, workload=workload, pack_identity=config["pack_identity"],
                    cmd=arm["cmd"], records=[], status="running")
    write(out / "run.json", manifest)
    with (out / "server.log").open("w", encoding="utf-8") as log, Sampler() as sampler:
        proc = subprocess.Popen(arm["cmd"], cwd=arm.get("cwd"), env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=os.name != "nt")
        owner = None
        manifest["pid"] = proc.pid
        try:
            owner = owner_api.windows_job(proc)
            wait_ready(proc, arm, args.ready_s)
            if workload.startswith("decode-"):
                kinds = ("prose", "edit", "json") if workload == "decode-all" else (workload.removeprefix("decode-"),)
                for ordinal, kind in enumerate(kinds):
                    prompts = config["decode_prompts"][kind]
                    first = "cold" if ordinal == 0 else "intermediate"
                    for state, concurrency in ((first, 1), ("warm", 1), ("warm", 4), ("warm", 8)):
                        rec = client(arm, prompts, concurrency, args.tokens,
                                     out / f"{kind}-{state}-c{concurrency}.json", args.timeout)
                        manifest["records"].append(dict(rec, kind=kind, concurrency=concurrency, state=state))
            else:
                prompts = config["prefill_prompts"][workload]
                for i, prompt in enumerate(prompts):
                    rec = client(arm, [prompt], 1, 1, out / f"prefill-{i}.json", args.timeout)
                    state = "cold" if i == 0 else "warm" if i == len(prompts) - 1 else "intermediate"
                    measured = json.loads((out / rec["file"]).read_text(encoding="utf-8"))["bursts"][0]["client"]["prompt_tokens"]
                    if not int(workload) * (1 - args.context_tolerance) <= measured <= int(workload) * (1 + args.context_tolerance):
                        raise ValueError(f"actual prompt {measured} outside target {workload} band")
                    if measured + 1 > args.max_context:
                        raise ValueError(f"prompt {measured} plus emitted token exceeds max context")
                    manifest["records"].append(dict(rec, kind="prefill", target_tokens=int(workload), state=state))
            manifest["status"] = "complete"
        except Exception as error:
            manifest["status"] = "failed"
            manifest["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            stop_owned(proc)
            if owner:
                owner[0].CloseHandle(owner[1])  # also owns children whose parent exited first
            manifest["samples"] = sampler.samples
            manifest["sampler_errors"] = sampler.errors
            write(out / "run.json", manifest)
    print(f"{arm['name']} pair {pair} {workload}: complete", flush=True)


def validate(config):
    if not isinstance(config.get("group_decode_classes", False), bool):
        raise ValueError("group_decode_classes must be boolean")
    if not config.get("pack_identity") or len(config["arms"]) != 2:
        raise ValueError("require pack_identity and exactly two arms")
    if len({arm["name"] for arm in config["arms"]}) != 2:
        raise ValueError("arm names must differ")
    for arm in config["arms"]:
        if not arm.get("cache_policy"):
            raise ValueError("each arm must document cache disabling in cache_policy")
        if not isinstance(arm["cmd"], list) or not arm["cmd"]:
            raise ValueError("server cmd must be a nonempty argv list")
        if not arm["url"].startswith("http://127.0.0.1:"):
            raise ValueError("require loopback server URL")
    hashes = {}
    for kind in ("prose", "edit", "json"):
        if not config["decode_prompts"].get(kind):
            raise ValueError(f"missing {kind} prompts")
    for target in ("8192", "32768", "43008"):
        prompts = config["prefill_prompts"][target]
        if len(prompts) < 2:
            raise ValueError("require at least two prefills per target")
        prefixes = [pathlib.Path(f).read_text(encoding="utf-8")[:64] for f in prompts]
        if len(set(prefixes)) != len(prefixes):
            raise ValueError("prefill prefixes must differ to avoid prefix-cache reuse")
    for files in list(config["decode_prompts"].values()) + list(config["prefill_prompts"].values()):
        for f in files:
            hashes[f] = hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
    return hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=pathlib.Path, required=True)
    p.add_argument("--out", type=pathlib.Path, required=True)
    p.add_argument("--pairs", type=int, default=3)
    p.add_argument("--tokens", type=int, default=512)
    p.add_argument("--ready-s", type=float, default=900)
    p.add_argument("--timeout", type=float, default=900)
    p.add_argument("--idle-memory-mib", type=float, default=2048)
    p.add_argument("--max-context", type=int, default=45056)
    p.add_argument("--context-tolerance", type=float, default=0.05)
    p.add_argument("--plan", action="store_true")
    p.add_argument("--run-index", type=int, help="run one fixed plan entry in a separately reserved FIFO window")
    p.add_argument("--resume", action="store_true", help="preserve completed entries of the identical frozen plan")
    p.add_argument("--release-wait-s", type=float, default=30, help="bounded wait for owned-process resource teardown")
    args = p.parse_args()
    if args.pairs < 3:
        p.error("strict ABAB requires at least three pairs")
    if not 0 <= args.context_tolerance < 1:
        p.error("context tolerance must be in [0,1)")
    if args.release_wait_s < 0:
        p.error("release-wait-s must be nonnegative")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    hashes = validate(config)
    workloads = ("decode-all",) if config.get("group_decode_classes") else ("decode-prose", "decode-edit", "decode-json", "8192", "32768", "43008")
    plan = [(workload, pair, arm) for workload in workloads
            for pair in range(1, args.pairs + 1) for arm in config["arms"]]
    if args.plan:
        print(json.dumps(dict(order=[f"{w}:{pair}:{a['name']}" for w, pair, a in plan], prompt_sha256=hashes), indent=1))
        return 0
    if args.run_index is not None and not 0 <= args.run_index < len(plan):
        p.error("run-index is outside the fixed plan")
    if (args.resume or args.run_index is not None) and args.out.exists() and any(args.out.iterdir()):
        if any(not (args.out / name).is_file() for name in ("config.json", "prompt-sha256.json")):
            p.error("nonempty output is missing frozen metadata; use a fresh output directory")
    args.out.mkdir(parents=True, exist_ok=args.run_index is not None or args.resume)
    for name, expected in (("config.json", config), ("prompt-sha256.json", hashes)):
        path = args.out / name
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != expected:
            p.error(f"existing {name} differs; use a fresh output directory")
        write(path, expected)
    for index, (workload, pair, arm) in enumerate(plan):
        if args.run_index is not None and index != args.run_index:
            continue
        entry = args.out / f"{index:02}-{arm['name']}-{workload}"
        if args.resume and (entry / "run.json").exists():
            saved = json.loads((entry / "run.json").read_text())
            expected = dict(arm=arm["name"], workload=workload, pair=pair, pack_identity=config["pack_identity"])
            if saved.get("status") != "complete" or any(saved.get(k) != v for k, v in expected.items()):
                p.error(f"{entry.name} is incomplete or mismatched; preserve it before retrying")
            if not saved.get("records") or any(not (entry / rec["file"]).is_file() for rec in saved["records"]):
                p.error(f"{entry.name} is missing completed receipts")
            print(f"resume kept {entry.name}", flush=True)
            continue
        run_one(config, arm, workload, pair, entry, args)
    if all((args.out / f"{i:02}-{a['name']}-{w}" / "run.json").exists() and
           json.loads((args.out / f"{i:02}-{a['name']}-{w}" / "run.json").read_text())["status"] == "complete"
           for i, (w, _, a) in enumerate(plan)):
        (args.out / ".done").touch()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
