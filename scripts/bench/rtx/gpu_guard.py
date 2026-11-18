#!/usr/bin/env python3
"""gpu_guard.py: run one GPU job under a 1 s telemetry guard that fails closed.

    python scripts/bench/rtx/gpu_guard.py --gpu-name "RTX PRO 6000" --out DIR --latch FILE [--timeout-s 300] -- CMD ...

What it does (read-only on the machine: it never changes driver, power, registry or clock settings):
  * resolves the GPU by NAME (and optional --pci) to its UUID and runs CMD with CUDA_VISIBLE_DEVICES=<UUID>, so a
    renumbered device can never be picked by index;
  * before the launch: refuses when the latch file exists, when the GPU is not visible, when its memory is above
    --max-idle-mib, or when a guarded Windows event happened in the last --quiet-before-s seconds;
  * while CMD runs: `nvidia-smi -i UUID --query-gpu=... -l 1` to DIR/smi.csv, a persistent System-log watcher
    (winevents.ps1: WHEA-Logger any id, nvlddmkm any id, Display 4101) to DIR/events.log, PCIe replay counter every
    --replay-every-s, child stdout/stderr to DIR/child.log (and echoed);
  * aborts on the FIRST: guarded event, telemetry silence > --stale-s (GPU gone), GPU missing from nvidia-smi, temperature
    margin to the slowdown limit (temperature.gpu.tlimit) < --tlimit-min-c or temperature >= --temp-max-c, PCIe replay
    increase, a child output line matching --fail-pattern (CUDA "unknown error", illegal address, launch failure), or the
    wall limit. Abort = kill the owned workload, write the configured latch (reason + time), exit non-zero;
  * after all workload processes exit: --post-s more seconds of monitoring; a non-zero child rc is a failure.
  * DIR/guard.json + one 'GUARD ...' summary line (peak W, max C, min margin, max MiB, largest 1 s power step up/down).

Exit codes: child rc when only the child failed; 90 event, 91 GPU lost / telemetry silent, 92 temperature, 93 PCIe
replay, 94 wall limit, 95 fail pattern, 96 refused before launch (nothing ran), 97 guard failure, 130 interruption.
All arms/agents must explicitly share --latch FILE (or GPU_GUARD_LATCH); omission disables cross-run STOP.
Windows event coverage is machine-wide, including faults on other devices; Linux has no event-log coverage.
Temperature, thermal margin and replay counter are required; unavailable sensors refuse/trip. Measurement-only
fields may be unavailable and receipts include busy-power/clock sample counts. Startup requires healthy monitors.
Linux commands must retain the owned process group (no setsid/daemon escape); Windows uses a Job Object.
"""
import argparse, json, math, os, re, shutil, signal, subprocess, sys, threading, time
from datetime import datetime

FIELDS = ("timestamp,uuid,pstate,temperature.gpu,temperature.gpu.tlimit,power.draw,enforced.power.limit,utilization.gpu,"
          "memory.used,memory.total,clocks.sm,clocks.mem,clocks_event_reasons.active,pcie.link.gen.current,"
          "pcie.link.width.current")
FAIL_DEFAULT = (r"failed: unknown error|unspecified launch failure|illegal memory access|illegal instruction|"
                r"No devices were found|GPU is lost|Xid|cudaErrorDevicesUnavailable|device-side assert")
WIN = os.name == "nt"
STARTUP_QUIET_S = 3.0
MONITOR_STARTUP_S = 15.0
REPLAY_QUERY_S = 15.0


def now():
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def smi(args, timeout=15):
    try:
        r = subprocess.run(["nvidia-smi"] + args, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout
    except Exception as e:  # nvidia-smi hung or missing: treat as not visible
        return 255, str(e)


def resolve(name, pci):
    rc, out = smi(["--query-gpu=index,uuid,name,pci.bus_id", "--format=csv,noheader"])
    if rc:
        return None, out.strip()
    for line in out.splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 4 and name in p[2] and (not pci or p[3].lower().endswith(pci.lower())):
            return p[1], line.strip()
    return None, out.strip()


def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


class WindowsJob:
    """Keep descendants owned even after the launcher exits; closing kills all members."""
    def __init__(self):
        import ctypes as c
        from ctypes import wintypes as w
        class Limits(c.Structure):
            _fields_ = [("process_time", c.c_int64), ("job_time", c.c_int64), ("flags", w.DWORD),
                        ("min_ws", c.c_size_t), ("max_ws", c.c_size_t), ("active", w.DWORD),
                        ("affinity", c.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]
        class Extended(c.Structure):
            _fields_ = [("basic", Limits), ("io", c.c_uint64 * 6), ("memory", c.c_size_t * 4)]
        class Accounting(c.Structure):
            _fields_ = [("times", c.c_int64 * 4), ("faults", w.DWORD), ("total", w.DWORD),
                        ("active", w.DWORD), ("terminated", w.DWORD)]
        self.c, self.Accounting = c, Accounting
        self.api = c.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([c.c_void_p, w.LPCWSTR], w.HANDLE),
            "SetInformationJobObject": ([w.HANDLE, c.c_int, c.c_void_p, w.DWORD], w.BOOL),
            "QueryInformationJobObject": ([w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.c_void_p], w.BOOL),
            "AssignProcessToJobObject": ([w.HANDLE, w.HANDLE], w.BOOL),
            "TerminateJobObject": ([w.HANDLE, w.UINT], w.BOOL),
            "CloseHandle": ([w.HANDLE], w.BOOL),
        }
        for name, (args, result) in signatures.items():
            fn = getattr(self.api, name)
            fn.argtypes, fn.restype = args, result
        self.handle = self.api.CreateJobObjectW(None, None)
        self.check(self.handle)
        limits = Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        try:
            self.check(self.api.SetInformationJobObject(self.handle, 9, c.byref(limits), c.sizeof(limits)))
        except BaseException:
            self.close()
            raise

    def check(self, result):
        if not result:
            raise self.c.WinError(self.c.get_last_error())

    def assign(self, child):
        self.check(self.api.AssignProcessToJobObject(self.handle, int(child._handle)))

    def active(self):
        info = self.Accounting()
        self.check(self.api.QueryInformationJobObject(self.handle, 1, self.c.byref(info), self.c.sizeof(info), None))
        return info.active != 0

    def kill(self):
        if self.handle:
            self.check(self.api.TerminateJobObject(self.handle, 97))

    def close(self):
        if self.handle:
            self.check(self.api.CloseHandle(self.handle))
            self.handle = None


class Guard:
    def __init__(self, a):
        self.a = a
        self.lock = threading.RLock()
        self.trip = None          # (code, reason)
        self.last_smi = None
        self.last_hb = None
        self.samples = []
        self.events = []
        self.child = None
        self.job = None
        self.launched = False
        self.latch_error = None
        self.stop = threading.Event()
        self.threads = []
        self.monitors = []
        self.replay_initial = None
        self.replay_at = None
        self.replay_request = threading.Event()

    def fire(self, code, reason):
        with self.lock:
            if not self.trip:
                self.trip = (code, reason)
                if self.a.latch:
                    try:
                        with open(self.a.latch, "a", encoding="utf-8") as f:
                            f.write(f"{now()} gpu_guard STOP code {code}: {reason} (out {self.a.out})\n")
                            f.flush()
                            os.fsync(f.fileno())
                    except OSError as e:
                        self.latch_error = str(e)
                try:
                    self.log(f"TRIP {code} {reason}" + (f"; latch write failed: {self.latch_error}" if self.latch_error else ""))
                finally:
                    self.kill_child()
            else:
                self.kill_child()  # Retry termination independently of choosing the first reason.

    def log(self, msg):
        line = f"{now()} {msg}"
        with open(os.path.join(self.a.out, "guard.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print("[gpu_guard] " + line, file=sys.stderr, flush=True)

    def kill_child(self):
        c = self.child
        if c is None:
            return
        if self.job:
            self.job.kill()
            if c.poll() is None:  # Also covers an assignment failure before admission.
                c.kill()
        else:
            try:
                os.killpg(c.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def launch(self, env, pat):
        with self.lock:
            if self.trip:
                return
            if WIN:
                self.job = WindowsJob()
                # The relay cannot start CMD until it belongs to our Job. All descendants inherit it.
                relay = ("import subprocess,sys; go=sys.stdin.buffer.read(1); "
                         "sys.exit(subprocess.call(sys.argv[1:], stdin=subprocess.DEVNULL) if go == b'1' else 97)")
                self.child = subprocess.Popen([sys.executable, "-c", relay] + self.a.cmd, stdin=subprocess.PIPE,
                                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                              env=env, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
                self.job.assign(self.child)
                if self.trip:
                    self.kill_child()
                    return
                self.child.stdin.write(b"1")
                self.child.stdin.close()
            else:
                self.child = subprocess.Popen(self.a.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                              env=env, start_new_session=True)
            self.launched = True
            if self.trip:
                self.kill_child()
            self.watch(self.child_reader, self.child, pat)

    def workload_active(self):
        if self.child is None:
            return False
        self.child.poll()  # Reap the root, but retain the group/job ownership.
        if self.job:
            return self.job.active()
        try:
            os.killpg(self.child.pid, 0)
            return True
        except ProcessLookupError:
            return False

    def watch(self, target, *args):
        def run():
            try:
                target(*args)
            except Exception as e:
                if not self.stop.is_set():
                    self.fire(97, f"{target.__name__} failed: {e}")
        t = threading.Thread(target=run)
        self.threads.append(t)
        t.start()

    def close(self):
        self.stop.set()
        self.replay_request.set()
        errors = []
        def attempt(fn):
            try:
                fn()
            except Exception as e:
                errors.append(str(e))
        try:
            with self.lock:  # Wait for STOP persistence, including an in-flight watcher trip.
                attempt(self.kill_child)
                if self.job:
                    attempt(self.job.close)
        finally:
            for p, _ in self.monitors:
                if p.poll() is None:
                    attempt(p.kill)
                attempt(lambda: p.wait(timeout=5))
            if self.child:
                attempt(lambda: self.child.wait(timeout=5))
            for t in self.threads:
                t.join()
            for p in [p for p, _ in self.monitors] + ([self.child] if self.child else []):
                if p.stdout:
                    p.stdout.close()
                if p.stdin:
                    p.stdin.close()
        if errors:
            self.fire(97, "cleanup failed: " + "; ".join(errors))

    def ready(self):
        return self.last_smi is not None and (not WIN or self.last_hb is not None) and self.replay_at is not None

    def check_health(self, started, launched=None):
        t = time.monotonic()
        for p, name in self.monitors:
            if p.poll() is not None:
                self.fire(91, f"{name} watcher exited")
        for last, limit, name in ((self.last_smi, self.a.stale_s, "telemetry"),
                                  (self.last_hb, 3 * self.a.stale_s, "event watcher")):
            if name == "event watcher" and not WIN:
                continue
            if last is not None and t - last > limit:
                self.fire(91, f"{name} silent for {t - last:.1f} s")
        if not self.ready() and t - started > MONITOR_STARTUP_S:
            self.fire(91, "monitor startup incomplete")
        if launched is not None and t - launched > self.a.timeout_s:
            self.fire(94, f"wall limit {self.a.timeout_s} s")

    def replay_worker(self, uuid):
        while not self.stop.is_set():
            self.replay_request.clear()
            # Keep this potentially hung query off the watchdog thread and cancel it on shutdown.
            queried = time.monotonic()
            p = subprocess.Popen(["nvidia-smi", "-i", uuid, "-q"], stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True)
            try:
                deadline = time.monotonic() + REPLAY_QUERY_S
                while not self.stop.is_set():
                    try:
                        out, _ = p.communicate(timeout=0.1)
                        break
                    except subprocess.TimeoutExpired:
                        if time.monotonic() >= deadline:
                            self.fire(93, "PCIe replay query timed out")
                            return
                else:
                    return
                m = re.search(r"Replays Since Reset\s*:\s*(\d+)", out)
                if p.returncode or not m:
                    self.fire(93, "PCIe replay monitoring unavailable")
                    return
                rp = int(m.group(1))
                if self.replay_initial is None:
                    self.replay_initial = rp
                elif rp > self.replay_initial:
                    self.fire(93, f"PCIe replays {self.replay_initial} -> {rp}")
                self.replay_at = queried
            finally:
                if p.poll() is None:
                    p.kill()
                p.communicate()
                p.stdout.close()
            self.replay_request.wait(self.a.replay_every_s)

    # ---- watchers -------------------------------------------------------------------------------------------
    def smi_reader(self, proc):
        with open(os.path.join(self.a.out, "smi.csv"), "a", encoding="utf-8") as out:
            out.write(FIELDS + "\n")
            for line in proc.stdout:
                out.write(line)
                out.flush()
                p = [x.strip() for x in line.split(",")]
                if len(p) < 15 or p[1] != self.uuid:
                    self.fire(91, f"invalid telemetry: {line.strip()[:120]}")
                    continue
                s = {"t": time.time(), "temp": num(p[3]), "tlimit": num(p[4]), "w": num(p[5]), "plim": num(p[6]),
                     "util": num(p[7]), "mib": num(p[8]), "sm": num(p[10]), "gen": num(p[13]), "width": num(p[14])}
                if s["temp"] is None or s["tlimit"] is None:
                    self.fire(91, "required thermal telemetry unavailable")
                    continue
                self.samples.append(s)
                if s["tlimit"] < self.a.tlimit_min_c:
                    self.fire(92, f"temperature margin {s['tlimit']} C < {self.a.tlimit_min_c} C (temp {s['temp']} C)")
                if s["temp"] >= self.a.temp_max_c:
                    self.fire(92, f"temperature {s['temp']} C >= {self.a.temp_max_c} C")
                self.last_smi = time.monotonic()

    def event_reader(self, proc):
        with open(os.path.join(self.a.out, "events.log"), "a", encoding="utf-8") as out:
            for line in proc.stdout:
                if line.startswith("HB "):
                    self.last_hb = time.monotonic()
                    continue
                out.write(line)
                out.flush()
                if line.startswith("EVT "):
                    self.events.append(line.strip()[:300])
                    self.fire(90, "windows event: " + line.strip()[4:180])
                elif line.startswith("ERR "):
                    self.fire(91, "event watcher failed: " + line.strip()[4:180])

    def child_reader(self, proc, pat):
        with open(os.path.join(self.a.out, "child.log"), "ab") as out:
            for raw in proc.stdout:
                out.write(raw)
                out.flush()
                sys.stdout.buffer.write(raw)
                sys.stdout.flush()
                if pat.search(raw.decode("utf-8", "replace")):
                    self.fire(95, "child output: " + raw.decode("utf-8", "replace").strip()[:160])

    def summary(self, child_rc, code, t0, t1):
        S = self.samples
        def mx(k):
            v = [s[k] for s in S if num(s[k]) is not None]
            return max(v) if v else None
        def mn(k):
            v = [s[k] for s in S if num(s[k]) is not None]
            return min(v) if v else None
        steps = [b["w"] - a["w"] for a, b in zip(S, S[1:]) if num(a["w"]) is not None and num(b["w"]) is not None]
        busy = [s for s in S if (num(s["util"]) or 0) >= 90]
        watts = [s["w"] for s in busy if num(s["w"]) is not None]
        clocks = [s["sm"] for s in busy if num(s["sm"]) is not None]
        d = {"status": "ok" if code == 0 else "FAIL", "exit": code, "child_rc": child_rc,
             "reason": self.trip[1] if self.trip else ("child rc %s" % child_rc if code else ""),
             "start": t0, "end": t1, "samples": len(S), "peak_w": mx("w"), "power_limit_w": mx("plim"),
             "max_temp_c": mx("temp"), "min_tlimit_c": mn("tlimit"), "max_mib": mx("mib"),
             "max_step_up_w": round(max(steps), 1) if steps else None,
             "max_step_down_w": round(min(steps), 1) if steps else None,
             "busy_s": len(busy), "busy_mean_w": round(sum(watts) / len(watts), 1) if watts else None,
             "busy_mean_sm_mhz": round(sum(clocks) / len(clocks)) if clocks else None,
             "busy_w_samples": len(watts), "busy_sm_samples": len(clocks),
             "coverage": {"windows_events": WIN, "thermal_samples": len(S),
                          "replay_available": self.replay_at is not None}, "latch_error": self.latch_error,
             "pcie_gen_min": mn("gen"), "pcie_width_min": mn("width"), "events": self.events,
             "cmd": self.a.cmd}
        with open(os.path.join(self.a.out, "guard.json"), "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1)
        print(f"GUARD {d['status']} exit {code} child_rc {child_rc} reason '{d['reason']}' peak {d['peak_w']} W "
              f"(limit {d['power_limit_w']}) maxT {d['max_temp_c']} C min margin {d['min_tlimit_c']} C "
              f"max {d['max_mib']} MiB step +{d['max_step_up_w']}/{d['max_step_down_w']} W busy {d['busy_s']} s "
              f"@ {d['busy_mean_w']} W {d['busy_mean_sm_mhz']} MHz events {len(self.events)}", flush=True)


def supervise(g, uuid, pat):
    a = g.a
    g.uuid = uuid
    started = time.monotonic()
    if WIN:
        since = datetime.fromtimestamp(time.time() - a.quiet_before_s).isoformat(timespec="seconds")
        ev = subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                               os.path.join(os.path.dirname(os.path.abspath(__file__)), "winevents.ps1"),
                               "-Since", since], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        g.monitors.append((ev, "event"))
        g.watch(g.event_reader, ev)
    else:
        g.log("not Windows: no system-event coverage (recorded in guard.json)")
    sm = subprocess.Popen(["nvidia-smi", "-i", uuid, "--query-gpu=" + FIELDS,
                           "--format=csv,noheader,nounits", "-l", "1"],
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    g.monitors.append((sm, "telemetry"))
    g.watch(g.smi_reader, sm)
    g.watch(g.replay_worker, uuid)
    while not g.trip:
        g.check_health(started)
        if g.ready() and time.monotonic() - started >= STARTUP_QUIET_S:
            break
        g.stop.wait(0.05)
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=uuid, GPU_GUARD_UUID=uuid)
    launched = time.monotonic()
    g.launch(env, pat)
    if not g.launched:
        return
    g.log("LAUNCH " + " ".join(a.cmd))
    while not g.trip and g.workload_active():
        g.check_health(started, launched)
        g.stop.wait(0.05)
    # The post window starts only after all owned workload processes have exited.
    end = time.monotonic() + a.post_s
    while not g.trip and time.monotonic() < end:
        g.check_health(started, launched)
        g.stop.wait(0.05)
    # Require a fresh final replay sample, still supervised for silence/wall deadlines.
    final = time.monotonic()
    g.replay_request.set()
    while not g.trip and (g.replay_at is None or g.replay_at < final):
        g.check_health(started, launched)
        g.stop.wait(0.05)
    g.check_health(started, launched)


class Refused(Exception):
    pass


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu-name", required=True, help="substring of the nvidia-smi name, e.g. 'RTX PRO 6000'")
    ap.add_argument("--pci", default="", help="optional PCI bus id suffix, e.g. 01:00.0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--latch", default=os.environ.get("GPU_GUARD_LATCH", ""),
                    help="STOP file: refuse to start while it exists; written on any trip")
    ap.add_argument("--timeout-s", type=int, default=300)
    ap.add_argument("--tlimit-min-c", type=float, default=5.0)
    ap.add_argument("--temp-max-c", type=float, default=88.0)
    ap.add_argument("--stale-s", type=float, default=5.0)
    ap.add_argument("--max-idle-mib", type=float, default=2048)
    ap.add_argument("--quiet-before-s", type=int, default=60)
    ap.add_argument("--replay-every-s", type=float, default=10.0)
    ap.add_argument("--post-s", type=float, default=5.0)
    ap.add_argument("--fail-pattern", default=FAIL_DEFAULT)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    if a.cmd and a.cmd[0] == "--":
        a.cmd = a.cmd[1:]
    if not a.cmd:
        ap.error("no command after --")
    # resolve through PATH: on Windows a bare 'bash' would otherwise start System32\bash.exe (WSL, ~10 s, other OS)
    a.cmd[0] = shutil.which(a.cmd[0]) or a.cmd[0]
    os.makedirs(a.out, exist_ok=True)
    g = Guard(a)
    t0 = now()

    def refuse(msg):
        raise Refused(msg)

    try:
        # Validate before any process (including monitors) can be started.
        pat = re.compile(a.fail_pattern)
        for name in ("timeout_s", "stale_s", "replay_every_s", "temp_max_c"):
            value = getattr(a, name)
            if not math.isfinite(value) or value <= 0:
                refuse(f"{name} must be finite and positive")
        for name in ("post_s", "quiet_before_s", "max_idle_mib", "tlimit_min_c"):
            value = getattr(a, name)
            if not math.isfinite(value) or value < 0:
                refuse(f"{name} must be finite and nonnegative")
        if a.latch and os.path.exists(a.latch):
            refuse("STOP latch exists")
        uuid, info = resolve(a.gpu_name, a.pci)
        if not uuid:
            refuse(f"no GPU named '{a.gpu_name}' {a.pci}: {info[:200]}")
        g.log(f"GPU {info}")
        rc, out = smi(["-i", uuid, "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
        memory = num(out.strip())
        if rc or memory is None or memory < 0 or memory > a.max_idle_mib:
            refuse(f"invalid or busy GPU memory before launch: {out.strip()[:80]} MiB")
        supervise(g, uuid, pat)
    except Refused as e:
        g.trip = (96, "refused: " + str(e))
    except KeyboardInterrupt:
        g.fire(130, "guard interrupted")
    except Exception as e:
        g.fire(97, f"guard failed: {type(e).__name__}: {e}")
    finally:
        # Do not allow a second Ctrl+C to interrupt owned-process cleanup / STOP persistence.
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            g.close()
        finally:
            signal.signal(signal.SIGINT, previous)
    child_rc = g.child.returncode if g.child else None
    code = g.trip[0] if g.trip else (child_rc or 0)
    if g.latch_error:
        code = 97
    if not g.launched and code != 130:
        code = 96
    g.summary(child_rc, code, t0, now())
    return code


if __name__ == "__main__":
    sys.exit(main())
