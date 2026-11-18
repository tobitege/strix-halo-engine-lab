#!/usr/bin/env python3
"""Transparent native stdout proxy and exact committed-token peak collector.

The caller declares full-line regexes with a named decimal ``token`` group and optional ``slot``.
Only declared committed-token protocol lines count. No text tokenization or draft estimate is used.
"""
import argparse
import ctypes
import importlib.util
import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import threading
import time


def windows_job(proc):
    """Close-on-parent-death ownership for the child tree, including forced wrapper termination."""
    if os.name != "nt":
        return None
    from ctypes import wintypes as w

    class Basic(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64), ("flags", w.DWORD),
                    ("minimum", ctypes.c_size_t), ("maximum", ctypes.c_size_t), ("active", w.DWORD),
                    ("affinity", ctypes.c_size_t), ("priority", w.DWORD), ("scheduling", w.DWORD)]

    class Extended(ctypes.Structure):
        _fields_ = [("basic", Basic), ("io", ctypes.c_uint64 * 6), ("process_memory", ctypes.c_size_t),
                    ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
    kernel.CreateJobObjectW.restype = w.HANDLE
    kernel.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
    kernel.SetInformationJobObject.restype = w.BOOL
    kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
    kernel.AssignProcessToJobObject.restype = w.BOOL
    kernel.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
    kernel.OpenProcess.restype = w.HANDLE
    kernel.CloseHandle.argtypes = [w.HANDLE]
    job = kernel.CreateJobObjectW(None, None)
    child = None
    try:
        limits = Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not job or not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            raise ctypes.WinError(ctypes.get_last_error())
        child = kernel.OpenProcess(0x0101, False, proc.pid)  # SET_QUOTA | TERMINATE
        if not child or not kernel.AssignProcessToJobObject(job, child):
            raise ctypes.WinError(ctypes.get_last_error())
        return kernel, job
    except BaseException:
        if job:
            kernel.CloseHandle(job)
        raise
    finally:
        if child:
            kernel.CloseHandle(child)


def patterns(values):
    result = [re.compile(value) for value in values]
    if not result or any("token" not in pattern.groupindex for pattern in result):
        raise ValueError("every full-line pattern needs a named token group")
    return result


def token_event(line, expressions):
    matches = [match for pattern in expressions if (match := pattern.fullmatch(line))]
    if len(matches) > 1:
        raise ValueError("ambiguous committed-token protocol line")
    if not matches:
        return None
    match = matches[0]
    token = int(match.group("token"))
    if token < 0:
        raise ValueError("negative token ID")
    return dict(token=token, slot=match.groupdict().get("slot"))


def run(args):
    expressions = patterns(args.token_pattern)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("missing native command")
    args.out = pathlib.Path(str(args.out).replace("{pid}", str(os.getpid())).replace("{time_ns}", str(time.time_ns())))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Unbuffered writes survive an owned server's forced process-tree cleanup. This is an instrumented peak run.
    with args.out.open("xb", buffering=0) as log:
        def record(value):
            log.write((json.dumps(value, separators=(",", ":")) + "\n").encode())

        record(dict(schema="native-emissions-v1", clock="perf_counter_ns", pid=os.getpid(),
                    overhead="stdout proxy and unbuffered per-token trace; separate from throughput"))
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        job = None
        try:
            job = windows_job(proc)

            def copy_input():
                try:
                    while data := os.read(sys.stdin.fileno(), 65536):
                        proc.stdin.write(data)
                        proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
                finally:
                    try:
                        proc.stdin.close()
                    except (BrokenPipeError, OSError):
                        pass

            threading.Thread(target=copy_input, daemon=True).start()
            for raw in iter(proc.stdout.readline, b""):
                stamp = time.perf_counter_ns()
                event = token_event(raw.decode("utf-8", errors="replace").rstrip("\r\n"), expressions)
                if event is not None:
                    record(dict(steady_ns=stamp, **event))
                sys.stdout.buffer.write(raw)
                sys.stdout.buffer.flush()
            code = proc.wait()
            record(dict(end=True, returncode=code))
            return code
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            if job is not None:
                job[0].CloseHandle(job[1])


def peak(trace, receipt):
    lines = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
    if not lines or lines[0].get("schema") != "native-emissions-v1" or lines[0].get("clock") != "perf_counter_ns":
        raise ValueError("unsupported native trace contract")
    events = json.loads(receipt.read_text(encoding="utf-8"))["events"]
    if not events or any(event.get("error") or type(event.get("tokens")) is not int
                         or "finished" not in event for event in events):
        raise ValueError("incomplete client events or final usage")
    begin = min(event["arrival"] for event in events) * 1e9
    end = max(event["finished"] for event in events) * 1e9
    stamps = []
    previous = -1
    for line in lines[1:]:
        if "steady_ns" not in line:
            continue
        stamp = line["steady_ns"]
        if type(stamp) is not int or stamp < previous or type(line.get("token")) is not int or line["token"] < 0:
            raise ValueError("invalid or nonmonotonic native publication")
        previous = stamp
        if begin <= stamp <= end:
            stamps.append(stamp)
    expected = sum(event["tokens"] for event in events)
    result = dict(schema="native-emission-peak-v1", valid=len(stamps) == expected, measured_tokens=len(stamps),
                  usage_tokens=expected, peak_1s_tok_s=None, scope="native committed IDs, stop tokens included as in usage",
                  clock="same-host perf_counter_ns", overhead=lines[0].get("overhead"))
    if not result["valid"]:
        result["reason"] = "native emitted count differs from summed final usage"
        return result
    spec = importlib.util.spec_from_file_location("serve_bench", pathlib.Path(__file__).resolve().parents[3] / "tools/serve_bench.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    origin = min(stamps) if stamps else 0
    result["peak_1s_tok_s"] = bench.peak_1s([(stamp - origin) / 1e9 for stamp in stamps], [1] * len(stamps))
    if result["peak_1s_tok_s"] is None:
        result["valid"] = False
        result["reason"] = "native emissions span less than one second"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--out", type=pathlib.Path, required=True)
    runner.add_argument("--token-pattern", action="append", required=True)
    runner.add_argument("command", nargs=argparse.REMAINDER)
    summary = commands.add_parser("peak")
    summary.add_argument("--trace", type=pathlib.Path, required=True)
    summary.add_argument("--receipt", type=pathlib.Path, required=True)
    summary.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "run":
        if os.name != "nt":
            def stop(signum, _):
                raise SystemExit(128 + signum)
            signal.signal(signal.SIGTERM, stop)
        return run(args)
    try:
        result = peak(args.trace, args.receipt)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        result = dict(schema="native-emission-peak-v1", valid=False, peak_1s_tok_s=None, reason=str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))
    return 0 if result["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
