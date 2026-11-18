#!/usr/bin/env python3
"""meminfo.py: one-line host memory state (GiB) recorded in every run's meta.json."""
values = {}
try:
    stream = open("/proc/meminfo")
except OSError:  # not Linux (RTX workstation): host memory is not part of the receipt there
    print("host-memory=n/a")
    raise SystemExit(0)
with stream:
    for line in stream:
        part = line.split()
        if part[0] in ("MemFree:", "MemAvailable:", "Cached:"):
            values[part[0][:-1]] = int(part[1]) / 1048576.0
print("free=%.1f cached=%.1f avail=%.1f" % (values.get("MemFree", 0), values.get("Cached", 0), values.get("MemAvailable", 0)))
