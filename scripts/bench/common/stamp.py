#!/usr/bin/env python3
"""stamp.py FILE: copy stdin to FILE and write FILE.ts with the arrival time of every line (receipt windows)."""
import sys
import time

with open(sys.argv[1], "w", buffering=1, errors="replace") as raw, open(sys.argv[1] + ".ts", "w", buffering=1) as ts:
    for line in sys.stdin.buffer:
        text = line.decode("utf-8", "replace")
        raw.write(text)
        ts.write("%.3f %s\n" % (time.time(), text[:100].rstrip()))
