#!/usr/bin/env python3
"""Pair-run helpers (no root needed).

pf_samp.py sample OUT SECONDS : 20 ms GPU/memory sampler. Columns: epoch t, sclk MHz, gpu busy %, power W
    (hwmon power1_average), KFD evicted_ms summed over processes, MemFree MiB, MemAvailable MiB.
pf_samp.py stamp OUT : copy stdin to OUT, each line prefixed with the epoch time (ms resolution).
"""
import glob
import re
import sys
import time


def read(path, default):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return default


def sample(out_path, seconds):
    end = time.time() + seconds
    sclk = (glob.glob('/sys/class/drm/card*/device/pp_dpm_sclk') or [''])[0]
    busy = sclk.replace('pp_dpm_sclk', 'gpu_busy_percent')
    power = (glob.glob(sclk.replace('pp_dpm_sclk', 'hwmon/hwmon*/power1_average')) or [''])[-1]
    with open(out_path, 'w') as out:
        out.write('t sclk busy w evict free avail\n')
        while time.time() < end:
            t = time.time()
            m = re.search(r'(\d+)Mhz \*', read(sclk, ''))
            b = read(busy, '-1').strip() or '-1'
            w = read(power, '')
            w = '%.2f' % (int(w) / 1e6) if w.strip().isdigit() else '-1'
            ev = 0
            for f in glob.glob('/sys/class/kfd/kfd/proc/*/stats_*/evicted_ms'):
                v = read(f, '0').strip()
                ev += int(v) if v.isdigit() else 0
            mi = read('/proc/meminfo', '')
            fr = re.search(r'MemFree:\s+(\d+)', mi)
            av = re.search(r'MemAvailable:\s+(\d+)', mi)
            out.write('%.3f %s %s %s %d %d %d\n' % (t, m.group(1) if m else '0', b, w, ev,
                                                    int(fr.group(1)) // 1024 if fr else -1,
                                                    int(av.group(1)) // 1024 if av else -1))
            out.flush()
            time.sleep(max(0.0, 0.02 - (time.time() - t)))


def stamp(out_path):
    with open(out_path, 'w') as out:
        for line in sys.stdin:
            out.write('%.3f %s' % (time.time(), line))
            out.flush()


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == 'sample':
        sample(sys.argv[2], float(sys.argv[3]))
    elif len(sys.argv) == 3 and sys.argv[1] == 'stamp':
        stamp(sys.argv[2])
    else:
        sys.exit(__doc__)
