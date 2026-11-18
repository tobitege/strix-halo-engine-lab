#!/usr/bin/env python3
"""Read-only KFD eviction sampler: logs every evicted_ms change of each KFD process with
/proc/vmstat + meminfo deltas over the previous interval. Usage: evsample.py SECONDS OUT [PERIOD]"""
import os, sys, time, glob
dur = float(sys.argv[1]); out = open(sys.argv[2], 'w', newline=''); per = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1
VM = ('pgsteal_kswapd', 'pgsteal_direct', 'pswpout', 'pswpin', 'pgmigrate_success', 'pgmigrate_fail', 'compact_stall',
      'compact_migrate_scanned', 'compact_isolated', 'compact_daemon_wake', 'thp_collapse_alloc', 'thp_fault_alloc',
      'workingset_refault_file', 'allocstall_normal', 'pgfault', 'pgmajfault', 'nr_mlock')
MI = ('MemAvailable', 'Cached', 'GPUActive', 'SwapFree', 'Mlocked', 'MemFree')
def vm():
    d = {}
    for l in open('/proc/vmstat'):
        k, v = l.split(); 
        if k in VM: d[k] = int(v)
    return d
def mi():
    d = {}
    for l in open('/proc/meminfo'):
        k, v = l.split(':'); 
        if k in MI: d[k] = int(v.split()[0]) // 1024
    return d
def ev():
    d = {}
    for p in glob.glob('/sys/class/kfd/kfd/proc/*'):
        pid = os.path.basename(p); t = 0
        for f in glob.glob(p + '/stats_*/evicted_ms'):
            try: t += int(open(f).read())
            except Exception: pass
        try: comm = open('/proc/%s/comm' % pid).read().strip()
        except Exception: comm = '?'
        d[pid] = (t, comm)
    return d
t0 = time.time(); v0 = vm(); e0 = ev(); vsec = [(t0, v0)]
out.write('# start %s host %s real=%.6f boot=%.6f\n' % (time.strftime('%H:%M:%S'), os.uname()[1], t0, time.clock_gettime(time.CLOCK_BOOTTIME)))
BUSY = next((f for f in glob.glob('/sys/class/drm/card*/device/gpu_busy_percent')), None); last_b = 0
for pid, (t, c) in e0.items(): out.write('# proc %s %s ev0=%d\n' % (pid, c, t))
last_mi = 0
while time.time() - t0 < dur:
    time.sleep(per); now = time.time(); v = vm(); e = ev()
    vsec.append((now, v)); vsec = [x for x in vsec if now - x[0] <= 2.0]
    for pid, (t, c) in e.items():
        if pid not in e0: out.write('%.2f %s NEW %s ev=%d\n' % (now - t0, time.strftime('%H:%M:%S'), c, t)); continue
        if t != e0[pid][0]:
            base = vsec[0][1]; dl = ' '.join('%s=%d' % (k, v[k] - base[k]) for k in VM if k in v and v[k] != base.get(k))
            m = mi(); out.write('%.2f %s EV %s %s +%d (=%d) | 2s: %s | MA=%d C=%d GA=%d SF=%d ML=%d\n' % (now - t0, time.strftime('%H:%M:%S'), pid, c,
                t - e0[pid][0], t, dl, m['MemAvailable'], m['Cached'], m.get('GPUActive', 0), m['SwapFree'], m['Mlocked']))
    for pid in e0:
        if pid not in e: out.write('%.2f %s GONE %s %s ev=%d\n' % (now - t0, time.strftime('%H:%M:%S'), pid, e0[pid][1], e0[pid][0]))
    if BUSY and now - last_b >= 0.25:
        last_b = now
        try: b = int(open(BUSY).read())
        except Exception: b = -1
        out.write('B %.2f %d %s\n' % (now - t0, b, ','.join('%d' % t for t, c in e.values())))
    if now - last_mi >= 10:
        m = mi(); last_mi = now; out.write('%.2f %s MEM %s\n' % (now - t0, time.strftime('%H:%M:%S'), ' '.join('%s=%d' % kv for kv in m.items())))
    out.flush(); e0 = e
d = vm(); out.write('# end totals: %s\n' % ' '.join('%s=%d' % (k, d[k] - v0[k]) for k in VM if k in d))
