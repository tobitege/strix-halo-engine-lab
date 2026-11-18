"""Per-kernel-class table of the LAST step in an nsys cuda_gpu_trace CSV (graph node trace).
usage: python nsys_step_table.py trace.csv [marker_kernel_substring=set_inputs_kernel]"""
import csv, re, sys, collections
rows = list(csv.reader(open(sys.argv[1], encoding='utf-8')))
h = rows[0]; ix = {n: i for i, n in enumerate(h)}
marker = sys.argv[2] if len(sys.argv) > 2 else 'set_inputs_kernel'
ev = [(int(r[ix['Start (ns)']]), int(r[ix['Duration (ns)']]), int(r[ix['GrdX']]), int(r[ix['GrdY']]), int(r[ix['GrdZ']]), int(r[ix['BlkX']]), r[ix['Name']]) for r in rows[1:] if len(r) > ix['Name'] and r[ix['GrdX']]]
ev.sort()
last = max(i for i, e in enumerate(ev) if marker in e[6])
st = ev[last:]
# one step = from the marker up to the next marker or the end
nxt = [i for i, e in enumerate(st[1:], 1) if marker in e[6]]
st = st[:nxt[0]] if nxt else st
span = st[-1][0] + st[-1][1] - st[0][0]
tot = sum(e[1] for e in st)
def short(n):
    n = re.sub(r'\(.*', '', n); n = n.replace('strix::cuda_core::', '').replace('strix::qwen35::', '').replace('<unnamed>::', '')
    return n[:70]
g = collections.OrderedDict()
for s, d, gx, gy, gz, bx, n in st:
    k = (short(n), gx, gy, gz, bx)
    c = g.setdefault(k, [0, 0]); c[0] += 1; c[1] += d
print('kernels %d span %.3f ms, sum(kernel) %.3f ms, gaps %.3f ms' % (len(st), span / 1e6, tot / 1e6, (span - tot) / 1e6))
for k, (c, d) in sorted(g.items(), key=lambda kv: -kv[1][1]):
    print('%-62s grid %5d,%d,%d blk %4d  n %3d  tot %8.1f us  mean %7.2f us' % (k[0][:62], k[1], k[2], k[3], k[4], c, d / 1e3, d / 1e3 / c))
