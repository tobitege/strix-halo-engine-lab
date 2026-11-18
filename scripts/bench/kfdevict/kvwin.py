#!/usr/bin/env python3
"""kvwin.py EVLOG...: sampler EV lines (evicted_ms jumps with 2 s vmstat deltas) that fall inside GPU busy windows (busy >= 80 %, >= 3 s)."""
import sys,re
sys.argv.pop(0)
for f in sys.argv:
    B=[];EV=[]
    for l in open(f,errors='replace'):
        p=l.split()
        if l.startswith('B '): B.append((float(p[1]),int(p[2])))
        elif len(p)>2 and p[2]=='EV': EV.append(l.rstrip())
    sg=[];cur=None
    for t,b in B:
        if b>=80:
            if cur and t-cur[1]<=1.0: cur[1]=t
            else:
                if cur: sg.append(cur)
                cur=[t,t]
    if cur: sg.append(cur)
    sg=[x for x in sg if x[1]-x[0]>=3]
    print('==',f,'busy',' '.join('%.0f-%.0f'%(a,b) for a,b in sg))
    for l in EV:
        t=float(l.split()[0])
        if any(a-0.3<=t<=b+0.3 for a,b in sg):
            l=re.sub(r'pgfault=\d+ ?','',l); l=re.sub(r'workingset_refault_file','wrf',l); l=re.sub(r'compact_','c_',l)
            print(l[:260])
