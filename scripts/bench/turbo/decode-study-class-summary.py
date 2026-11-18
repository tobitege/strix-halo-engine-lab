#!/usr/bin/env python3
"""Summarize fixed-class emitted tokens and sampled GPU work per token."""
import argparse, bisect, json, re, sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');a=p.parse_args()
classes=['code','prose','math','tools']
failed = False
for run in a.runs:
 root=Path(run);clock=[]
 try:
  for s in (root/'clk.txt').read_text(errors='replace').splitlines():
   m=re.search(r'^([0-9.]+).*?(\d+)Mhz.*busy=(\d+)',s)
   if m:clock.append((float(m[1]),float(m[2]),float(m[3])/100))
  rows=[json.loads(s) for s in (root/'res.jsonl').read_text().splitlines()]
  if not rows: raise ValueError('no request records')
 except Exception as e:print(root.name,'ERROR',str(e));failed=True;continue
 for request,row in enumerate(rows):
  if row.get('errors') or len(row['lanes']) != int(row['arm'][1:]): failed = True
  for r in row['lanes']:
   if not r or 'error' in r or r['tokens'] <= 0:print(root.name,request,'ERROR',r);failed=True;continue
   tok=r['tokens'];cy=r.get('cycles') or 0;acc=r.get('acc') or 0;prop=r.get('prop') or 0
   cycle=acc+cy;cycle_valid=(tok-cycle==1);cycle_E=round(cycle/max(1,cy),3) if cycle_valid else None
   start=row['t0']+r['start']+((r.get('queue_ms') or 0)+(r.get('prefill_ms') or 0))/1000
   end=row['t0']+r['end'];win=end-start;weighted=0;observed=0;mhz_sum=0;busy_sum=0
   for i,(t,mhz,busy) in enumerate(clock):
    stop=clock[i+1][0] if i+1<len(clock) else t+.25
    dt=max(0,min(stop,end)-max(t,start))
    weighted+=dt*mhz*busy;observed+=dt;mhz_sum+=dt*mhz;busy_sum+=dt*busy
   lane=r['lane'];cls=classes[lane] if lane<len(classes) else str(lane)
   print(root.name,'pass',request+1,cls,'prompt',r.get('prompt'),'tok',tok,'cycles',cy,'E_cycle',cycle_E,'E_client_legacy',round(tok/max(1,cy),3),'first_token_check',cycle_valid,'accepted',acc,'proposed',prop,'accept',round(acc/max(1,prop),4),'tps',round(tok/max(.0001,win),2),'mhz',round(mhz_sum/max(.0001,observed)),'busy',round(busy_sum/max(.0001,observed),3),'Mbusycyc/client_tok_legacy',round(weighted/max(1,tok),3),'clockcoverage',round(observed/max(.0001,win),3))
sys.exit(1 if failed else 0)
