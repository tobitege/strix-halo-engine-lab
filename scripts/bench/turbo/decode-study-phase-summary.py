#!/usr/bin/env python3
"""Summarize emitted-cycle receipts; runtime verify includes commit bookkeeping."""
import argparse,collections,json,re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('runs',nargs='+');p.add_argument('--warm-only',action='store_true');a=p.parse_args()
for name in a.runs:
 d=Path(name)
 try:err=(d/'srv.err').read_text(errors='replace');requests=[json.loads(x) for x in (d/'res.jsonl').read_text().splitlines()]
 except Exception as e:print(d.name,'ERROR',str(e));continue
 units=[]
 for cohort_pass,request in enumerate(requests,1):
  if request.get('arm','').startswith('S'):
   for lane in request['lanes']:
    if not lane or 'error' in lane:continue
    unit=dict(request);unit['lanes']=[lane];unit['_cohort_pass']=cohort_pass;unit['_class_lane']=lane['lane'];units.append(unit)
  else:
   request['_cohort_pass']=cohort_pass;request['_class_lane']=None;units.append(request)
 requests=units
 cycles=[];host={}
 for line in err.splitlines():
  if line.startswith('serve cycle '):
   c=dict(re.findall(r'(\w+)=([^ ]+)',line))
   if int(c.get('emitted',0))>0:cycles.append(c)
  m=re.search(r'cycle host timing\s+(.+?)\s+(\d+)\s+x\s+([0-9.]+) ms',line)
  if m:host[m[1].strip()]={'n':int(m[2]),'mean_ms':float(m[3])}
 groups=[];scope='unresolved';counts=[max([r.get('cycles') or 0 for r in req['lanes'] if r and 'error' not in r] or [0]) for req in requests]
 anchor=d/'clock-anchor.json'
 if anchor.exists():
  clk=json.loads(anchor.read_text());offset=clk['epoch_ns']-clk['monotonic_ns'];groups=[]
  for req in requests:
   lanes=[r for r in req['lanes'] if r and 'error' not in r]
   l=req['t0']+min(r['start'] for r in lanes);r=req['t0']+max(r['end'] for r in lanes)
   groups.append([c for c in cycles if l<=(int(c['begin_ns'])+offset)/1e9 and (int(c['end_ns'])+offset)/1e9<=r])
  if [len(g) for g in groups]==counts:scope='paired epoch/monotonic anchor + client spans'
  else:groups=[]
 if not groups and sum(counts)==len(cycles):
  start=0
  for n in counts:groups.append(cycles[start:start+n]);start+=n
  scope='sequential cycle-count partition (anchor unavailable/mismatch)'
 if not groups:groups=[cycles];scope='process aggregate; request attribution unresolved'
 clock=[];proc=[]
 if (d/'clk.txt').exists():
  for line in (d/'clk.txt').read_text(errors='replace').splitlines():
   m=re.search(r'^([0-9.]+).*?(\d+)Mhz.*busy=(\d+)(?:.*mem_kb=(\d+))?',line)
   if m:clock.append((float(m[1]),float(m[2]),float(m[3])/100,int(m[4]) if m[4] else None))
 if (d/'startup-proc.jsonl').exists():
  proc=[json.loads(x) for x in (d/'startup-proc.jsonl').read_text().splitlines()]
 summary=[]
 for i,g in enumerate(groups):
  sums={k:sum(float(c[k]) for c in g) for k in ('draft_ms','verify_ms','host_ms','ms')};n=len(g);tok=sum(int(c['emitted']) for c in g)
  physical=collections.Counter(int(c['rows']) for c in g);useful=collections.Counter(int(c.get('useful',0)) for c in g);pad=sum(int(c.get('pad',0)) for c in g);depth=[]
  for c in g:
   vals=list(map(int,c.get('draft_rows','').split('/'))) if c.get('draft_rows') else []
   if len(depth)<len(vals):depth.extend([0]*(len(vals)-len(depth)))
   for idx,val in enumerate(vals):depth[idx]+=val
  req=requests[i] if len(groups)==len(requests) else None;lanes=[r for r in req['lanes'] if r and 'error' not in r] if req else []
  complete=sum(r['tokens'] for r in lanes);proposed=sum(r.get('prop') or 0 for r in lanes);accepted=sum(r.get('acc') or 0 for r in lanes)
  gpu={'source':'250ms sampled frequency x gpu_busy_percent within receipt intervals','coverage':0.0};faults={}
  if anchor.exists():
   interval=[((int(c['begin_ns'])+offset)/1e9,(int(c['end_ns'])+offset)/1e9) for c in g];covered=0.0;mc=0.0;fm=0.0;busy=0.0;mem=[]
   for z,(t,f,b,memory) in enumerate(clock):
    stop=clock[z+1][0] if z+1<len(clock) else t+.25
    dt=sum(max(0.0,min(stop,r)-max(t,l)) for l,r in interval)
    if dt:covered+=dt;mc+=dt*f*b;fm+=dt*f;busy+=dt*b;mem.extend([memory] if memory is not None else [])
   duration=sum(r-l for l,r in interval)
   gpu.update({'coverage':covered/max(.0001,duration),'mean_mhz':fm/max(.0001,covered),'mean_busy':busy/max(.0001,covered),'Mbusycycles_per_cycle_emitted_token':mc/max(1,tok),'min_MemAvailable_GiB':min(mem)/1048576 if mem else None})
  if req and lanes and proc:
   begin=req['t0']+min(r['start'] for r in lanes);end=req['t0']+max(r['end'] for r in lanes);sample=[r for r in proc if begin<=r['epoch_ns']/1e9<=end]
   if len(sample)>=2:
    first,last=sample[0],sample[-1];faults={'source':'process counters at samples within client request; excludes unsampled edges','coverage':(last['epoch_ns']-first['epoch_ns'])/1e9/max(.0001,end-begin),'minor':last['minflt']-first['minflt'],'major':last['majflt']-first['majflt'],'read_bytes':last['read_bytes']-first['read_bytes']}
  rec={'pass':req.get('_cohort_pass',i+1) if req else i+1,'class_lane':req.get('_class_lane') if req else None,'scope':scope,'cycles':n,'cycle_emitted':tok,'client_completion_tokens':complete,'prefill_or_noncycle_emitted':complete-tok,'E':tok/max(1,n),'physical_rows_histogram':dict(physical),'useful_rows_histogram':dict(useful),'padding_rows':pad,'candidate_depth_lane_counts':depth,'json_proposed':proposed,'json_accepted':accepted,'sum_ms':sums,'mean_ms':{k:v/max(1,n) for k,v in sums.items()},'receipt_tps':tok/max(.0001,sums['ms']/1000),'sampled_gpu':gpu,'sampled_request_process_faults':faults,'prompt_metrics':[{'prompt':r.get('prompt'),'prefill_ms':r.get('prefill_ms'),'queue_ms':r.get('queue_ms')} for r in lanes]};summary.append(rec)
  if a.warm_only and rec['pass']==1:continue
  print(d.name,'pass',rec['pass'],'lane',rec['class_lane'],'cycle_tok',tok,'passes',n,'E',round(rec['E'],4),'ms/pass',{k:round(v,4) for k,v in rec['mean_ms'].items()},'cycle_tps',round(rec['receipt_tps'],3))
  print(d.name,'cycle-clock','MHz',round(gpu.get('mean_mhz',0)),'busy',round(gpu.get('mean_busy',0),4),'Mbusycycles/cycle-token',round(gpu.get('Mbusycycles_per_cycle_emitted_token',0),4),'MemAvailable_GiB',round(gpu['min_MemAvailable_GiB'],2) if gpu.get('min_MemAvailable_GiB') is not None else None)
  print(d.name,'work','physical',dict(physical),'useful',dict(useful),'pad',pad,'client_first_or_noncycle',complete-tok,'accepted/proposed',f'{accepted}/{proposed}','depth_lane_counts',depth)
 (d/'cycle-receipt-summary.json').write_text(json.dumps(summary,indent=2)+'\n');(d/'phase-marks.json').write_text(json.dumps(host,indent=2)+'\n')
 top=sorted(host.items(),key=lambda kv:kv[1]['n']*kv[1]['mean_ms'],reverse=True)[:6]
 print(d.name,'host_exit_top6',','.join(k+':'+str(v['n'])+'x'+format(v['mean_ms'],'.3f') for k,v in top)[:250])
