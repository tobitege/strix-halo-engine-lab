#!/usr/bin/env python3
"""Summarize ROCprof traces within emitted-token serving cycles (no startup kernels)."""
import argparse, bisect, collections, csv, glob, json, re
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('trace');p.add_argument('--limit',type=int,default=14);p.add_argument('--scope',choices=('c1','c8','all'),default='c1');a=p.parse_args()
root=Path(a.run);err=(root/'srv.err').read_text(errors='replace'); intervals=[];cycle_meta=[];phase=collections.defaultdict(float);rows=collections.Counter();emitted=0
anchor_pre=json.loads((root/'clock-anchor.json').read_text()) if (root/'clock-anchor.json').exists() else None
client_scope_spans=[]
for kind,filename in (('c1','res.jsonl'),('c8','c8-res.jsonl')):
 if not (root/filename).exists():continue
 for q,line in enumerate((root/filename).read_text().splitlines()):
  req=json.loads(line);valid=[r for r in req['lanes'] if r and 'error' not in r]
  if valid:client_scope_spans.append((int((req['t0']+min(r['start'] for r in valid))*1e9),int((req['t0']+max(r['end'] for r in valid))*1e9),kind,q+1))
def wanted_scope(epoch_ns):
 return a.scope=='all' or any(l<=epoch_ns<=r and kind==a.scope for l,r,kind,q in client_scope_spans)
for line in err.splitlines():
 if not line.startswith('serve cycle '):continue
 d=dict(re.findall(r'(\w+)=([^ ]+)',line));n=int(d.get('emitted',0))
 if n<=0:continue
 if anchor_pre and not wanted_scope(int(d['begin_ns'])+anchor_pre['epoch_ns']-anchor_pre['monotonic_ns']):continue
 intervals.append((int(d['begin_ns']),int(d['end_ns'])));cycle_meta.append(d);emitted+=n;rows[int(d.get('rows',0))]+=1
 for k in ('draft_ms','verify_ms','host_ms','ms'):phase[k]+=float(d[k])
anchor_path=root/'clock-anchor.json'
if anchor_path.exists():
 anchor=json.loads(anchor_path.read_text());offset=anchor['boottime_ns']-anchor['monotonic_ns']
 intervals=[(l+offset,r+offset) for l,r in intervals]
 print('clock_alignment','serve CLOCK_MONOTONIC -> profiler CLOCK_BOOTTIME','offset_ns',offset,'anchor',anchor)
else:
 print('clock_alignment','UNSEALED no paired clock anchor; interval selection is diagnostic only')
intervals.sort();starts=[x[0] for x in intervals]
def cycle_index(x):
 i=bisect.bisect_right(starts,x)-1
 return i if i>=0 and x<=intervals[i][1] else -1
def inc(x):return cycle_index(x)>=0
spans=[]
if anchor_path.exists():
 boot_epoch=anchor['boottime_ns']-anchor['epoch_ns']
 spans=[(l+boot_epoch,r+boot_epoch,f'{kind.upper()}-request{q}') for l,r,kind,q in client_scope_spans]
def class_key(ci):
 t=intervals[ci][0];request=next((q for l,r,q in spans if l<=t<=r),0)
 return f"{request}/R{cycle_meta[ci]['rows']}"
byclass={}
for ci,meta in enumerate(cycle_meta):
 key=class_key(ci);c=byclass.setdefault(key,{'cycles':0,'emitted':0,'kernels':{},'gpu_intervals':[],'host_phases':{},'hip_api':{}});c['cycles']+=1;c['emitted']+=int(meta['emitted'])
def merge_ns(v):
 end=0;total=0
 for l,r in sorted(v):
  if r>end:total+=r-max(l,end);end=r
 return total
agg=collections.defaultdict(list);allagg=collections.defaultdict(list);gpu=[];calls=0;scratch=0;raw=[]
files=glob.glob(str(Path(a.trace)/'**/*kernel_trace.csv'),recursive=True)
for f in files:
 for d in csv.DictReader(open(f)):
  l=int(d['Start_Timestamp']);r=int(d['End_Timestamp']);name=d['Kernel_Name'];allagg[name].append((r-l)/1e3)
  if not inc(l):continue
  calls+=1;gpu.append((l,r));scratch+=int(int(d.get('Scratch_Size','0'))>0)
  key=re.sub(r'\(anonymous namespace\)::|strix::|q8_rows::','',name)
  m=re.search(r'(project_rows_\w+)<(\w+Site)<(\d+)u?',key)
  key=(m.group(1)+' '+m.group(2)+' R'+m.group(3)) if m else re.sub(r'^void ','',key).split('(')[0][:115]
  agg[key].append((r-l)/1e3)
  c=byclass[class_key(cycle_index(l))];c['kernels'].setdefault(key,[]).append((r-l)/1e3);c['gpu_intervals'].append((l,r))
print('trace_files',len(files),'cycles',len(intervals),'emitted',emitted,'rows',dict(rows))
print('phase_ms',dict(phase),'E',round(emitted/max(1,len(intervals)),4))
print('decode_kernel_calls',calls,'gpu_busy_ms',round(merge_ns(gpu)/1e6,3),'sum_kernel_ms',round(sum(map(sum,agg.values()))/1e3,3),'scratch_calls',scratch)
print('decode kernel | calls | mean_us | total_ms')
for k,v in sorted(agg.items(),key=lambda kv:sum(kv[1]),reverse=True)[:a.limit]:print(k,len(v),round(sum(v)/len(v),3),round(sum(v)/1e3,3),sep=' | ')
if not calls:
 print('UNALIGNED: zero selected kernels; no kernel absence verdict. Aggregate trace below includes startup.')
 for n,v in sorted(allagg.items(),key=lambda kv:sum(kv[1]),reverse=True)[:a.limit]:print('UNSCOPED',n[:145],len(v),round(sum(v)/len(v),3),round(sum(v)/1e3,3),sep=' | ')
 raise SystemExit(2)
print('decode projection sites (all)')
for k,v in sorted(agg.items()):
 if 'RowsSite' in k:print(k,len(v),round(sum(v)/len(v),3),round(sum(v)/1e3,3),sep=' | ')
host=collections.defaultdict(list)
for match in re.finditer(r'slow phase t=(\d+) bt=(\d+) (.+?) ([0-9.]+)\s*$',err,re.M):
 if inc(int(match[2])):
  host[match[3]].append(float(match[4]));c=byclass[class_key(cycle_index(int(match[2])))];c['host_phases'].setdefault(match[3],[]).append(float(match[4]))
(root/f'decode-{a.scope}-host-phases.json').write_text(json.dumps(dict(host),indent=2)+'\n')
print('scoped live host phases | calls | mean_ms | total_ms')
for n,v in sorted(host.items(),key=lambda kv:sum(kv[1]),reverse=True)[:10]:print(n,len(v),round(sum(v)/len(v),4),round(sum(v),4),sep=' | ')
api=collections.Counter();apit=collections.defaultdict(float)
for f in glob.glob(str(Path(a.trace)/'**/*hip_api_trace.csv'),recursive=True):
 for d in csv.DictReader(open(f)):
  if not inc(int(d['Start_Timestamp'])):continue
  n=d.get('Function',d.get('Name',d.get('Operation','unknown')));api[n]+=1;duration=(int(d['End_Timestamp'])-int(d['Start_Timestamp']))/1e6;apit[n]+=duration
  c=byclass[class_key(cycle_index(int(d['Start_Timestamp'])))];v=c['hip_api'].setdefault(n,{'calls':0,'ms':0.0});v['calls']+=1;v['ms']+=duration
for key,c in sorted(byclass.items()):
 c['gpu_busy_ms']=merge_ns(c.pop('gpu_intervals'))/1e6
 c['sum_kernel_ms']=sum(sum(v) for v in c['kernels'].values())/1000
 target={n:round(sum(v)/len(v),4) for n,v in c['host_phases'].items() if any(word in n for word in ('window launch','verify tail','commit','reconcile'))}
 print('row_scope',key,'cycles',c['cycles'],'GPUms/pass',round(c['gpu_busy_ms']/max(1,c['cycles']),4),'host_phase_mean_ms',target)
(root/f'decode-{a.scope}-row-classes.json').write_text(json.dumps(byclass,indent=2)+'\n')
if a.scope=='c1':(root/'decode-row-classes.json').write_text(json.dumps(byclass,indent=2)+'\n')
print('hip_api top',[(k,v,round(apit[k],3)) for k,v in api.most_common(8)])
print('startup_or_decode_quantize_kernels',[(n[:110],len(v),round(sum(v)/1e3,3)) for n,v in allagg.items() if re.search('onfly|iu4.*quant|quant.*iu4',n,re.I)][:6])
