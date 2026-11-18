#!/usr/bin/env python3
"""Sample only an owned launcher's decoder descendants during startup."""
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('parent',type=int);p.add_argument('binary');p.add_argument('output');a=p.parse_args()
root=Path('/proc');wanted=Path(a.binary).name
with open(a.output,'w',buffering=1) as out:
 while (root/str(a.parent)).exists():
  pending=[a.parent];seen=set()
  while pending:
   pid=pending.pop()
   if pid in seen:continue
   seen.add(pid);base=root/str(pid)
   try:
    pending.extend(map(int,(base/'task'/str(pid)/'children').read_text().split()))
    command=(base/'cmdline').read_bytes().split(b'\0')[0].decode(errors='replace')
    if Path(command).name!=wanted:continue
    stat=(base/'stat').read_text().rsplit(') ',1)[1].split();io={}
    for line in (base/'io').read_text().splitlines():
     key,value=line.split(':',1);io[key]=int(value)
    r={'epoch_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns(),'pid':pid,'state':stat[0],'minflt':int(stat[7]),'majflt':int(stat[9]),'wchan':(base/'wchan').read_text().strip(),'read_bytes':io.get('read_bytes'),'rchar':io.get('rchar')}
    out.write(json.dumps(r,separators=(',',':'))+'\n')
   except (OSError,ValueError,IndexError):pass
  time.sleep(.25)
