import json,os,statistics,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent; sys.path.insert(0,str(H.parent/'m4_remeasure')); from campaign import locked,CACHE,ROOT
from gpu_clients import snapshot,delta
import gpufreq
with locked('ship-before-components') as meta:
 for attempt in range(2):
  p=subprocess.Popen([str(CACHE/'profile-ship'),str(CACHE)],cwd=ROOT,stdout=subprocess.PIPE,stdin=subprocess.PIPE,text=True); rows=[]
  for line in p.stdout:
   r=json.loads(line)
   if r.get('event')=='ready':
    a=snapshot(p.pid); fa=gpufreq.snapshot(); p.stdin.write('go\n');p.stdin.flush();continue
   if r.get('event')=='finished':
    fb=gpufreq.snapshot(); b=snapshot(p.pid);p.stdin.write('finish\n');p.stdin.flush();continue
   r.update(attempt=attempt,conditions=meta,foreign=delta(a,b,r['times']['gpu_s']),frequency=gpufreq.delta(fa,fb)); rows.append(r)
   with (H/'profile-runs.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
  assert p.wait()==0
  timed=[r for r in rows if not r['warmup']]
  print({k:statistics.median(r['times'][k] for r in timed) for k in timed[0]['times']},flush=True)
  if all(not r['foreign']['foreign_over_2pct'] for r in timed):break
