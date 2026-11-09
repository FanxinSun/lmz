import json,statistics,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent/'m4_remeasure'));from campaign import CACHE,ROOT,PYTHON,locked,notice
from gpu_clients import snapshot,delta
import gpufreq
subprocess.run(['clang','-O3','-fobjc-arc','-dynamiclib','-framework','Foundation','-framework','Metal',str(H/'wait_timer.m'),'-o',str(CACHE/'wait-timer.dylib')],check=True)
for mode in ['head-shared','head-perstream','new-aligned','new-prepared-shared','new-prepared-perstream','new-fused','new-prepared-fused']:
 with locked('ship-api-'+mode) as meta:
  for attempt in range(2):
   allrows=[];groups=7 if mode.startswith('head') else 1
   for group in range(groups):
    p=subprocess.Popen([PYTHON,str(H/'api_child.py'),mode,'1' if groups==7 else '7'],cwd=ROOT,stdout=subprocess.PIPE,stdin=subprocess.PIPE,text=True);rows=[]
    for line in p.stdout:
     row=json.loads(line)
     if row.get('event')=='ready':a=snapshot(p.pid);fa=gpufreq.snapshot();p.stdin.write('go\n');p.stdin.flush();continue
     if row.get('event')=='finished':fb=gpufreq.snapshot();b=snapshot(p.pid);p.stdin.write('finish\n');p.stdin.flush();continue
     row.update(attempt=attempt,group=group);rows.append(row)
    assert p.wait()==0,(mode,group)
    timed=[r for r in rows if not r['warmup']];foreign=delta(a,b,sum(r['gpu_s'] for r in timed));freq=gpufreq.delta(fa,fb)
    for row in rows:
     row.update(conditions=meta,foreign=foreign,frequency=freq)
     with (H/'api-b1-runs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    allrows+=timed
   print(mode,attempt,'API',statistics.median(r['gbps'] for r in allrows),'kernel',statistics.median(r['kernel_gbps'] for r in allrows),flush=True)
   if all(not r['foreign']['foreign_over_2pct'] for r in allrows):break
notice('E3 same-session API audit','HEAD library (unmodified git-show snapshot) vs new aligned/prepared/shared/perstream/fused API rows complete, with GPU wait timing instrumented externally for HEAD, true GPU-time foreign checks and both attempts retained. HEAD uses isolated short-lived processes to avoid its MRC per-call retained-buffer accumulation; each timed sample follows two warmups.')
