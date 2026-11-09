"""Sequential locked campaigns; save every GPU timestamp and Python wall timing."""
import json, statistics, subprocess, sys, time
from campaign import HERE, ROOT, CACHE, PYTHON, append, locked, notice
BIN=CACHE/'harness'
from gpu_clients import snapshot,delta
import gpufreq

def harness(mode,g,runs,meta,data_dir=CACHE):
 selected=[]
 frequency=meta['campaign'].startswith('phase3b')
 for attempt in range(2):
  p=subprocess.Popen([str(BIN),str(data_dir),mode,str(g),str(runs)],cwd=ROOT,stdout=subprocess.PIPE,stdin=subprocess.PIPE,text=True)
  rows=[]; before=None; after=None
  for line in p.stdout:
   row=json.loads(line)
   if row.get('event')=='ready':
    before=snapshot(p.pid)
    if frequency: freq_before=gpufreq.snapshot()
    p.stdin.write('go\n'); p.stdin.flush(); continue
   row['workload']='real_qwen3' if data_dir==CACHE else 'synthetic_historical_shape'
   row['attempt']=attempt; row['harness_pid']=p.pid; rows.append(row)
   if row['run']==runs-1:
    if frequency: freq_after=gpufreq.snapshot()
    after=snapshot(p.pid)
  assert p.wait()==0,(mode,g)
  timed=[r for r in rows if not r['warmup']]
  hygiene=delta(before,after,sum(r['gpu_s'] for r in timed))
  freq=gpufreq.delta(freq_before,freq_after) if frequency else {}
  for row in rows:
   if frequency:
    row['gpu_frequency']=freq; row['f_avg_active_mhz']=freq['f_avg_active_mhz']; row['by_state_s']=freq['by_state_s']
    if g<=30:
     row['k_ns']=128*g/row['gbps']; row['k_cycles']=row['k_ns']*freq['f_avg_active_mhz']/1000
   append(dict(row,**hygiene),meta)
  rate=statistics.median(r['gbps'] for r in timed)
  print(mode,g,'attempt',attempt,'median GB/s',rate,'foreign fraction',hygiene['foreign_fraction'],'top',hygiene['foreign_top'],flush=True)
  selected=timed
  if not hygiene['foreign_over_2pct']: break
 return selected
if __name__=='__main__':
 phase=sys.argv[1]
 if phase in ('2','2-resume'):
  subprocess.run(['swiftc','-O',str(HERE/'harness.swift'),'-o',str(BIN)],check=True)
  with locked('phase2') as meta:
   if phase=='2': harness('shipped',1787,7,meta)
   harness('empty',1,7,meta)
   # Run package timing in a separate process so the harness buffers are released.
   subprocess.run([PYTHON,str(HERE/'published.py')],cwd=ROOT,check=True)
  notice('Phase 2','Shipped runtime-source kernel and trivial-dispatch control measured with GPU timestamps; Python published-method and verify results saved in runs.jsonl.')
 elif phase=='3':
  with locked('phase3-drift') as meta:
   a=harness('shipped',1787,7,meta); b=harness('stride',1787,7,meta)
  ar=statistics.median(r['gbps'] for r in a); br=statistics.median(r['gbps'] for r in b); drift=br/ar-1
  (HERE/'drift.json').write_text(json.dumps(dict(shipped_gbps=ar,stride_gbps=br,relative_drift=drift,pass_3pct=abs(drift)<=.03),indent=2))
  if abs(drift)>.03:
   notice('Phase 3 STOP',f'Instrument drift {drift:.2%} exceeds 3%; grid sweep not run. Shipped {ar:.3f}, wrapper {br:.3f} GB/s.'); sys.exit(3)
  # Separate lock periods cap even the low-grid campaign well below 20 minutes.
  for grids in ([1,2,3,4,5,6,8,10,12,15],[20,25,30,40,50,60,80,100,120,160,200,300,500,1000]):
   with locked('phase3-G'+str(grids[0])+'-'+str(grids[-1])) as meta:
    for g in grids: harness('stride',g,5,meta)
  notice('Phase 3',f'Grid sweep complete; full-grid wrapper drift {drift:.2%}; every G byte-identical. Results in runs.jsonl.')

 elif phase=='2-synthetic':
  import os
  synthetic=CACHE/'synthetic'; synthetic.mkdir(exist_ok=True)
  with locked('phase2-synthetic') as meta:
   subprocess.run([PYTHON,'scratchpad/gpu/prep_bench_shared.py',str(synthetic),'4096','32768'],cwd=ROOT,env=dict(os.environ,PYTHONPATH=str(ROOT)),check=True)
   harness('shipped',256,7,meta,synthetic)
  notice('Phase 2 synthetic reconciliation','Historical synthetic 4,096-stream shape rerun with shipped runtime-source kernel, seven timed GPU runs; all output verified.')

 elif phase=='3b':
  with locked('phase3b-frequency') as meta:
   for g in [1,2,5,10,20,30,40,80,200,1787]: harness('stride',g,5,meta)
  notice('Phase 3b','Row-specific IOReport active residency-weighted frequency sweep complete; own PID excluded from foreign GPU accounting, required repeats retained, all outputs byte-identical.')
