"""D2 campaigns. New files only; all dispatches append to runs-phase4.jsonl."""
import collections,json,statistics,subprocess,sys,time
from campaign import HERE,ROOT,CACHE,PYTHON,locked,notice,shell
from gpu_clients import snapshot as foreign_snapshot,delta as foreign_delta
import gpufreq
BIN=CACHE/'harness-phase4'
def append(row,meta):
 with (HERE/'runs-phase4.jsonl').open('a') as f: f.write(json.dumps(dict(row,conditions=meta))+'\n')
def clock_attribution(freq,gpu_sum):
 table=freq['table_mhz']; p15_mhz=table[15]; p15=freq['by_state_s'].get('P15',0); qualifies=p15>=.95*gpu_sum
 return dict(p15_s=p15,summed_kernel_s=gpu_sum,p15_to_kernel_ratio=p15/gpu_sum,p15_ge_95pct=qualifies,kernel_attributed_mhz=p15_mhz if qualifies else freq['f_avg_active_mhz'],kernel_clock_method='P15' if qualifies else 'f_avg_active',pstate_table_nonmonotonic=True)
def harness(mode,g,runs,meta,tag,workers=None):
 selected=[]
 for attempt in range(2):
  p=subprocess.Popen([str(BIN),str(CACHE),mode,str(g),str(runs)],cwd=ROOT,stdout=subprocess.PIPE,stdin=subprocess.PIPE,text=True)
  rows=[]; before=after=None
  for line in p.stdout:
   row=json.loads(line)
   if row.get('event')=='ready':
    before=foreign_snapshot(p.pid); fa=gpufreq.snapshot()
    if workers: workers.begin_window()
    window_start=time.perf_counter(); p.stdin.write('go\n'); p.stdin.flush(); continue
   if row.get('event')=='finished':
    window_end=time.perf_counter()
    if workers: cpu=workers.end_window()
    fb=gpufreq.snapshot(); after=foreign_snapshot(p.pid); p.stdin.write('finish\n'); p.stdin.flush(); continue
   row.update(tag=tag,attempt=attempt,harness_pid=p.pid); rows.append(row)
  rc=p.wait()
  if rc!=0: raise RuntimeError((mode,g,rc))
  timed=[r for r in rows if not r['warmup']]; total=sum(r['gpu_s'] for r in timed)
  hygiene=foreign_delta(before,after,total); freq=gpufreq.delta(fa,fb); attr=clock_attribution(freq,total)
  context=dict(**hygiene,gpu_frequency=freq,clock_attribution=attr,window_s=window_end-window_start)
  if workers:
   context['cpu_contention']=cpu
   expansion=json.loads((CACHE/'prep.json').read_text())['expansion']; traffic_factor=1+1/expansion
   kernel_rate=statistics.median(r['exponent_gbps'] for r in timed)
   context['traffic_sanity']=dict(gpu_logical_traffic_gbps=kernel_rate*traffic_factor,cpu_counted_traffic_gbps=cpu['cpu_traffic_gbps'],kernel_rate_plus_cpu_gbps=kernel_rate*traffic_factor+cpu['cpu_traffic_gbps'],same_window_gpu_traffic_gbps=sum(r['exponent_bytes'] for r in timed)*traffic_factor/cpu['window_s']/1e9,limit_gbps=120)
   context['traffic_sanity']['pass_120']=context['traffic_sanity']['kernel_rate_plus_cpu_gbps']<=120
  for row in rows:
   row.update(context)
   # Cost arithmetic is stored for all G as a launch-lane diagnostic; only
   # pre-saturation G can interpret launched lanes as resident lanes.
   lanes=g*row['threads_per_tg']; kns=lanes/row['exponent_gbps']
   row.update(k_ns_if_G_resident=kns,k_cycles_avg_if_G_resident=kns*freq['f_avg_active_mhz']/1000,k_cycles_attributed_if_G_resident=kns*attr['kernel_attributed_mhz']/1000)
   append(row,meta)
  print(tag,g,'attempt',attempt,'GB/s',statistics.median(r['gbps'] for r in timed),'foreign',hygiene['foreign_fraction'],'P15/kernel',attr['p15_to_kernel_ratio'],flush=True)
  selected=timed
  if not hygiene['foreign_over_2pct'] and ('traffic_sanity' not in context or context['traffic_sanity']['pass_120']): break
 if workers and not context['traffic_sanity']['pass_120']: raise RuntimeError('CPU+GPU logical traffic exceeds 120 GB/s; reject row')
 return selected

def run_perstream():
 with locked('phase4-perstream-prep'):
  subprocess.run([PYTHON,str(HERE/'prep_phase4.py'),'perstream'],cwd=ROOT,check=True)
 with locked('phase4-perstream-drift') as meta:
  a=harness('perstream',7146,7,meta,'perstream-shipped'); b=harness('perstream-stride',7146,7,meta,'perstream-drift')
 ar=statistics.median(r['gbps'] for r in a); br=statistics.median(r['gbps'] for r in b); drift=br/ar-1
 result=dict(shipped_gbps=ar,stride_gbps=br,relative_drift=drift,pass_3pct=abs(drift)<=.03)
 (HERE/'drift-phase4-perstream.json').write_text(json.dumps(result,indent=2))
 if abs(drift)>.03:
  notice('D2 item 1 STOP',f'Perstream wrapper drift {drift:.2%} exceeds 3%; no perstream grid sweep. Full-grid evidence retained.'); return
 for grids in ([1,2,3,4,5,6,8,10,12,15,20,25,30,40],[50,60,80,100,120,160,200,300,500,1000]):
  with locked('phase4-perstream-G'+str(grids[0])+'-'+str(grids[-1])) as meta:
   for g in grids: harness('perstream-stride',g,5,meta,'perstream-sweep')
 notice('D2 item 1',f'Native per-chunk streams all round-trip; perstream full grid {ar:.4f} GB/s, wrapper drift {drift:.2%}; fixed-traffic grid sweep complete, every G byte-identical. runs-phase4.jsonl.')
if __name__=='__main__':
 phase=sys.argv[1]
 if phase=='build':
  subprocess.run(['swiftc','-O',str(HERE/'harness_phase4.swift'),'-o',str(BIN)],cwd=ROOT,check=True)
 elif phase=='perstream': run_perstream()

 elif phase=='contention':
  from contention_phase4 import CopyWorkers
  # G=40 needs the validated shared stride wrapper to keep fixed full traffic.
  for n in [0,2,4,8,10,0]:
   tag='contention-N'+str(n)+('-end' if n==0 and 'seen_zero' in globals() else '')
   if n==0: seen_zero=True
   with locked('phase4-'+tag) as meta:
    meta['memory_stats_before']=shell(['vm_stat'])
    workers=CopyWorkers(n)
    try:
     for g in [40,1787]: harness('shared-stride',g,7,meta,tag,workers)
    finally:
     workers.close()
     (CACHE/('phase4-'+tag+'-memory-after.txt')).write_text(shell(['vm_stat']))
  notice('D2 item 3','Shared fixed-traffic G=40/full-grid CPU contention N=0,2,4,8,10,0-end measured; every worker copy rate, logical read+write traffic and 120 GB/s sanity check retained; all GPU output checks pass.')
 elif phase=='fused':
  with locked('phase4-fused-prep'):
   subprocess.run([PYTHON,str(HERE/'prep_phase4.py'),'fused'],cwd=ROOT,check=True)
  with locked('phase4-fused-drift') as meta:
   a=harness('fused',1787,7,meta,'fused-shipped'); b=harness('fused-stride',1787,7,meta,'fused-drift')
  ar=statistics.median(r['gbps'] for r in a); br=statistics.median(r['gbps'] for r in b); drift=br/ar-1
  (HERE/'drift-phase4-fused.json').write_text(json.dumps(dict(shipped_bf16_gbps=ar,stride_bf16_gbps=br,relative_drift=drift,pass_3pct=abs(drift)<=.03),indent=2))
  if abs(drift)>.03:
   notice('D2 item 2 STOP',f'Fused full-grid {ar:.4f} BF16 GB/s; wrapper drift {drift:.2%} exceeds 3%, no fused sweep.'); sys.exit(3)
  for grids in ([1,2,5,10,20,30,40],[60,80,120,200,500]):
   with locked('phase4-fused-G'+str(grids[0])+'-'+str(grids[-1])) as meta:
    for g in grids: harness('fused-stride',g,5,meta,'fused-sweep')
  notice('D2 item 2',f'Fused original BF16 reference byte-identical; shipped {ar:.4f} BF16 GB/s / {ar/2:.4f} exponent GB/s. Wrapper drift {drift:.2%}; verbatim-body grid sweep complete.')
