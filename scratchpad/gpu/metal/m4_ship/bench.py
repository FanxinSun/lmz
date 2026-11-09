import ctypes,json,mmap,os,statistics,struct,subprocess,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent; sys.path.insert(0,str(H.parent/'m4_remeasure')); from campaign import ROOT,CACHE,PYTHON,locked,notice
sys.path.insert(0,str(ROOT)); from lmz import gpu
from gpu_clients import snapshot,delta
import gpufreq

def record(mode,fn,ref,meta):
 rows=[]
 for run in range(-2,7):
  a=snapshot(os.getpid()); fa=gpufreq.snapshot(); t=time.perf_counter(); got=fn(); wall=time.perf_counter()-t; fb=gpufreq.snapshot(); b=snapshot(os.getpid())
  ok=got is not None and got==ref
  row=dict(mode=mode,run=run,warmup=run<0,wall_s=wall,gbps=len(ref)/wall/1e9,identical=ok,frequency=gpufreq.delta(fa,fb),foreign=delta(a,b,wall),conditions=meta)
  with (H/'runs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
  assert ok,(mode,gpu.last_error())
  if run>=0: rows.append(row)
 print(mode,statistics.median(r['gbps'] for r in rows),flush=True); return rows

def load(per=False):
 blob=(CACHE/('perstream.bin' if per else 'streams.bin')).read_bytes(); n,p=struct.unpack_from('<II',blob); return n,p,blob[8:524],blob[524:524+n*16],blob[524+n*16:]
if __name__=='__main__':
 phase=sys.argv[1]
 if phase=='e1':
  with locked('ship-e1-prep'):
   subprocess.run([PYTHON,str(H.parent/'m4_remeasure/prep_phase4.py'),'perstream'],cwd=ROOT,check=True)
  with locked('ship-e1-api') as meta:
   n,p,h,o,s=load(True); ref=(CACHE/'ref.bin').read_bytes(); out=bytearray(len(ref))
   record('e1-auto',lambda:gpu.decode_batch(s,o,n,p,out=out),ref,meta)
   lib=gpu._load(); lib.lmz_gpu_metal_perstream_tgs.restype=ctypes.c_uint; print('tuned',lib.lmz_gpu_metal_perstream_tgs(),flush=True)
   (H/'e1-tuned.json').write_text(json.dumps(dict(tgs=lib.lmz_gpu_metal_perstream_tgs())))
  notice('E1','Grid-stride + device-measured tuning implemented; real per-chunk API output identical. Candidate choice and raw wall runs in scratchpad/gpu/metal/m4_ship/. Kernel-only old baseline retained; new kernel/API validation continues in E3.')
 elif phase=='baseline':
  with locked('ship-e3-before') as meta:
   n,p,h,o,s=load(); ref=(CACHE/'ref.bin').read_bytes(); out=bytearray(len(ref))
   record('e3-before',lambda:gpu.decode_batch(s,o,n,p,out=out,header=h),ref,meta)
