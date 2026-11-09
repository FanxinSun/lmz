import ctypes,json,mmap,os,statistics,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H));from bench import load,record
sys.path.insert(0,str(H.parent/'m4_remeasure'));from campaign import locked,CACHE,notice
from lmz import gpu

def aligned(data):
 m=mmap.mmap(-1,((len(data)+mmap.PAGESIZE-1)//mmap.PAGESIZE)*mmap.PAGESIZE);m[:len(data)]=data;return m
with locked('ship-e3-after') as meta:
 n,p,h,o,s=load();ref=(CACHE/'ref.bin').read_bytes();src=aligned(s);off=aligned(o);dst=mmap.mmap(-1,len(ref))
 record('e3-aligned',lambda:gpu.decode_batch(src,off,n,p,out=dst,header=h),ref,meta)
 record('e3-owned',lambda:gpu.decode_batch(src,off,n,p,header=h),ref,meta)
 record('e3-fallback',lambda:gpu.decode_batch(s,o,n,p,header=h),ref,meta)
 lib=gpu._load(); timing=lib.lmz_gpu_metal_last_timing();print('last stages',[timing[i] for i in range(9)],flush=True)
 # Pointer entry now wraps the true coded extent; supplied output is page-aligned.
 record('e3-device',lambda: (memoryview(dst) if gpu.decode_batch_dev(gpu._ptr(src)[0],gpu._ptr(off)[0],n,p,gpu._ptr(dst)[0],header=gpu._ptr(h)[0])==0 else None),ref,meta)
 n,p,h,o,s=load(True);src=aligned(s);off=aligned(o);out=mmap.mmap(-1,len(ref))
 for setting in ['auto','1','3','7','full']:
  if setting=='auto':os.environ.pop('LMZ_METAL_PERSTREAM_TGS',None)
  else:os.environ['LMZ_METAL_PERSTREAM_TGS']=str((n+3)//4 if setting=='full' else int(setting))
  record('e1-e3-perstream-'+setting,lambda:gpu.decode_batch(src,off,n,p,out=out),ref,meta)
  timing=lib.lmz_gpu_metal_last_timing();
  with (H/'kernel-timings.jsonl').open('a') as f:f.write(json.dumps(dict(mode=setting,stats=[timing[i] for i in range(9)],conditions=meta))+'\n')
  print('kernel G',setting,'GB/s',len(ref)/timing[3]/1e9,flush=True)
 os.environ.pop('LMZ_METAL_PERSTREAM_TGS',None)
notice('E3','Sized Metal host ABI, strict page-range no-copy buffers, retained owned output views and buffer-export lifetime support implemented; real shared/per-chunk outputs byte-identical, both aligned and copying fallback paths measured. Detailed profile and after-runs in m4_ship/.')
