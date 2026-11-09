import ctypes,json,mmap,os,struct,sys,time,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H));from bench import load
sys.path.insert(0,str(H.parent/'m4_remeasure'));from campaign import CACHE,ROOT
sys.path.insert(0,str(ROOT));from lmz import gpu
mode=sys.argv[1]; repeats=int(sys.argv[2]); per='perstream' in mode; fused='fused' in mode
source_sha256=hashlib.sha256((CACHE/'head/lmz_rans.metal' if mode.startswith('head') else ROOT/'lmz/gpu/lmz_rans.metal').read_bytes()).hexdigest()
n,p,h,o,s=load(per); ref=(CACHE/('bf16.bin' if fused else 'ref.bin')).read_bytes()
def aligned(data):
 m=mmap.mmap(-1,((len(data)+mmap.PAGESIZE-1)//mmap.PAGESIZE)*mmap.PAGESIZE);m[:len(data)]=data;return m
if mode.startswith('head'):
 lib=ctypes.CDLL(str(CACHE/'head/lmzgpu.dylib'));fn=lib.lmz_gpu_decode_batch;fn.restype=ctypes.c_int;fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p,ctypes.c_uint,ctypes.c_uint,ctypes.c_void_p]
 timer=ctypes.CDLL(str(CACHE/'wait-timer.dylib'));timer.install_wait_timer();timer.last_wait_gpu_seconds.restype=ctypes.c_double
 out=bytearray(len(ref));sp,sa=gpu._ptr(s);op,oa=gpu._ptr(o);hp,ha=gpu._ptr(h);dp,da=gpu._ptr(out)
 def decode():
  assert fn(None if per else hp,sp,len(s),op,n,p,dp)==0
  return memoryview(out)
 def gpu_time():return timer.last_wait_gpu_seconds()
else:
 src,off=(s,o) if "fallback" in mode else (aligned(s),aligned(o));out=None if "fallback" in mode else mmap.mmap(-1,len(ref));sm=aligned((CACHE/'smplane.bin').read_bytes()) if fused else None
 setup_s=None
 if 'prepared' in mode:
  setup_start=time.perf_counter()
  prepared=gpu.prepare_metal_batch(src,off,n,p,header=None if per else h,smplane=sm,out=out)
  setup_s=time.perf_counter()-setup_start
  assert prepared is not None,gpu.last_error()
  decode=prepared.decode
 elif fused: decode=lambda:gpu.decode_bf16_fused(src,off,sm,n,p,h,out=out)
 else: decode=lambda:gpu.decode_batch(src,off,n,p,header=None if per else h,out=out)
 def gpu_time():return gpu._load().lmz_gpu_metal_last_timing()[3]
for run in range(-2,repeats):
 if run==0:print('{"event":"ready"}',flush=True);assert input()=='go'
 t=time.perf_counter();got=decode();wall=time.perf_counter()-t;gs=gpu_time()
 if run==repeats-1:print('{"event":"finished"}',flush=True);assert input()=='finish'
 verified=os.environ.get("LMZ_BENCH_CHECK_ALL","1")=="1" or run in (-1,0,repeats-1)
 ok=(got is not None and got==ref) if verified else None
 assert got is not None and (ok is not False),(mode,gpu.last_error())
 row=dict(source_sha256=source_sha256,mode=mode,run=run,warmup=run<0,wall_s=wall,gpu_s=gs,gbps=len(ref)/wall/1e9,kernel_gbps=len(ref)/gs/1e9,identical=ok,verified=verified,check_all=os.environ.get("LMZ_BENCH_CHECK_ALL","1")=="1")
 if not mode.startswith('head'):
  row['prepare_s']=setup_s
  lib=gpu._load();row['stages']=[lib.lmz_gpu_metal_last_timing()[i] for i in range(9)]
  lib.lmz_gpu_metal_perstream_tgs.restype=ctypes.c_uint;row['tuned_tgs']=lib.lmz_gpu_metal_perstream_tgs()
 print(json.dumps(row),flush=True)
