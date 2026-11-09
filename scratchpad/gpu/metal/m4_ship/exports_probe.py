import ctypes,json,mmap,os,statistics,struct,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent/'m4_remeasure'));from campaign import locked,CACHE,ROOT,notice
sys.path.insert(0,str(ROOT));from lmz import gpu,kernels
from gpu_clients import snapshot,delta
with locked('ship-existing-exports') as meta:
 new=gpu._load();assert new is not None
 old=ctypes.CDLL(str(CACHE/'head/lmzgpu.dylib'))
 for lib in [old,new]:
  lib.lmz_gpu_make_mtl_buffer.restype=ctypes.c_void_p;lib.lmz_gpu_make_mtl_buffer.argtypes=[ctypes.c_void_p,ctypes.c_size_t,ctypes.POINTER(ctypes.c_size_t)]
  lib.lmz_gpu_release_mtl_buffer.argtypes=[ctypes.c_void_p];lib.lmz_gpu_release_mtl_buffer.restype=None
  lib.lmz_gpu_decode_batch_dev.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint,ctypes.c_uint,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint];lib.lmz_gpu_decode_batch_dev.restype=ctypes.c_int
 def aligned(data,minimum=0):
  size=((max(len(data),minimum)+mmap.PAGESIZE-1)//mmap.PAGESIZE)*mmap.PAGESIZE
  m=mmap.mmap(-1,size);m[:len(data)]=data;return m
 results=[];data=aligned((CACHE/'streams.bin').read_bytes()[524+28584*16:])
 for name,lib in [('HEAD',old),('new',new)]:
  for i in range(-2,7):
   offset=ctypes.c_size_t();t=time.perf_counter();handle=lib.lmz_gpu_make_mtl_buffer(gpu._ptr(data)[0],len(data),ctypes.byref(offset));wall=time.perf_counter()-t
   assert handle and new.lmz_gpu_mtl_contents(handle)==gpu._ptr(data)[0]
   lib.lmz_gpu_release_mtl_buffer(handle)
   results.append(dict(mode=name+'-make_mtl_buffer',run=i,warmup=i<0,wall_s=wall,aliases=True))
 hdr=json.loads((CACHE/'prep.json').read_text()); header=(CACHE/'streams.bin').read_bytes()[8:524]
 with (CACHE/'ref.bin').open('rb') as f:ref=f.read(2048*128)
 coded=bytearray();offs=bytearray()
 for i in range(2048):
  stream=kernels.rans_encode_shared(ref[i*128:(i+1)*128],header);offs+=struct.pack('<QQ',len(coded),len(stream));coded+=stream
 src=aligned(coded,2048*128+gpu.pad_bytes());off=aligned(offs);out=aligned(bytes(len(ref)));h=aligned(header)
 for name,lib in [('HEAD',old),('new',new)]:
  for attempt in range(2):
   a=snapshot(os.getpid());group=[]
   for i in range(-2,7):
    t=time.perf_counter();rc=lib.lmz_gpu_decode_batch_dev(gpu._ptr(h)[0],gpu._ptr(src)[0],gpu._ptr(off)[0],2048,128,gpu._ptr(out)[0],None,0);wall=time.perf_counter()-t
    assert rc==0 and out[:len(ref)]==ref
    row=dict(mode=name+'-decode_batch_dev-safe-small',attempt=attempt,run=i,warmup=i<0,wall_s=wall,gbps=len(ref)/wall/1e9,identical=True);group.append(row)
   b=snapshot(os.getpid());hygiene=delta(a,b,sum(r['wall_s'] for r in group));results+= [dict(r,foreign=hygiene) for r in group]
   if not hygiene['foreign_over_2pct']:break
 (H/'existing-exports.json').write_text(json.dumps(dict(rows=results,conditions=meta,note='HEAD device ABI on real 32-KiB streams is unsafe to benchmark: it declares only 128*nstr+pad coded bytes. This valid 128-byte-plane shared workload fits that extent. The before/after export alias checks use whole-page buffers.'),indent=2))
 print('existing make/export and safe small device ABI checked',flush=True)
