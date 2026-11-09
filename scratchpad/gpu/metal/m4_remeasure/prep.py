"""Real Qwen3 BF16 exponent planes; native shared encoder, ordinary decoder oracle."""
import ctypes as C, glob, json, mmap, os, struct, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
OUT=Path.home()/'Library/Caches/lmz-m4-remeasure'; OUT.mkdir(parents=True,exist_ok=True)
N=28584; PLANE=32768; TOTAL=N*PLANE
so=max((ROOT/'lmz/native').glob('lmzcore-arm64-*.dylib'),key=lambda p:p.stat().st_mtime)
lib=C.CDLL(str(so)); v=C.c_void_p; z=C.c_size_t
for name,rest,args in [('lmz_rans_bound',z,[z]),('lmz_rans_freqs',C.c_int,[v,v]),('lmz_rans_encode_shared',C.c_long,[v,z,v,z,v]),('lmz_rans_decode',C.c_int,[v,z,v,z])]:
 f=getattr(lib,name); f.restype=rest; f.argtypes=args
used=[]; done=0; hist=np.zeros(256,dtype=np.uint64)
with open(OUT/'ref.bin','wb') as dst:
 for path in sorted((Path.home()/'models/Qwen3-4B-Instruct-2507').glob('*.safetensors')):
  with open(path,'rb') as f:
   h=struct.unpack('<Q',f.read(8))[0]; metas=json.loads(f.read(h)); base=8+h
   for name,meta in sorted(((k,m) for k,m in metas.items() if k!='__metadata__'),key=lambda x:x[1]['data_offsets'][0]):
    if meta['dtype']!='BF16' or done==TOTAL: continue
    a,b=meta['data_offsets']; take=min((b-a)//2,TOTAL-done); f.seek(base+a)
    w=np.frombuffer(f.read(take*2),dtype='<u2'); plane=((w>>7)&255).astype(np.uint8)
    dst.write(plane.tobytes()); hist+=np.bincount(plane,minlength=256).astype(np.uint64)
    used.append(dict(file=str(path),tensor=name,elements=take,tensor_elements=(b-a)//2)); done+=take
  if done==TOTAL: break
assert done==TOTAL
counts=(C.c_uint64*256)(*map(int,hist)); freqs=(C.c_uint16*256)(); assert lib.lmz_rans_freqs(counts,freqs)==0
header=b'R1\0\0'+struct.pack('<256H',*freqs)
cap=lib.lmz_rans_bound(PLANE); buf=C.create_string_buffer(cap); dec=C.create_string_buffer(PLANE)
offsets=[]; coded_total=0; t=time.perf_counter()
with open(OUT/'ref.bin','rb') as ref, open(OUT/'streams.bin','wb+') as dst:
 dst.write(struct.pack('<II',N,PLANE)+header+bytes(N*16)); pos=0
 for i in range(N):
  seg=ref.read(PLANE); n=lib.lmz_rans_encode_shared(seg,PLANE,buf,cap,freqs); assert n>0
  coded=buf.raw[:n]; ordinary=header+coded
  assert lib.lmz_rans_decode(ordinary,len(ordinary),dec,PLANE)==0 and dec.raw==seg, i
  offsets.append((pos,n)); dst.write(coded); pad=(-n)%16; dst.write(bytes(pad)); pos+=n+pad; coded_total+=n
  if i%4000==0: print('verified',i,flush=True)
 dst.seek(524)
 for o,n in offsets: dst.write(struct.pack('<QQ',o,n))
meta=dict(nstr=N,plane=PLANE,plain_bytes=TOTAL,coded_bytes=coded_total,shared_header_bytes=516,padded_payload_bytes=pos,expansion=TOTAL/(coded_total+516),native_library=str(so),verified_streams=N,encoding_and_verification_s=time.perf_counter()-t,tensors=used,cleanup=[str(OUT/'ref.bin'),str(OUT/'streams.bin'),str(OUT/'prep.json')])
(OUT/'prep.json').write_text(json.dumps(meta,indent=2)); print(json.dumps({k:v for k,v in meta.items() if k!='tensors'},indent=2))
