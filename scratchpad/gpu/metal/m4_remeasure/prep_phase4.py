"""D2: native ordinary streams and original BF16/sign-mantissa references."""
import ctypes as C,json,struct,sys,time
from pathlib import Path
import numpy as np
from campaign import ROOT,CACHE
mode=sys.argv[1]; meta=json.loads((CACHE/'prep.json').read_text())
lib=C.CDLL(meta['native_library']); v=C.c_void_p; z=C.c_size_t
for name,rest,args in [('lmz_rans_bound',z,[z]),('lmz_rans_encode',C.c_long,[v,z,v,z]),('lmz_rans_decode',C.c_int,[v,z,v,z])]:
 f=getattr(lib,name); f.restype=rest; f.argtypes=args
n=meta['nstr']; plane=meta['plane']
if mode=='perstream':
 cap=lib.lmz_rans_bound(plane); buf=C.create_string_buffer(cap); out=C.create_string_buffer(plane); offsets=[]; coded=0; pos=0; t=time.perf_counter()
 with (CACHE/'ref.bin').open('rb') as ref,(CACHE/'perstream.bin').open('wb+') as dst:
  # Same transport fixture layout; the unused global-header slot is zero.
  dst.write(struct.pack('<II',n,plane)+bytes(516+n*16))
  for i in range(n):
   seg=ref.read(plane); size=lib.lmz_rans_encode(seg,plane,buf,cap); assert size>0
   payload=buf.raw[:size]; assert payload[:2]==b'R1'
   assert lib.lmz_rans_decode(payload,size,out,plane)==0 and out.raw==seg,i
   offsets.append((pos,size)); dst.write(payload); pad=(-size)%16; dst.write(bytes(pad)); pos+=size+pad; coded+=size
  dst.seek(524)
  for o,size in offsets: dst.write(struct.pack('<QQ',o,size))
 result=dict(nstr=n,plane=plane,plain_bytes=n*plane,coded_bytes=coded,padded_payload_bytes=pos,expansion=n*plane/coded,verified_streams=n,native_library=meta['native_library'],seconds=time.perf_counter()-t,tensors=meta['tensors'])
 (CACHE/'perstream-prep.json').write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k!='tensors'}),flush=True)
elif mode=='fused':
 count=0
 with (CACHE/'bf16.bin').open('wb') as bf,(CACHE/'smplane.bin').open('wb') as sm,(CACHE/'ref.bin').open('rb') as exp:
  for item in meta['tensors']:
   with open(item['file'],'rb') as f:
    h=struct.unpack('<Q',f.read(8))[0]; hdr=json.loads(f.read(h)); start=hdr[item['tensor']]['data_offsets'][0]; f.seek(8+h+start)
    remaining=item['elements']
    while remaining:
     take=min(remaining,4*1024*1024); raw=f.read(2*take); assert len(raw)==2*take
     w=np.frombuffer(raw,dtype='<u2'); e=((w>>7)&255).astype(np.uint8).tobytes(); assert e==exp.read(take)
     signmant=(((w>>8)&128)|(w&127)).astype(np.uint8).tobytes()
     bf.write(raw); sm.write(signmant); remaining-=take; count+=take
 assert count==n*plane
 result=dict(elements=count,bf16_bytes=count*2,smplane_bytes=count,reference='original BF16 bytes read directly from safetensors; exponent plane independently matched to Phase-1 ref.bin',tensors=meta['tensors'])
 (CACHE/'fused-prep.json').write_text(json.dumps(result,indent=2)); print('fused reference bytes',count*2,flush=True)
