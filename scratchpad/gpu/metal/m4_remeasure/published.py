import json, statistics, struct, sys, time
from campaign import ROOT,HERE,CACHE,append
sys.path.insert(0,str(ROOT))
from lmz import gpu
blob=(CACHE/'streams.bin').read_bytes(); n,plane=struct.unpack_from('<II',blob)
header=blob[8:524]; offsets=blob[524:524+n*16]; payload=blob[524+n*16:]; del blob
ref=(CACHE/'ref.bin').read_bytes(); out=bytearray(n*plane)
meta=json.loads((CACHE/'phase2-conditions.json').read_text())
for run in range(-2,7):
 t=time.perf_counter(); got=gpu.decode_batch(payload,offsets,n,plane,out=out,header=header); wall=time.perf_counter()-t
 ok=got is not None and out==ref
 append(dict(mode='python_decode_batch',G=1787,run=run,warmup=run<0,wall_s=wall,plain_bytes=n*plane,gbps=n*plane/wall/1e9,verified=True,identical=ok),meta)
 assert ok,gpu.last_error()
print('Python wall timing done',flush=True)
# Free the full-grid workload before constructing the shipped verification shapes.
del ref,out,payload,offsets,header,got
start=time.perf_counter(); report=gpu.verify(); wall=time.perf_counter()-start
append(dict(mode='verify',report=report,wall_s=wall),meta)
(HERE/'verify.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report),flush=True)
