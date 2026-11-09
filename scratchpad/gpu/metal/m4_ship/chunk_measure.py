import json,os,statistics,struct,sys,time
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent/'m4_remeasure'));from campaign import locked,CACHE,ROOT,notice
sys.path.insert(0,str(ROOT));from lmz import gpu,kernels,codec
from lmz.format import CODEC_BF16
with locked('ship-fused-chunk-comparison') as meta:
 with (CACHE/'bf16.bin').open('rb') as f: raw=f.read(4<<20)
 for size in [64<<10,1<<20,4<<20]:
  want=raw[:size];planes=kernels.split_bf16(want);n=size//2;exp=planes[:n];sm=planes[n:];stream=kernels.rans_encode(exp)
  payload=struct.pack('<II',len(stream),len(sm))+stream+sm
  for mode in ['cpu-neon','gpu-two-plane','gpu-fused-optin']:
   os.environ['LMZ_METAL_FUSED_CHUNK']='1' if mode.endswith('optin') else '0'
   rows=[]
   for run in range(-2,7):
    t=time.perf_counter();got=(codec.decode_chunk(payload,CODEC_BF16,2,3,size,verify=False) if mode=='cpu-neon' else gpu.decode_chunk(payload,CODEC_BF16,2,3,size));wall=time.perf_counter()-t
    assert got is not None and got==want
    row=dict(mode=mode,bf16_bytes=size,run=run,warmup=run<0,wall_s=wall,gbps=size/wall/1e9,identical=True,conditions=meta)
    with (H/'chunk-runs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    if run>=0:rows.append(row)
   print(size,mode,statistics.median(r['gbps'] for r in rows),flush=True)
 os.environ.pop('LMZ_METAL_FUSED_CHUNK',None)
notice('E2 chunk audit','Real Qwen3 CODEC_BF16 64 KiB/1 MiB/4 MiB chunk API wall comparison complete against CPU native merge and prior two-plane GPU path. Fused chunk dispatch is OFF by default, opt-in LMZ_METAL_FUSED_CHUNK=1. Public many-stream fused API remains available; raw output identity and rates in m4_ship/chunk-runs.jsonl.')
