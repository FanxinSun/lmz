import json,mmap,os,struct,sys
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H));from bench import load
sys.path.insert(0,str(H.parent/'m4_remeasure'));from campaign import locked,CACHE,notice
from lmz import gpu
with locked('ship-real-settings') as meta:
 n,p,h,o,s=load(True);ref=(CACHE/'ref.bin').read_bytes()
 def aligned(data):
  m=mmap.mmap(-1,((len(data)+mmap.PAGESIZE-1)//mmap.PAGESIZE)*mmap.PAGESIZE);m[:len(data)]=data;return m
 src,off=aligned(s),aligned(o);out=mmap.mmap(-1,len(ref));results=[]
 for grid in [1,3,7,(n+3)//4]:
  os.environ['LMZ_METAL_PERSTREAM_TGS']=str(grid)
  got=gpu.decode_batch(src,off,n,p,out=out)
  assert got is not None and got==ref
  results.append(dict(nstr=n,plane=p,forced_grid=grid,identical=True))
 os.environ.pop('LMZ_METAL_PERSTREAM_TGS',None)
 (H/'real-settings-check.json').write_text(json.dumps(dict(results=results,conditions=meta,note='Correctness-only runs; no rate claims.'),indent=2))
notice('E1 final real settings','Final B1/ARC host and HEAD-body stride kernel decode all 28,584 real streams identically at forced G=1,3,7 and the full grid. These supplement the 10 synthetic partial-tail counts at the same forced settings.')
