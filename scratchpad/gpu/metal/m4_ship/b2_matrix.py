import json,os,sys
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent/'m4_remeasure'));import measure_phase4 as m
from campaign import CACHE,ROOT,locked,notice
m.BIN=CACHE/'kernel-ship'
def append(row,meta):
 with (H/'kernel-b2-runs.jsonl').open('a') as f:f.write(json.dumps(dict(row,conditions=meta))+'\n')
m.append=append
for version in ['head','b2']:
 os.environ['LMZ_KERNEL_SOURCE']=str(CACHE/'head/lmz_rans.metal') if version=='head' else str(ROOT/'lmz/gpu/lmz_rans.metal')
 with locked('ship-kernel-'+version+'-b2-audit') as meta:
  for mode,g in [('shared',1787),('shared-stride',1),('shared-stride',10),('shared-stride',40),('perstream',7146),('perstream-stride',30),('fused',1787)]:
   m.harness(mode,g,7 if g in (1787,7146) else 5,meta,version+'-'+mode)
notice('E3 B2 matrix','Branchless clamped-refill matrix complete against same-session HEAD, all byte-identical with row timing and hygiene in kernel-b2-runs.jsonl; acceptance decision follows comparison.')
