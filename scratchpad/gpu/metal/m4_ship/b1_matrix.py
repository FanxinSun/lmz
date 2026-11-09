import json,os,sys
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent/'m4_remeasure'));import measure_phase4 as m
from campaign import CACHE,ROOT,locked,notice
m.BIN=CACHE/'kernel-ship'
def append(row,meta):
 with (H/'kernel-b1-runs.jsonl').open('a') as f:f.write(json.dumps(dict(row,conditions=meta))+'\n')
m.append=append
for version in ['head','b1']:
 os.environ['LMZ_KERNEL_SOURCE']=str(CACHE/'head/lmz_rans.metal') if version=='head' else str(ROOT/'lmz/gpu/lmz_rans.metal')
 with locked('ship-kernel-'+version+'-b1-audit') as meta:
  for mode,g in [('shared',1787),('shared-stride',1),('perstream',7146),('fused',1787)]:
   m.harness(mode,g,7 if g in (1787,7146) else 5,meta,version+'-'+mode)
notice('E3 B1 kernel audit','Restored HEAD kernel bodies; quick same-session shared full/G1, perstream full and fused kernel comparison complete with all output checks and hygiene in kernel-b1-runs.jsonl. New E1 stride retains the padded SIMD bound and unmodified HEAD refill; cost records remain unchanged.')
