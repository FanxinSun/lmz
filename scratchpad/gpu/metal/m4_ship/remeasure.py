import json,os,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent/'m4_remeasure'));import measure_phase4 as m
from campaign import CACHE,ROOT,PYTHON,locked,notice
# The manager explicitly requested read-only git show HEAD snapshots for before rows.
head=CACHE/'head';head.mkdir(exist_ok=True)
for filename in ['lmz_rans.metal','lmzgpu_metal.m']:
 (head/filename).write_bytes(subprocess.check_output(['git','show','HEAD:lmz/gpu/'+filename],cwd=ROOT))
subprocess.run(['clang','-O3','-fPIC','-shared','-framework','Metal','-framework','Foundation','-o',str(head/'lmzgpu.dylib'),str(head/'lmzgpu_metal.m')],check=True,capture_output=True)
s=(H.parent/'m4_remeasure/harness_phase4.swift').read_text()
s=s.replace('let source=try String(contentsOfFile:"lmz/gpu/lmz_rans.metal",encoding:.utf8)','let source=try String(contentsOfFile:ProcessInfo.processInfo.environment["LMZ_KERNEL_SOURCE"] ?? "lmz/gpu/lmz_rans.metal",encoding:.utf8)')
s=s.replace('if stride { extra=', 'if stride && !(perstream && source.contains("kernel void lmz_decode_perstream_stride")) { extra=')
(H/'kernel.swift').write_text(s)
subprocess.run(['swiftc','-O',str(H/'kernel.swift'),'-o',str(CACHE/'kernel-ship')],check=True)
m.BIN=CACHE/'kernel-ship'
def append(row,meta):
 with (H/'kernel-runs.jsonl').open('a') as f:f.write(json.dumps(dict(row,conditions=meta))+'\n')
m.append=append
with locked('ship-fused-real-prep'):
 subprocess.run([PYTHON,str(H.parent/'m4_remeasure/prep_phase4.py'),'fused'],cwd=ROOT,check=True)
for version in ['head','new']:
 os.environ['LMZ_KERNEL_SOURCE']=str(head/'lmz_rans.metal') if version=='head' else str(ROOT/'lmz/gpu/lmz_rans.metal')
 jobs=[('shared',1787),('shared-stride',1),('shared-stride',10),('shared-stride',40),('perstream',7146),('perstream-stride',80),('fused',1787)]
 if version=='new':jobs += [('perstream-stride',1),('perstream-stride',10),('perstream-stride',30)]
 with locked('ship-kernel-'+version) as meta:
  for mode,g in jobs: m.harness(mode,g,7 if g in (1787,7146,80) else 5,meta,version+'-'+mode)
notice('E3 kernel audit','Same-session HEAD/new kernel matrix complete: shared full/G1/G10/G40, perstream full/tuned and new low-grid costs, fused full. Protected-page and byte-identity checks retained; P15 clock and foreign-work hygiene in m4_ship/kernel-runs.jsonl. HEAD obtained by explicitly requested read-only git show; no git state changes.')
