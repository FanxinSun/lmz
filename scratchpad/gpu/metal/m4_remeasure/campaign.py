"""GPU/CPU lock and measurement conditions, shared by all campaign entrypoints."""
import contextlib, datetime, json, os, re, statistics, subprocess, time
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]
CACHE=Path.home()/'Library/Caches/lmz-m4-remeasure'
PYTHON='/Users/masshj/system-env/customize-llm/venv/bin/python'
def shell(args):
 p=subprocess.run(args,capture_output=True,text=True); return p.stdout.strip()+ ('\n'+p.stderr.strip() if p.stderr.strip() else '')
def notice(phase,body):
 p=Path.home()/'.claude/agy-bridge/inbox'/f'{datetime.datetime.now():%Y%m%d-%H%M%S}_lmz-codex.md'
 p.write_text(f'# lmz-codex {phase}\n\n'+body+'\n')
 subprocess.run([str(Path.home()/'.claude/agy-bridge/send_to_tty.sh'),'lmsluice-manager',f'[CODEX-FEEDBACK] {phase}: {body.splitlines()[0]} — report: {p}'],check=True)
@contextlib.contextmanager
def locked(name):
 lock=Path.home()/'.claude/agy-bridge/locks/m4-gpu'; lock.parent.mkdir(parents=True,exist_ok=True)
 while True:
  try: lock.mkdir(); break
  except FileExistsError:
   print('waiting for GPU lock',flush=True)
   if time.time()-lock.stat().st_mtime>1800: notice('lock wait','Other executor has held GPU lock for >30 min; not stealing.')
   time.sleep(15)
 (lock/'owner').write_text(f'lmz-codex {int(time.time())}\n')
 try:
  batches=[]
  for attempt in range(11):
   samples=[]
   for _ in range(5):
    txt=shell(['ioreg','-r','-c','IOAccelerator','-l','-w0'])
    vals=re.findall(r'"Device Utilization %"=(\d+)',txt); samples.append([int(x) for x in vals]); time.sleep(1)
   batches.append(samples); values=[v for s in samples for v in s]
   print(name,'idle utilization',samples,flush=True)
   if not values or statistics.median(values)<=5 or attempt==10: break
   time.sleep(60)
  gitdir=ROOT.parent/'.git/modules/lmz'; head=(gitdir/'HEAD').read_text().strip(); sha=(gitdir/head[5:]).read_text().strip() if head.startswith('ref: ') else head
  meta=dict(campaign=name,timestamp=datetime.datetime.now().astimezone().isoformat(),git_sha=sha,idle_samples=batches,idle_flag=not values or statistics.median(values)>5,vm=shell(['pgrep','-x','prl_vm_app']),therm=shell(['pmset','-g','therm']),power=shell(['pmset','-g','batt']),os=shell(['sw_vers']),memory=shell(['sysctl','hw.memsize']),gpu_registry=shell(['ioreg','-r','-c','AGXAccelerator','-l','-w0']))
  # Preserve only relevant GPU properties, not the whole registry dump.
  meta['gpu_registry']='\n'.join(l for l in meta['gpu_registry'].splitlines() if 'gpu-core-count' in l or 'model' in l)
  CACHE.mkdir(parents=True,exist_ok=True); (CACHE/f'{name}-conditions.json').write_text(json.dumps(meta,indent=2))
  yield meta
 finally:
  (lock/'owner').unlink(); lock.rmdir()
def append(row,meta):
 with (HERE/'runs.jsonl').open('a') as f: f.write(json.dumps(dict(row,conditions=meta))+'\n')
if __name__=='__main__':
 import sys
 if sys.argv[1]=='prep':
  with locked('prep'):
   subprocess.run([PYTHON,str(HERE/'prep.py')],cwd=ROOT,check=True)
  notice('Phase 1','Real Qwen3 shared-table data ready; all 28,584 streams verified with ordinary native decoder. Metadata: '+str(CACHE/'prep.json'))
