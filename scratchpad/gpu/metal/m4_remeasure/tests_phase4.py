"""Run the exact pytest baseline, retaining every attempt and environment result."""
import json,subprocess
from campaign import HERE,ROOT,PYTHON,locked,notice
results=[]
for attempt in [1,2]:
 with locked('phase4-final-tests-'+str(attempt)):
  p=subprocess.run([PYTHON,'-m','pytest','-q','tests'],cwd=ROOT,capture_output=True,text=True)
  (HERE/f'tests-phase4-attempt{attempt}.txt').write_text(p.stdout+p.stderr)
  result=[l for l in p.stdout.splitlines() if ' passed' in l and ' failed' in l][-1]
  r=dict(attempt=attempt,exit_code=p.returncode,result=result,matches_baseline='6 failed, 142 passed' in result)
  results.append(r); (HERE/'tests-phase4.json').write_text(json.dumps(dict(attempts=results,final=r,baseline=dict(passed=142,failed=6,total=148)),indent=2)); print(json.dumps(r),flush=True)
 if r['matches_baseline']: break
 # A new failure is an unresolved concern warranting one exact retry; no test fix.
notice('D2 final tests',result+'; matches requested 142/6 baseline: '+str(r['matches_baseline']))
