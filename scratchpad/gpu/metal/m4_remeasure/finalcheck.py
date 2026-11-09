"""Repeat the exact pytest baseline without changing shipped test handling."""
import json, subprocess
from campaign import HERE,ROOT,PYTHON,locked,notice
with locked('final-tests-repeat'):
 p=subprocess.run([PYTHON,'-m','pytest','-q','tests'],cwd=ROOT,capture_output=True,text=True)
 (HERE/'final-tests.txt').write_text(p.stdout+p.stderr)
 last=[line for line in p.stdout.splitlines() if ' passed' in line and ' failed' in line][-1]
 result=dict(exit_code=p.returncode,result=last,matches_baseline='6 failed, 141 passed' in last)
 (HERE/'final-tests.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result),flush=True)
 assert result['matches_baseline'],result
notice('Final validation',last+'; same baseline failures, no shipped source edits.')
