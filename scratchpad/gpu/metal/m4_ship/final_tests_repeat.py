import json,os,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent/'m4_remeasure'));from campaign import locked,ROOT,PYTHON,notice
LOCK_OWNER=Path.home()/".claude/agy-bridge/locks/m4-gpu/owner"
results=[]
with locked('ship-final-tests-repeat'):
 parent_env=dict(os.environ,LMZ_PARENT_LOCK_TOKEN=LOCK_OWNER.read_text())
 for mode,cmd,env in [('pytest',[PYTHON,str(H/'pytest_locked.py')],parent_env),('no-gpu',[PYTHON,'tests/test_lmz.py'],dict(os.environ,LMZ_NO_GPU='1'))]:
  p=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True)
  (H/('tests-repeat-'+mode+'.txt')).write_text(p.stdout+p.stderr)
  result=dict(mode=mode,exit_code=p.returncode,tail=p.stdout.splitlines()[-12:]);results.append(result);print(json.dumps(result),flush=True)
(H/'test-results-repeat.json').write_text(json.dumps(results,indent=2))
notice('E4 test runners','Both requested runners completed. Exact results are in m4_ship/test-results.json and tests-pytest.txt/tests-no-gpu.txt. No-GPU mapped Metal-export test now raises Skip; CUDA source unchanged. Remaining failures, if any, are identified in the final checkpoint.')
