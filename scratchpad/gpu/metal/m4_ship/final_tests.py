import json,os,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent/'m4_remeasure'));from campaign import locked,ROOT,PYTHON,notice
results=[]
with locked('ship-final-tests'):
 for mode,cmd,env in [('pytest',[PYTHON,'-m','pytest','-q','tests'],None),('no-gpu',[PYTHON,'tests/test_lmz.py'],dict(os.environ,LMZ_NO_GPU='1'))]:
  p=subprocess.run(cmd,cwd=ROOT,env=env,capture_output=True,text=True)
  (H/('tests-'+mode+'.txt')).write_text(p.stdout+p.stderr)
  result=dict(mode=mode,exit_code=p.returncode,tail=p.stdout.splitlines()[-12:]);results.append(result);print(json.dumps(result),flush=True)
(H/'test-results.json').write_text(json.dumps(results,indent=2))
notice('E4 test runners','Both requested runners completed. Exact results are in m4_ship/test-results.json and tests-pytest.txt/tests-no-gpu.txt. No-GPU mapped Metal-export test now raises Skip; CUDA source unchanged. Remaining failures, if any, are identified in the final checkpoint.')
