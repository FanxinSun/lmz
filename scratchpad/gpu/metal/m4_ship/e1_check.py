import sys,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent/'m4_remeasure'));from campaign import locked,ROOT,PYTHON,notice
with locked('ship-e1-tail-tests'):
 p=subprocess.run([PYTHON,'-m','pytest','-q','tests/test_lmz.py::test_metal_perstream_stride_partial_groups'],cwd=ROOT,capture_output=True,text=True)
 (H/'e1-tail-tests.txt').write_text(p.stdout+p.stderr);print(p.stdout+p.stderr,flush=True);assert p.returncode==0
notice('E1 tail audit','Padded SIMD-group stride bound fixed; 10 partial-tail stream counts pass full and forced 1/3/7 grids. Tuning oracle capped at 256 MiB; IOKit core discovery replaces subprocess. Real-data settings and kernel-only performance retained for final E3 benchmark.')
