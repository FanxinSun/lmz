import sys,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path.insert(0,str(H.parent/'m4_remeasure'));from campaign import locked,ROOT,PYTHON,notice
with locked('ship-e3-guard-tests'):
 p=subprocess.run([PYTHON,'-X','faulthandler','-m','pytest','-q','tests/test_lmz.py::test_metal_guarded_pages_and_retained_views','tests/test_lmz.py::test_metal_fused_and_opt_in_chunk','tests/test_lmz.py::test_metal_perstream_stride_partial_groups'],cwd=ROOT,capture_output=True,text=True)
 (H/'audit-tests.txt').write_text(p.stdout+p.stderr);print(p.stdout+p.stderr,flush=True);assert p.returncode==0
notice('E3 protected-page audit','B1 page audit: enough slack before PROT_NONE uses no-copy; exact boundary uses padded copy; both decode byte-identical. Physically short and declared-short inputs do not crash, and a following valid decode works. Retained prepared view, fused output, opt-in chunk and partial SIMD-tail tests pass. ARC is required and now enabled: previous MRC builds made bridge-retained dictionaries dangle and leaked new buffers.')
