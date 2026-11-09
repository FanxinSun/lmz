"""Default per-chunk API rate on the FINAL code: plain decode_batch (no tuning asked) vs the explicitly tuned path."""
import mmap, os, struct, sys, time, statistics as st
from pathlib import Path
sys.path.insert(0, "/Users/masshj/YFCE/lmz"); sys.path.insert(0, "/Users/masshj/YFCE/lmz/scratchpad/gpu/metal/m4_ship")
from bench import load
from lmz import gpu
CACHE = Path.home() / "Library/Caches/lmz-m4-remeasure"
n, p, h, o, s = load(True); ref = (CACHE / "ref.bin").read_bytes()
def aligned(d):
    m = mmap.mmap(-1, ((len(d) + mmap.PAGESIZE - 1) // mmap.PAGESIZE) * mmap.PAGESIZE); m[:len(d)] = d; return m
src, off = aligned(s), aligned(o); out = mmap.mmap(-1, len(ref))
mode = sys.argv[1]
def run():
    t = time.perf_counter(); got = gpu.decode_batch(src, off, n, p, out=out) if mode == "default" else prepared.decode()
    dt = time.perf_counter() - t; assert got is not None
    return dt, got
if mode == "tuned":
    t0 = time.perf_counter(); prepared = gpu.prepare_metal_batch(src, off, n, p, out=out); prep = time.perf_counter() - t0
else: prep = 0.0
for _ in range(2): run()
rates = []
for i in range(7):
    dt, got = run(); rates.append(len(ref) / dt / 1e9)
ok = bytes(got) == ref
import ctypes; lib = gpu._load(); lib.lmz_gpu_metal_perstream_tgs.restype = ctypes.c_uint
print(f"{mode:<8} API {st.median(rates):6.3f} GB/s [{min(rates):.3f}-{max(rates):.3f}]  prepare/tune {prep:5.2f} s  tuned grid {lib.lmz_gpu_metal_perstream_tgs()}  identical {ok}")
