"""Vulkan (MoltenVK) on the SAME 936.6 MB workload as the Metal 25.2 GB/s measurement: full grid + low-occupancy stride points."""
import hashlib, statistics as st, struct, sys
from pathlib import Path
sys.path.insert(0, "/Users/masshj/YFCE/lmz"); sys.path.insert(0, "/Users/masshj/YFCE/lmz/scratchpad/gpu/metal/m4_ship")
from bench import load
from lmz.gpu import vulkan
CACHE = Path.home() / "Library/Caches/lmz-m4-remeasure"
n, p, h, o, s = load(False); ref = (CACHE / "ref.bin").read_bytes()
s, o, h = bytes(s), bytes(o), bytes(h)
print(f"workload {n} streams x {p} B = {n*p/1e6:.1f} MB; coded {len(s)/1e6:.1f} MB; device {vulkan.device_name()}")
want = hashlib.sha256(ref).hexdigest()
def run(grid, reps):
    rates = []
    for i in range(reps + 1):
        out = vulkan.decode_batch(s, o, n, p, h, grid=grid)
        assert out is not None, vulkan.last_error()
        t = vulkan.last_timing()
        if i: rates.append(n * p / t["gpu_seconds"] / 1e9)
    ok = hashlib.sha256(bytes(out)).hexdigest() == want
    return st.median(rates), min(rates), max(rates), ok
for grid, reps in ((None, 7), (40, 5), (10, 3), (1, 2)):
    m, lo, hi, ok = run(grid, reps)
    print(f"  grid={'full' if grid is None else grid:<5} kernel {m:7.3f} GB/s [{lo:.3f}-{hi:.3f}]  identical {ok}")
