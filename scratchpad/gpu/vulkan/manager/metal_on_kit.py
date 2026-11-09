"""Native Metal on the SAME 2048-stream kit workload the Vulkan kit uses (like-for-like), GPU-timestamp kernel time."""
import hashlib, struct, sys, statistics as st, ctypes, mmap
sys.path.insert(0, "/Users/masshj/YFCE/lmz")
from lmz import gpu
with open("/Users/masshj/YFCE/lmz/scratchpad/gpu/vulkan/kit/kit-workload.bin", "rb") as f:
    f.read(8); want = f.read(64).decode("ascii"); nstr, plane = struct.unpack("<II", f.read(8))
    hdr = f.read(516); offs = f.read(nstr * 16); streams = f.read()
def aligned(d):
    m = mmap.mmap(-1, ((len(d) + mmap.PAGESIZE - 1) // mmap.PAGESIZE) * mmap.PAGESIZE); m[:len(d)] = d; return m
src, off = aligned(streams), aligned(offs); out = mmap.mmap(-1, nstr * plane)
lib = gpu._load(); lib.lmz_gpu_metal_last_timing.restype = ctypes.POINTER(ctypes.c_double)
for _ in range(3): gpu.decode_batch(src, off, nstr, plane, out=out, header=hdr)
rates, walls = [], []
for _ in range(11):
    import time; t = time.perf_counter(); got = gpu.decode_batch(src, off, nstr, plane, out=out, header=hdr); walls.append(time.perf_counter() - t)
    rates.append(nstr * plane / lib.lmz_gpu_metal_last_timing()[3] / 1e9)
ok = hashlib.sha256(bytes(got)).hexdigest() == want
print(f"native Metal, {nstr} streams x {plane} B = {nstr*plane/1e6:.1f} MB: kernel {st.median(rates):.2f} GB/s [{min(rates):.2f}-{max(rates):.2f}], API {nstr*plane/st.median(walls)/1e9:.2f} GB/s, sha256 match {ok}")
