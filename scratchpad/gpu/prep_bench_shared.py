"""Prepare streams.bin and ref.bin for Metal benchmark using native lmz kernels."""
import os
import random
import struct
import sys

from lmz import kernels

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/metalbench"
NSTR = int(sys.argv[2]) if len(sys.argv) > 2 else 4096  # 4096 * 32KB = 134 MB
PLANE = int(sys.argv[3]) if len(sys.argv) > 3 else 32768

print(f"Generating synthetic benchmark set: {NSTR} streams of {PLANE} bytes...")
TARGET_BITS = 2.8
CENTRE = 0x86

def _slots(decay):
    w = [decay ** abs(v - CENTRE) for v in range(256)]
    tot = sum(w)
    exact = [256 * x / tot for x in w]
    n = [int(x) for x in exact]
    for v in sorted(range(256), key=lambda i: exact[i] - n[i], reverse=True)[:256 - sum(n)]:
        n[v] += 1
    return n

def _bits(n):
    from math import log2
    return -sum((c / 256) * log2(c / 256) for c in n if c)

lo, hi = 0.01, 0.999
for _ in range(60):
    mid = (lo + hi) / 2
    if _bits(_slots(mid)) < TARGET_BITS:
        lo = mid
    else:
        hi = mid
slots = _slots((lo + hi) / 2)
table = bytes(v for v in range(256) for _ in range(slots[v]))

rnd = random.Random(20260821)
ref_planes = [rnd.randbytes(PLANE).translate(table) for _ in range(NSTR)]
ref_bytes = b"".join(ref_planes)

# Shared frequency table from pooled sample
counts = kernels.histogram(ref_planes[0])
freqs = kernels.rans_freqs(counts)
hdr = kernels.rans_shared_header(freqs)
assert len(hdr) == 516

streams, offsets = bytearray(), bytearray()
for i, plain in enumerate(ref_planes):
    coded = kernels.rans_encode_shared(plain, hdr)
    assert coded is not None, f"encode failed for stream {i}"
    offsets += struct.pack("<QQ", len(streams), len(coded))
    streams += coded

os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/streams.bin", "wb") as fh:
    fh.write(struct.pack("<II", NSTR, PLANE))
    fh.write(hdr)
    fh.write(bytes(offsets))
    fh.write(bytes(streams))

with open(f"{OUT}/ref.bin", "wb") as fh:
    fh.write(ref_bytes)

raw = NSTR * PLANE
print(f"Wrote {OUT}: {NSTR} x {PLANE} B = {raw/1e6:.1f} MB plain, "
      f"{len(streams)/1e6:.1f} MB coded ({100*(1-len(streams)/raw):.1f}% saved)")
