"""Build a libbench input set from lmz's own encoder, with no checkpoint.

`prep_shared.py` and `prep_fused.py` need a real safetensors file and write
about 3 GB. That is the right way to measure a claim about real weights, and
it is the wrong way to answer "what does this card do", because nobody
renting an A100 for twenty minutes is going to reproduce it first.

This writes the same layout from a synthetic exponent plane: a skew close to
what BF16 exponents actually have, coded by `lmz_rans_encode` itself, so the
streams are real lmz streams and the decoder has nothing easier to do.

    python3 scratchpad/gpu/prep_synth.py /tmp/lmzsweep [nstr] [plane]
    nvcc -O3 -std=c++17 -arch=native -o libbench scratchpad/gpu/cuda/libbench.cu
    ./libbench perstream /tmp/lmzsweep
"""
import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from lmz import kernels   # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/lmzsweep"
NSTR = int(sys.argv[2]) if len(sys.argv) > 2 else 28577
PLANE = int(sys.argv[3]) if len(sys.argv) > 3 else 32768

if not kernels.have_rans():
    sys.exit("the native kernel is needed to write the streams")

# A BF16 exponent plane is a few dozen live values with a sharp mode, which is
# what makes it worth coding at all: measured around 2.8 bits a symbol over
# real weights. That number is not decoration -- how often a state refills is
# a direct function of it, and refill rate is what this kernel's speed is
# made of. Coding a flatter buffer would measure a different machine.
TARGET_BITS = 2.8
CENTRE = 0x86


def _slots(decay):
    """256 translate slots in proportion to a two-sided geometric.

    Largest remainder, so the slots sum to exactly 256 and the distribution
    the buffer ends up with is the one that was asked for rather than one
    random sampling of it.
    """
    w = [decay ** abs(v - CENTRE) for v in range(256)]
    tot = sum(w)
    exact = [256 * x / tot for x in w]
    n = [int(x) for x in exact]
    for v in sorted(range(256), key=lambda i: exact[i] - n[i], reverse=True)[
            :256 - sum(n)]:
        n[v] += 1
    return n


def _bits(n):
    from math import log2
    return -sum((c / 256) * log2(c / 256) for c in n if c)


# Bisect the decay to land on the entropy real exponents have.
lo, hi = 0.01, 0.999
for _ in range(60):
    mid = (lo + hi) / 2
    if _bits(_slots(mid)) < TARGET_BITS:
        lo = mid
    else:
        hi = mid
slots = _slots((lo + hi) / 2)
table = bytes(v for v in range(256) for _ in range(slots[v]))
assert len(table) == 256, len(table)
print(f"synthetic plane: {_bits(slots):.2f} bits/symbol over "
      f"{sum(1 for c in slots if c)} live values")

rnd = random.Random(20260821)
streams, offsets, ref = bytearray(), bytearray(), bytearray()
for i in range(NSTR):
    plain = rnd.randbytes(PLANE).translate(table)
    coded = kernels.rans_encode(plain, kernels.histogram(plain))
    if coded is None:
        sys.exit(f"the coder declined stream {i}")
    offsets += struct.pack("<QQ", len(streams), len(coded))
    streams += coded
    ref += plain
    if i % 4096 == 0:
        print(f"\r  {i}/{NSTR}", end="", flush=True)

os.makedirs(OUT, exist_ok=True)
with open(f"{OUT}/streams.bin", "wb") as fh:
    fh.write(struct.pack("<II", NSTR, PLANE))
    fh.write(bytes(offsets))
    fh.write(bytes(streams))
with open(f"{OUT}/ref.bin", "wb") as fh:
    fh.write(bytes(ref))
raw = NSTR * PLANE
print(f"\r{OUT}: {NSTR} x {PLANE} B = {raw/1e6:.1f} MB plain, "
      f"{len(streams)/1e6:.1f} MB coded ({100*(1-len(streams)/raw):.1f}% saved)")
