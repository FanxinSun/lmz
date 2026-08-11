"""Reference BF16 for the fused decoder, taken straight from the model file.

The planes were built by walking tensors in data_offsets order and
concatenating, so element i of the plane arrays is BF16 element i of that same
concatenation -- the reference is just those raw bytes, truncated to the
element count the planes cover. Deriving it from the planes instead would make
the test circular.
"""
import json, os, struct
import numpy as np

SRC = "/home/rog/.cache/lmz-bench/model.safetensors"
OUT = "/home/rog/.cache/lmz-gpu-shared"
PLANE = 32768
NSTR = 28577                      # what the plane harness produced

f = open(SRC, "rb")
hlen = struct.unpack("<Q", f.read(8))[0]
hdr = json.loads(f.read(hlen))
base = 8 + hlen
need_elems = NSTR * PLANE
got = bytearray()
for name, meta in sorted((kv for kv in hdr.items() if kv[0] != "__metadata__"),
                         key=lambda kv: kv[1]["data_offsets"][0]):
    if meta["dtype"] != "BF16":
        continue
    s, e = meta["data_offsets"]
    f.seek(base + s)
    got += f.read(e - s)
    if len(got) >= need_elems * 2:
        break
got = bytes(got[:need_elems * 2])
assert len(got) == need_elems * 2
with open(f"{OUT}/bf16.bin", "wb") as fh:
    fh.write(got)
print(f"wrote {OUT}/bf16.bin  {len(got)/1e6:.1f} MB  ({need_elems/1e6:.1f} M BF16 elements)")

# cross-check: merging the two plane files must reproduce it, using lmz's own
# merge formula. If this fails the planes and the reference disagree.
exp = np.fromfile(f"{OUT}/ref.bin", dtype=np.uint8, count=need_elems)
sm = np.fromfile("/home/rog/.cache/lmz-gpu-sm/ref.bin", dtype=np.uint8, count=need_elems)
n = 4 << 20
a, b = exp[:n].astype(np.uint32), sm[:n].astype(np.uint32)
w = ((b & 0x80) << 8) | (a << 7) | (b & 0x7F)
merged = np.empty(n * 2, dtype=np.uint8)
merged[0::2] = (w & 0xFF).astype(np.uint8)
merged[1::2] = (w >> 8).astype(np.uint8)
ok = merged.tobytes() == got[:n * 2]
print(f"merge(exp, sm) == model bytes over first {n/1e6:.1f} M elements: {ok}")
assert ok

# the two halves of the compressed side, for the end-to-end timing
print(f"\ncompressed side:")
a1 = os.path.getsize(f"{OUT}/streams.bin")
a2 = os.path.getsize("/home/rog/.cache/lmz-gpu-sm/ref.bin")
print(f"  exponent streams {a1/1e6:9.1f} MB")
print(f"  sm plane, raw    {a2/1e6:9.1f} MB")
print(f"  total            {(a1+a2)/1e6:9.1f} MB  vs {len(got)/1e6:.1f} MB plain"
      f"  ({100*(1-(a1+a2)/len(got)):.2f}% saved)")
