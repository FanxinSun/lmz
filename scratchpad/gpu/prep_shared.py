"""Build a shared-table stream set and prove it round-trips through lmz itself.

Verification is deliberately indirect: prepend the shared 516-byte header to a
shared-table stream and it becomes a valid ordinary lmz stream, so lmz's own
lmz_rans_decode -- untouched -- is the oracle.
"""
import ctypes, glob, json, os, struct, sys, time
import numpy as np

SRC = "/home/rog/.cache/lmz-bench/model.safetensors"
OUT = os.environ.get("OUTDIR", "/home/rog/.cache/lmz-gpu-shared")
PLANE = 32768
KIND = os.environ.get("KIND", "exp")   # exp | sm
os.makedirs(OUT, exist_ok=True)

so = sorted(glob.glob("/home/rog/business/lmz/lmz/native/lmzcore-x86_64-*.so"),
            key=os.path.getmtime)[-1]
lmz = ctypes.CDLL(so)
v = ctypes.c_void_p
lmz.lmz_rans_decode.restype = ctypes.c_int
lmz.lmz_rans_decode.argtypes = [v, ctypes.c_size_t, v, ctypes.c_size_t]
lmz.lmz_rans_encode.restype = ctypes.c_long
lmz.lmz_rans_encode.argtypes = [v, ctypes.c_size_t, v, ctypes.c_size_t]
lmz.lmz_rans_bound.restype = ctypes.c_size_t
lmz.lmz_rans_bound.argtypes = [ctypes.c_size_t]

ext = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "libshared_enc.so"))
ext.lmzx_hist.argtypes = [v, ctypes.c_size_t, v]
ext.lmzx_normalize.restype = ctypes.c_int
ext.lmzx_normalize.argtypes = [v, v]
ext.lmzx_encode_shared.restype = ctypes.c_long
ext.lmzx_encode_shared.argtypes = [v, ctypes.c_size_t, v, v, ctypes.c_size_t]

# ---- real exponent planes ------------------------------------------------
f = open(SRC, "rb")
hlen = struct.unpack("<Q", f.read(8))[0]
hdr = json.loads(f.read(hlen))
base = 8 + hlen
planes = bytearray()
for name, meta in sorted((kv for kv in hdr.items() if kv[0] != "__metadata__"),
                         key=lambda kv: kv[1]["data_offsets"][0]):
    if meta["dtype"] != "BF16":
        continue
    s, e = meta["data_offsets"]
    f.seek(base + s)
    raw = np.frombuffer(f.read(e - s), dtype=np.uint8)
    hi, lo = raw[1::2], raw[0::2]
    w = lo.astype(np.uint16) | (hi.astype(np.uint16) << 8)
    if KIND == "exp":                       # lmz split_bf16 pa: (w >> 7) & 0xFF
        pl = ((w >> 7) & 0xFF).astype(np.uint8)
    else:                                   # pb: ((w >> 8) & 0x80) | (w & 0x7F)
        pl = ((((w >> 8) & 0x80) | (w & 0x7F))).astype(np.uint8)
    planes += pl.tobytes()
nstr = len(planes) // PLANE
planes = bytes(planes[:nstr * PLANE])
print(f"[{KIND}] {nstr} planes x {PLANE} B = {len(planes)/1e6:.1f} MB of real BF16 exponent")

# ---- one table for the whole model --------------------------------------
counts = (ctypes.c_uint64 * 256)()
ext.lmzx_hist(planes, len(planes), counts)
freqs = (ctypes.c_uint16 * 256)()
assert ext.lmzx_normalize(counts, freqs) == 0, "global normalise failed"
nz = sum(1 for i in range(256) if freqs[i])
print(f"shared table: {nz} symbols with non-zero frequency")

shared_header = bytearray(516)
shared_header[0:2] = b"R1"
for s in range(256):
    shared_header[4 + 2 * s] = freqs[s] & 0xFF
    shared_header[5 + 2 * s] = freqs[s] >> 8
shared_header = bytes(shared_header)

# ---- encode both ways ----------------------------------------------------
cap = lmz.lmz_rans_bound(PLANE)
buf = ctypes.create_string_buffer(cap)
per_stream_total = 0
shared_streams, offs = bytearray(), []
t0 = time.perf_counter()
for i in range(nstr):
    seg = planes[i * PLANE:(i + 1) * PLANE]
    assert len(seg) == PLANE
    n = lmz.lmz_rans_encode(seg, PLANE, buf, cap)      # lmz today
    assert n > 0
    per_stream_total += n
    m = ext.lmzx_encode_shared(seg, PLANE, freqs, buf, cap)  # shared table
    assert m > 0, f"shared encode failed on {i}: {m}"
    offs.append((len(shared_streams), m))
    shared_streams += buf.raw[:m]
    shared_streams += b"\0" * (-len(shared_streams) % 16)   # 16 B aligned starts
enc_s = time.perf_counter() - t0
shared_total = sum(n for _, n in offs) + 516   # padding is an artifact of this harness
raw = nstr * PLANE
print(f"\n{'':22}{'bytes':>14}{'of raw':>10}")
print(f"{'per-stream tables':22}{per_stream_total:14,}{100*per_stream_total/raw:9.2f}%")
print(f"{'one shared table':22}{shared_total:14,}{100*shared_total/raw:9.2f}%")
d = per_stream_total - shared_total
print(f"{'difference':22}{d:14,}{100*d/raw:9.2f}%  "
      f"({'shared is smaller' if d > 0 else 'shared is LARGER'})")

# ---- round-trip through lmz's own decoder -------------------------------
ref = ctypes.create_string_buffer(PLANE)
bad = 0
for i in range(0, nstr, max(1, nstr // 400)):       # sample 400 streams
    o, n = offs[i]
    stream = shared_header + bytes(shared_streams[o:o + n])
    rc = lmz.lmz_rans_decode(stream, len(stream), ref, PLANE)
    if rc != 0 or ref.raw[:PLANE] != planes[i * PLANE:(i + 1) * PLANE]:
        bad += 1
print(f"\nround-trip through lmz_rans_decode: {'OK' if not bad else f'{bad} FAILURES'} "
      f"(sampled {len(range(0, nstr, max(1, nstr//400)))} streams)")
assert not bad

with open(f"{OUT}/streams.bin", "wb") as fh:
    fh.write(struct.pack("<II", nstr, PLANE))
    fh.write(shared_header)
    for o, n in offs:
        fh.write(struct.pack("<QQ", o, n))
    fh.write(bytes(shared_streams))
with open(f"{OUT}/ref.bin", "wb") as fh:
    fh.write(planes)
print(f"wrote {OUT}/streams.bin + ref.bin")
