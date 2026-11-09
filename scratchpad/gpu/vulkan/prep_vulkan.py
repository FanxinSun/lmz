"""Prepare real Qwen3 BF16 exponent plane dataset for Vulkan decoder tests and kit."""
import hashlib
import json
from pathlib import Path
import struct
import time
import numpy as np
from lmz import kernels

OUT = Path.home() / "Library/Caches/lmz-vulkan"
OUT.mkdir(parents=True, exist_ok=True)

N = 2048
PLANE = 32768
TOTAL = N * PLANE

print(f"Preparing {N} streams of {PLANE} bytes ({TOTAL / 1e6:.1f} MB plain)...")

model_dir = Path.home() / "models/Qwen3-4B-Instruct-2507"
safetensors = sorted(model_dir.glob("*.safetensors"))
assert safetensors, f"No safetensors found in {model_dir}"

done = 0
hist = np.zeros(256, dtype=np.uint64)
used_tensors = []

ref_file = OUT / "ref.bin"
with open(ref_file, "wb") as dst:
    for path in safetensors:
        with open(path, "rb") as f:
            h_len = struct.unpack("<Q", f.read(8))[0]
            metas = json.loads(f.read(h_len))
            base = 8 + h_len
            for name, meta in sorted(
                ((k, m) for k, m in metas.items() if k != "__metadata__"),
                key=lambda x: x[1]["data_offsets"][0],
            ):
                if meta.get("dtype") != "BF16" or done == TOTAL:
                    continue
                a, b = meta["data_offsets"]
                take = min((b - a) // 2, TOTAL - done)
                f.seek(base + a)
                raw_words = np.frombuffer(f.read(take * 2), dtype="<u2")
                # Exponent plane: (bf16 >> 7) & 0xff
                plane_bytes = ((raw_words >> 7) & 0xFF).astype(np.uint8)
                dst.write(plane_bytes.tobytes())
                hist += np.bincount(plane_bytes, minlength=256).astype(np.uint64)
                used_tensors.append(dict(file=path.name, tensor=name, elements=take))
                done += take
        if done == TOTAL:
            break

assert done == TOTAL, f"Expected {TOTAL} bytes, got {done}"
print(f"Extracted {done} bytes of exponent data.")

# SHA-256 of plain reference data
with open(ref_file, "rb") as f:
    ref_sha256 = hashlib.sha256(f.read()).hexdigest()
print(f"Plain ref SHA-256: {ref_sha256}")

# Calculate histogram, freqs and header
freqs = kernels.rans_freqs(list(map(int, hist)))
header = kernels.rans_shared_header(freqs)
assert len(header) == 516

# Encode each stream
streams_file = OUT / "streams.bin"
offsets = []
coded_total = 0
pos = 0
t0 = time.perf_counter()

print("Encoding streams with shared table...")
with open(ref_file, "rb") as ref, open(streams_file, "wb+") as dst:
    # Header area: N (u32), PLANE (u32), header (516 B), offsets (N * 16 B)
    header_area_size = 8 + 516 + N * 16
    dst.write(struct.pack("<II", N, PLANE))
    dst.write(header)
    dst.write(bytes(N * 16))

    for i in range(N):
        seg = ref.read(PLANE)
        coded = kernels.rans_encode_shared(seg, header)
        assert coded is not None, f"Encoding failed for stream {i}"

        # Verify on CPU
        dec = kernels.rans_decode_shared(coded, PLANE, header)
        assert dec == seg, f"CPU decode mismatch for stream {i}"

        offsets.append((pos, len(coded)))
        dst.write(coded)
        # 16-byte align padding
        pad = (-len(coded)) % 16
        if pad > 0:
            dst.write(bytes(pad))
        pos += len(coded) + pad
        coded_total += len(coded)

        if (i + 1) % 500 == 0 or (i + 1) == N:
            print(f"  Verified and encoded {i + 1}/{N} streams...")

    # Write offsets table
    dst.seek(8 + 516)
    for o, n_bytes in offsets:
        dst.write(struct.pack("<QQ", o, n_bytes))

encode_time = time.perf_counter() - t0
print(f"Encoding finished in {encode_time:.2f}s. Coded bytes: {coded_total} (ratio: {TOTAL / coded_total:.2f}x)")

meta = {
    "nstr": N,
    "plane": PLANE,
    "plain_bytes": TOTAL,
    "coded_bytes": coded_total,
    "padded_payload_bytes": pos,
    "plain_sha256": ref_sha256,
    "header_bytes": 516,
    "encode_time_s": encode_time,
    "tensors_used": len(used_tensors),
}
(OUT / "prep.json").write_text(json.dumps(meta, indent=2))
print(f"Metadata written to {OUT / 'prep.json'}")
print("Data preparation complete.")
