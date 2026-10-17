"""Cold-Disk to Unified-Memory Benchmark (LMZ-I0) on Apple Silicon M4.

Compares:
1. Raw safetensors (mmap / sequential read into unified memory)
2. lmz CPU streaming decode (MappedArchive with NEON acceleration)
3. lmz Metal GPU streaming decode (MappedArchive with Metal shared-table acceleration)
"""

import json
import os
import struct
import sys
import tempfile
import time

# Add repository root to path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import lmz


def make_weights_bf16(nelem: int, seed: int = 1) -> bytes:
    """BF16 words whose exponents cluster the way real weights do."""
    # Fast generator using bytearray and tile
    chunk_size = min(nelem, 32768)
    out = bytearray()
    x = seed | 1
    for _ in range(chunk_size):
        x = (x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        mant = (x >> 11) & 0x7F
        exp = 118 + ((x >> 40) % 6)
        sign = (x >> 63) & 1
        out += ((sign << 15) | (exp << 7) | mant).to_bytes(2, "little")
    
    # Repeat to fill requested nelem
    full = bytearray()
    reps = nelem // chunk_size
    rem = nelem % chunk_size
    for _ in range(reps):
        full += out
    if rem:
        full += out[:rem * 2]
    return bytes(full)


def write_safetensors(path: str, tensors: list[tuple[str, str, list, bytes]]) -> None:
    header, offset, blobs = {}, 0, []
    for name, dtype, shape, raw in tensors:
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    header["__metadata__"] = {"format": "pt"}
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * ((-len(blob)) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for raw in blobs:
            fh.write(raw)


def run_benchmark():
    with tempfile.TemporaryDirectory() as d:
        st_path = os.path.join(d, "model.safetensors")
        lmz_path = os.path.join(d, "model.lmz")
        
        target_size_mb = 128
        num_tensors = 16
        bytes_per_tensor = (target_size_mb * 1024 * 1024) // num_tensors
        nelem_per_tensor = bytes_per_tensor // 2
        
        print(f"Generating {target_size_mb} MB safetensors ({num_tensors} tensors × {bytes_per_tensor / 1e6:.1f} MB)...")
        tensors = []
        for i in range(num_tensors):
            tname = f"model.layers.{i}.weight"
            raw = make_weights_bf16(nelem_per_tensor, seed=100 + i)
            tensors.append((tname, "BF16", [nelem_per_tensor], raw))
        write_safetensors(st_path, tensors)
        st_size = os.path.getsize(st_path)
        
        print("\nCompressing with lmz (page-mapped, shared tables)...")
        t0 = time.perf_counter()
        lmz.compress(st_path, lmz_path, mapped=True, shared_tables=True, chunk_size=64*1024)
        comp_time = time.perf_counter() - t0
        lmz_size = os.path.getsize(lmz_path)
        ratio = (1.0 - lmz_size / st_size) * 100.0
        print(f"Compressed in {comp_time:.2f}s: {st_size / 1e6:.1f} MB -> {lmz_size / 1e6:.1f} MB ({ratio:.1f}% saved)")
        
        # 1. Baseline: Raw safetensors mmap / sequential read into unified memory
        print("\n--- 1. Raw safetensors mmap & touch ---")
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            with open(st_path, "rb") as fh:
                import mmap
                mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
                # Touch pages (read full buffer)
                buf = bytes(mm)
                mm.close()
            times.append(time.perf_counter() - t0)
        st_time = min(times)
        st_bw = (st_size / (1024**3)) / st_time
        print(f"Raw mmap + read: {st_time*1000:.2f} ms ({st_bw:.2f} GB/s)")
        
        # 2. lmz CPU streaming decode (MappedArchive)
        print("\n--- 2. lmz CPU Streaming Decode (NEON) ---")
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            with lmz.MappedArchive(lmz_path, cache_blocks=64, gpu=False) as arc:
                buf = arc.read(0, st_size)
            times.append(time.perf_counter() - t0)
        cpu_time = min(times)
        cpu_bw = (st_size / (1024**3)) / cpu_time
        print(f"lmz CPU decode: {cpu_time*1000:.2f} ms ({cpu_bw:.2f} GB/s, effective uncompressed)")
        
        # 3. lmz Metal GPU streaming decode (MappedArchive with GPU decode)
        print("\n--- 3. lmz Metal GPU Streaming Decode ---")
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            with lmz.MappedArchive(lmz_path, cache_blocks=64, gpu=True) as arc:
                buf = arc.read(0, st_size)
            times.append(time.perf_counter() - t0)
        gpu_time = min(times)
        gpu_bw = (st_size / (1024**3)) / gpu_time
        print(f"lmz GPU decode: {gpu_time*1000:.2f} ms ({gpu_bw:.2f} GB/s, effective uncompressed)")

        print("\n================ Comparative Summary ================")
        print(f"Artifact Size: Raw = {st_size/1e6:.1f} MB, lmz = {lmz_size/1e6:.1f} MB (saved {ratio:.1f}%)")
        print(f"Raw safetensors mmap read: {st_time*1000:.1f} ms | {st_bw:.2f} GB/s")
        print(f"lmz CPU streaming decode:  {cpu_time*1000:.1f} ms | {cpu_bw:.2f} GB/s")
        print(f"lmz Metal GPU decode:      {gpu_time*1000:.1f} ms | {gpu_bw:.2f} GB/s")
        print("=====================================================")


if __name__ == "__main__":
    run_benchmark()
