"""Package standalone Vulkan benchmark kit for other machines."""
import json
from pathlib import Path
import shutil
import struct

ROOT = Path(__file__).resolve().parents[3]  # lmz root
CACHE = Path.home() / "Library/Caches/lmz-vulkan"
KIT_DIR = ROOT / "scratchpad/gpu/vulkan/kit"
KIT_DIR.mkdir(parents=True, exist_ok=True)

print("Packaging Vulkan Standalone Kit...")

# 1. Create kit-workload.bin
streams_bin = CACHE / "streams.bin"
prep_json = CACHE / "prep.json"
assert streams_bin.is_file(), f"Missing {streams_bin}"
assert prep_json.is_file(), f"Missing {prep_json}"

meta = json.loads(prep_json.read_text())
plain_sha256 = meta["plain_sha256"].encode("ascii")
assert len(plain_sha256) == 64

with open(streams_bin, "rb") as src, open(KIT_DIR / "kit-workload.bin", "wb") as dst:
    nstr, plane = struct.unpack("<II", src.read(8))
    hdr = src.read(516)
    offsets = src.read(nstr * 16)
    payload = src.read()

    # Magic: 8 bytes
    dst.write(b"LMZVKWK\0")
    # Expected SHA256: 64 bytes
    dst.write(plain_sha256)
    # nstr, plane: 8 bytes
    dst.write(struct.pack("<II", nstr, plane))
    # hdr: 516 bytes
    dst.write(hdr)
    # offsets: nstr * 16 bytes
    dst.write(offsets)
    # payload: streams
    dst.write(payload)

wk_size = (KIT_DIR / "kit-workload.bin").stat().st_size
print(f"Created kit-workload.bin ({wk_size / 1e6:.2f} MB, {nstr} streams x {plane} B)")

# 2. Copy binaries and host module
shutil.copy2(ROOT / "lmz/gpu/vulkan.py", KIT_DIR / "vulkan.py")
shutil.copy2(ROOT / "lmz/gpu/lmz_rans.spv", KIT_DIR / "lmz_rans.spv")
shutil.copy2(ROOT / "lmz/gpu/lmz_rans_stride.spv", KIT_DIR / "lmz_rans_stride.spv")
print("Copied vulkan.py, lmz_rans.spv, and lmz_rans_stride.spv to kit.")

# 3. Create README.txt
readme_text = """========================================================================
  lmz Standalone Vulkan rANS Benchmark Kit
========================================================================

PURPOSE
-------
This kit measures GPU rANS compute decode throughput on Vulkan 1.1+
compatible graphics hardware (AMD, Intel, NVIDIA, and mobile SoCs).
It requires only standard Python 3 and the platform's Vulkan runtime.
No C/C++ compiler, SDK, or pip packages are needed.

FILES
-----
- vulkan_sweep.py:       The benchmark driver (standard library only).
- vulkan.py:             Pure ctypes Vulkan loader and compute host path.
- lmz_rans.spv:          Precompiled SPIR-V 1.3 direct compute decoder.
- lmz_rans_stride.spv:   Precompiled SPIR-V 1.3 grid-stride compute decoder.
- kit-workload.bin:      Real Qwen3 BF16 exponent plane dataset (2048 streams
                         x 32 KiB = 67.1 MB plain) with embedded expected SHA-256.
- README.txt:            This documentation.

REQUIREMENTS
------------
1. Python 3.9 or newer, standard library only (tested on 3.9.6 and 3.11; 3.8 is untested).
2. A GPU with a Vulkan 1.1+ driver supporting compute, subgroup ballot
   (subgroup size multiple of 8 >= 8), shaderInt64, shaderInt8, and
   storageBuffer8BitAccess.

HOW TO RUN
----------
On Linux or Windows:
    python vulkan_sweep.py

On macOS (MoltenVK):
    export VK_ICD_FILENAMES=/opt/homebrew/opt/molten-vk/etc/vulkan/icd.d/MoltenVK_icd.json
    export DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib
    python3 vulkan_sweep.py

WHAT THE KIT DOES
-----------------
1. Queries and prints device properties (deviceName, subgroupSize, limits).
2. Verifies decode correctness by comparing plain SHA-256 against the
   workload's embedded oracle hash.
3. Measures sustained full-grid kernel execution throughput (GB/s).
4. Sweeps grid sizes (G = 1..200) with the grid-stride kernel and computes
   k_ns (nanoseconds per decoded byte per resident lane).
5. Writes full structured benchmark data to 'vulkan_sweep_results.json'.

RUN IT TWICE, AND READ THE RATES AS INDICATIVE
----------------------------------------------
The workload is small (67 MB), so each timed kernel lasts milliseconds to a fraction
of a second, and a GPU's clock state changes the rate. On one Apple M4 the SAME code
gave 0.51 GB/s in one run and 0.37 GB/s in another at G = 1 (a single workgroup barely
loads the GPU, so it may not boost), and the native Metal decoder gave 11.1 and 17.9 GB/s
on the same 67 MB. So: use "Best performance" power mode, run the kit at least twice,
at different times, send every result file, and do not read one run as a clean number.
Correctness (the SHA-256 check) does not depend on any of this.

WHAT THE KIT DOES NOT MEASURE
-----------------------------
0. The pool-per-unit question the 2-CU investigation needs is answered by the D3D11
   script in the parent kit (igpu-pool.ps1), not by this one.
1. This kit measures compute decode kernel time (via GPU timestamp queries);
   it does not include CPU host setup, buffer allocation, or PCIe/host transfer time.
2. Results on macOS represent MoltenVK translation over Apple Metal, not a native
   Vulkan driver implementation.

WHAT TO SEND BACK
-----------------
Please submit:
1. Console stdout log.
2. The generated 'vulkan_sweep_results.json' file.
========================================================================
"""
(KIT_DIR / "README.txt").write_text(readme_text)
print("Created README.txt.")

print("Kit packaging complete.")
