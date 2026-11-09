========================================================================
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
