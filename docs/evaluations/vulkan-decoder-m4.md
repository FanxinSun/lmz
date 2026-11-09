# Vulkan rANS Compute Decoder on Apple M4 (MoltenVK)

**Date:** 2026-11-09  
**Machine:** Mac mini Apple M4, 10 GPU cores, 24 GiB unified memory, macOS 15.4.1 (24E263).  
**Operating Conditions:** AC power, idle state monitored; every GPU dispatch executed under mutual exclusion lock (`~/.claude/agy-bridge/locks/m4-gpu`).  
**Software:** MoltenVK 1.2+ translation layer over Metal; SPIR-V 1.3 compiled via `glslangValidator --target-env vulkan1.1` and validated with `spirv-val`; Python 3.11 with pure standard-library `ctypes` host path.

> [!IMPORTANT]  
> **Translation Layer Notice:**  
> All performance figures in this report represent **Vulkan through MoltenVK on an Apple M4: a translation layer over Metal, not a native Vulkan driver**. This measurement validates functional correctness, confirms the portable shader and host harness architecture, and measures the translation layer cost. It is **not** a measurement of native AMD, Intel, or NVIDIA Vulkan drivers.

---

## 1. Executive Summary

A portable, standalone Vulkan 1.1 compute decoder for `lmz` has been implemented, verified, and measured.

1. **Shader Fixes:** Identified and resolved three blocking defects in `lmz/gpu/lmz_rans.comp`:
   - Replaced subgroup-wide ballot counting with per-8-lane group ballot masking for arbitrary subgroup sizes $\ge 8$.
   - Replaced divergent early return before `barrier()` with uniform lockstep execution and guarded output stores.
   - Enabled explicit int8/int64 and 8-bit storage extensions, adhering to standard SPIR-V 1.3 / Vulkan 1.1 compute requirements.
2. **Pure `ctypes` Host Driver:** Implemented in `lmz/gpu/vulkan.py` using standard library only. Dynamically queries all device limits and extensions, resolves MoltenVK portability enumeration, and queries GPU timestamps via `vkCmdWriteTimestamp`.
3. **Correctness (Hard Gate):** 105 of 105 test subtests in `tests/test_vulkan.py` passed with 100% byte equality against CPU decoders across all shapes (`_SHAPES`), planes (4096 and 32768), stream counts ($n_{str} \in [1..1025]$), and real Qwen3 BF16 exponent plane data (2048 streams $\times$ 32 KiB). Zero validation errors under `VK_LAYER_KHRONOS_validation`.
4. **Throughput through MoltenVK:**
   - On a 2,048-stream (67 MB) workload the kit measured 11.71 GB/s (median of 7, range 10.72-14.77) and a 15.06 GB/s sweep peak at G = 40. **That workload is too small to say anything about translation cost**: the same machine's native Metal decoder, on the same 67 MB, measured 11.1 GB/s in one run and 17.9 in the next (GPU clock state; the Vulkan rows themselves averaged 1,230 of 1,578 MHz). An earlier draft of this document called the Vulkan rate "46% of native Metal" by comparing this 67 MB batch with Metal's 936 MB figure; that comparison is withdrawn.
   - **Like-for-like, on the identical 936.6 MB real workload** (manager measurement, 2026-11-09, byte-identical by SHA-256): Vulkan through MoltenVK decodes at **16.6 GB/s full-grid against native Metal's 25.2 (0.66x)**, and the ratio is flat across occupancy (G=40: 0.64-0.65, G=10: 0.64, G=1: 0.63). A constant ratio is a per-lane cost: MoltenVK's translated shader costs about 1.55x per lane (k = 425 vs 270 lane-cycles per byte at G = 1). Both reach their peak at G = 40, the same 4-threadgroups-per-core bound, which is set by the shared-memory footprint and not by the API.
5. **Standalone Kit:** Packaged in `scratchpad/gpu/vulkan/kit/` with embedded self-verifying workload (`kit-workload.bin`, 21.5 MB), verified under clean `env -i`.

---

## 2. Code & Architecture Audit

### 2.1 Shader Specification (`lmz_rans.comp` & `lmz_rans_stride.comp`)

| Property | Vulkan Shared Direct (`lmz_rans.spv`) | Vulkan Stride (`lmz_rans_stride.spv`) | Metal Direct Reference |
|---|---|---|---|
| Workgroup Size (`local_size_x`) | 128 invocations | 128 invocations | 128 threads |
| Stream Groups per Workgroup | 16 groups (8 lanes each) | 16 groups (8 lanes each) | 16 groups |
| Subgroup Ballot Requirement | Multiple of 8 $\ge 8$ (`VK_SUBGROUP_FEATURE_BALLOT_BIT`) | Same | Simdgroup 32 |
| LUT Memory (`s_lut`) | 16,384 B (`shared uint[4096]`) | 16,384 B | 16,384 B |
| LUT Starts (`s_starts`) | 1,024 B (`shared uint[256]`) | 1,024 B | 1,024 B |
| Output Stage (`s_stage`) | 2,048 B (`shared uint8_t[2048]`) | 2,048 B | 2,048 B |
| Total Shared Memory Footprint | **19,456 B** | **19,456 B** | **19,456 B** (compiled) |
| Padding Streams Handling | Lockstep decode of stream 0; store guarded by `gid < pc.nstr` | Lockstep decode; store guarded | Same |
| Barriers | 2 workgroup `barrier()` calls in decode loop | Same | 2 `simdgroup_barrier` calls |
| Output Plane Addressing | `out_planes[gid * plane + out_idx + b]` | Same | Same |

### 2.2 Fix of the Three Shader Defects

1. **Subgroup-Wide Ballot Refill Defect:**
   Vulkan `subgroupBallot` returns a `uvec4` (128 bits total). In subgroups of width 32 (M4) or 64 (AMD RDNA), counting bits across the full ballot returned the refill demand of *other* stream groups.
   *Resolution:* Masked 8-lane group ballot extraction:
   ```glsl
   uint lane = gl_SubgroupInvocationID;
   uint gbase = lane & ~7u;
   uint word_idx = gbase >> 5;
   uint bit_shift = gbase & 31u;
   uvec4 ball = subgroupBallot(need);
   uint raw_word = ball[word_idx];
   uint gm = (raw_word >> bit_shift) & 0xffu;
   uint before = bitCount(gm & ((1u << k) - 1u));
   consumed += 2u * bitCount(gm);
   ```
   Since 32 is a multiple of 8, each 8-lane group is guaranteed to lie within a single 32-bit word of the ballot vector, ensuring portability across subgroup sizes 8, 16, 32, 64, and 128.

2. **Divergent Control Flow and Barrier Desynchronization:**
   Original GLSL code returned early on padding streams: `if (gid >= pc.nstr) return;`. In workgroups with mixed active and padding streams (e.g. the final workgroup when $n_{str} \pmod{16} \neq 0$), active threads encountered `barrier()` while padding threads had returned, causing undefined behavior or hang.
   *Resolution:* All 128 threads remain active in lockstep (`src_id = gid < pc.nstr ? gid : 0u`), safely decoding stream 0 into their slice of shared memory. Writes to device memory are guarded by `if (gid < pc.nstr)`.

3. **Missing Arithmetic & Storage Extensions:**
   GLSL requires explicit extensions for 8-bit and 64-bit storage buffers:
   ```glsl
   #extension GL_KHR_shader_subgroup_basic : enable
   #extension GL_KHR_shader_subgroup_ballot : enable
   #extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
   #extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
   #extension GL_EXT_shader_8bit_storage : require
   ```

---

## 3. Correctness Verification (`tests/test_vulkan.py`)

All tests were executed under the mutual exclusion GPU lock (`m4-gpu`).

```bash
python -m pytest -v tests/test_vulkan.py
```

### 3.1 Test Results Summary

| Suite / Test | Parameters | Subtests | Status | Execution Time |
|---|---|---:|:---:|---:|
| `test_01_probe_and_device` | Device detection & limits query | 1 | PASSED | 0.05 s |
| `test_02_synthetic_shapes_plane_4096` | 5 shapes $\times$ 10 stream counts | 50 | PASSED | 0.82 s |
| `test_03_synthetic_shapes_plane_32768` | 5 shapes $\times$ 10 stream counts | 50 | PASSED | 0.91 s |
| `test_04_real_qwen3_exponent_plane` | 2048 streams $\times$ 32 KiB (Direct + Stride $G \in \{1,2,5,10,40\}$) | 6 | PASSED | 0.16 s |
| **Total** | | **107** | **100% PASSED** | **1.94 s** |

### 3.2 Dataset Specifications
- **Synthetic Shapes:** `single symbol`, `two symbols`, `near uniform`, `dominant + tail`, `skewed`.
- **Stream Counts:** $n_{str} \in \{1, 2, 3, 5, 8, 9, 17, 129, 1023, 1025\}$.
- **Real Model Data:** 2048 streams of 32,768 bytes extracted from `~/models/Qwen3-4B-Instruct-2507/*.safetensors` BF16 exponent plane (`(word >> 7) & 0xff`), total 67,108,864 plain bytes (67.1 MB), encoded with a single shared 516-byte frequency header.
  - Plain Ref SHA-256: `718dddc0032b7f0ea386142fa07ce4d280429bd9f0f507e17f7ab623cdb3662a`
  - Vulkan Direct Decode SHA-256: `718dddc0032b7f0ea386142fa07ce4d280429bd9f0f507e17f7ab623cdb3662a` (**Byte-identical**)
  - Vulkan Stride Decode ($G=1, 2, 5, 10, 40$) SHA-256: `718dddc0032b7f0ea386142fa07ce4d280429bd9f0f507e17f7ab623cdb3662a` (**Byte-identical**)

### 3.3 Validation Layer Results
Running under `LMZ_VK_VALIDATE=1` with `VK_LAYER_KHRONOS_validation` enabled:
- **Zero validation errors**
- **Zero validation warnings**

---

## 4. Measured Performance through MoltenVK

> [!NOTE]  
> All measurements in this section were taken using kernel-only GPU timestamp queries (`vkCmdWriteTimestamp`), calibrated by `timestampPeriod = 1.0` ns/tick, on an Apple M4 with concurrently measured IOReport GPU clock sampling.

### 4.1 Sustained Full-Grid Throughput

- **Workload:** 2048 streams $\times$ 32,768 bytes = 67,108,864 plain bytes.
- **Protocol:** 2 warmup runs, 7 timed runs. First and last runs verified against oracle SHA-256.
- **Grid Configuration:** 128 workgroups (128 threads each = 16,384 resident threads).
- **GPU Clock (IOReport):** Average active frequency **1229.5 MHz**, peak **1578.0 MHz**.

| Run | GPU Time (ms) | Plain Throughput (GB/s) | Validation |
|:---:|:---:|:---:|:---:|
| 1 | 5.73 | 11.71 | SHA-256 Verified |
| 2 | 5.75 | 11.67 | - |
| 3 | 5.90 | 11.38 | - |
| 4 | 5.71 | 11.76 | - |
| 5 | 5.39 | 12.44 | - |
| 6 | 4.54 | 14.77 | - |
| 7 | 6.26 | 10.72 | SHA-256 Verified |
| **Median** | **5.73 ms** | **11.71 GB/s** | **All Byte-Identical** |

#### Comparison with Native Metal Decoder (corrected):

The 67 MB batch above cannot be compared with Metal's 936 MB figure, and the earlier "46.5%" did exactly that. Two measurements replace it.

*Same 67 MB data, native Metal* (`scratchpad/gpu/vulkan/manager/metal_on_kit.py`): kernel 11.11 GB/s in one run (range 10.4-12.6), then 17.9 in a later one. Short kernels on a dynamically clocked GPU swing by 60%, so no
translation-cost conclusion can be drawn from this batch size.

*Identical 936.6 MB real workload, both paths* (`scratchpad/gpu/vulkan/manager/vulkan_big.py`; Metal figures from the Phase 3-5 records): Machine/conditions: Mac mini Apple M4, 10 GPU cores, 24 GiB, macOS 15.4.1, desktop GPU clients present,
kernel GPU timestamps, output SHA-256-identical to the reference in every Vulkan run.

| Occupancy | Native Metal GB/s | Vulkan via MoltenVK GB/s (min-max) | Ratio |
|---|---:|---:|---:|
| full grid | 25.19-25.22 | 16.56 (15.00-16.91) | 0.66 |
| stride G = 40 | 25.67-25.90 | 16.64 (16.58-16.71) | 0.64-0.65 |
| stride G = 10 | 7.26-7.32 | 4.654 (4.651-4.661) | 0.64 |
| stride G = 1 | 0.750 | 0.475 | 0.63 |

The flat ratio says what the cost is: **about 1.55x per lane** in the translated shader, not a bandwidth or tail effect. The causes are not isolated here. Candidates are SPIR-V to Metal Shading Language translation by
SPIRV-Cross inside MoltenVK, 8-bit and 64-bit storage emulation, and the translation of subgroup ballots; this measurement does not separate them. It is the cost of *this translation layer*; a native AMD, Intel or NVIDIA
Vulkan driver compiles SPIR-V directly and may cost anything from the same to much less, which only a run on one of those devices can tell.

---

### 4.2 Grid-Stride Sweep & Lane-Cycle Scaling

Fixed byte traffic (all 2048 streams $\times$ 32 KiB) dispatched using `lmz_rans_stride.spv` across workgroup counts $G \in [1..128]$ (resident lanes $= G \times 128$).
Clock measured per row via IOReport P-state residency.

| $G$ (Workgroups) | Resident Lanes | Median Rate (GB/s) | $k_{ns}$ (ns/B/lane) | $f_{\text{avg}}$ (MHz) | Lane-Cycles / Byte | Scaling Regime |
|---:|---:|---:|---:|---:|---:|:---|
| **1** | 128 | 0.51 | 249.79 | 1575.0 | 393.4 | Linear (compute-bound) |
| **2** | 256 | 1.03 | 249.72 | 1573.4 | 392.9 | Linear (compute-bound) |
| **3** | 384 | 1.52 | 252.22 | 1571.5 | 396.4 | Linear (compute-bound) |
| **5** | 640 | 2.50 | 255.84 | 1569.6 | 401.6 | Linear (compute-bound) |
| **10** | 1,280 | 4.96 | 258.08 | 1560.1 | 402.6 | Linear (compute-bound) |
| **20** | 2,560 | 8.96 | 285.72 | 1552.5 | 443.6 | Knee onset |
| **30** | 3,840 | 12.26 | 313.34 | 1529.1 | 479.1 | Approaching saturation |
| **40** | 5,120 | **15.06** | 339.96 | 1517.3 | 515.8 | **Peak Throughput** |
| **80** | 10,240 | 14.90 | 687.43 | 1536.9 | 1056.5 | Memory/dispatch saturated |
| **128** | 16,384 | 14.88 | 1101.22 | 1520.1 | 1673.9 | Full-grid saturated |

#### Observations:
1. **Constant Computational Cost in Linear Region ($G \le 10$):**
   Across $G = 1$ to $G = 10$, throughput scales strictly proportionally to the number of active workgroups ($\approx 0.50$ GB/s per workgroup). The apparent per-byte computational cost is constant at **$\approx 250$ ns/B/lane**, corresponding to **$\approx 393\text{--}403$ lane-cycles per decoded exponent byte** at $\approx 1.57$ GHz.
2. **Occupancy Knee at $G = 40$:**
   Peak throughput reaches **15.06 GB/s** at $G = 40$. On this 10-core M4 with 19,456 B shared memory per threadgroup, hardware limits occupancy to 4 threadgroups per core ($10 \times 4 = 40$ resident threadgroups). Beyond $G = 40$, throughput levels off at $\approx 14.9$ GB/s, confirming device saturation.

---

## 5. Standalone Benchmark Kit (`scratchpad/gpu/vulkan/kit/`)

A self-contained kit was created for benchmarking target hardware (e.g. 2-CU AMD iGPUs on Windows/Linux) without requiring the `lmz` source tree or developer toolchains:

### 5.1 Kit File Inventory

| File | Size | Purpose |
|---|---:|---|
| `vulkan_sweep.py` | 6.7 KB | Standalone runner (standard library only). |
| `vulkan.py` | 31.2 KB | Pure `ctypes` Vulkan compute host path. |
| `lmz_rans.spv` | 11.5 KB | Precompiled direct rANS decoder SPIR-V 1.3 binary. |
| `lmz_rans_stride.spv` | 12.1 KB | Precompiled grid-stride rANS decoder SPIR-V 1.3 binary. |
| `kit-workload.bin` | 21.5 MB | 2048 streams $\times$ 32 KiB Qwen3 exponent data with oracle hash. |
| `README.txt` | 2.1 KB | Execution instructions and documentation. |

### 5.2 Clean Environment Verification
The kit was validated from an isolated directory under clean `env -i`:
```bash
env -i \
  HOME="$HOME" \
  PATH="/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin" \
  VK_ICD_FILENAMES="/opt/homebrew/opt/molten-vk/etc/vulkan/icd.d/MoltenVK_icd.json" \
  DYLD_FALLBACK_LIBRARY_PATH="/opt/homebrew/lib" \
  /Users/masshj/system-env/customize-llm/venv/bin/python /Users/masshj/YFCE/lmz/scratchpad/gpu/vulkan/kit/vulkan_sweep.py
```
- Exit status: `0`.
- Oracle verification: `PASS: Correctness verified! Plain output SHA-256 matches exactly.`
- Results successfully written to `vulkan_sweep_results.json`.

---

## 6. Files Created & Modified

All edits strictly observed repository file ownership boundaries:

### Permitted Files Created / Modified:
- `lmz/gpu/lmz_rans.comp` (Updated shader source)
- `lmz/gpu/build_spirv.sh` (SPIR-V build and validation script)
- `lmz/gpu/lmz_rans.spv` (Compiled binary)
- `lmz/gpu/lmz_rans_stride.spv` (Compiled binary)
- `lmz/gpu/vulkan.py` (Full pure `ctypes` host path)
- `tests/test_vulkan.py` (Correctness test suite)
- `lmz/docs/evaluations/vulkan-decoder-m4.md` (This document)
- `scratchpad/gpu/vulkan/prep_vulkan.py` (Data preparation)
- `scratchpad/gpu/vulkan/measure_vulkan.py` (Measurement harness with IOReport)
- `scratchpad/gpu/vulkan/package_kit.py` (Kit packager)
- `scratchpad/gpu/vulkan/kit/*` (Standalone kit distribution)

### Untouched Files:
- No Metal sources (`lmz_rans.metal`, `lmzgpu_metal.m`) touched.
- No CUDA sources (`lmzgpu.cu`) touched.
- `lmz/gpu/__init__.py` and `build.py` untouched (Vulkan remains cleanly opt-in).
- No git commands or file operations outside the brief scope.
