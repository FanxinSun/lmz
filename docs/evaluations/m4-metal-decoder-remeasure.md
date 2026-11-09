# Apple M4 Metal decoder remeasurement

Measured 2026-11-09. Scope: new measurement files only; shipped decoder and model files unchanged. Machine: Mac mini Apple M4, 10 GPU cores, 24 GiB unified memory (hw.memsize=25769803776). Conditions and every timed run will be preserved in `scratchpad/gpu/metal/m4_remeasure/runs.jsonl`. No downloads, installs, or git operations.

## Phase 0 audit

Machine/conditions for this code-comparison table: Apple M4 Metal source versus RTX 5080 CUDA source at the published 192-thread configuration; static inspection, no performance inference.

| Property | Metal shared direct | CUDA k_shared |
|---|---|---|
| Threads / groups per TG or block | 128 / 16 | 192 / 24 |
| LUT | 16,384 B | 16,384 B |
| LUT construction starts | 1,024 B | 1,024 B static shared |
| Output stage per group | 128 B | 128 B |
| Input ring per group | 256 B declared, unused for false template | 512 B |
| Declared total | 23,552 B | 32,768 B including static starts |
| Reads | dependent device byte loads | cp.async four-slot ring, aligned source |
| Grid | one stream per 8 lanes, no stride | same |
| Partial last TG/block | padding groups decode stream zero; stores guarded by gid<nstr | same |
| LUT barriers | two threadgroup barriers | two __syncthreads |
| Per-stream barriers | simdgroup barriers only | masked/full __syncwarp and pipeline waits |

Metal's per-stream template has no `threadgroup_barrier`, so a stream grid-stride wrapper can call it after constructing the LUT once per TG. The wrapper calls the shipped template directly, appended to the unchanged source at runtime.

The historical input is synthetic: `scratchpad/gpu/prep_bench_shared.py` builds random byte planes translated through an approximately 2.8-bit distribution centered at 0x86, seed 20260821, 4,096 × 32,768 B. Its shared table comes from the first plane, not real BF16 weights. The direct numerator is 134,217,728 exponent-plane bytes. 134,217,728 / 0.00556 / 1e9 = 24.14 GB/s. Fused output counts twice that many bytes: 268,435,456 / 0.00739 / 1e9 = 36.32 GB/s. Counting the original exponent byte count instead produces the incompatible 3.69 ms row. It is not evidence of a second measurement.

**Correction to the brief's timer attribution:** the Python `decode_batch` route includes a streams padding copy, newBufferWithBytes copies of streams/offsets/header, an output allocation, synchronous GPU execution and output memcpy. A Python perf_counter around that call is a warm host-to-host wall measurement, not kernel timing. However, `scratchpad/gpu/metal/bench.swift` already preallocates buffers, uses gpuEndTime-gpuStartTime and reports the best of seven. Its direct/fused names match the first historical evaluation table. There is no retained raw run log establishing which harness produced the exact published 24.1 and 36.3. The claim that they necessarily came from the Python timer is unsupported. The evaluation's later IO table reuses the decoder rates as cold-load numbers without corresponding batched/fused arms in `scratchpad/bench_unified_memory_io.py`.

Other findings: the old evaluation says 16 GB but this host has 24 GiB. `plan/PLAN.md` simultaneously says Metal is written but unrun and marks LMZ-T0 COMPLETE. The evaluation's 19,456 B footprint omits the declared input ring; the compiler may eliminate it for the direct specialization, so pipeline static memory must be distinguished from declarations. The CUDA cost model also omits its 1,024 B static starts from the stated allocation formula (not enough to change four-block residency on the measured 5080). `cost_model()` is CUDA evidence, not an already measured Metal cost. Source records disagree on the CUDA source model (scratchpad README Qwen2.5; cost_model Llama), so this campaign matches its stream shape, not a claimed identical source distribution.

The normal standalone test runner is documented in `tests/test_lmz.py`; pytest handles its custom Skip exceptions as failures. Requested pytest baseline: **141 passed, 6 failed**, 147 collected, 1 warning, 85.46 s. Failures are the CUDA-only device entry test and five FUSE mount tests missing fusermount3; no fixes made. Baseline suite ran before campaign lock acquisition; its outputs are not calibration data.

## Reproduction

Requires an installed Apple developer toolchain (`swiftc`, Metal and Foundation SDK), existing Python environment with NumPy and pytest, local Qwen3-4B-Instruct-2507 BF16 safetensors, and the existing newest lmz native dylib. This harness needs a compiler to build. Shipped lmz's own Metal host path builds via clang on first use unless a matching prebuilt dylib exists; a stock Mac without developer tools cannot build that path. This campaign does not fix packaging or toolchain requirements.

From the lmz root, use `/Users/masshj/system-env/customize-llm/venv/bin/python scratchpad/gpu/metal/m4_remeasure/campaign.py prep`, then the same Python with `measure.py 2`, `measure.py 3`, `measure.py 2-synthetic`, and `measure.py 3b`. These entrypoints take the shared GPU lock, sample idle utilization and save conditions. Low-grid runs and high-grid runs have separate lock periods. The harness uses newLibraryWithSource on `lmz/gpu/lmz_rans.metal` at runtime, not a copied kernel. Output comparisons follow GPU completion and are excluded from both GPU and recorded command-buffer wall timing. Each dispatch uses 128 threads. Phase 2 uses ceil(28,584/16)=1,787 TGs. The output is poisoned before runs; first and last timed outputs at every G are compared across all bytes.

## Measured results

Machine/conditions: Mac mini Apple M4 (10 GPU cores), 24 GiB unified, macOS 15.4.1 (24E263), AC power; no pmset thermal/performance warning, Parallels VM running and untouched; 2026-11-09. Decimal GB/s. Two warmups, then seven timed runs per full-grid arm, five per sweep G. All decoder measurement campaign idle medians ≤5% after retries; per-row foreign-time flags below.

Real source: local Qwen3-4B-Instruct-2507, BF16 exponent `(word >> 7) & 255`, taken in file/data-offset order until 936,640,512 bytes. This is the first 936.64 MB of exponent plane, not a whole-model sample. One global table covers all streams; all 28,584 streams round-trip after prepending the 516-byte shared header through the unmodified native ordinary-stream decoder. Native library: `/Users/masshj/YFCE/lmz/lmz/native/lmzcore-arm64-cd8585a1c0bf.dylib`. Coded bytes: 314,804,856; including shared header: 314,805,372; 16-byte-aligned payload: 315,004,416. Expansion = 936,640,512 / 314,805,372 = 2.975300. All tensor names, consumed element counts and file paths are in summary.json; the data spans the embedding, complete tensors from layers 0, 1, 10, 11, 12 and the beginning of layer 13, ending with a partial layer-13 gate projection.

The direct compiled pipeline reports **19,456 B** static TG memory. Thus the 23,552 B source declaration is not the actual direct-specialization allocation: the compiler eliminates the unused 4,096 B input ring. The old evaluation footprint is correct for this pipeline, although its source inventory is incomplete.

Machine/conditions: Mac mini Apple M4 (10 GPU cores), 24 GiB unified, macOS 15.4.1 (24E263), AC power; no pmset thermal/performance warning, Parallels VM running and untouched; 2026-11-09. Decimal GB/s. Two warmups, then seven timed runs per full-grid arm, five per sweep G. All decoder measurement campaign idle medians ≤5% after retries; per-row foreign-time flags below. First two rows precede foreign-client instrumentation; paired rows use it.

| Arm | Median GPU ms (min–max) | Median wall ms (min–max) | Median plain GB/s | Foreign time / own timed GPU sum |
|---|---:|---:|---:|---:|
| Real shipped, original Phase 2 | 37.1538 (37.1081–37.3259) | 37.9108 (37.7801–37.9925) | 25.2098 | not collected |
| Real Python decode_batch, out supplied | — | 473.0328 (282.2653–1016.0967) | 1.9801 | not collected |
| Real shipped, paired drift check retry | 37.1812 (37.0999–40.9217) | 37.8809 (37.7303–41.6962) | 25.1913 | 18.36% (flagged) |
| Real stride wrapper, paired retry | 36.8286 (36.7327–37.2603) | 37.5264 (37.3330–37.8316) | 25.4324 | 16.40% (flagged) |
| Trivial one-TG command-buffer control | 0.0028 (0.0027–0.0031) | 0.1735 (0.1659–0.1885) | 0.0000 | not collected |
| Historical synthetic shape, shipped retry | 5.7795 (5.7364–5.8301) | 6.4488 (6.2648–6.5814) | 23.2231 | 13.48% (flagged) |

Python wall timing includes synchronous buffer setup/copies, GPU dispatch/wait and output memcpy; its caller output bytearray is preallocated. It excludes file loading and output comparison. Harness wall timing covers command-buffer creation through completion; preallocated inputs and output, output poisoning, correctness comparisons and shader compilation are outside that timer. GPU time is GPUEndTime−GPUStartTime. The trivial kernel control uses the same buffers/encoder setup, writes one uint and launches one 128-thread TG. It is not a mathematically empty command buffer and its overhead has not been subtracted from decoder times. First/last timed harness output comparisons cover the full reference, and every Python timed output is compared; all pass.

Shipped `lmz.gpu.verify()` (no quick mode): ok=True, checked=30, failures=[], its own synthetic per-stream-table wall-timed throughput 2.52 GB/s. That is a separate workload and path, not the shared-table kernel rate.

Reconciliation of 24.1: the retained synthetic generator reproduces 134.2 MB plain / 46.7 MB coded / 65.2% saved, and the unchanged shipped kernel now gives 23.223 GB/s median kernel-only, versus the historical 24.1 (3.64% lower). The original real-data kernel result is 25.210 GB/s (4.61% above 24.1). These are workload/conditions-specific results, not evidence that the old timer was the Python API. The existing Swift GPU timer and the old fused 268.4 MB / 7.39 ms arithmetic support a kernel-only interpretation of the first historical table. Raw historical traces are absent, so exact numerical provenance cannot be recovered. Its later cold-IO and fused 3.69 ms rows cannot be treated as equivalent measurements. The current API wall result is about 12.7× slower than kernel-only on the real full-grid data, demonstrating why the methods must be labeled separately.

## Sweep and cost arithmetic

Machine/conditions: Mac mini Apple M4 (10 GPU cores), 24 GiB unified, macOS 15.4.1 (24E263), AC power; no pmset thermal/performance warning, Parallels VM running and untouched; 2026-11-09. Decimal GB/s. Two warmups, then seven timed runs per full-grid arm, five per sweep G. All decoder measurement campaign idle medians ≤5% after retries; per-row foreign-time flags below. Every row below uses the required retry because its first attempt exceeded 2% foreign time. Both attempts and all warmups are retained in runs.jsonl. All rows decode the identical 28,584 × 32,768 B workload; all first/last timed output comparisons pass.

| TGs G | Median GPU ms | Plain GB/s (min–max) | GB/s per launched TG | Foreign % | Largest foreign contributor |
|---:|---:|---:|---:|---:|---|
| 1 | 1254.823 | 0.7464 (0.7441–0.7469) | 0.74643 | 3.48 | 428 WindowServer |
| 2 | 643.973 | 1.4545 (1.3849–1.4628) | 0.72724 | 28.33 | 77552 Google Chrome He |
| 3 | 420.851 | 2.2256 (2.2220–2.2288) | 0.74186 | 4.71 | 428 WindowServer |
| 4 | 316.944 | 2.9552 (2.9427–2.9630) | 0.73881 | 7.21 | 428 WindowServer |
| 5 | 253.893 | 3.6891 (3.6862–3.6907) | 0.73782 | 5.29 | 428 WindowServer |
| 6 | 212.107 | 4.4159 (4.4129–4.4188) | 0.73598 | 6.78 | 428 WindowServer |
| 8 | 159.790 | 5.8617 (5.8545–5.8694) | 0.73271 | 5.05 | 428 WindowServer |
| 10 | 128.710 | 7.2772 (7.2377–7.3038) | 0.72772 | 6.88 | 428 WindowServer |
| 12 | 108.001 | 8.6725 (8.6464–8.6902) | 0.72271 | 7.49 | 428 WindowServer |
| 15 | 87.024 | 10.7630 (10.6795–10.7819) | 0.71753 | 7.65 | 428 WindowServer |
| 20 | 66.400 | 14.1060 (14.0329–14.1099) | 0.70530 | 3.41 | 624 Terminal |
| 25 | 54.502 | 17.1854 (16.3334–17.2325) | 0.68742 | 3.46 | 624 Terminal |
| 30 | 46.138 | 20.3008 (19.1381–20.4001) | 0.67669 | 3.78 | 624 Terminal |
| 40 | 36.198 | 25.8758 (22.7329–25.9369) | 0.64690 | 4.80 | 624 Terminal |
| 50 | 44.127 | 21.2261 (17.4759–27.5597) | 0.42452 | 15.08 | 624 Terminal |
| 60 | 45.917 | 20.3986 (20.3804–21.6971) | 0.33998 | 14.14 | 624 Terminal |
| 80 | 36.120 | 25.9312 (24.5330–25.9644) | 0.32414 | 12.23 | 624 Terminal |
| 100 | 41.608 | 22.5109 (21.4920–30.8153) | 0.22511 | 14.83 | 624 Terminal |
| 120 | 45.037 | 20.7969 (17.9635–25.5742) | 0.17331 | 13.75 | 624 Terminal |
| 160 | 42.592 | 21.9912 (21.9345–25.3711) | 0.13744 | 17.08 | 624 Terminal |
| 200 | 36.070 | 25.9671 (24.0029–26.9060) | 0.12984 | 26.54 | 624 Terminal |
| 300 | 37.498 | 24.9787 (23.7196–25.2417) | 0.08326 | 7.66 | 624 Terminal |
| 500 | 36.955 | 25.3453 (22.8761–25.6648) | 0.05069 | 18.67 | 624 Terminal |
| 1000 | 36.397 | 25.7338 (22.0932–25.8655) | 0.02573 | 31.82 | 624 Terminal |
| 1787 | 36.829 | 25.4324 (25.1378–25.4988) | 0.01423 | 16.40 | 624 Terminal |

Full-grid drift = (25.432405 / 25.191262 − 1) × 100 = 0.957%, within the required 3%. The instrument calls shipped decode_plane_shared<false> unchanged, and build_lut_shared once before the loop.

A conservative linear region is G=[1, 2, 3, 4, 5, 6, 8, 10, 12]. Fitting rate = slope × G through the origin gives slope = Σ(G × rate) / Σ(G²) = **0.728895 GB/s per TG**. The largest relative residual is 2.41%. Gradual slope loss becomes visible around G=15–30; the first full-rate peak is G=40 at 25.8758 GB/s. Extrapolating the low-grid slope to that peak gives 35.50 TGs. Together with the first observed peak at 40 TGs, this suggests roughly 35–40 simultaneously useful TGs, or **about 3.5–4 TGs per core across ten cores**. The integer candidate is four TGs/core. This is an inference from the curve, not measured hardware residency: no Metal per-core shared-memory pool/residency counter was obtained.

The later rate is not a smooth bandwidth plateau: G=50/60/100/120/160 have substantial troughs, while G=40/80/200 approach 26 GB/s. Wave scheduling, TG distribution and background load are possible explanations; this campaign does not distinguish them. Dividing all launched lanes by those later rates would invent a compute constant. The maximum TG memory per launch is not a per-core pool and cannot settle this occupancy question.

Using only the strict linear rows, assuming G resident TGs, resident_lanes=128G and rate in GB/s (numerically bytes/ns): **k_ns = 128G / rate = 171.48–177.11 lane-ns per decoded byte**. For example, G=1: 128/0.746432=171.48; G=12: 1536/8.672517=177.11. The reciprocal fitted slope gives 128/0.728895=175.61 lane-ns/B. A sensitivity interval that admits all rising pre-peak rows through G=30 is 171.48–189.16; those less-linear rows can mix scheduling/early saturation effects and are excluded from the primary interval. Saturated rows are excluded entirely, following cost_model()['provenance']['supersedes']. This interval comes from 1–12 launched/resident TGs in total (partial GPU activation), not from four resident TGs on every core. The inferred four-TG/core bend is a separate quantity; this campaign does not establish a cost interval specifically at that full occupancy.

Traffic factor = 1 + 1/expansion = **1.336101**. At full-grid wrapper rate 25.432405 GB/s, implied traffic = 33.9802 GB/s and achieved_fraction = 25.432405 × 1.336101 / nominal 120 = **0.283169**. This is an effective fraction computed from byte counts and nominal DRAM specification, not a DRAM-counter measurement or evidence that DRAM binds. The first-peak calculation is 0.288106. Padding adds 199,560 B of stored input; using padded payload changes the traffic factor to 1.336313.

## Phase 3b: measured GPU frequency and lane-cycles

Machine/conditions: Mac mini Apple M4 (10 GPU cores), 24 GiB unified, macOS 15.4.1 (24E263), AC power; no pmset thermal/performance warning, Parallels VM running and untouched; 2026-11-09. Decimal GB/s. Two warmups, then seven timed runs per full-grid arm, five per sweep G. All decoder measurement campaign idle medians ≤5% after retries; per-row foreign-time flags below. Phase 3b repeats G=1,2,5,10,20,30,40,80,200,1787 with five timed runs and two warmups per attempt. G=1/2 pass the 2% foreign-time criterion on the first attempt; all higher rows use the required retry and remain flagged. Every output check passes.

The supplied gpufreq.py was copied unchanged from `/Users/masshj/.claude/agy-bridge/inbox/gpufreq.py` into this measurement directory. It subscribes without sudo to IOReport GPU Stats / GPU Performance States / GPUPH, snapshots immediately after the foreign-client pre-snapshot before go, then again after the final timed result beside the foreign-client post-snapshot. **f_avg_active_mhz is measured residency-weighted GPU frequency**, excluding OFF time: Σ(state_seconds × state_MHz) / Σ(active_state_seconds). The state table comes from IORegistry pmgr voltage-states9 (u32 Hz, u32 mV pairs), mapped by index exactly. It is **non-monotonic in index order**; sorting frequencies would assign residency to the wrong clock. The sampler assumes the local 24 MHz residency timebase; the active-frequency ratio does not depend on that scale. It is a device-wide active-state window, not per-dispatch clock telemetry or an FMA estimate. Host gaps, checks and foreign GPU work can contribute to the window.

State index 0=OFF; P1..P15 MHz in index order: `[338, 618, 796, 928, 952, 1056, 1053, 1170, 1152, 1278, 1204, 1338, 1326, 1470, 1578]`. The maximum is 1578 MHz; examples of reversals are P6=1056/P7=1053 and P8=1170/P9=1152. The directly sampled frequency resolves the low-grid ramp concern for these runs: G=1/2 average about 1575 MHz, near the highest state, rather than requiring a guessed clock.

Machine/conditions: Mac mini Apple M4 (10 GPU cores), 24 GiB unified, macOS 15.4.1 (24E263), AC power; no pmset thermal/performance warning, Parallels VM running and untouched; 2026-11-09. Decimal GB/s. Two warmups, then seven timed runs per full-grid arm, five per sweep G. All decoder measurement campaign idle medians ≤5% after retries; per-row foreign-time flags below. Phase 3b; row medians below saturation use 128G resident lanes. Cycles are arithmetic derived from measured rate and measured row frequency.

| G | GB/s | Active MHz | k_ns lane-ns/B | k lane-cycles/B | Foreign % |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.7497 | 1574.758 | 170.74 | 268.87 | 0.77 |
| 2 | 1.4938 | 1574.765 | 171.37 | 269.87 | 1.39 |
| 5 | 3.7016 | 1571.937 | 172.90 | 271.79 | 2.41 |
| 10 | 7.3040 | 1558.158 | 175.25 | 273.06 | 3.40 |
| 20 | 14.0387 | 1562.196 | 182.35 | 284.87 | 2.21 |
| 30 | 20.3894 | 1537.793 | 188.33 | 289.62 | 5.33 |
| 40 | 25.8005 | 1509.174 | — | — | 5.95 |
| 80 | 25.9253 | 1540.219 | — | — | 6.10 |
| 200 | 23.2109 | 1517.045 | — | — | 11.58 |
| 1787 | 25.4000 | 1549.208 | — | — | 9.89 |

Machine/conditions: Mac mini Apple M4 (10 GPU cores), 24 GiB unified, macOS 15.4.1 (24E263), AC power; no pmset thermal/performance warning, Parallels VM running and untouched; 2026-11-09. Decimal GB/s. Two warmups, then seven timed runs per full-grid arm, five per sweep G. All decoder measurement campaign idle medians ≤5% after retries; per-row foreign-time flags below. Phase 3b IOReport state-duration records for the selected attempt, seconds of GPU residency, excluding OFF. Full per-state data, OFF time, active time and frequency table are stored in each run and summary.json.

| G | Active s | OFF s | by_state_s |
|---:|---:|---:|---|
| 1 | 6.284314 | 0.925931 | P1=0.003205, P3=0.016922, P8=0.005901, P10=0.002536, P15=6.255750 |
| 2 | 3.159367 | 0.272034 | P1=0.002282, P3=0.006210, P8=0.004357, P10=0.002524, P15=3.143994 |
| 5 | 1.290239 | 0.199080 | P1=0.001447, P3=0.004465, P8=0.004353, P10=0.002537, P15=1.277438 |
| 10 | 0.672413 | 0.200604 | P1=0.001992, P3=0.010652, P8=0.004369, P10=0.002531, P15=0.652868 |
| 20 | 0.346860 | 0.207174 | P1=0.001087, P3=0.002012, P8=0.004419, P10=0.002522, P15=0.336819 |
| 30 | 0.259501 | 0.195588 | P1=0.002433, P3=0.006262, P8=0.004347, P10=0.002490, P15=0.243970 |
| 40 | 0.207321 | 0.196872 | P1=0.001866, P3=0.012034, P8=0.004392, P10=0.002509, P15=0.186520 |
| 80 | 0.206338 | 0.195360 | P1=0.001832, P3=0.003825, P8=0.004351, P10=0.002524, P15=0.193805 |
| 200 | 0.219918 | 0.194314 | P1=0.001647, P3=0.011322, P8=0.004324, P10=0.002485, P15=0.200140 |
| 1787 | 0.203344 | 0.201454 | P1=0.001940, P3=0.001167, P8=0.004358, P10=0.002530, P15=0.193350 |

For each below-saturation row, **k_cycles = (128G / rate_GBps) × f_avg_active_mhz / 1000**. Example G=1: (128 / 0.749687878) × 1574.757703/1000 = 268.87 lane-cycles/B. Across G=[1, 2, 5, 10, 20, 30], row-median k_ns is 170.74–188.33 and the **conditional lane-cycle interval is 268.87–289.62**. Strict linear subset G=1,2,5,10 gives 268.87–273.06. Only the G=[1, 2] rows pass the foreign-time criterion, giving 268.87–269.87 across their medians at partial GPU activation. That two-row range is not a validated four-TG/core cost bracket. Rows G≥40 are excluded from k conversion because saturation/scheduling obscures compute cost.

This supersedes the earlier no-clock status: a separate FMA clock result at the originally requested probe path is no longer needed. The conversion uses sampled IOReport residency and the indexed IORegistry frequency table. It does not remove the foreign-work flags on G≥5 or prove a fully resident per-core cost. No old Phase-3 row is retroactively assigned the Phase-3b clock.

## Measurement limits and foreign clients

The coordinating manager session (lmsluice-manager, Claude Code) amended hygiene after the initial Phase-2 campaign began. Those original runs have campaign idle samples but no per-G foreign snapshots. Phase 3b adds per-row IOReport frequency sampling. Subsequent paired drift checks and the entire sweep have before/after AGXDeviceUserClient snapshots, summing every AppUsage accumulatedGPUTime counter except clients owned by the harness PID. The harness gates the first timed dispatch until the parent snapshot completes; the parent snapshots again upon the final run result. The window includes host gaps/output checks as well as timed dispatches, so foreign time is a conservative activity indicator, not exact simultaneous interference. Client IDs, creator PIDs/names, counter totals, deltas and top contributor are retained. Disappearing clients are separately listed; no counter deltas are silently assigned to our PID.

Every original Phase-3 amended sweep row exceeded 2%, was redone exactly once, and remained flagged. Phase 3b has two clean rows, G=1 and G=2, as shown above. Results select the required second attempt, never the faster attempt. Original attempts remain available. The recorded contributors include WindowServer, Terminal and Google Chrome Helper. The client inventory also contains prl_client_app, prl_vm_app and a com.microsoft.te client; their presence is not evidence they were the top contributor. No VM, applications or other executor sessions were changed. GPU lock prevents cooperating campaigns overlapping, but does not suppress desktop clients. Initial prep idle samples were 24/22/18/5/5%; its median failed, so it waited and rechecked 4/0/0/0/0 before encoding. All successful start checks and thermal/power/OS/VM records are in summary.json.

**Consequently the broader k_ns/k_cycles intervals and inferred occupancy are conditional observations under the retained desktop background.** Only Phase-3b G=1/2 satisfy the foreign-time criterion; they measure partial GPU activation rather than full per-core occupancy. The broader multi-occupancy interval does not meet the ≤2% foreign-time criterion across all its rows. The rate curve is still useful evidence; a stronger claim needs another campaign with acceptable foreign-time rows.

## Validation and artifacts

The first final rerun had 140 passed / 7 failed: an additional custom Skip from WebDAV mount error 19. Its output/result are retained in final-tests-attempt1.*; this is an OS mount availability failure with unchanged source. The second exact suite run cleared that mount failure and matched the baseline pass/failure counts.

Exact final pytest rerun: **6 failed, 141 passed, 1 warning in 83.82s (0:01:23)**. Collected the original 147-test total. Pass/failure counts match baseline: **True**. Full output is in final-tests.txt. Reproduction of the baseline issue: `/Users/masshj/system-env/customize-llm/venv/bin/python -m pytest -q tests`; individual minimal examples are `tests/test_lmz.py::test_gpu_device_entry_point_owns_nothing` (CUDA-only Skip) and `tests/test_lmz.py::test_fs_writes_are_compressed_and_read_back` (missing fusermount3 Skip). No fixes applied.

All implementation/report files are new, under the requested scratchpad directory and this new evaluation path. No git commands, downloads, installs or shipped-source edits were made. A read-only direct parse of the existing Git index and blob hashing checked all 83 indexed entries: zero content mismatches or missing files (tracked-content-check.json). SHA was read from the local ref file; shipped source SHA256 hashes are in summary.json. Optional TG variants, fused real BF16 and per-stream real-data measurements were not run. They are not implied by the shared-table results. This shared-table input is not a newly written lmz archive or a cold-file/inference benchmark.

Build and run prerequisites and entrypoints are above. Regenerate summary/report after a completed campaign with analyze.py then render.py using the same Python. runs.jsonl contains every timed dispatch, warmup and both attempts; foreign row-level accounting is repeated on each member run for grouping, and must not be summed repeatedly. verify.json, drift.json, development-notes.md and final-tests.* retain separate checks.

Cleanup, outside the repo (the complete list also appears in summary.json):

- `/Users/masshj/Library/Caches/lmz-m4-remeasure/final-tests-conditions.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/final-tests-repeat-conditions.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/streams.bin`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/prep.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/prep-conditions.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase2-conditions.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase3-G1-15-conditions.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/harness`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/ref.bin`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase3-G20-1000-conditions.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase3b-frequency-conditions.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase3-drift-conditions.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase2-synthetic-conditions.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/synthetic/streams.bin`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/synthetic/ref.bin`
- `/tmp/lmz-m4-harness` (initial development build)
