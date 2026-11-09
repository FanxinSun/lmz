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


## Phase 4 — 2026-11-09, D2 remaining kernels, CPU contention and clock attribution

This section appends new evidence to the accepted Phase 0–3b record. The original report prefix and old runs are preserved. The shared prep cache had been deleted after the prior campaign; committed campaign.py prep regenerated it from the same local Qwen3 tensors. Again, all 28,584 shared streams round-trip through the unchanged ordinary native decoder. No model downloads, installs, sudo, git commands or shipped-source edits were used. Baseline for this round is 148 tests: 142 passed and six custom Skip exceptions reported as failures by pytest.

The new harness compiles the shipped Metal source unchanged at runtime; stride modes append wrapper-only source generated by build_phase4_wrappers.py. Both the perstream and fused wrapper stream bodies are extracted verbatim. The only stream-body change is replacing gid initialization by a grid-stride loop. Perstream table construction remains inside the stream body; fused shared-LUT construction stays before the loop. There is no threadgroup_barrier in either stream body, only simdgroup_barrier. Source/body hashes are in wrapper-phase4-audit.json.

### 1. Ordinary per-chunk-table streams

All 28,584 × 32,768 B exponent streams are independently encoded by native lmz_rans_encode, each with its own 516-byte header, and every stream is verified by untouched lmz_rans_decode. Plain bytes: 936,640,512; coded bytes including all stream headers: 319,218,502; padded payload: 319,417,248; expansion 2.934167. The source tensors/partial final tensor are the same as Phase 1 and listed in summary-phase4.json. The fixture's unused shared-header slot is zero; each actual ordinary payload starts with R1. Alignment padding is a harness artifact, not additional codec metadata.


Machine/conditions: Mac mini Apple M4, 10 GPU cores, 24 GiB unified; macOS 15.4.1 (24E263), AC power, Parallels VM running and untouched, 2026-11-09. Kernel GPU timestamps, 2 warmups; 7 timed runs per current full-grid arm and each CPU-contention row, 5 per sweep G and each old-3b row. Per-campaign idle, thermal, power and SHA records plus per-row foreign GPU accounting and IOReport residency are retained in runs-phase4.jsonl/summary-phase4.json. Foreign percentages shown are selected-attempt values.

| Arm | Median GPU ms (min–max) | Plain GB/s | TG threads | Compiled TG bytes | Foreign % |
|---|---:|---:|---:|---:|---:|
| Shipped full grid, G=7146 | 204.646 (200.976–208.816) | 4.5769 | 32 | 20992 | 27.08 |
| Stride full grid, G=7146 | 206.843 (205.553–209.133) | 4.5283 | 32 | 20992 | 24.16 |

Drift = -1.062%, within 3%; sweep permitted. Shipped geometry is exactly 32 threads/TG, four streams/TG, ceil(28,584/4)=7,146 TGs. The compiled 20,992 B agrees with 16,384 B symbols + 4,096 B per-group start/frequency tables + 512 B staging.


Machine/conditions: Mac mini Apple M4, 10 GPU cores, 24 GiB unified; macOS 15.4.1 (24E263), AC power, Parallels VM running and untouched, 2026-11-09. Kernel GPU timestamps, 2 warmups; 7 timed runs per current full-grid arm and each CPU-contention row, 5 per sweep G and each old-3b row. Per-campaign idle, thermal, power and SHA records plus per-row foreign GPU accounting and IOReport residency are retained in runs-phase4.jsonl/summary-phase4.json. Foreign percentages shown are selected-attempt values.

| G | GPU ms | Plain GB/s (min–max) | GB/s per launched TG | k_ns† | k_avg† cycles/B | k_attributed† cycles/B | Foreign % |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6542.400 | 0.1432 (0.1379–0.1445) | 0.143165 | 223.52 | 352.46 | 352.71 | 4.77 |
| 2 | 3239.906 | 0.2891 (0.2888–0.2892) | 0.144547 | 221.38 | 349.23 | 349.34 | 1.66 |
| 3 | 2184.304 | 0.4288 (0.4286–0.4297) | 0.142935 | 223.88 | 353.22 | 353.28 | 3.81 |
| 4 | 1620.693 | 0.5779 (0.5767–0.5785) | 0.144482 | 221.48 | 349.26 | 349.50 | 2.40 |
| 5 | 1296.894 | 0.7222 (0.7212–0.7236) | 0.144444 | 221.54 | 349.28 | 349.59 | 2.96 |
| 6 | 1081.466 | 0.8661 (0.8625–0.8667) | 0.144347 | 221.69 | 349.45 | 349.82 | 2.96 |
| 8 | 810.897 | 1.1551 (1.1508–1.1557) | 0.144383 | 221.63 | 349.01 | 349.74 | 2.94 |
| 10 | 652.184 | 1.4362 (1.4348–1.4365) | 0.143616 | 222.82 | 351.42 | 351.60 | 3.90 |
| 12 | 546.764 | 1.7131 (1.7006–1.7147) | 0.142755 | 224.16 | 353.43 | 353.72 | 6.17 |
| 15 | 434.324 | 2.1565 (2.1350–2.1595) | 0.143770 | 222.58 | 349.97 | 351.23 | 2.00 |
| 20 | 326.909 | 2.8651 (2.8384–2.8673) | 0.143257 | 223.38 | 351.70 | 352.49 | 2.25 |
| 25 | 262.045 | 3.5744 (3.5465–3.5788) | 0.142974 | 223.82 | 351.60 | 353.18 | 2.08 |
| 30 | 219.755 | 4.2622 (4.1502–4.2751) | 0.142074 | 225.24 | 352.53 | 355.42 | 2.58 |
| 40 | 198.674 | 4.7144 (4.2302–5.6760) | 0.117861 | — | — | — | 8.63 |
| 50 | 159.293 | 5.8800 (4.9461–6.8058) | 0.117599 | — | — | — | 8.24 |
| 60 | 142.831 | 6.5577 (5.0753–7.5595) | 0.109295 | — | — | — | 9.37 |
| 80 | 112.465 | 8.3283 (5.6134–8.9815) | 0.104104 | — | — | — | 9.92 |
| 100 | 88.722 | 10.5570 (7.0366–11.8897) | 0.105570 | — | — | — | 9.82 |
| 120 | 95.604 | 9.7971 (7.5183–10.3483) | 0.081642 | — | — | — | 16.61 |
| 160 | 95.570 | 9.8006 (7.1569–11.6663) | 0.061254 | — | — | — | 34.32 |
| 200 | 115.391 | 8.1171 (7.9483–9.8069) | 0.040586 | — | — | — | 38.63 |
| 300 | 99.988 | 9.3676 (6.6496–9.7543) | 0.031225 | — | — | — | 35.67 |
| 500 | 122.456 | 7.6488 (6.2726–9.6129) | 0.015298 | — | — | — | 32.00 |
| 1000 | 180.043 | 5.2023 (5.1524–5.5063) | 0.005202 | — | — | — | 17.10 |

†Only low-grid rows G≤30 interpret G as resident TGs: resident_lanes=32G, k_ns=32G/rate_GBps, k_avg=k_ns×f_avg_active_MHz/1000, k_attributed=k_ns×kernel_attributed_MHz/1000. G=[1, 2, 3, 4, 5, 6, 8, 10] fits a through-origin slope of 0.144036 GB/s/TG, max relative residual 0.764%. Across G=[1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30], k_ns=221.38–225.24, average-active k=349.01–353.43, and **kernel-attributed k=349.34–355.42 lane-cycles per decoded exponent byte**. This is the per-chunk Metal evidence missing from cost_model; no shipped cost-model edit is made here.

The higher-grid curve is not a simple saturation plateau. It accelerates beyond G=30 and reaches about 10.56 GB/s around G=100, then falls toward the full-grid rate near 4.5 GB/s. The supplied separate probe reported three 20,992-B TGs per core, but **this decoder sweep does not independently validate a 30-TG residency ceiling**. At the low-grid slope, 10.56 GB/s corresponds to roughly 73 useful TG-equivalents, inconsistent with naively capping active TGs at 30 without another performance effect. Scheduling, occupancy-sensitive latency hiding, cache/table construction and compiler behavior are not separated by this campaign. The wrapper passed the required full-grid drift and all byte checks; a full-grid pass alone does not prove compiler equivalence at every low-grid occupancy. No single full-grid fit is substituted for the measured low-grid interval, and no unsupported pool size or residency is published as fact.

### 4. Clock attribution, including the old Phase-3b rows

The rule requested in D2 is applied to every new row and re-read from every old Phase-3b attempt: use P15 MHz iff P15 residency ≥ 0.95 × summed timed kernel GPU seconds; otherwise use measured f_avg_active. The condition, ratio and chosen clock are printed below. P15 is 1578 MHz from IORegistry pmgr voltage-states9. The P-state table is non-monotonic in index order and is kept indexed as supplied. gpufreq.py remains unchanged. A residency-weighted average includes lower-state/background time between dispatches and can under-read the kernel clock; the P15 rule is an attribution criterion, not a hardware per-kernel timestamp or proof that foreign work contributed zero P15 time. Both estimates are retained. No clock telemetry exists for pre-3b runs, so none is assigned retroactively.


Machine/conditions: Mac mini Apple M4, 10 GPU cores, 24 GiB unified; macOS 15.4.1 (24E263), AC power, Parallels VM running and untouched, 2026-11-09. Kernel GPU timestamps, 2 warmups; 7 timed runs per current full-grid arm and each CPU-contention row, 5 per sweep G and each old-3b row. Per-campaign idle, thermal, power and SHA records plus per-row foreign GPU accounting and IOReport residency are retained in runs-phase4.jsonl/summary-phase4.json. Foreign percentages shown are selected-attempt values.

| Old 3b G | f_avg active MHz | P15 s / kernel s | P15 ≥95%? | Attributed MHz | k_avg‡ | k_attributed‡ |
|---:|---:|---:|---|---:|---:|---:|
| 1 | 1574.76 | 1.0008 | yes | 1578.00 | 268.87 | 269.42 |
| 2 | 1574.76 | 1.0028 | yes | 1578.00 | 269.87 | 270.43 |
| 5 | 1571.94 | 1.0060 | yes | 1578.00 | 271.79 | 272.83 |
| 10 | 1558.16 | 1.0119 | yes | 1578.00 | 273.06 | 276.54 |
| 20 | 1562.20 | 1.0035 | yes | 1578.00 | 284.87 | 287.75 |
| 30 | 1537.79 | 1.0408 | yes | 1578.00 | 289.62 | 297.19 |
| 40 | 1509.17 | 1.0144 | yes | 1578.00 | 299.49 | 313.15 |
| 80 | 1540.22 | 1.0640 | yes | 1578.00 | 304.18 | 311.64 |
| 200 | 1517.04 | 1.0280 | yes | 1578.00 | 334.64 | 348.08 |
| 1787 | 1549.21 | 1.0354 | yes | 1578.00 | 312.28 | 318.08 |

‡Old shared rows use R=min(G,40) TGs and 128 lanes/TG; above the first bend those are apparent costs conditional on the prior 40-TG inference, not additional compute-bound measurements. All attempts, not only selected retries, are in clock-attribution-phase4-old3b.json. The manager’s approximately 269 cycles/B at G=1 and 313 at G=40 are confirmed: 269.42 and 313.15. At G=40 the average-active estimate is only 299.49. Thus the previous f_avg-active ceiling cannot simply be read as a kernel-clock-attributed ceiling. This correction does not modify the shipped shared cost record.


Machine/conditions: Mac mini Apple M4, 10 GPU cores, 24 GiB unified; macOS 15.4.1 (24E263), AC power, Parallels VM running and untouched, 2026-11-09. Kernel GPU timestamps, 2 warmups; 7 timed runs per current full-grid arm and each CPU-contention row, 5 per sweep G and each old-3b row. Per-campaign idle, thermal, power and SHA records plus per-row foreign GPU accounting and IOReport residency are retained in runs-phase4.jsonl/summary-phase4.json. Foreign percentages shown are selected-attempt values.

| New tag / G | f_avg active MHz | P15 s / kernel s | P15 ≥95%? | Attributed MHz | k_avg§ | k_attributed§ |
|---|---:|---:|---|---:|---:|---:|
| perstream-shipped / 7146 | 1576.02 | 1.0123 | yes | 1578.00 | R×11.0190 | R×11.0328 |
| perstream-drift / 7146 | 1575.97 | 1.0142 | yes | 1578.00 | R×11.1369 | R×11.1513 |
| perstream-sweep / 1 | 1576.88 | 0.9993 | yes | 1578.00 | 352.46 | 352.71 |
| perstream-sweep / 2 | 1577.49 | 1.0003 | yes | 1578.00 | 349.23 | 349.34 |
| perstream-sweep / 3 | 1577.72 | 1.0016 | yes | 1578.00 | 353.22 | 353.28 |
| perstream-sweep / 4 | 1576.90 | 0.9990 | yes | 1578.00 | 349.26 | 349.50 |
| perstream-sweep / 5 | 1576.60 | 1.0004 | yes | 1578.00 | 349.28 | 349.59 |
| perstream-sweep / 6 | 1576.30 | 1.0014 | yes | 1578.00 | 349.45 | 349.82 |
| perstream-sweep / 8 | 1574.71 | 1.0006 | yes | 1578.00 | 349.01 | 349.74 |
| perstream-sweep / 10 | 1577.17 | 1.0063 | yes | 1578.00 | 351.42 | 351.60 |
| perstream-sweep / 12 | 1576.66 | 1.0121 | yes | 1578.00 | 353.43 | 353.72 |
| perstream-sweep / 15 | 1572.37 | 0.9978 | yes | 1578.00 | 349.97 | 351.23 |
| perstream-sweep / 20 | 1574.48 | 1.0015 | yes | 1578.00 | 351.70 | 352.49 |
| perstream-sweep / 25 | 1570.94 | 0.9989 | yes | 1578.00 | 351.60 | 353.18 |
| perstream-sweep / 30 | 1565.16 | 0.9909 | yes | 1578.00 | 352.53 | 355.42 |
| perstream-sweep / 40 | 1575.37 | 1.0175 | yes | 1578.00 | R×10.6930 | R×10.7109 |
| perstream-sweep / 50 | 1561.22 | 0.9938 | yes | 1578.00 | R×8.4965 | R×8.5878 |
| perstream-sweep / 60 | 1560.82 | 1.0078 | yes | 1578.00 | R×7.6164 | R×7.7003 |
| perstream-sweep / 80 | 1568.55 | 1.0191 | yes | 1578.00 | R×6.0269 | R×6.0632 |
| perstream-sweep / 100 | 1556.47 | 1.0003 | yes | 1578.00 | R×4.7179 | R×4.7832 |
| perstream-sweep / 120 | 1560.48 | 1.0031 | yes | 1578.00 | R×5.0970 | R×5.1542 |
| perstream-sweep / 160 | 1574.26 | 1.0220 | yes | 1578.00 | R×5.1401 | R×5.1523 |
| perstream-sweep / 200 | 1566.93 | 1.0034 | yes | 1578.00 | R×6.1773 | R×6.2209 |
| perstream-sweep / 300 | 1573.92 | 1.0251 | yes | 1578.00 | R×5.3766 | R×5.3905 |
| perstream-sweep / 500 | 1557.27 | 0.9862 | yes | 1578.00 | R×6.5151 | R×6.6018 |
| perstream-sweep / 1000 | 1569.98 | 0.9957 | yes | 1578.00 | R×9.6571 | R×9.7064 |
| contention-N0 / 40 | 1568.79 | 1.0715 | yes | 1578.00 | 309.85 | 311.67 |
| contention-N0 / 1787 | 1544.48 | 1.0382 | yes | 1578.00 | 312.17 | 318.95 |
| contention-N2 / 40 | 1546.98 | 1.0516 | yes | 1578.00 | 368.24 | 375.63 |
| contention-N2 / 1787 | 1544.22 | 1.0303 | yes | 1578.00 | 376.82 | 385.06 |
| contention-N4 / 40 | 1514.29 | 1.0016 | yes | 1578.00 | 368.37 | 383.87 |
| contention-N4 / 1787 | 1532.32 | 1.0266 | yes | 1578.00 | 380.38 | 391.72 |
| contention-N8 / 40 | 1497.61 | 0.9627 | yes | 1578.00 | 410.18 | 432.20 |
| contention-N8 / 1787 | 1547.86 | 1.2619 | yes | 1578.00 | 429.66 | 438.03 |
| contention-N10 / 40 | 1503.36 | 1.2481 | yes | 1578.00 | 413.64 | 434.18 |
| contention-N10 / 1787 | 1517.56 | 1.1391 | yes | 1578.00 | 430.05 | 447.17 |
| contention-N0-end / 40 | 1547.45 | 1.0279 | yes | 1578.00 | 306.72 | 312.78 |
| contention-N0-end / 1787 | 1567.19 | 1.0775 | yes | 1578.00 | 316.07 | 318.26 |
| fused-shipped / 1787 | 1526.33 | 1.0191 | yes | 1578.00 | 398.97 | 412.48 |
| fused-drift / 1787 | 1541.18 | 1.0496 | yes | 1578.00 | 409.94 | 419.73 |
| fused-sweep / 1 | 1574.18 | 1.0020 | yes | 1578.00 | 306.94 | 307.68 |
| fused-sweep / 2 | 1572.64 | 1.0011 | yes | 1578.00 | 307.90 | 308.95 |
| fused-sweep / 5 | 1565.46 | 1.0008 | yes | 1578.00 | 314.45 | 316.97 |
| fused-sweep / 10 | 1557.94 | 1.0090 | yes | 1578.00 | 325.30 | 329.49 |
| fused-sweep / 20 | 1539.33 | 0.9988 | yes | 1578.00 | 347.88 | 356.62 |
| fused-sweep / 30 | 1507.54 | 1.0029 | yes | 1578.00 | 361.44 | 378.33 |
| fused-sweep / 40 | 1532.31 | 1.0094 | yes | 1578.00 | 395.37 | 407.16 |
| fused-sweep / 60 | 1529.17 | 0.9970 | yes | 1578.00 | 439.91 | 453.96 |
| fused-sweep / 80 | 1177.77 | 1.0117 | yes | 1578.00 | 311.17 | 416.91 |
| fused-sweep / 120 | 1452.66 | 4.1225 | yes | 1578.00 | 418.59 | 454.70 |
| fused-sweep / 200 | 1534.22 | 1.0132 | yes | 1578.00 | 418.54 | 430.48 |
| fused-sweep / 500 | 1508.32 | 0.9860 | yes | 1578.00 | 391.26 | 409.33 |

§Unit is lane-cycles per exponent-plane decoded byte. For shared/fused apparent costs R=min(G,40) is an explicit curve-derived assumption; saturated or contended rows do not isolate compute cost. Fused per-BF16-output-byte cost would be half these numbers. For perstream G≤30 use R=G; above it R is left symbolic, because actual resident TG count is unresolved, and each multiplier is the measured lane-cycle coefficient per resident TG. New raw records retain the separate launched-lane diagnostic too; it is not a valid cost when launched TGs are not all resident. All state durations and both clock inputs are in each row; group-level accounting repeated on member dispatches must not be summed repeatedly.

### 3. Shared decoder under measured CPU memory contention

G=40 uses the previously validated shared stride wrapper so it decodes all 28,584 streams, as does G=1787. Dispatching the original non-stride kernel at G=40 would decode only 640 streams and would not be the same fixed-traffic comparison. The N=0 controls, N=2/4/8/10 campaigns and final N=0 control use the same wrapper and data. Each worker owns two 256,000,000-byte uint8 NumPy arrays (512 MB/thread); repeated np.copyto releases the GIL and streams through arrays far larger than cache. Arrays are allocated/touched before streaming begins, outside measurements; workers stop and their arrays are released after each N campaign. The VM is untouched.

Each worker rate is actual complete copy bytes divided by measured elapsed seconds inside those np.copyto calls, using only calls wholly within the row window. CPU traffic=2×copy rate, labeled read+write. Partial boundary copies are excluded, never prorated. The separately retained copied-bytes/window rate is a conservative window-count lower bound. Complete-copy counts, copy-call durations, per-worker intervals and both rate definitions are in raw rows. GPU kernels are timed by GPU timestamps; poisoning and comparisons are outside those timestamps. Both CPU-active-call rates and GPU-only rates omit host gaps, so their sum is a rate sanity check, not a DRAM-counter measurement. The same-wall-window GPU and counted-copy rates are also retained for comparison.


Machine/conditions: Mac mini Apple M4, 10 GPU cores, 24 GiB unified; macOS 15.4.1 (24E263), AC power, Parallels VM running and untouched, 2026-11-09. Kernel GPU timestamps, 2 warmups; 7 timed runs per current full-grid arm and each CPU-contention row, 5 per sweep G and each old-3b row. Per-campaign idle, thermal, power and SHA records plus per-row foreign GPU accounting and IOReport residency are retained in runs-phase4.jsonl/summary-phase4.json. Foreign percentages shown are selected-attempt values.

| N / control | G | Decoder GB/s | CPU copy GB/s | CPU read+write GB/s | GPU+CPU logical GB/s | f_avg / attributed MHz | Foreign % |
|---|---:|---:|---:|---:|---:|---:|---:|
| contention-N0 | 40 | 25.9227 | 0.000 | 0.000 | 34.635 | 1568.8 / 1578.0 | 3.72 |
| contention-N0 | 1787 | 25.3312 | 0.000 | 0.000 | 33.845 | 1544.5 / 1578.0 | 21.42 |
| contention-N2 | 40 | 21.5090 | 35.314 | 70.627 | 99.365 | 1547.0 / 1578.0 | 8.29 |
| contention-N2 | 1787 | 20.9821 | 35.630 | 71.260 | 99.294 | 1544.2 / 1578.0 | 22.26 |
| contention-N4 | 40 | 21.0472 | 37.566 | 75.132 | 103.253 | 1514.3 / 1578.0 | 6.84 |
| contention-N4 | 1787 | 20.6253 | 37.375 | 74.750 | 102.308 | 1532.3 / 1578.0 | 19.85 |
| contention-N8 | 40 | 18.6937 | 40.887 | 81.775 | 106.752 | 1497.6 / 1578.0 | 9.22 |
| contention-N8 | 1787 | 18.4449 | 39.860 | 79.720 | 104.364 | 1547.9 / 1578.0 | 36.17 |
| contention-N10 | 40 | 18.6083 | 39.323 | 78.646 | 103.508 | 1503.4 / 1578.0 | 28.13 |
| contention-N10 | 1787 | 18.0677 | 40.397 | 80.793 | 104.933 | 1517.6 / 1578.0 | 43.03 |
| contention-N0-end | 40 | 25.8311 | 0.000 | 0.000 | 34.513 | 1547.4 / 1578.0 | 7.71 |
| contention-N0-end | 1787 | 25.3864 | 0.000 | 0.000 | 33.919 | 1567.2 / 1578.0 | 16.43 |

Machine/conditions: Mac mini Apple M4, 10 GPU cores, 24 GiB unified; macOS 15.4.1 (24E263), AC power, Parallels VM running and untouched, 2026-11-09. Kernel GPU timestamps, 2 warmups; 7 timed runs per current full-grid arm and each CPU-contention row, 5 per sweep G and each old-3b row. Per-campaign idle, thermal, power and SHA records plus per-row foreign GPU accounting and IOReport residency are retained in runs-phase4.jsonl/summary-phase4.json. Foreign percentages shown are selected-attempt values.

| N / control | G | Per-worker measured copy GB/s, worker order 0..N−1 | Full copies counted per worker |
|---|---:|---|---|
| contention-N0 | 40 | none | none |
| contention-N0 | 1787 | none | none |
| contention-N2 | 40 | 17.658, 17.656 | 40, 39 |
| contention-N2 | 1787 | 17.773, 17.857 | 40, 39 |
| contention-N4 | 40 | 9.390, 9.445, 9.349, 9.382 | 22, 22, 22, 22 |
| contention-N4 | 1787 | 9.445, 9.293, 9.299, 9.338 | 22, 23, 22, 23 |
| contention-N8 | 40 | 5.569, 5.238, 5.271, 4.825, 5.117, 5.010, 5.080, 4.777 | 17, 16, 16, 14, 15, 16, 15, 14 |
| contention-N8 | 1787 | 4.855, 5.174, 4.650, 5.123, 5.294, 4.566, 5.220, 4.979 | 14, 15, 13, 15, 15, 13, 15, 15 |
| contention-N10 | 40 | 3.252, 3.970, 4.157, 4.177, 3.957, 4.139, 4.081, 3.956, 3.413, 4.221 | 11, 12, 14, 13, 13, 13, 13, 12, 11, 13 |
| contention-N10 | 1787 | 4.324, 4.130, 4.289, 3.795, 3.254, 4.078, 4.027, 4.553, 4.274, 3.673 | 13, 12, 13, 12, 9, 12, 12, 13, 13, 11 |
| contention-N0-end | 40 | none | none |
| contention-N0-end | 1787 | none | none |

Sanity: shared traffic factor = 1+1/2.975300 = 1.336101. For every attempt, GPU kernel median rate × traffic factor + measured CPU read/write rate ≤120 GB/s; maximum observed sum is 106.752 GB/s. No rejected-over-120 attempt is hidden. This counts logical coded-read/plain-write plus CPU read/write bytes, not physical DRAM transactions/cache/write-allocation overhead.

G=40: N=10 changes decode from 25.9227 to 18.6083 GB/s (-28.22%). Ending N=0 gives 25.8311 GB/s, drift -0.353%.

G=1787: N=10 changes decode from 25.3312 to 18.0677 GB/s (-28.67%). Ending N=0 gives 25.3864 GB/s, drift +0.218%.

CPU contention degrades decode on this unified-memory host under the recorded background. It does not identify bandwidth as the sole cause: CPU scheduling, shared power/clock and host submission gaps also change, and some attempts fail the P15 coverage threshold and use f_avg-active instead. All per-row foreign-work flags remain visible. vm_stat snapshots are retained before and after each N campaign. Global counters show no new pageouts, swapouts or compressions, but swapins (20–5,238 pages per campaign) and decompressions (92–14,134 pages) occurred. These are shared-machine counters and cannot be assigned solely to the harness. This is a decoder/copy campaign, not an end-to-end model load or inference benchmark.

### 2. Fused real BF16

Reference is 1,873,281,024 original BF16 bytes read directly from the same safetensors tensors, not reconstructed by the tested merge. The independently checked exponent plane matches ref.bin; raw sign/mantissa is ((word>>8)&0x80)|(word&0x7f), 936,640,512 bytes. Shipped geometry is 128 threads/TG and ceil(28,584/16)=1,787 TGs. Buffers are allocated and populated once before warmups. The source entrypoint expects coded streams at buffer 0, offsets 1, smplane 2, BF16 output 3, shared header 4 and Params 5. The current Objective-C host C ABI does not expose a fused decode function; this harness dispatches the shipped Metal entrypoint directly.


Machine/conditions: Mac mini Apple M4, 10 GPU cores, 24 GiB unified; macOS 15.4.1 (24E263), AC power, Parallels VM running and untouched, 2026-11-09. Kernel GPU timestamps, 2 warmups; 7 timed runs per current full-grid arm and each CPU-contention row, 5 per sweep G and each old-3b row. Per-campaign idle, thermal, power and SHA records plus per-row foreign GPU accounting and IOReport residency are retained in runs-phase4.jsonl/summary-phase4.json. Foreign percentages shown are selected-attempt values.

| Fused arm / G | GPU ms (min–max) | BF16-output GB/s | Exponent-plane GB/s | Compiled TG bytes | Foreign % |
|---|---:|---:|---:|---:|---:|
| fused-shipped / 1787 | 47.818 (47.696–54.874) | 39.1749 | 19.5875 | 19456 | 25.19 |
| fused-drift / 1787 | 48.660 (47.466–49.826) | 38.4975 | 19.2487 | 19456 | 24.67 |
| fused-sweep / 1 | 1426.785 (1423.651–1436.326) | 1.3129 | 0.6565 | 19456 | 2.64 |
| fused-sweep / 2 | 716.341 (715.829–721.514) | 2.6151 | 1.3075 | 19456 | 1.97 |
| fused-sweep / 5 | 293.973 (293.396–298.116) | 6.3723 | 3.1861 | 19456 | 2.32 |
| fused-sweep / 10 | 152.792 (152.301–157.238) | 12.2604 | 6.1302 | 19456 | 3.02 |
| fused-sweep / 20 | 82.687 (82.338–86.117) | 22.6552 | 11.3276 | 19456 | 4.58 |
| fused-sweep / 30 | 58.480 (58.402–64.928) | 32.0329 | 16.0165 | 19456 | 6.89 |
| fused-sweep / 40 | 47.202 (47.054–50.002) | 39.6862 | 19.8431 | 19456 | 4.27 |
| fused-sweep / 60 | 52.628 (41.048–58.746) | 35.5950 | 17.7975 | 19456 | 12.04 |
| fused-sweep / 80 | 48.332 (46.617–49.691) | 38.7586 | 19.3793 | 19456 | 96.26 |
| fused-sweep / 120 | 52.714 (49.281–57.436) | 35.5368 | 17.7684 | 19456 | 341.64 |
| fused-sweep / 200 | 49.906 (46.722–52.285) | 37.5362 | 18.7681 | 19456 | 22.47 |
| fused-sweep / 500 | 47.454 (46.595–51.549) | 39.4759 | 19.7380 | 19456 | 30.05 |

Full-grid wrapper drift -1.729%; within 3%: True. Both numerators use the same measured GPU time: BF16_bytes=1,873,281,024 and exponent_bytes=936,640,512. BF16-output GB/s is exactly twice exponent-plane GB/s. This removes the old numerator ambiguity; the historical synthetic 36.3 figure is not represented as real BF16 evidence. Original BF16 byte comparisons run on first and last timed dispatch of every attempt/G, and pass.

### Conditions, validation, reproduction and cleanup

Each run holds the shared m4-gpu mkdir lock; each lock period is bounded below 20 minutes, and no timed GPU process overlaps another harness. Start idle samples are recorded and rechecked according to the brief. AGXDeviceUserClient before/after counters exclude the harness PID. Above 2% foreign summed GPU time each row is repeated once and both attempts remain append-only; reports select the required retry, not the fastest. Some rows still exceed 2% after retry. In particular, fused G=120 retry has foreign time over three times the timed kernel sum; its rate and P15 attribution are heavily contaminated and retained as flagged evidence, not a clean result. Such rows and the resulting broad cost intervals are conditional on those background conditions, not clean uncontended constants. The clock interval includes measured device-wide state residency; the P15 attribution rule is retained even when background may contribute P15 time.

Final exact pytest baseline: **6 failed, 142 passed, 1 warning in 84.26s (0:01:24)**. Matches requested 142 passed / 6 skip-as-fail: True. All attempts are retained in tests-phase4-attempt*.txt and tests-phase4.json; no test handling or shipped-source fixes.

New raw log: runs-phase4.jsonl (778 rows, 204 warmups, 574 timed dispatches); original runs.jsonl is preserved. Summary: summary-phase4.json. New host harness: harness_phase4.swift, built by swiftc -O against Foundation/Metal; needs installed Apple developer tools. New wrappers: wrapper_phase4.metal, generated by build_phase4_wrappers.py from shipped source. New prep: prep_phase4.py. Orchestration: measure_phase4.py build/perstream/contention/fused. Clock re-read: attribute_phase4.py. Analysis/render: analyze_phase4.py and render_phase4.py. Reuse the original locked campaign.py prep first. All source paths are relative to the lmz root; use the existing Python environment /Users/masshj/system-env/customize-llm/venv/bin/python. No dependency installs are needed.

Tracked-file validation reads the Git index directly without invoking git. Initial tracked-phase4-initial.json records the original report prefix length/hash; final validation permits only this requested report append and requires the prefix hash to match. Shipped sources remain byte-identical to the index. Cached data remain available for review; no cleanup is performed.

New/recreated external cache artifacts for user cleanup:

- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase4-fused-G1-40-conditions.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase4-contention-N8-conditions.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase4-perstream-G1-40-conditions.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/smplane.bin`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase4-contention-N2-memory-after.txt`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/streams.bin`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase4-contention-N0-conditions.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/prep.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/perstream-prep.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase4-contention-N0-end-memory-after.txt`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/fused-prep.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/prep-conditions.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase4-contention-N8-memory-after.txt`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase4-perstream-prep-conditions.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase4-fused-prep-conditions.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase4-contention-N0-memory-after.txt`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase4-perstream-G50-1000-conditions.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase4-fused-G60-500-conditions.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/harness-phase4`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase4-contention-N4-conditions.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/ref.bin`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase4-contention-N4-memory-after.txt`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/bf16.bin`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/perstream.bin`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase4-contention-N0-end-conditions.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase4-contention-N10-memory-after.txt`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase4-perstream-drift-conditions.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase4-contention-N10-conditions.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase4-final-tests-1-conditions.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase4-contention-N2-conditions.json`
- `/Users/masshj/Library/Caches/lmz-m4-remeasure/phase4-fused-drift-conditions.json`


## Phase 5 — 2026-11-09, the shipped Metal host path, and what it cost to keep it honest

Executed by a codex session under the lmsluice manager's brief E; audited by the manager against the raw rows
(`scratchpad/gpu/metal/m4_ship/`). The Vulkan decoder is a separate record, [vulkan-decoder-m4.md](vulkan-decoder-m4.md).

**What changed in shipped code** (`lmz/gpu/lmz_rans.metal`, `lmzgpu_metal.m`, `__init__.py`, `build.py`, tests):

- `lmz_rans.metal` is the HEAD source byte-for-byte plus one new kernel, `lmz_decode_perstream_stride`: HEAD's per-stream body
  in a grid-stride loop whose bound is rounded up to four streams so the SIMD-group barriers and ballots stay in lockstep.
- A sized host ABI (`lmz_gpu_metal_decode`, `_decode_bf16`, `_prepare`/`_run_prepared`, `_decode_batch_dev_sized`) that wraps whole
  aligned pages of the caller's memory instead of copying, validates every offset, length and shared header, and returns either a view
  of the caller's output or a Metal-owned buffer whose lifetime follows the Python view.
- `decode_bf16_fused` (the fused BF16 kernel, previously unreachable) and an opt-in `decode_chunk` fast path (`LMZ_METAL_FUSED_CHUNK=1`).
- The library is now built with **ARC** (`-fobjc-arc`, plus `-framework IOKit`). **HEAD was built without it**, so every
  `newBufferWith*` was never released (each decode leaked its output buffer, ~936 MB on this workload) and `__bridge_retained`
  handles were plain casts that dangled once an autorelease pool drained (codex observed a crash). This is a pre-existing bug, not a regression.
- The pre-existing test `test_mapped_archive_zero_copy_and_mtl_buffer` failed instead of skipping under `LMZ_NO_GPU=1`; it skips now.

**Decisions, in the order they were forced by measurement.**

1. *A bounds-checked kernel was built first* (refills guarded by the stream's own end, which removes the trailing-padding
   requirement). Same-session against HEAD it was **0.90-0.94x** on every case (shared 25.17 -> 22.93 GB/s, per-chunk 0.90x, fused 0.94x).
2. *A branchless clamp* recovered some of it (0.931-0.958x) and still failed the pre-set bound of 3% (shared full grid 0.941x; k at G=1 270 -> 288).
   On a latency-bound dependent chain every extra operation on the address path shows up.
3. *Final: the HEAD kernels unchanged, with the slack guaranteed by the host.* A coded buffer is wrapped without a copy only if
   `page_round_up(extent) - extent >= lmz_gpu_pad_bytes()` (576 B); otherwise it is copied into a zero-padded allocation. The trust model is the CUDA
   one (below). The published Metal `k` records therefore stay valid: the kernels are the ones that were measured.
4. *Per-chunk grid tuning was made explicit.* The first version tuned inside the first `decode_batch` of 1,024+ per-chunk streams, which costs ~20
   batch decodes (4.6-4.9 s on this workload) for a grid that is 2.6x faster than the shipped one: it repays only after ~30 batches of this size, so
   a one-shot caller would have decoded 13x slower. It now runs only when asked (`prepare_metal_batch`, or `LMZ_METAL_PERSTREAM_TGS=auto`).
5. *The fused `decode_chunk` fast path stays off by default* (table below).

**Trust model.** The kernels may read up to `lmz_gpu_pad_bytes()` past the end of a stream, as the CUDA kernels do, so the coded input needs
that much readable slack, which the host provides. A *corrupt* archive can make a kernel read past its buffer; the call then fails
(command-buffer error, the call returns None), a caller-owned output may be partly written, and nothing is written outside the output.
A test places the final coded byte against a `PROT_NONE` page (both the no-copy case with slack and the exact-boundary case that must take the copy path) and checks
that a physically short stream neither crashes nor poisons the next decode. **Untrusted archives should not be fed to the GPU path unverified.**

### Kernel-only: HEAD against the final kernels

Machine/conditions: Mac mini Apple M4 (10 GPU cores), 24 GiB unified, macOS 15.4.1, AC power, Parallels VM running and untouched, 2026-11-09; real Qwen3-4B-Instruct-2507 exponent planes, 28,584 streams x 32,768 B (936.6 MB plain); decimal GB/s; GPU lock and idle checks; desktop GPU clients present in most rows, every row over 2% foreign time retried once and the second attempt kept (both attempts in the logs). Kernel GPU timestamps, 14 timed runs (two attempts) per row.

| Kernel | HEAD GB/s | Final GB/s | Ratio |
|---|---:|---:|---:|
| shared table, full grid (1,787 TGs) | 25.219 | 25.190 | 0.999 |
| shared table, stride G=1 | 0.750 | 0.750 | 1.000 |
| per-chunk tables, shipped geometry (7,146 TGs) | 4.422 | 4.413 | 0.998 |
| fused BF16 (BF16 bytes out), full grid | 39.276 | 39.270 | 1.000 |

### Through the Python API

Machine/conditions: Mac mini Apple M4 (10 GPU cores), 24 GiB unified, macOS 15.4.1, AC power, Parallels VM running and untouched, 2026-11-09; real Qwen3-4B-Instruct-2507 exponent planes, 28,584 streams x 32,768 B (936.6 MB plain); decimal GB/s; GPU lock and idle checks; desktop GPU clients present in most rows, every row over 2% foreign time retried once and the second attempt kept (both attempts in the logs). API wall time, seven timed calls after two warm-ups. HEAD is the unmodified library (built from `git show HEAD`, with its original flags), each timed sample in
a fresh short-lived process. The same HEAD, called repeatedly inside one process, measured 1.98 GB/s in Phase 2; codex attributes the difference to the leak above, which was not isolated.

| Path | API GB/s (min-max) | Kernel inside the call |
|---|---:|---:|
| HEAD, shared table | 4.68 | 22.8 |
| HEAD, per-chunk tables | 2.47 | 4.3 |
| new, shared table, page-aligned no-copy input, output read back every call | 15.04 (12.6-15.8) | 22.9 |
| new, shared table, prepared, output read back every call | 22.62 (21.1-23.7) | 23.0 |
| new, shared table, prepared, output left on the GPU* | 24.97 (22.0-25.1) | 25.2 |
| new, fused BF16, prepared, output read back (BF16 bytes) | 18.35 (17.8-18.8) | 36.4 |
| new, fused BF16, prepared, output left on the GPU* | 38.91 (17.8-39.1) | 39.2 |
| new, per-chunk, **plain `decode_batch`** (shipped geometry, nothing tuned) | 4.0-4.1 | 4.4 |
| new, per-chunk, `prepare_metal_batch` (tuned: 80 TGs in codex's runs, 160 in the manager's) | 10.3-11.4 (5.1-12.1) | 8.8-8.9 |

\* Intermediate outputs are not read by the CPU; the first and last timed outputs are compared byte-for-byte. Which condition applies depends on the consumer: a GPU consumer
(model weights that stay in unified memory) is the second, a CPU consumer the first. Both are kept. The per-chunk tuned rows are noisy (the low end is rows with foreign GPU work).
One-time costs are separate: `prepare_metal_batch` on the per-chunk batch took 4.6-4.9 s; these are not cold model-load numbers.

### The fused kernel inside `decode_chunk`

Machine/conditions: Mac mini Apple M4 (10 GPU cores), 24 GiB unified, macOS 15.4.1, AC power, Parallels VM running and untouched, 2026-11-09; real Qwen3-4B-Instruct-2507 exponent planes, 28,584 streams x 32,768 B (936.6 MB plain); decimal GB/s; GPU lock and idle checks; desktop GPU clients present in most rows, every row over 2% foreign time retried once and the second attempt kept (both attempts in the logs). Real `CODEC_BF16` chunks, byte-identical, seven timed API runs after two warm-ups, BF16 bytes per second.

| Chunk (BF16 bytes) | CPU native decode + merge | previous GPU two-plane path | opt-in fused path |
|---:|---:|---:|---:|
| 65,536 | 2.152 | 0.035 | 0.044 |
| 1,048,576 | 1.407 | 0.084 | 0.105 |
| 4,194,304 | 1.322 | 0.087 | 0.104 |

Every chunk of an ordinary archive has its own table, so the fused kernel gets one stream per dispatch (8 lanes); it is 13-48x slower than the CPU (13x at 1 and 4 MiB, 48x at 64 KiB). It is useful to
a caller holding many streams against **one** table (39 GB/s above), not to today's archives, hence off by default.

### Tests

- pytest (`tests/`): **151 passed, 6 "failed"**, the 6 being lmz's own `Skip` exceptions that pytest counts as failures (the CUDA-only device test and five FUSE tests needing
  `fusermount3`); the baseline was 142 passed with the same six. The nine extra passes are four Metal tests (partial SIMD tails at 10 stream counts under forced grids of 1/3/7/full, the
  guard-page and lifetime test, fused and opt-in chunk identity, explicit tuning) and five Vulkan tests.
- `LMZ_NO_GPU=1 python tests/test_lmz.py`: 124/144 passed, 20 skipped, no failures, stable over four consecutive runs from a quiet start. **One existing test is environment-flaky:** `test_mount_readahead_serves_the_same_bytes` (macOS loopback WebDAV mount). Run soon after another mount session (the pytest `fs_`/mount tests, or the full pytest run) it turns from a pass into a skip, `mount_webdav failed (19)` (123 passed, 21 skipped, reproduced on demand); after the full pytest run it was once short a test altogether (123 passed + 20 skipped = 143 of 144), which was not captured in a log. It touches `mount_macos.py`, not anything changed here, and it did not occur in the earlier baseline.
- On all 28,584 real streams the final host path and stride kernel decode byte-identically at forced grids of 1, 3, 7 and the full grid.

### Not established here

An uncontended rate (desktop GPU clients were present throughout); a cold end-to-end model load; any device other than this M4; that one process-cached tuned grid suits a later,
different workload; the CPU-contention cost of the *new* host path (Phase 4's contention numbers are for the kernel); behaviour of the corrupt-archive path beyond "does not crash or hang and the next decode works".

Reproduction: `scratchpad/gpu/metal/m4_ship/` (drivers, raw `*.jsonl`, `summary.json` with the cache cleanup list); `default_perstream.py` there regenerates the plain-vs-tuned rows.
