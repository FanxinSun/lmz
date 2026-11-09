"""Vulkan rANS benchmark on Apple M4 through MoltenVK with GPU timestamps and IOReport clock sampling."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import struct
import sys
import time

# Ensure imports
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gpufreq
from lmz.gpu import vulkan

# Exclusive GPU access for timed runs is the caller's job (a measurement campaign serialises its own runs).



def main():
    print("=" * 80)
    print("Vulkan through MoltenVK on an Apple M4: a translation layer over Metal,")
    print("not a native Vulkan driver.")
    print("=" * 80)

    # 1. Load data
    cache_dir = Path.home() / "Library/Caches/lmz-vulkan"
    streams_file = cache_dir / "streams.bin"
    ref_file = cache_dir / "ref.bin"
    prep_file = cache_dir / "prep.json"

    assert streams_file.is_file(), f"Missing {streams_file}"
    assert ref_file.is_file(), f"Missing {ref_file}"
    prep_meta = json.loads(prep_file.read_text())
    expected_sha256 = prep_meta["plain_sha256"]

    with open(streams_file, "rb") as f:
        nstr, plane = struct.unpack("<II", f.read(8))
        hdr = f.read(516)
        offsets = f.read(nstr * 16)
        streams = f.read()

    plain_bytes = nstr * plane
    print(f"Workload: {nstr} streams x {plane} bytes ({plain_bytes / 1e6:.1f} MB plain)")
    print(f"Expected SHA-256: {expected_sha256}")

    # 2. Context & Device
    ctx = vulkan.get_context()
    props = vulkan.VkPhysicalDeviceProperties()
    ctx.lib.vkGetPhysicalDeviceProperties(ctx.phys_dev, vulkan.ctypes.byref(props))

    sub_props = vulkan.VkPhysicalDeviceSubgroupProperties(
        sType=vulkan.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
        pNext=None
    )
    props2 = vulkan.VkPhysicalDeviceProperties2(
        sType=vulkan.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        pNext=vulkan.ctypes.cast(vulkan.ctypes.byref(sub_props), vulkan.ctypes.c_void_p)
    )
    ctx.lib.vkGetPhysicalDeviceProperties2(ctx.phys_dev, vulkan.ctypes.byref(props2))

    device_name = props.deviceName.decode("utf-8", "replace").strip("\x00")
    print(f"Device: {device_name}")
    print(f"Subgroup Size: {sub_props.subgroupSize}, Shared Memory: {props.limits.maxComputeSharedMemorySize} B")
    print("-" * 80)

    results = {
        "label": "Vulkan through MoltenVK on an Apple M4: a translation layer over Metal, not a native Vulkan driver",
        "device": device_name,
        "limits": {
            "subgroup_size": sub_props.subgroupSize,
            "max_shared_memory_bytes": props.limits.maxComputeSharedMemorySize,
            "timestamp_period_ns": props.limits.timestampPeriod,
        },
        "workload": {
            "nstr": nstr,
            "plane": plane,
            "plain_bytes": plain_bytes,
            "expected_sha256": expected_sha256,
        },
        "full_grid": {},
        "stride_sweep": [],
    }

    # 3. Full-Grid Benchmark (Direct Kernel)
    print(">>> Running Full-Grid Benchmark (Direct Kernel, 128 workgroups)...")
    # 2 warmups
    for _ in range(2):
        vulkan.decode_batch(streams, offsets, nstr, plane, hdr)

    runs = 7
    full_timings = []
    # Sample IOReport clock across sustained runs
    clock_a = gpufreq.snapshot()
    for r in range(runs):
        out = vulkan.decode_batch(streams, offsets, nstr, plane, hdr)
        t = vulkan.last_timing()
        assert t is not None
        if r == 0 or r == runs - 1:
            h = hashlib.sha256(out).hexdigest()
            assert h == expected_sha256, f"Run {r} corrupted output!"
        sec = t["gpu_seconds"]
        gbps = (plain_bytes / sec) / 1e9
        full_timings.append({"run": r, "gpu_seconds": sec, "gbps": gbps, "ticks": t["gpu_ticks"]})
    clock_b = gpufreq.snapshot()
    clock_delta = gpufreq.delta(clock_a, clock_b)

    rates = sorted([x["gbps"] for x in full_timings])
    med_gbps = rates[len(rates) // 2]
    f_avg = clock_delta.get("f_avg_active_mhz")
    f_max = clock_delta.get("f_max_mhz")

    print(f"Full-Grid Median Rate: {med_gbps:.2f} GB/s (over {runs} runs)")
    print(f"GPU Clock during window: avg {f_avg:.1f} MHz, max {f_max} MHz")
    print(f"Individual runs (GB/s): {[round(x['gbps'], 2) for x in full_timings]}")
    results["full_grid"] = {
        "median_gbps": med_gbps,
        "runs": full_timings,
        "clock": clock_delta,
    }
    print("-" * 80)

    # 4. Grid-Stride Sweep
    # Candidates: G = 1, 2, 3, 5, 10, 20, 30, 40, 80, 128 (full)
    full_g = (nstr + 15) // 16
    grid_candidates = [1, 2, 3, 5, 10, 20, 30, 40, 80, full_g]
    grid_list = sorted(list(set(grid_candidates)))

    print(">>> Running Grid-Stride Sweep Benchmark...")
    print(f"{'G':>4} | {'Resident Lanes':>14} | {'Rate (GB/s)':>12} | {'k_ns (ns/B/lane)':>18} | {'Lane-Cycles':>12} | {'Clock (MHz)':>12}")
    print("-" * 80)

    for g in grid_list:
        # Warmup
        vulkan.decode_batch(streams, offsets, nstr, plane, hdr, grid=g)

        # Timed runs with clock sampling
        n_repeats = 5
        g_timings = []
        ca = gpufreq.snapshot()
        for r in range(n_repeats):
            out = vulkan.decode_batch(streams, offsets, nstr, plane, hdr, grid=g)
            t = vulkan.last_timing()
            assert t is not None
            if r == 0 or r == n_repeats - 1:
                assert hashlib.sha256(out).hexdigest() == expected_sha256
            sec = t["gpu_seconds"]
            gbps = (plain_bytes / sec) / 1e9
            g_timings.append({"run": r, "gpu_seconds": sec, "gbps": gbps})
        cb = gpufreq.snapshot()
        cdelta = gpufreq.delta(ca, cb)

        g_rates = sorted([x["gbps"] for x in g_timings])
        med_rate = g_rates[len(g_rates) // 2]
        resident_lanes = g * 128
        k_ns = (resident_lanes / med_rate) if med_rate > 0 else 0.0

        # Clock & lane-cycles
        row_f_avg = cdelta.get("f_avg_active_mhz")
        if row_f_avg is None:
            row_f_avg = f_avg or 1578.0
        lane_cycles = k_ns * row_f_avg / 1000.0

        print(f"{g:4d} | {resident_lanes:14d} | {med_rate:12.2f} | {k_ns:18.2f} | {lane_cycles:12.1f} | {row_f_avg:12.1f}")

        results["stride_sweep"].append({
            "G": g,
            "resident_lanes": resident_lanes,
            "median_gbps": med_rate,
            "k_ns": k_ns,
            "lane_cycles": lane_cycles,
            "f_avg_mhz": row_f_avg,
            "runs": g_timings,
            "clock": cdelta,
        })

    out_file = HERE / "vulkan_measurements.json"
    out_file.write_text(json.dumps(results, indent=2))
    print("=" * 80)
    print(f"Measurements saved to {out_file.name}")
    print("=" * 80)


if __name__ == "__main__":
    main()
