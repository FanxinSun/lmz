#!/usr/bin/env python3
"""Standalone Vulkan rANS sweep benchmark.

Imports only vulkan.py from the current directory, using standard library only.
Loads kit-workload.bin, verifies correctness against the embedded expected SHA-256,
and executes full-grid and grid-stride benchmarks, emitting JSON results.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import struct
import sys
import time
from pathlib import Path

# Add current directory to path so vulkan can be imported
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import vulkan

def main():
    print("=" * 70)
    print("  lmz Standalone Vulkan rANS Benchmark Kit")
    print("=" * 70)

    # 1. Probe & Query Device Limits
    ok, dev_str = vulkan.probe()
    if not ok:
        print(f"Error: Vulkan compute not available: {dev_str}")
        sys.exit(1)

    print(f"Device: {dev_str}")
    ctx = vulkan.get_context()
    props = vulkan.VkPhysicalDeviceProperties()
    ctx.lib.vkGetPhysicalDeviceProperties(ctx.phys_dev, vulkan.ctypes.byref(props))

    # Query subgroup properties
    sub_props = vulkan.VkPhysicalDeviceSubgroupProperties(
        sType=vulkan.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES,
        pNext=None
    )
    props2 = vulkan.VkPhysicalDeviceProperties2(
        sType=vulkan.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        pNext=vulkan.ctypes.cast(vulkan.ctypes.byref(sub_props), vulkan.ctypes.c_void_p)
    )
    ctx.lib.vkGetPhysicalDeviceProperties2(ctx.phys_dev, vulkan.ctypes.byref(props2))

    limits = props.limits
    print(f"  Driver Version:               {props.driverVersion}")
    print(f"  Vulkan API Version:           {props.apiVersion >> 22}.{(props.apiVersion >> 12) & 0x3ff}.{props.apiVersion & 0xfff}")
    print(f"  Subgroup Size:                {sub_props.subgroupSize}")
    print(f"  Max Workgroup Invocations:    {limits.maxComputeWorkGroupInvocations}")
    print(f"  Max Shared Memory Size (B):   {limits.maxComputeSharedMemorySize}")
    print(f"  Timestamp Period (ns):        {limits.timestampPeriod}")
    print("-" * 70)

    # 2. Load kit-workload.bin
    wk_file = HERE / "kit-workload.bin"
    if not wk_file.is_file():
        print(f"Error: Workload file not found at {wk_file}")
        sys.exit(1)

    print(f"Loading workload from {wk_file.name} ({wk_file.stat().st_size / 1e6:.1f} MB)...")
    with open(wk_file, "rb") as f:
        magic = f.read(8)
        if magic != b"LMZVKWK\0":
            print(f"Error: Invalid workload magic: {magic}")
            sys.exit(1)
        expected_sha256 = f.read(64).decode("ascii")
        nstr, plane = struct.unpack("<II", f.read(8))
        hdr = f.read(516)
        offsets = f.read(nstr * 16)
        streams = f.read()

    plain_bytes = nstr * plane
    print(f"  Streams (nstr):               {nstr}")
    print(f"  Plane size:                   {plane} B")
    print(f"  Total plain bytes:            {plain_bytes} B ({plain_bytes / 1e6:.1f} MB)")
    print(f"  Expected plain SHA-256:       {expected_sha256}")
    print("-" * 70)

    # 3. Correctness Verification
    print("Verifying correctness against expected SHA-256...")
    out = vulkan.decode_batch(streams, offsets, nstr, plane, hdr)
    if out is None:
        print(f"Error: decode_batch failed: {vulkan.last_error()}")
        sys.exit(1)

    got_hash = hashlib.sha256(out).hexdigest()
    if got_hash != expected_sha256:
        print(f"FAIL: Hash mismatch!\n  Expected: {expected_sha256}\n  Got:      {got_hash}")
        sys.exit(1)
    print(f"PASS: Correctness verified! Plain output SHA-256 matches exactly.")
    print("-" * 70)

    # 4. Benchmarking
    print("Benchmarking...")
    results = {
        "timestamp": time.time(),
        "platform": platform.platform(),
        "device": dev_str,
        "limits": {
            "subgroup_size": sub_props.subgroupSize,
            "max_workgroup_invocations": limits.maxComputeWorkGroupInvocations,
            "max_shared_memory_bytes": limits.maxComputeSharedMemorySize,
            "timestamp_period_ns": limits.timestampPeriod,
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

    # Full Grid Benchmark
    print("1. Full Grid Benchmark (Direct Kernel):")
    for _ in range(2):
        vulkan.decode_batch(streams, offsets, nstr, plane, hdr)

    full_timings = []
    runs = 7
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
        print(f"  Run {r + 1}/{runs}: {sec * 1e3:7.3f} ms  ->  {gbps:6.2f} GB/s")

    gbps_vals = [x["gbps"] for x in full_timings]
    gbps_vals.sort()
    median_gbps = gbps_vals[len(gbps_vals) // 2]
    print(f"  => Median Full Grid Rate: {median_gbps:.2f} GB/s")
    results["full_grid"] = {
        "runs": full_timings,
        "median_gbps": median_gbps,
    }
    print("-" * 70)

    # Grid Stride Sweep
    print("2. Grid-Stride Sweep Benchmark (Stride Kernel):")
    grid_candidates = [1, 2, 3, 5, 10, 20, 30, 40, 80, 200]
    full_groups = (nstr + 15) // 16
    grid_list = [g for g in grid_candidates if g < full_groups] + [full_groups]

    for g in grid_list:
        vulkan.decode_batch(streams, offsets, nstr, plane, hdr, grid=g)
        g_timings = []
        n_repeats = 5
        for r in range(n_repeats):
            out = vulkan.decode_batch(streams, offsets, nstr, plane, hdr, grid=g)
            t = vulkan.last_timing()
            assert t is not None
            if r == 0 or r == n_repeats - 1:
                assert hashlib.sha256(out).hexdigest() == expected_sha256
            sec = t["gpu_seconds"]
            gbps = (plain_bytes / sec) / 1e9
            g_timings.append({"run": r, "gpu_seconds": sec, "gbps": gbps})

        rates = [x["gbps"] for x in g_timings]
        rates.sort()
        med_rate = rates[len(rates) // 2]
        resident_lanes = g * 128
        k_ns = (resident_lanes / med_rate) if med_rate > 0 else 0.0

        print(f"  G = {g:3d}: Median Rate = {med_rate:6.2f} GB/s  |  k_ns = {k_ns:7.2f} ns/B/lane")
        results["stride_sweep"].append({
            "G": g,
            "resident_lanes": resident_lanes,
            "median_gbps": med_rate,
            "k_ns": k_ns,
            "runs": g_timings,
        })

    # Save JSON output
    out_json = HERE / "vulkan_sweep_results.json"
    out_json.write_text(json.dumps(results, indent=2))
    print("=" * 70)
    print(f"Benchmark complete! Results written to {out_json.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
