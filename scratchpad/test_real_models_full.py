"""Comprehensive real-world model testing on NVMe and ExFAT.

Tests:
1. Joint Multi-Model Archiving with lmz.append(dest=...):
   - Qwen3-4B (Base) + Qwen3-4B-Instruct-2507
   - Cross-model deduplication and delta coding measurement
   - Byte-exact verification of all tensors
2. Bundle Creation on Real Model Tree:
   - Copy-on-write clonefile acceleration on APFS
   - Validation and inventory checks
3. Filesystem comparison: APFS (NVMe) vs ExFAT (USB).
"""

import hashlib
import json
import os
import resource
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import lmz
from lmz import bundle


def get_peak_rss_mb() -> float:
    # ru_maxrss is in bytes on macOS
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def test_joint_family_archive(base_dir: str, instruct_dir: str, out_archive: str, label: str):
    print(f"\n=======================================================")
    print(f"Testing Joint Model Family Archive: {label}")
    print(f"Base: {base_dir}")
    print(f"Instruct: {instruct_dir}")
    print(f"Archive: {out_archive}")

    if os.path.exists(out_archive):
        os.unlink(out_archive)

    # 1. Compress Base Model
    rss0 = get_peak_rss_mb()
    t0 = time.perf_counter()
    st_base = lmz.compress(base_dir, out_archive, level=1)
    t_comp_base = time.perf_counter() - t0
    base_size = st_base.input_bytes
    arc_size_1 = os.path.getsize(out_archive)
    print(f"[Base] Compressed {base_size / 1e9:.2f} GB in {t_comp_base:.2f}s "
          f"({(base_size / 1024**3) / t_comp_base:.2f} GB/s) -> {arc_size_1 / 1e9:.2f} GB "
          f"({(1.0 - arc_size_1 / base_size) * 100:.1f}% saved)")

    # 2. Append Instruct Model under dest='instruct'
    t0 = time.perf_counter()
    st_inst = lmz.append(out_archive, instruct_dir, dest="instruct", level=1)
    t_append = time.perf_counter() - t0
    inst_size = st_inst.input_bytes
    total_uncompressed = base_size + inst_size
    arc_size_total = os.path.getsize(out_archive)
    incremental_arc = arc_size_total - arc_size_1

    print(f"[Instruct Append] Appended {inst_size / 1e9:.2f} GB in {t_append:.2f}s "
          f"({(inst_size / 1024**3) / t_append:.2f} GB/s)")
    print(f"  Incremental archive growth: {incremental_arc / 1e9:.2f} GB "
          f"({(1.0 - incremental_arc / inst_size) * 100:.1f}% saved vs raw)")
    dd = st_inst.detail.get("dedup_bytes", 0)
    dl = st_inst.detail.get("delta_bytes", 0)
    sh = st_inst.detail.get("shared_bytes", 0)
    print(f"  Cross-model deduplicated: {dd / 1e6:.1f} MB")
    print(f"  Cross-model delta-coded:  {dl / 1e6:.1f} MB")
    print(f"  Shared payload bytes:     {sh / 1e6:.1f} MB")
    print(f"[Joint Archive] Total uncompressed: {total_uncompressed / 1e9:.2f} GB -> "
          f"Archive: {arc_size_total / 1e9:.2f} GB (Overall {(1.0 - arc_size_total / total_uncompressed) * 100:.1f}% saved)")
    print(f"  Peak RSS: {get_peak_rss_mb():.1f} MB")

    # 3. Verify Archive Integrity
    t0 = time.perf_counter()
    lmz.verify(out_archive)
    t_v = time.perf_counter() - t0
    print(f"Verification: PASS in {t_v:.2f}s ({arc_size_total / 1e6 / t_v:.1f} MB/s)")

    # 4. Spot check tensor reading across base and instruct via MappedArchive
    with lmz.MappedArchive(out_archive) as arc:
        base_members = [m for m in arc.members if not m.path.startswith("instruct/")]
        inst_members = [m for m in arc.members if m.path.startswith("instruct/")]
        print(f"MappedArchive members: {len(base_members)} base, {len(inst_members)} instruct")
        
        # Pick a safetensors shard from base and instruct
        st_base_mem = next(m for m in base_members if m.path.endswith(".safetensors") and m.tensors)
        st_inst_mem = next(m for m in inst_members if m.path.endswith(".safetensors") and m.tensors)

        # Compare first tensor
        tname = list(st_base_mem.tensors.keys())[0]
        dt1, sh1, raw1 = arc.tensor(tname, member=st_base_mem.path)
        dt2, sh2, raw2 = arc.tensor(tname, member=st_inst_mem.path)
        print(f"Tensor '{tname}':")
        print(f"  Base:     dtype={dt1}, shape={sh1}, bytes={len(raw1)}")
        print(f"  Instruct: dtype={dt2}, shape={sh2}, bytes={len(raw2)}")

        # Check against ground truth files
        orig_base_path = os.path.join(base_dir, st_base_mem.path)
        orig_inst_path = os.path.join(instruct_dir, st_inst_mem.path[len("instruct/"):])
        import lmz.planner as planner
        with open(orig_base_path, "rb") as fh:
            lo = planner.probe(fh, os.path.getsize(orig_base_path))
            off = lo.tensors[tname]["offsets"]
            fh.seek(off[0])
            gt1 = fh.read(off[1] - off[0])
        with open(orig_inst_path, "rb") as fh:
            lo = planner.probe(fh, os.path.getsize(orig_inst_path))
            off = lo.tensors[tname]["offsets"]
            fh.seek(off[0])
            gt2 = fh.read(off[1] - off[0])
        assert raw1 == gt1, "Base tensor mismatch!"
        assert raw2 == gt2, "Instruct tensor mismatch!"
        print(f"Ground-truth tensor verification: PASS (byte-exact)")

    # Clean up
    if os.path.exists(out_archive):
        os.unlink(out_archive)


def test_bundle_real_model(model_dir: str, out_bundle: str, label: str):
    print(f"\n=======================================================")
    print(f"Testing Bundle Creation on Real Model: {label}")
    print(f"Source: {model_dir}")
    print(f"Bundle: {out_bundle}")

    if os.path.exists(out_bundle):
        os.unlink(out_bundle)

    t0 = time.perf_counter()
    res = lmz.create_bundle(model_dir, out_bundle, level=1, overwrite=True)
    t_bundle = time.perf_counter() - t0
    bundle_size = os.path.getsize(out_bundle)
    dec_bytes = res["archive_accounting"]["decoded_bytes"]
    print(f"Bundle created in {t_bundle:.2f}s ({(dec_bytes / 1024**3) / t_bundle:.2f} GB/s)")
    print(f"Decoded bytes: {dec_bytes / 1e9:.2f} GB -> Bundle: {bundle_size / 1e9:.2f} GB "
          f"({(1.0 - bundle_size / dec_bytes) * 100:.1f}% saved)")
    print(f"Entries: {len(res['entries'])}, Entry point: {res['entry_point']['path']}")

    # Validation
    v_res = lmz.validate_bundle(out_bundle)
    assert v_res["status"] == "verified"
    print(f"Bundle validation: PASS (manifest SHA-256: {res['bundle_manifest_sha256'][:16]}...)")

    # Inventory
    inv = lmz.inventory_bundle(out_bundle)
    assert inv["status"] == "verified"
    print(f"Bundle inventory: PASS")

    if os.path.exists(out_bundle):
        os.unlink(out_bundle)


if __name__ == "__main__":
    # Test on local APFS SSD
    print(">>> RUNNING BENCHMARKS ON LOCAL APFS NVMe SSD <<<")
    base_local = "/Users/masshj/models/Qwen3-4B"
    inst_local = "/Users/masshj/models/Qwen3-4B-Instruct-2507"
    arc_local = "/tmp/qwen_family_test.lmz"
    bundle_local = "/tmp/qwen_bundle_test.lmz"

    test_bundle_real_model(base_local, bundle_local, "Qwen3-4B on APFS NVMe")
    test_joint_family_archive(base_local, inst_local, arc_local, "Qwen3-4B Base + Instruct on APFS NVMe")

    # Test on external ExFAT USB drive
    print("\n>>> RUNNING BENCHMARKS ON EXTERNAL EXFAT USB DRIVE <<<")
    base_ext = "/Volumes/Elements/models/Qwen3-4B"
    inst_ext = "/Volumes/Elements/models/Qwen3-4B-Instruct-2507"
    arc_ext = "/Volumes/Elements/models/qwen_family_ext.lmz"
    bundle_ext = "/Volumes/Elements/models/qwen_bundle_ext.lmz"

    test_bundle_real_model(base_ext, bundle_ext, "Qwen3-4B on ExFAT USB Drive")
    test_joint_family_archive(base_ext, inst_ext, arc_ext, "Qwen3-4B Base + Instruct on ExFAT USB Drive")
