"""Vulkan rANS compute decoder correctness tests."""
import hashlib
from pathlib import Path
import random
import shutil
import struct
import time
import unittest
import zlib

from lmz import kernels
from lmz.gpu import _SHAPES, vulkan

class TestVulkanDecoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        if os.environ.get("LMZ_NO_GPU") == "1":
            raise unittest.SkipTest("LMZ_NO_GPU set")

        ok, msg = vulkan.probe()
        if not ok:
            raise unittest.SkipTest(f"Vulkan compute unavailable: {msg}")

    def test_01_probe_and_device(self):
        ok, msg = vulkan.probe()
        self.assertTrue(ok, f"probe() failed: {msg}")
        self.assertTrue(vulkan.available())
        dev_name = vulkan.device_name()
        self.assertIsNotNone(dev_name)
        self.assertTrue(dev_name.startswith("vulkan:"))

    def _run_shape_batch(self, table: bytes, plane: int, nstr: int, seed: int):
        rnd = random.Random(seed)
        bufs = []
        for _ in range(nstr):
            buf = bytes(rnd.randbytes(plane)).translate(table)
            bufs.append(buf)

        # Build shared table for this batch
        all_bytes = b"".join(bufs)
        hist = kernels.histogram(all_bytes)
        freqs = kernels.rans_freqs(hist)
        hdr = kernels.rans_shared_header(freqs)

        streams = bytearray()
        offsets = bytearray()
        off = 0
        for b in bufs:
            coded = kernels.rans_encode_shared(b, hdr)
            self.assertIsNotNone(coded, "CPU shared encode returned None")
            offsets += struct.pack("<QQ", off, len(coded))
            streams += coded
            # Pad to 16 bytes
            pad = (-len(coded)) % 16
            if pad:
                streams += bytes(pad)
            off += len(coded) + pad

        out = vulkan.decode_batch(bytes(streams), bytes(offsets), nstr, plane, hdr)
        self.assertIsNotNone(out, f"decode_batch failed: {vulkan.last_error()}")
        self.assertEqual(len(out), nstr * plane)
        self.assertEqual(bytes(out), all_bytes, f"Mismatch for shape, plane={plane}, nstr={nstr}")

    def test_02_synthetic_shapes_plane_4096(self):
        nstr_list = [1, 2, 3, 5, 8, 9, 17, 129, 1023, 1025]
        for shape_name, table in _SHAPES.items():
            for nstr in nstr_list:
                seed = zlib.crc32(f"{shape_name}:4096:{nstr}".encode()) & 0xFFFF   # stable across processes
                with self.subTest(shape=shape_name, plane=4096, nstr=nstr):
                    self._run_shape_batch(table, 4096, nstr, seed)

    def test_03_synthetic_shapes_plane_32768(self):
        nstr_list = [1, 2, 3, 5, 8, 9, 17, 129, 1023, 1025]
        for shape_name, table in _SHAPES.items():
            for nstr in nstr_list:
                seed = zlib.crc32(f"{shape_name}:32768:{nstr}".encode()) & 0xFFFF
                with self.subTest(shape=shape_name, plane=32768, nstr=nstr):
                    self._run_shape_batch(table, 32768, nstr, seed)

    def test_05_declines_a_batch_over_the_storage_buffer_limit(self):
        """The shader indexes storage buffers with 32 bits and a descriptor range may not
        exceed maxStorageBufferRange: a batch over the device's limit must be declined
        with a reason, never decoded with wrapped offsets."""
        rnd = random.Random(7)
        plain = bytes(rnd.randbytes(4096)).translate(_SHAPES["skewed"])
        hdr = kernels.rans_shared_header(kernels.rans_freqs(kernels.histogram(plain)))
        coded = kernels.rans_encode_shared(plain, hdr)
        offsets = struct.pack("<QQ", 0, len(coded))
        ctx = vulkan.get_context(None)
        saved = ctx.max_buffer_bytes
        try:
            out = vulkan.decode_batch(coded, offsets, 1, 4096, hdr)
            self.assertEqual(bytes(out), plain, "the unrestricted decode must still work")
            ctx.max_buffer_bytes = 1024                  # pretend the device allows only 1 KiB
            self.assertIsNone(vulkan.decode_batch(coded, offsets, 1, 4096, hdr))
            self.assertIn("maxStorageBufferRange", vulkan.last_error())
        finally:
            ctx.max_buffer_bytes = saved
        self.assertEqual(bytes(vulkan.decode_batch(coded, offsets, 1, 4096, hdr)), plain)

    def test_04_real_qwen3_exponent_plane(self):
        cache_dir = Path.home() / "Library/Caches/lmz-vulkan"
        streams_file = cache_dir / "streams.bin"
        ref_file = cache_dir / "ref.bin"
        if not streams_file.is_file() or not ref_file.is_file():
            self.skipTest(f"Real data not prepared at {cache_dir}")

        with open(streams_file, "rb") as f:
            nstr, plane = struct.unpack("<II", f.read(8))
            hdr = f.read(516)
            offsets = f.read(nstr * 16)
            streams = f.read()

        with open(ref_file, "rb") as f:
            expected_plain = f.read()

        self.assertGreaterEqual(nstr, 2048)
        self.assertEqual(plane, 32768)
        self.assertEqual(len(expected_plain), nstr * plane)

        # 1. Direct kernel decode
        out = vulkan.decode_batch(streams, offsets, nstr, plane, hdr)
        self.assertIsNotNone(out, f"Direct decode failed: {vulkan.last_error()}")
        self.assertEqual(bytes(out), expected_plain, "Direct decoded bytes do not match expected plain ref!")

        got_hash = hashlib.sha256(bytes(out)).hexdigest()
        want_hash = hashlib.sha256(expected_plain).hexdigest()
        self.assertEqual(got_hash, want_hash)
        print(f"\nReal Qwen3 Direct Match ({nstr} streams, {plane} B, SHA-256 {got_hash[:16]}...) Timing: {vulkan.last_timing()}")

        # 2. Grid-stride kernel decode (various grid sizes)
        for g in [1, 2, 5, 10, 40]:
            with self.subTest(grid=g):
                out_stride = vulkan.decode_batch(streams, offsets, nstr, plane, hdr, grid=g)
                self.assertIsNotNone(out_stride, f"Stride (grid={g}) failed: {vulkan.last_error()}")
                self.assertEqual(bytes(out_stride), expected_plain, f"Stride (grid={g}) mismatch!")
                print(f"Real Qwen3 Stride G={g} Match! Timing: {vulkan.last_timing()}")


if __name__ == "__main__":
    unittest.main()
