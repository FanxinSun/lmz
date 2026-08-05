"""Test suite for lmz.

Runs standalone (`python3 tests/test_lmz.py`) and under pytest. Every
round-trip check compares bytes, not just sizes: the whole point of the tool
is that decompression reproduces the input exactly.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lmz  # noqa: E402
from lmz import codec, entropy, kernels, planner  # noqa: E402
from lmz import format as lmzformat  # noqa: E402
from lmz.format import ArchiveReader, FormatError  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLI = [sys.executable, os.path.join(ROOT, "lmz-cli")]


def digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rand(n: int, seed: int = 0) -> bytes:
    """Deterministic pseudo-random bytes without requiring numpy."""
    out = bytearray()
    x = seed | 1
    while len(out) < n:
        x = (x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        out += x.to_bytes(8, "little")
    return bytes(out[:n])


def weights_bf16(nelem: int, seed: int = 1) -> bytes:
    """BF16 words whose exponents cluster the way real weights do."""
    out = bytearray()
    x = seed | 1
    for _ in range(nelem):
        x = (x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        mant = (x >> 11) & 0x7F
        exp = 118 + ((x >> 40) % 6)  # a narrow band, as in a trained model
        sign = (x >> 63) & 1
        out += ((sign << 15) | (exp << 7) | mant).to_bytes(2, "little")
    return bytes(out)


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


def weights_bf16_cond(nelem: int, seed: int = 1) -> bytes:
    """BF16 whose mantissa alphabet narrows as the exponent grows.

    Real weights carry a small exponent-mantissa dependence; this exaggerates
    it so the conditioned codec must win and therefore must be chosen.
    """
    out = bytearray()
    x = seed | 1
    for _ in range(nelem):
        x = (x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        exp = 118 + ((x >> 40) % 6)
        mant = (x >> 11) & 0x7F
        if exp >= 121:
            mant &= 0x0F
        sign = (x >> 63) & 1
        out += ((sign << 15) | (exp << 7) | mant).to_bytes(2, "little")
    return bytes(out)


def write_torch_bin(path: str, storages: list[tuple[str, str, bytes]]) -> None:
    """A torch.save-style zip: data.pkl naming storage classes, data/<key> blobs.

    The pickle is a valid opcode stream that pushes each storage class GLOBAL
    followed by its key string -- the pattern the planner's scanner keys on --
    without needing torch to produce it.
    """
    import zipfile

    pkl = bytearray(b"\x80\x02")  # PROTO 2
    for key, cls, _raw in storages:
        pkl += b"ctorch\n" + cls.encode() + b"\n"
        k = key.encode()
        pkl += b"X" + len(k).to_bytes(4, "little") + k
        loc = b"cpu"
        pkl += b"X" + len(loc).to_bytes(4, "little") + loc
        pkl += b"0" * 3  # pop the three pushed objects
    pkl += b"N."  # NONE, STOP

    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("archive/data.pkl", bytes(pkl))
        for key, _cls, raw in storages:
            zf.writestr(f"archive/data/{key}", raw)


def q80_blocks(nblocks: int, seed: int = 1) -> bytes:
    """Synthetic Q8_0 blocks: clustered fp16 scales, gaussian-ish int8 quants.

    The scale-low byte depends on the scale-high byte and the quants use a
    narrowed alphabet, mirroring the structure of real quantised weights that
    the block codec exists to exploit.
    """
    out = bytearray()
    x = seed | 1
    for _ in range(nblocks):
        x = (x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        hi = 0x24 | ((x >> 5) & 0x3)
        lo = ((x >> 8) & 0x3F) | ((hi & 1) << 6)
        out.append(lo)
        out.append(hi)
        for _ in range(32):
            x = (x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            q = (((x >> 0) & 0xFF) + ((x >> 8) & 0xFF)
                 + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)) >> 2
            out.append(q & 0xFF)
    return bytes(out)


def write_gguf(path: str, tensors: list[tuple[str, int, list, bytes]],
               alignment: int = 32) -> None:
    """Minimal GGUF writer: magic, KVs, tensor index, aligned data."""
    def s(text: bytes) -> bytes:
        return struct.pack("<Q", len(text)) + text

    kv = s(b"general.alignment") + struct.pack("<I", 4) + struct.pack("<I", alignment)
    kv += s(b"general.name") + struct.pack("<I", 8) + s(b"test-model")
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 2) + kv

    infos, offset, blobs = b"", 0, []
    for name, ttype, dims, raw in tensors:
        infos += s(name.encode()) + struct.pack("<I", len(dims))
        infos += b"".join(struct.pack("<Q", d) for d in dims)
        infos += struct.pack("<IQ", ttype, offset)
        blobs.append(raw)
        offset += len(raw)
        offset += (-offset) % alignment

    body = head + infos
    pad = (-len(body)) % alignment
    with open(path, "wb") as fh:
        fh.write(body + b"\0" * pad)
        for raw in blobs:
            fh.write(raw)
            fh.write(b"\0" * ((-len(raw)) % alignment))


# --------------------------------------------------------------------- tests


def test_kernels_roundtrip():
    """Split must equal a transpose, and merge must undo it, at every size."""
    for esize in (1, 2, 4, 8):
        for nelem in (0, 1, 2, 3, 7, 15, 16, 17, 31, 33, 64, 1000, 4097, 70000):
            src = rand(nelem * esize, nelem + esize)
            planes = kernels.split(src, esize)
            expect = b"".join(bytes(src[k::esize]) for k in range(esize))
            assert bytes(planes) == expect, f"split esize={esize} n={nelem}"
            assert bytes(kernels.merge(planes, esize, nelem)) == src
            if nelem:
                out = bytearray(nelem * esize)
                parts = [(bytes(planes[k * nelem:(k + 1) * nelem]), 0)
                         for k in range(esize)]
                got = kernels.merge_planes(parts, esize, nelem, out)
                assert bytes(got) == src, f"merge_planes esize={esize} n={nelem}"


def test_kernels_reject_bad_input():
    for bad in (3, 5, 16, 0):
        try:
            kernels.split(b"\0" * 48, bad)
        except ValueError:
            continue
        raise AssertionError(f"element size {bad} should be rejected")
    try:
        kernels.split(b"\0" * 7, 2)
        raise AssertionError("misaligned length should be rejected")
    except ValueError:
        pass


def test_kernel_fallback_matches_native():
    """The pure-Python path must agree with the SIMD one byte for byte."""
    script = (
        "import os,sys;os.environ['LMZ_NO_NATIVE']='1';sys.path.insert(0,%r);"
        "from lmz import kernels;"
        "assert not kernels.backend().startswith('native'), kernels.backend();"
        "src=bytes((i*7+3)%%256 for i in range(4096));"
        "print(kernels.backend());"
        "print(bytes(kernels.split(src,4)).hex()[:64]);"
        "print(bytes(kernels.merge(kernels.split(src,8),8,512)).hex()[:64])" % ROOT
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    backend, split_hex, merge_hex = out.stdout.strip().splitlines()
    src = bytes((i * 7 + 3) % 256 for i in range(4096))
    assert split_hex == bytes(kernels.split(src, 4)).hex()[:64], backend
    assert merge_hex == bytes(kernels.merge(kernels.split(src, 8), 8, 512)).hex()[:64]


def test_codec_roundtrip_all_dtypes():
    cases = [
        ("bf16 weights", weights_bf16(200000), 2),
        ("random noise", rand(400000, 3), 2),
        ("fp32-ish", weights_bf16(100000) + bytes(200000), 4),
        ("all zeros", bytes(300000), 2),
        ("bytes", rand(100000, 5), 1),
        ("int64", b"".join((i % 1000).to_bytes(8, "little") for i in range(20000)), 8),
    ]
    for name, data, esize in cases:
        for level in (1, 3):
            parts, cid, flags, crc = codec.encode_chunk(data, esize, level, True)
            payload = b"".join(bytes(p) for p in parts)
            got = codec.decode_chunk(payload, cid, esize, flags, len(data), crc, True)
            assert bytes(got) == data, f"{name} level={level}"


def test_bf16_field_split_roundtrip():
    """Splitting on bfloat16 field bounds must still be exactly reversible."""
    for n in (0, 1, 2, 3, 7, 16, 17, 255, 4096, 65537):
        src = rand(n * 2, n + 3)
        planes = kernels.split_bf16(src)
        words = [src[2 * i] | (src[2 * i + 1] << 8) for i in range(n)]
        assert bytes(planes[:n]) == bytes((w >> 7) & 0xFF for w in words)
        assert bytes(planes[n:]) == bytes(((w >> 8) & 0x80) | (w & 0x7F)
                                          for w in words)
        back = kernels.merge_bf16([(bytes(planes[:n]), 0), (bytes(planes[n:]), 0)], n)
        assert bytes(back) == src, f"bf16 merge mismatch at n={n}"


def test_bf16_chunk_beats_byte_split():
    """The field split should encode real BF16 weights smaller than a byte split."""
    data = weights_bf16(400000, 7)
    field = codec.encode_chunk(data, 2, 1, False, kind=planner.KIND_BF16)
    byte = codec.encode_chunk(data, 2, 1, False, kind=planner.KIND_BYTES)
    size = lambda r: sum(len(p) for p in r[0])  # noqa: E731
    assert field[1] == lmzformat.CODEC_BF16, field[1]
    assert byte[1] == lmzformat.CODEC_SPLIT, byte[1]
    for parts, cid, flags, crc in (field, byte):
        payload = b"".join(bytes(p) for p in parts)
        got = codec.decode_chunk(payload, cid, 2, flags, len(data), crc, True)
        assert bytes(got) == data
    assert size(field) <= size(byte), (size(field), size(byte))


def test_rans_adversarial_distributions():
    """Distributions that break naive rANS implementations.

    A stream of one repeated symbol drives that symbol's frequency to the full
    probability scale. That value needs 13 bits, and it was being packed into
    a 12-bit field of the decode table, where it wrapped to zero and made the
    decoder read past the end of its input. Nothing but degenerate input shows
    it, so every shape that could reach full scale is pinned here.

    Sizes straddle the 8-way interleave boundary so both the block loop and
    the ragged tail are exercised, including the state assignment between them.
    """
    if not kernels.have_rans():
        return
    shapes = {
        "single symbol": lambda n: bytes([7]) * n,
        "two symbols": lambda n: bytes((i & 1) * 200 for i in range(n)),
        "whole alphabet": lambda n: bytes(i % 256 for i in range(n)),
        "one rare outlier": lambda n: bytes([0] * (n - 1) + [255]),
        "uniform noise": lambda n: rand(n, n + 5),
    }
    for n in (1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 255, 256, 4096, 65537):
        for name, make in shapes.items():
            data = make(n)
            enc = kernels.rans_encode(data)
            assert enc is not None, f"{name}, n={n}: encode failed"
            assert bytes(kernels.rans_decode(enc, n)) == data, f"{name}, n={n}"

    # A single-symbol stream costs zero bits per symbol, so the whole megabyte
    # must collapse to the frequency table and the flushed states. If the
    # frequency field ever overflows again this balloons instead.
    enc = kernels.rans_encode(bytes([42]) * (1 << 20))
    assert len(enc) < 1024, f"single-symbol stream took {len(enc)} bytes"


def test_rans_majority_symbol_states():
    """A >50% symbol drives encoder states past 2^31.

    The fixed-point reciprocal the encoder once used is only exact below
    2^31, so on rare states the quotient came out one high and the symbol
    landed in a neighbour's slot. Scattered minority symbols steer states
    through the risky window; contiguous runs happen not to, which is how a
    real Q8_0 norm plane (3998:98, scattered) shipped the first failure.
    """
    if not kernels.have_rans():
        return
    # Each (seed, majority freq, length) was verified to make the reciprocal
    # encoder emit a stream its own decoder rejects. Do not "simplify" the
    # generator: the failure needs specific state trajectories, and most
    # arrangements with the same histogram decode fine.
    for seed, fmaj, n in ((95, 3998, 4096), (165, 3998, 4096),
                          (1, 3900, 65536), (3, 3900, 65536)):
        x = seed | 1
        buf = bytearray([0x3E]) * n
        minority = n * (4096 - fmaj) // 4096
        placed = 0
        while placed < minority:
            x = (x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
            pos = x % n
            if buf[pos] == 0x3E:
                buf[pos] = 0x3F
                placed += 1
        data = bytes(buf)
        enc = kernels.rans_encode(data)
        assert enc is not None
        assert bytes(kernels.rans_decode(enc, n)) == data, \
            f"seed {seed}, majority {fmaj}/4096 over {n} bytes"


def test_rans_rejects_malformed():
    if not kernels.have_rans():
        return
    good = kernels.rans_encode(rand(10000, 3))
    for name, bad in (("truncated", good[:len(good) // 2]),
                      ("bad magic", b"XX" + good[2:]),
                      ("empty", b""),
                      ("header only", good[:516])):
        try:
            kernels.rans_decode(bad, 10000)
        except (ValueError, RuntimeError):
            continue
        raise AssertionError(f"malformed input ({name}) was not rejected")


def test_rans_matches_entropy():
    """The rANS coder must land near the order-0 bound, not merely below raw."""
    if not kernels.have_rans():
        return
    import math

    data = weights_bf16(500000, 11)
    planes = kernels.split_bf16(data)
    n = len(data) // 2
    for name, plane in (("exponent", bytes(planes[:n])),
                        ("sign+mantissa", bytes(planes[n:]))):
        counts = kernels.histogram(plane)
        total = sum(counts)
        h = -sum((c / total) * math.log2(c / total) for c in counts if c)
        enc = kernels.rans_encode(plane)
        assert bytes(kernels.rans_decode(enc, len(plane))) == plane
        bound = len(plane) * h / 8
        assert len(enc) < bound * 1.02, (name, len(enc), bound)


def test_codec_detects_corruption():
    data = weights_bf16(100000)
    parts, cid, flags, crc = codec.encode_chunk(data, 2, 1, True)
    payload = bytearray(b"".join(bytes(p) for p in parts))
    # Flip a byte in each plane in turn: damage to the stored mantissa plane
    # is caught by the checksum, damage to the coded exponent plane usually
    # fails inside the entropy decoder first. Both must report the same way.
    hdr = codec._plane_header(2)
    first_plane_len = hdr.unpack_from(payload, 0)[0]
    for offset in (hdr.size + 4, hdr.size + first_plane_len + 4):
        damaged = bytearray(payload)
        damaged[offset] ^= 0xFF
        try:
            codec.decode_chunk(bytes(damaged), cid, 2, flags, len(data), crc, True)
        except FormatError:
            continue
        raise AssertionError(f"corruption at offset {offset} was not detected")


def test_incompressible_data_is_stored():
    """Random data must not be inflated, and must skip the entropy coder."""
    data = rand(1 << 20, 11)
    parts, cid, flags, crc = codec.encode_chunk(data, 2, 1, True)
    total = sum(len(p) for p in parts)
    assert total <= len(data) + 64, f"grew from {len(data)} to {total}"
    assert cid == 0, f"expected stored codec for noise, got {cid}"


def test_safetensors_layout():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.safetensors")
        write_safetensors(path, [
            ("a.weight", "BF16", [64, 64], weights_bf16(4096)),
            ("b.weight", "F32", [32, 32], rand(4096, 7)),
            ("c.bias", "BF16", [128], weights_bf16(128)),
        ])
        with open(path, "rb") as fh:
            layout = planner.probe(fh, os.path.getsize(path))
        assert layout.kind == "safetensors"
        assert set(layout.tensors) == {"a.weight", "b.weight", "c.bias"}
        assert layout.tensors["a.weight"]["dtype"] == "BF16"
        sizes = {r.esize for r in layout.regions}
        assert sizes == {2, 4}, sizes
        chunks = planner.chunkify(layout, os.path.getsize(path), 1 << 20)
        assert chunks[0][0] == 0 and chunks[-1][1] == os.path.getsize(path)
        for start, end, _, _, _ in chunks:
            assert end > start
        # BF16 regions must be tagged so the codec can split on field bounds.
        kinds = {k for _, _, _, k, _ in chunks}
        assert planner.KIND_BF16 in kinds, kinds


def test_gguf_layout():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.gguf")
        write_gguf(path, [
            ("tok.weight", 1, [64, 64], rand(8192, 2)),      # F16
            ("blk.0.w", 30, [64, 64], weights_bf16(4096)),   # BF16
            ("blk.0.q", 12, [256, 4], rand(4 * 144, 4)),     # Q4_K
        ])
        with open(path, "rb") as fh:
            layout = planner.probe(fh, os.path.getsize(path))
        assert layout.kind == "gguf", layout.kind
        assert set(layout.tensors) == {"tok.weight", "blk.0.w", "blk.0.q"}
        assert layout.tensors["blk.0.w"]["dtype"] == "BF16"
        assert layout.tensors["blk.0.q"]["dtype"] == "Q4_K"


def test_unknown_format_is_opaque():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "x.bin")
        with open(path, "wb") as fh:
            fh.write(b"not a model at all" * 1000)
        with open(path, "rb") as fh:
            layout = planner.probe(fh, os.path.getsize(path))
        assert layout.kind == "raw" and layout.regions == []


def roundtrip(tmp: str, src: str, **kw) -> lmz.Stats:
    archive = os.path.join(tmp, "a.lmz")
    out = os.path.join(tmp, "out")
    before = digest(src) if os.path.isfile(src) else None
    stats = lmz.compress(src, archive, **kw)
    lmz.decompress(archive, out, overwrite=True)
    if before is not None:
        assert digest(out) == before, "decompressed file differs from the original"
    return stats


def test_single_file_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.safetensors")
        write_safetensors(path, [
            ("w1", "BF16", [512, 512], weights_bf16(512 * 512)),
            ("w2", "F32", [128, 128], rand(128 * 128 * 4, 9)),
            ("w3", "I8", [4096], rand(4096, 13)),
        ])
        stats = roundtrip(d, path, chunk_size=1 << 18)
        assert stats.ratio > 1.15, f"expected real compression, got {stats.ratio:.3f}x"


def test_roundtrip_edge_sizes():
    with tempfile.TemporaryDirectory() as d:
        for n in (0, 1, 7, 4096, 100000):
            path = os.path.join(d, f"f{n}.bin")
            with open(path, "wb") as fh:
                fh.write(rand(n, n + 1))
            archive = os.path.join(d, f"f{n}.lmz")
            out = os.path.join(d, f"o{n}.bin")
            lmz.compress(path, archive, chunk_size=1 << 16)
            lmz.decompress(archive, out, overwrite=True)
            assert digest(out) == digest(path), f"size {n}"


def test_directory_roundtrip_preserves_tree():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "model")
        os.makedirs(os.path.join(src, "nested"))
        write_safetensors(os.path.join(src, "model-00001.safetensors"),
                          [("w", "BF16", [256, 256], weights_bf16(65536))])
        write_safetensors(os.path.join(src, "model-00002.safetensors"),
                          [("v", "BF16", [256, 256], weights_bf16(65536, 5))])
        with open(os.path.join(src, "config.json"), "w") as fh:
            json.dump({"hidden": 256, "layers": 2}, fh)
        with open(os.path.join(src, "nested", "tokenizer.bin"), "wb") as fh:
            fh.write(rand(50000, 21))

        archive = os.path.join(d, "m.lmz")
        out = os.path.join(d, "restored")
        stats = lmz.compress(src, archive)
        assert stats.files == 4
        lmz.decompress(archive, out, overwrite=True)
        for root, _, files in os.walk(src):
            for name in files:
                a = os.path.join(root, name)
                b = os.path.join(out, os.path.relpath(a, src))
                assert os.path.exists(b), f"missing {b}"
                assert digest(a) == digest(b), f"differs: {name}"


def test_workers_do_not_change_output():
    """The archive must be deterministic regardless of thread count."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.safetensors")
        write_safetensors(path, [("w", "BF16", [512, 256], weights_bf16(131072))])
        digests = set()
        for workers in (1, 4, 16):
            archive = os.path.join(d, f"a{workers}.lmz")
            lmz.compress(path, archive, workers=workers, chunk_size=1 << 16)
            digests.add(digest(archive))
        assert len(digests) == 1, "archive contents depend on worker count"


def test_verify_and_info():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.safetensors")
        write_safetensors(path, [("w", "BF16", [256, 256], weights_bf16(65536))])
        archive = os.path.join(d, "a.lmz")
        lmz.compress(path, archive)
        stats = lmz.verify(archive)
        assert stats.chunks > 0
        meta = lmz.info(archive)
        assert meta["original_size"] == os.path.getsize(path)
        assert meta["members"][0]["kind"] == "safetensors"
        assert "w" in meta["members"][0]["tensors"]


def test_verify_rejects_damaged_archive():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.safetensors")
        write_safetensors(path, [("w", "BF16", [512, 512], weights_bf16(262144))])
        archive = os.path.join(d, "a.lmz")
        lmz.compress(path, archive)

        flipped = os.path.join(d, "flipped.lmz")
        shutil.copy(archive, flipped)
        with open(flipped, "r+b") as fh:
            fh.seek(os.path.getsize(archive) // 2)
            byte = fh.read(1)
            fh.seek(-1, io.SEEK_CUR)
            fh.write(bytes([byte[0] ^ 0xFF]))
        try:
            lmz.verify(flipped)
            raise AssertionError("a flipped byte went undetected")
        except FormatError:
            pass

        truncated = os.path.join(d, "cut.lmz")
        with open(archive, "rb") as a, open(truncated, "wb") as b:
            b.write(a.read(os.path.getsize(archive) // 2))
        try:
            lmz.verify(truncated)
            raise AssertionError("truncation went undetected")
        except (FormatError, ValueError):
            pass

        with open(os.path.join(d, "junk.lmz"), "wb") as fh:
            fh.write(b"definitely not an archive" * 100)
        try:
            lmz.info(os.path.join(d, "junk.lmz"))
            raise AssertionError("non-archive accepted")
        except FormatError:
            pass


def test_rejects_path_traversal():
    """A hostile member path must not be able to write outside the target."""
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "model")
        os.makedirs(src)
        with open(os.path.join(src, "a.bin"), "wb") as fh:
            fh.write(rand(1000, 2))
        with open(os.path.join(src, "b.bin"), "wb") as fh:
            fh.write(rand(1000, 3))
        archive = os.path.join(d, "a.lmz")
        lmz.compress(src, archive)

        from lmz import api
        for evil in ("../escaped.bin", "/etc/passwd", "a/../../x"):
            try:
                api._safe_member_path(d, evil)
                raise AssertionError(f"accepted hostile path {evil!r}")
            except FormatError:
                pass
        assert api._safe_member_path(d, "sub/ok.bin") == os.path.join(d, "sub", "ok.bin")


def test_read_tensor_matches_original():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.safetensors")
        big = weights_bf16(300000, 4)
        small = weights_bf16(64, 9)
        write_safetensors(path, [("big", "BF16", [300000], big),
                                 ("small", "BF16", [64], small)])
        archive = os.path.join(d, "a.lmz")
        lmz.compress(path, archive, chunk_size=1 << 16)  # force it to span chunks
        dtype, shape, raw = lmz.read_tensor(archive, "big")
        assert dtype == "BF16" and shape == [300000]
        assert raw == big, "extracted tensor differs from the original"
        _, _, raw2 = lmz.read_tensor(archive, "small")
        assert raw2 == small
        try:
            lmz.read_tensor(archive, "missing")
            raise AssertionError("missing tensor should raise")
        except KeyError:
            pass


def test_options_are_honoured():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.safetensors")
        write_safetensors(path, [("w", "BF16", [256, 256], weights_bf16(65536))])
        for kw in ({"checksum": False}, {"level": 5}, {"chunk_size": 1 << 16},
                   {"workers": 1}):
            archive = os.path.join(d, "a.lmz")
            out = os.path.join(d, "o.safetensors")
            lmz.compress(path, archive, **kw)
            lmz.decompress(archive, out, overwrite=True)
            assert digest(out) == digest(path), kw
        with open(archive, "rb") as fh:
            assert ArchiveReader(fh).manifest["checksum"] in ("crc32", "none")


def test_refuses_to_clobber():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.bin")
        with open(path, "wb") as fh:
            fh.write(rand(5000, 1))
        archive = os.path.join(d, "a.lmz")
        lmz.compress(path, archive)
        out = os.path.join(d, "out.bin")
        lmz.decompress(archive, out)
        try:
            lmz.decompress(archive, out)
            raise AssertionError("existing output was silently overwritten")
        except FileExistsError:
            pass
        lmz.decompress(archive, out, overwrite=True)


def test_cli():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.safetensors")
        write_safetensors(path, [("w", "BF16", [512, 512], weights_bf16(262144))])
        archive = os.path.join(d, "m.lmz")
        out = os.path.join(d, "out.safetensors")

        def run(*args, expect=0):
            r = subprocess.run(CLI + list(args), capture_output=True, text=True)
            assert r.returncode == expect, f"{args}\n{r.stdout}\n{r.stderr}"
            return r.stdout

        run("compress", path, archive, "-q")
        assert "safetensors" in run("info", archive)
        assert "BF16" in run("info", archive, "--tensors")
        json.loads(run("info", archive, "--json"))
        run("verify", archive, "-q")
        run("decompress", archive, out, "-q")
        assert digest(out) == digest(path)
        run("decompress", archive, out, expect=1)  # refuses to clobber
        run("decompress", archive, out, "-f", "-q")
        assert "lmz" in run("doctor")
        assert "ratio" in run("bench", path)
        tensor_out = os.path.join(d, "w.bin")
        run("cat", archive, "w", "-o", tensor_out)
        assert os.path.getsize(tensor_out) == 512 * 512 * 2


def test_backends_reported():
    b = lmz.backends()
    assert b["kernel"] and b["entropy"] and b["workers"] >= 1
    assert entropy.HAVE_ZSTD or entropy.DEFAULT_METHOD == entropy.METHOD_DEFLATE


def test_bucket_kernels():
    """The bucket map and the partition must be exact inverses of each other."""
    data = weights_bf16_cond(100000, 3)
    planes = kernels.split_bf16(data)
    n = len(data) // 2
    ctx = bytes(planes[:n])
    val = bytes(planes[n:])
    hist = kernels.histogram(ctx)
    for k in (1, 2, 8, 32):
        lut = kernels.bucket_lut(hist, k)
        assert len(lut) == 256
        assert list(lut) == sorted(lut), "buckets must be contiguous"
        assert max(lut) < k
        counts = kernels.bucket_counts(hist, lut, k)
        assert sum(counts) == n
        part, pcounts = kernels.bucket_partition(ctx, val, lut, k)
        assert pcounts == counts and len(part) == n
        streams, pos = [], 0
        for c in counts:
            streams.append((bytes(part[pos:pos + c]), 0))
            pos += c
        back = kernels.bucket_unpartition(ctx, streams, lut, k)
        assert bytes(back) == val, f"unpartition mismatch at k={k}"
    for bad in (0, 33):
        try:
            kernels.bucket_lut(hist, bad)
            raise AssertionError(f"bucket count {bad} should be rejected")
        except ValueError:
            pass
    # Degenerate context: everything lands in one bucket, the rest stay empty.
    one = bytes([7]) * 1000
    lut = kernels.bucket_lut(kernels.histogram(one), 8)
    part, counts = kernels.bucket_partition(one, rand(1000, 9), lut, 8)
    assert sorted(counts, reverse=True)[0] == 1000 and sum(counts) == 1000


def test_bucket_fallback_matches_native():
    """The pure-Python bucket path must agree with the C one byte for byte."""
    script = (
        "import os,sys;os.environ['LMZ_NO_NATIVE']='1';sys.path.insert(0,%r);"
        "from lmz import kernels;"
        "assert not kernels.backend().startswith('native');"
        "ctx=bytes((i*13+5)%%37 for i in range(20000));"
        "val=bytes((i*7+1)%%256 for i in range(20000));"
        "h=kernels.histogram(ctx);lut=kernels.bucket_lut(h,8);"
        "part,counts=kernels.bucket_partition(ctx,val,lut,8);"
        "streams=[];pos=0\n"
        "for c in counts: streams.append((bytes(part[pos:pos+c]),0));pos+=c\n"
        "back=kernels.bucket_unpartition(ctx,streams,lut,8);"
        "print(lut.hex());print(bytes(part).hex()[:64]);print(bytes(back).hex()[:64])"
        % ROOT
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    lut_hex, part_hex, back_hex = out.stdout.strip().splitlines()
    ctx = bytes((i * 13 + 5) % 37 for i in range(20000))
    val = bytes((i * 7 + 1) % 256 for i in range(20000))
    h = kernels.histogram(ctx)
    lut = kernels.bucket_lut(h, 8)
    part, counts = kernels.bucket_partition(ctx, val, lut, 8)
    assert lut.hex() == lut_hex
    assert bytes(part).hex()[:64] == part_hex
    assert bytes(val).hex()[:64] == back_hex


def test_bf16_conditional_roundtrip():
    """Exponent-dependent mantissas must trigger the conditioned codec and win."""
    if not kernels.have_rans():
        return
    data = weights_bf16_cond(codec.COND_MIN_ELEMS, 5)
    parts, cid, flags, crc = codec.encode_chunk(data, 2, 1, True,
                                                kind=planner.KIND_BF16)
    assert cid == lmzformat.CODEC_BF16C, f"expected conditioned codec, got {cid}"
    payload = b"".join(bytes(p) for p in parts)
    got = codec.decode_chunk(payload, cid, 2, flags, len(data), crc, True)
    assert bytes(got) == data

    # And it must actually be smaller than the unconditioned field split.
    import lmz.codec as codec_mod
    saved_min = codec_mod.COND_MIN_ELEMS
    try:
        codec_mod.COND_MIN_ELEMS = 1 << 62  # forbid conditioning
        plain_parts, plain_cid, pf, pc = codec.encode_chunk(
            data, 2, 1, True, kind=planner.KIND_BF16)
    finally:
        codec_mod.COND_MIN_ELEMS = saved_min
    assert plain_cid == lmzformat.CODEC_BF16
    assert sum(len(p) for p in parts) < sum(len(p) for p in plain_parts)


def test_bf16_conditional_declines_uncorrelated():
    """With no exponent-mantissa link the plain field split must stay."""
    if not kernels.have_rans():
        return
    data = weights_bf16(codec.COND_MIN_ELEMS, 7)
    parts, cid, flags, crc = codec.encode_chunk(data, 2, 1, True,
                                                kind=planner.KIND_BF16)
    assert cid == lmzformat.CODEC_BF16, f"conditioning should decline, got {cid}"
    payload = b"".join(bytes(p) for p in parts)
    assert bytes(codec.decode_chunk(payload, cid, 2, flags, len(data), crc,
                                    True)) == data


def test_bf16_conditional_detects_corruption():
    if not kernels.have_rans():
        return
    data = weights_bf16_cond(codec.COND_MIN_ELEMS, 11)
    parts, cid, flags, crc = codec.encode_chunk(data, 2, 1, True,
                                                kind=planner.KIND_BF16)
    assert cid == lmzformat.CODEC_BF16C
    payload = bytearray(b"".join(bytes(p) for p in parts))
    for offset in (2, codec._bf16c_hdr.size + 8, len(payload) - 8):
        damaged = bytearray(payload)
        damaged[offset] ^= 0xFF
        try:
            codec.decode_chunk(bytes(damaged), cid, 2, flags, len(data), crc, True)
            raise AssertionError(f"corruption at offset {offset} went undetected")
        except FormatError:
            pass


def test_dedup_across_files():
    """A tensor shipped twice under different names must be stored once."""
    shared = weights_bf16(150000, 21)
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "model")
        os.makedirs(src)
        # Distinct odd seeds: the generator ORs the seed with 1, so an even
        # seed and its successor would silently produce identical tensors.
        write_safetensors(os.path.join(src, "m1.safetensors"), [
            ("w.emb", "BF16", [150000], shared),
            ("w.a", "BF16", [80000], weights_bf16(80000, 25)),
        ])
        write_safetensors(os.path.join(src, "m2.safetensors"), [
            ("tok", "BF16", [150000], shared),
            ("w.b", "BF16", [80000], weights_bf16(80000, 27)),
        ])
        plain = os.path.join(d, "plain.lmz")
        deduped = os.path.join(d, "dedup.lmz")
        lmz.compress(src, plain, dedup=False)
        stats = lmz.compress(src, deduped)
        assert stats.detail["dedup_bytes"] == len(shared)
        assert "ref" in lmz.info(deduped)["codecs"], lmz.info(deduped)["codecs"]
        assert "ref" not in lmz.info(plain)["codecs"]
        # The duplicate must cost (almost) nothing beyond its chunk records.
        assert os.path.getsize(deduped) + len(shared) // 2 < os.path.getsize(plain)

        out = os.path.join(d, "restored")
        lmz.decompress(deduped, out)
        for name in ("m1.safetensors", "m2.safetensors"):
            assert digest(os.path.join(src, name)) == digest(os.path.join(out, name))

        # Extraction must resolve refs too.
        dtype, shape, raw = lmz.read_tensor(deduped, "tok", member="m2.safetensors")
        assert raw == shared


def test_dedup_within_file():
    """Tied embeddings: the same bytes twice inside one member."""
    shared = weights_bf16(120000, 31)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.safetensors")
        write_safetensors(path, [
            ("embed", "BF16", [120000], shared),
            ("mid", "BF16", [50000], weights_bf16(50000, 32)),
            ("head", "BF16", [120000], shared),
        ])
        archive = os.path.join(d, "a.lmz")
        stats = lmz.compress(path, archive)
        assert stats.detail["dedup_bytes"] == len(shared)
        out = os.path.join(d, "o.safetensors")
        lmz.decompress(archive, out)
        assert digest(out) == digest(path)
        _, _, raw = lmz.read_tensor(archive, "head")
        assert raw == shared


def test_dedup_deterministic_across_workers():
    shared = weights_bf16(100000, 41)
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "model")
        os.makedirs(src)
        write_safetensors(os.path.join(src, "a.safetensors"),
                          [("x", "BF16", [100000], shared)])
        write_safetensors(os.path.join(src, "b.safetensors"),
                          [("y", "BF16", [100000], shared)])
        digests = set()
        for workers in (1, 4, 16):
            archive = os.path.join(d, f"a{workers}.lmz")
            lmz.compress(src, archive, workers=workers, chunk_size=1 << 16)
            digests.add(digest(archive))
        assert len(digests) == 1, "dedup made archives depend on worker count"


def test_ref_corruption_rejected():
    """Hostile ref chunks must fail cleanly, not loop or write garbage."""
    from lmz.format import CODEC_REF, CODEC_STORED, ArchiveWriter

    with tempfile.TemporaryDirectory() as d:
        member = {"path": "x.bin", "size": 16, "dst": 0, "kind": "raw",
                  "mode": 0o644, "crc": 0}

        # A ref whose target range is covered by the ref itself.
        selfref = os.path.join(d, "selfref.lmz")
        with open(selfref, "wb") as fh:
            w = ArchiveWriter(fh, {"members": [member], "total_size": 16})
            w.append([struct.pack("<Q", 0)], 0, 16, 0, CODEC_REF, 1, 0)
            w.close(16)
        # A ref pointing outside every chunk.
        wild = os.path.join(d, "wild.lmz")
        with open(wild, "wb") as fh:
            w = ArchiveWriter(fh, {"members": [member], "total_size": 16})
            w.append([b"\xAA" * 8], 0, 8, 0, CODEC_STORED, 1, 0)
            w.append([struct.pack("<Q", 1 << 40)], 8, 8, 0, CODEC_REF, 1, 0)
            w.close(16)

        for path in (selfref, wild):
            for op in (lmz.verify,
                       lambda p: lmz.decompress(p, os.path.join(d, "out"),
                                                overwrite=True)):
                try:
                    op(path)
                    raise AssertionError(f"{os.path.basename(path)} was accepted")
                except FormatError:
                    pass


def test_pytorch_bin_layout_and_roundtrip():
    """A torch-style zip checkpoint must be typed, split and restored exactly."""
    # FP32 upcast from BF16: alternating (0, 0, mantissa, sign+exp) bytes.
    words = weights_bf16(120000, 51)
    f32 = bytearray()
    for i in range(0, len(words), 2):
        f32 += b"\x00\x00" + words[i:i + 2]
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "pytorch_model.bin")
        write_torch_bin(path, [
            ("0", "FloatStorage", bytes(f32)),
            ("1", "HalfStorage", rand(65536, 52)),
            ("2", "BFloat16Storage", weights_bf16(65536, 53)),
        ])
        with open(path, "rb") as fh:
            layout = planner.probe(fh, os.path.getsize(path))
        assert layout.kind == "pytorch", layout.kind
        assert layout.tensors["data/0"]["dtype"] == "F32"
        assert layout.tensors["data/2"]["dtype"] == "BF16"
        esizes = {r.esize for r in layout.regions}
        assert 4 in esizes and 2 in esizes, esizes

        archive = os.path.join(d, "a.lmz")
        stats = lmz.compress(path, archive, chunk_size=1 << 18)
        out = os.path.join(d, "restored.bin")
        lmz.decompress(archive, out)
        assert digest(out) == digest(path)
        # Two of four byte planes in the fp32 payload are zeros; anything
        # near or below 1.5x means the container was not really parsed.
        assert stats.ratio > 1.5, f"fp32-from-bf16 only reached {stats.ratio:.3f}x"


def test_stride_kernels():
    """The strided split must equal slicing and merge must undo it."""
    for period in (1, 2, 17, 34, 64):
        for nblocks in (0, 1, 7, 100, 4097):
            src = rand(nblocks * period, nblocks + period)
            planes = kernels.split_stride(src, period)
            expect = b"".join(src[k::period] for k in range(period))
            assert bytes(planes) == expect, f"split period={period} n={nblocks}"
            streams = [(bytes(planes), k * nblocks) for k in range(period)]
            back = kernels.merge_stride(streams, nblocks, period)
            assert bytes(back) == src, f"merge period={period} n={nblocks}"
    for bad in (0, 65):
        try:
            kernels.split_stride(b"\0" * 130 * max(bad, 1), bad)
            raise AssertionError(f"period {bad} should be rejected")
        except ValueError:
            pass


def test_stride_fallback_matches_native():
    script = (
        "import os,sys;os.environ['LMZ_NO_NATIVE']='1';sys.path.insert(0,%r);"
        "from lmz import kernels;"
        "assert not kernels.backend().startswith('native');"
        "src=bytes((i*11+3)%%256 for i in range(34*500));"
        "p=kernels.split_stride(src,34);"
        "s=[(bytes(p),k*500) for k in range(34)];"
        "print(bytes(p).hex()[:64]);"
        "print(bytes(kernels.merge_stride(s,500,34)).hex()[:64])" % ROOT
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    split_hex, merge_hex = out.stdout.strip().splitlines()
    src = bytes((i * 11 + 3) % 256 for i in range(34 * 500))
    assert split_hex == bytes(kernels.split_stride(src, 34)).hex()[:64]
    assert merge_hex == src.hex()[:64]


def test_q80_block_codec_roundtrip():
    """Q8_0 chunks must choose the block codec and restore exactly."""
    if not kernels.have_rans():
        return
    data = q80_blocks(codec.Q80_MIN_BLOCKS, 3)
    parts, cid, flags, crc = codec.encode_chunk(data, 34, 1, True,
                                                kind=planner.KIND_Q80)
    assert cid == lmzformat.CODEC_BLK, f"expected block codec, got {cid}"
    payload = b"".join(bytes(p) for p in parts)
    assert len(payload) < len(data) * 0.95, "block codec should win clearly"
    got = codec.decode_chunk(payload, cid, 34, flags, len(data), crc, True)
    assert bytes(got) == data

    for offset in (3, codec._q80_hdr.size + 4, len(payload) - 4):
        damaged = bytearray(payload)
        damaged[offset] ^= 0xFF
        try:
            codec.decode_chunk(bytes(damaged), cid, 34, flags, len(data), crc, True)
            raise AssertionError(f"corruption at offset {offset} went undetected")
        except FormatError:
            pass


def test_q80_gguf_file_roundtrip():
    """A Q8_0 GGUF must be block-coded through the whole pipeline."""
    if not kernels.have_rans():
        return
    nblocks = 150000
    raw = q80_blocks(nblocks, 7)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.gguf")
        write_gguf(path, [("blk.0.w", 8, [32 * nblocks], raw)])
        with open(path, "rb") as fh:
            layout = planner.probe(fh, os.path.getsize(path))
        region = next(r for r in layout.regions if r.kind == planner.KIND_Q80)
        assert region.esize == 34

        archive = os.path.join(d, "a.lmz")
        stats = lmz.compress(path, archive, chunk_size=2 << 20)
        assert "q8-block" in lmz.info(archive)["codecs"]
        out = os.path.join(d, "o.gguf")
        lmz.decompress(archive, out)
        assert digest(out) == digest(path)
        assert stats.saved > 0.05, f"only saved {stats.saved:.2%}"


def test_v1_archives_still_read():
    """Version-1 archives predate refs and conditioning; they must still open."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.safetensors")
        write_safetensors(path, [("w", "BF16", [65536], weights_bf16(65536, 61))])
        archive = os.path.join(d, "a.lmz")
        lmz.compress(path, archive, dedup=False)
        with open(archive, "r+b") as fh:
            fh.seek(4)  # the version field of the header
            fh.write(struct.pack("<H", 1))
        out = os.path.join(d, "o.safetensors")
        lmz.decompress(archive, out)
        assert digest(out) == digest(path)


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            import traceback

            traceback.print_exc()
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
