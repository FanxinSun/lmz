"""Test suite for lmz.

Runs standalone (`python3 tests/test_lmz.py`) and under pytest. Every
round-trip check compares bytes, not just sizes: the whole point of the tool
is that decompression reproduces the input exactly.
"""

from __future__ import annotations

import errno
import hashlib
import io
import json
import math
import os
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lmz  # noqa: E402
from lmz import codec, entropy, freqs, kernels, planner  # noqa: E402
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


def weights_f32_from_f16(nelem: int, seed: int = 1, band: int = 0) -> bytes:
    """FP32 words holding values that only ever had fp16 precision.

    The shape whisper and bge-m3 really ship in, and the one shared tables are
    for: the low 13 mantissa bits are zero, so byte 0 is constant and byte 1
    takes eight values, while byte 3 carries the sign and exponent and so
    moves with `band` -- a stand-in for the per-tensor scale that makes one
    tensor's high byte a poor fit for another's.
    """
    out = bytearray()
    x = seed | 1
    for _ in range(nelem):
        x = (x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        mant = (x >> 11) & 0x3FF                     # 10 bits, as fp16 has
        exp = 118 + band + ((x >> 40) % 5)
        sign = (x >> 63) & 1
        out += ((sign << 31) | (exp << 23) | (mant << 13)).to_bytes(4, "little")
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


def quant_blocks(ttype: int, nblocks: int, seed: int = 1) -> bytes:
    """Synthetic GGUF blocks carrying the structure the block codec looks for.

    Each field is filled according to its role in the layout: a 2-byte field
    is an fp16 scale (few exponent values, with the low byte partly determined
    by the high byte), a wide field is quants on a narrowed gaussian-ish
    alphabet, and the narrow fields in between are packed sub-scales. That is
    the shape real quantised weights have, and it exercises all three group
    modes -- conditioned pair, per-position and concatenated.
    """
    period, groups = planner.BLOCK_LAYOUTS[ttype]
    out = bytearray()
    x = seed | 1

    def nxt() -> int:
        nonlocal x
        x = (x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return x

    for _ in range(nblocks):
        for _start, width in groups:
            if width == 2:
                v = nxt()
                hi = 0x20 | ((v >> 5) & 0x1F)   # 32 exponent values
                # Two bits shared with the top of the exponent, which is what
                # equal-mass bucketing of the high byte can actually see.
                out.append(((v >> 16) & 0x3F) | (((hi >> 2) & 0x3) << 6))
                out.append(hi)
            elif width >= 32:
                for _ in range(width):
                    v = nxt()
                    out.append(((v & 0xFF) + ((v >> 8) & 0xFF) + ((v >> 16) & 0xFF)
                                + ((v >> 24) & 0xFF)) >> 2 & 0xFF)
            else:
                # Packed sub-scales: each byte position packs a different mix
                # of fields, so the alphabets differ by position the way a
                # k-quant's 6-bit scale array does.
                for j in range(width):
                    out.append((nxt() >> 11) & (0x3F >> (j & 0x3)))
    assert len(out) == nblocks * period
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


def test_bf16_conditioning_beats_a_byte_split():
    """What earns the BF16 path its ratio is the conditioning, not the split.

    Splitting on the field boundary rather than the byte boundary moves almost
    nothing on its own. It puts the whole exponent in one plane and sign with
    mantissa in the other, where a byte split puts the sign alongside the
    exponent's high bits and leaves its low bit with the mantissa -- and
    measured across several distributions the two land within a few tenths of a
    percent, falling either way. This assertion used to be made on the split
    alone and passed by four bytes on 530 KiB, which is a coin flip rather than
    a property.

    The win is in coding the sign+mantissa plane against the exponent, and that
    needs the rANS coder and a chunk with elements enough to pay for one
    frequency table per bucket. Given both, it is worth about 7% here and is
    asserted with a margin clear of noise, so that losing the conditioning fails
    rather than squeaks through.
    """
    if not kernels.have_rans():
        return
    data = weights_bf16_cond(1_200_000, 7)
    field = codec.encode_chunk(data, 2, 1, False, kind=planner.KIND_BF16)
    byte = codec.encode_chunk(data, 2, 1, False, kind=planner.KIND_BYTES)
    size = lambda r: sum(len(p) for p in r[0])  # noqa: E731
    assert field[1] == lmzformat.CODEC_BF16C, field[1]
    assert byte[1] == lmzformat.CODEC_SPLIT, byte[1]
    for parts, cid, flags, crc in (field, byte):
        payload = b"".join(bytes(p) for p in parts)
        got = codec.decode_chunk(payload, cid, 2, flags, len(data), crc, True)
        assert bytes(got) == data
    assert size(field) < size(byte) * 0.97, (size(field), size(byte))


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


def test_rans_vector_path_writes_the_same_bytes():
    """The vector coder is a speed change, not a format change.

    It steps the same eight interleaved states eight at a time, so it has to
    write byte for byte what the portable loop writes -- otherwise an archive
    would depend on which machine produced it, and a build without the vector
    path could not reproduce one that had it.

    Sizes straddle the length the vector path starts at and the eight-symbol
    group boundary either side of it, so the ragged tail and the handover from
    scalar to vector are both exercised.
    """
    if not kernels.have_rans():
        return
    sizes = (4088, 4095, 4096, 4097, 4103, 4104, 4111, 8192, 65537)
    shapes = {
        "weights": lambda n: weights_bf16(n // 2 + 1)[:n],
        "noise": lambda n: rand(n, n + 3),
        "few symbols": lambda n: bytes((i * i) % 7 for i in range(n)),
        # Drives one frequency past half the scale, which is where the vector
        # path's reciprocal stops being exact and it must decline.
        "one dominant": lambda n: bytes(5 if i % 8 else (i & 0xFF)
                                        for i in range(n)),
        "single symbol": lambda n: bytes([200]) * n,
    }
    for n in sizes:
        for name, make in shapes.items():
            data = make(n)
            assert len(data) == n, (name, n)
            portable = kernels.rans_encode(data, portable=True)
            vector = kernels.rans_encode(data)
            assert vector == portable, f"{name}, n={n}"
            assert bytes(kernels.rans_decode(vector, n)) == data, f"{name}, n={n}"


def test_rans_accepts_a_precomputed_histogram():
    """Handing the coder counts it would otherwise take must change nothing.

    The caller usually has them, because whether to code at all is decided
    from a histogram. Counts of the wrong bytes are the hazard: a symbol given
    frequency zero cannot be coded at all, so the kernel checks their total
    against the buffer and counts for itself rather than trusting them.
    """
    if not kernels.have_rans():
        return
    cases = (("weights", weights_bf16(200000, 5)),
             ("noise", rand(200000, 6)),
             ("few symbols", bytes((i * i) % 7 for i in range(200000))))
    for name, data in cases:
        hist = kernels.histogram(data)
        with_hist = kernels.rans_encode(data, hist)
        plain = kernels.rans_encode(data)
        assert with_hist == plain, name
        assert bytes(kernels.rans_decode(with_hist, len(data))) == data, name
        # A histogram of other bytes. The total gives it away and the kernel
        # counts again, so the stream is the same one either way.
        wrong = kernels.histogram(data[:len(data) // 2])
        assert kernels.rans_encode(data, wrong) == plain, f"{name}, wrong counts"


def test_rans_cost_is_a_floor():
    """Streams are skipped on this figure, so it must not exceed the truth.

    `_rans_cost` prices a stream against the frequencies the coder will use,
    and the encoder is never run when the price cannot clear the threshold. If
    it ever came out above what the coder charges, a stream that would have
    paid its way could be stored instead. Near-uniform alphabets are where the
    twelve-bit quantisation bites hardest and the margin is thinnest.
    """
    if not kernels.have_rans():
        return
    data = weights_bf16(400000, 12)
    planes = kernels.split_bf16(data)
    n = len(data) // 2
    cases = (("exponent", bytes(planes[:n])),
             ("sign+mantissa", bytes(planes[n:])),
             ("uniform", rand(300000, 13)),
             ("nearly uniform",
              bytes((i * 251) % 256 if i % 17 else 5 for i in range(300000))),
             ("single symbol", bytes([3]) * 100000),
             ("two symbols", bytes((i & 1) * 200 for i in range(100000))))
    for name, buf in cases:
        hist = kernels.histogram(buf)
        cost = codec._rans_cost(hist, len(buf))
        actual = len(kernels.rans_encode(buf, hist))
        assert cost is not None, name
        assert cost <= actual, f"{name}: priced {cost}, coder charged {actual}"
        # And close enough that the floor is worth deciding on: a figure far
        # under the truth would refuse to skip anything.
        assert actual - cost <= 2 * codec.RANS_COST_SLACK, (name, cost, actual)
        # Counts of other bytes price a stream that is not this one, so the
        # decision they would inform is declined rather than made.
        assert codec._rans_cost(kernels.histogram(buf[:len(buf) // 2]),
                                len(buf)) is None, name


def test_bucket_partition_accepts_a_precomputed_histogram():
    """Same contract as the coder's: use the caller's counts, verify the total."""
    ctx = rand(50000, 21)
    val = rand(50000, 22)
    hist = kernels.histogram(ctx)
    lut = kernels.bucket_lut(hist, 8)
    plain = kernels.bucket_partition(ctx, val, lut, 8)
    assert kernels.bucket_partition(ctx, val, lut, 8, hist=hist) == plain
    wrong = kernels.histogram(ctx[:100])
    assert kernels.bucket_partition(ctx, val, lut, 8, hist=wrong) == plain


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
        for start, end, *_rest in chunks:
            assert end > start
        # BF16 regions must be tagged so the codec can split on field bounds.
        kinds = {c[3] for c in chunks}
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


def nudged_bf16(raw: bytes, seed: int, rate: int = 3) -> bytes:
    """The same weights after a little more training.

    Most values move by a step or two in the low mantissa bits and a few not
    at all, which is what a checkpoint 1000 steps later actually looks like:
    nothing dedups, but the XOR is almost all zeros.
    """
    out = bytearray(raw)
    x = seed | 1
    for i in range(0, len(out) - 1, 2):
        x = (x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        v = x >> 33
        if v % rate:
            out[i] ^= (v >> 8) & 0x07  # low mantissa bits only
    return bytes(out)


def test_delta_against_an_earlier_file():
    """A near-copy must be coded as a difference, and come back exactly."""
    base = weights_bf16(700000, 31)
    moved = nudged_bf16(base, 5)
    assert moved != base
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "run")
        os.makedirs(src)
        # Same tensor names in both, which is what a checkpoint series gives.
        for name, blob in (("ckpt-1.safetensors", base),
                           ("ckpt-2.safetensors", moved)):
            write_safetensors(os.path.join(src, name),
                              [("layer.0.weight", "BF16", [700000], blob)])
        plain = os.path.join(d, "plain.lmz")
        delta = os.path.join(d, "delta.lmz")
        lmz.compress(src, plain, delta=False)
        stats = lmz.compress(src, delta)

        assert stats.detail["delta_bytes"] == len(base), stats.detail
        assert "delta" in lmz.info(delta)["codecs"], lmz.info(delta)["codecs"]
        assert "delta" not in lmz.info(plain)["codecs"]
        assert os.path.getsize(delta) < os.path.getsize(plain) * 0.8, (
            os.path.getsize(delta), os.path.getsize(plain))

        out = os.path.join(d, "restored")
        lmz.decompress(delta, out)
        for name in ("ckpt-1.safetensors", "ckpt-2.safetensors"):
            assert digest(os.path.join(src, name)) == digest(os.path.join(out, name))
        lmz.verify(delta)

        # Extraction has to resolve the source range too.
        _dt, _sh, raw = lmz.read_tensor(delta, "layer.0.weight",
                                        member="ckpt-2.safetensors")
        assert raw == moved


def test_delta_declines_when_the_files_differ():
    """Unrelated tensors must not be coded as differences from each other."""
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "run")
        os.makedirs(src)
        for i, seed in enumerate((41, 43)):
            write_safetensors(os.path.join(src, f"m{i}.safetensors"),
                              [("layer.0.weight", "BF16", [700000],
                                weights_bf16(700000, seed))])
        arc = os.path.join(d, "a.lmz")
        stats = lmz.compress(src, arc)
        assert stats.detail["delta_bytes"] == 0, stats.detail
        assert "delta" not in lmz.info(arc)["codecs"]


def test_delta_corruption_rejected():
    """A damaged delta must be rejected, never returned as wrong bytes."""
    base = weights_bf16(700000, 31)
    moved = nudged_bf16(base, 5)
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "run")
        os.makedirs(src)
        for name, blob in (("a.safetensors", base), ("b.safetensors", moved)):
            write_safetensors(os.path.join(src, name),
                              [("layer.0.weight", "BF16", [700000], blob)])
        arc = os.path.join(d, "a.lmz")
        lmz.compress(src, arc)
        with open(arc, "rb") as fh:
            reader = ArchiveReader(fh)
            target = next(c for c in reader.chunks
                          if c.codec == lmzformat.CODEC_DELTA)

        # A source offset pointing at the delta's own output is circular.
        for off, val in ((0, struct.pack("<Q", target.dst)),
                         (0, struct.pack("<Q", 1 << 62)),
                         (8, b"\xff")):
            blob = bytearray(open(arc, "rb").read())
            blob[target.off + off:target.off + off + len(val)] = val
            bad = os.path.join(d, "bad.lmz")
            with open(bad, "wb") as fh:
                fh.write(blob)
            try:
                lmz.verify(bad)
            except (FormatError, ValueError):
                continue
            raise AssertionError(f"damage at delta byte {off} was accepted")


def test_mapped_archive_random_access():
    """A page-mapped archive must serve any byte range from one small block."""
    blob = weights_bf16(900000, 51)
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "m.safetensors")
        write_safetensors(src, [("a.weight", "BF16", [900000], blob)])
        raw = open(src, "rb").read()

        plain = os.path.join(d, "plain.lmz")
        mapped = os.path.join(d, "mapped.lmz")
        lmz.compress(src, plain)
        lmz.compress(src, mapped, mapped=True)

        with lmz.MappedArchive(mapped) as a:
            assert a.page_mapped and not a.aligned
            assert a.block_bytes <= lmz.api.DEFAULT_BLOCK_SIZE
            assert a.size == len(raw)
            # every read must match the original file exactly
            for off, ln in ((0, 1), (len(raw) - 1, 1), (5, 100000),
                            (len(raw) // 3, 70000), (0, len(raw))):
                assert a.read(off, ln) == raw[off:off + ln], (off, ln)
            # reading past the end clamps rather than failing
            assert a.read(len(raw) - 10, 999) == raw[-10:]
            assert a.read(len(raw), 10) == b""

        # a one-byte read must not expand more than one block
        with lmz.MappedArchive(mapped) as a:
            a.read(len(raw) // 2, 1)
            assert a.decoded_bytes <= lmz.api.DEFAULT_BLOCK_SIZE, a.decoded_bytes
        with lmz.MappedArchive(plain) as a:
            a.read(len(raw) // 2, 1)
            big = a.decoded_bytes
        assert big > lmz.api.DEFAULT_BLOCK_SIZE * 8, big

        # neither the cache nor the read pool may change what is returned
        for kw in ({"cache_blocks": 1}, {"workers": 1}, {"workers": 2},
                   {"workers": 2, "cache_blocks": 1}):
            with lmz.MappedArchive(mapped, **kw) as a:
                assert a.read(1000, 200000) == raw[1000:201000], kw
                assert a.read(0, len(raw)) == raw, kw
        assert lmz.info(mapped)["manifest"]["mapped"] is True


def test_mapped_archive_still_decompresses_whole():
    """Small blocks and padding must not disturb the ordinary paths."""
    blob = weights_bf16(400000, 53)
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "m.safetensors")
        write_safetensors(src, [("a.weight", "BF16", [400000], blob)])
        for name, kw in (("mapped.lmz", {"mapped": True}),
                         ("aligned.lmz", {"mapped": True, "align": True})):
            arc = os.path.join(d, name)
            lmz.compress(src, arc, **kw)
            lmz.verify(arc)
            out = os.path.join(d, "out-" + name)
            lmz.decompress(arc, out)
            assert digest(src) == digest(out), name
            _dt, _sh, got = lmz.read_tensor(arc, "a.weight")
            assert got == blob, name


def test_aligned_archive_starts_blocks_on_page_boundaries():
    """--align must actually align, and cost only the padding it writes."""
    blob = weights_bf16(600000, 55)
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "m.safetensors")
        write_safetensors(src, [("a.weight", "BF16", [600000], blob)])
        loose = os.path.join(d, "loose.lmz")
        tight = os.path.join(d, "tight.lmz")
        lmz.compress(src, loose, mapped=True)
        stats = lmz.compress(src, tight, mapped=True, align=True)

        with open(tight, "rb") as fh:
            reader = ArchiveReader(fh)
        assert reader.aligned and reader.page_mapped
        for c in reader.chunks:
            assert c.off % lmzformat.PAGE_ALIGN == 0, c.off
        # Padding accounts for the growth. Not to the byte: larger offsets
        # make the chunk table compress slightly differently.
        pad = stats.detail["padding_bytes"]
        grew = os.path.getsize(tight) - os.path.getsize(loose)
        assert 0 < pad <= lmzformat.PAGE_ALIGN * len(reader.chunks), pad
        assert abs(grew - pad) < (1 << 16), (grew, pad)

        with open(loose, "rb") as fh:
            assert not ArchiveReader(fh).aligned

        try:
            lmz.compress(src, os.path.join(d, "x.lmz"), align=True)
        except ValueError:
            pass
        else:
            raise AssertionError("align without mapped should be rejected")


def test_append_grows_an_archive_like_one_shot():
    """A series grown one file at a time must match compressing it together.

    This is the workflow the delta coding exists for: checkpoints appear over
    hours, and recompressing every earlier one to add the next is not a thing
    anybody would do.
    """
    steps = [weights_bf16(700000, 61)]
    for i in range(3):
        steps.append(nudged_bf16(steps[-1], 70 + i))
    with tempfile.TemporaryDirectory() as d:
        names = []
        for i, blob in enumerate(steps):
            p = os.path.join(d, f"ck{i}.safetensors")
            write_safetensors(p, [("layer.0.weight", "BF16", [700000], blob)])
            names.append(p)

        grown = os.path.join(d, "grown.lmz")
        lmz.compress(names[0], grown)
        for p in names[1:]:
            stats = lmz.append(grown, p)
            assert stats.detail["appended"] is True

        series = os.path.join(d, "series")
        os.makedirs(series)
        for p in names:
            shutil.copy(p, series)
        one = os.path.join(d, "one.lmz")
        lmz.compress(series, one, workers=4)

        # Within a rounding of each other: appending must not cost ratio.
        g, o = os.path.getsize(grown), os.path.getsize(one)
        assert abs(g - o) < max(4096, o // 200), (g, o)
        assert g < sum(len(b) for b in steps) * 0.9

        lmz.verify(grown)
        out = os.path.join(d, "back")
        lmz.decompress(grown, out)
        for i, p in enumerate(names):
            assert digest(p) == digest(os.path.join(out, f"ck{i}.safetensors"))

        # and the appended members must be reachable one at a time
        for i, p in enumerate(names):
            got = os.path.join(d, f"got{i}.safetensors")
            lmz.extract(grown, f"ck{i}.safetensors", got)
            assert digest(got) == digest(p), i
            _dt, _sh, raw = lmz.read_tensor(grown, "layer.0.weight",
                                            member=f"ck{i}.safetensors")
            assert raw == steps[i], i


def test_append_rejects_a_name_already_present():
    blob = weights_bf16(300000, 63)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.safetensors")
        write_safetensors(p, [("w", "BF16", [300000], blob)])
        arc = os.path.join(d, "a.lmz")
        lmz.compress(p, arc)
        before = os.path.getsize(arc)
        try:
            lmz.append(arc, p)
        except ValueError:
            pass
        else:
            raise AssertionError("appending a duplicate member should fail")
        assert os.path.getsize(arc) == before, "a rejected append must not write"
        lmz.verify(arc)


def test_append_keeps_the_archive_kind():
    """Appending to a page-mapped archive must not silently un-map it."""
    a = weights_bf16(700000, 65)
    b = nudged_bf16(a, 67)
    with tempfile.TemporaryDirectory() as d:
        pa = os.path.join(d, "a.safetensors")
        pb = os.path.join(d, "b.safetensors")
        write_safetensors(pa, [("w", "BF16", [700000], a)])
        write_safetensors(pb, [("w", "BF16", [700000], b)])
        arc = os.path.join(d, "m.lmz")
        lmz.compress(pa, arc, mapped=True)
        lmz.append(arc, pb)
        with lmz.MappedArchive(arc) as m:
            assert m.page_mapped, "the flag must survive an append"
            assert m.block_bytes <= lmz.api.DEFAULT_BLOCK_SIZE
            assert m.tensor("w", "a.safetensors")[2] == a
            assert m.tensor("w", "b.safetensors")[2] == b
        lmz.verify(arc)
        out = os.path.join(d, "back")
        lmz.decompress(arc, out)
        assert digest(pb) == digest(os.path.join(out, "b.safetensors"))


def test_extract_one_member():
    blobs = [weights_bf16(200000, 71), weights_bf16(200000, 73)]
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "pair")
        os.makedirs(src)
        for i, blob in enumerate(blobs):
            write_safetensors(os.path.join(src, f"m{i}.safetensors"),
                              [("w", "BF16", [200000], blob)])
        arc = os.path.join(d, "a.lmz")
        lmz.compress(src, arc)
        got = os.path.join(d, "m1.out")
        lmz.extract(arc, "m1.safetensors", got)
        assert digest(os.path.join(src, "m1.safetensors")) == digest(got)

        try:
            lmz.extract(arc, "m1.safetensors", got)
        except FileExistsError:
            pass
        else:
            raise AssertionError("extract must not clobber without overwrite")
        lmz.extract(arc, "m1.safetensors", got, overwrite=True)

        try:
            lmz.extract(arc, "nope.safetensors", os.path.join(d, "x"))
        except KeyError:
            pass
        else:
            raise AssertionError("extracting a missing member should fail")


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


def _pb_varint(v: int) -> bytes:
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        out.append(b | (0x80 if v else 0))
        if not v:
            return bytes(out)


def _pb_len(field: int, payload: bytes) -> bytes:
    return _pb_varint(field << 3 | 2) + _pb_varint(len(payload)) + payload


def _pb_int(field: int, value: int) -> bytes:
    return _pb_varint(field << 3) + _pb_varint(value)


def _onnx_tensor(name: str, dtype_id: int, dims, raw: bytes) -> bytes:
    """One TensorProto, written the way an exporter writes it."""
    return (b"".join(_pb_int(1, d) for d in dims)
            + _pb_int(2, dtype_id)
            + _pb_len(8, name.encode())
            + _pb_len(9, raw))


def _onnx_model(tensors) -> bytes:
    graph = b"".join(_pb_len(5, t) for t in tensors) + _pb_len(2, b"main")
    return _pb_int(1, 8) + _pb_len(7, graph) + _pb_len(8, b"test")


def test_onnx_initializers_are_addressed_and_typed():
    """ONNX is what perception ships in, and its weights must be typed.

    Without a parse every initializer travels as opaque bytes, and the
    byte-position planes cannot phase-align because nothing knows the element
    width -- an fp16 exponent then smears across positions instead of
    collecting in one plane. The offsets have to land on the real bytes, and
    the dtypes have to reach the same splitter safetensors uses.
    """
    f32 = rand(4096, 3)
    f16 = rand(2048, 5)
    i8 = rand(256, 7)
    model = _onnx_model([
        _onnx_tensor("conv1.weight", 1, [16, 64], f32),      # FLOAT
        _onnx_tensor("conv2.weight", 10, [8, 128], f16),     # FLOAT16
        _onnx_tensor("quant.weight", 3, [16, 16], i8),       # INT8
        _onnx_tensor("attn.weight", 16, [2, 2], rand(8, 11)),  # BFLOAT16
    ])

    layout = planner.probe(io.BytesIO(model), len(model))
    assert layout.kind == "onnx", layout.kind
    assert set(layout.tensors) == {"conv1.weight", "conv2.weight",
                                   "quant.weight", "attn.weight"}
    assert layout.tensors["conv1.weight"]["dtype"] == "F32"
    assert layout.tensors["conv1.weight"]["shape"] == [16, 64]
    assert layout.tensors["quant.weight"]["dtype"] == "I8"

    # The offsets must point at the weights themselves, not at the message.
    for name, payload in (("conv1.weight", f32), ("conv2.weight", f16),
                          ("quant.weight", i8)):
        start, end = layout.tensors[name]["offsets"]
        assert model[start:end] == payload, name

    # And each region must carry the element width its splitter needs.
    widths = sorted((r.end - r.start, r.esize, r.kind) for r in layout.regions)
    assert (len(i8), 1, planner.KIND_BYTES) in widths
    assert (len(f16), 2, planner.KIND_BYTES) in widths
    assert (len(f32), 4, planner.KIND_BYTES) in widths
    # BF16 reaches the conditioned path, exactly as it does from safetensors.
    assert (8, 2, planner.KIND_BF16) in widths


def test_onnx_that_is_not_onnx_degrades_to_bytes():
    """Anything unexpected must become opaque bytes, never an exception.

    `probe` runs on every file that is compressed, so a parser that raises on
    a malformed or merely unfamiliar file would take the whole archive with
    it. ONNX has no magic number either, so this parser is offered every file
    the earlier ones declined.
    """
    good = _onnx_model([_onnx_tensor("w", 1, [4], rand(16, 2))])
    assert planner.probe(io.BytesIO(good), len(good)).kind == "onnx"

    cases = {
        "empty": b"",
        "not a protobuf": b"\xff\xff\xff\xff" + b"x" * 100,
        "starts right, says nothing": b"\x08" + b"\x00" * 50,
        "truncated mid-field": (_pb_int(1, 8) + _pb_varint(7 << 3 | 2)
                                + _pb_varint(1000) + b"short"),
        "no initializers": _pb_int(1, 8) + _pb_len(7, _pb_len(2, b"main")),
        "initializer with no data": _onnx_model(
            [_onnx_tensor("w", 1, [4], b"")]),
        "length prefix past the file": (_pb_int(1, 8) + _pb_varint(7 << 3 | 2)
                                        + _pb_varint(1 << 40) + b"x" * 10),
        "deprecated group wire type": (_pb_int(1, 8)
                                       + _pb_len(7, _pb_varint(5 << 3 | 3)
                                                 + b"\x00")),
    }
    for name, data in cases.items():
        assert planner.probe(io.BytesIO(data), len(data)).kind == "raw", name

    # External-data tensors name a sidecar instead of carrying raw_data. There
    # is nothing in the file to address, so it is not an ONNX layout here.
    external = _pb_int(1, 8) + _pb_len(7, _pb_len(
        5, _pb_int(2, 1) + _pb_len(8, b"w") + _pb_len(12, b"weights.bin")))
    assert planner.probe(io.BytesIO(external), len(external)).kind == "raw"

    # A dtype this build does not know, and a payload that is not a whole
    # number of elements, both fall back to untyped bytes -- still addressed,
    # just not split.
    for model in (_onnx_model([_onnx_tensor("w", 999, [4], b"abcd" * 4)]),
                  _onnx_model([_onnx_tensor("w", 1, [1], b"abc")])):
        layout = planner.probe(io.BytesIO(model), len(model))
        assert layout.kind == "onnx"
        assert layout.tensors["w"]["dtype"] == "U8"
        assert all(r.esize == 1 for r in layout.regions)


def test_onnx_round_trips_through_an_archive():
    """The parse has to survive the whole path, not just the planner."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.onnx")
        # fp16 weights with the structure real ones have: an exponent that
        # occupies a handful of values and a mantissa that does not. Random
        # bytes would only prove the parse survives, not that it helps.
        payloads = []
        for i in range(6):
            noise = rand(1 << 16, i + 1)
            payloads.append(bytes(
                b if k % 2 == 0 else 0x38 | (b & 7)
                for k, b in enumerate(noise)))
        model = _onnx_model([
            _onnx_tensor(f"layer{i}.weight", 10, [16, 2048], p)
            for i, p in enumerate(payloads)])
        with open(path, "wb") as fh:
            fh.write(model)

        archive = os.path.join(d, "m.lmz")
        lmz.compress(path, archive)
        out = os.path.join(d, "out.onnx")
        lmz.decompress(archive, out)
        assert digest(out) == digest(path)

        info = lmz.info(archive)
        assert info["members"][0]["kind"] == "onnx"
        assert len(info["members"][0]["tensors"]) == 6

        # Typed beats opaque, which is the whole reason to parse the
        # container: without the element width the byte-position planes
        # cannot phase-align, and an fp16 exponent smears across positions
        # instead of collecting in one plane. Measured against the same bytes
        # with the parser switched off.
        real = planner.parse_onnx
        try:
            planner.parse_onnx = lambda f, size: None
            blob = os.path.join(d, "blob.lmz")
            lmz.compress(path, blob)
            assert planner.probe(io.BytesIO(model), len(model)).kind == "raw"
        finally:
            planner.parse_onnx = real
        assert os.path.getsize(archive) < os.path.getsize(blob), (
            os.path.getsize(archive), os.path.getsize(blob))


def test_int8_weights_are_coded_with_rans_not_zstd():
    """int8 planes must reach the coder the GPU can decode.

    A 1-byte element has no plane split, so an int8 tensor takes the general
    path. rANS used to be priced there only where zstd had failed to dent the
    stream by 1/32 -- a bar quantised weights never clear, because a
    per-channel scale maps each filter's peak to 127 and leaves the body
    concentrated enough for zstd to save a comfortable tenth. So rANS was
    never tried, and `lmz.gpu`, which decodes rANS and nothing else, could
    never touch an int8 archive.

    The size difference is small either way. The point is which decoder can
    read the result.
    """
    if not kernels.have_rans():
        raise Skip("no native rANS kernel")

    # A per-channel int8 weight plane: symmetric and peaked, but with no long
    # repeats -- the distribution is the only structure, which is exactly the
    # case a symbol coder wins and a match finder does not. Mapping distinct
    # random bytes through a bell curve keeps every byte independent.
    n = 1 << 20
    bell = bytes(max(0, min(255, int(128 + 34 * math.sqrt(-2 * math.log(
        (i + 0.5) / 256)) * math.cos(i * 2.39996))))
        for i in range(256))
    plane = rand(n, 9).translate(bell)

    used, payload = codec._encode_stream(plane, 1, entropy.METHOD_ZSTD)
    assert used == entropy.METHOD_RANS, entropy.METHOD_NAMES.get(used)
    assert len(payload) < n
    assert bytes(entropy.decompress(payload, used, n)) == plane

    # Where a general-purpose coder genuinely wins -- long repeats, which are
    # matches rather than symbol statistics -- it must still be chosen, and
    # that holds whichever general coder this interpreter has. Naming zstd
    # here would only test which Python is running: before 3.14 without the
    # package there is none, and the fallback is deflate.
    repetitive = b"the quick brown fox jumps over the lazy dog. " * 20000
    used, out = codec._encode_stream(repetitive, 1, entropy.METHOD_ZSTD)
    assert used in (entropy.METHOD_ZSTD, entropy.METHOD_DEFLATE), \
        entropy.METHOD_NAMES.get(used)
    # rANS on this stream is ~500 KB against deflate's ~5 KB, so choosing it
    # would be a 90x regression rather than a close call.
    assert len(out) < len(repetitive) // 100

    # And incompressible bytes must still be stored, without either coder
    # being run to find that out.
    used, out = codec._encode_stream(rand(1 << 20, 77), 1, entropy.METHOD_ZSTD)
    assert used == entropy.METHOD_STORED


def test_a_missing_general_coder_does_not_hand_everything_to_rans():
    """Without zstd, the fallback must still be priced against rANS.

    Before 3.14 and without the `zstandard` package there is no zstd, so a
    caller asking for it gets UnsupportedMethod. That used to leave the
    general path with no candidate at all, and the branch that takes rANS
    when nothing else worked would then accept it unpriced -- on text with
    long repeats that is ~500 KB against deflate's ~5 KB, a ninety-fold
    regression on exactly the interpreters that already compress worst.
    """
    if not kernels.have_rans():
        raise Skip("no native rANS kernel")

    repetitive = b"the quick brown fox jumps over the lazy dog. " * 20000
    saved = (entropy._zstd_compress, entropy.HAVE_ZSTD, entropy.DEFAULT_METHOD)
    try:
        entropy._zstd_compress = None
        entropy.HAVE_ZSTD = False
        entropy.DEFAULT_METHOD = entropy.METHOD_DEFLATE
        used, out = codec._encode_stream(repetitive, 1, entropy.METHOD_ZSTD)
        assert used == entropy.METHOD_DEFLATE, entropy.METHOD_NAMES.get(used)
        assert len(out) < len(repetitive) // 100, len(out)
    finally:
        (entropy._zstd_compress, entropy.HAVE_ZSTD,
         entropy.DEFAULT_METHOD) = saved

    # And with zstd back, the same stream picks it up again.
    used, _ = codec._encode_stream(repetitive, 1, entropy.METHOD_ZSTD)
    assert used in (entropy.METHOD_ZSTD, entropy.METHOD_DEFLATE)


def test_shared_table_streams_are_ordinary_streams():
    """A shared-table stream must decode through the untouched decoder.

    The whole basis of the shared form is that lifting the 516-byte header out
    of a stream leaves the coded bytes unchanged: prepend the table again and
    it is an ordinary lmz stream. That makes `lmz_rans_decode` -- which knows
    nothing about any of this -- the oracle, so there is no second decoder to
    keep in step with the first.
    """
    if not kernels.have_rans():
        raise Skip("no native rANS kernel")

    plane, nstr = 8192, 24
    # Peaked, the way a quantised weight plane is: a shared table is only
    # interesting where the streams resemble each other.
    table_map = bytes((abs(i - 128) // 3 + 100) % 256 for i in range(256))
    segs = [rand(plane, seed + 1).translate(table_map) for seed in range(nstr)]

    total = [0] * 256
    for seg in segs:
        for i, c in enumerate(kernels.histogram(seg)):
            total[i] += c
    shared = kernels.rans_table(total)
    assert shared is not None and len(shared) == kernels.RANS_HEADER
    assert shared[:2] == b"R1"

    per_stream = headerless = 0
    for seg in segs:
        own = kernels.rans_encode(seg, kernels.histogram(seg))
        coded = kernels.rans_encode_shared(seg, shared)
        assert coded is not None
        per_stream += len(own)
        headerless += len(coded)
        # The oracle, and the direct path, must both give the bytes back.
        assert bytes(kernels.rans_decode(shared + coded, plane)) == seg
        assert bytes(kernels.rans_decode_shared(coded, plane, shared)) == seg

    # 516 bytes a stream, paid once instead of nstr times. The coded bytes
    # themselves are not identical to the per-stream form -- a shared table
    # holds slightly different probabilities, so it codes slightly worse per
    # stream and wins by not repeating the header.
    assert headerless + len(shared) < per_stream

    # Coding a stream against its own table is exactly the ordinary form with
    # the header lifted off, which is the property the oracle above rests on.
    lone = kernels.histogram(segs[0])
    assert (len(kernels.rans_encode(segs[0], lone))
            == len(kernels.rans_encode_shared(segs[0], kernels.rans_table(lone)))
            + kernels.RANS_HEADER)


def test_shared_table_refuses_what_it_cannot_code():
    """A table that cannot represent a symbol must refuse, not mis-code.

    A shared table comes from counts the caller gathered, and a caller that
    gathers them over the wrong set produces a table with a zero frequency for
    a symbol that really occurs. The coder cannot represent it; encoding
    anyway would emit a stream that decodes to different bytes, which is the
    one outcome an archiver must never produce.
    """
    if not kernels.have_rans():
        raise Skip("no native rANS kernel")

    partial = kernels.rans_table(kernels.histogram(bytes([1, 2, 3]) * 400))
    assert partial is not None
    assert kernels.rans_encode_shared(bytes([1, 2, 3]) * 400, partial) is not None
    # 200 never appeared in the counts, so it has no frequency to code with.
    assert kernels.rans_encode_shared(bytes([1, 2, 200]) * 400, partial) is None

    # A table whose frequencies do not sum to the coder's scale is malformed
    # in both directions.
    broken = bytearray(partial)
    broken[4] = (broken[4] + 1) & 0xFF
    assert kernels.rans_encode_shared(bytes([1, 2, 3]) * 400, bytes(broken)) is None
    try:
        kernels.rans_decode_shared(b"\0" * 64, 64, bytes(broken))
    except ValueError:
        pass
    else:
        raise AssertionError("a malformed table must not decode")

    try:
        kernels.rans_encode_shared(b"abc" * 100, partial[:-1])
    except ValueError:
        pass
    else:
        raise AssertionError("a short table must be rejected")

    # One symbol at the full probability scale is the edge the stream header
    # biases for; it has to survive the shared path too.
    one = kernels.rans_table(kernels.histogram(b"\x07" * 5000))
    coded = kernels.rans_encode_shared(b"\x07" * 5000, one)
    assert bytes(kernels.rans_decode_shared(coded, 5000, one)) == b"\x07" * 5000


def _shared_model(path, tensors=24, nelem=12000):
    """A checkpoint whose tensors sit at different scales, as layers do."""
    write_safetensors(path, [
        (f"layer.{i}.weight", "F32", [nelem],
         weights_f32_from_f16(nelem, seed=i + 1, band=(i % 7)))
        for i in range(tensors)])


def test_shared_tables_shrink_a_page_mapped_archive():
    """The fixed cost a small archive cannot amortise, actually removed.

    A 64 KiB block archive pays 516 bytes of frequency table per stream, and
    on a model of a few megabytes that is a material fraction of the payload.
    This is the whole justification for the format option, so it is asserted
    as a size, not just as a round trip.
    """
    if not kernels.have_rans():
        raise Skip("no native rANS kernel")
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "m.safetensors")
        _shared_model(src)
        before = digest(src)

        sizes = {}
        for tag, shared in (("own", False), ("shared", True)):
            arc = os.path.join(d, f"{tag}.lmz")
            out = os.path.join(d, f"out-{tag}")
            lmz.compress(src, arc, mapped=True, shared_tables=shared)
            lmz.decompress(arc, out, overwrite=True)
            assert digest(out) == before, f"{tag} did not round-trip"
            sizes[tag] = os.path.getsize(arc)

        assert sizes["shared"] < sizes["own"], (
            f"shared tables did not pay: {sizes['shared']} vs {sizes['own']}")


def test_shared_tables_are_decided_per_plane():
    """Pooling wins on some planes of a file and loses on others.

    An fp32 checkpoint upcast from fp16 is the case: its low byte planes are
    near-constant and pool perfectly, while its high byte carries a per-tensor
    scale and pools badly. An all-or-nothing rule would throw away the planes
    that did win, so the manifest is expected to hold *some* kinds and not
    all of them -- and every chunk must still round-trip.
    """
    if not kernels.have_rans():
        raise Skip("no native rANS kernel")
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "m.safetensors")
        _shared_model(src)
        arc = os.path.join(d, "a.lmz")
        lmz.compress(src, arc, mapped=True, shared_tables=True)
        with open(arc, "rb") as fh:
            reader = lmzformat.ArchiveReader(fh)
            kinds = set((reader.manifest.get("tables") or {}))
            codecs = {c.codec for c in reader.chunks}
        assert kinds, "no plane kind was worth a shared table"
        assert kinds != {f"{lmzformat.CODEC_SPLIT}:4:{k}" for k in range(4)}, \
            "every plane pooled; this fixture no longer tests the mixed case"
        assert lmzformat.CODEC_SPLIT_ST in codecs, \
            "tables were written but no chunk used them"


def test_shared_table_chunk_needs_its_manifest():
    """A v7 chunk without its tables is corruption, not a wrong answer.

    The coded bytes carry no frequencies of their own, so a reader that has
    lost the manifest's table set cannot decode them and must say so rather
    than produce plausible garbage.
    """
    if not kernels.have_rans():
        raise Skip("no native rANS kernel")
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "m.safetensors")
        _shared_model(src)
        arc = os.path.join(d, "a.lmz")
        lmz.compress(src, arc, mapped=True, shared_tables=True)
        with open(arc, "rb") as fh:
            reader = lmzformat.ArchiveReader(fh)
            chunk = next(c for c in reader.chunks
                         if c.codec == lmzformat.CODEC_SPLIT_ST)
            fh.seek(chunk.off)
            payload = fh.read(chunk.clen)
        try:
            codec.decode_chunk(payload, chunk.codec, chunk.esize, chunk.flags,
                               chunk.rlen, chunk.crc, True, tables=None)
        except lmzformat.CorruptArchive:
            pass
        else:
            raise AssertionError("a shared-table chunk decoded without tables")


def test_version_stamp_follows_the_codecs_used():
    """The stamp is a requirement on the reader, not a build number.

    Writing the newest version unconditionally would make every ordinary
    archive this build produces unreadable to the previous release, which can
    decode all of it. Only an archive that really uses a v7 codec is v7.
    """
    if not kernels.have_rans():
        raise Skip("no native rANS kernel")
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "m.safetensors")
        _shared_model(src)
        for shared, want in ((False, lmzformat.BASE_VERSION), (True, 7)):
            arc = os.path.join(d, f"v{int(shared)}.lmz")
            lmz.compress(src, arc, mapped=True, shared_tables=shared)
            with open(arc, "rb") as fh:
                _magic, ver = lmzformat.HEADER.unpack(
                    fh.read(lmzformat.HEADER_SIZE))[:2]
            assert ver == want, f"shared={shared} stamped v{ver}, wanted v{want}"


def test_primer_declines_when_pooling_loses():
    """The decision is a measurement, and it has to be able to say no.

    Streams drawn from unrelated distributions cost more against one pooled
    table than they do against their own, header included. A primer that
    kept a table there would make the archive bigger, which is the failure
    mode the whole-archive table showed on real models.
    """
    if not kernels.have_rans():
        raise Skip("no native rANS kernel")
    alike, unlike = freqs.Primer(), freqs.Primer()
    for i in range(16):
        same = bytes((j * 7 + 3) % 32 for j in range(4096))
        alike.add("2:4:0", kernels.histogram(same))
        # Each stream a narrow band of its own, disjoint from its neighbours'.
        band = bytes((i * 16 + (j % 16)) & 0xFF for j in range(4096))
        unlike.add("2:4:0", kernels.histogram(band))
    assert alike.build(), "identical streams must pool"
    assert not unlike.build(), "disjoint streams must not pool"


def test_stride_kernels():
    """The strided split must equal slicing and merge must undo it."""
    for period in (1, 2, 17, 34, 64, 144, 210, 256):
        for nblocks in (0, 1, 7, 100, 4097):
            src = rand(nblocks * period, nblocks + period)
            planes = kernels.split_stride(src, period)
            expect = b"".join(src[k::period] for k in range(period))
            assert bytes(planes) == expect, f"split period={period} n={nblocks}"
            streams = [(bytes(planes), k * nblocks) for k in range(period)]
            back = kernels.merge_stride(streams, nblocks, period)
            assert bytes(back) == src, f"merge period={period} n={nblocks}"
    for bad in (0, 257):
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


def test_block_layouts_tile_their_blocks():
    """Every GGUF field layout must cover its block exactly, once."""
    for ttype, (period, groups) in planner.BLOCK_LAYOUTS.items():
        name, _blk, per_blk, _es = planner.GGML_TYPES[ttype]
        assert period == per_blk, f"{name}: layout says {period}, table says {per_blk}"
        assert period <= kernels.MAX_PERIOD, f"{name}: period {period} exceeds kernel"
        pos = 0
        for start, width in groups:
            assert start == pos and width > 0, f"{name}: field at {start} leaves a gap"
            pos += width
        assert pos == period, f"{name}: fields cover {pos} of {period} bytes"


def test_block_codec_roundtrip():
    """Every block quantisation must choose the block codec and restore exactly."""
    if not kernels.have_rans():
        return
    for ttype in (8, 10, 11, 12, 13, 14, 23):  # Q8_0, Q2_K..Q6_K, IQ4_XS
        period = planner.BLOCK_LAYOUTS[ttype][0]
        name = planner.GGML_TYPES[ttype][0]
        data = quant_blocks(ttype, codec.BLOCK_MIN_BLOCKS * 4, ttype)
        parts, cid, flags, crc = codec.encode_chunk(data, period, 1, True,
                                                    kind=planner.KIND_BLOCK,
                                                    btype=ttype)
        assert cid == lmzformat.CODEC_GBLK, f"{name}: got codec {cid}"
        payload = b"".join(bytes(p) for p in parts)
        assert len(payload) < len(data) * 0.95, f"{name}: block codec should win"
        got = codec.decode_chunk(payload, cid, period, flags, len(data), crc, True)
        assert bytes(got) == data, f"{name}: round-trip differs"

        # Damage must never pass as good data. It may be rejected, or it may
        # land on a byte no decoder ever consults -- an interleaved rANS stream
        # ends with a refill whose value cannot reach a symbol. What must not
        # happen is a clean return of the wrong bytes.
        head = codec._gblk_hdr.size + len(planner.BLOCK_LAYOUTS[ttype][1]) * 4
        for offset in [1, 3, head, head + 7] + [
                len(payload) * k // 11 for k in range(1, 11)]:
            damaged = bytearray(payload)
            damaged[offset] ^= 0xFF
            try:
                got = codec.decode_chunk(bytes(damaged), cid, period, flags,
                                         len(data), crc, True)
            except FormatError:
                continue
            assert bytes(got) == data, (
                f"{name}: damage at {offset} decoded to wrong bytes silently")


def test_block_codec_uses_every_group_mode():
    """The three field modes are chosen by measurement, so all must be reachable."""
    if not kernels.have_rans():
        return
    seen = set()
    for ttype in (8, 12, 14):
        period, groups = planner.BLOCK_LAYOUTS[ttype]
        data = quant_blocks(ttype, codec.BLOCK_MIN_BLOCKS * 8, ttype)
        parts, cid, _flags, _crc = codec.encode_chunk(data, period, 1, True,
                                                      kind=planner.KIND_BLOCK,
                                                      btype=ttype)
        assert cid == lmzformat.CODEC_GBLK
        payload = b"".join(bytes(p) for p in parts)
        pos = codec._gblk_hdr.size
        for _ in groups:
            _start, _width, mode = codec._gblk_grp.unpack_from(payload, pos)
            pos += codec._gblk_grp.size
            seen.add(mode)
    assert seen == {codec.GRP_CONCAT, codec.GRP_PLANES, codec.GRP_COND}, seen


def test_block_codec_declines_on_noise():
    """Blocks with no structure must fall through to plain storage, not the split."""
    if not kernels.have_rans():
        return
    period = planner.BLOCK_LAYOUTS[12][0]
    data = rand(period * codec.BLOCK_MIN_BLOCKS * 2, 99)
    parts, cid, flags, crc = codec.encode_chunk(data, period, 1, True,
                                                kind=planner.KIND_BLOCK, btype=12)
    assert cid != lmzformat.CODEC_GBLK, "noise must not pay for a block split"
    payload = b"".join(bytes(p) for p in parts)
    assert bytes(codec.decode_chunk(payload, cid, period, flags, len(data),
                                    crc, True)) == data


def test_zstd_backend_preference_does_not_change_what_is_readable():
    """Which binding wins varies by platform; what it writes must not.

    Both are libzstd, so the choice is only about speed: CPython's bundled build
    measured roughly 7x slower than the same library through the `zstandard`
    package on macOS, which is why the preference flips there. An archive has to
    stay readable either way, so where both are installed their frames must be
    mutually decodable -- otherwise the preference would be a format decision
    wearing a performance disguise.
    """
    from unittest import mock

    with mock.patch.object(sys, "platform", "darwin"):
        assert [f.__name__ for f in entropy._backend_order()] == \
            ["_load_package", "_load_stdlib"], "macOS must prefer the package"
    for other in ("linux", "win32"):
        with mock.patch.object(sys, "platform", other):
            assert [f.__name__ for f in entropy._backend_order()] == \
                ["_load_stdlib", "_load_package"], f"{other} must prefer stdlib"

    loaded = []
    for loader in (entropy._load_stdlib, entropy._load_package):
        try:
            loaded.append(loader())
        except ImportError:
            pass          # only one binding installed here, which is normal
    if not loaded:
        raise Skip("no zstd binding is available")
    data = (bytes(range(256)) * 512) + b"contents number 0\n" * 100
    for comp, _d, _e, name in loaded:
        frame = comp(data, 3)
        for _c, dec, _e2, reader in loaded:
            assert dec(frame) == data, f"{reader} could not read {name}'s frame"


def test_method_tally_counts_only_the_attempt_that_won():
    """Every input byte is credited to exactly one coder, once.

    encode_chunk tries several codings and keeps the smallest: a block split, a
    conditioned BF16 chunk, a plain plane split. An attempt that loses has
    already coded its streams, and those streams never reach the archive, so
    counting them would credit a coder for work that was thrown away and would
    describe the search rather than the file. Summing the tally back to the
    exact input size is precisely the assertion that this did not happen -- a
    committed loser shows up as roughly twice the bytes.
    """
    if not kernels.have_rans():
        raise Skip("the tally distinguishes coders only when rANS is available")
    period = planner.BLOCK_LAYOUTS[12][0]
    nblk = codec.BLOCK_MIN_BLOCKS * 2
    cases = [
        # A block split that is attempted and declined: the interesting one.
        ("gblk declined", rand(period * nblk, 99), period, planner.KIND_BLOCK, 12),
        ("gblk kept", quant_blocks(12, nblk), period, planner.KIND_BLOCK, 12),
        ("bf16 conditioned", weights_bf16_cond(1 << 20, 3), 2, planner.KIND_BF16, -1),
        ("bf16 plain", weights_bf16(400000, 7), 2, planner.KIND_BF16, -1),
        ("byte split", weights_bf16(400000, 7), 2, planner.KIND_BYTES, -1),
        ("noise", rand(1 << 20, 11), 2, planner.KIND_BYTES, -1),
        ("all zeros", b"\x00" * (1 << 20), 4, planner.KIND_BYTES, -1),
    ]
    for label, data, esize, kind, btype in cases:
        tally = codec.MethodTally()
        codec.encode_chunk(data, esize, 1, False, kind=kind, btype=btype,
                           tally=tally)
        rows = tally.to_json()
        raw = sum(r["raw"] for r in rows.values())
        assert raw == len(data), f"{label}: {raw} bytes tallied for {len(data)}"
        assert all(r["streams"] > 0 for r in rows.values()), (label, rows)


def test_method_tally_measures_the_losing_coder():
    """measure_alt has to record a size the winner actually beat.

    The plane path runs rANS alone, so the archive records what the winner cost
    and nothing about what the alternative would have. Asking for the loser to
    be coded too is the only way to answer "by how much", and the answer is
    only meaningful if the winner is genuinely no larger.
    """
    if not kernels.have_rans() or not entropy.HAVE_ZSTD:
        raise Skip("needs both coders to have a contest")
    tally = codec.MethodTally(measure_alt=True)
    data = weights_bf16(400000, 7)
    codec.encode_chunk(data, 2, 1, False, kind=planner.KIND_BF16, tally=tally)
    rows = tally.to_json()
    rans = rows.get("rans")
    assert rans and rans.get("contested"), f"no contested streams: {rows}"
    assert rans["contested_coded"] <= rans["contested_alt"], \
        f"the chosen coder was larger: {rans}"
    # And without asking, nothing is contested: the loser is never coded.
    quiet = codec.MethodTally()
    codec.encode_chunk(data, 2, 1, False, kind=planner.KIND_BF16, tally=quiet)
    assert not quiet.to_json().get("rans", {}).get("contested"), \
        "the alternative was coded without being asked for"


def test_info_reports_which_coder_did_the_work():
    """The tally has to survive into the archive, since that is where it is read.

    A chunk's codec is the framing, not the coder: a bf16-split chunk says
    nothing about whether rANS or zstd earned its bytes. The manifest carries
    the answer so `lmz info` can report it without decompressing anything.
    """
    if not kernels.have_rans():
        raise Skip("the tally distinguishes coders only when rANS is available")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.safetensors")
        write_safetensors(path, [
            ("w1", "BF16", [1024, 1024], weights_bf16(1024 * 1024)),
            ("w2", "I8", [8192], rand(8192, 13)),
        ])
        archive = os.path.join(d, "m.lmz")
        stats = lmz.compress(path, archive)
        assert stats.detail.get("methods"), "compress reported no method tally"

        methods = lmz.api.info(archive)["methods"]
        assert methods, "the archive did not carry the tally"
        assert methods.get("rans", {}).get("raw"), \
            f"rANS coded nothing on BF16 weights: {methods}"
        # Same invariant as at chunk level, now across the whole archive.
        raw = sum(m["raw"] for m in methods.values())
        assert raw == os.path.getsize(path), \
            f"{raw} bytes tallied for a {os.path.getsize(path)} byte input"


def _k4_quant(v: int, scale: int) -> int:
    """A 4-bit quant whose spread follows its own sub-block's scale."""
    span = 1 + scale * 7 // 63
    return max(0, min(15, 8 + (v % (2 * span + 1)) - span))


def k4_blocks(nblocks: int, ttype: int = 12, seed: int = 5) -> bytes:
    """k-quant super-blocks whose quants really depend on their sub-block.

    Built the way ggml builds them: eight 6-bit scales and eight 6-bit mins
    packed by get_scale_min_k4's own straddling bit layout, and quants
    interleaved two sub-blocks to a byte. A sub-block with a small scale holds
    a narrower alphabet, which is the structure the sub-block mode collects
    and which no byte-plane split can see.
    """
    period, groups = planner.BLOCK_LAYOUTS[ttype]
    qstart = next(s for s, w in groups if w == 128)
    x = seed | 1

    def nxt() -> int:
        nonlocal x
        x = (x * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)
        return x >> 11

    out = bytearray()
    for _ in range(nblocks):
        blk = bytearray(period)
        for off in (0, 2):  # d, dmin: an fp16 with a narrow exponent range
            blk[off] = nxt() & 0xFF
            blk[off + 1] = 0x20 | (nxt() & 0x0F)

        sc = [nxt() % 64 for _ in range(8)]
        mn = [nxt() % 64 for _ in range(8)]
        s = bytearray(12)
        for j in range(4):
            s[j] = sc[j]
            s[j + 4] = mn[j]
        for j in range(4, 8):
            s[j + 4] = (sc[j] & 0x0F) | ((mn[j] & 0x0F) << 4)
            s[j - 4] |= ((sc[j] >> 4) & 0x3) << 6
            s[j] |= ((mn[j] >> 4) & 0x3) << 6
        blk[4:16] = s

        for i in range(16, qstart):  # Q5_K's qh sits between scales and quants
            blk[i] = nxt() & 0xFF
        for g in range(4):
            for t in range(32):
                blk[qstart + 32 * g + t] = (_k4_quant(nxt(), sc[2 * g])
                                            | (_k4_quant(nxt(), sc[2 * g + 1]) << 4))
        out += blk
    assert len(out) == nblocks * period
    return bytes(out)


def block_modes(payload: bytes):
    """Parse a block payload's field table: {start: (width, mode, sub)}."""
    _period, ngroups = codec._gblk_hdr.unpack_from(payload, 0)
    pos = codec._gblk_hdr.size
    out = {}
    for _ in range(ngroups):
        start, width, mode = codec._gblk_grp.unpack_from(payload, pos)
        pos += codec._gblk_grp.size
        sub = None
        if mode == codec.GRP_SUB:
            sub = codec._gblk_sub.unpack_from(payload, pos)
            pos += codec._gblk_sub.size
        out[start] = (width, mode, sub, pos)
    return out


def encode_block_chunk(data, ttype, subctx=True):
    period = planner.BLOCK_LAYOUTS[ttype][0]
    saved = codec.SUBBLOCK_CTX
    if not subctx:
        codec.SUBBLOCK_CTX = {}
    try:
        parts, cid, flags, crc = codec.encode_chunk(
            data, period, 1, True, kind=planner.KIND_BLOCK, btype=ttype)
    finally:
        codec.SUBBLOCK_CTX = saved
    return b"".join(bytes(p) for p in parts), cid, flags, crc


_K4_KERNEL_PROBE = (
    "from lmz import kernels;"
    "out=[]\n"
    "for nb in (0,1,3,37,260):\n"
    "    sp=bytes((i*37+nb)%256 for i in range(12*nb));"
    "q=bytes((i*89+nb*5)%256 for i in range(128*nb));"
    "sc,mn=kernels.k4_scales(sp,nb);"
    "pk=kernels.k4_pack(q,nb);\n"
    "    assert bytes(kernels.k4_unpack(pk,nb))==q, 'pack is not invertible'\n"
    "    out.append((bytes(sc)+bytes(mn)+bytes(pk)).hex())\n"
    "print(':'.join(out))"
)


def test_k4_kernels_agree_and_reverse():
    """The sub-block kernels must agree across backends and undo themselves."""
    script = ("import os,sys;os.environ['LMZ_NO_NATIVE']='1';"
              "sys.path.insert(0,%r);"
              "from lmz import kernels as _k;"
              "assert not _k.backend().startswith('native');" % ROOT
              ) + _K4_KERNEL_PROBE
    out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, cwd=ROOT)
    assert out.returncode == 0, out.stderr

    native = subprocess.run(
        [sys.executable, "-c", "import sys;sys.path.insert(0,%r);" % ROOT
         + _K4_KERNEL_PROBE],
        capture_output=True, text=True, cwd=ROOT)
    assert native.returncode == 0, native.stderr
    assert out.stdout == native.stdout, "k4 kernel backends disagree"

    # and the unpacking must be ggml's get_scale_min_k4, value for value
    nb = 64
    planes = [bytes(((i * 37 + j * 11) & 0xFF) for i in range(nb)) for j in range(12)]
    sc, mn = kernels.k4_scales(b"".join(planes), nb)
    for i in range(nb):
        q = [planes[j][i] for j in range(12)]
        for j in range(8):
            if j < 4:
                d, m = q[j] & 63, q[j + 4] & 63
            else:
                d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4)
                m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4)
            assert sc[j * nb + i] == d and mn[j * nb + i] == m, "not get_scale_min_k4"


def test_subblock_mode_is_chosen_and_reverses():
    """Quants that depend on their sub-block take the sub mode, and come back."""
    if not kernels.have_rans():
        return
    for ttype in sorted(planner.SUBBLOCK_CTX):
        name = planner.GGML_TYPES[ttype][0]
        period = planner.BLOCK_LAYOUTS[ttype][0]
        qstart = planner.SUBBLOCK_CTX[ttype][0]
        data = k4_blocks(codec.BLOCK_MIN_BLOCKS * 4, ttype, ttype)

        payload, cid, flags, crc = encode_block_chunk(data, ttype)
        assert cid == lmzformat.CODEC_GBLK, f"{name}: got codec {cid}"
        modes = block_modes(payload)
        assert modes[qstart][1] == codec.GRP_SUB, f"{name}: quants took another mode"
        got = codec.decode_chunk(payload, cid, period, flags, len(data), crc, True)
        assert bytes(got) == data, f"{name}: round-trip differs"

        # and it must be chosen because it wins, not merely because it is offered
        plain, _c, _f, _r = encode_block_chunk(data, ttype, subctx=False)
        assert len(payload) < len(plain), (
            f"{name}: sub mode cost {len(payload)} against {len(plain)}")


def test_subblock_mode_spans_a_gap_before_the_quants():
    """A context field need not sit immediately before the field it conditions.

    Q5_K puts qh between its scales and its quants. It is not registered --
    on real weights it gains 0.000 points, because qs holds only the low four
    bits of a five-bit quant -- but the payload describes whatever layout it
    used, so the path has to work and is exercised here.
    """
    if not kernels.have_rans():
        return
    ttype, qstart = 13, 48
    period = planner.BLOCK_LAYOUTS[ttype][0]
    data = k4_blocks(codec.BLOCK_MIN_BLOCKS * 4, ttype, ttype)
    saved = codec.SUBBLOCK_CTX
    codec.SUBBLOCK_CTX = {ttype: (qstart, 4, planner.SUB_K4)}
    try:
        payload, cid, flags, crc = encode_block_chunk(data, ttype)
    finally:
        codec.SUBBLOCK_CTX = saved
    assert block_modes(payload)[qstart][1] == codec.GRP_SUB
    got = codec.decode_chunk(payload, cid, period, flags, len(data), crc, True)
    assert bytes(got) == data, "Q5_K layout round-trip differs"


def test_subblock_mode_declines_without_structure():
    """Quants unrelated to their sub-block must not pay for the extra tables."""
    if not kernels.have_rans():
        return
    for ttype in sorted(planner.SUBBLOCK_CTX):
        qstart = planner.SUBBLOCK_CTX[ttype][0]
        data = quant_blocks(ttype, codec.BLOCK_MIN_BLOCKS * 4, ttype)
        payload, cid, _flags, _crc = encode_block_chunk(data, ttype)
        if cid != lmzformat.CODEC_GBLK:
            continue
        assert block_modes(payload)[qstart][1] != codec.GRP_SUB, (
            f"{planner.GGML_TYPES[ttype][0]}: sub mode taken with nothing to gain")


def test_subblock_descriptor_is_validated():
    """A damaged sub-block descriptor must be rejected, never acted on."""
    if not kernels.have_rans():
        return
    ttype, period = 12, planner.BLOCK_LAYOUTS[12][0]
    qstart = planner.SUBBLOCK_CTX[ttype][0]
    data = k4_blocks(codec.BLOCK_MIN_BLOCKS * 4, ttype, ttype)
    payload, cid, flags, crc = encode_block_chunk(data, ttype)
    end = block_modes(payload)[qstart][3]
    off = end - codec._gblk_sub.size
    good = codec._gblk_sub.unpack_from(payload, off)

    # ctx_start past the quants would let a field read lanes it has not decoded
    for field, value in ((0, qstart), (0, 200), (1, 11), (2, 7), (3, 1),
                         (4, 0), (5, 0), (4, 31), (5, 31)):
        bad = list(good)
        bad[field] = value
        damaged = bytearray(payload)
        damaged[off:off + codec._gblk_sub.size] = codec._gblk_sub.pack(*bad)
        try:
            got = codec.decode_chunk(bytes(damaged), cid, period, flags,
                                     len(data), crc, True)
        except FormatError:
            continue
        assert bytes(got) == data, (
            f"field {field}={value} decoded to wrong bytes silently")


def test_kquant_gguf_file_roundtrip():
    """A Q4_K GGUF must be block-coded through the whole pipeline."""
    if not kernels.have_rans():
        return
    nblocks = 40000
    raw = quant_blocks(12, nblocks, 7)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.gguf")
        write_gguf(path, [("blk.0.w", 12, [256 * nblocks], raw)])
        with open(path, "rb") as fh:
            layout = planner.probe(fh, os.path.getsize(path))
        region = next(r for r in layout.regions if r.kind == planner.KIND_BLOCK)
        assert region.esize == 144 and region.btype == 12

        archive = os.path.join(d, "a.lmz")
        stats = lmz.compress(path, archive, chunk_size=2 << 20)
        assert "blk-split" in lmz.info(archive)["codecs"]
        out = os.path.join(d, "o.gguf")
        lmz.decompress(archive, out)
        assert digest(out) == digest(path)
        assert stats.saved > 0.05, f"only saved {stats.saved:.2%}"


def test_q80_gguf_file_roundtrip():
    """A Q8_0 GGUF must be block-coded through the whole pipeline."""
    if not kernels.have_rans():
        return
    nblocks = 150000
    raw = quant_blocks(8, nblocks, 7)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.gguf")
        write_gguf(path, [("blk.0.w", 8, [32 * nblocks], raw)])
        with open(path, "rb") as fh:
            layout = planner.probe(fh, os.path.getsize(path))
        region = next(r for r in layout.regions if r.kind == planner.KIND_BLOCK)
        assert region.esize == 34 and region.btype == 8

        archive = os.path.join(d, "a.lmz")
        stats = lmz.compress(path, archive, chunk_size=2 << 20)
        assert "blk-split" in lmz.info(archive)["codecs"]
        out = os.path.join(d, "o.gguf")
        lmz.decompress(archive, out)
        assert digest(out) == digest(path)
        assert stats.saved > 0.05, f"only saved {stats.saved:.2%}"


def test_v3_q8_block_chunks_still_decode():
    """v3 wrote a Q8_0-only block payload; this build must still read it.

    Built here with every plane stored rather than entropy coded, which
    exercises the v3 header and plane order without needing the v3 encoder.
    """
    nblocks = 300
    data = quant_blocks(8, nblocks, 5)
    planes = bytes(kernels.split_stride(data, 34))
    lo, hi, quants = (planes[:nblocks], planes[nblocks:2 * nblocks],
                      planes[2 * nblocks:])
    stored = entropy.METHOD_STORED
    header = codec._q80_hdr.pack(0, stored, stored, *([stored] * 7), stored,
                                 nblocks, nblocks, *([0] * 7), len(quants))
    payload = header + hi + lo + quants
    got = codec.decode_chunk(payload, lmzformat.CODEC_BLK, 34, 0, len(data), 0,
                             False)
    assert bytes(got) == data


def test_opening_a_huge_index_stays_cheap():
    """Opening must not scale with the number of chunks.

    The chunk table used to become one Python object per chunk at open: for a
    70B checkpoint that is 70 MB of records turning into ~0.5 GB of objects
    and about five seconds, paid by every process before it read a byte, and
    it is why mounting a large store was expensive. The records are
    fixed-width, so the decompressed bytes are already the index.

    Sized so the old behaviour cannot pass: 400k chunks is roughly 100 MB of
    objects, against a 12.8 MB table.
    """
    from lmz.format import RECORD, ChunkTable

    n = 400_000
    table = b"".join(RECORD.pack(i * 65536, i * 65536, 65536, 65536, 0, 1, 2, 0)
                     for i in range(n))

    tracemalloc.start()
    try:
        before = tracemalloc.get_traced_memory()[0]
        t0 = time.perf_counter()
        chunks = ChunkTable(table)
        assert len(chunks) == n
        cost = tracemalloc.get_traced_memory()[0] - before
        elapsed = time.perf_counter() - t0
    finally:
        tracemalloc.stop()

    # Materialising even one field of every chunk would be megabytes; this is
    # a couple of attributes regardless of how many chunks there are.
    assert cost < 100_000, f"opening allocated {cost} bytes for {n} chunks"
    assert elapsed < 0.5, f"opening took {elapsed:.2f}s for {n} chunks"

    # And it still behaves like the list it replaced.
    assert chunks[0].dst == 0
    assert chunks[-1].dst == (n - 1) * 65536
    assert chunks[5].off == 5 * 65536
    assert chunks[-1] == chunks[n - 1]
    assert [c.dst for c in chunks[:3]] == [0, 65536, 131072]
    try:
        chunks[n]
    except IndexError:
        pass
    else:
        raise AssertionError("an out-of-range index must raise IndexError")

    # The column shortcut has to agree with walking the records, or the
    # aggregate paths that use it would report something different from the
    # ones that iterate.
    assert sum(chunks.column("clen")) == sum(c.clen for c in chunks)
    assert list(chunks.column("dst")) == [c.dst for c in chunks]


def test_chunk_order_survives_a_table_written_out_of_order():
    """Chunks are appended as workers finish, so dst order is not the file's.

    order_by_dst() skips the sort when the table is already ascending, which
    is the common case -- and a wrong sortedness test would silently hand back
    an unordered list, which every offset lookup downstream depends on.
    """
    from lmz.format import RECORD, ChunkTable

    order = [3, 0, 4, 1, 2]
    table = b"".join(RECORD.pack(1000 + i * 10, d * 65536, 10, 65536, 0, 1, 2, 0)
                     for i, d in enumerate(order))
    chunks = ChunkTable(table)

    assert [c.dst for c in chunks] == [d * 65536 for d in order]
    assert [c.dst for c in chunks.order_by_dst()] == [d * 65536 for d in sorted(order)]
    # The payload offset must travel with its record rather than the position.
    assert [c.off for c in chunks.order_by_dst()] == [1010, 1030, 1040, 1000, 1020]

    ascending = b"".join(RECORD.pack(100, d * 65536, 10, 65536, 0, 1, 2, 0)
                         for d in range(5))
    assert [c.dst for c in ChunkTable(ascending).order_by_dst()] == \
        [d * 65536 for d in range(5)]

    # Equal destinations are not ascending-violating, and a single record and
    # an empty table are the boundaries the zip-based check has to survive.
    same = b"".join(RECORD.pack(100, 0, 10, 0, 0, 1, 2, 0) for _ in range(3))
    assert len(ChunkTable(same).order_by_dst()) == 3
    assert len(ChunkTable(RECORD.pack(1, 2, 3, 4, 0, 1, 2, 0)).order_by_dst()) == 1
    assert ChunkTable(b"").order_by_dst() == []
    assert len(ChunkTable(b"")) == 0

    try:
        ChunkTable(b"\0" * (RECORD.size + 1))
    except FormatError:
        pass
    else:
        raise AssertionError("a partial record must be rejected")


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


# ------------------------------------------------------- the store and mount


def sample_model(root: str) -> str:
    """A small two-file model directory, one file in a subdirectory."""
    os.makedirs(os.path.join(root, "original"), exist_ok=True)
    write_safetensors(
        os.path.join(root, "model.safetensors"),
        [("embed", "BF16", [512, 64], weights_bf16(512 * 64, 3)),
         ("lm_head", "BF16", [512, 64], weights_bf16(512 * 64, 4))])
    write_safetensors(
        os.path.join(root, "original", "consolidated.safetensors"),
        [("w", "BF16", [256, 64], weights_bf16(256 * 64, 5))])
    with open(os.path.join(root, "config.json"), "w") as fh:
        json.dump({"model_type": "llama", "hidden_size": 64}, fh)
    return root


def test_store_add_list_and_remove():
    from lmz.store import Store

    with tempfile.TemporaryDirectory() as d:
        src = sample_model(os.path.join(d, "src"))
        store = Store(os.path.join(d, "store"))
        assert store.models() == []
        entry = store.add(src, "demo")
        assert entry.name == "demo" and entry.files == 3
        assert entry.stored_size < entry.original_size
        assert sorted(entry.members) == [
            "config.json", "model.safetensors",
            "original/consolidated.safetensors"]
        assert [e.name for e in store.models()] == ["demo"]

        # A second add under the same name needs force, and must not have
        # damaged the first one when it refuses.
        try:
            store.add(src, "demo")
            raise AssertionError("expected a refusal")
        except ValueError:
            pass
        assert store.get("demo").original_size == entry.original_size

        store.remove("demo")
        assert store.models() == []
        assert not os.path.exists(store.archive_path(entry))


def test_store_names_are_normalised():
    from lmz.store import normalise_name

    assert normalise_name("meta-llama/Llama-3.1-8B") == "meta-llama-Llama-3.1-8B"
    assert normalise_name("  spaced name  ") == "spaced-name"
    assert normalise_name("../../etc/passwd") == "etc-passwd"
    for bad in ("", "   ", "..", "/"):
        try:
            normalise_name(bad)
            raise AssertionError(f"expected {bad!r} to be refused")
        except ValueError:
            pass


def test_store_rebuild_recovers_the_index():
    from lmz.store import Store

    with tempfile.TemporaryDirectory() as d:
        src = sample_model(os.path.join(d, "src"))
        store = Store(os.path.join(d, "store"))
        store.add(src, "demo")
        os.unlink(store.index_path)
        assert store.models() == []
        rebuilt = store.rebuild()
        assert [e.name for e in rebuilt] == ["demo"]
        assert store.get("demo").files == 3


def test_store_reads_tensors_without_expanding():
    from lmz.store import Store

    with tempfile.TemporaryDirectory() as d:
        src = sample_model(os.path.join(d, "src"))
        store = Store(os.path.join(d, "store"))
        store.add(src, "demo")
        with open(os.path.join(src, "model.safetensors"), "rb") as fh:
            want = fh.read()
        with store.open("demo") as arc:
            assert arc.page_mapped, "a stored model must be page-mapped"
            got = arc.read_member("model.safetensors")
            assert got == want
            dtype, shape, raw = arc.tensor("embed")
            assert dtype == "BF16" and shape == [512, 64]
            assert raw == weights_bf16(512 * 64, 3)


def test_fuse_tree_builds_nested_paths():
    from lmz.fuse import Tree

    tree = Tree()
    tree.add_file("m/config.json", 12, ("m", 0))
    leaf = tree.add_file("m/original/w.safetensors", 34, ("m", 12))
    root = tree.nodes[1]
    assert set(root.children) == {"m"}
    model = root.children["m"]
    assert set(model.children) == {"config.json", "original"}
    assert model.children["original"].children["w.safetensors"] is leaf
    assert tree.total_size == 46
    assert tree.nodes[leaf.parent].name == "original"
    for bad in ("../escape", "m/../../etc/passwd", "", "."):
        try:
            tree.add_file(bad, 1, None)
            raise AssertionError(f"expected {bad!r} to be refused")
        except ValueError:
            pass


def test_coalesced_payload_reads_match():
    """Reading a run of blocks in one pread must equal reading them singly."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.safetensors")
        write_safetensors(path, [("w", "BF16", [1 << 17], weights_bf16(1 << 17, 9))])
        archive = os.path.join(d, "a.lmz")
        lmz.compress(path, archive, mapped=True)
        with open(path, "rb") as fh:
            want = fh.read()
        with lmz.MappedArchive(archive, cache_blocks=1) as arc:
            assert arc.read(0, len(want)) == want
        # Defeat the coalescing and confirm the same answer.
        with lmz.MappedArchive(archive, cache_blocks=1) as arc:
            arc.COALESCE_MAX = 0
            assert arc.read(0, len(want)) == want


def test_prefetch_does_not_change_what_is_read():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.safetensors")
        write_safetensors(path, [("w", "BF16", [1 << 17], weights_bf16(1 << 17, 11))])
        archive = os.path.join(d, "a.lmz")
        lmz.compress(path, archive, mapped=True)
        with open(path, "rb") as fh:
            want = fh.read()
        with lmz.MappedArchive(archive, cache_blocks=256) as arc:
            assert arc.prefetch(0, len(want)) > 0
            assert arc.prefetch(0, len(want)) == 0  # already cached, no work
            assert arc.read(0, len(want)) == want
            # Out of range and empty requests are no-ops, not errors.
            assert arc.prefetch(arc.size, 4096) == 0
            assert arc.prefetch(0, 0) == 0


def test_mapped_archive_reads_concurrently():
    """One reader shared by many threads, which is how the mount uses it."""
    import threading as _threading

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "m.safetensors")
        write_safetensors(path, [("w", "BF16", [1 << 18], weights_bf16(1 << 18, 13))])
        archive = os.path.join(d, "a.lmz")
        lmz.compress(path, archive, mapped=True)
        with open(path, "rb") as fh:
            want = fh.read()

        # A cache far too small for the file, so blocks are constantly
        # evicted and re-decoded and the claim path is actually exercised.
        with lmz.MappedArchive(archive, cache_blocks=4, workers=1) as arc:
            errors = []
            step = len(want) // 8

            def reader(k):
                try:
                    for _ in range(6):
                        # Overlapping ranges, so threads collide on blocks.
                        off = (k * step // 2) % max(1, len(want) - step)
                        got = arc.read(off, step)
                        if got != want[off:off + step]:
                            errors.append(f"thread {k} mismatch at {off}")
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"thread {k}: {type(exc).__name__}: {exc}")

            threads = [_threading.Thread(target=reader, args=(k,))
                       for k in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=120)
            assert not any(t.is_alive() for t in threads), "a reader deadlocked"
            assert not errors, errors[:4]


def test_readahead_predicts_only_sequential_streams():
    from lmz.store import Readahead

    seen = []

    class FakeArchive:
        def prefetch(self, offset, length):
            seen.append((offset, length))
            return 0

    arc = FakeArchive()
    ra = Readahead(1, window=1 << 20, slice=1 << 18)
    key = ("m", 0)
    ra.note(arc, key, 0, 1 << 16)          # first read: nothing predicted
    assert ra.issued == 0
    ra.note(arc, key, 1 << 16, 1 << 16)    # continues: predict ahead
    first = ra.issued
    assert first > 0
    ra.note(arc, key, 1 << 17, 1 << 16)    # still sequential, but already
    assert ra.issued == first, "the frontier must not re-queue served slices"
    ra.note(arc, key, 99 << 20, 1 << 16)   # a seek resets the stream
    assert ra.issued == first
    ra.close()


def _gpu_streams(nstr: int, plane: int, seed: int = 12345):
    """`nstr` plane-sized buffers with a skewed distribution, and their streams.

    Skewed because a uniform buffer is refused by the coder as not worth
    coding, and packed back to back with no padding and no alignment because
    that is how an archive stores them. Both matter: the coded bytes of most
    streams land on an odd address, and the kernel's cp.async needs a
    16-byte-aligned source; and it prefetches past the last stream, so the
    slack for that has to come from somewhere the caller did not provide.
    Padding or aligning here would test a layout lmz never writes.
    """
    x = seed
    plains, streams, offsets = [], bytearray(), bytearray()
    for _ in range(nstr):
        buf = bytearray(plane)
        for i in range(plane):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF
            # Three-way mixture: one dominant symbol, a narrow band, and a
            # tail, so the frequency table has structure and is not degenerate.
            r = x >> 8
            buf[i] = 0x40 if r % 3 else ((r >> 4) & 0x0F) + (0x80 if r & 8 else 0)
        coded = kernels.rans_encode(buf, kernels.histogram(buf))
        assert coded, "the coder declined a buffer it should have coded"
        offsets += struct.pack("<QQ", len(streams), len(coded))
        streams += coded
        plains.append(bytes(buf))
    return plains, streams, bytes(offsets)


def test_gpu_decode_matches_cpu():
    """The GPU decoder reproduces lmz_rans_decode exactly, or it is not shipped.

    This is the whole contract: an archive decoded on a GPU and the same
    archive decoded on a CPU are the same bytes. There is no tolerance to
    check, because a decoder either reproduces the plaintext or it does not.
    """
    from lmz import gpu

    ok, why = gpu.available()
    if not ok:
        raise Skip(f"no GPU decoder: {why}")
    if not kernels.have_rans():
        raise Skip("building the streams needs the native coder")

    nstr, plane = 64, 4096
    assert plane % gpu.grain() == 0
    plains, streams, offsets = _gpu_streams(nstr, plane)

    out = gpu.decode_batch(streams, offsets, nstr, plane)
    assert out is not None, f"decode_batch declined: {gpu.last_error()}"
    assert len(out) == nstr * plane
    for i, want in enumerate(plains):
        got = bytes(out[i * plane:(i + 1) * plane])
        assert got == want, f"stream {i} differs from the plaintext"
        # And against the CPU decoder on the same bytes, not just the source,
        # so a shared bug in encode+expectation cannot hide here.
        o, n = struct.unpack_from("<QQ", offsets, i * 16)
        assert bytes(kernels.rans_decode(bytes(streams[o:o + n]), plane)) == want


# Byte-value distributions the one benchmark workload never contained. Built
# by translating uniform random bytes, which is a C-speed way to get an exact
# shape -- a per-byte Python loop over a 32 KiB plane is not worth the seconds.
_GPU_SHAPES = {
    # One symbol at the full probability scale. The CPU coder had a
    # frequency-field overflow here once; the GPU table packs freq-1 into
    # twelve bits for the same reason and wants the same check.
    "single": bytes([0x5A]) * 256,
    "two": bytes([0x01]) * 128 + bytes([0xFE]) * 128,
    "flat": bytes(range(256)),
    "tail": bytes([0x20]) * 250 + bytes(range(6)),
    "skewed": bytes([0x40]) * 170 + bytes(range(16)) * 5 + bytes([0x7F]) * 6,
}


def test_gpu_decode_over_distributions_and_shapes():
    """Every distribution the coder emits, and batch shapes that do not divide.

    The 936 MB workload the kernel was tuned on is one distribution at one
    size. These are the ones that break tables and block arithmetic: a single
    symbol at full scale, a batch that is not a whole number of blocks, and
    the smallest plane the kernel will take.
    """
    import random

    from lmz import gpu

    ok, why = gpu.available()
    if not ok:
        raise Skip(f"no GPU decoder: {why}")
    if not kernels.have_rans():
        raise Skip("building the streams needs the native coder")

    checked = 0
    for name, table in _GPU_SHAPES.items():
        for plane in (gpu.grain(), 4096):
            # 1 is a single group in a block of many; 17 and 129 leave a
            # partial block whose spare groups decode stream 0 and must not
            # write anything.
            for nstr in (1, 17, 129):
                rnd = random.Random(hash((name, plane, nstr)) & 0xFFFF)
                bufs, streams, offsets = [], bytearray(), bytearray()
                for _ in range(nstr):
                    buf = rnd.randbytes(plane).translate(table)
                    coded = kernels.rans_encode(buf, kernels.histogram(buf))
                    if coded is None:
                        break
                    offsets += struct.pack("<QQ", len(streams), len(coded))
                    streams += coded
                    bufs.append(buf)
                if len(bufs) != nstr:
                    continue
                out = gpu.decode_batch(streams, bytes(offsets), nstr, plane)
                assert out is not None, f"{name} p={plane} n={nstr}: {gpu.last_error()}"
                for i, want in enumerate(bufs):
                    assert bytes(out[i * plane:(i + 1) * plane]) == want, \
                        f"{name} p={plane} n={nstr}: stream {i} differs"
                checked += 1
    assert checked >= 20, f"only {checked} shapes actually ran"


def test_encode_options_match_the_signature():
    """The declaration must not be able to drift from what compress accepts.

    A table of options that has quietly fallen behind the function is worse
    than no table, because a caller trusts it. So this checks the two against
    each other by introspection rather than against a copy of the list: a new
    keyword with no entry fails here, and so does an entry for a keyword that
    no longer exists or whose default has moved.
    """
    import inspect

    opts = lmz.encode_options()
    sig = inspect.signature(lmz.compress)
    keywords = {name: p for name, p in sig.parameters.items()
                if p.kind is inspect.Parameter.KEYWORD_ONLY}

    assert set(opts) == set(keywords), (
        set(opts) ^ set(keywords), "declaration and signature disagree")
    for name, param in keywords.items():
        assert opts[name].default == param.default, name
        assert opts[name].kind in ("format", "schedule", "observe"), name
        assert opts[name].describes, name
        assert opts[name].name == name

    # src and dst are positional, and deliberately not options.
    assert "src" not in opts and "dst" not in opts

    # Exactly one scheduling knob today. If a second appears, it needs the
    # same "fallback, never a decision" treatment, so make that deliberate.
    assert [n for n, o in opts.items() if o.kind == "schedule"] == ["workers"]

    # Undeclared keywords are rejected by the language, which is why the
    # table does not have to police the call.
    try:
        lmz.compress("/nonexistent", "/nonexistent.lmz", not_an_option=1)
    except TypeError:
        pass
    else:
        raise AssertionError("an undeclared keyword must not be accepted")


def test_scheduling_options_do_not_change_the_bytes():
    """`workers` is declared `schedule`; that has to be true of the archive.

    This is what makes the format/schedule split a property of lmz rather
    than a naming convention. A future option misclassified as scheduling --
    one that quietly changed chunking, say -- would produce different bytes
    at different thread counts and fail here.
    """
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "m.bin")
        # Enough chunks that the work really is spread across the workers.
        with open(src, "wb") as fh:
            fh.write((bytes(range(256)) * 4096) * 24)

        digests = set()
        for workers in (1, 2, 8, 16):
            archive = os.path.join(d, f"w{workers}.lmz")
            lmz.compress(src, archive, workers=workers)
            digests.add(digest(archive))
        assert len(digests) == 1, f"{len(digests)} distinct archives"

        # And the schedule really did vary -- a default that collapsed to one
        # thread would pass the check above without testing anything.
        assert lmz.encode_options()["workers"].default is None


def test_gpu_cost_model_is_publishable():
    """The cost model is a contract, so its shape is checked even with no GPU.

    A consumer builds a rate curve from these constants and decides whether
    the coded route beats the plain one on its machine. A key that changes
    name or a point published where the truth is an interval both land as a
    wrong decision downstream rather than as an error, so the shape is pinned
    here and the provenance is required to be present.
    """
    from lmz import gpu

    cm = gpu.cost_model()
    for key in ("lanes", "states", "grain", "bytes_per_symbol",
                "k_cycles_per_byte", "expansion", "bound", "provenance"):
        assert key in cm, key

    # Exact properties of the format: 8 interleaved states is what makes one
    # stream 8 lanes of work, and the grain follows from it.
    assert cm["lanes"] == 8 and cm["states"] == 8
    assert cm["bytes_per_symbol"] == 1
    assert cm["grain"] == gpu.grain()

    # k is not pinned to a point, and must not be published as one.
    lo, hi = cm["k_cycles_per_byte"]
    assert 0 < lo < hi, cm["k_cycles_per_byte"]

    assert cm["expansion"] > 1
    assert cm["bound"]["compute_below_threads"] > 0
    assert 0 < cm["bound"]["saturates_at_fraction_of_peak_dram"] <= 1

    # A number without its conditions is not a measurement.
    prov = cm["provenance"]
    for key in ("kernel", "device", "machine", "archive", "method",
                "k_derivation", "status"):
        assert prov.get(key), key

    # The shared-memory layout, as numbers rather than prose. A consumer
    # divides its own device's budget by these to learn how many blocks it can
    # hold, and compares that against the occupancy k was measured at: hold at
    # least as many and the interval brackets the device, hold fewer and it is
    # only a floor. Prose in provenance cannot be computed with, and a
    # consumer restating it in its own constants is a copy that drifts.
    assert cm["shmem_lut_bytes"] == gpu.PROB_SCALE * 4
    assert cm["shmem_per_group_bytes"] == gpu.GRAIN + gpu.BUFB
    lo_blocks, hi_blocks = cm["blocks_per_unit_at_measurement"]
    assert 0 < lo_blocks <= hi_blocks

    # Those constants mirror the kernel, so they are checked against what the
    # library itself reports rather than trusted. `grain` is the one the C ABI
    # exposes directly; if it disagrees, the mirror has drifted.
    assert gpu.GRAIN == gpu.grain()
    assert gpu.NST == cm["lanes"] == cm["states"]

    # A block's request must be positive and must actually fit somewhere: a
    # layout needing more than the 48 KiB every CUDA device grants without
    # opting in would decline on hardware the kernel claims to support.
    one_group = cm["shmem_lut_bytes"] + cm["shmem_per_group_bytes"]
    assert 0 < one_group <= 99 * 1024


def test_gpu_device_entry_point_owns_nothing():
    """The device path must take the caller's memory and synchronise nothing.

    A consumer that has built its own context, staging ring and event chain
    cannot share a device with a library that quietly builds its own. This is
    the entry point that lets it drive lmz instead, so what is checked is that
    it exists, that it is exported, and -- where there is a GPU -- that it
    decodes correctly against pointers the caller allocated in a context the
    caller created.
    """
    from lmz import gpu

    assert "decode_batch_dev" in gpu.__all__
    assert callable(gpu.decode_batch_dev)
    # Declining without a device is a status, not an exception: the caller is
    # mid-pipeline and the decision to fall back is theirs.
    assert gpu.decode_batch_dev(1, 1, 0, 128, 1) == gpu.OK  # nstr == 0

    ok, _why = gpu.available()
    if not ok:
        assert gpu.decode_batch_dev(1, 1, 4, 128, 1) == gpu.ENODEV
        raise Skip("no GPU decoder")

    import ctypes
    import struct

    cuda = ctypes.CDLL("libcuda.so.1")
    assert cuda.cuInit(0) == 0
    dev, ctx = ctypes.c_int(), ctypes.c_void_p()
    assert cuda.cuDeviceGet(ctypes.byref(dev), 0) == 0
    assert cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev) == 0
    try:
        plane, nstr = 4096, 32
        table = bytes((abs(i - 128) // 3 + 100) % 256 for i in range(256))
        segs = [rand(plane, i + 1).translate(table) for i in range(nstr)]
        streams, offsets = bytearray(), bytearray()
        for seg in segs:
            coded = kernels.rans_encode(seg, kernels.histogram(seg))
            offsets += struct.pack("<QQ", len(streams), len(coded))
            streams += coded
        # The caller owns the padding here, because the allocation is the
        # caller's -- decode_batch adds it on the device, this cannot.
        blob = bytes(streams) + b"\0" * gpu.pad_bytes()

        ptrs = []

        def alloc(n):
            p = ctypes.c_void_p()
            assert cuda.cuMemAlloc_v2(ctypes.byref(p), ctypes.c_size_t(n)) == 0
            ptrs.append(p)
            return p

        d_streams, d_off, d_out = alloc(len(blob)), alloc(len(offsets)), \
            alloc(nstr * plane)
        cuda.cuMemcpyHtoD_v2(d_streams, blob, ctypes.c_size_t(len(blob)))
        cuda.cuMemcpyHtoD_v2(d_off, bytes(offsets), ctypes.c_size_t(len(offsets)))

        rc = gpu.decode_batch_dev(d_streams.value, d_off.value, nstr, plane,
                                  d_out.value)
        assert rc == gpu.OK, (rc, gpu.last_error())
        # It did not synchronise; the caller does, which is the point.
        assert cuda.cuCtxSynchronize() == 0
        host = ctypes.create_string_buffer(nstr * plane)
        cuda.cuMemcpyDtoH_v2(host, d_out, ctypes.c_size_t(nstr * plane))
        assert host.raw[:nstr * plane] == b"".join(segs)
    finally:
        for p in ptrs:
            cuda.cuMemFree_v2(p)
        cuda.cuCtxDestroy_v2(ctx)


def test_gpu_verify_reports_this_machine():
    """`lmz doctor --gpu-verify` is the field report, so it has to be right.

    It is the one thing a stranger with a Turing card will run, and its answer
    is the only evidence that will ever exist for that architecture. A verdict
    that is wrong in either direction is worse than no command at all.
    """
    from lmz import gpu

    ok, why = gpu.available()
    r = gpu.verify(quick=True)
    assert r["lmz"] == lmz.__version__
    if not ok:
        assert r["ok"] is False and r["device"] is None and r["why"]
        raise Skip(f"no GPU decoder: {why}")
    assert r["ok"] is True, r["failures"]
    assert r["checked"] >= 5 and not r["failures"]
    assert r["gbps"] and r["gbps"] > 0

    out = subprocess.run([*CLI, "doctor", "--gpu-verify"], capture_output=True,
                         text=True)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "byte-identical" in out.stdout and "verdict  OK" in out.stdout


def test_gpu_refuses_a_device_that_disagrees_with_the_cpu():
    """A card whose answer differs from the CPU's is not used at all.

    The kernel has been executed on one architecture. A silently wrong decoder
    is worse than an absent one -- the caller gets weights back rather than an
    error -- so the guard is that the first decode has a known answer, and
    this checks the guard rather than the kernel.
    """
    from lmz import gpu

    ok, why = gpu.available()
    if not ok:
        raise Skip(f"no GPU decoder: {why}")

    saved_state, saved_lib, saved_test = gpu._state, gpu._lib, gpu._selftest
    try:
        gpu._selftest = lambda lib: False
        gpu._state, gpu._lib = "unloaded", None
        live, reason = gpu.available()
        assert live is False, "a disagreeing device was used anyway"
        assert "did not reproduce" in reason, reason
        assert gpu.decode_batch(b"", b"", 1, 128) is None
    finally:
        gpu._selftest = saved_test
        gpu._state, gpu._lib = "unloaded", None
        gpu.available()          # re-probe honestly for whatever runs next
    assert gpu.backend().startswith("cuda:")


def test_gpu_probe_survives_a_driver_that_crashes():
    """Loading the CUDA library must not be able to kill the caller.

    A driver that is half-removed or mid-upgrade leaves libcuda.so.1 on disk
    with an initialiser that faults, and `ctypes.CDLL` then takes the whole
    interpreter down -- no return code involved. That happened for real, so
    the first load is done in a child process that is allowed to die. This
    runs everywhere: it is the plumbing, not the driver.
    """
    from lmz import gpu

    # What matters is that it declines with a reason and does not raise. The
    # wording is the platform's -- macOS dlopen and glibc describe a missing
    # library quite differently -- so asserting on it tests dyld, not lmz.
    ok, why = gpu._probe_elsewhere("/nonexistent/nothing-here.so")
    assert ok is False and why.strip(), repr(why)

    junk = os.path.join(tempfile.mkdtemp(), "junk.so")
    with open(junk, "wb") as fh:
        fh.write(b"not an ELF file at all")
    ok, why = gpu._probe_elsewhere(junk)
    assert ok is False and why.strip(), repr(why)

    # A child that dies from a signal must be reported as a crash rather than
    # raising, because that is the case the guard exists for.
    saved = gpu._PROBE_SOURCE
    try:
        gpu._PROBE_SOURCE = "import os, signal\nos.kill(os.getpid(), signal.SIGSEGV)\n"
        ok, why = gpu._probe_elsewhere(junk)
        assert ok is False, why
        assert "crashed" in why and "staying on the CPU" in why, why
    finally:
        gpu._PROBE_SOURCE = saved


def test_colab_notebook_still_opens():
    """The badge in the README is the only path most people will ever take.

    It is a JSON file that gets hand-edited whenever the evidence changes, and
    a stray comma makes Colab show a load error instead of the notebook -- at
    which point the field reports simply stop arriving and nothing says why.
    The runtime metadata matters just as much: without `accelerator` and a T4
    `gpuType` the notebook opens on a CPU runtime, prints "no CUDA device",
    and asks a card-specific question of no card at all.
    """
    path = os.path.join(ROOT, "docs", "verify-on-colab.ipynb")
    with open(path, encoding="utf-8") as fh:
        nb = json.load(fh)

    assert nb["metadata"]["accelerator"] == "GPU"
    assert nb["metadata"]["colab"]["gpuType"] == "T4"
    assert nb["nbformat"] == 4

    cells = nb["cells"]
    assert cells, "an empty notebook opens fine and asks nothing"
    for c in cells:
        assert c["cell_type"] in ("markdown", "code")
        # nbformat allows a bare string here and Colab reads it, but an editor
        # that rewrites one cell that way turns the next prose change into a
        # single 3000-character line nobody can review. Keep the line lists.
        assert isinstance(c["source"], list), c["cell_type"]
        if c["cell_type"] == "code":
            assert "outputs" in c and "execution_count" in c

    body = "\n".join("".join(c["source"]) for c in cells)
    # The two things the notebook exists to do: install the published wheel
    # and run the check. A doc edit that loses either leaves a page that reads
    # well and reports nothing.
    assert "pip install -q lmzip" in body
    assert "doctor --gpu-verify" in body
    assert "github.com/FanxinSun/lmz/issues" in body


def test_gpu_build_declines_hardware_it_cannot_target():
    """A card below the floor is declined with a reason, not a compiler error.

    This runs everywhere, GPU or not: it is the build logic, and the point is
    that lmz says "compute capability 6.1 is below 7.5" rather than passing
    -arch=sm_61 to an nvcc that will answer "unsupported gpu architecture".
    """
    from lmz.gpu import build as gpubuild

    saved = list(gpubuild._arch_cache)
    try:
        gpubuild._arch_cache[:] = [(6, 1)]           # a Pascal card
        assert gpubuild.build(force=True) is None
        assert "6.1" in gpubuild.last_error and "below" in gpubuild.last_error
    finally:
        gpubuild._arch_cache[:] = saved


def test_gpu_declines_rather_than_guesses():
    """Every shape it cannot take comes back as None, which means "use the CPU".

    A decoder that is optional has to be unambiguous about declining: the
    caller's fallback is a correct decode, so silence is fine and a wrong
    answer is not.
    """
    from lmz import gpu

    ok, why = gpu.available()
    if not ok:
        raise Skip(f"no GPU decoder: {why}")
    if not kernels.have_rans():
        raise Skip("building the streams needs the native coder")

    plains, streams, offsets = _gpu_streams(2, 4096)
    # An empty batch is not a decline -- there was nothing to decode and it
    # succeeded. None is reserved for "this needs the CPU".
    assert bytes(gpu.decode_batch(streams, offsets, 0, 4096)) == b""
    # A plane that is not a whole number of grains: the kernel retires
    # gpu.grain() bytes per group per step and cannot stop part way.
    assert gpu.decode_batch(streams, offsets, 2, 4096 + 1) is None
    assert "grain" in gpu.last_error()

    # Corruption is not a shape it declines, it is an error it reports.
    broken = bytearray(streams)
    broken[0] = ord("X")
    try:
        gpu.decode_batch(broken, offsets, 2, 4096)
    except ValueError:
        pass
    else:
        raise AssertionError("a stream that is not lmz rANS decoded anyway")


def test_gpu_is_optional():
    """Nothing about lmz requires CUDA, and asking must not build a compiler.

    The package ships a .cu and no CUDA. `pip install lmzip` on a machine with
    no toolkit and no card has to behave exactly as it did before the GPU
    decoder existed, so the import must be free and the probe must be opt-in.
    """
    from lmz import gpu

    assert gpu.grain() > 0 and gpu.header_bytes() == 516

    # Importing lmz must not drag in the CUDA module at all.
    out = subprocess.run(
        [sys.executable, "-c",
         "import lmz, sys; sys.exit(1 if 'lmz.gpu' in sys.modules else 0)"],
        cwd=ROOT, capture_output=True)
    assert out.returncode == 0, "import lmz imported lmz.gpu"

    # backends() reports what is known; it must not go and run nvcc.
    out = subprocess.run(
        [sys.executable, "-c",
         "import lmz, sys; b = lmz.backends();"
         " sys.exit(0 if b['gpu'] == 'not probed' else 1)"],
        cwd=ROOT, capture_output=True)
    assert out.returncode == 0, "backends() probed the GPU as a side effect"

    # And LMZ_NO_GPU turns it off wherever it would otherwise be on.
    env = dict(os.environ, LMZ_NO_GPU="1")
    out = subprocess.run(
        [sys.executable, "-c",
         "from lmz import gpu; ok, why = gpu.available();"
         " print(ok, why)"],
        cwd=ROOT, capture_output=True, text=True, env=env)
    assert out.stdout.startswith("False"), out.stdout
    assert "LMZ_NO_GPU" in out.stdout


class Skip(Exception):
    """Raised by a test that cannot run in this environment."""


class MountedStore:
    """Serve a store from a separate process, for the length of a `with`.

    Separate because a thread that faults on its own mount blocks in the
    kernel holding the GIL, so an in-process server can never answer it.
    """

    def __init__(self, store_root: str, point: str, *extra: str):
        self.point = point
        self.args = list(extra)
        self.store_root = store_root
        self.proc = None

    def __enter__(self):
        from lmz import fuse

        ok, why = fuse.available()
        if not ok:
            raise Skip(why)
        os.makedirs(self.point, exist_ok=True)
        self.proc = subprocess.Popen(
            CLI + ["mount", self.point, "--store", self.store_root, "-q"]
            + self.args,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for _ in range(200):
            if os.path.ismount(self.point):
                return self
            if self.proc.poll() is not None:
                err = self.proc.stderr.read().decode().strip()
                raise Skip(f"mount exited: {err or self.proc.returncode}")
            time.sleep(0.05)
        self.__exit__(None, None, None)
        raise AssertionError("mount did not come up within 10 s")

    def __exit__(self, *_exc):
        from lmz import fuse

        fuse.unmount(self.point)
        if self.proc is not None:
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            if self.proc.stderr:
                self.proc.stderr.close()


def test_mount_serves_the_original_bytes():
    from lmz.store import Store

    with tempfile.TemporaryDirectory() as d:
        src = sample_model(os.path.join(d, "src"))
        root = os.path.join(d, "store")
        Store(root).add(src, "demo")
        point = os.path.join(d, "mnt")
        with MountedStore(root, point):
            assert os.listdir(point) == ["demo"]
            served = os.path.join(point, "demo")
            assert sorted(os.listdir(served)) == [
                "config.json", "model.safetensors", "original"]
            for rel in ("config.json", "model.safetensors",
                        "original/consolidated.safetensors"):
                a, b = os.path.join(src, rel), os.path.join(served, rel)
                assert os.path.getsize(a) == os.path.getsize(b), rel
                assert digest(a) == digest(b), rel

            # Random access must not need the bytes before it.
            big = os.path.join(served, "model.safetensors")
            with open(os.path.join(src, "model.safetensors"), "rb") as fh:
                want = fh.read()
            with open(big, "rb") as fh:
                for off in (0, 1, 4095, 4096, len(want) // 3, len(want) - 17):
                    fh.seek(off)
                    assert fh.read(64) == want[off:off + 64], off
                fh.seek(0, os.SEEK_END)
                assert fh.tell() == len(want)


def test_mount_is_read_only():
    from lmz.store import Store

    with tempfile.TemporaryDirectory() as d:
        src = sample_model(os.path.join(d, "src"))
        root = os.path.join(d, "store")
        Store(root).add(src, "demo")
        point = os.path.join(d, "mnt")
        with MountedStore(root, point):
            served = os.path.join(point, "demo")
            for attempt in (lambda: open(os.path.join(served, "new"), "wb"),
                            lambda: open(os.path.join(served, "config.json"), "wb"),
                            lambda: os.unlink(os.path.join(served, "config.json")),
                            lambda: os.mkdir(os.path.join(served, "sub"))):
                try:
                    attempt()
                    raise AssertionError("expected a read-only filesystem")
                except OSError as exc:
                    assert exc.errno in (errno.EROFS, errno.EACCES, errno.EPERM), exc


def test_mount_serves_only_the_named_models():
    from lmz.store import Store

    with tempfile.TemporaryDirectory() as d:
        src = sample_model(os.path.join(d, "src"))
        root = os.path.join(d, "store")
        store = Store(root)
        store.add(src, "keep")
        store.add(src, "hide")
        point = os.path.join(d, "mnt")
        with MountedStore(root, point, "--model", "keep"):
            assert os.listdir(point) == ["keep"]


def test_mount_readahead_serves_the_same_bytes():
    """Speculation may only change timing, never a byte of what is served."""
    from lmz.store import Store

    with tempfile.TemporaryDirectory() as d:
        src = sample_model(os.path.join(d, "src"))
        root = os.path.join(d, "store")
        Store(root).add(src, "demo")
        want = digest(os.path.join(src, "model.safetensors"))
        for flags in (("--readahead", "0"), ("--readahead", "4")):
            point = os.path.join(d, "mnt" + flags[1])
            with MountedStore(root, point, *flags):
                served = os.path.join(point, "demo", "model.safetensors")
                assert digest(served) == want, flags
                with open(served, "rb") as fh:  # a seek must reset the stream
                    fh.seek(4096)
                    head = fh.read(1024)
                    fh.seek(0)
                    assert fh.read(4096 + 1024)[4096:] == head


def test_mount_matches_a_full_decompression():
    """What the mount serves and what `decompress` writes must agree."""
    from lmz.store import Store

    with tempfile.TemporaryDirectory() as d:
        src = sample_model(os.path.join(d, "src"))
        root = os.path.join(d, "store")
        store = Store(root)
        store.add(src, "demo")
        expanded = os.path.join(d, "expanded")
        store.extract("demo", expanded)
        point = os.path.join(d, "mnt")
        with MountedStore(root, point):
            served = os.path.join(point, "demo")
            for dirpath, _dirs, files in os.walk(expanded):
                for name in files:
                    a = os.path.join(dirpath, name)
                    rel = os.path.relpath(a, expanded)
                    assert digest(a) == digest(os.path.join(served, rel)), rel


class MountedFS:
    """A compressed read-write filesystem, served from its own process."""

    def __init__(self, backing: str, point: str, *extra: str):
        self.backing, self.point = backing, point
        self.args = list(extra)
        self.proc = None

    def __enter__(self):
        from lmz import fuse

        ok, why = fuse.available()
        if not ok:
            raise Skip(why)
        os.makedirs(self.point, exist_ok=True)
        os.makedirs(self.backing, exist_ok=True)
        self.proc = subprocess.Popen(
            CLI + ["fs", self.backing, self.point, "-q"] + self.args,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        for _ in range(200):
            if os.path.ismount(self.point):
                return self
            if self.proc.poll() is not None:
                raise Skip(f"fs exited: {self.proc.stderr.read().decode().strip()}")
            time.sleep(0.05)
        self.__exit__(None, None, None)
        raise AssertionError("filesystem did not come up within 10 s")

    def settle(self, timeout=60):
        """Wait for every pending commit to land in the backing store."""
        scratch = os.path.join(self.backing, ".lmz-scratch")
        deadline = time.time() + timeout
        while time.time() < deadline:
            busy = [f for f in os.listdir(scratch)] if os.path.isdir(scratch) else []
            if not busy:
                return
            time.sleep(0.05)
        raise AssertionError("commits did not settle")

    def __exit__(self, *_exc):
        from lmz import fuse

        fuse.unmount(self.point)
        if self.proc is not None:
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
            if self.proc.stderr:
                self.proc.stderr.close()


def test_fs_writes_are_compressed_and_read_back():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "model.safetensors")
        write_safetensors(src, [("w", "BF16", [1 << 16], weights_bf16(1 << 16, 21))])
        back, point = os.path.join(d, "back"), os.path.join(d, "mnt")
        with MountedFS(back, point) as fs:
            dst = os.path.join(point, "model.safetensors")
            shutil.copyfile(src, dst)
            fs.settle()
            assert digest(dst) == digest(src), "bytes changed through the mount"
            assert os.path.getsize(dst) == os.path.getsize(src)
            stored = os.path.join(back, "model.safetensors.lmz")
            assert os.path.exists(stored), "was not stored as an archive"
            assert os.path.getsize(stored) < os.path.getsize(src) * 0.9, \
                "a bf16 model should compress well through the filesystem"


def test_fs_write_then_immediately_read():
    """close() returns before the kernel delivers RELEASE, so this races."""
    with tempfile.TemporaryDirectory() as d:
        back, point = os.path.join(d, "back"), os.path.join(d, "mnt")
        with MountedFS(back, point):
            for i in range(12):
                p = os.path.join(point, f"f{i}.txt")
                body = f"contents number {i}\n" * 40
                with open(p, "w") as fh:
                    fh.write(body)
                with open(p) as fh:          # no sync, no settle: on purpose
                    assert fh.read() == body, f"f{i} read back wrong"


def test_fs_never_publishes_a_size_it_knows_is_stale():
    """A stat of a file mid-write must not see the placeholder, ever.

    Until the commit lands, what is on disk is an empty placeholder and the real
    length is the scratch copy's. node_for is handed that length so it assigns
    the node's size once, already right. Correcting it afterwards -- which is
    what this used to do -- still publishes the placeholder for as long as the
    correction takes, and the node is shared, so a second server thread
    answering a stat inside that window reports zero bytes. The reader then
    believes the file is empty and stops before asking for any of it.

    With the correction applied after the fact, a watching thread saw the wrong
    size 50% of the time, so a short sample is more than enough to catch a
    regression.
    """
    from lmz import lmzfs

    with tempfile.TemporaryDirectory() as d:
        fs = lmzfs.LmzFS(os.path.join(d, "back"), os.path.join(d, "mnt"))
        try:
            root = fs.btree.root
            body = b"contents number 0\n" * 40
            node, fh = fs.fs_create(root, "w.bin", 0o644, os.O_WRONLY)
            fs.fs_write(node, fh, 0, body)

            # The on-disk placeholder is empty; the node must not report that.
            assert os.path.getsize(fs.btree.rawfile("w.bin")) == 0
            fs.resolve(root, "w.bin")
            live = fs.btree.by_path["w.bin"]
            assert live.size == len(body), live.size

            seen, stop = set(), threading.Event()

            def lookups():
                while not stop.is_set():
                    fs.resolve(root, "w.bin")

            def watch():
                while not stop.is_set():
                    seen.add(live.size)

            threads = [threading.Thread(target=lookups) for _ in range(2)]
            threads.append(threading.Thread(target=watch))
            for t in threads:
                t.start()
            time.sleep(0.3)
            stop.set()
            for t in threads:
                t.join()
            assert seen == {len(body)}, f"a stat could have seen {sorted(seen)}"
        finally:
            fs.close()


def test_fs_scratch_fd_closes_only_once_unreachable():
    """The commit must not free the descriptor while a reader can still find it.

    fs_release commits with the file still registered in _open_writes, so a read
    arriving mid-commit is served from the scratch copy. Freeing the descriptor
    before unregistering leaves that reader preading a number the kernel has
    already handed to the next open(), and it gets an unrelated file's bytes
    back. That is what turned `write, close, read` into wrong contents on the
    free-threaded build, where the server lifts its cap from 2 threads to 16 and
    the window is wide enough to land in.

    Driven through the handlers directly rather than through a mount, so it
    checks the ordering itself instead of trying to win a race.
    """
    from lmz import lmzfs

    with tempfile.TemporaryDirectory() as d:
        fs = lmzfs.LmzFS(os.path.join(d, "back"), os.path.join(d, "mnt"))
        closes = []
        original = lmzfs.Handle.close

        def watched(handle):
            with fs._lock:
                closes.append((handle.rel,
                               fs._open_writes.get(handle.rel) is handle))
            original(handle)

        lmzfs.Handle.close = watched
        try:
            node, fh = fs.fs_create(fs.btree.root, "w.bin", 0o644, os.O_WRONLY)
            fs.fs_write(node, fh, 0, b"weights" * 20000)
            fs.fs_release(node, fh)
        finally:
            lmzfs.Handle.close = original
            fs.close()

        assert closes, "the scratch descriptor was never closed"
        for rel, reachable in closes:
            assert not reachable, \
                f"{rel}: descriptor closed while still listed in _open_writes"


def test_fs_preserves_metadata_and_namespace():
    with tempfile.TemporaryDirectory() as d:
        back, point = os.path.join(d, "back"), os.path.join(d, "mnt")
        with MountedFS(back, point) as fs:
            os.makedirs(os.path.join(point, "a", "b"))
            deep = os.path.join(point, "a", "b", "deep.txt")
            with open(deep, "w") as fh:
                fh.write("x" * 9000)
            fs.settle()
            os.chmod(deep, 0o640)
            assert stat.S_IMODE(os.stat(deep).st_mode) == 0o640
            assert os.path.getsize(deep) == 9000

            assert sorted(os.listdir(point)) == ["a"]
            assert sorted(os.listdir(os.path.join(point, "a", "b"))) == ["deep.txt"]

            moved = os.path.join(point, "a", "moved.txt")
            os.rename(deep, moved)
            assert os.path.getsize(moved) == 9000
            assert not os.path.exists(deep)

            os.unlink(moved)
            assert not os.path.exists(moved)
            os.rmdir(os.path.join(point, "a", "b"))
            os.rmdir(os.path.join(point, "a"))
            assert os.listdir(point) == []


def test_fs_declines_to_compress_what_will_not_shrink():
    """Random bytes must not be inflated by being stored."""
    with tempfile.TemporaryDirectory() as d:
        back, point = os.path.join(d, "back"), os.path.join(d, "mnt")
        with MountedFS(back, point) as fs:
            blob = rand(400 << 10, 5)
            p = os.path.join(point, "noise.bin")
            with open(p, "wb") as fh:
                fh.write(blob)
            fs.settle()
            assert os.path.exists(os.path.join(back, "noise.bin.lmr")), \
                "incompressible data should be stored raw, not in a container"
            with open(p, "rb") as fh:
                assert fh.read() == blob
            # Tiny files skip the container too: its header would dwarf them.
            tiny = os.path.join(point, "tiny.txt")
            with open(tiny, "w") as fh:
                fh.write("hi")
            fs.settle()
            assert os.path.exists(os.path.join(back, "tiny.txt.lmr"))
            with open(tiny) as fh:
                assert fh.read() == "hi"


def test_fs_rewrite_replaces_contents():
    with tempfile.TemporaryDirectory() as d:
        back, point = os.path.join(d, "back"), os.path.join(d, "mnt")
        with MountedFS(back, point) as fs:
            p = os.path.join(point, "w.safetensors")
            write_safetensors(p, [("w", "BF16", [1 << 15], weights_bf16(1 << 15, 3))])
            fs.settle()
            first = digest(p)
            write_safetensors(p, [("w", "BF16", [1 << 15], weights_bf16(1 << 15, 9))])
            fs.settle()
            assert digest(p) != first, "rewrite did not take"
            # Exactly one backing form survives; a stale twin would shadow it.
            assert os.path.exists(os.path.join(back, "w.safetensors.lmz"))
            assert not os.path.exists(os.path.join(back, "w.safetensors.lmr"))

            with open(p, "r+b") as fh:      # truncate through the mount
                fh.truncate(1024)
            fs.settle()
            assert os.path.getsize(p) == 1024


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed, skipped = [], []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Skip as exc:
            skipped.append(name)
            print(f"  SKIP  {name}: {exc}")
        except Exception as exc:
            failed.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            import traceback

            traceback.print_exc()
    tail = f", {len(skipped)} skipped" if skipped else ""
    print(f"\n{len(tests) - len(failed) - len(skipped)}/{len(tests)} passed{tail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
