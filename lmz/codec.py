"""Per-chunk encode and decode.

A chunk is split into byte planes and each plane is judged on its own. That
adaptivity is where the compression actually comes from, and it is what a
fixed codec cannot do: in BF16 weights the exponent plane shrinks by ~2.6x
while the mantissa plane is incompressible noise, and in an FP32 checkpoint
that was upcast from BF16 two planes are entirely zero. Measuring each plane
handles all of those without knowing which case it is in.

Planes that look random are stored without ever being handed to the entropy
coder, which saves most of the time that would be spent proving they are
incompressible.
"""

from __future__ import annotations

import struct
import zlib
from math import log2

from . import entropy, kernels
from .format import (CODEC_BF16, CODEC_SPLIT, CODEC_STORED, CODEC_ZSTD,
                     CorruptArchive)
from .planner import KIND_BF16, KIND_BYTES

# Above this many bits per byte a stream is genuinely noise and is stored.
# The threshold sits this close to 8 because rANS can profitably code a
# mantissa plane at ~7.90 bits, which anything LZ-based has to turn down.
NOISE_BITS = 7.98

# Bytes sampled when estimating entropy, taken from three places so a
# non-uniform tensor is not judged by its opening bytes alone.
SAMPLE_BYTES = 3 * 65536

_hdr_cache: dict[int, struct.Struct] = {}


def _plane_header(nplanes: int) -> struct.Struct:
    s = _hdr_cache.get(nplanes)
    if s is None:
        s = _hdr_cache[nplanes] = struct.Struct(f"<{nplanes}I")
    return s


def _sample(buf) -> bytes:
    n = len(buf)
    if n <= SAMPLE_BYTES:
        return bytes(buf)
    w = SAMPLE_BYTES // 3
    mid = (n - w) // 2
    mv = memoryview(buf)
    return b"".join((mv[:w], mv[mid:mid + w], mv[n - w:]))


def estimate_entropy(buf) -> float:
    """Order-0 entropy in bits per byte, from a sample of the buffer."""
    s = _sample(buf)
    if not s:
        return 0.0
    counts = kernels.histogram(s)
    total = len(s)
    h = 0.0
    for c in counts:
        if c:
            p = c / total
            h -= p * log2(p)
    return h


def _encode_stream(buf, level: int, method: int, plane: bool = False):
    """Compress one stream unless doing so is not worth it.

    Returns (method_used, payload).

    Planes get the order-0 rANS coder, which is what the data actually calls
    for. A mantissa plane sits at about 7.90 bits per byte -- a real 1.3%
    of structure, but so little that a general-purpose coder's framing and
    match search eat the whole gain and then some. rANS charges fractional
    bits and almost no overhead, so it collects that 1.3% where zstd could
    only decline the job.
    """
    n = len(buf)
    if n == 0:
        return entropy.METHOD_STORED, b""
    if estimate_entropy(buf) >= NOISE_BITS:
        return entropy.METHOD_STORED, buf

    best_method, best = entropy.METHOD_STORED, None
    if plane and kernels.have_rans():
        best = kernels.rans_encode(buf)
        if best is not None:
            best_method = entropy.METHOD_RANS
    else:
        try:
            best = entropy.compress(buf, level, method)
            best_method = method
        except Exception:
            best = None
        # A stream the general-purpose coder barely dented may still be a
        # plain symbol distribution -- quantised INT8 weights, for one.
        if kernels.have_rans() and (best is None or len(best) > n - (n >> 5)):
            alt = kernels.rans_encode(buf)
            if alt is not None and (best is None or len(alt) < len(best)):
                best_method, best = entropy.METHOD_RANS, alt

    if best is None or len(best) + _min_gain(n) >= n:
        return entropy.METHOD_STORED, buf
    return best_method, best


def _min_gain(n: int) -> int:
    """Smallest saving worth encoding for: the coder's own header, plus a bit."""
    return max(1024, n >> 8)


def encode_chunk(data, esize: int, level: int = 1, checksum: bool = True,
                 method: int = -1, kind: int = KIND_BYTES):
    """Encode one chunk.

    Returns (payload, codec, flags, crc). `payload` may be a list of buffers,
    which the caller writes in order.
    """
    if method < 0:
        method = entropy.DEFAULT_METHOD
    crc = zlib.crc32(data) & 0xFFFFFFFF if checksum else 0
    n = len(data)

    splittable = esize > 1 and n % esize == 0 and esize in kernels.SUPPORTED_ESIZES
    if kind == KIND_BF16 and esize == 2 and n % 2 == 0:
        nplanes, cid = 2, CODEC_BF16
        planes = kernels.split_bf16(data)
    elif splittable:
        nplanes, cid = esize, CODEC_SPLIT
        planes = kernels.split(data, esize)
    else:
        nplanes = 0

    if nplanes:
        nelem = n // nplanes
        mv = memoryview(planes)
        parts, lengths, flags = [], [], 0
        total = 0
        any_compressed = False
        for k in range(nplanes):
            used, payload = _encode_stream(mv[k * nelem:(k + 1) * nelem], level,
                                           method, plane=True)
            flags |= (used & 0x3) << (2 * k)
            lengths.append(len(payload))
            parts.append(payload)
            total += len(payload)
            if used != entropy.METHOD_STORED:
                any_compressed = True
        # If nothing compressed, the split only shuffled bytes; storing the
        # original avoids paying for a merge on the way back.
        if any_compressed and total + _plane_header(nplanes).size < n:
            return ([_plane_header(nplanes).pack(*lengths)] + parts,
                    cid, flags, crc)
        return ([data], CODEC_STORED, 0, crc)

    used, payload = _encode_stream(data, level, method)
    if used == entropy.METHOD_STORED:
        return ([data], CODEC_STORED, 0, crc)
    return ([payload], CODEC_ZSTD, used & 0x3, crc)


def _decode_stream(raw, method: int, what: str, out_len: int | None = None):
    """Decompress one stream, reporting backend failures as corruption.

    Damaged data usually surfaces as an error from the entropy coder rather
    than a checksum mismatch, because decoding fails before there is anything
    to checksum. Callers should see one recognisable error either way.
    """
    try:
        return entropy.decompress(raw, method, out_len)
    except entropy.UnsupportedMethod:
        raise
    except Exception as exc:
        raise CorruptArchive(f"{what} failed to decode: {exc}") from exc


def decode_chunk(payload, codec: int, esize: int, flags: int, rlen: int,
                 crc: int = 0, verify: bool = True, out=None):
    """Decode one chunk back to its exact original bytes.

    `out` is an optional scratch buffer to decode into. Callers that hand one
    over must finish with the returned view before reusing the buffer.
    """
    if codec == CODEC_STORED:
        result = payload
        if len(result) != rlen:
            raise ValueError(f"stored chunk is {len(result)} bytes, expected {rlen}")
    elif codec == CODEC_ZSTD:
        result = _decode_stream(payload, flags & 0x3, "chunk", rlen)
        if len(result) != rlen:
            raise CorruptArchive(
                f"chunk decoded to {len(result)} bytes, expected {rlen}")
    elif codec in (CODEC_SPLIT, CODEC_BF16):
        if esize not in kernels.SUPPORTED_ESIZES or esize < 2 or rlen % esize:
            raise ValueError(f"split chunk has invalid element size {esize}")
        if codec == CODEC_BF16 and esize != 2:
            raise ValueError("a bf16 chunk must have 2-byte elements")
        nplanes = 2 if codec == CODEC_BF16 else esize
        nelem = rlen // nplanes
        hdr = _plane_header(nplanes)
        if len(payload) < hdr.size:
            raise ValueError("split chunk is truncated")
        lengths = hdr.unpack_from(payload, 0)
        mv = memoryview(payload)
        pos = hdr.size
        planes = []
        for k, ln in enumerate(lengths):
            if pos + ln > len(payload):
                raise ValueError("split chunk is truncated")
            m = (flags >> (2 * k)) & 0x3
            if m == entropy.METHOD_STORED:
                if ln != nelem:
                    raise ValueError(
                        f"plane {k} holds {ln} bytes, expected {nelem}")
                planes.append((payload, pos))
            else:
                block = _decode_stream(mv[pos:pos + ln], m, f"plane {k}", nelem)
                if len(block) != nelem:
                    raise CorruptArchive(
                        f"plane {k} decoded to {len(block)} bytes, expected {nelem}")
                planes.append((block, 0))
            pos += ln
        buf = out if (out is not None and len(out) >= rlen) else bytearray(rlen)
        if codec == CODEC_BF16:
            result = kernels.merge_bf16(planes, nelem, buf)
        else:
            result = kernels.merge_planes(planes, esize, nelem, buf)
    else:
        raise ValueError(f"unknown codec id {codec}")

    if verify and crc:
        if (zlib.crc32(result) & 0xFFFFFFFF) != crc:
            raise CorruptArchive("chunk failed its checksum; the archive is corrupt")
    return result
