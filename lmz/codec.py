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
from .format import (CODEC_BF16, CODEC_BF16C, CODEC_BLK, CODEC_SPLIT,
                     CODEC_STORED, CODEC_ZSTD, CorruptArchive)
from .planner import KIND_BF16, KIND_BYTES, KIND_Q80

# Above this many bits per byte a stream is genuinely noise and is stored.
# The threshold sits this close to 8 because rANS can profitably code a
# mantissa plane at ~7.90 bits, which anything LZ-based has to turn down.
NOISE_BITS = 7.98

# Bytes sampled when estimating entropy, taken from three places so a
# non-uniform tensor is not judged by its opening bytes alone.
SAMPLE_BYTES = 3 * 65536

# Exponent-conditioned coding of the sign+mantissa plane. Eight equal-mass
# exponent buckets were measured to capture the whole usable dependence on
# real weights (the full 256-way context gains nothing further), and below
# ~1M elements the extra frequency tables cost more than conditioning saves,
# so small chunks skip the attempt entirely.
COND_BUCKETS = 8
COND_MIN_ELEMS = 1 << 20

# What one rANS stream costs beyond its coded bytes: the 516-byte frequency
# header plus the flushed states.
RANS_OVERHEAD = 516 + 4 * 8

_bf16c_hdr = struct.Struct(f"<BB{COND_BUCKETS}BI{COND_BUCKETS}I")

# A GGUF Q8_0 block: 2-byte fp16 scale then 32 int8 quants. Coding scale
# bytes and quant bytes in one alphabet was measured to waste 2% of the
# payload; below ~32k blocks the per-stream tables eat the gain.
Q80_PERIOD = 34
Q80_MIN_BLOCKS = 1 << 15

_q80_hdr = struct.Struct(f"<BB{COND_BUCKETS}BBI{COND_BUCKETS}II")

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


def _hist_entropy(counts, total: int) -> float:
    if not total:
        return 0.0
    h = 0.0
    for c in counts:
        if c:
            p = c / total
            h -= p * log2(p)
    return h


def _est_stream(total: int, h: float) -> int:
    """Predicted rANS stream size; a stream is never stored above raw size."""
    if total == 0:
        return 0
    return min(total, int(total * h / 8) + RANS_OVERHEAD)


def _encode_bf16_cond(exp, sm, nelem: int, level: int, method: int):
    """Code the sign+mantissa plane with one rANS table per exponent bucket.

    Decides from exact histograms whether conditioning beats one shared
    table, and only encodes the winning side, so declining costs two
    histogram passes and a partition rather than a wasted encode. Returns
    (parts, flags) or None to fall back to the plain field split.

    The bucket map is derived from the exponent histogram, which the decoder
    reconstructs from the decoded exponent plane -- nothing about the map is
    stored in the archive.
    """
    ehist = kernels.histogram(exp)
    shist = kernels.histogram(sm)
    h_sm = _hist_entropy(shist, nelem)
    lut = kernels.bucket_lut(ehist, COND_BUCKETS)
    part, counts = kernels.bucket_partition(exp, sm, lut, COND_BUCKETS)
    pmv = memoryview(part)

    est_cond = 0
    pos = 0
    for c in counts:
        if c:
            hseg = _hist_entropy(kernels.histogram(pmv[pos:pos + c]), c)
            est_cond += _est_stream(c, hseg)
            pos += c
    est_plain = _est_stream(nelem, h_sm)
    # The conditional side pays its larger header and one table per bucket;
    # demand a clear win so estimate noise cannot flip the choice.
    if est_cond + (_bf16c_hdr.size - _plane_header(2).size) + 512 >= est_plain:
        return None

    exp_used, exp_payload = _encode_stream(exp, level, method, plane=True)
    sm_methods = []
    seg_lens = []
    parts = []
    pos = 0
    for c in counts:
        if c == 0:
            sm_methods.append(entropy.METHOD_STORED)
            seg_lens.append(0)
            continue
        used, payload = _encode_stream(pmv[pos:pos + c], level, method, plane=True)
        sm_methods.append(used)
        seg_lens.append(len(payload))
        parts.append(payload)
        pos += c
    header = _bf16c_hdr.pack(COND_BUCKETS, exp_used, *sm_methods,
                             len(exp_payload), *seg_lens)
    return [header, exp_payload] + parts, 0


def _encode_q80(data, nblocks: int, level: int, method: int):
    """Split Q8_0 blocks into scale planes and a quants plane, then code each.

    The two scale bytes are the halves of one fp16, and they are correlated:
    the low byte conditioned on the high byte's bucket was measured two bits
    below its own entropy. The same exact-histogram comparison as the BF16
    conditioner decides whether that conditioning pays; the quant bytes are a
    single near-uniform alphabet either way. Returns (parts, flags) or None.
    """
    planes = kernels.split_stride(data, Q80_PERIOD)
    mv = memoryview(planes)
    lo, hi = mv[:nblocks], mv[nblocks:2 * nblocks]
    quants = mv[2 * nblocks:]

    hhist = kernels.histogram(hi)
    lut = kernels.bucket_lut(hhist, COND_BUCKETS)
    part, counts = kernels.bucket_partition(hi, lo, lut, COND_BUCKETS)
    pmv = memoryview(part)
    est_cond = 0
    pos = 0
    for c in counts:
        if c:
            est_cond += _est_stream(c, _hist_entropy(kernels.histogram(
                pmv[pos:pos + c]), c))
            pos += c
    est_plain = _est_stream(nblocks, _hist_entropy(kernels.histogram(lo), nblocks))

    hi_used, hi_payload = _encode_stream(hi, level, method, plane=True)
    lo_methods = [entropy.METHOD_STORED] * COND_BUCKETS
    lo_lens = [0] * COND_BUCKETS
    lo_parts = []
    if est_cond + 512 < est_plain:
        k = COND_BUCKETS
        pos = 0
        for b, c in enumerate(counts):
            if c == 0:
                continue
            used, payload = _encode_stream(pmv[pos:pos + c], level, method,
                                           plane=True)
            lo_methods[b] = used
            lo_lens[b] = len(payload)
            lo_parts.append(payload)
            pos += c
    else:
        k = 0
        used, payload = _encode_stream(lo, level, method, plane=True)
        lo_methods[0] = used
        lo_lens[0] = len(payload)
        lo_parts.append(payload)
    q_used, q_payload = _encode_stream(quants, level, method, plane=True)

    header = _q80_hdr.pack(k, hi_used, *lo_methods, q_used,
                           len(hi_payload), *lo_lens, len(q_payload))
    return [header, hi_payload] + lo_parts + [q_payload], 0


def _decode_q80(payload, nblocks: int, out=None):
    """Decode a block-split Q8_0 chunk back to interleaved 34-byte blocks."""
    hdr = _q80_hdr
    if len(payload) < hdr.size:
        raise CorruptArchive("block chunk is truncated")
    fields = hdr.unpack_from(payload, 0)
    k = fields[0]
    if k not in (0, COND_BUCKETS):
        raise CorruptArchive(f"block chunk declares {k} scale buckets")
    hi_method = fields[1]
    lo_methods = fields[2:2 + COND_BUCKETS]
    q_method = fields[2 + COND_BUCKETS]
    hi_len = fields[3 + COND_BUCKETS]
    lo_lens = fields[4 + COND_BUCKETS:4 + 2 * COND_BUCKETS]
    q_len = fields[4 + 2 * COND_BUCKETS]
    for m in (hi_method, q_method, *lo_methods):
        if m not in entropy.METHOD_NAMES:
            raise CorruptArchive(f"block chunk names method {m}")

    mv = memoryview(payload)
    pos = hdr.size

    def take(ln, m, what, want):
        nonlocal pos
        if pos + ln > len(payload):
            raise CorruptArchive("block chunk is truncated")
        if m == entropy.METHOD_STORED:
            if ln != want:
                raise CorruptArchive(f"{what} holds {ln} bytes, expected {want}")
            block = (payload, pos)
        else:
            data = _decode_stream(mv[pos:pos + ln], m, what, want)
            if len(data) != want:
                raise CorruptArchive(
                    f"{what} decoded to {len(data)} bytes, expected {want}")
            block = (data, 0)
        pos += ln
        return block

    hi_obj, hi_off = take(hi_len, hi_method, "scale-high plane", nblocks)
    hi = bytes(memoryview(hi_obj)[hi_off:hi_off + nblocks])

    if k == 0:
        lo_obj, lo_off = take(lo_lens[0], lo_methods[0], "scale-low plane",
                              nblocks)
        for b in range(1, COND_BUCKETS):
            if lo_lens[b]:
                raise CorruptArchive("unbucketed block chunk carries bucket data")
        lo = (lo_obj, lo_off)
    else:
        hhist = kernels.histogram(hi)
        lut = kernels.bucket_lut(hhist, COND_BUCKETS)
        counts = kernels.bucket_counts(hhist, lut, COND_BUCKETS)
        streams = []
        for b in range(COND_BUCKETS):
            want = counts[b]
            if want == 0:
                if lo_lens[b]:
                    raise CorruptArchive(
                        f"scale bucket {b} should be empty but holds data")
                streams.append((b"", 0))
            else:
                streams.append(take(lo_lens[b], lo_methods[b],
                                    f"scale bucket {b}", want))
        lo = (kernels.bucket_unpartition(hi, streams, lut, COND_BUCKETS), 0)

    q_obj, q_off = take(q_len, q_method, "quant plane", 32 * nblocks)

    streams = [lo, (hi, 0)]
    streams += [(q_obj, q_off + j * nblocks) for j in range(32)]
    buf = out if (out is not None and len(out) >= nblocks * Q80_PERIOD) else \
        bytearray(nblocks * Q80_PERIOD)
    return kernels.merge_stride(streams, nblocks, Q80_PERIOD, buf)


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

    if (kind == KIND_Q80 and esize == Q80_PERIOD and n % Q80_PERIOD == 0
            and n // Q80_PERIOD >= Q80_MIN_BLOCKS and kernels.have_rans()):
        blk = _encode_q80(data, n // Q80_PERIOD, level, method)
        if blk is not None:
            parts, flags = blk
            if sum(len(p) for p in parts) + _min_gain(n) < n:
                return parts, CODEC_BLK, flags, crc
        # Otherwise fall through: 34 is not a splittable width, so the chunk
        # takes the generic single-stream path exactly as it did before.

    splittable = esize > 1 and n % esize == 0 and esize in kernels.SUPPORTED_ESIZES
    if kind == KIND_BF16 and esize == 2 and n % 2 == 0:
        nplanes, cid = 2, CODEC_BF16
        planes = kernels.split_bf16(data)
        nelem = n // 2
        if nelem >= COND_MIN_ELEMS and kernels.have_rans():
            mv = memoryview(planes)
            cond = _encode_bf16_cond(mv[:nelem], mv[nelem:], nelem, level, method)
            if cond is not None:
                parts, flags = cond
                if sum(len(p) for p in parts) + _min_gain(n) < n:
                    return parts, CODEC_BF16C, flags, crc
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


def _decode_bf16c(payload, nelem: int, out=None):
    """Decode a conditioned bf16 chunk back to interleaved bytes.

    The exponent plane comes out first; its histogram rebuilds the bucket
    map, which yields each bucket's expected length, and only then can the
    sign+mantissa segments be decoded and dealt back into element order.
    """
    hdr = _bf16c_hdr
    if len(payload) < hdr.size:
        raise CorruptArchive("conditioned bf16 chunk is truncated")
    fields = hdr.unpack_from(payload, 0)
    k = fields[0]
    if k != COND_BUCKETS:
        raise CorruptArchive(f"conditioned bf16 chunk declares {k} buckets")
    exp_method = fields[1]
    sm_methods = fields[2:2 + k]
    exp_len = fields[2 + k]
    seg_lens = fields[3 + k:3 + k + k]
    # These bytes came from the payload, not the validated chunk record; a
    # value outside the method alphabet is damage, not a future format.
    for m in (exp_method, *sm_methods):
        if m not in entropy.METHOD_NAMES:
            raise CorruptArchive(f"conditioned bf16 chunk names method {m}")

    mv = memoryview(payload)
    pos = hdr.size
    if pos + exp_len > len(payload):
        raise CorruptArchive("conditioned bf16 chunk is truncated")
    if exp_method == entropy.METHOD_STORED:
        if exp_len != nelem:
            raise CorruptArchive(
                f"exponent plane holds {exp_len} bytes, expected {nelem}")
        exp = bytes(mv[pos:pos + exp_len])
    else:
        exp = _decode_stream(mv[pos:pos + exp_len], exp_method,
                             "exponent plane", nelem)
        if len(exp) != nelem:
            raise CorruptArchive(
                f"exponent plane decoded to {len(exp)} bytes, expected {nelem}")
    pos += exp_len

    ehist = kernels.histogram(exp)
    lut = kernels.bucket_lut(ehist, k)
    counts = kernels.bucket_counts(ehist, lut, k)

    streams = []
    for b in range(k):
        ln = seg_lens[b]
        want = counts[b]
        if pos + ln > len(payload):
            raise CorruptArchive("conditioned bf16 chunk is truncated")
        m = sm_methods[b]
        if want == 0:
            if ln:
                raise CorruptArchive(f"bucket {b} should be empty but holds {ln} bytes")
            streams.append((b"", 0))
        elif m == entropy.METHOD_STORED:
            if ln != want:
                raise CorruptArchive(
                    f"bucket {b} holds {ln} bytes, expected {want}")
            streams.append((payload, pos))
        else:
            block = _decode_stream(mv[pos:pos + ln], m, f"bucket {b}", want)
            if len(block) != want:
                raise CorruptArchive(
                    f"bucket {b} decoded to {len(block)} bytes, expected {want}")
            streams.append((block, 0))
        pos += ln

    sm = kernels.bucket_unpartition(exp, streams, lut, k)
    buf = out if (out is not None and len(out) >= nelem * 2) else bytearray(nelem * 2)
    return kernels.merge_bf16([(exp, 0), (sm, 0)], nelem, buf)


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
    elif codec == CODEC_BF16C:
        if esize != 2 or rlen % 2:
            raise ValueError("a conditioned bf16 chunk must have 2-byte elements")
        result = _decode_bf16c(payload, rlen // 2, out)
    elif codec == CODEC_BLK:
        if esize != Q80_PERIOD or rlen % Q80_PERIOD:
            raise ValueError(
                f"a block chunk must be whole {Q80_PERIOD}-byte blocks")
        result = _decode_q80(payload, rlen // Q80_PERIOD, out)
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
