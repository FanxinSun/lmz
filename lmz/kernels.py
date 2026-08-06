"""Byte-plane split and merge.

`split` turns an array of fixed-size elements into one contiguous plane per
byte position; `merge` is its exact inverse. Splitting is what exposes the
structure an entropy coder can use: in BF16 the sign+exponent byte takes only
a few dozen distinct values while the mantissa byte is close to uniform noise,
and separating them lets each be handled on its own terms.

Three backends are tried in order. The native one is SIMD and releases the
GIL, so it scales across threads. The others exist so the tool still works
with no compiler and no third-party packages.
"""

from __future__ import annotations

import ctypes
import os
import threading

_lock = threading.Lock()
_lib = None
_state = "unloaded"

SUPPORTED_ESIZES = (1, 2, 4, 8)

# Widest fixed-period record split_stride/merge_stride accept, matching
# LMZ_MAX_PERIOD in the kernel. The widest GGUF block in use is Q6_K's 210.
MAX_PERIOD = 256


def _load_native():
    """Load the compiled kernel, building it on first use. None if unavailable."""
    global _lib, _state
    if _state != "unloaded":
        return _lib
    with _lock:
        if _state != "unloaded":
            return _lib
        if os.environ.get("LMZ_NO_NATIVE"):
            _state = "disabled"
            return None
        try:
            from .native import build as _build

            path = _build.build()
            if not path:
                _state = "unavailable"
                return None
            lib = ctypes.CDLL(path)
            lib.lmz_abi_version.restype = ctypes.c_int
            if lib.lmz_abi_version() != 9:
                _state = "abi-mismatch"
                return None
            lib.lmz_isa.restype = ctypes.c_char_p
            v = ctypes.c_void_p
            lib.lmz_split.restype = ctypes.c_int
            lib.lmz_split.argtypes = [v, v, ctypes.c_size_t, ctypes.c_size_t, v]
            lib.lmz_merge.restype = ctypes.c_int
            lib.lmz_merge.argtypes = [v, v, ctypes.c_size_t, ctypes.c_size_t, v]
            lib.lmz_merge_planes.restype = ctypes.c_int
            lib.lmz_merge_planes.argtypes = [ctypes.POINTER(ctypes.c_void_p), v,
                                             ctypes.c_size_t, ctypes.c_size_t, v]
            lib.lmz_scratch_size.restype = ctypes.c_size_t
            lib.lmz_scratch_size.argtypes = [ctypes.c_size_t, ctypes.c_size_t]
            lib.lmz_hist.argtypes = [v, ctypes.c_size_t, v]
            lib.lmz_distinct.restype = ctypes.c_int
            lib.lmz_distinct.argtypes = [v, ctypes.c_size_t]
            lib.lmz_split_bf16.restype = ctypes.c_int
            lib.lmz_split_bf16.argtypes = [v, v, v, ctypes.c_size_t]
            lib.lmz_merge_bf16.restype = ctypes.c_int
            lib.lmz_merge_bf16.argtypes = [v, v, v, ctypes.c_size_t]
            lib.lmz_split_stride.restype = ctypes.c_int
            lib.lmz_split_stride.argtypes = [v, v, ctypes.c_size_t, ctypes.c_size_t]
            lib.lmz_merge_stride.restype = ctypes.c_int
            lib.lmz_merge_stride.argtypes = [ctypes.POINTER(ctypes.c_void_p), v,
                                             ctypes.c_size_t, ctypes.c_size_t]
            lib.lmz_bucket_lut.restype = ctypes.c_int
            lib.lmz_bucket_lut.argtypes = [v, ctypes.c_size_t, v]
            lib.lmz_bucket_partition.restype = ctypes.c_int
            lib.lmz_bucket_partition.argtypes = [v, v, ctypes.c_size_t, v,
                                                 ctypes.c_size_t, v, v]
            lib.lmz_bucket_unpartition.restype = ctypes.c_int
            lib.lmz_bucket_unpartition.argtypes = [v, ctypes.POINTER(ctypes.c_void_p),
                                                   ctypes.c_size_t, v,
                                                   ctypes.c_size_t, v]
            lib.lmz_xor.restype = ctypes.c_int
            lib.lmz_xor.argtypes = [v, v, ctypes.c_size_t, v]
            lib.lmz_k4_scales.restype = ctypes.c_int
            lib.lmz_k4_scales.argtypes = [v, ctypes.c_size_t, v, v]
            lib.lmz_k4_pack.restype = ctypes.c_int
            lib.lmz_k4_pack.argtypes = [v, ctypes.c_size_t, v]
            lib.lmz_k4_unpack.restype = ctypes.c_int
            lib.lmz_k4_unpack.argtypes = [v, ctypes.c_size_t, v]
            lib.lmz_rans_bound.restype = ctypes.c_size_t
            lib.lmz_rans_bound.argtypes = [ctypes.c_size_t]
            lib.lmz_rans_encode.restype = ctypes.c_long
            lib.lmz_rans_encode.argtypes = [v, ctypes.c_size_t, v, ctypes.c_size_t]
            lib.lmz_rans_decode.restype = ctypes.c_int
            lib.lmz_rans_decode.argtypes = [v, ctypes.c_size_t, v, ctypes.c_size_t]
            _lib = lib
            _state = "native:" + lib.lmz_isa().decode()
        except Exception:
            _lib = None
            _state = "unavailable"
    return _lib


def backend() -> str:
    """Name of the active backend, for diagnostics."""
    _load_native()
    if _lib is not None:
        return _state
    try:
        import numpy  # noqa: F401

        return "numpy"
    except ImportError:
        return "python"


def _ptr(obj):
    """Address of a buffer plus a reference that must outlive the call.

    Read-only bytes are addressed in place rather than copied; anything else
    is exported through the buffer protocol.
    """
    if len(obj) == 0:
        return 0, None
    if type(obj) is bytes:
        p = ctypes.c_char_p(obj)
        return ctypes.cast(p, ctypes.c_void_p).value, (p, obj)
    try:
        arr = (ctypes.c_ubyte * len(obj)).from_buffer(obj)
        return ctypes.addressof(arr), arr
    except TypeError:
        b = bytes(obj)  # read-only memoryview or other exporter
        p = ctypes.c_char_p(b)
        return ctypes.cast(p, ctypes.c_void_p).value, (p, b)


# Element sizes above 2 need a working area as large as the chunk. Reusing it
# per thread keeps that off the hot path.
_scratch = threading.local()


def _get_scratch(nbytes: int):
    if nbytes == 0:
        return 0, None
    buf = getattr(_scratch, "buf", None)
    if buf is None or len(buf) < nbytes:
        buf = bytearray(nbytes)
        _scratch.buf = buf
    return _ptr(buf)


def split(src, esize: int) -> bytearray:
    """Deinterleave `src` into `esize` planes, concatenated in byte-position order.

    len(src) must be a multiple of esize.
    """
    if esize not in SUPPORTED_ESIZES:
        raise ValueError(f"unsupported element size {esize}")
    n = len(src)
    if esize == 1:
        return bytearray(src)
    if n % esize:
        raise ValueError(f"buffer of {n} bytes is not a multiple of element size {esize}")
    nelem = n // esize
    out = bytearray(n)
    if n == 0:
        return out

    lib = _load_native()
    if lib is not None:
        src_addr, _a = _ptr(src)
        dst_addr, _b = _ptr(out)
        scr_addr, _c = _get_scratch(lib.lmz_scratch_size(nelem, esize))
        rc = lib.lmz_split(src_addr, dst_addr, nelem, esize, scr_addr)
        if rc != 0:
            raise ValueError(f"unsupported element size {esize}")
        return out

    try:
        import numpy as np

        a = np.frombuffer(src, dtype=np.uint8).reshape(nelem, esize)
        np.frombuffer(out, dtype=np.uint8).reshape(esize, nelem)[:] = a.T
        return out
    except ImportError:
        pass

    # Extended slicing runs in C and is respectable even without numpy.
    mv = bytes(src)
    for k in range(esize):
        out[k * nelem:(k + 1) * nelem] = mv[k::esize]
    return out


def merge(planes, esize: int, nelem: int) -> bytearray:
    """Rebuild interleaved elements from concatenated planes. Inverse of `split`."""
    if esize not in SUPPORTED_ESIZES:
        raise ValueError(f"unsupported element size {esize}")
    n = nelem * esize
    if esize == 1:
        return bytearray(planes)
    if len(planes) != n:
        raise ValueError(f"expected {n} plane bytes, got {len(planes)}")
    out = bytearray(n)
    if n == 0:
        return out

    lib = _load_native()
    if lib is not None:
        src_addr, _a = _ptr(planes)
        dst_addr, _b = _ptr(out)
        scr_addr, _c = _get_scratch(lib.lmz_scratch_size(nelem, esize))
        rc = lib.lmz_merge(src_addr, dst_addr, nelem, esize, scr_addr)
        if rc != 0:
            raise ValueError(f"unsupported element size {esize}")
        return out

    try:
        import numpy as np

        src = np.frombuffer(planes, dtype=np.uint8).reshape(esize, nelem)
        np.frombuffer(out, dtype=np.uint8).reshape(nelem, esize)[:] = src.T
        return out
    except ImportError:
        pass

    src = bytes(planes)
    for k in range(esize):
        out[k::esize] = src[k * nelem:(k + 1) * nelem]
    return out


def merge_planes(planes, esize: int, nelem: int, out):
    """Interleave planes given as (buffer, offset) pairs into `out`.

    The decoder gets its planes as separate objects: some are freshly
    decompressed, others are slices of the payload that were stored as-is.
    Taking them where they lie avoids copying the whole chunk into one
    contiguous block purely to satisfy the kernel's calling convention.

    Returns a memoryview of the filled portion of `out`.
    """
    n = nelem * esize
    if len(planes) != esize:
        raise ValueError(f"expected {esize} planes, got {len(planes)}")
    if len(out) < n:
        raise ValueError(f"output buffer holds {len(out)} bytes, need {n}")
    view = memoryview(out)
    if n == 0:
        return view[:0]
    if esize == 1:
        obj, off = planes[0]
        view[:n] = memoryview(obj)[off:off + n]
        return view[:n]

    lib = _load_native()
    if lib is not None:
        keep = []
        addrs = (ctypes.c_void_p * esize)()
        for k, (obj, off) in enumerate(planes):
            addr, ref = _ptr(obj)
            keep.append(ref)
            addrs[k] = addr + off
        dst, ref = _ptr(out)
        keep.append(ref)
        scr, ref = _get_scratch(lib.lmz_scratch_size(nelem, esize))
        keep.append(ref)
        rc = lib.lmz_merge_planes(addrs, dst, nelem, esize, scr)
        if rc != 0:
            raise ValueError(f"unsupported element size {esize}")
        return view[:n]

    gathered = bytearray(n)
    for k, (obj, off) in enumerate(planes):
        gathered[k * nelem:(k + 1) * nelem] = memoryview(obj)[off:off + nelem]
    view[:n] = merge(gathered, esize, nelem)
    return view[:n]


def split_bf16(src) -> bytearray:
    """Split BF16 words into an exponent plane and a sign+mantissa plane.

    Unlike the byte split this cuts on the float's own field boundaries, so
    the 8-bit exponent stays in one piece instead of being divided across
    two alphabets. Output is the two planes back to back.
    """
    n = len(src)
    if n % 2:
        raise ValueError(f"buffer of {n} bytes is not a whole number of BF16 values")
    nelem = n // 2
    out = bytearray(n)
    if nelem == 0:
        return out
    lib = _load_native()
    if lib is not None:
        s, _a = _ptr(src)
        d, _b = _ptr(out)
        lib.lmz_split_bf16(s, d, d + nelem, nelem)
        return out
    try:
        import numpy as np

        w = np.frombuffer(src, dtype="<u2")
        view = np.frombuffer(out, dtype=np.uint8)
        view[:nelem] = ((w >> 7) & 0xFF).astype(np.uint8)
        view[nelem:] = (((w >> 8) & 0x80) | (w & 0x7F)).astype(np.uint8)
        return out
    except ImportError:
        pass

    # Last resort. A per-element Python loop is far too slow for a real model,
    # but it keeps the tool correct rather than absent.
    mv = memoryview(src)
    for i in range(nelem):
        w = mv[2 * i] | (mv[2 * i + 1] << 8)
        out[i] = (w >> 7) & 0xFF
        out[nelem + i] = ((w >> 8) & 0x80) | (w & 0x7F)
    return out


def merge_bf16(planes, nelem: int, out=None):
    """Rebuild BF16 words from the two field planes. Inverse of `split_bf16`."""
    n = nelem * 2
    dst = out if (out is not None and len(out) >= n) else bytearray(n)
    if nelem == 0:
        return memoryview(dst)[:0]
    lib = _load_native()
    if lib is not None:
        keep = []
        addrs = []
        for obj, off in planes:
            a, ref = _ptr(obj)
            keep.append(ref)
            addrs.append(a + off)
        d, ref = _ptr(dst)
        keep.append(ref)
        lib.lmz_merge_bf16(addrs[0], addrs[1], d, nelem)
        return memoryview(dst)[:n]
    try:
        import numpy as np

        def plane(idx):
            obj, off = planes[idx]
            return np.frombuffer(obj, dtype=np.uint8, count=nelem, offset=off)

        pa = plane(0).astype(np.uint16)
        pb = plane(1).astype(np.uint16)
        words = ((pb & 0x80) << 8) | (pa << 7) | (pb & 0x7F)
        np.frombuffer(dst, dtype="<u2", count=nelem)[:] = words
        return memoryview(dst)[:n]
    except ImportError:
        pass

    pa = memoryview(planes[0][0])[planes[0][1]:planes[0][1] + nelem]
    pb = memoryview(planes[1][0])[planes[1][1]:planes[1][1] + nelem]
    for i in range(nelem):
        w = ((pb[i] & 0x80) << 8) | (pa[i] << 7) | (pb[i] & 0x7F)
        dst[2 * i] = w & 0xFF
        dst[2 * i + 1] = w >> 8
    return memoryview(dst)[:n]


def split_stride(src, period: int) -> bytearray:
    """Deinterleave fixed-period records into one plane per byte position.

    For periods the power-of-two kernel cannot express, like Q8_0's 34-byte
    blocks or Q6_K's 210-byte ones. Output is position-major: plane k occupies
    [k*n, (k+1)*n).
    """
    n = len(src)
    if period <= 0 or period > MAX_PERIOD:
        raise ValueError(f"unsupported record period {period}")
    if n % period:
        raise ValueError(f"buffer of {n} bytes is not whole {period}-byte records")
    nblocks = n // period
    out = bytearray(n)
    if n == 0:
        return out
    lib = _load_native()
    if lib is not None:
        sa, _a = _ptr(src)
        da, _b = _ptr(out)
        if lib.lmz_split_stride(sa, da, nblocks, period) != 0:
            raise ValueError(f"unsupported record period {period}")
        return out
    sb = bytes(src)
    for k in range(period):
        out[k * nblocks:(k + 1) * nblocks] = sb[k::period]
    return out


def merge_stride(streams, nblocks: int, period: int, out=None):
    """Re-interleave planes into fixed-period records. Inverse of split_stride.

    `streams` is one (buffer, offset) per byte position, so decoded planes
    are consumed where they lie.
    """
    n = nblocks * period
    if len(streams) != period:
        raise ValueError(f"expected {period} planes, got {len(streams)}")
    dst = out if (out is not None and len(out) >= n) else bytearray(n)
    if n == 0:
        return memoryview(dst)[:0]
    lib = _load_native()
    if lib is not None:
        keep = []
        addrs = (ctypes.c_void_p * period)()
        for k, (obj, off) in enumerate(streams):
            addr, ref = _ptr(obj)
            keep.append(ref)
            addrs[k] = (addr + off) if addr else 0
        da, _d = _ptr(dst)
        if lib.lmz_merge_stride(addrs, da, nblocks, period) != 0:
            raise ValueError(f"unsupported record period {period}")
        return memoryview(dst)[:n]
    view = memoryview(dst)
    for k, (obj, off) in enumerate(streams):
        view[k:n:period] = memoryview(obj)[off:off + nblocks]
    return view[:n]


def bucket_lut(hist, k: int) -> bytes:
    """Contiguous equal-mass buckets over the byte alphabet.

    A pure function of the histogram, identical between the native and Python
    paths, so an encoder and decoder that see the same symbol counts always
    derive the same map and it never needs to be stored.
    """
    lib = _load_native()
    if lib is not None:
        arr = (ctypes.c_uint64 * 256)(*hist)
        lut = ctypes.create_string_buffer(256)
        if lib.lmz_bucket_lut(ctypes.byref(arr), k, lut) != 0:
            raise ValueError(f"unsupported bucket count {k}")
        return lut.raw
    if not (1 <= k <= 32):
        raise ValueError(f"unsupported bucket count {k}")
    total = sum(hist)
    if total == 0:
        return bytes(256)
    out = bytearray(256)
    acc = 0
    b = 0
    for s in range(256):
        out[s] = b
        acc += hist[s]
        while b + 1 < k and acc * k >= (b + 1) * total:
            b += 1
    return bytes(out)


def bucket_counts(hist, lut: bytes, k: int) -> list[int]:
    """Elements per bucket, from the context histogram and the bucket map."""
    counts = [0] * k
    for s in range(256):
        counts[lut[s]] += hist[s]
    return counts


def bucket_partition(ctx, val, lut: bytes, k: int):
    """Group val[i] into per-bucket segments by ctx[i]'s bucket.

    Returns (buffer of concatenated segments, per-bucket lengths). Order is
    preserved within each bucket, which is what makes it invertible.
    """
    n = len(ctx)
    if len(val) != n:
        raise ValueError("context and value planes differ in length")
    out = bytearray(n)
    lib = _load_native()
    if lib is not None:
        counts = (ctypes.c_uint64 * k)()
        ca, _a = _ptr(ctx)
        va, _b = _ptr(val)
        la, _c = _ptr(lut)
        oa, _d = _ptr(out)
        if lib.lmz_bucket_partition(ca, va, n, la, k, oa, ctypes.byref(counts)) != 0:
            raise ValueError(f"unsupported bucket count {k}")
        return out, list(counts)
    counts = bucket_counts(histogram(ctx), lut, k)
    cur = [0] * k
    pos = 0
    for b in range(k):
        cur[b] = pos
        pos += counts[b]
    cb = bytes(ctx)
    vb = bytes(val)
    for i in range(n):
        b = lut[cb[i]]
        out[cur[b]] = vb[i]
        cur[b] += 1
    return out, counts


def bucket_unpartition(ctx, streams, lut: bytes, k: int, out=None):
    """Inverse of bucket_partition. `streams` is a list of (buffer, offset)."""
    n = len(ctx)
    dst = out if (out is not None and len(out) >= n) else bytearray(n)
    if n == 0:
        return memoryview(dst)[:0]
    lib = _load_native()
    if lib is not None:
        keep = []
        addrs = (ctypes.c_void_p * k)()
        for b, (obj, off) in enumerate(streams):
            addr, ref = _ptr(obj)
            keep.append(ref)
            addrs[b] = (addr + off) if addr else 0
        ca, _a = _ptr(ctx)
        la, _b = _ptr(lut)
        da, _c = _ptr(dst)
        if lib.lmz_bucket_unpartition(ca, addrs, n, la, k, da) != 0:
            raise ValueError(f"unsupported bucket count {k}")
        return memoryview(dst)[:n]
    cur = [off for (_obj, off) in streams]
    views = [memoryview(obj) for (obj, _off) in streams]
    cb = bytes(ctx)
    for i in range(n):
        b = lut[cb[i]]
        dst[i] = views[b][cur[b]]
        cur[b] += 1
    return memoryview(dst)[:n]


def xor_bytes(a, b, out=None):
    """a ^ b, byte for byte. Both must be the same length.

    This is the whole of a delta chunk's arithmetic. XOR rather than
    subtraction because a borrow propagates into the high byte and destroys
    the very thing being exploited -- that the high byte usually does not
    change at all. Measured against integer subtraction and against an
    order-preserving monotone map on real checkpoints and fine-tunes, plain
    XOR won every time.
    """
    n = len(a)
    if len(b) != n:
        raise ValueError(f"cannot xor {n} bytes against {len(b)}")
    dst = out if (out is not None and len(out) >= n) else bytearray(n)
    if n == 0:
        return memoryview(dst)[:0]

    lib = _load_native()
    if lib is not None:
        aa, _x = _ptr(a)
        bb, _y = _ptr(b)
        dd, _z = _ptr(dst)
        lib.lmz_xor(aa, bb, n, dd)
        return memoryview(dst)[:n]
    try:
        import numpy as np

        na = np.frombuffer(bytes(a), dtype=np.uint8)
        nb = np.frombuffer(bytes(b), dtype=np.uint8)
        np.frombuffer(dst, dtype=np.uint8, count=n)[:] = na ^ nb
        return memoryview(dst)[:n]
    except ImportError:
        pass

    ba, bb = bytes(a), bytes(b)
    dst[:n] = (int.from_bytes(ba, "big") ^ int.from_bytes(bb, "big")).to_bytes(
        n, "big")
    return memoryview(dst)[:n]


# A Q4_K/Q5_K super-block: 256 weights in 8 sub-blocks, each with a 6-bit
# scale and a 6-bit min packed into a shared 12-byte field.
K4_SUBBLOCKS = 8
K4_SCALE_BYTES = 12
K4_QUANT_BYTES = 128


def k4_scales(planes, nblocks: int):
    """Unpack ggml's 12 packed scale bytes into 8 scale and 8 min planes.

    `planes` is the 12 byte-planes of the scales field, back to back. Returns
    (scales, mins), each 8 planes of `nblocks`. Four of each pair straddle a
    byte boundary, which is why a byte-plane split cannot see them.

    Used only to derive a coding context, never to reconstruct: the packed
    bytes are still coded as their own field.
    """
    n = K4_SUBBLOCKS * nblocks
    sc, mn = bytearray(n), bytearray(n)
    if nblocks == 0:
        return sc, mn
    if len(planes) < K4_SCALE_BYTES * nblocks:
        raise ValueError("scale field is shorter than 12 planes")

    lib = _load_native()
    if lib is not None:
        sp, _a = _ptr(planes)
        da, _b = _ptr(sc)
        db, _c = _ptr(mn)
        lib.lmz_k4_scales(sp, nblocks, da, db)
        return sc, mn
    try:
        import numpy as np

        sp = np.frombuffer(bytes(planes), dtype=np.uint8,
                           count=K4_SCALE_BYTES * nblocks).reshape(12, nblocks)
        a = np.frombuffer(sc, dtype=np.uint8).reshape(8, nblocks)
        b = np.frombuffer(mn, dtype=np.uint8).reshape(8, nblocks)
        a[0:4] = sp[0:4] & 63
        b[0:4] = sp[4:8] & 63
        a[4:8] = (sp[8:12] & 0x0F) | ((sp[0:4] >> 6) << 4)
        b[4:8] = (sp[8:12] >> 4) | ((sp[4:8] >> 6) << 4)
        return sc, mn
    except ImportError:
        pass

    mv = memoryview(planes)
    for j in range(4):
        a, b = mv[j * nblocks:(j + 1) * nblocks], mv[(j + 4) * nblocks:(j + 5) * nblocks]
        for i in range(nblocks):
            sc[j * nblocks + i] = a[i] & 63
            mn[j * nblocks + i] = b[i] & 63
    for j in range(4, 8):
        lo = mv[(j + 4) * nblocks:(j + 5) * nblocks]
        hs = mv[(j - 4) * nblocks:(j - 3) * nblocks]
        hm = mv[j * nblocks:(j + 1) * nblocks]
        for i in range(nblocks):
            sc[j * nblocks + i] = (lo[i] & 0x0F) | ((hs[i] >> 6) << 4)
            mn[j * nblocks + i] = (lo[i] >> 4) | ((hm[i] >> 6) << 4)
    return sc, mn


def k4_pack(qplanes, nblocks: int) -> bytearray:
    """Regroup 128 quant planes so each output plane holds one sub-block.

    ggml puts sub-block 2g in the low nibbles of quant bytes [32g, 32g+32) and
    2g+1 in their high nibbles, so every byte plane mixes two alphabets. This
    pairs two nibbles of the same sub-block into each output byte: one symbol
    per two quants, so the coder still runs at byte rate, but with a single
    context per byte.
    """
    n = K4_QUANT_BYTES * nblocks
    out = bytearray(n)
    if nblocks == 0:
        return out
    if len(qplanes) < n:
        raise ValueError("quant field is shorter than 128 planes")

    lib = _load_native()
    if lib is not None:
        sa, _a = _ptr(qplanes)
        da, _b = _ptr(out)
        lib.lmz_k4_pack(sa, nblocks, da)
        return out
    try:
        import numpy as np

        q = np.frombuffer(bytes(qplanes), dtype=np.uint8,
                          count=n).reshape(128, nblocks)
        o = np.frombuffer(out, dtype=np.uint8).reshape(128, nblocks)
        for s in range(8):
            g, sh = s >> 1, 4 if (s & 1) else 0
            a = (q[32 * g:32 * g + 32:2] >> sh) & 0x0F
            b = (q[32 * g + 1:32 * g + 32:2] >> sh) & 0x0F
            o[16 * s:16 * s + 16] = a | (b << 4)
        return out
    except ImportError:
        pass

    mv = memoryview(qplanes)
    for s in range(8):
        g, sh = s >> 1, 4 if (s & 1) else 0
        for i in range(16):
            a = mv[(32 * g + 2 * i) * nblocks:(32 * g + 2 * i + 1) * nblocks]
            b = mv[(32 * g + 2 * i + 1) * nblocks:(32 * g + 2 * i + 2) * nblocks]
            base = (16 * s + i) * nblocks
            for k in range(nblocks):
                out[base + k] = ((a[k] >> sh) & 0x0F) | (((b[k] >> sh) & 0x0F) << 4)
    return out


def k4_unpack(packed, nblocks: int) -> bytearray:
    """Inverse of k4_pack: sub-block planes back to ggml's quant planes."""
    n = K4_QUANT_BYTES * nblocks
    out = bytearray(n)
    if nblocks == 0:
        return out
    if len(packed) < n:
        raise ValueError("packed field is shorter than 128 planes")

    lib = _load_native()
    if lib is not None:
        sa, _a = _ptr(packed)
        da, _b = _ptr(out)
        lib.lmz_k4_unpack(sa, nblocks, da)
        return out
    try:
        import numpy as np

        p = np.frombuffer(bytes(packed), dtype=np.uint8,
                          count=n).reshape(128, nblocks)
        o = np.frombuffer(out, dtype=np.uint8).reshape(128, nblocks)
        for g in range(4):
            lo = p[32 * g:32 * g + 16]
            hi = p[32 * g + 16:32 * g + 32]
            for half in (0, 1):
                sh = 4 * half
                o[32 * g + half:32 * g + 32:2] = \
                    ((lo >> sh) & 0x0F) | (((hi >> sh) & 0x0F) << 4)
        return out
    except ImportError:
        pass

    mv = memoryview(packed)
    for c in range(128):
        g, t = c >> 5, c & 31
        sh = 4 * (t & 1)
        lo = mv[(16 * (2 * g) + (t >> 1)) * nblocks:][:nblocks]
        hi = mv[(16 * (2 * g + 1) + (t >> 1)) * nblocks:][:nblocks]
        base = c * nblocks
        for k in range(nblocks):
            out[base + k] = ((lo[k] >> sh) & 0x0F) | (((hi[k] >> sh) & 0x0F) << 4)
    return out


def have_rans() -> bool:
    """rANS needs the native kernel; there is no practical pure-Python coder."""
    return _load_native() is not None


# Encoding runs backwards from the end of a scratch buffer, so it needs room
# for the worst case up front. Reused per thread.
_rans_scratch = threading.local()


def rans_encode(src) -> bytes | None:
    """Order-0 rANS encode. Returns None if unavailable or the input is empty."""
    lib = _load_native()
    if lib is None or len(src) == 0:
        return None
    n = len(src)
    cap = lib.lmz_rans_bound(n)
    buf = getattr(_rans_scratch, "buf", None)
    if buf is None or len(buf) < cap:
        buf = _rans_scratch.buf = bytearray(cap)
    src_addr, _a = _ptr(src)
    dst_addr, _b = _ptr(buf)
    written = lib.lmz_rans_encode(src_addr, n, dst_addr, cap)
    if written < 0:
        return None
    return bytes(memoryview(buf)[:written])


def rans_decode(payload, n: int, out=None):
    """Decode a rANS stream to exactly `n` bytes."""
    lib = _load_native()
    if lib is None:
        raise RuntimeError(
            "this archive uses lmz's rANS coder, which needs the native kernel; "
            "a C compiler (cc/gcc/clang) is required to build it")
    dst = out if (out is not None and len(out) >= n) else bytearray(n)
    if n == 0:
        return memoryview(dst)[:0]
    src_addr, _a = _ptr(payload)
    dst_addr, _b = _ptr(dst)
    if lib.lmz_rans_decode(src_addr, len(payload), dst_addr, n) != 0:
        raise ValueError("malformed rANS stream")
    return memoryview(dst)[:n]


def histogram(buf) -> list[int]:
    """Byte-value counts, 256 entries."""
    if len(buf) == 0:
        return [0] * 256
    lib = _load_native()
    if lib is not None:
        hist = (ctypes.c_uint64 * 256)()
        addr, _keep = _ptr(buf)
        lib.lmz_hist(addr, len(buf), ctypes.byref(hist))
        return list(hist)
    try:
        import numpy as np

        return np.bincount(np.frombuffer(buf, dtype=np.uint8), minlength=256).tolist()
    except ImportError:
        counts = [0] * 256
        for b in bytes(buf):
            counts[b] += 1
        return counts
