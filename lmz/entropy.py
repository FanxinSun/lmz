"""Entropy coding backend.

Zstandard is the intended coder and ships in Python 3.14's standard library;
the third-party `zstandard` package is used when running on older versions.
Deflate is the last resort so the tool degrades rather than failing outright.

The method used for each stream is recorded in the archive, so a file written
by one backend stays readable by any build that has the corresponding decoder.
Which one wrote it changes nothing but speed: both emit ordinary zstd frames.
"""

from __future__ import annotations

import sys
import zlib

METHOD_STORED = 0
METHOD_ZSTD = 1
METHOD_DEFLATE = 2
# lmz's own order-0 rANS. Only worth using on byte planes, where the data is
# a plain symbol stream with no repeats for a general-purpose coder to find.
METHOD_RANS = 3

METHOD_NAMES = {METHOD_STORED: "stored", METHOD_ZSTD: "zstd",
                METHOD_DEFLATE: "deflate", METHOD_RANS: "rans"}

_zstd_compress = None
_zstd_decompress = None
BACKEND = "deflate"
ZstdTruncated = ValueError  # replaced with the backend's error type below

def _load_stdlib():
    from compression import zstd as _z          # Python 3.14+

    def compress(d, lvl):
        return _z.compress(d, lvl)

    def decompress(d):
        # A one-shot decompressor object is markedly faster than the
        # module-level decompress(), which loops looking for further frames.
        dec = _z.ZstdDecompressor()
        out = dec.decompress(d)
        if not dec.eof:
            raise _z.ZstdError("compressed stream ends mid-frame")
        return out

    return compress, decompress, _z.ZstdError, f"zstd {_z.zstd_version} (stdlib)"


def _load_package():
    import zstandard as _zs

    def compress(d, lvl):
        return _zs.ZstdCompressor(level=lvl).compress(d)

    def decompress(d):
        return _zs.ZstdDecompressor().decompress(d)

    return (compress, decompress, _zs.ZstdError,
            f"zstd {_zs.__version__} (zstandard)")


def _backend_order():
    """Which zstd binding to try first. Both are libzstd; only speed differs.

    On macOS, CPython's bundled zstd measured 103 MiB/s compressing where the
    same work through the `zstandard` package measured 767, on hardware that is
    otherwise the fastest of any platform tested here. lmz hands libzstd whole
    multi-megabyte chunks, so per-call binding overhead cannot account for a gap
    that size -- it is the bundled build. No other platform showed a gap, so the
    preference is narrowed to where the evidence is, and it costs nothing
    anywhere: with only one binding installed this changes no behaviour at all.
    """
    if sys.platform == "darwin":
        return (_load_package, _load_stdlib)
    return (_load_stdlib, _load_package)


for _loader in _backend_order():
    try:
        _zstd_compress, _zstd_decompress, ZstdTruncated, BACKEND = _loader()
        break
    except ImportError:
        continue

HAVE_ZSTD = _zstd_compress is not None

# What a new archive uses. Falls back only when zstd is genuinely absent.
DEFAULT_METHOD = METHOD_ZSTD if HAVE_ZSTD else METHOD_DEFLATE


class UnsupportedMethod(Exception):
    """The archive needs a codec this interpreter does not have."""


def compress(data, level: int = 1, method: int = -1) -> bytes:
    if method < 0:
        method = DEFAULT_METHOD
    if method == METHOD_RANS:
        from . import kernels

        out = kernels.rans_encode(data)
        if out is None:
            raise UnsupportedMethod("the rANS coder needs the native kernel")
        return out
    if method == METHOD_ZSTD:
        if not HAVE_ZSTD:
            raise UnsupportedMethod("zstd is not available")
        return _zstd_compress(data, level)
    if method == METHOD_DEFLATE:
        return zlib.compress(data, min(max(level, 1), 9))
    if method == METHOD_STORED:
        return bytes(data)
    raise UnsupportedMethod(f"unknown compression method {method}")


def decompress(data, method: int = -1, out_len: int | None = None):
    if method < 0:
        method = METHOD_ZSTD if HAVE_ZSTD else METHOD_DEFLATE
    if method == METHOD_RANS:
        from . import kernels

        if out_len is None:
            raise ValueError("decoding rANS requires the decoded length")
        return kernels.rans_decode(data, out_len)
    if method == METHOD_ZSTD:
        if not HAVE_ZSTD:
            raise UnsupportedMethod(
                "this archive uses zstd, which is unavailable here; "
                "use Python 3.14+ or install the 'zstandard' package")
        return _zstd_decompress(data)
    if method == METHOD_DEFLATE:
        return zlib.decompress(data)
    if method == METHOD_STORED:
        return bytes(data)
    raise UnsupportedMethod(f"unknown compression method {method}")
