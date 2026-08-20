"""The CUDA decoder, if this machine has one.

lmz decodes on the CPU and always will; this is the path for the case the CPU
cannot serve, which is a model that has to arrive in VRAM. The kernel decodes
lmz's own rANS -- the same coded bytes, the same 8-state interleave, no format
change -- so a GPU and a CPU decode of the same archive are the same bytes.

Nothing here is required. `pip install lmzip` installs no CUDA, imports no CUDA
and works unchanged without a GPU: `available()` reports why, and every caller
falls back. Set LMZ_NO_GPU to force that path.

The unit of work is a *batch* of streams, not one stream. A single lmz rANS
stream has 8 interleaved states and therefore exactly 8 lanes of work in it, so
one stream cannot fill a GPU no matter how large it is; what fills a GPU is
many streams at once. That is a property of the format, not of this code.
"""

from __future__ import annotations

import ctypes
import os
import threading

__all__ = ["available", "backend", "decode_batch", "grain", "header_bytes",
           "last_error", "pad_bytes", "state"]

# Kept in step with ABI_VERSION in lmzgpu.cu. A library built from an older
# source is ignored rather than called with the wrong argument list.
ABI_VERSION = 1

OK = 0
ENODEV = -1
EUNSUPPORTED = -2
EBADSTREAM = -3
ECUDA = -4

_lock = threading.Lock()
_lib = None
_state = "unloaded"
_why = "not probed"


def _load():
    global _lib, _state, _why
    if _state != "unloaded":
        return _lib
    with _lock:
        if _state != "unloaded":
            return _lib
        if os.environ.get("LMZ_NO_GPU"):
            _state, _why = "disabled", "disabled by LMZ_NO_GPU"
            return None
        try:
            from . import build as _build

            # build() first, and only ask why afterwards: an already-built
            # library returns without running nvcc at all, and the reasons
            # cost a subprocess each to establish.
            path = _build.build()
            if not path:
                _state = "unavailable"
                if _build.find_compiler() is None:
                    _why = "no nvcc (a CUDA toolkit is needed once, to build)"
                else:
                    _why = "the CUDA decoder did not build"
                    if _build.last_error:
                        _why += f" ({_build.last_error})"
                return None
            lib = ctypes.CDLL(path)
            v, c_int, c_uint = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint
            lib.lmz_gpu_abi_version.restype = c_int
            if lib.lmz_gpu_abi_version() != ABI_VERSION:
                _state, _why = "abi-mismatch", "the built CUDA decoder is from another source"
                return None
            for name in ("lmz_gpu_grain", "lmz_gpu_header_bytes", "lmz_gpu_pad_bytes"):
                getattr(lib, name).restype = c_int
            lib.lmz_gpu_device_name.restype = c_int
            lib.lmz_gpu_device_name.argtypes = [ctypes.c_char_p, c_int]
            lib.lmz_gpu_last_error.restype = ctypes.c_char_p
            lib.lmz_gpu_validate.restype = c_int
            lib.lmz_gpu_validate.argtypes = [v, v, c_uint]
            lib.lmz_gpu_decode_batch.restype = c_int
            lib.lmz_gpu_decode_batch.argtypes = [v, v, ctypes.c_size_t, v,
                                                 c_uint, c_uint, v]
            lib.lmz_gpu_decode_batch_dev.restype = c_int
            lib.lmz_gpu_decode_batch_dev.argtypes = [v, v, v, c_uint, c_uint, v,
                                                     v, c_uint]
            name = ctypes.create_string_buffer(256)
            if lib.lmz_gpu_device_name(name, 256) != OK:
                # Built, but there is no device to run it on: a build host, or
                # a container with the toolkit and no card passed through.
                _state, _why = "unavailable", "nvcc is present but no CUDA device is"
                return None
            _lib = lib
            _state = "cuda:" + name.value.decode()
            _why = ""
        except Exception as exc:
            _lib = None
            _state, _why = "unavailable", f"{type(exc).__name__}: {exc}"
    return _lib


def available() -> tuple[bool, str]:
    """Whether the GPU decoder can run, and if not, the reason.

    Probes on the first call, which means building the library if nvcc is here
    and it has not been built yet -- about a second, once per install.
    """
    _load()
    return (_lib is not None), _why


def state() -> str:
    """What is known about the GPU decoder without going and finding out.

    `backends()` reports this rather than `backend()` so that asking a
    diagnostic question does not run a compiler as a side effect.
    """
    return _state if _state != "unloaded" else "not probed"


def backend() -> str:
    """Name of the GPU backend, for diagnostics. Mirrors kernels.backend()."""
    _load()
    return _state if _lib is not None else "unavailable"


def last_error() -> str:
    """Why the last decode declined, in the library's words. Empty if it did not.

    Every failure the decoder can have is one `decode_batch` turns into None
    and the caller turns into a CPU decode, so without this the difference
    between "no card", "this block size wants more shared memory than the
    device grants" and "the launch was rejected" never reaches anyone.
    """
    lib = _load()
    if lib is None:
        return _why
    return lib.lmz_gpu_last_error().decode(errors="replace")


def _const(name: str, default: int) -> int:
    lib = _load()
    return getattr(lib, name)() if lib is not None else default


def grain() -> int:
    """Bytes a group of 8 lanes retires per step; a plane must be a multiple."""
    return _const("lmz_gpu_grain", 128)


def header_bytes() -> int:
    """Size of a per-chunk frequency table: magic, reserved, 256 uint16."""
    return _const("lmz_gpu_header_bytes", 516)


def pad_bytes() -> int:
    """Readable slack the kernel prefetches past the last stream."""
    return _const("lmz_gpu_pad_bytes", 576)


def _ptr(obj):
    """Address of a buffer plus a reference that must outlive the call."""
    if obj is None or len(obj) == 0:
        return None, None
    if type(obj) is bytes:
        p = ctypes.c_char_p(obj)
        return ctypes.cast(p, ctypes.c_void_p).value, (p, obj)
    try:
        arr = (ctypes.c_ubyte * len(obj)).from_buffer(obj)
        return ctypes.addressof(arr), arr
    except TypeError:
        b = bytes(obj)
        p = ctypes.c_char_p(b)
        return ctypes.cast(p, ctypes.c_void_p).value, (p, b)


def decode_batch(streams, offsets, nstr: int, plane: int, out=None, header=None):
    """Decode `nstr` lmz rANS streams to `plane` bytes each. None if unavailable.

    `streams` holds the streams concatenated exactly as an archive stores them:
    no padding and no alignment. The kernel prefetches past its cursor, and the
    slack for that is added on the device here, so a caller of this function
    owes nothing -- it is `lmz_gpu_decode_batch_dev`, which takes memory the
    caller allocated, that needs `pad_bytes()` of readable space after the last
    stream. `offsets` is 2*nstr little-endian uint64, (byte offset, byte
    length) per stream; only the offset is read.

    `header` is a 516-byte frequency table shared by every stream in the batch,
    for the layout that factors the table out of the chunk. Passing None means
    each stream carries its own, which is what an archive written today holds.

    Returns a memoryview of nstr*plane bytes, or None when there is no GPU
    decoder or the batch is a shape it does not take -- both of which mean the
    caller should decode on the CPU. Raises ValueError on a stream that is not
    lmz rANS at all, because that is not a fallback, it is corruption.
    """
    lib = _load()
    if lib is None:
        return None
    if nstr == 0:
        return memoryview(b"")
    # The shape rules are the library's, not this file's: it is the side that
    # knows the grain and the device's shared-memory budget, and duplicating
    # the check here would shadow the reason it gives for declining.
    if nstr < 0:
        raise ValueError("nstr is negative")
    if len(offsets) < nstr * 16:
        raise ValueError(f"offsets holds {len(offsets)} bytes, need {nstr * 16}")
    if header is not None and len(header) != header_bytes():
        raise ValueError(f"a shared table is {header_bytes()} bytes, not {len(header)}")

    need = nstr * plane
    dst = out if (out is not None and len(out) >= need) else bytearray(need)
    s_addr, _a = _ptr(streams)
    o_addr, _b = _ptr(offsets)
    d_addr, _c = _ptr(dst)
    h_addr, _d = _ptr(header)

    rc = lib.lmz_gpu_decode_batch(h_addr, s_addr, len(streams), o_addr,
                                  nstr, plane, d_addr)
    if rc == OK:
        return memoryview(dst)[:need]
    if rc == EBADSTREAM:
        raise ValueError("malformed rANS stream")
    return None
