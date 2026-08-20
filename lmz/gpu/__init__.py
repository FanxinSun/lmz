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
import subprocess
import sys
import threading

__all__ = ["available", "backend", "decode_batch", "grain", "header_bytes",
           "last_error", "pad_bytes", "state", "verify"]

# Kept in step with ABI_VERSION in lmzgpu.cu. A library built from an older
# source is ignored rather than called with the wrong argument list.
ABI_VERSION = 1

OK = 0
ENODEV = -1
EUNSUPPORTED = -2
EBADSTREAM = -3
ECUDA = -4

# The dangerous part of using a GPU is the first touch of the driver, and it
# is dangerous in a way no return code reaches: a driver that is half-removed
# or mid-upgrade leaves libcuda.so.1 on disk with an initialiser that faults,
# so `ctypes.CDLL` takes the whole interpreter down. That was not hypothetical
# -- it happened on the machine this was written on, when a driver update
# landed underneath a running session and left nvidia-smi answering nothing.
#
# lmz's promise is that the GPU decoder is optional in every direction and the
# CPU path is unaffected. A segmentation fault in someone's process is not a
# fallback, so the first load happens in a child that is allowed to die.
_PROBE_SOURCE = """
import ctypes, sys
try:
    lib = ctypes.CDLL(sys.argv[1])
    lib.lmz_gpu_abi_version.restype = ctypes.c_int
    if lib.lmz_gpu_abi_version() != ABI:
        sys.exit(2)
    lib.lmz_gpu_device_name.restype = ctypes.c_int
    lib.lmz_gpu_device_name.argtypes = [ctypes.c_char_p, ctypes.c_int]
    buf = ctypes.create_string_buffer(256)
    if lib.lmz_gpu_device_name(buf, 256) != 0:
        sys.exit(3)
    sys.stdout.write(buf.value.decode())
except Exception as exc:
    sys.stderr.write(f"{type(exc).__name__}: {exc}")
    sys.exit(4)
"""


def _probe_elsewhere(path: str) -> tuple[bool, str]:
    """Load the library in a child process. (ok, device name) or (False, why).

    Costs one interpreter start and one CUDA context, once, and only on a
    machine that has both a toolkit and something that looks like a driver.
    Everywhere else `build()` has already declined and none of this runs.
    """
    src = _PROBE_SOURCE.replace("ABI", str(ABI_VERSION))
    try:
        p = subprocess.run([sys.executable, "-c", src, path],
                           capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return False, "the CUDA driver did not answer within 120s"
    except Exception as exc:
        return False, f"could not probe the CUDA driver: {exc}"
    if p.returncode == 0:
        return True, p.stdout.strip()
    if p.returncode == 2:
        return False, "the built CUDA decoder is from another source"
    if p.returncode == 3:
        return False, "nvcc is present but no CUDA device is"
    if p.returncode == 4:
        return False, (p.stderr.strip().splitlines() or ["the library did not load"])[-1]
    # Anything else is a death rather than a decision: a negative code is the
    # POSIX signal that killed it, and Windows reports its own status codes.
    sig = f"signal {-p.returncode}" if p.returncode < 0 else f"status {p.returncode:#x}"
    return False, (f"the CUDA driver crashed while loading ({sig}); the driver "
                   f"is broken or mid-upgrade, so lmz is staying on the CPU")


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

            path = _build.build()
            if not path:
                # build() states its own reason, and it is the side that knows
                # whether the compiler was reached at all.
                _state = "unavailable"
                _why = _build.last_error or "the CUDA decoder did not build"
                return None
            # Nothing above has touched the driver. This is where that happens,
            # and it happens somewhere a fault cannot reach this process.
            safe, _probed = _probe_elsewhere(path)
            if not safe:
                _state, _why = "unavailable", _probed
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
            device = _probed
            if not _selftest(lib):
                _state = "unavailable"
                _why = (f"the CUDA decoder did not reproduce the CPU decoder's "
                        f"bytes on {device}; staying on the CPU")
                return None
            _lib = lib
            _state = "cuda:" + device
            _why = ""
        except Exception as exc:
            _lib = None
            _state, _why = "unavailable", f"{type(exc).__name__}: {exc}"
    return _lib


def _selftest(lib) -> bool:
    """Decode a stream this machine just encoded, and check the CPU agrees.

    This kernel has been run on one card. It compiles for Turing through
    Blackwell and is clean under compute-sanitizer, but neither of those is
    the same as having executed it on a Turing -- and a decoder that is
    silently wrong is far worse than one that is absent, because the caller
    gets weights rather than an error.

    So the first thing the GPU decoder ever does is decode something whose
    answer is already known, and a device that disagrees is not used. It costs
    one launch, after a CUDA context that was being created anyway.
    """
    from .. import kernels

    if not kernels.have_rans():
        # No native coder means no rANS archives exist to decode, so there is
        # nothing to be wrong about and nothing to check against.
        return True
    plane, nstr = 128, 8
    # Skewed enough that the coder takes it, and mixed enough that the
    # frequency table has more than one entry to get wrong.
    plain = bytes(((i * 37) & 0x0F) if i % 3 else 0x40 for i in range(plane))
    coded = kernels.rans_encode(plain, kernels.histogram(plain))
    if not coded:
        return True
    import struct

    streams, offsets = bytearray(), bytearray()
    for _ in range(nstr):
        offsets += struct.pack("<QQ", len(streams), len(coded))
        streams += coded
    out = bytearray(nstr * plane)
    s_addr, _a = _ptr(streams)
    o_addr, _b = _ptr(bytes(offsets))
    d_addr, _c = _ptr(out)
    rc = lib.lmz_gpu_decode_batch(None, s_addr, len(streams), o_addr,
                                  nstr, plane, d_addr)
    if rc != OK:
        # It declined rather than answered -- no device, no shared memory for
        # this shape. That is the CPU path working as designed, not a wrong
        # decoder, so do not brand the card as broken.
        return rc != EBADSTREAM
    return bytes(out) == plain * nstr


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


# Byte-value distributions that break frequency tables, as translate() tables
# over uniform random bytes -- exact shapes at C speed. Shared with the test
# suite's idea of what is worth checking.
_SHAPES = {
    "single symbol": bytes([0x5A]) * 256,
    "two symbols": bytes([0x01]) * 128 + bytes([0xFE]) * 128,
    "near uniform": bytes(range(256)),
    "dominant + tail": bytes([0x20]) * 250 + bytes(range(6)),
    "skewed": bytes([0x40]) * 170 + bytes(range(16)) * 5 + bytes([0x7F]) * 6,
}


def verify(quick: bool = False) -> dict:
    """Decode every awkward shape on this device and check the CPU agrees.

    The kernel has been *run* on one architecture. It is sanitizer-clean and
    compiles from sm_75 to sm_121, but Turing gets genuinely different code --
    cp.async has no instruction there and the intrinsic falls back to a
    synchronous copy -- so an untested card is not merely untested hardware.

    This exists so anyone holding one can settle it in a few seconds and paste
    the answer. It needs no data files and no network: the streams are built
    by lmz's own encoder here and checked against lmz's own decoder, so the
    oracle travels with the question.
    """
    import random
    import struct
    import time

    from .. import __version__, kernels

    report = {"lmz": __version__, "device": None, "ok": False, "checked": 0,
              "failures": [], "gbps": None, "why": ""}
    ok, why = available()
    if not ok:
        report["why"] = why
        return report
    report["device"] = backend().removeprefix("cuda:")
    if not kernels.have_rans():
        report["why"] = "no native coder, so there is nothing to check against"
        return report

    def batch(table, plane, nstr, seed):
        rnd = random.Random(seed)
        bufs, streams, offsets = [], bytearray(), bytearray()
        for _ in range(nstr):
            buf = rnd.randbytes(plane).translate(table)
            coded = kernels.rans_encode(buf, kernels.histogram(buf))
            if coded is None:
                return None
            offsets += struct.pack("<QQ", len(streams), len(coded))
            streams += coded
            bufs.append(buf)
        return bufs, streams, bytes(offsets)

    for name, table in _SHAPES.items():
        for plane in (grain(), 4096) if not quick else (4096,):
            for nstr in (1, 17, 129):
                made = batch(table, plane, nstr, hash((name, plane, nstr)) & 0xFFFF)
                if made is None:
                    continue
                bufs, streams, offsets = made
                out = decode_batch(streams, offsets, nstr, plane)
                if out is None:
                    report["failures"].append(
                        f"{name} plane={plane} n={nstr}: declined -- {last_error()}")
                    continue
                for i, want in enumerate(bufs):
                    if bytes(out[i * plane:(i + 1) * plane]) != want:
                        report["failures"].append(
                            f"{name} plane={plane} n={nstr}: stream {i} differs")
                        break
                report["checked"] += 1

    # Throughput on a batch big enough to mean something, and every stream
    # distinct: decoding one stream twice finds its coded bytes in L2 and
    # reports a rate that does not exist.
    plane, nstr = 32768, 256 if quick else 2048
    made = batch(_SHAPES["skewed"], plane, nstr, 4242)
    if made is not None:
        bufs, streams, offsets = made
        out = bytearray(nstr * plane)
        decode_batch(streams, offsets, nstr, plane, out=out)   # warm the context
        t = time.perf_counter()
        got = decode_batch(streams, offsets, nstr, plane, out=out)
        dt = time.perf_counter() - t
        if got is not None and dt > 0:
            report["gbps"] = round(nstr * plane / dt / 1e9, 2)
        if got is None or bytes(out) != b"".join(bufs):
            report["failures"].append("throughput batch did not decode correctly")

    report["ok"] = report["checked"] > 0 and not report["failures"]
    return report


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
