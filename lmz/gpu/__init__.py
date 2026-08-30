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

__all__ = ["available", "backend", "cost_model", "decode_batch",
           "decode_batch_dev", "grain", "header_bytes", "last_error",
           "pad_bytes", "state", "verify"]

# Kept in step with ABI_VERSION in lmzgpu.cu. A library built from an older
# source is ignored rather than called with the wrong argument list.
ABI_VERSION = 1

# The shared-table kernel's shared-memory layout, mirroring lmzgpu.cu. A block
# asks for one LUT plus a slice per 8-lane group, and how many blocks a device
# can hold is what decides whether the published k interval brackets that
# device or only floors it. The values are checked against the kernel's own
# reported constants in the test suite, so this cannot drift silently.
PROB_SCALE = 1 << 12       # 4096 probability slots, one 32-bit LUT entry each
NST = 8                    # lanes in a group; the format's interleave
STAGE_ITERS = 16
GRAIN = NST * STAGE_ITERS  # 128 bytes a group retires per outer step
BUFB = 512                 # the cp.async ring, four 128-byte slots

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


def _cost_model_per_chunk() -> dict:
    """The per-chunk kernel's constants. See `cost_model`.

    Reported separately because the two kernels differ in the thing that
    decides throughput. This one builds a narrow table per group -- 4 KiB of
    slot->symbol plus 1 KiB of symbol->(start, freq) -- with no fixed cost, so
    a block's request scales entirely with its group count. The shared-table
    kernel pays 16 KiB once and 640 B a group.

    The consequence is occupancy, not arithmetic. At 5.6 KiB a group this
    kernel cannot exceed 128 threads inside a 99 KiB block, and on the card it
    was measured on `pick_tpb` lands on 32 threads at 4 blocks a unit for every
    request -- 10752 resident lanes against the shared kernel's 21504 to 64512.
    That ~6x is most of the gap between 110 and 418 GB/s, rather than the
    second dependent shared load the two-table design was expected to cost.

    So `blocks_per_unit_at_measurement` is a single value here and not a range:
    there was only ever one launch to measure.
    """
    return {
        "lanes": 8,
        "states": 8,
        "grain": grain(),
        "bytes_per_symbol": 1,
        "kernel": "per_chunk",
        # 257-291, and it is a SINGLE-POINT FIT, not a bracket. There is only
        # one launch to measure -- see `k_derivation` -- so this width is
        # run-to-run spread on one configuration, where the shared kernel's
        # interval brackets 3 and 4 resident blocks. Same field, weaker claim.
        "k_cycles_per_byte": (257, 291),
        "k_is_single_point_fit": True,
        "expansion": 2.79,
        "bound": {
            # There is no crossover to report: every block size the caller can
            # ask for resolves to the same launch, so nothing here is tunable.
            "compute_below_threads": None,
            "saturates_at_fraction_of_peak_dram": 0.15,
        },
        "shmem_lut_bytes": 0,
        "shmem_per_group_bytes": PROB_SCALE + 256 * 4 + GRAIN + BUFB,
        "blocks_per_unit_at_measurement": (4, 4),
        "provenance": {
            "kernel": "lmzgpu.cu k_perstream, a frequency table per chunk",
            "device": "NVIDIA GeForce RTX 5080, sm_120, 84 SMs, "
                      "256-bit GDDR7, ~960 GB/s peak, ~2.66 GHz sustained",
            "machine": "9800X3D, WSL2, CUDA 13.2, driver 610.88",
            "archive": "936.4 MB of real BF16 exponent planes from a Llama "
                       "checkpoint, 336.1 MB coded, 32768-byte streams",
            "method": "block-size sweep at fixed byte traffic, two runs "
                      "agreeing within 0.3%. Every requested size resolves to "
                      "the same launch, so the sweep has one real row rather "
                      "than seven -- a caller cannot tune this kernel by "
                      "asking for a wider block.",
            "k_derivation": "measured: resident lanes x sustained clock / "
                            "decode rate, at the single launch pick_tpb "
                            "chooses (32 threads, 4 blocks a unit). It "
                            "overlaps the shared kernel's 230-330, which is "
                            "the expected result -- same algorithm, same cost "
                            "a byte, different residency. THE WIDTH MEANS "
                            "SOMETHING DIFFERENT THOUGH: the shared kernel's "
                            "interval brackets two occupancies, this one is "
                            "run-to-run spread over 12 runs of one launch "
                            "(98.2-111.2 GB/s). A sweep that collapses to a "
                            "point cannot bracket an interval, so treat this "
                            "as a point estimate with noise, not a range a "
                            "device might land anywhere inside.",
            "status": "one device, and one launch on it. The interval is "
                      "narrower than the shared kernel's because there was no "
                      "occupancy to vary, not because it is better known.",
        },
    }


def cost_model(kernel: str = "shared") -> dict:
    """What the decoder costs, so a caller can predict it on its own machine.

    `kernel` selects which of the two lmz ships: "shared", for a batch given
    one frequency table, and "per_chunk", for streams that each carry their
    own -- which is what every archive written today holds. They are the same
    algorithm and cost about the same per byte; what differs is how many lanes
    a device can keep resident, and that is most of the throughput gap between
    them.

    A consumer deciding whether decoding beats not compressing needs the
    kernel's constants, not lmz's opinion about their machine. These are the
    constants; the curve built from them -- against that machine's bandwidth,
    clock and storage rate -- belongs to the caller.

    Every value carries its provenance, and where a value is not pinned to a
    point the interval is given rather than a midpoint, so uncertainty
    propagates instead of being inherited as fact.

    Keys:

    `lanes`, `grain`, `bytes_per_symbol`, `states` -- exact properties of the
    format and the kernel, true on every device.

    `k_cycles_per_byte` -- (low, high) lane-cycles of compute per decoded
    byte. **An interval, and latency-bound rather than throughput-bound**: the
    inner loop is a dependent shared-memory load feeding a state update
    feeding the next load, so this does not scale with a device's FP32 rate
    and must not be estimated from it. Occupancy hides part of the chain,
    which is why it is a range and not a constant.

    `expansion` -- decoded bytes out per coded byte in, on the archive named
    in provenance. A caller wanting DRAM traffic per decoded byte uses
    1 + 1/expansion: the plaintext is written and the coded bytes are read.

    `bound` -- which resource binds, and where the crossover sits. This is the
    part a consumer most needs and cannot derive: below the named occupancy
    the kernel is compute-bound and rate tracks resident lanes; above it the
    rate saturates against achieved bandwidth.

    Nothing here is measured on the calling machine -- that would be a policy
    measurement, which is the caller's. `verify()` reports what this device
    actually did.
    """
    if kernel not in ("shared", "per_chunk"):
        raise ValueError(f"kernel must be 'shared' or 'per_chunk', not {kernel!r}")
    if kernel == "per_chunk":
        return _cost_model_per_chunk()
    return {
        "lanes": 8,
        "states": 8,
        "grain": grain(),
        "bytes_per_symbol": 1,
        "kernel": "shared",
        "k_cycles_per_byte": (230, 330),
        # A real bracket: the low end came from a row holding 4 resident
        # blocks and the high end from one holding 3, so a device inside that
        # occupancy range lands inside the interval. Contrast the per-chunk
        # kernel, where the sweep collapses to one launch and the same field
        # is a point estimate with noise around it.
        "k_is_single_point_fit": False,
        "expansion": 2.88,
        "bound": {
            "compute_below_threads": 192,
            "saturates_at_fraction_of_peak_dram": 0.59,
        },
        # What a block of the shared-table kernel asks for, so a caller can
        # work out how many fit in its own device's budget. Both come from
        # the kernel's own constants rather than being restated here.
        "shmem_lut_bytes": PROB_SCALE * 4,
        "shmem_per_group_bytes": GRAIN + BUFB,
        # Resident blocks per unit across the rows k was taken from -- a
        # range, because it is one: the low end of k came from a row holding
        # 4 blocks and the high end from a row holding 3.
        #
        # This is the field that says whether k is a bracket or a floor. A
        # device that holds at least the low end hides at least as much of
        # the dependent-load chain as the measurement did, so the interval
        # brackets it. A device that holds fewer hides less, and k should be
        # read as a floor with no upper bound published -- which is the case
        # on a small integrated adapter, and exactly the case the interval
        # must not be quoted as a bracket for.
        "blocks_per_unit_at_measurement": (3, 4),
        "provenance": {
            "kernel": "lmzgpu.cu k_shared, shared frequency table",
            "device": "NVIDIA GeForce RTX 5080, sm_120, 84 SMs, "
                      "256-bit GDDR7, ~960 GB/s peak, ~2.66 GHz sustained",
            "machine": "9800X3D, WSL2, CUDA 13.2, driver 610.88",
            "archive": "936.4 MB of real BF16 exponent planes from a Llama "
                       "checkpoint, 325.2 MB coded, 32768-byte streams",
            "method": "occupancy sweep at fixed byte traffic -- threads per "
                      "block 64..384 over identical input, two runs agreeing "
                      "within 1%. Bytes moved are the same in every row, so "
                      "the 1.70x spread is compute alone.",
            "k_derivation": "measured, not derived: lanes resident x "
                            "sustained clock / decode rate, over the rows "
                            "below saturation. Resident lanes come from the "
                            "kernel's own shared-memory request (16 KiB LUT "
                            "plus 640 B a group) against the device's 99 KiB "
                            "opt-in block.",
            "status": "one device. k is a property of this kernel rather than "
                      "of the host, but the interval was taken on a single "
                      "high-bandwidth card; a device with a low FLOP-per-byte "
                      "ratio would tighten it and has not been measured.",
        },
    }


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


def decode_batch_dev(streams, offsets, nstr: int, plane: int, out,
                     header=None, stream=None, tpb: int = 0) -> int:
    """Decode a batch that is already in device memory. Returns a status code.

    Every pointer is the caller's: `streams`, `offsets`, `out` and `header` are
    device addresses as plain integers, and `stream` is a CUDA stream handle,
    or None for the default stream. This allocates nothing, copies nothing,
    creates no context and **does not synchronise** -- when it returns, the
    work has been enqueued and not necessarily done. The caller decides when
    the result is needed, because the caller owns the stream.

    That is the whole reason this exists next to `decode_batch`. A consumer
    that has already built a context, a staging ring and an event chain cannot
    share a device with a library that quietly builds its own; the host-array
    form above allocates and synchronises by construction, which makes it the
    wrong shape for anyone driving the device themselves.

    The caller also owns the padding: the kernel prefetches past its cursor, so
    `streams` must have `pad_bytes()` of readable slack after the last stream.
    `decode_batch` adds that on the device itself; here it cannot, because the
    allocation is not this function's to make.

    Returns OK, or one of ENODEV / EUNSUPPORTED / EBADSTREAM / ECUDA -- the
    same codes the C ABI uses, with `last_error()` carrying the sentence. A
    status rather than an exception because the caller is mid-pipeline and the
    decision to fall back is theirs.
    """
    lib = _load()
    if lib is None:
        return ENODEV
    if nstr == 0:
        return OK
    if nstr < 0:
        raise ValueError("nstr is negative")
    if out is None or streams is None or offsets is None:
        raise ValueError("streams, offsets and out are device pointers")
    return lib.lmz_gpu_decode_batch_dev(
        ctypes.c_void_p(header or 0), ctypes.c_void_p(streams),
        ctypes.c_void_p(offsets), nstr, plane, ctypes.c_void_p(out),
        ctypes.c_void_p(stream or 0), tpb)
