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
import struct
import subprocess
import sys
import threading

from . import vulkan

__all__ = ["available", "backend", "can_decode_chunk", "cost_model",
           "decode_batch", "decode_batch_dev", "decode_chunk", "decode_chunks",
           "dequant_q8_0_to_fp16", "grain", "header_bytes", "last_error",
           "make_mtl_buffer", "pad_bytes", "project_fp8_to_bf16",
           "release_mtl_buffer", "prepare_metal_batch", "decode_bf16_fused", "state", "verify", "vulkan"]

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
        return False, "the built GPU decoder is from another source"
    if p.returncode == 3:
        dev_tool = "Metal" if sys.platform == "darwin" else "CUDA"
        return False, f"compiler is present but no {dev_tool} device is"
    if p.returncode == 4:
        return False, (p.stderr.strip().splitlines() or ["the library did not load"])[-1]
    # Anything else is a death rather than a decision: a negative code is the
    # POSIX signal that killed it, and Windows reports its own status codes.
    sig = f"signal {-p.returncode}" if p.returncode < 0 else f"status {p.returncode:#x}"
    dev_name = "Metal" if sys.platform == "darwin" else "CUDA"
    return False, (f"the {dev_name} driver crashed while loading ({sig}); the driver "
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
                target_name = "Metal" if sys.platform == "darwin" else "CUDA"
                _why = _build.last_error or f"the {target_name} decoder did not build"
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
                _state, _why = "abi-mismatch", "the built GPU decoder is from another source"
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
            if hasattr(lib, "lmz_gpu_dequant_q8_0_to_fp16"):
                lib.lmz_gpu_dequant_q8_0_to_fp16.restype = c_int
                lib.lmz_gpu_dequant_q8_0_to_fp16.argtypes = [v, v, c_uint]
            if hasattr(lib, "lmz_gpu_project_fp8_to_bf16"):
                lib.lmz_gpu_project_fp8_to_bf16.restype = c_int
                lib.lmz_gpu_project_fp8_to_bf16.argtypes = [v, v, c_uint]
            if hasattr(lib, "lmz_gpu_make_mtl_buffer"):
                lib.lmz_gpu_make_mtl_buffer.restype = v
                lib.lmz_gpu_make_mtl_buffer.argtypes = [v, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
            if hasattr(lib, "lmz_gpu_release_mtl_buffer"):
                lib.lmz_gpu_release_mtl_buffer.restype = None
                lib.lmz_gpu_release_mtl_buffer.argtypes = [v]
            if hasattr(lib, "lmz_gpu_metal_decode"):
                lib.lmz_gpu_metal_decode.restype = c_int
                lib.lmz_gpu_metal_decode.argtypes = [v, v, ctypes.c_size_t, v, c_uint, c_uint,
                                                     v, ctypes.c_size_t, ctypes.POINTER(v), ctypes.POINTER(v)]
                lib.lmz_gpu_metal_decode_bf16.restype = c_int
                lib.lmz_gpu_metal_decode_bf16.argtypes = [v,v,ctypes.c_size_t,v,c_uint,c_uint,v,ctypes.c_size_t,v,ctypes.c_size_t,ctypes.POINTER(v),ctypes.POINTER(v)]
                lib.lmz_gpu_metal_prepare.restype = v
                lib.lmz_gpu_metal_prepare.argtypes = [v,v,ctypes.c_size_t,v,c_uint,c_uint,v,ctypes.c_size_t,v,ctypes.c_size_t]
                lib.lmz_gpu_metal_run_prepared.restype = c_int
                lib.lmz_gpu_metal_run_prepared.argtypes = [v]
                lib.lmz_gpu_metal_prepared_contents.restype = v
                lib.lmz_gpu_metal_prepared_contents.argtypes = [v]
                lib.lmz_gpu_metal_release_prepared.argtypes = [v]
                lib.lmz_gpu_metal_release_prepared.restype = None
                lib.lmz_gpu_metal_decode_batch_dev_sized.restype = c_int
                lib.lmz_gpu_metal_decode_batch_dev_sized.argtypes = [v,v,v,c_uint,c_uint,v,v,c_uint,ctypes.c_size_t,ctypes.c_size_t]
                lib.lmz_gpu_metal_last_timing.restype = ctypes.POINTER(ctypes.c_double)
                lib.lmz_gpu_mtl_contents.restype = v
                lib.lmz_gpu_mtl_contents.argtypes = [v]
            device = _probed
            prefix = "metal:" if sys.platform == "darwin" or "Metal" in device else "cuda:"
            if not _selftest(lib):
                _state = "unavailable"
                _why = (f"the {prefix[:-1].upper()} decoder did not reproduce the CPU decoder's "
                        f"bytes on {device}; staying on the CPU")
                return None
            _lib = lib
            _state = prefix + device
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
        "backend": "cuda",
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
        # Same formula as the shared kernel; see cost_model() for why both the
        # per-block cap and the per-unit pool are needed rather than either
        # alone. Here the fixed term is zero, so shm is entirely per-group --
        # which is why this kernel cannot exceed 128 threads a block inside a
        # 99 KiB cap and lands on 32 whatever the caller asks for.
        "residency_formula": (
            "blocks = floor(pool / shm) if shm <= cap else 0, where "
            "shm = shmem_lut_bytes + (threads / lanes) * shmem_per_group_bytes, "
            "cap = max shared memory for one block (CUDA "
            "sharedMemPerBlockOptin), pool = shared memory per compute unit "
            "(CUDA sharedMemPerMultiprocessor). They agree on discrete NVIDIA "
            "parts and diverge on integrated ones, which is where it matters. "
            "A third limit exists and is not in the formula because it does "
            "not bind on the device measured: a compute unit also caps "
            "resident THREADS (CUDA maxThreadsPerMultiProcessor, 1536 here). "
            "Shared memory binds first at every block size on this card "
            "because the fixed LUT is large, so blocks = min(pool/shm, "
            "threads_per_unit/threads) reduces to the shm term -- but that is "
            "a property of this kernel's footprint, not a guarantee, and a "
            "smaller table would change it. Take the min if you have both. "
            "NOT EVERY API REPORTS THEM: Vulkan exposes no per-unit pool at "
            "all, and its maxComputeSharedMemorySize is not the opt-in cap "
            "either -- on this card Vulkan reports 49152 where CUDA reports "
            "an opt-in cap of 101376 and a pool of 102400, i.e. it "
            "corresponds to CUDA's non-opt-in sharedMemPerBlock. So a "
            "Vulkan-only consumer has a figure that understates the cap, and "
            "no pool at all. Substituting it for either understates "
            "residency; substituting it for both understates it twice. That "
            "yields a floor, never an estimate -- and pool >= cap always "
            "holds, since a unit that could not hold one legally-sized "
            "workgroup could not schedule one. Say which you used."
        ),
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


def _cost_model_metal_shared() -> dict:
    """The Metal shared-table kernel's own cost, measured on an Apple M4.

    A different kernel from CUDA k_shared, not a port of its constants: 128
    threads a threadgroup, direct device reads in the refill (no cp.async
    ring), and a compiled footprint of 19,456 B. Its k is therefore its own
    number. Borrowing CUDA's would predict the M4 26-44% too fast.
    """
    return {
        "lanes": 8,
        "states": 8,
        "grain": grain(),
        "bytes_per_symbol": 1,
        "kernel": "shared",
        "backend": "metal",
        # Lane-cycles per decoded byte across the measured occupancies, 1 to 4
        # threadgroups per core, at the KERNEL's clock: IOReport state P15
        # (1578 MHz), whose residency matched the summed kernel time in every
        # row (0.98-1.06x). The residency-weighted average clock, diluted by
        # background low-state time between runs, gave 268.9-300.4 and
        # under-reads most where kernel windows are shortest. Exact bracket of
        # the row medians below, unrounded.
        "k_cycles_per_byte": (269.4, 313.1),
        "k_by_threadgroups_per_core": {
            1: (269.4, 276.5), 2: (287.8, 288.1),
            3: (297.2, 298.3), 4: (311.6, 313.1),
        },
        "k_is_single_point_fit": False,
        "expansion": 2.975,
        "bound": {
            # Compute-bound at the shipped 128 threads at every occupancy the
            # M4 can hold. The key doubles as the block size for consumers that
            # read it that way, and 128 is correct for both readings.
            "compute_below_threads": 128,
            # NOT REACHED: the kernel tops out at its residency limit with
            # DRAM at ~28% of spec, so no saturation fraction exists to publish.
            # Writing the 0.28 here would tell a consumer that bandwidth binds
            # at the compute ceiling, which is the reverse of what was measured.
            "saturates_at_fraction_of_peak_dram": None,
            "observed_fraction_of_peak_dram_at_ceiling": 0.28,
        },
        # The fixed per-threadgroup term is the LUT (16,384 B) plus the 1 KiB
        # `starts` array; the per-group term is the 128 B output stage. The
        # declared 256 B input ring is removed by the compiler in the direct
        # specialization -- the pipeline reports 19,456 B static threadgroup
        # memory, which these two fields reproduce exactly at 128 threads:
        # 17,408 + (128 / 8) * 128 = 19,456. (CUDA's entry counts only the
        # LUT in its fixed term and omits its own 1 KiB static `starts`.)
        "shmem_lut_bytes": PROB_SCALE * 4 + 256 * 4,
        "shmem_per_group_bytes": GRAIN,
        "compiled_threadgroup_bytes_at_128_threads": 19_456,
        "blocks_per_unit_at_measurement": (1, 4),
        "residency_formula": (
            "blocks = floor(pool / shm) if shm <= cap else 0, where "
            "shm = shmem_lut_bytes + (threads / lanes) * shmem_per_group_bytes. "
            "Metal reports the cap (maxThreadgroupMemoryLength, 32768 on the "
            "M4) and NO per-core pool; the M4's pool was measured by occupancy "
            "staircase as 77,824 <= pool < 83,968 B (lmsluice "
            "probe/metal-bench.js), which gives 4 threadgroups per core at "
            "19,456 B -- the same 4 the decoder grid sweep's knee implies. Say "
            "which pool you used; a cap substituted for the pool gives 1."
        ),
        "provenance": {
            "kernel": "lmz_rans.metal lmz_decode_plane_direct "
                      "(decode_plane_shared<false>), shared frequency table",
            "device": "Apple M4 (Mac16,10), 10 GPU cores, 1.51-1.58 GHz "
                      "measured per row (IOReport P15 = 1578 MHz), LPDDR5X "
                      "120 GB/s spec, ~102 GB/s measured streaming read",
            "machine": "Mac mini, 24 GB unified memory, macOS 15.4.1",
            "archive": "936.6 MB of real BF16 exponent planes from "
                       "Qwen3-4B-Instruct-2507 (first 936.6 MB in file order), "
                       "314.8 MB coded, 28,584 streams of 32,768 bytes, one "
                       "shared table",
            "method": "grid sweep at fixed byte traffic -- the shipped "
                      "decode_plane_shared<false> wrapped in a grid-stride "
                      "loop and nothing else (drift +0.96% against the shipped "
                      "kernel at full grid), 1 to 1787 threadgroups, "
                      "byte-identical at every size, GPU timestamps; row "
                      "frequency from IOReport GPU P-state residency",
            "k_derivation": "measured: resident lanes (128 x resident "
                            "threadgroups) x the kernel's clock (P15, 1578 MHz, "
                            "attributed by P15 residency >= 0.95 x summed "
                            "kernel time) / decode rate, rows at 1-4 "
                            "threadgroups per core",
            "scaling": "scale by resident lanes x clock, never by FP32 rate. "
                       "On this device k RISES with co-resident threadgroups "
                       "(269.4 at <=1 per core to ~313 at 4): co-resident "
                       "groups contend for the core rather than only hiding "
                       "each other's latency -- the opposite direction to the "
                       "CUDA reading, so a device holding fewer than 4 per core "
                       "sits toward the LOW end, not beyond the high end",
            "supersedes": "nothing: the first measured Metal cost. The earlier "
                          "24.1/36.3 GB/s Metal figures were synthetic-input "
                          "rates, not a cost model",
            "status": "one device. Desktop GPU clients (WindowServer, Terminal, "
                      "a browser) were present in most rows and only the 1-2 "
                      "threadgroup rows met a <=2% foreign-time criterion, so "
                      "the interval is conditional on that background. The "
                      "4-per-core row is residency-limited at ~28% of DRAM, "
                      "i.e. still a compute measurement; the re-measurement "
                      "report excludes it from its primary interval. A CPU "
                      "streaming 71-81 GB/s (measured read+write) on the same "
                      "bus slowed the decoder 17-29% although DRAM did not "
                      "bind it. The kernel is byte-identical to the one "
                      "measured: the 2026-11-09 host-path rewrite did not "
                      "change it. Record: "
                      "docs/evaluations/m4-metal-decoder-remeasure.md "
                      "(Phases 3b-5)",
        },
    }


def _cost_model_metal_per_chunk() -> dict:
    """The Metal per-chunk-table kernel, measured on an Apple M4.

    What every archive written today holds: each stream carries its own table,
    so the kernel builds one per stream in threadgroup memory -- 32 threads, 4
    streams a threadgroup, 20,992 B compiled -- and pays for it per byte.
    """
    return {
        "lanes": 8,
        "states": 8,
        "grain": grain(),
        "bytes_per_symbol": 1,
        "kernel": "per_chunk",
        "backend": "metal",
        # At the kernel's clock (P15, attributed as for the shared record),
        # over the rows where the threadgroup count is the resident count:
        # 1-30 threadgroups, at most 3 per core, linear to within 1%.
        "k_cycles_per_byte": (349.3, 355.4),
        "k_is_single_point_fit": False,
        "expansion": 2.934,
        "bound": {
            "compute_below_threads": 32,
            "saturates_at_fraction_of_peak_dram": None,
        },
        # No fixed term: every table is per stream, so per group. 4,096 B of
        # symbols + 1,024 B of starts/frequencies + the 128 B output stage =
        # 5,248 B a group, 20,992 B at 32 threads -- the compiled figure.
        "shmem_lut_bytes": 0,
        "shmem_per_group_bytes": 5_248,
        "compiled_threadgroup_bytes_at_32_threads": 20_992,
        "blocks_per_unit_at_measurement": (1, 3),
        # The launch geometry matters more here than anywhere else measured:
        # one threadgroup per four streams is far from the best grid. These two
        # are KERNEL-ONLY rates of the unchanged kernel. Since 2026-11-09 the host
        # path defaults to that shipped geometry and tuning is opt-in
        # (prepare_metal_batch, or LMZ_METAL_PERSTREAM_TGS=auto): through the API
        # the default decodes 4.0 GB/s and a tuned grid (80-160 TGs here, varying
        # with load) 10-11 GB/s, at a one-time cost of ~20 batch decodes.
        "shipped_geometry_gbps": 4.577,
        "best_measured_grid": {"threadgroups": 100, "gbps": 10.557},
        "residency_formula": (
            "blocks = floor(pool / shm) if shm <= cap else 0, where "
            "shm = shmem_lut_bytes + (threads / lanes) * shmem_per_group_bytes. "
            "On the M4's staircase-measured pool (77,824 <= pool < 83,968 B) "
            "this gives 3 per core at 20,992 B, which the probe's staircase "
            "also measured. UNRESOLVED ABOVE THAT: the decoder's rate keeps "
            "rising past 30 threadgroups, to ~10.6 GB/s at 100 -- about 73 "
            "threadgroups' worth at the low-grid cost -- so the measured curve "
            "does not support a hard 3-per-core ceiling for this kernel, and "
            "none is published. k is valid only at the occupancies listed."
        ),
        "provenance": {
            "kernel": "lmz_rans.metal lmz_decode_perstream, a table per stream",
            "device": "Apple M4 (Mac16,10), 10 GPU cores, kernel clock P15 = "
                      "1578 MHz (IOReport), LPDDR5X 120 GB/s spec",
            "machine": "Mac mini, 24 GB unified memory, macOS 15.4.1",
            "archive": "the same 936.6 MB of real Qwen3-4B-Instruct-2507 BF16 "
                       "exponent planes, 28,584 ordinary lmz streams of "
                       "32,768 bytes each with its own 516-byte table, "
                       "319.2 MB coded including the tables",
            "method": "grid sweep at fixed byte traffic -- the shipped "
                      "per-stream body verbatim in a grid-stride loop (drift "
                      "-1.06% against the shipped kernel at full grid), 1 to "
                      "7146 threadgroups, byte-identical at every size, GPU "
                      "timestamps, IOReport residency per row",
            "k_derivation": "measured: 32 x threadgroups x 1578 MHz / decode "
                            "rate over 1-30 threadgroups",
            "scaling": "by resident lanes x clock, at 1-3 threadgroups per "
                       "core only; above that the occupancy is unresolved",
            "supersedes": "nothing: the first measured Metal per-chunk cost",
            "status": "one device; desktop GPU clients present in most rows. "
                      "The kernels are byte-identical to those measured: the "
                      "2026-11-09 host-path rewrite (zero-copy buffers, ARC) did "
                      "not change them, and a bounds-guarded variant that did "
                      "(4-10% slower) was not kept. "
                      "Record: docs/evaluations/m4-metal-decoder-remeasure.md "
                      "(Phases 4 and 5)",
        },
    }


def cost_model(kernel: str = "shared", backend: str = "cuda") -> dict:
    """What the decoder costs, so a caller can predict it on its own machine.

    `backend` names whose kernel the constants describe: "cuda" (the default,
    and the record every existing caller gets) or "metal". The two kernels
    share the algorithm and format but not their shape, so their k differ.
    Both Metal kernels have their own measured records; an unknown backend
    raises rather than returning CUDA's numbers under another name.

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
    if backend not in ("cuda", "metal"):
        raise ValueError(f"backend must be 'cuda' or 'metal', not {backend!r}")
    if backend == "metal":
        if kernel == "per_chunk":
            return _cost_model_metal_per_chunk()
        return _cost_model_metal_shared()
    if kernel == "per_chunk":
        return _cost_model_per_chunk()
    return {
        "lanes": 8,
        "states": 8,
        "grain": grain(),
        "bytes_per_symbol": 1,
        "kernel": "shared",
        "backend": "cuda",
        # Validated across 1191x of compute by shrinking the grid at fixed
        # byte traffic, which is the only way to reach the compute-bound
        # regime on a card that saturates on bandwidth at 192 threads. The
        # first published interval, (230, 330), was derived from full-size
        # runs where bandwidth binds -- dividing a bandwidth-limited rate by
        # resident lanes does not yield a compute constant. It was right in
        # direction and 20% wide in the wrong place; this is the same
        # quantity measured where it is actually visible.
        "k_cycles_per_byte": (217, 248),
        # A real bracket: 14 grid sizes from 1 block to 1191, 217-248 across
        # the ten that are genuinely compute-bound. Contrast the per-chunk
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
        # HOW TO USE THE THREE FIELDS ABOVE, because the divisor is not
        # inferable from them and a consumer that guesses it wrong finds out
        # only on a device where it matters.
        #
        #     shm    = shmem_lut_bytes + (threads / lanes) * shmem_per_group_bytes
        #     blocks = shm <= cap ? floor(pool / shm) : 0
        #
        # where `cap` is the most shared memory ONE BLOCK may have -- opting in
        # if the API requires it, CUDA's sharedMemPerBlockOptin -- and `pool`
        # is what a compute unit hands out in total, CUDA's
        # sharedMemPerMultiprocessor. Both are needed: the cap decides whether
        # a block size is legal at all, the pool decides how many fit.
        #
        # On every discrete NVIDIA part they are within a kilobyte of each
        # other (this card: 101376 and 102400) and either answers. They
        # diverge exactly where the answer matters -- an integrated adapter
        # capping a block at 32 KiB out of a 64 KiB pool holds 1 block by the
        # cap and 3 by the pool -- so the formula uses both rather than
        # whichever is to hand.
        #
        # The figures published here were taken that way. A caller that
        # reproduces them and does not get 4 blocks at 64 threads and 2 at 384
        # on an RTX 5080 is using a different divisor, and the projection it
        # builds will be wrong in proportion.
        "residency_formula": (
            "blocks = floor(pool / shm) if shm <= cap else 0, where "
            "shm = shmem_lut_bytes + (threads / lanes) * shmem_per_group_bytes, "
            "cap = max shared memory for one block (CUDA "
            "sharedMemPerBlockOptin), pool = shared memory per compute unit "
            "(CUDA sharedMemPerMultiprocessor). They agree on discrete NVIDIA "
            "parts and diverge on integrated ones, which is where it matters. "
            "A third limit exists and is not in the formula because it does "
            "not bind on the device measured: a compute unit also caps "
            "resident THREADS (CUDA maxThreadsPerMultiProcessor, 1536 here). "
            "Shared memory binds first at every block size on this card "
            "because the fixed LUT is large, so blocks = min(pool/shm, "
            "threads_per_unit/threads) reduces to the shm term -- but that is "
            "a property of this kernel's footprint, not a guarantee, and a "
            "smaller table would change it. Take the min if you have both. "
            "NOT EVERY API REPORTS THEM: Vulkan exposes no per-unit pool at "
            "all, and its maxComputeSharedMemorySize is not the opt-in cap "
            "either -- on this card Vulkan reports 49152 where CUDA reports "
            "an opt-in cap of 101376 and a pool of 102400, i.e. it "
            "corresponds to CUDA's non-opt-in sharedMemPerBlock. So a "
            "Vulkan-only consumer has a figure that understates the cap, and "
            "no pool at all. Substituting it for either understates "
            "residency; substituting it for both understates it twice. That "
            "yields a floor, never an estimate -- and pool >= cap always "
            "holds, since a unit that could not hold one legally-sized "
            "workgroup could not schedule one. Say which you used."
        ),
        "provenance": {
            "kernel": "lmzgpu.cu k_shared, shared frequency table",
            "device": "NVIDIA GeForce RTX 5080, sm_120, 84 SMs, "
                      "256-bit GDDR7, ~960 GB/s peak, ~2.66 GHz sustained",
            "machine": "9800X3D, WSL2, CUDA 13.2, driver 610.88",
            "archive": "936.4 MB of real BF16 exponent planes from a Llama "
                       "checkpoint, 325.2 MB coded, 32768-byte streams",
            "method": "grid sweep at fixed byte traffic -- 1 to 1191 blocks "
                      "over identical input, byte-identical at every size, "
                      "linear in resident blocks to within 3% below 84 "
                      "blocks and bending as bandwidth takes over. Also an "
                      "occupancy sweep -- threads per "
                      "block 64..384 over identical input, two runs agreeing "
                      "within 1%. Bytes moved are the same in every row, so "
                      "the 1.70x spread is compute alone.",
            "scaling": "k is dependent-load latency partly hidden by "
                       "occupancy, not an issue rate: the decode rate is "
                       "linear in resident lanes across an 84x range, where "
                       "an issue-limited kernel would flatten once the issue "
                       "slots filled. So scale it by resident lanes x clock. "
                       "Never by a device's FP32 rate -- the inner loop is a "
                       "dependent shared-memory load feeding a state update, "
                       "and FMA throughput says nothing about it.",
            "supersedes": "an earlier build of this file published k as "
                          "(230, 330), fitted across a full-size sweep of "
                          "64-384 threads a block. The defect was that those "
                          "rows are not governed by the same resource: at 64 "
                          "threads the kernel is compute-bound (245.7 GB/s "
                          "against a 264 ceiling), and from 96 upward it is "
                          "bandwidth-bound (417.3 against 791 at 384). "
                          "Dividing every row's rate by its resident lanes "
                          "and reading one constant out of the result mixes a "
                          "compute measurement with six memory measurements, "
                          "and a rate limited by the memory system yields "
                          "whatever k makes the arithmetic close rather than "
                          "the kernel's actual cost. That is why the old "
                          "interval was plausible, why its low end was nearly "
                          "right -- the one compute-bound row carried it -- "
                          "and why its high end was not. (217, 248) is the "
                          "same quantity taken only where compute binds. A "
                          "refinement of a badly-conditioned fit, not a "
                          "contradiction: nothing about the kernel changed "
                          "between the two.",
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
    report["device"] = backend().removeprefix("cuda:").removeprefix("metal:")
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


class _MetalBuffer(ctypes.Structure):
    """A retained contiguous Python buffer export, including read-only mmaps."""
    _fields_ = [("buf", ctypes.c_void_p), ("obj", ctypes.c_void_p),
                ("size", ctypes.c_ssize_t), ("itemsize", ctypes.c_ssize_t),
                ("readonly", ctypes.c_int), ("ndim", ctypes.c_int),
                ("format", ctypes.c_void_p), ("shape", ctypes.c_void_p),
                ("strides", ctypes.c_void_p), ("suboffsets", ctypes.c_void_p),
                ("internal", ctypes.c_void_p)]

    def __init__(self, value, writable=False):
        super().__init__()
        get = ctypes.pythonapi.PyObject_GetBuffer
        get.argtypes = [ctypes.py_object, ctypes.c_void_p, ctypes.c_int]
        get.restype = ctypes.c_int
        get(value, ctypes.byref(self), 1 if writable else 0)

    def __del__(self):
        if self.obj:
            release = ctypes.pythonapi.PyBuffer_Release
            release.argtypes = [ctypes.c_void_p]
            release.restype = None
            release(ctypes.byref(self))


class _MetalOwned:
    def __init__(self, lib, handle):
        self.lib, self.handle = lib, handle

    def __del__(self):
        self.lib.lmz_gpu_release_mtl_buffer(self.handle)


def _metal_decode(lib, streams, offsets, nstr, plane, out, header, smplane=None):
    src, off = _MetalBuffer(streams), _MetalBuffer(offsets)
    hdr = _MetalBuffer(header) if header is not None else None
    dst = _MetalBuffer(out, writable=True) if out is not None else None
    need = nstr * plane * (2 if smplane is not None else 1)
    sm = _MetalBuffer(smplane) if smplane is not None else None
    if dst is not None and dst.size < need:
        raise ValueError("output is too small")
    owned, address = ctypes.c_void_p(), ctypes.c_void_p()
    args = (hdr.buf if hdr else None, src.buf, src.size, off.buf, nstr, plane)
    output = (dst.buf if dst else None, dst.size if dst else 0, ctypes.byref(owned), ctypes.byref(address))
    rc = (lib.lmz_gpu_metal_decode_bf16(*args, sm.buf, sm.size, *output) if sm is not None
          else lib.lmz_gpu_metal_decode(*args, *output))
    if rc == EBADSTREAM:
        raise ValueError("malformed rANS stream")
    if rc != OK:
        return None
    if out is not None:
        return memoryview(out).cast("B")[:need]
    array = (ctypes.c_ubyte * need).from_address(address.value)
    array._metal_owner = _MetalOwned(lib, owned.value)
    return memoryview(array).cast("B")


class _MetalBatch:
    def __init__(self, lib, handle, refs, size):
        self.lib, self.handle, self.refs, self.size = lib, handle, refs, size

    def decode(self):
        """Synchronously decode into the retained output; prior views are overwritten."""
        if self.lib.lmz_gpu_metal_run_prepared(self.handle) != OK:
            return None
        address = self.lib.lmz_gpu_metal_prepared_contents(self.handle)
        array = (ctypes.c_ubyte * self.size).from_address(address)
        array._metal_owner = self
        return memoryview(array).cast("B")

    def __del__(self):
        self.lib.lmz_gpu_metal_release_prepared(self.handle)


def prepare_metal_batch(streams, offsets, nstr, plane, out=None, header=None, smplane=None):
    """Prepare reusable synchronous Metal bindings, or None without Metal.

    Preparation validates with one decode and may autotune per-stream geometry.
    Inputs must remain unchanged for the lifetime of the object. Views returned
    by .decode() share its output and keep all bindings/exported memory alive;
    the next decode overwrites prior views. Aligned coded inputs share their
    mapped final page only when it contains pad_bytes() of readable tail space;
    otherwise they are copied once with zero padding. This assumes valid rANS
    streams, like CUDA; corrupted payloads may fail the command buffer. Other
    inputs share complete aligned pages or are copied once. An explicit output must span whole
    aligned writable pages. smplane selects fused BF16 and requires header.
    """
    lib = _load()
    if lib is None or not _state.startswith("metal:") or not hasattr(lib, "lmz_gpu_metal_prepare"):
        return None
    _check_metal_shape(offsets, nstr, plane, header)
    if not nstr: return None
    src, off = _MetalBuffer(streams), _MetalBuffer(offsets)
    hdr = _MetalBuffer(header) if header is not None else None
    sm = _MetalBuffer(smplane) if smplane is not None else None
    dst = _MetalBuffer(out, writable=True) if out is not None else None
    if sm is not None and hdr is None: raise ValueError("fused decode requires header")
    handle = lib.lmz_gpu_metal_prepare(hdr.buf if hdr else None, src.buf, src.size, off.buf,
              nstr, plane, sm.buf if sm else None, sm.size if sm else 0,
              dst.buf if dst else None, dst.size if dst else 0)
    if not handle: return None
    return _MetalBatch(lib, handle, (src, off, hdr, sm, dst), nstr * plane * (2 if sm else 1))


def _check_metal_shape(offsets, nstr, plane, header):
    if nstr < 0 or nstr > 0xFFFFFFFF or plane <= 0 or plane > 0xFFFFFFFF:
        raise ValueError("invalid batch dimensions")
    if memoryview(offsets).nbytes < nstr * 16: raise ValueError("offsets too short")
    if header is not None and memoryview(header).nbytes != 516: raise ValueError("shared header must be 516 bytes")


def decode_bf16_fused(streams, offsets, smplane, nstr, plane, header, out=None):
    """Metal rANS exponent + raw sign/mantissa merge to original BF16 bytes.

    Offsets describe headerless exponent streams against the required shared
    header. smplane holds nstr*plane raw sign/mantissa bytes. Returns an owned
    Metal-backed view or a view of writable out; None means CPU fallback.
    CUDA is unaffected and declines this Metal-only entrypoint.
    """
    lib = _load()
    if lib is None or not _state.startswith("metal:") or not hasattr(lib, "lmz_gpu_metal_decode_bf16"):
        return None
    _check_metal_shape(offsets, nstr, plane, header)
    if header is None: raise ValueError("fused decode requires header")
    if not nstr: return memoryview(b"")
    return _metal_decode(lib, streams, offsets, nstr, plane, out, header, smplane)


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

    if _state.startswith("metal:") and hasattr(lib, "lmz_gpu_metal_decode"):
        if out is not None and memoryview(out).nbytes < nstr * plane:
            out = None  # preserve the legacy undersized-scratch fallback
        return _metal_decode(lib, streams, offsets, nstr, plane, out, header)

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
                     header=None, stream=None, tpb: int = 0,
                     streams_nbytes: int = 0, out_nbytes: int = 0) -> int:
    """Decode a batch that is already in device memory. Returns a status code.

    On Metal these are CPU-accessible unified-memory pointers. The default
    queue is synchronous; an explicit Metal queue enqueues asynchronously and
    the caller must keep all pointed-to memory alive through completion.
    streams_nbytes/out_nbytes optionally declare the exported capacities; the
    coded extent is checked against streams_nbytes when supplied. Metal wraps
    safe aligned ranges and otherwise copies, including output copy-back.

    On CUDA every pointer is the caller's: `streams`, `offsets`, `out` and `header` are
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
    if _state.startswith("metal:") and hasattr(lib, "lmz_gpu_metal_decode_batch_dev_sized"):
        if streams_nbytes < 0 or out_nbytes < 0:
            raise ValueError("negative exported capacity")
        return lib.lmz_gpu_metal_decode_batch_dev_sized(header or 0, streams, offsets,
                 nstr, plane, out, stream or 0, tpb, streams_nbytes, out_nbytes)
    return lib.lmz_gpu_decode_batch_dev(
        ctypes.c_void_p(header or 0), ctypes.c_void_p(streams),
        ctypes.c_void_p(offsets), nstr, plane, ctypes.c_void_p(out),
        ctypes.c_void_p(stream or 0), tpb)


def dequant_q8_0_to_fp16(blocks, nblocks: int, out=None):
    """GPU in-kernel dequantization of Q8_0 blocks to FP16 half-precision words."""
    lib = _load()
    if lib is None or not hasattr(lib, "lmz_gpu_dequant_q8_0_to_fp16"):
        return None
    need = nblocks * 32 * 2
    dst = out if (out is not None and len(out) >= need) else bytearray(need)
    b_addr, _a = _ptr(blocks)
    d_addr, _b = _ptr(dst)
    rc = lib.lmz_gpu_dequant_q8_0_to_fp16(b_addr, d_addr, nblocks)
    if rc == OK:
        return memoryview(dst)[:need]
    return None


def project_fp8_to_bf16(fp8_data, nelem: int, out=None):
    """GPU in-kernel projection of FP8 (E4M3) elements to BF16 words."""
    lib = _load()
    if lib is None or not hasattr(lib, "lmz_gpu_project_fp8_to_bf16"):
        return None
    need = nelem * 2
    dst = out if (out is not None and len(out) >= need) else bytearray(need)
    s_addr, _a = _ptr(fp8_data)
    d_addr, _b = _ptr(dst)
    rc = lib.lmz_gpu_project_fp8_to_bf16(s_addr, d_addr, nelem)
    if rc == OK:
        return memoryview(dst)[:need]
    return None


_metal_exports = {}


def make_mtl_buffer(data):
    """Wrap or share memoryview/bytes into an Apple Silicon MTLBuffer.
    Returns (raw_ptr, offset, length) or None if Metal is unavailable.
    """
    lib = _load()
    if lib is None or not hasattr(lib, "lmz_gpu_make_mtl_buffer"):
        return None
    ref = _MetalBuffer(data)
    ptr = ref.buf
    if ptr is None:
        return None
    length = ref.size
    out_off = ctypes.c_size_t(0)
    buf_ptr = lib.lmz_gpu_make_mtl_buffer(ptr, length, ctypes.byref(out_off))
    if not buf_ptr:
        return None
    _metal_exports[buf_ptr] = ref
    return buf_ptr, out_off.value, length


def release_mtl_buffer(buf_ptr):
    """Release an MTLBuffer handle returned by make_mtl_buffer."""
    lib = _load()
    if lib is not None and hasattr(lib, "lmz_gpu_release_mtl_buffer") and buf_ptr:
        lib.lmz_gpu_release_mtl_buffer(buf_ptr)
        _metal_exports.pop(buf_ptr, None)


def can_decode_chunk(codec: int, esize: int, flags: int, rlen: int) -> bool:
    """Whether the GPU decoder can take this chunk."""
    lib = _load()
    if lib is None:
        return False
    from ..format import CODEC_SPLIT, CODEC_BF16, CODEC_SPLIT_ST, CODEC_BF16C
    if codec in (CODEC_SPLIT, CODEC_BF16, CODEC_SPLIT_ST):
        if esize not in (2, 4, 8) or (rlen % esize) != 0:
            return False
        nplanes = 2 if codec == CODEC_BF16 else esize
        nelem = rlen // nplanes
        g = grain()
        if nelem == 0 or (nelem % g) != 0:
            return False
        for k in range(nplanes):
            m = (flags >> (2 * k)) & 0x3
            if m not in (0, 3):  # 0 is stored, 3 is rANS
                return False
        return True
    elif codec == CODEC_BF16C:
        if esize != 2 or (rlen % 2) != 0:
            return False
        nelem = rlen // 2
        g = grain()
        if nelem == 0 or (nelem % g) != 0:
            return False
        return True
    return False


def decode_chunk(payload, codec: int, esize: int, flags: int, rlen: int,
                 tables=None, out=None):
    """Decode one split, shared-table, or bf16-cond chunk using the GPU decoder.

    Supports CODEC_SPLIT, CODEC_BF16, CODEC_SPLIT_ST (v7 shared tables), and CODEC_BF16C.
    Returns a memoryview of the merged plaintext bytes, or None if the GPU
    decoder declines.
    """
    if not can_decode_chunk(codec, esize, flags, rlen):
        return None
    from ..format import CODEC_SPLIT, CODEC_BF16, CODEC_SPLIT_ST, CODEC_BF16C
    from .. import kernels

    if codec == CODEC_BF16C:
        from ..codec import _bf16c_hdr
        if len(payload) < _bf16c_hdr.size:
            return None
        nelem = rlen // 2
        k, exp_method, *rest = _bf16c_hdr.unpack_from(payload, 0)
        sm_methods = rest[:k]
        exp_len = rest[k]
        seg_lens = rest[k + 1:]
        pos = _bf16c_hdr.size
        if pos + exp_len > len(payload):
            return None
        exp_payload = payload[pos:pos + exp_len]
        pos += exp_len

        if exp_method == 0:
            if exp_len != nelem:
                return None
            exp = exp_payload
        elif exp_method == 3:
            off_bytes = struct.pack("<QQ", 0, exp_len)
            exp = decode_batch(exp_payload, off_bytes, 1, nelem)
            if exp is None:
                exp = kernels.rans_decode(exp_payload, nelem)
            if exp is None or len(exp) != nelem:
                return None
        else:
            return None

        ehist = kernels.histogram(exp)
        lut = kernels.bucket_lut(ehist, k)
        counts = kernels.bucket_counts(ehist, lut, k)

        streams = []
        mv = memoryview(payload)
        for b in range(k):
            ln = seg_lens[b]
            want = counts[b]
            if pos + ln > len(payload):
                return None
            m = sm_methods[b]
            if want == 0:
                if ln != 0:
                    return None
                streams.append((b"", 0))
            elif m == 0:
                if ln != want:
                    return None
                streams.append((payload, pos))
            elif m == 3:
                block = kernels.rans_decode(mv[pos:pos + ln], want)
                if block is None or len(block) != want:
                    return None
                streams.append((block, 0))
            else:
                return None
            pos += ln

        need = nelem * 2
        dst = out if (out is not None and len(out) >= need) else bytearray(need)
        res = kernels.unpartition_merge_bf16(exp, streams, lut, k, dst)
        return memoryview(res)[:need]

    nplanes = 2 if codec == CODEC_BF16 else esize
    nelem = rlen // nplanes
    hdr_size = nplanes * 4
    if len(payload) < hdr_size:
        return None
    lengths = struct.unpack_from(f"<{nplanes}I", payload, 0)

    pos = hdr_size
    streams = []
    plane_hdrs = []
    plane_methods = []
    for k in range(nplanes):
        ln = lengths[k]
        if pos + ln > len(payload):
            return None
        m = (flags >> (2 * k)) & 0x3
        plane_methods.append(m)
        streams.append(payload[pos:pos + ln])
        pos += ln

        hdr = None
        if codec == CODEC_SPLIT_ST and tables is not None:
            hdr = tables.get(CODEC_SPLIT, esize, k)
        plane_hdrs.append(hdr)

    if (os.environ.get("LMZ_METAL_FUSED_CHUNK") == "1" and
            codec == CODEC_BF16 and plane_methods == [3, 0] and
            nelem % grain() == 0 and len(streams[1]) == nelem and
            len(streams[0]) >= 516 + 32 and bytes(streams[0][:2]) == b"R1"):
        exponent = memoryview(streams[0])
        fused = decode_bf16_fused(exponent[516:], struct.pack("<QQ", 0, len(exponent)-516),
                                  streams[1], 1, nelem, exponent[:516], out=out)
        if fused is not None:
            return fused

    decoded_planes = []
    for k in range(nplanes):
        m = plane_methods[k]
        s_bytes = streams[k]
        if m == 0:
            if len(s_bytes) != nelem:
                return None
            decoded_planes.append(s_bytes)
        elif m == 3:
            off_bytes = struct.pack("<QQ", 0, len(s_bytes))
            dec = decode_batch(s_bytes, off_bytes, 1, nelem, header=plane_hdrs[k])
            if dec is None:
                return None
            decoded_planes.append(dec)
        else:
            return None

    need = nplanes * nelem
    dst = out if (out is not None and len(out) >= need) else bytearray(need)
    pairs = [(p, 0) for p in decoded_planes]
    if nplanes == 2 and codec == CODEC_BF16:
        kernels.merge_bf16(pairs, nelem, dst)
    else:
        kernels.merge_planes(pairs, nplanes, nelem, dst)
    return memoryview(dst)[:need]


def decode_chunks(chunks_and_payloads, tables=None):
    """Decode a batch of (chunk, payload) pairs using the GPU decoder.

    Groups rANS streams across chunks by plane kind to maximize GPU parallel occupancy.
    Returns a list of decoded memoryviews, or None if the batch cannot be decoded.
    """
    if not chunks_and_payloads:
        return []
    lib = _load()
    if lib is None:
        return None
    from ..format import CODEC_SPLIT, CODEC_BF16, CODEC_SPLIT_ST, CODEC_BF16C
    from .. import kernels

    for item in chunks_and_payloads:
        c = item[0]
        codec = getattr(c, "codec", None)
        esize = getattr(c, "esize", None)
        flags = getattr(c, "flags", None)
        rlen = getattr(c, "rlen", None)
        if not can_decode_chunk(codec, esize, flags, rlen):
            return None

    results = []
    first_c = chunks_and_payloads[0][0]
    uniform = all(
        item[0].codec == first_c.codec and item[0].esize == first_c.esize and item[0].rlen == first_c.rlen
        for item in chunks_and_payloads
    )

    if not uniform or len(chunks_and_payloads) == 1:
        for c, p in chunks_and_payloads:
            dec = decode_chunk(p, c.codec, c.esize, c.flags, c.rlen, tables=tables)
            if dec is None:
                return None
            results.append(dec)
        return results

    N = len(chunks_and_payloads)
    codec = first_c.codec
    esize = first_c.esize
    rlen = first_c.rlen

    if codec == CODEC_BF16C:
        from ..codec import _bf16c_hdr
        nelem = rlen // 2
        hdr_size = _bf16c_hdr.size
        parsed = []
        for c, p in chunks_and_payloads:
            if len(p) < hdr_size:
                return None
            k, exp_method, *rest = _bf16c_hdr.unpack_from(p, 0)
            sm_methods = rest[:k]
            exp_len = rest[k]
            seg_lens = rest[k + 1:]
            pos = hdr_size
            if pos + exp_len > len(p):
                return None
            exp_payload = p[pos:pos + exp_len]
            pos += exp_len
            parsed.append((k, exp_method, sm_methods, exp_len, seg_lens, exp_payload, pos, p))

        all_rans = all(item[1] == 3 for item in parsed)
        if all_rans:
            cat = bytearray()
            offs = bytearray()
            c_off = 0
            for item in parsed:
                s = item[5]
                slen = len(s)
                offs += struct.pack("<QQ", c_off, slen)
                cat += s
                c_off += slen
            batch_exp = decode_batch(bytes(cat), bytes(offs), N, nelem)
            if batch_exp is not None:
                decoded_exps = [batch_exp[i * nelem:(i + 1) * nelem] for i in range(N)]
            else:
                decoded_exps = [kernels.rans_decode(item[5], nelem) for item in parsed]
        else:
            decoded_exps = []
            for item in parsed:
                if item[1] == 0:
                    decoded_exps.append(item[5])
                else:
                    d = kernels.rans_decode(item[5], nelem)
                    if d is None:
                        return None
                    decoded_exps.append(d)

        for i in range(N):
            exp = decoded_exps[i]
            k, exp_method, sm_methods, exp_len, seg_lens, exp_payload, pos, p = parsed[i]
            ehist = kernels.histogram(exp)
            lut = kernels.bucket_lut(ehist, k)
            counts = kernels.bucket_counts(ehist, lut, k)

            streams = []
            mv = memoryview(p)
            for b in range(k):
                ln = seg_lens[b]
                want = counts[b]
                m = sm_methods[b]
                if want == 0:
                    streams.append((b"", 0))
                elif m == 0:
                    streams.append((p, pos))
                elif m == 3:
                    block = kernels.rans_decode(mv[pos:pos + ln], want)
                    if block is None:
                        return None
                    streams.append((block, 0))
                else:
                    return None
                pos += ln

            dst = bytearray(nelem * 2)
            res = kernels.unpartition_merge_bf16(exp, streams, lut, k, dst)
            results.append(memoryview(res))
        return results
    nplanes = 2 if codec == CODEC_BF16 else esize
    nelem = rlen // nplanes
    hdr_size = nplanes * 4

    chunk_streams = []
    chunk_methods = []
    for c, p in chunks_and_payloads:
        if len(p) < hdr_size:
            return None
        lengths = struct.unpack_from(f"<{nplanes}I", p, 0)
        pos = hdr_size
        p_streams = []
        p_methods = []
        for k in range(nplanes):
            ln = lengths[k]
            if pos + ln > len(p):
                return None
            p_methods.append((c.flags >> (2 * k)) & 0x3)
            p_streams.append(p[pos:pos + ln])
            pos += ln
        chunk_streams.append(p_streams)
        chunk_methods.append(p_methods)

    decoded_by_plane = []
    for k in range(nplanes):
        hdr = None
        if codec == CODEC_SPLIT_ST and tables is not None:
            hdr = tables.get(CODEC_SPLIT, esize, k)

        methods_k = [chunk_methods[i][k] for i in range(N)]
        if all(m == 0 for m in methods_k):
            plane_k_decoded = [chunk_streams[i][k] for i in range(N)]
            decoded_by_plane.append(plane_k_decoded)
        elif all(m == 3 for m in methods_k):
            cat_streams = bytearray()
            offsets = bytearray()
            off = 0
            for i in range(N):
                s = chunk_streams[i][k]
                slen = len(s)
                offsets += struct.pack("<QQ", off, slen)
                cat_streams += s
                off += slen
            batch_dec = decode_batch(bytes(cat_streams), bytes(offsets), N, nelem, header=hdr)
            if batch_dec is None:
                return None
            plane_k_decoded = [batch_dec[i * nelem:(i + 1) * nelem] for i in range(N)]
            decoded_by_plane.append(plane_k_decoded)
        else:
            plane_k_decoded = []
            for i in range(N):
                m = methods_k[i]
                s = chunk_streams[i][k]
                if m == 0:
                    plane_k_decoded.append(s)
                else:
                    off_bytes = struct.pack("<QQ", 0, len(s))
                    d = decode_batch(s, off_bytes, 1, nelem, header=hdr)
                    if d is None:
                        return None
                    plane_k_decoded.append(d)
            decoded_by_plane.append(plane_k_decoded)

    for i in range(N):
        planes_i = [(decoded_by_plane[k][i], 0) for k in range(nplanes)]
        dst = bytearray(rlen)
        if nplanes == 2 and codec == CODEC_BF16:
            kernels.merge_bf16(planes_i, nelem, dst)
        else:
            kernels.merge_planes(planes_i, nplanes, nelem, dst)
        results.append(memoryview(dst))

    return results

