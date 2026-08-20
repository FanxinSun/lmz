"""Compile the CUDA decoder.

Same contract as lmz/native/build.py, and for the same reason: the wheel stays
pure Python, nothing platform-specific is ever uploaded, and a machine without
a toolchain is a normal condition rather than an error. The shared library is
written next to this file, inside the package; nothing is installed and no
system state is modified.

`pip install lmzip` therefore does not need CUDA. If nvcc and a device are both
present the GPU decoder appears on first use, and if either is missing the
decoder stays on the CPU and says so in `lmz doctor`.
"""

from __future__ import annotations

import glob
import hashlib
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "lmzgpu.cu")

_SUFFIX = ".dll" if os.name == "nt" else ".so"

# Why the last build attempt failed, quoted by `lmz doctor`. A build that
# never ran leaves this empty.
last_error = ""

# Architectures to emit when the local device cannot be identified. Volta
# through Ada; Blackwell is deliberately absent because an nvcc older than 12.8
# cannot target it and would fail the whole build for the sake of a guess.
_FALLBACK_ARCHS = ("70", "80", "86", "89")


_arch_cache: list = []


def _device_arch() -> str | None:
    """Compute capability of device 0 as `sm_120`, or None if it cannot be read.

    Asked of the driver rather than of a CUDA context, so this costs no GPU
    memory and works before anything of ours has initialised. Cached, because
    it names the built artifact and so is asked for on every probe.
    """
    if _arch_cache:
        return _arch_cache[0]
    arch = _query_arch()
    _arch_cache.append(arch)
    return arch


def _query_arch() -> str | None:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return None
    try:
        out = subprocess.run([smi, "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    first = out.stdout.strip().splitlines()
    if not first:
        return None
    m = re.match(r"^\s*(\d+)\.(\d+)\s*$", first[0])
    return f"sm_{m.group(1)}{m.group(2)}" if m else None


def _nvcc_version(path: str) -> tuple[int, int]:
    """(major, minor) of an nvcc, or (0, 0) if it will not say."""
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True,
                             timeout=30)
    except Exception:
        return (0, 0)
    m = re.search(r"release (\d+)\.(\d+)", out.stdout)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def find_compilers() -> list[str]:
    """Every nvcc worth trying, newest first.

    Newest first rather than PATH first because a machine very often has an old
    toolkit on PATH beside a new one in /usr/local, and the old one cannot
    target a new card: CUDA 12.4 answers `-arch=sm_120` with "not defined for
    option gpu-architecture" and there is no way to discover that except by
    asking. An explicit NVCC still wins outright -- that is the caller saying
    which one they mean.
    """
    pinned = os.environ.get("NVCC")
    if pinned:
        path = shutil.which(pinned) or (pinned if os.path.exists(pinned) else None)
        if path:
            return [path]

    seen, cands = set(), []
    for path in [shutil.which("nvcc"), *glob.glob("/usr/local/cuda*/bin/nvcc")]:
        if not path:
            continue
        real = os.path.realpath(path)
        if real in seen:
            continue
        seen.add(real)
        cands.append(path)
    return sorted(cands, key=_nvcc_version, reverse=True)


def find_compiler() -> str | None:
    """The nvcc that would be tried first, or None if there is none."""
    c = find_compilers()
    return c[0] if c else None


def _source_tag() -> str:
    """Identity of the artifact: source, machine, and the arch it was built for.

    The arch is in the tag because -arch bakes the target into the binary and
    the machine name does not change when the card does. Without it, moving an
    installed package between two x86_64 boxes with different GPUs would reuse
    a library that cannot launch.
    """
    try:
        with open(SOURCE, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()[:12]
    except OSError:
        digest = "nosrc"
    return f"{platform.machine()}-{_device_arch() or 'fat'}-{digest}"


def library_path() -> str:
    return os.path.join(HERE, f"lmzgpu-{_source_tag()}{_SUFFIX}")


def build(force: bool = False, verbose: bool = False) -> str | None:
    """Build the CUDA decoder if needed and return its path, or None.

    Never raises: no nvcc, no device and a failed compile are all ordinary.
    """
    out = library_path()
    if os.path.exists(out) and not force:
        return out
    if not os.path.exists(SOURCE):
        return None

    global last_error
    nvccs = find_compilers()
    if not nvccs:
        last_error = "no nvcc found"
        if verbose:
            print("lmz: no nvcc found; the decoder stays on the CPU", file=sys.stderr)
        return None

    arch = _device_arch()
    if arch:
        gencode = [f"-arch={arch}"]
    else:
        gencode = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in _FALLBACK_ARCHS]

    # -cudart static so the result depends on the driver alone. A .so living
    # inside a Python package must not break because a CUDA toolkit moved.
    flags = ["-O3", "-std=c++17", "-cudart", "static",
             "-Xcompiler", "-fPIC", "-shared", *gencode]

    for nvcc in nvccs:
        fd, tmp = tempfile.mkstemp(dir=HERE, suffix=_SUFFIX)
        os.close(fd)
        cmd = [nvcc, *flags, "-o", tmp, SOURCE]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if proc.returncode == 0:
                os.replace(tmp, out)
                last_error = ""
                if verbose:
                    print(f"lmz: built {os.path.basename(out)} with {nvcc}",
                          file=sys.stderr)
                return out
            last_error = (proc.stderr.strip().splitlines() or ["nvcc failed"])[-1]
            if verbose:
                print(f"lmz: {nvcc} failed:\n{proc.stderr}", file=sys.stderr)
        except Exception as exc:  # nvcc vanished mid-flight, timeout, sandbox
            last_error = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"lmz: {nvcc} failed: {exc}", file=sys.stderr)
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return None


def clean() -> int:
    """Remove built libraries. Returns how many were deleted."""
    n = 0
    for name in os.listdir(HERE):
        if name.startswith("lmzgpu-") and name.endswith(_SUFFIX):
            try:
                os.unlink(os.path.join(HERE, name))
                n += 1
            except OSError:
                pass
    return n


if __name__ == "__main__":
    path = build(force="--force" in sys.argv, verbose=True)
    print(path or "no CUDA decoder")
