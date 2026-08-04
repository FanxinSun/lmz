"""Compile the native byte-plane kernel.

The shared library is written next to this file, inside the package. Nothing
is installed and no interpreter or system state is modified; if a compiler is
missing or the build fails, callers fall back to a pure-Python path.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, "lmzcore.c")

_SUFFIX = ".dylib" if sys.platform == "darwin" else (".dll" if os.name == "nt" else ".so")


def _source_tag() -> str:
    """Identity of the artifact: source contents plus target machine."""
    try:
        with open(SOURCE, "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()[:12]
    except OSError:
        digest = "nosrc"
    return f"{platform.machine()}-{digest}"


def library_path() -> str:
    return os.path.join(HERE, f"lmzcore-{_source_tag()}{_SUFFIX}")


def find_compiler() -> str | None:
    for name in filter(None, [os.environ.get("CC"), "cc", "gcc", "clang"]):
        path = shutil.which(name)
        if path:
            return path
    return None


def build(force: bool = False, verbose: bool = False) -> str | None:
    """Build the kernel if needed and return its path, or None if unavailable.

    Never raises: a missing compiler is a normal condition, not an error.
    """
    out = library_path()
    if os.path.exists(out) and not force:
        return out
    if not os.path.exists(SOURCE):
        return None

    cc = find_compiler()
    if cc is None:
        if verbose:
            print("lmz: no C compiler found; using fallback kernels", file=sys.stderr)
        return None

    flags = ["-O3", "-fPIC", "-std=c11", "-fvisibility=hidden", "-w"]
    if sys.platform == "darwin":
        link = ["-dynamiclib"]
    elif os.name == "nt":
        link = ["-shared"]
    else:
        link = ["-shared"]

    # Compile to a temporary name in the same directory, then move into place,
    # so concurrent builds and interrupted runs cannot leave a partial library.
    fd, tmp = tempfile.mkstemp(dir=HERE, suffix=_SUFFIX)
    os.close(fd)
    cmd = [cc, *flags, *link, "-o", tmp, SOURCE]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if proc.returncode != 0:
            if verbose:
                print(f"lmz: kernel build failed:\n{proc.stderr}", file=sys.stderr)
            os.unlink(tmp)
            return None
        os.replace(tmp, out)
        if verbose:
            print(f"lmz: built {os.path.basename(out)}", file=sys.stderr)
        return out
    except Exception as exc:  # compiler missing mid-flight, timeout, sandbox, ...
        if verbose:
            print(f"lmz: kernel build failed: {exc}", file=sys.stderr)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        return None


def clean() -> int:
    """Remove built libraries. Returns how many were deleted."""
    n = 0
    for name in os.listdir(HERE):
        if name.startswith("lmzcore-") and name.endswith(_SUFFIX):
            try:
                os.unlink(os.path.join(HERE, name))
                n += 1
            except OSError:
                pass
    return n


if __name__ == "__main__":
    path = build(force="--force" in sys.argv, verbose=True)
    print(path or "no native kernel")
