"""Compress, decompress, verify and inspect archives.

Compression reads chunks through positional reads, encodes them across a
thread pool, and appends the results in order. Decompression is the reverse
and needs no ordering at all: every chunk knows its destination offset, so
threads decode and place them with positional writes and never synchronise.
That is what keeps decompression close to disk speed on a large model.
"""

from __future__ import annotations

import io
import os
import posixpath
import sys
import threading
import time
from bisect import bisect_right
from dataclasses import dataclass, field

from . import codec as _codec
from . import entropy, kernels
from .format import ArchiveReader, ArchiveWriter, FormatError, Member
from .parallel import default_workers, ordered_map, unordered_map
from .planner import chunkify, probe

DEFAULT_CHUNK_SIZE = 8 << 20
DEFAULT_LEVEL = 1

# Files that are not model weights but belong with them; kept so a compressed
# model directory round-trips into something directly usable.
SKIP_NAMES = {".DS_Store"}


@dataclass
class Stats:
    input_bytes: int = 0
    output_bytes: int = 0
    seconds: float = 0.0
    chunks: int = 0
    files: int = 0
    detail: dict = field(default_factory=dict)

    @property
    def ratio(self) -> float:
        return self.input_bytes / self.output_bytes if self.output_bytes else 0.0

    @property
    def saved(self) -> float:
        if not self.input_bytes:
            return 0.0
        return 1.0 - self.output_bytes / self.input_bytes

    @property
    def throughput(self) -> float:
        """Input bytes per second."""
        return self.input_bytes / self.seconds if self.seconds > 0 else 0.0


class _FdPool:
    """Per-thread file descriptors, so positional I/O needs no locking."""

    def __init__(self, paths: list[str], flags: int):
        self.paths = paths
        self.flags = flags
        self._local = threading.local()
        self._all: list[int] = []
        self._lock = threading.Lock()

    def get(self, idx: int) -> int:
        table = getattr(self._local, "table", None)
        if table is None:
            table = self._local.table = {}
        fd = table.get(idx)
        if fd is None:
            fd = os.open(self.paths[idx], self.flags)
            table[idx] = fd
            with self._lock:
                self._all.append(fd)
        return fd

    def close(self) -> None:
        with self._lock:
            for fd in self._all:
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._all.clear()


def _preallocate(fd: int, size: int) -> None:
    """Reserve the output file's blocks up front.

    Chunks are written at scattered offsets, and on a sparse file each write
    faults in and clears a fresh page -- measured at roughly a ninth of the
    speed of writing into space already reserved. On a disk filesystem
    reserving 2 GiB is a metadata update costing about 15 ms, so it repays
    that many times over.

    RAM-backed filesystems are the exception: there a block is a page of
    memory, so reserving means allocating the file's entire size in one
    serial pass (1.3 s for the same 2 GiB) to speed up writes that were
    already cheap. Those are left sparse so the worker threads can fault
    pages in as they go, in parallel.

    fallocate is called directly rather than through posix_fallocate because
    the latter quietly emulates itself by writing zeros across the whole file
    where the filesystem has no support -- tolerable for a small file,
    ruinous for a 400 GB one.
    """
    if size <= 0:
        return
    fn = _linux_fallocate()
    if fn is not None and _fs_magic(fd) not in (TMPFS_MAGIC, RAMFS_MAGIC):
        if fn(fd, 0, 0, size) == 0:
            return
    try:
        os.ftruncate(fd, size)
    except OSError:
        pass


TMPFS_MAGIC = 0x01021994
RAMFS_MAGIC = 0x858458F6


def _fs_magic(fd: int):
    """Filesystem type behind an open fd, or None if it cannot be determined."""
    import ctypes

    if not sys.platform.startswith("linux") or ctypes.sizeof(ctypes.c_void_p) != 8:
        return None
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        buf = ctypes.create_string_buffer(128)
        if libc.fstatfs(fd, buf) != 0:
            return None
        # f_type is the first field of struct statfs, one word wide on LP64.
        return int.from_bytes(buf.raw[:8], sys.byteorder, signed=True)
    except Exception:
        return None


_decode_scratch = threading.local()


def _scratch(nbytes: int) -> bytearray:
    """A per-thread decode buffer, so the hot path allocates nothing."""
    buf = getattr(_decode_scratch, "buf", None)
    if buf is None or len(buf) < nbytes:
        buf = _decode_scratch.buf = bytearray(nbytes)
    return buf


_fallocate_fn = False


def _linux_fallocate():
    global _fallocate_fn
    if _fallocate_fn is False:
        _fallocate_fn = None
        if sys.platform.startswith("linux"):
            try:
                import ctypes

                libc = ctypes.CDLL(None, use_errno=True)
                fn = libc.fallocate
                fn.restype = ctypes.c_int
                fn.argtypes = [ctypes.c_int, ctypes.c_int,
                               ctypes.c_int64, ctypes.c_int64]
                _fallocate_fn = fn
            except Exception:
                _fallocate_fn = None
    return _fallocate_fn


def _iter_files(root: str):
    """Regular files under `root`, in a stable order, as relative posix paths."""
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for name in sorted(filenames):
            if name in SKIP_NAMES:
                continue
            full = os.path.join(dirpath, name)
            if not os.path.isfile(full) or os.path.islink(full):
                continue
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            yield rel, full


def _safe_member_path(base: str, rel: str) -> str:
    """Resolve an archive member path under `base`, refusing escapes."""
    rel = rel.replace("\\", "/")
    if posixpath.isabs(rel) or (len(rel) > 1 and rel[1] == ":"):
        raise FormatError(f"archive member has an absolute path: {rel!r}")
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise FormatError(f"archive member escapes the output directory: {rel!r}")
    if not parts:
        raise FormatError("archive member has an empty path")
    return os.path.join(base, *parts)


# ------------------------------------------------------------------ compress


def compress(src: str, dst: str, *, level: int = DEFAULT_LEVEL,
             workers: int | None = None, chunk_size: int = DEFAULT_CHUNK_SIZE,
             checksum: bool = True, progress=None) -> Stats:
    """Compress a file or directory into the archive at `dst`."""
    workers = workers or default_workers()
    chunk_size = max(chunk_size, 1 << 16)
    started = time.perf_counter()

    if os.path.isdir(src):
        entries = list(_iter_files(src))
        if not entries:
            raise ValueError(f"{src} contains no regular files")
    elif os.path.isfile(src):
        entries = [(os.path.basename(src), src)]
    else:
        raise FileNotFoundError(src)

    members: list[Member] = []
    paths: list[str] = []
    plan: list[tuple[int, int, int, int, int]] = []  # member, start, end, esize, kind
    base = 0
    for rel, full in entries:
        size = os.path.getsize(full)
        with open(full, "rb") as fh:
            layout = probe(fh, size)
        member = Member(path=rel, size=size, dst=base, kind=layout.kind,
                        mode=os.stat(full).st_mode & 0o777, tensors=layout.tensors)
        idx = len(members)
        members.append(member)
        paths.append(full)
        for start, end, esize, kind in chunkify(layout, size, chunk_size):
            plan.append((idx, start, end, esize, kind))
        base += size

    total_in = base
    manifest = {
        "version": 1,
        "tool": f"lmz {_version()}",
        "level": level,
        "chunk_size": chunk_size,
        "entropy": entropy.BACKEND,
        "checksum": "crc32" if checksum else "none",
        "total_size": total_in,
        "members": [m.to_json() for m in members],
    }

    pool = _FdPool(paths, os.O_RDONLY)
    done = 0
    stats = Stats(input_bytes=total_in, files=len(members), chunks=len(plan))

    def work(task):
        idx, start, end, esize, kind = task
        data = os.pread(pool.get(idx), end - start, start)
        if len(data) != end - start:
            raise IOError(f"short read on {members[idx].path} at {start}")
        parts, cid, flags, crc = _codec.encode_chunk(data, esize, level, checksum,
                                                     kind=kind)
        return members[idx].dst + start, end - start, parts, cid, esize, flags, crc

    tmp = dst + ".part"
    try:
        with open(tmp, "wb", buffering=1 << 20) as out:
            writer = ArchiveWriter(out, manifest)
            for dst_off, rlen, parts, cid, esize, flags, crc in ordered_map(
                    work, plan, workers, lookahead=workers * 2):
                writer.append(parts, dst_off, rlen, crc, cid, esize, flags)
                done += rlen
                if progress:
                    progress(done, total_in)
            writer.close(total_in)
        os.replace(tmp, dst)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    finally:
        pool.close()

    stats.output_bytes = os.path.getsize(dst)
    stats.seconds = time.perf_counter() - started
    return stats


# ---------------------------------------------------------------- decompress


def decompress(src: str, dst: str, *, workers: int | None = None,
               verify_checksums: bool = True, progress=None,
               overwrite: bool = False) -> Stats:
    """Restore an archive to `dst`, byte for byte.

    With a single member, `dst` is the output file unless it is an existing
    directory. With several, `dst` is the directory they are written into.
    """
    workers = workers or default_workers()
    started = time.perf_counter()

    with open(src, "rb") as fh:
        reader = ArchiveReader(fh)
        members = reader.members
        chunks = reader.chunks
        manifest = reader.manifest

    single = len(members) == 1 and "/" not in members[0].path
    if single and not os.path.isdir(dst):
        out_paths = [dst]
        parent = os.path.dirname(os.path.abspath(dst))
        os.makedirs(parent, exist_ok=True)
    else:
        os.makedirs(dst, exist_ok=True)
        out_paths = [_safe_member_path(dst, m.path) for m in members]

    for path, member in zip(out_paths, members):
        if os.path.exists(path) and not overwrite:
            raise FileExistsError(
                f"{path} already exists (pass overwrite=True to replace it)")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        # Chunks land at arbitrary offsets, so the file is created at its
        # final size before any of them are written.
        with open(path, "wb") as fh:
            if member.size:
                _preallocate(fh.fileno(), member.size)

    starts = [m.dst for m in members]
    total_out = sum(m.size for m in members)
    _check_coverage(chunks, total_out)

    in_pool = _FdPool([src], os.O_RDONLY)
    out_pool = _FdPool(out_paths, os.O_WRONLY)
    done = 0

    def work(chunk):
        payload = os.pread(in_pool.get(0), chunk.clen, chunk.off)
        if len(payload) != chunk.clen:
            raise FormatError("archive is truncated")
        data = _codec.decode_chunk(payload, chunk.codec, chunk.esize, chunk.flags,
                                   chunk.rlen, chunk.crc, verify_checksums,
                                   out=_scratch(chunk.rlen))
        i = bisect_right(starts, chunk.dst) - 1
        if i < 0 or chunk.dst + chunk.rlen > members[i].dst + members[i].size:
            raise FormatError(f"chunk at {chunk.dst} does not fit any member")
        written = os.pwrite(out_pool.get(i), data, chunk.dst - members[i].dst)
        if written != len(data):
            raise IOError(f"short write to {out_paths[i]}")
        return chunk.rlen

    try:
        for rlen in unordered_map(work, chunks, workers, lookahead=workers * 3):
            done += rlen
            if progress:
                progress(done, total_out)
    finally:
        in_pool.close()
        out_pool.close()

    for path, member in zip(out_paths, members):
        try:
            os.chmod(path, member.mode or 0o644)
        except OSError:
            pass

    return Stats(input_bytes=total_out, output_bytes=os.path.getsize(src),
                 seconds=time.perf_counter() - started, chunks=len(chunks),
                 files=len(members), detail={"manifest": manifest})


def _check_coverage(chunks, total: int) -> None:
    """Confirm the chunks tile the output exactly, with no gap or overlap."""
    pos = 0
    for c in sorted(chunks, key=lambda c: c.dst):
        if c.dst != pos:
            raise FormatError(
                f"archive does not cover its output: gap or overlap at byte {pos}")
        pos += c.rlen
    if pos != total:
        raise FormatError(
            f"archive covers {pos} bytes but declares {total}")


# -------------------------------------------------------------------- verify


def verify(src: str, *, workers: int | None = None, progress=None) -> Stats:
    """Decode every chunk and check it, without writing anything."""
    workers = workers or default_workers()
    started = time.perf_counter()

    with open(src, "rb") as fh:
        reader = ArchiveReader(fh)
        chunks = reader.chunks
        members = reader.members

    total = sum(m.size for m in members)
    _check_coverage(chunks, total)
    pool = _FdPool([src], os.O_RDONLY)
    done = 0

    def work(chunk):
        payload = os.pread(pool.get(0), chunk.clen, chunk.off)
        if len(payload) != chunk.clen:
            raise FormatError("archive is truncated")
        _codec.decode_chunk(payload, chunk.codec, chunk.esize, chunk.flags,
                            chunk.rlen, chunk.crc, True, out=_scratch(chunk.rlen))
        return chunk.rlen

    try:
        for rlen in unordered_map(work, chunks, workers, lookahead=workers * 3):
            done += rlen
            if progress:
                progress(done, total)
    finally:
        pool.close()

    return Stats(input_bytes=total, output_bytes=os.path.getsize(src),
                 seconds=time.perf_counter() - started, chunks=len(chunks),
                 files=len(members))


# --------------------------------------------------------------- inspection


def info(src: str) -> dict:
    """Archive metadata: members, tensors, and how the chunks were encoded."""
    with open(src, "rb") as fh:
        reader = ArchiveReader(fh)
        by_codec: dict[str, list[int]] = {}
        for c in reader.chunks:
            name = {0: "stored", 1: "entropy", 2: "split",
                    3: "bf16-split"}.get(c.codec, "?")
            slot = by_codec.setdefault(name, [0, 0, 0])
            slot[0] += 1
            slot[1] += c.rlen
            slot[2] += c.clen
        return {
            "path": src,
            "archive_size": reader.file_size,
            "original_size": reader.original_size,
            "chunks": len(reader.chunks),
            "manifest": reader.manifest,
            "members": [m.to_json() for m in reader.members],
            "codecs": by_codec,
            "payload_bytes": sum(c.clen for c in reader.chunks),
        }


def read_tensor(src: str, name: str, member: str | None = None) -> tuple[str, list, bytes]:
    """Extract one tensor without decompressing the rest of the archive.

    Returns (dtype, shape, raw little-endian bytes).
    """
    with open(src, "rb") as fh:
        reader = ArchiveReader(fh)
        target = None
        for m in reader.members:
            if member is not None and m.path != member:
                continue
            if m.tensors and name in m.tensors:
                target = (m, m.tensors[name])
                break
        if target is None:
            where = f" in {member}" if member else ""
            raise KeyError(f"no tensor named {name!r}{where}")

        m, meta = target
        start, end = meta["offsets"]
        lo, hi = m.dst + start, m.dst + end
        starts = [c.dst for c in reader.chunks]
        i = max(0, bisect_right(starts, lo) - 1)
        out = bytearray()
        for c in reader.chunks[i:]:
            if c.dst >= hi:
                break
            if c.dst + c.rlen <= lo:
                continue
            fh.seek(c.off)
            payload = fh.read(c.clen)
            data = _codec.decode_chunk(payload, c.codec, c.esize, c.flags,
                                       c.rlen, c.crc, True)
            a = max(lo, c.dst) - c.dst
            b = min(hi, c.dst + c.rlen) - c.dst
            out += data[a:b]
        return meta.get("dtype", ""), meta.get("shape", []), bytes(out)


def _version() -> str:
    from . import __version__

    return __version__


def backends() -> dict:
    """What the tool is actually using, for diagnostics."""
    return {"kernel": kernels.backend(), "entropy": entropy.BACKEND,
            "workers": default_workers()}
