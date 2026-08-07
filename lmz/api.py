"""Compress, decompress, verify and inspect archives.

Compression reads chunks through positional reads, encodes them across a
thread pool, and appends the results in order. Decompression is the reverse
and needs no ordering at all: every chunk knows its destination offset, so
threads decode and place them with positional writes and never synchronise.
That is what keeps decompression close to disk speed on a large model.
"""

from __future__ import annotations

import hashlib
import io
import os
import posixpath
import struct
import sys
import threading
import time
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass, field

from . import codec as _codec
from . import entropy, kernels
from .format import (CODEC_DELTA, CODEC_REF, FLAG_ALIGNED, FLAG_PAGE_MAPPED,
                     HEADER_SIZE, PAGE_ALIGN, ArchiveReader, ArchiveWriter,
                     CorruptArchive, FormatError, Member)
from .parallel import (default_workers, gil_enabled, ordered_map,
                        unordered_map)
from .planner import KIND_DELTA, KIND_REF, chunkify, probe

DEFAULT_CHUNK_SIZE = 8 << 20
DEFAULT_LEVEL = 1

# Block size for page-mapped archives. Measured on real BF16 weights, this is
# where the two curves cross: 64 KiB keeps 33.14% against 34.34% at 8 MiB --
# 1.2 points -- while a block decodes in 74 us instead of 14 ms. 74 us is
# faster than the flash read it replaces on any phone or NVMe drive, so
# decompression stops being a cost and hides behind I/O. Below this it falls
# away fast: 16 KiB gives 30.65%, and at 4 KiB the frequency tables cost more
# than the data saves and every block is stored raw.
DEFAULT_BLOCK_SIZE = 64 << 10

# Tensors below this size are not worth deduplicating: the bookkeeping and
# hashing outweigh a few kilobytes of norm weights.
DEDUP_MIN_TENSOR = 64 << 10

# Delta coding: a tensor that is merely *close* to one in an earlier member.
# A training run rewrites every weight each checkpoint and moves each a
# little, so nothing dedups while the XOR is nearly all zeros -- measured at
# 69.2% against 56.3% for coding the checkpoint alone, with not one tensor
# byte-identical. The floor matches dedup's, because the candidates that
# matter are not always large: a real Adam state saves two moments per
# parameter as separate ~91 KB storages, and one of the two is where the whole
# win lives. The sample is large enough that a block-quantised candidate still
# clears BLOCK_MIN_BLOCKS and so is judged by the coder that would really run
# on it.
DELTA_MIN_TENSOR = DEDUP_MIN_TENSOR
DELTA_SAMPLE = 1 << 20
DELTA_MIN_GAIN = 0.05  # the difference must beat coding alone by this much

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


def _find_duplicate_tensors(paths: list[str], members: list[Member],
                            layouts: list, workers: int):
    """Byte-identical tensors across (and within) members, found by hashing.

    Checkpoints genuinely repeat themselves: a consolidated file shipped
    alongside its own shards, tied embeddings stored twice. Tensors are
    grouped by size, then by a sampled digest, and only the still-colliding
    groups are hashed in full, so the extra reads stay proportional to data
    that really is duplicated. Returns ({member index: [(start, end, source
    output offset)]}, deduplicated bytes).
    """
    by_size: dict[int, list] = {}
    for idx, layout in enumerate(layouts):
        if layout is None or not layout.tensors:
            continue
        seen: set[tuple[int, int]] = set()
        for meta in layout.tensors.values():
            s, e = meta["offsets"]
            # Aliased names over one range must not become self-referencing.
            if e - s >= DEDUP_MIN_TENSOR and (s, e) not in seen:
                seen.add((s, e))
                by_size.setdefault(e - s, []).append((idx, s, e))
    flat = [item for group in by_size.values() if len(group) >= 2
            for item in group]
    if not flat:
        return {}, 0

    pool = _FdPool(paths, os.O_RDONLY)
    try:
        def sample_digest(item):
            idx, s, e = item
            n = e - s
            h = hashlib.blake2b(digest_size=16)
            fd = pool.get(idx)
            if n <= 3 * 65536:
                h.update(os.pread(fd, n, s))
            else:
                for off in (0, (n // 2) & ~63, n - 65536):
                    h.update(os.pread(fd, 65536, s + off))
            return item, h.digest()

        sampled: dict = {}
        for item, d in unordered_map(sample_digest, flat, workers):
            sampled.setdefault((item[2] - item[1], d), []).append(item)

        def full_digest(item):
            idx, s, e = item
            h = hashlib.blake2b(digest_size=32)
            fd = pool.get(idx)
            pos = s
            while pos < e:
                block = os.pread(fd, min(1 << 22, e - pos), pos)
                if not block:
                    raise IOError(f"short read hashing {paths[idx]} at {pos}")
                h.update(block)
                pos += len(block)
            return item, h.digest()

        flat2 = [item for group in sampled.values() if len(group) >= 2
                 for item in group]
        confirmed: dict = {}
        for item, d in unordered_map(full_digest, flat2, workers):
            confirmed.setdefault((item[2] - item[1], d), []).append(item)
    finally:
        pool.close()

    refs: dict[int, list] = {}
    saved = 0
    for group in confirmed.values():
        if len(group) < 2:
            continue
        group.sort()
        pidx, ps, _pe = group[0]
        src = members[pidx].dst + ps
        for idx, s, e in group[1:]:
            refs.setdefault(idx, []).append((s, e, src))
            saved += e - s
    return refs, saved


def append(archive: str, src: str, *, level: int = DEFAULT_LEVEL,
           workers: int | None = None, checksum: bool = True,
           delta: bool = True, progress=None) -> Stats:
    """Add files to an existing archive, coded against what is already in it.

    This is what makes the corpus-level savings reachable in practice. A
    training run produces checkpoints over days, and the 65% that delta coding
    finds between them is only available if they share an archive -- but
    recompressing every earlier checkpoint to add one is not a workflow. Here
    the new files are coded against the archive's own contents, which are read
    back by decoding rather than from the original files, so the sources may
    be long gone.

    Only the tail is rewritten: payloads are appended where the old chunk
    table began, and the table, manifest and footer are rebuilt after them.
    """
    workers = workers or default_workers()
    started = time.perf_counter()

    with open(archive, "rb") as fh:
        reader = ArchiveReader(fh)
    old_members = reader.members
    old_chunks = reader.chunks
    old_manifest = reader.manifest
    chunk_size = max(int(old_manifest.get("chunk_size") or DEFAULT_CHUNK_SIZE),
                     1 << 16)
    table_off = reader.table_off
    base_total = sum(m.size for m in old_members)

    if os.path.isdir(src):
        entries = list(_iter_files(src))
        if not entries:
            raise ValueError(f"{src} contains no regular files")
    elif os.path.isfile(src):
        entries = [(os.path.basename(src), src)]
    else:
        raise FileNotFoundError(src)

    taken = {m.path for m in old_members}
    members: list[Member] = []
    paths: list[str] = []
    layouts: list = []
    sizes: list[int] = []
    base = base_total
    for rel, full in entries:
        if rel in taken:
            raise ValueError(f"{rel} is already in {archive}")
        size = os.path.getsize(full)
        with open(full, "rb") as fh:
            layout = probe(fh, size)
        members.append(Member(path=rel, size=size, dst=base, kind=layout.kind,
                              mode=os.stat(full).st_mode & 0o777,
                              tensors=layout.tensors))
        paths.append(full)
        layouts.append(layout)
        sizes.append(size)
        base += size
    total_new = base - base_total

    arc = MappedArchive(archive, verify=False, cache_blocks=8)
    try:
        deltas_by_member: dict[int, list] = {}
        delta_bytes = 0
        if delta:
            deltas_by_member, delta_bytes = _find_append_deltas(
                paths, members, layouts, old_members, arc, level)

        plan: list[tuple] = []
        for idx, layout in enumerate(layouts):
            for task in chunkify(layout, sizes[idx], chunk_size,
                                 deltas=deltas_by_member.get(idx)):
                plan.append((idx,) + task)

        pool = _FdPool(paths, os.O_RDONLY)
        done = 0
        stats = Stats(input_bytes=total_new, files=len(members),
                      chunks=len(plan),
                      detail={"delta_bytes": delta_bytes, "appended": True})

        def work(task):
            idx, start, end, esize, kind, src_off, btype, ikind = task
            data = os.pread(pool.get(idx), end - start, start)
            if len(data) != end - start:
                raise IOError(f"short read on {members[idx].path} at {start}")
            if kind == KIND_DELTA:
                base_bytes = arc.read(src_off, end - start)
                parts, flags, crc = _codec.encode_delta(
                    data, base_bytes, esize, level, checksum, kind=ikind,
                    btype=btype, src=src_off)
                return (members[idx].dst + start, end - start, parts,
                        CODEC_DELTA, esize, flags, crc)
            parts, cid, flags, crc = _codec.encode_chunk(
                data, esize, level, checksum, kind=kind, btype=btype)
            return (members[idx].dst + start, end - start, parts, cid, esize,
                    flags, crc)

        def work_batch(batch):
            return [work(t) for t in batch]

        manifest = dict(old_manifest)
        manifest["members"] = [m.to_json() for m in old_members + members]
        manifest["total_size"] = base
        if delta_bytes:
            manifest["delta_bytes"] = (old_manifest.get("delta_bytes", 0)
                                       + delta_bytes)

        with open(archive, "r+b") as out:
            writer = ArchiveWriter(out, manifest, flags=reader.flags,
                                   align=PAGE_ALIGN if reader.aligned else 0,
                                   resume_at=table_off, chunks=old_chunks)
            nw = _worker_cap([e - s for _i, s, e, *_r in plan], workers)
            for results in ordered_map(
                    work_batch, _batches(plan, lambda t: t[2] - t[1]), nw,
                    lookahead=nw * 2):
                for dst_off, rlen, parts, cid, esize, flags, crc in results:
                    writer.append(parts, dst_off, rlen, crc, cid, esize, flags)
                    done += rlen
                if progress:
                    progress(done, total_new)
            writer.close(base)
            # An append can leave a shorter tail than the one it replaced.
            out.truncate(writer.end)
        pool.close()
    finally:
        arc.close()

    stats.output_bytes = os.path.getsize(archive)
    stats.seconds = time.perf_counter() - started
    return stats


def _find_append_deltas(paths, members, layouts, old_members, arc, level):
    """New tensors that are close to one already in the archive.

    Same rule as within a single job -- matched by name, dtype and size, and
    decided by encoding a sample both ways -- except the base is decoded out
    of the archive rather than read from a file that may no longer exist.
    """
    # A delta may only name a plain chunk -- resolution is a single hop by
    # design -- so a tensor that is itself stored as a difference cannot be a
    # base. In a checkpoint series that leaves the first checkpoint as the
    # only base, and the difference grows with the gap: measured 12.9 points
    # at a 1000-step distance against 11.6 at 2000. Later checkpoints
    # therefore gain a little less than a fresh archive would give them.
    coded = [(c.dst, c.dst + c.rlen) for c in arc._chunks
             if c.codec in (CODEC_DELTA, CODEC_REF)]

    def is_plain(lo, hi):
        return not any(cs < hi and lo < ce for cs, ce in coded)

    known: dict = {}
    for m in old_members:
        for name, meta in (m.tensors or {}).items():
            s, e = meta["offsets"]
            if e - s < DELTA_MIN_TENSOR or not is_plain(m.dst + s, m.dst + e):
                continue
            # Latest wins: the nearer the base, the smaller the difference.
            known[(name, meta.get("dtype", ""), e - s)] = m.dst + s

    deltas: dict[int, list] = {}
    covered = 0
    for idx, layout in enumerate(layouts):
        region_at = {r.start: r for r in (layout.regions if layout else ())}
        for name, meta in ((layout.tensors or {}) if layout else {}).items():
            s, e = meta["offsets"]
            key = (name, meta.get("dtype", ""), e - s)
            src_off = known.get(key)
            r = region_at.get(s)
            if src_off is None or r is None or r.start != s or r.end != e:
                continue
            take = min(DELTA_SAMPLE, e - s)
            take -= take % max(r.esize, 1)
            if take <= 0:
                continue
            off = ((e - s - take) // 2 // max(r.esize, 1)) * max(r.esize, 1)
            with open(paths[idx], "rb") as fh:
                fh.seek(s + off)
                b = fh.read(take)
            a = arc.read(src_off + off, take)
            if len(a) != take or len(b) != take or a == b:
                # An identical sample still deltas to zeros, which codes for
                # almost nothing; there is no separate ref path to take here.
                if a != b:
                    continue
            diff = bytes(kernels.xor_bytes(b, a))

            def coded(buf):
                parts = _codec.encode_chunk(buf, r.esize, level, False,
                                            kind=r.kind, btype=r.btype)[0]
                return sum(len(p) for p in parts)

            if coded(diff) < coded(b) * (1 - DELTA_MIN_GAIN):
                deltas.setdefault(idx, []).append((s, e, src_off))
                covered += e - s
    for items in deltas.values():
        items.sort()
    return deltas, covered


def extract(archive: str, member: str, dst: str, *, overwrite: bool = False,
            workers: int | None = None) -> Stats:
    """Write one member out without expanding the rest of the archive."""
    started = time.perf_counter()
    if os.path.exists(dst) and not overwrite:
        raise FileExistsError(
            f"{dst} already exists (pass overwrite=True to replace it)")
    with MappedArchive(archive, cache_blocks=16, workers=workers) as arc:
        target = next((m for m in arc.members if m.path == member), None)
        if target is None:
            raise KeyError(f"no member named {member!r} in {archive}")
        parent = os.path.dirname(os.path.abspath(dst))
        os.makedirs(parent, exist_ok=True)
        with open(dst, "wb") as fh:
            pos = 0
            while pos < target.size:
                n = min(8 << 20, target.size - pos)
                fh.write(arc.read(target.dst + pos, n))
                pos += n
        try:
            os.chmod(dst, target.mode or 0o644)
        except OSError:
            pass
    return Stats(input_bytes=target.size, output_bytes=os.path.getsize(archive),
                 seconds=time.perf_counter() - started, files=1)


def _find_delta_candidates(paths, members, layouts, refs, workers, level):
    """Tensors that are close to one in an earlier member, but not identical.

    Matched by name, dtype and size -- which is exactly what a checkpoint
    series or a fine-tune of a known base produces -- then decided by encoding
    a sample both ways. That costs two encodes per candidate and settles the
    question with the coder that will actually run, so a pair that has drifted
    too far declines on its own rather than on a rule of thumb.

    Sources are always the earliest member holding the tensor, so a delta
    never points at another delta and resolving one stays a single hop.
    Returns ({member index: [(start, end, source output offset)]}, bytes covered).
    """
    taken = {}
    for idx, items in (refs or {}).items():
        taken[idx] = {(s, e) for s, e, _src in items}

    groups: dict = {}
    region_at: list = []
    for idx, layout in enumerate(layouts):
        at = {}
        if layout is not None:
            for r in layout.regions:
                at[r.start] = r
            for name, meta in (layout.tensors or {}).items():
                s, e = meta["offsets"]
                if e - s < DELTA_MIN_TENSOR or (s, e) in taken.get(idx, ()):
                    continue
                groups.setdefault((name, meta.get("dtype", ""), e - s), []) \
                    .append((idx, s, e))
        region_at.append(at)

    work = []
    for group in groups.values():
        if len(group) < 2:
            continue
        group.sort()
        pidx, ps, _pe = group[0]
        for idx, s, e in group[1:]:
            r = region_at[idx].get(s)
            if r is None or r.start != s or r.end != e:
                continue  # only whole, singly-owned regions
            work.append((pidx, ps, idx, s, e, r))
    if not work:
        return {}, 0

    pool = _FdPool(paths, os.O_RDONLY)
    try:
        def decide(item):
            pidx, ps, idx, s, e, r = item
            n = e - s
            take = min(DELTA_SAMPLE, n)
            take -= take % max(r.esize, 1)
            if take <= 0:
                return None
            off = ((n - take) // 2 // max(r.esize, 1)) * max(r.esize, 1)
            a = os.pread(pool.get(pidx), take, ps + off)
            b = os.pread(pool.get(idx), take, s + off)
            if len(a) != take or len(b) != take or a == b:
                return None  # identical samples are dedup's job, not this one
            diff = bytes(kernels.xor_bytes(b, a))

            def coded(buf):
                parts = _codec.encode_chunk(buf, r.esize, level, False,
                                            kind=r.kind, btype=r.btype)[0]
                return sum(len(p) for p in parts)

            if coded(diff) < coded(b) * (1 - DELTA_MIN_GAIN):
                return idx, s, e, members[pidx].dst + ps
            return None

        deltas: dict[int, list] = {}
        covered = 0
        for got in unordered_map(decide, work, workers):
            if got is None:
                continue
            idx, s, e, src = got
            deltas.setdefault(idx, []).append((s, e, src))
            covered += e - s
    finally:
        pool.close()
    for items in deltas.values():
        items.sort()
    return deltas, covered


def _locate_output(members, off: int):
    """Which member an output offset falls in, and where inside its file."""
    starts = [m.dst for m in members]
    i = bisect_right(starts, off) - 1
    if i < 0 or off >= members[i].dst + members[i].size:
        raise ValueError(f"output offset {off} falls in no member")
    return i, off - members[i].dst


def compress(src: str, dst: str, *, level: int = DEFAULT_LEVEL,
             workers: int | None = None, chunk_size: int | None = None,
             checksum: bool = True, dedup: bool = True, delta: bool = True,
             mapped: bool = False, align: bool = False, progress=None) -> Stats:
    """Compress a file or directory into the archive at `dst`.

    `mapped` cuts the archive into small blocks so any byte range can be
    decoded on its own, which is what `MappedArchive` needs and what a
    demand-paging loader needs underneath it. `align` additionally starts
    every payload on a 4 KiB boundary; it costs the padding it writes and only
    repays it where reads bypass the page cache, so it is off by default.
    """
    workers = workers or default_workers()
    if chunk_size is None:
        chunk_size = DEFAULT_BLOCK_SIZE if mapped else DEFAULT_CHUNK_SIZE
    chunk_size = max(chunk_size, 1 << 16)
    if align and not mapped:
        raise ValueError("align only makes sense for a page-mapped archive")
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
    layouts: list = []
    sizes: list[int] = []
    base = 0
    for rel, full in entries:
        size = os.path.getsize(full)
        with open(full, "rb") as fh:
            layout = probe(fh, size)
        members.append(Member(path=rel, size=size, dst=base, kind=layout.kind,
                              mode=os.stat(full).st_mode & 0o777,
                              tensors=layout.tensors))
        paths.append(full)
        layouts.append(layout)
        sizes.append(size)
        base += size
    total_in = base

    refs_by_member: dict[int, list] = {}
    dedup_bytes = 0
    if dedup:
        refs_by_member, dedup_bytes = _find_duplicate_tensors(
            paths, members, layouts, workers)

    # Deltas come after dedup so a tensor that is byte-identical becomes a ref,
    # which costs eight bytes, rather than a difference of all zeros.
    deltas_by_member: dict[int, list] = {}
    delta_bytes = 0
    if delta and len(members) > 1:
        deltas_by_member, delta_bytes = _find_delta_candidates(
            paths, members, layouts, refs_by_member, workers, level)

    plan: list[tuple] = []  # member, start, end, esize, kind, src, btype, ikind
    for idx, layout in enumerate(layouts):
        for task in chunkify(layout, sizes[idx], chunk_size,
                             refs=refs_by_member.get(idx),
                             deltas=deltas_by_member.get(idx)):
            plan.append((idx,) + task)

    manifest = {
        "version": 1,
        "tool": f"lmz {_version()}",
        "level": level,
        "chunk_size": chunk_size,
        "mapped": bool(mapped),
        "aligned": bool(align),
        "entropy": entropy.BACKEND,
        "checksum": "crc32" if checksum else "none",
        "total_size": total_in,
        "members": [m.to_json() for m in members],
    }
    if dedup_bytes:
        manifest["dedup_bytes"] = dedup_bytes
    if delta_bytes:
        manifest["delta_bytes"] = delta_bytes

    pool = _FdPool(paths, os.O_RDONLY)
    done = 0
    stats = Stats(input_bytes=total_in, files=len(members), chunks=len(plan),
                  detail={"dedup_bytes": dedup_bytes,
                          "delta_bytes": delta_bytes})

    def work(task):
        idx, start, end, esize, kind, ref_src, btype, ikind = task
        if kind == KIND_REF:
            # The bytes equal an earlier output range; nothing is read and
            # nothing needs a checksum of its own -- the source chunks carry
            # theirs, and resolution decodes through them.
            return (members[idx].dst + start, end - start,
                    [struct.pack("<Q", ref_src)], CODEC_REF, 1, 0, 0)
        data = os.pread(pool.get(idx), end - start, start)
        if len(data) != end - start:
            raise IOError(f"short read on {members[idx].path} at {start}")
        if kind == KIND_DELTA:
            midx, moff = _locate_output(members, ref_src)
            base = os.pread(pool.get(midx), end - start, moff)
            if len(base) != end - start:
                raise IOError(f"short read on {members[midx].path} at {moff}")
            parts, flags, crc = _codec.encode_delta(
                data, base, esize, level, checksum, kind=ikind, btype=btype,
                src=ref_src)
            return (members[idx].dst + start, end - start, parts, CODEC_DELTA,
                    esize, flags, crc)
        parts, cid, flags, crc = _codec.encode_chunk(data, esize, level, checksum,
                                                     kind=kind, btype=btype)
        return members[idx].dst + start, end - start, parts, cid, esize, flags, crc

    def work_batch(batch):
        return [work(task) for task in batch]

    tmp = dst + ".part"
    try:
        with open(tmp, "wb", buffering=1 << 20) as out:
            hflags = (FLAG_PAGE_MAPPED if mapped else 0) | \
                     (FLAG_ALIGNED if align else 0)
            writer = ArchiveWriter(out, manifest, flags=hflags,
                                   align=PAGE_ALIGN if align else 0)
            # Encoding has the same granularity problem as decoding, and the
            # same two-part answer: a 64 KiB chunk is a few microseconds of
            # interpreter around its native work, so twelve threads spend
            # their time on the GIL rather than on coding.
            nw = _worker_cap([e - s for _i, s, e, *_r in plan], workers)
            for results in ordered_map(
                    work_batch, _batches(plan, lambda t: t[2] - t[1]), nw,
                    lookahead=nw * 2):
                for dst_off, rlen, parts, cid, esize, flags, crc in results:
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

    stats.detail["padding_bytes"] = writer.padding
    stats.output_bytes = os.path.getsize(dst)
    stats.seconds = time.perf_counter() - started
    return stats


# ---------------------------------------------------------------- decompress


def _range_resolver(chunks, payload_of, verify_checksums: bool):
    """A function that materialises the bytes at any earlier output range.

    Source chunks are decoded straight from the archive rather than read back
    from the output, so resolution never depends on another chunk having been
    written first and decompression stays a bag of independent jobs. A source
    may not itself be a ref or a delta, which keeps resolution a single hop
    instead of a chain of unknown depth.
    """
    ordered = sorted(chunks, key=lambda c: c.dst)
    starts = [c.dst for c in ordered]

    def resolve(src: int, rlen: int) -> bytearray:
        out = bytearray(rlen)
        pos, end = src, src + rlen
        i = bisect_right(starts, pos) - 1
        while pos < end:
            c = ordered[i] if 0 <= i < len(ordered) else None
            if c is None or not (c.dst <= pos < c.dst + c.rlen):
                raise CorruptArchive(f"source at byte {pos} is not covered")
            if c.codec in (CODEC_REF, CODEC_DELTA):
                raise CorruptArchive(
                    "a ref or delta chunk points at another one")
            data = _codec.decode_chunk(payload_of(c), c.codec, c.esize, c.flags,
                                       c.rlen, c.crc, verify_checksums)
            a = pos - c.dst
            b = min(end - c.dst, c.rlen)
            out[pos - src:pos - src + (b - a)] = data[a:b]
            pos = c.dst + b
            i += 1
        return out

    return resolve


def _chunk_decoder(chunks, payload_of, verify_checksums: bool):
    """Decode any chunk, resolving ref and delta sources as needed."""
    resolve = _range_resolver(chunks, payload_of, verify_checksums)

    def decode(chunk, payload, out=None):
        if chunk.codec == CODEC_REF:
            if len(payload) != 8:
                raise CorruptArchive("ref chunk payload must be 8 bytes")
            (src,) = struct.unpack("<Q", bytes(payload))
            return resolve(src, chunk.rlen)
        if chunk.codec == CODEC_DELTA:
            src, _inner, _off = _codec.delta_source(payload)
            return _codec.decode_delta(payload, chunk.esize, chunk.flags,
                                       chunk.rlen, resolve(src, chunk.rlen),
                                       chunk.crc, verify_checksums)
        return _codec.decode_chunk(payload, chunk.codec, chunk.esize,
                                   chunk.flags, chunk.rlen, chunk.crc,
                                   verify_checksums, out=out)

    return decode


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

    def payload_of(c):
        payload = os.pread(in_pool.get(0), c.clen, c.off)
        if len(payload) != c.clen:
            raise FormatError("archive is truncated")
        return payload

    decode = _chunk_decoder(chunks, payload_of, verify_checksums)

    def work(batch):
        done_bytes = 0
        for chunk in batch:
            payload = payload_of(chunk)
            data = decode(chunk, payload, out=_scratch(chunk.rlen))
            i = bisect_right(starts, chunk.dst) - 1
            if i < 0 or chunk.dst + chunk.rlen > members[i].dst + members[i].size:
                raise FormatError(f"chunk at {chunk.dst} does not fit any member")
            written = os.pwrite(out_pool.get(i), data, chunk.dst - members[i].dst)
            if written != len(data):
                raise IOError(f"short write to {out_paths[i]}")
            done_bytes += chunk.rlen
        return done_bytes

    try:
        nw = _decode_workers(chunks, workers)
        for rlen in unordered_map(work, _batches(chunks, lambda c: c.rlen), nw,
                                  lookahead=nw * 3):
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


# How much decoded output one pool task should carry. A task's dispatch costs
# roughly what decoding a 64 KiB block costs, so a page-mapped archive -- 16k
# blocks where a default one has 119 -- spent most of decompression feeding
# the pool rather than decoding, and ran 6x slower for it. Grouping restores
# the ratio without changing a byte of what is decoded, and leaves the default
# archive alone: at 8 MiB a chunk already exceeds this on its own.
DECODE_BATCH = 4 << 20


# Chunk size at which decoding stops being Python-bound. A chunk costs about
# 4 us of interpreter around however much native decoding it carries, so a
# 64 KiB block is ~5% Python and a 1 MiB one is a fraction of a percent.
# Measured at twelve threads, best of three, on a 1 GiB BF16 model:
#
#   chunk    threads capped    uncapped
#    64 KiB       1.05 GB/s    0.34 GB/s
#   256 KiB       1.31         1.23
#     1 MiB       1.39         4.29
#
# so the boundary is between 256 KiB and 1 MiB, and it is sharp. A first
# attempt put it at 1 MiB, which capped the row that scales best and cost 3x
# -- single runs had hidden it, because this measurement swings 30% run to run
# and only best-of-three is stable.
THREADS_SCALE_ABOVE = 512 << 10
GIL_DECODE_WORKERS = 2


def _worker_cap(sizes, workers: int) -> int:
    """How many threads work of this granularity is actually worth.

    Capping is the opposite of what a thread pool usually wants, and it is
    what the measurement says: with small chunks the GIL handoff between one
    chunk's native calls and the next costs more than another core brings.
    A free-threaded interpreter has no such point, so the cap lifts there.
    """
    if not sizes or workers <= GIL_DECODE_WORKERS or not gil_enabled():
        return workers
    if sum(sizes) / len(sizes) >= THREADS_SCALE_ABOVE:
        return workers
    return GIL_DECODE_WORKERS


def _decode_workers(chunks, workers: int) -> int:
    return _worker_cap([c.rlen for c in chunks], workers)


def _batches(items, size_of, target: int = DECODE_BATCH):
    """Group consecutive items so each task carries a worthwhile amount."""
    out, cur, acc = [], [], 0
    for it in items:
        cur.append(it)
        acc += size_of(it)
        if acc >= target:
            out.append(cur)
            cur, acc = [], 0
    if cur:
        out.append(cur)
    return out


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

    ordered = sorted(chunks, key=lambda c: c.dst)
    ordered_starts = [c.dst for c in ordered]

    def check_source(src: int, rlen: int, what: str):
        """The whole source range must lie on plain chunks.

        Those sources are each decoded and checksummed by their own turn in
        this same pass, so decoding them again here would only repeat work.
        """
        pos, end = src, src + rlen
        i = bisect_right(ordered_starts, pos) - 1
        while pos < end:
            c = ordered[i] if 0 <= i < len(ordered) else None
            if c is None or not (c.dst <= pos < c.dst + c.rlen):
                raise CorruptArchive(f"{what} at byte {pos} is not covered")
            if c.codec in (CODEC_REF, CODEC_DELTA):
                raise CorruptArchive("a ref or delta chunk points at another one")
            pos = c.dst + min(end - c.dst, c.rlen)
            i += 1

    def payload_of(c):
        payload = os.pread(pool.get(0), c.clen, c.off)
        if len(payload) != c.clen:
            raise FormatError("archive is truncated")
        return payload

    resolve = _range_resolver(chunks, payload_of, True)

    def work(batch):
        done_bytes = 0
        for chunk in batch:
            payload = payload_of(chunk)
            if chunk.codec == CODEC_REF:
                if len(payload) != 8:
                    raise CorruptArchive("ref chunk payload must be 8 bytes")
                (src,) = struct.unpack("<Q", bytes(payload))
                check_source(src, chunk.rlen, "ref target")
            elif chunk.codec == CODEC_DELTA:
                # A delta carries real data, so it is decoded and checksummed
                # in full -- unlike a ref, nothing else in this pass would
                # catch a damaged difference.
                src, _inner, _off = _codec.delta_source(payload)
                check_source(src, chunk.rlen, "delta source")
                _codec.decode_delta(payload, chunk.esize, chunk.flags,
                                    chunk.rlen, resolve(src, chunk.rlen),
                                    chunk.crc, True)
            else:
                _codec.decode_chunk(payload, chunk.codec, chunk.esize,
                                    chunk.flags, chunk.rlen, chunk.crc, True,
                                    out=_scratch(chunk.rlen))
            done_bytes += chunk.rlen
        return done_bytes

    try:
        nw = _decode_workers(chunks, workers)
        for rlen in unordered_map(work, _batches(chunks, lambda c: c.rlen), nw,
                                  lookahead=nw * 3):
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
            name = {0: "stored", 1: "entropy", 2: "split", 3: "bf16-split",
                    4: "ref", 5: "bf16-cond", 6: "q8-block",
                    7: "blk-split", 8: "delta"}.get(c.codec, "?")
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


class MappedArchive:
    """Random access into an archive without expanding it.

    The chunk table is already an index: every block records where it lands in
    the output and where its payload sits in the archive, so reading a byte
    range is a binary search plus a decode of the blocks it touches. What
    decides whether that is cheap is the block size. At the 8 MiB default,
    reading a 200-byte bias decodes 8.39 MB -- 32768 times what was asked for.
    At the 64 KiB blocks `compress(mapped=True)` writes, the same read decodes
    64 KiB in 74 us, which is quicker than the flash read it replaces.

    Decoded blocks are cached, so walking a tensor decodes each block once and
    a second read inside one costs nothing. `decoded_bytes` records what was
    actually expanded, which is the honest way to see the amplification.

    Not thread-safe: give each thread its own instance.
    """

    # Decoding a block is 48 us of work against 1 us of pread, so reads are
    # compute-bound and the kernel releases the GIL for the whole of a chunk.
    # Two is still the measured optimum: 0.90 GB/s on one thread, 1.73 on
    # two, 1.69 on four and 0.85 on eight. Folding a chunk into one native
    # call removed the old inversion -- four threads used to be *slower* than
    # one -- but it did not buy a third and fourth thread, because what little
    # Python is left between decodes turns into GIL contention.
    #
    # Four does pay at sys.setswitchinterval(50us) instead of the 5 ms
    # default: 1.69 -> 2.05 GB/s. That is a process-wide setting and belongs
    # to whoever embeds this, not to a library, so the default stays at two.
    MAX_READ_WORKERS = 2

    # Below this many blocks the pool costs more to feed than it saves.
    PARALLEL_MIN_BLOCKS = 4

    def __init__(self, path: str, *, verify: bool = True,
                 cache_blocks: int = 32, workers: int | None = None):
        self.path = path
        self._fh = open(path, "rb")
        try:
            self._reader = ArchiveReader(self._fh)
        except BaseException:
            self._fh.close()
            raise
        self._chunks = sorted(self._reader.chunks, key=lambda c: c.dst)
        self._starts = [c.dst for c in self._chunks]
        self._decode = _chunk_decoder(self._chunks, self._payload_of, verify)
        self._cache: "OrderedDict[int, bytes]" = OrderedDict()
        self._cap = max(1, cache_blocks)
        self._lock = threading.Lock()
        self._pool = None
        self._workers = (self._auto_workers() if workers is None
                         else max(1, workers))
        self.decoded_bytes = 0
        self.blocks_decoded = 0
        self.cache_hits = 0

    # -- introspection ----------------------------------------------------
    @property
    def members(self):
        return self._reader.members

    @property
    def size(self) -> int:
        """Total size of everything the archive restores to."""
        return sum(m.size for m in self._reader.members)

    @property
    def page_mapped(self) -> bool:
        return self._reader.page_mapped

    @property
    def aligned(self) -> bool:
        return self._reader.aligned

    @property
    def block_bytes(self) -> int:
        """Largest block in the archive: what a one-byte read can cost."""
        return max((c.rlen for c in self._chunks), default=0)

    # -- reading ----------------------------------------------------------
    def _payload_of(self, c):
        payload = os.pread(self._fh.fileno(), c.clen, c.off)
        if len(payload) != c.clen:
            raise FormatError("archive is truncated")
        return payload

    @classmethod
    def _auto_workers(cls) -> int:
        """Threads for reads: two under the GIL, as many as there are without.

        The cap exists only because the Python between native calls is
        serialised. A free-threaded build has no such point, so the limit is
        the machine. Untested here -- this box has no free-threaded
        interpreter -- so the cap is lifted only where the interpreter itself
        says the GIL is gone.
        """
        if not gil_enabled():
            return default_workers()
        return min(default_workers(), cls.MAX_READ_WORKERS)

    def _lookup(self, idx: list) -> list:
        """Cache hits for every index, under one lock rather than one each."""
        with self._lock:
            out = []
            for i in idx:
                got = self._cache.get(i)
                if got is not None:
                    self._cache.move_to_end(i)
                    self.cache_hits += 1
                out.append(got)
            return out

    def _store(self, items) -> None:
        with self._lock:
            for i, data in items:
                self.decoded_bytes += len(data)
                self.blocks_decoded += 1
                self._cache[i] = data
            while len(self._cache) > self._cap:
                self._cache.popitem(last=False)

    def _raw(self, i: int) -> bytes:
        # Decode into a per-thread buffer and copy out for the cache: a fresh
        # 64 KiB bytearray per block has to be allocated and zeroed with the
        # GIL held, which cost 1.69 against 1.40 GB/s across four threads.
        c = self._chunks[i]
        return bytes(self._decode(c, self._payload_of(c), out=_scratch(c.rlen)))

    def _decode_run(self, idx: list) -> dict:
        """Decode a run of blocks, taking the cache lock once at the end.

        Locking per block instead put four threads *behind* two, because a
        bulk read acquires it twice per block and a decode is only ~70 us.
        """
        got = {i: self._raw(i) for i in idx}
        self._store(got.items())
        return got

    def _block(self, i: int) -> bytes:
        got = self._lookup([i])[0]
        if got is None:
            got = self._raw(i)
            self._store(((i, got),))
        return got

    def _blocks(self, idx: list) -> list:
        """Decode every block in `idx`, using the pool when it is worth feeding."""
        got = self._lookup(idx)
        missing = [i for i, d in zip(idx, got) if d is None]
        if len(missing) >= self.PARALLEL_MIN_BLOCKS and self._workers > 1:
            if self._pool is None:
                from concurrent.futures import ThreadPoolExecutor

                self._pool = ThreadPoolExecutor(
                    self._workers, thread_name_prefix="lmz-read")
            # One task per run of blocks, never one per block: handing a
            # future to the pool costs about what decoding a 64 KiB block
            # costs, so per-block dispatch spends the parallelism on its own
            # bookkeeping and lands slower than a plain loop.
            step = -(-len(missing) // self._workers)
            runs = [missing[j:j + step] for j in range(0, len(missing), step)]
            done = {}
            for part in self._pool.map(self._decode_run, runs):
                done.update(part)
        else:
            done = self._decode_run(missing)
        return [d if d is not None else done[i] for i, d in zip(idx, got)]

    def _locate(self, pos: int) -> int:
        i = bisect_right(self._starts, pos) - 1
        if not 0 <= i < len(self._chunks):
            raise CorruptArchive(f"byte {pos} is not covered by any chunk")
        c = self._chunks[i]
        if not (c.dst <= pos < c.dst + c.rlen):
            raise CorruptArchive(f"byte {pos} is not covered by any chunk")
        return i

    def read(self, offset: int, length: int) -> bytes:
        """Bytes [offset, offset+length) of the restored output."""
        if offset < 0 or length < 0:
            raise ValueError("offset and length must not be negative")
        end = min(offset + length, self.size)
        if offset >= end:
            return b""

        first = self._locate(offset)
        c = self._chunks[first]
        # The common case by count -- a small read inside one block -- takes
        # nothing but a slice. Assembling a buffer for it copied every byte
        # twice over for no reason.
        if end <= c.dst + c.rlen:
            a = offset - c.dst
            return bytes(self._block(first)[a:a + (end - offset)])

        idx, pos = [], offset
        i = first
        while pos < end:
            i = self._locate(pos)
            idx.append(i)
            pos = self._chunks[i].dst + self._chunks[i].rlen

        out = bytearray(end - offset)
        for i, data in zip(idx, self._blocks(idx)):
            c = self._chunks[i]
            a = max(offset, c.dst) - c.dst
            b = min(end, c.dst + c.rlen) - c.dst
            out[c.dst + a - offset:c.dst + b - offset] = data[a:b]
        return bytes(out)

    def read_member(self, member: str, offset: int = 0,
                    length: int | None = None) -> bytes:
        for m in self._reader.members:
            if m.path == member:
                n = m.size - offset if length is None else length
                return self.read(m.dst + offset, max(0, min(n, m.size - offset)))
        raise KeyError(f"no member named {member!r}")

    def tensor(self, name: str, member: str | None = None):
        """One tensor as (dtype, shape, raw little-endian bytes)."""
        for m in self._reader.members:
            if member is not None and m.path != member:
                continue
            if m.tensors and name in m.tensors:
                meta = m.tensors[name]
                start, end = meta["offsets"]
                raw = self.read(m.dst + start, end - start)
                return meta.get("dtype", ""), meta.get("shape", []), raw
        where = f" in {member}" if member else ""
        raise KeyError(f"no tensor named {name!r}{where}")

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        self._cache.clear()
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def read_tensor(src: str, name: str, member: str | None = None) -> tuple[str, list, bytes]:
    """Extract one tensor without decompressing the rest of the archive.

    Returns (dtype, shape, raw little-endian bytes).
    """
    with MappedArchive(src, cache_blocks=4) as arc:
        return arc.tensor(name, member)


def _version() -> str:
    from . import __version__

    return __version__


def backends() -> dict:
    """What the tool is actually using, for diagnostics."""
    return {"kernel": kernels.backend(), "entropy": entropy.BACKEND,
            "workers": default_workers()}
