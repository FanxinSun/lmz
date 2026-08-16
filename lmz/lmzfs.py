"""A read-write filesystem that compresses what it stores.

Every file written through the mount is compressed on the way down and
decoded on the way back up, so an application sees ordinary files and the
disk holds lmz archives. Unlike the model store, nothing has to be registered
first and nothing is read-only: this is a place to put files.

The codec is chosen per file, by the same planner the command line uses. A
safetensors or GGUF file gets the field-split and block-split coders that are
the reason this project exists; anything else falls through to the generic
entropy coder, which is zstd. That is the honest division of labour -- lmz
beats zstd by eleven points on float weights and ties it everywhere else, so
a general-purpose compressed filesystem should be judged on not being *worse*
off the weights and much better on them.

Layout of the backing directory mirrors the mount, with one suffix per file
telling you how it is stored:

    backing/model.safetensors.lmz    compressed
    backing/photo.jpg.lmr            stored raw, compression declined
    backing/subdir/                  a directory, as itself

The suffix is what keeps the mapping unambiguous -- a file genuinely called
`a.lmz` lands at `a.lmz.lmz` and still reads back under its own name.
Metadata needs no sidecar: mode, owner and times are the backing file's own,
and the logical size is the archive's, which sits at a fixed offset in its
32-byte header and costs one short read to fetch.

Writes are buffered. A file opened for writing is materialised into a scratch
file, written there, and compressed once on the last close. Compressing
incrementally is not possible when the coder needs whole chunks to measure a
histogram, and rewriting a byte in the middle of an archive is not possible
at all, so this is the honest shape rather than a compromise: it suits files
that are written once and read many times, which is what large model files
are.
"""

from __future__ import annotations

import errno
import os
import posixpath
import shutil
import stat as statmod
import threading
import time
from collections import OrderedDict

from . import api
from .format import HEADER, HEADER_SIZE, MAGIC, FormatError
from .fuse import (S_IFDIR, S_IFLNK, S_IFREG, FuseServer, Node, Tree,
                   default_threads)

ARCHIVE = ".lmz"   # compressed
RAW = ".lmr"       # stored as-is, because compressing it did not pay
SCRATCH = ".lmz-scratch"

# Below this, the container's own header, chunk table and manifest cost more
# than any coder can save, so small files are stored raw without trying.
MIN_COMPRESS = 1 << 12

# Compression must beat this much of the original to be worth keeping, which
# is the same bargain a compressing filesystem makes: spending CPU on a JPEG
# to save nothing is worse than not trying.
MIN_GAIN = 0.02

# Blocks a stored file is cut into. The store's reasoning applies unchanged:
# small enough that reading a little decodes a little, large enough that the
# frequency tables are not most of the payload.
BLOCK = 64 << 10


class BackingTree(Tree):
    """Inodes over a real directory, discovered as they are asked for.

    The read-only tree is built once because the archive cannot change. Here
    the backing directory is the truth and may be changed by this filesystem
    at any moment, so nodes are created on lookup and their attributes are
    re-read from disk before they are reported.
    """

    def __init__(self, backing: str):
        super().__init__()
        self.backing = backing
        self.root.source = ""
        # The base class builds a read-only root (0555). Under
        # `default_permissions` the kernel enforces exactly what we report, so
        # a writable filesystem whose root says 0555 refuses every create with
        # EACCES before the request ever reaches this code.
        self.stat_root()
        self.by_path: dict[str, Node] = {"": self.root}
        self._lock = threading.Lock()

    def stat_root(self) -> None:
        try:
            st = os.stat(self.backing)
        except OSError:
            return
        self.root.mode = S_IFDIR | (st.st_mode & 0o7777)
        self.root.mtime = int(st.st_mtime)
        self.root.uid, self.root.gid = st.st_uid, st.st_gid

    # -- paths -------------------------------------------------------------
    def real(self, rel: str) -> str:
        return os.path.join(self.backing, rel) if rel else self.backing

    def archive(self, rel: str) -> str:
        return self.real(rel) + ARCHIVE

    def rawfile(self, rel: str) -> str:
        return self.real(rel) + RAW

    def stored(self, rel: str):
        """(path, is_archive) for however `rel` is stored, or (None, False)."""
        a = self.archive(rel)
        if os.path.lexists(a):
            return a, True
        r = self.rawfile(rel)
        if os.path.lexists(r):
            return r, False
        return None, False

    # -- nodes -------------------------------------------------------------
    def node_for(self, rel: str, parent: Node) -> Node | None:
        """The node for a logical path, created or refreshed from disk."""
        real = self.real(rel)
        name = posixpath.basename(rel)
        if os.path.isdir(real):
            mode, size, target = S_IFDIR, 0, None
        else:
            path, is_arc = self.stored(rel)
            if path is None:
                self.drop(rel)
                return None
            st = os.lstat(path)
            if statmod.S_ISLNK(st.st_mode):
                mode, size = S_IFLNK, st.st_size
                target = os.readlink(path)
            else:
                mode = S_IFREG
                size = archive_size(path) if is_arc else st.st_size
                target = None
        with self._lock:
            node = self.by_path.get(rel)
            if node is None:
                node = self._new(name, mode, size, rel, parent.ino)
                self.by_path[rel] = node
        try:
            st = os.lstat(real if os.path.isdir(real) else
                          (self.stored(rel)[0] or real))
        except OSError:
            return None
        node.mode = (mode | (st.st_mode & 0o7777)) if mode != S_IFLNK else \
            (S_IFLNK | 0o777)
        node.size = size
        node.mtime = int(st.st_mtime)
        node.uid, node.gid = st.st_uid, st.st_gid
        node.target = target
        node.parent = parent.ino
        return node

    def drop(self, rel: str) -> None:
        with self._lock:
            node = self.by_path.pop(rel, None)
            if node is not None:
                self.nodes.pop(node.ino, None)

    def rename_path(self, old: str, new: str) -> None:
        with self._lock:
            moved = [p for p in self.by_path if p == old or p.startswith(old + "/")]
            for p in moved:
                node = self.by_path.pop(p)
                np = new + p[len(old):]
                node.source = np
                node.name = posixpath.basename(np)
                self.by_path[np] = node


def archive_size(path: str) -> int:
    """The logical size an archive restores to, from its header alone.

    Sixteen bytes off the front rather than a parse of the manifest, because
    this runs on every stat of every file.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(HEADER_SIZE)
        if len(head) < HEADER_SIZE:
            return 0
        magic, _v, _f, original, _r1, _r2 = HEADER.unpack(head)
        return original if magic == MAGIC else 0
    except OSError:
        return 0


class Handle:
    """One open file. A writable handle owns a scratch copy of the contents."""

    __slots__ = ("rel", "fd", "path", "writable", "dirty", "refs")

    def __init__(self, rel, fd, path, writable):
        self.rel = rel
        self.fd = fd
        self.path = path
        self.writable = writable
        self.dirty = False
        self.refs = 1

    def close(self) -> None:
        """Drop the scratch descriptor. Idempotent, and safe to call twice."""
        fd, self.fd = self.fd, -1
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


class LmzFS(FuseServer):
    """Compressed storage behind an ordinary filesystem interface."""

    def __init__(self, backing: str, point: str, *, threads=None,
                 allow_other=False, level=api.DEFAULT_LEVEL,
                 block_size: int = BLOCK, cache_files: int = 8,
                 min_gain: float = MIN_GAIN):
        os.makedirs(backing, exist_ok=True)
        self.backing = os.path.abspath(backing)
        self.scratch_dir = os.path.join(self.backing, SCRATCH)
        os.makedirs(self.scratch_dir, exist_ok=True)
        for stale in os.listdir(self.scratch_dir):
            try:
                os.unlink(os.path.join(self.scratch_dir, stale))
            except OSError:
                pass
        tree = BackingTree(self.backing)
        super().__init__(tree, point, threads=threads or default_threads(),
                         allow_other=allow_other, read_only=False)
        self.btree = tree
        self.level = level
        self.block_size = block_size
        self.min_gain = min_gain
        self._handles: dict[int, Handle] = {}
        self._open_writes: dict[str, Handle] = {}
        self._next_fh = 1
        self._lock = threading.RLock()
        self._readers: "OrderedDict[str, api.MappedArchive]" = OrderedDict()
        self._reader_cap = max(1, cache_files)
        self.files_written = 0
        self.bytes_in = 0
        self.bytes_out = 0
        self.stored_raw = 0

    # -- helpers -----------------------------------------------------------
    def _rel(self, node: Node) -> str:
        return node.source or ""

    def _child_rel(self, parent: Node, name: str) -> str:
        if name in ("", ".", "..") or "/" in name:
            raise OSError(errno.EINVAL, "bad name")
        base = self._rel(parent)
        return f"{base}/{name}" if base else name

    def _reader(self, rel: str):
        with self._lock:
            arc = self._readers.get(rel)
            if arc is not None:
                self._readers.move_to_end(rel)
                return arc
        arc = api.MappedArchive(self.btree.archive(rel), verify=False,
                                cache_blocks=128, workers=1)
        with self._lock:
            self._readers[rel] = arc
            while len(self._readers) > self._reader_cap:
                _k, old = self._readers.popitem(last=False)
                try:
                    old.close()
                except OSError:
                    pass
        return arc

    def _evict(self, rel: str) -> None:
        with self._lock:
            arc = self._readers.pop(rel, None)
        if arc is not None:
            try:
                arc.close()
            except OSError:
                pass

    # -- tree hooks --------------------------------------------------------
    def _live_size(self, node: Node, rel: str) -> Node:
        """Report the scratch copy's size while a write is still open.

        Until the commit lands, what is on disk is an empty placeholder. A
        stat that believed it would report zero bytes, and a reader that
        trusted the stat would stop before asking for any -- which is exactly
        how `echo > f && cat f` came back empty even though the data was
        there to be read.
        """
        if node is None:
            return node
        with self._lock:
            handle = self._open_writes.get(rel)
            if handle is not None:
                # Measured under the lock: the descriptor is closed in the same
                # critical section that unregisters the handle, so an fstat
                # outside it can size whatever file inherited the number.
                try:
                    node.size = os.fstat(handle.fd).st_size
                except OSError:
                    pass
        return node

    def resolve(self, node: Node, name: str):
        if not node.is_dir or name in (SCRATCH,):
            return None
        try:
            rel = self._child_rel(node, name)
            return self._live_size(self.btree.node_for(rel, node), rel)
        except OSError:
            return None

    def refresh(self, node: Node) -> None:
        if node.ino == 1:
            self.btree.stat_root()
            return
        parent = self.tree.nodes.get(node.parent, self.tree.root)
        rel = self._rel(node)
        self._live_size(self.btree.node_for(rel, parent), rel)

    def entries(self, node: Node):
        rel = self._rel(node)
        try:
            names = os.listdir(self.btree.real(rel))
        except OSError:
            return []
        out = []
        for entry in sorted(names):
            if entry == SCRATCH:
                continue
            logical = (entry[:-len(ARCHIVE)] if entry.endswith(ARCHIVE)
                       else entry[:-len(RAW)] if entry.endswith(RAW) else entry)
            crel = f"{rel}/{logical}" if rel else logical
            child = self._live_size(self.btree.node_for(crel, node), crel)
            if child is not None:
                out.append((logical, child))
        return out

    # -- reading -----------------------------------------------------------
    def read_file(self, node: Node, offset: int, size: int, active: int = 1):
        rel = self._rel(node)
        with self._lock:
            live = self._open_writes.get(rel)
            if live is not None:                   # being written: read scratch
                # Held across the pread deliberately. fs_release closes this
                # descriptor in the same critical section that unregisters the
                # handle, so reading outside the lock can land on a number that
                # has already been freed -- and handed to the next open().
                return os.pread(live.fd, size, offset)
        path, is_arc = self.btree.stored(rel)
        if path is None:
            raise OSError(errno.ENOENT, rel)
        if not is_arc:
            with open(path, "rb") as fh:
                return os.pread(fh.fileno(), size, offset)
        return self._reader(rel).read(offset, size)

    # -- writing -----------------------------------------------------------
    def _scratch_for(self, rel: str, truncate: bool) -> Handle:
        """A writable scratch copy of `rel`, materialised if it exists."""
        safe = rel.replace("/", "%2f")
        path = os.path.join(self.scratch_dir, f"{safe}.{os.getpid()}")
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o600)
        if not truncate:
            src, is_arc = self.btree.stored(rel)
            if src is not None:
                if is_arc:
                    self._evict(rel)
                    api.decompress(src, path, overwrite=True)
                    fd2 = os.open(path, os.O_RDWR)
                    os.close(fd)
                    fd = fd2
                else:
                    with open(src, "rb") as fh:
                        shutil.copyfileobj(fh, os.fdopen(os.dup(fd), "wb"))
        return Handle(rel, fd, path, True)

    def fs_open(self, node: Node, flags: int) -> int:
        writable = bool(flags & (os.O_WRONLY | os.O_RDWR)) or bool(flags & os.O_TRUNC)
        if not writable:
            return 0
        rel = self._rel(node)
        with self._lock:
            live = self._open_writes.get(rel)
            if live is not None:
                live.refs += 1
                fh = self._next_fh
                self._next_fh += 1
                self._handles[fh] = live
                return fh
        handle = self._scratch_for(rel, bool(flags & os.O_TRUNC))
        with self._lock:
            fh = self._next_fh
            self._next_fh += 1
            self._handles[fh] = handle
            self._open_writes[rel] = handle
        return fh

    def fs_create(self, parent: Node, name: str, mode: int, flags: int):
        rel = self._child_rel(parent, name)
        if self.btree.stored(rel)[0] is not None or os.path.isdir(self.btree.real(rel)):
            raise OSError(errno.EEXIST, name)
        # Claim the name immediately so a concurrent lookup can see it.
        raw = self.btree.rawfile(rel)
        with open(raw, "wb"):
            pass
        os.chmod(raw, mode & 0o7777)
        handle = self._scratch_for(rel, True)
        handle.dirty = True
        with self._lock:
            fh = self._next_fh
            self._next_fh += 1
            self._handles[fh] = handle
            self._open_writes[rel] = handle
        node = self.btree.node_for(rel, parent)
        if node is None:
            raise OSError(errno.EIO, name)
        return node, fh

    def fs_write(self, node: Node, fh: int, offset: int, data) -> int:
        with self._lock:
            handle = self._handles.get(fh)
        if handle is None or not handle.writable:
            raise OSError(errno.EBADF, "not open for writing")
        n = os.pwrite(handle.fd, data, offset)
        handle.dirty = True
        return n

    def fs_release(self, node: Node, fh: int) -> None:
        with self._lock:
            handle = self._handles.pop(fh, None)
            if handle is None or not handle.writable:
                return
            handle.refs -= 1
            if handle.refs > 0:
                return
        # The file stays registered as "being written" for the whole of the
        # commit, so a reader arriving in that window is still served from the
        # scratch copy. close() returns before the kernel delivers RELEASE, so
        # `echo > f && cat f` really does race this, and unregistering first
        # made it read back empty.
        try:
            self._commit(handle)
        finally:
            # Unregister and close together, under the lock: past this point no
            # reader can still be holding the handle, and until it no reader
            # can find a closed descriptor.
            with self._lock:
                self._open_writes.pop(handle.rel, None)
                handle.close()

    def fs_fsync(self, node: Node, fh: int) -> None:
        with self._lock:
            handle = self._handles.get(fh)
        if handle is not None and handle.writable:
            os.fsync(handle.fd)

    # Ordinary files reach the generic coder, which is zstd, and there the
    # level is the whole of the ratio -- a compressing filesystem uses zstd 3
    # and level 1 gave away one to two points against it on text and code. On
    # model weights the level does almost nothing (the README measures level 1
    # as usually the smallest as well as the fastest) and level 3 only costs
    # time, so the file's own structure picks.
    GENERIC_LEVEL = 3

    def _level_for(self, path: str) -> int:
        from .planner import probe

        try:
            with open(path, "rb") as fh:
                layout = probe(fh, os.path.getsize(path))
        except Exception:
            return self.GENERIC_LEVEL
        return self.level if layout.kind != "raw" else self.GENERIC_LEVEL

    def _commit(self, handle: Handle) -> None:
        """Compress the scratch copy into the backing store, or store it raw."""
        rel = handle.rel
        try:
            os.fsync(handle.fd)
            size = os.fstat(handle.fd).st_size
        except OSError:
            size = 0

        arc_path, raw_path = self.btree.archive(rel), self.btree.rawfile(rel)
        # The scratch file was made 0600 for safety; the file's real mode
        # lives on whatever is already stored, and os.replace would otherwise
        # carry the scratch's mode over the top of it.
        mode = None
        for existing in (arc_path, raw_path):
            if os.path.lexists(existing):
                try:
                    mode = statmod.S_IMODE(os.stat(existing).st_mode)
                except OSError:
                    pass
                break
        tmp = handle.path + ".out"
        keep_raw = size < MIN_COMPRESS
        if not keep_raw:
            try:
                stats = api.compress(handle.path, tmp, mapped=True,
                                     chunk_size=self.block_size,
                                     level=self._level_for(handle.path))
                keep_raw = stats.output_bytes > size * (1 - self.min_gain)
            except Exception:
                keep_raw = True
            if keep_raw and os.path.exists(tmp):
                os.unlink(tmp)

        self._evict(rel)
        try:
            if keep_raw:
                os.replace(handle.path, raw_path)
                if os.path.lexists(arc_path):
                    os.unlink(arc_path)
                self.stored_raw += 1
                self.bytes_out += size
            else:
                os.replace(tmp, arc_path)
                if os.path.lexists(raw_path):
                    os.unlink(raw_path)
                self.bytes_out += os.path.getsize(arc_path)
            self.files_written += 1
            self.bytes_in += size
            if mode is not None:
                try:
                    os.chmod(arc_path if not keep_raw else raw_path, mode)
                except OSError:
                    pass
        finally:
            # The descriptor outlives this function on purpose: reads racing the
            # commit are served from it, and an open fd survives the rename that
            # puts the result in place. Closing it here would free the number
            # while `_open_writes` still hands the handle out, and the next
            # open() is then given it -- so the caller closes it instead, in the
            # same critical section that unregisters the handle.
            for leftover in (handle.path, tmp):
                if os.path.lexists(leftover):
                    try:
                        os.unlink(leftover)
                    except OSError:
                        pass

    # -- namespace ---------------------------------------------------------
    def fs_setattr(self, node, valid, size, mode, atime, mtime, uid, gid):
        from .fuse import (FATTR_GID, FATTR_MODE, FATTR_MTIME, FATTR_SIZE,
                           FATTR_UID)

        rel = self._rel(node)
        if valid & FATTR_SIZE:
            with self._lock:
                handle = self._open_writes.get(rel)
                if handle is not None:             # same rule as read_file
                    os.ftruncate(handle.fd, size)
                    handle.dirty = True
            if handle is None:
                # Never registered, so nothing else can reach this one and the
                # close is ours to make as soon as the commit is done.
                handle = self._scratch_for(rel, False)
                handle.dirty = True
                os.ftruncate(handle.fd, size)
                try:
                    self._commit(handle)
                finally:
                    handle.close()
        target = self.btree.stored(rel)[0] or self.btree.real(rel)
        if valid & FATTR_MODE:
            os.chmod(target, mode & 0o7777)
        if valid & (FATTR_UID | FATTR_GID):
            try:
                os.chown(target, uid if valid & FATTR_UID else -1,
                         gid if valid & FATTR_GID else -1)
            except OSError:
                pass  # unprivileged mounts cannot change ownership
        if valid & FATTR_MTIME:
            try:
                os.utime(target, (mtime, mtime))
            except OSError:
                pass

    def fs_unlink(self, parent: Node, name: str) -> None:
        rel = self._child_rel(parent, name)
        path = self.btree.stored(rel)[0]
        if path is None:
            raise OSError(errno.ENOENT, name)
        self._evict(rel)
        os.unlink(path)
        self.btree.drop(rel)

    def fs_mkdir(self, parent: Node, name: str, mode: int):
        rel = self._child_rel(parent, name)
        os.mkdir(self.btree.real(rel), mode & 0o7777)
        node = self.btree.node_for(rel, parent)
        if node is None:
            raise OSError(errno.EIO, name)
        return node

    def fs_rmdir(self, parent: Node, name: str) -> None:
        rel = self._child_rel(parent, name)
        os.rmdir(self.btree.real(rel))
        self.btree.drop(rel)

    def fs_rename(self, parent, name, newparent, newname) -> None:
        old = self._child_rel(parent, name)
        new = self._child_rel(newparent, newname)
        if os.path.isdir(self.btree.real(old)):
            os.rename(self.btree.real(old), self.btree.real(new))
        else:
            src, is_arc = self.btree.stored(old)
            if src is None:
                raise OSError(errno.ENOENT, name)
            self._evict(old)
            self._evict(new)
            for stale in (self.btree.archive(new), self.btree.rawfile(new)):
                if os.path.lexists(stale):
                    os.unlink(stale)
            os.rename(src, self.btree.archive(new) if is_arc
                      else self.btree.rawfile(new))
        self.btree.rename_path(old, new)

    def fs_symlink(self, parent: Node, name: str, target: str):
        rel = self._child_rel(parent, name)
        os.symlink(target, self.btree.rawfile(rel))
        node = self.btree.node_for(rel, parent)
        if node is None:
            raise OSError(errno.EIO, name)
        return node

    def fs_statfs(self):
        st = os.statvfs(self.backing)
        return (st.f_blocks * st.f_frsize // 4096,
                st.f_bfree * st.f_frsize // 4096,
                st.f_bavail * st.f_frsize // 4096, st.f_files, st.f_ffree)

    # -- lifecycle ---------------------------------------------------------
    def stats(self) -> dict:
        saved = (1 - self.bytes_out / self.bytes_in) if self.bytes_in else 0.0
        return {"files_written": self.files_written,
                "bytes_in": self.bytes_in, "bytes_out": self.bytes_out,
                "saved": saved, "stored_raw": self.stored_raw,
                "open_readers": len(self._readers)}

    def close(self) -> None:
        super().close()
        with self._lock:
            handles = list(self._handles.values())
            self._handles.clear()
            self._open_writes.clear()
            readers, self._readers = list(self._readers.values()), OrderedDict()
        for handle in handles:
            try:
                if handle.writable and handle.refs > 0:
                    handle.refs = 0
                    self._commit(handle)
            except Exception:
                pass
            finally:
                handle.close()
        for arc in readers:
            try:
                arc.close()
            except OSError:
                pass


def mount(backing: str, point: str, **kwargs) -> LmzFS:
    """Build the filesystem. The caller runs `serve()`."""
    return LmzFS(backing, point, **kwargs)
