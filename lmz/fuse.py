"""A read-only filesystem, spoken straight to the kernel.

This is what turns lmz from a tool you run into a protocol you read through.
A compressed model stays compressed on disk; the mount presents it as the
ordinary files a runtime expects, and the bytes are decoded on the way out.
llama.cpp, vLLM and transformers need no patching, no plugin and no knowledge
that lmz exists -- they call open() and read() and get the original bytes.

There is no libfuse here and nothing to install. `fusermount3` performs the
privileged mount and hands back a file descriptor over a Unix socket, after
which the kernel's FUSE protocol is a sequence of fixed structs on that
descriptor -- so the whole thing is struct.pack and os.read, which is exactly
what the rest of this package already restricts itself to.

Two facts decide the shape of everything below.

The server may not live in the reader's process. A thread that page-faults on
an mmap of its own mount blocks inside the kernel while still holding the GIL,
so the thread that would answer the fault can never run and the process
deadlocks. Since mmap is how llama.cpp loads a model, `lmz mount` is a
separate process by construction rather than by preference.

Requests are answered concurrently and each answer is one write. The kernel
multiplexes many outstanding requests on the one descriptor and tags each with
a unique id, so N server threads each read, decode and reply independently and
never coordinate. Parallelism therefore comes from concurrent requests, not
from splitting one request, which is why the archives are opened with their
own thread pools disabled.
"""

from __future__ import annotations

import array
import errno
import os
import socket
import struct
import subprocess
import sys
import threading

# ---------------------------------------------------------------- the protocol
#
# Field order and widths are linux/include/uapi/linux/fuse.h. Structs are
# append-only across ABI revisions, so a short reply is a valid older reply and
# a long request is read by its declared length -- which is what lets one
# implementation talk to kernels that disagree about the tail.

IN_HDR = struct.Struct("<IIQQIIIHH")        # len, opcode, unique, nodeid, ...
OUT_HDR = struct.Struct("<IiQ")             # len, error, unique
ATTR = struct.Struct("<QQQQQQIIIIIIIIII")   # fuse_attr
ENTRY_OUT = struct.Struct("<QQQQII")        # + fuse_attr
ATTR_OUT = struct.Struct("<QII")            # + fuse_attr
INIT_OUT = struct.Struct("<IIIIHHIIHH32s")
OPEN_OUT = struct.Struct("<QIi")
READ_IN = struct.Struct("<QQIIQII")
DIRENT = struct.Struct("<QQII")             # + name, padded to 8
KSTATFS = struct.Struct("<QQQQQIIII24s")

assert (IN_HDR.size, OUT_HDR.size, ATTR.size) == (40, 16, 88)
assert (ENTRY_OUT.size + ATTR.size, ATTR_OUT.size + ATTR.size) == (128, 104)
assert (INIT_OUT.size, OPEN_OUT.size, READ_IN.size) == (64, 16, 40)

(LOOKUP, FORGET, GETATTR, SETATTR, READLINK, SYMLINK) = 1, 2, 3, 4, 5, 6
(MKNOD, MKDIR, UNLINK, RMDIR, RENAME, LINK) = 8, 9, 10, 11, 12, 13
(OPEN, READ, WRITE, STATFS, RELEASE) = 14, 15, 16, 17, 18
(FSYNC, SETXATTR, GETXATTR, LISTXATTR, REMOVEXATTR) = 20, 21, 22, 23, 24
(FLUSH, INIT, OPENDIR, READDIR, RELEASEDIR) = 25, 26, 27, 28, 29
(FSYNCDIR, ACCESS, CREATE, INTERRUPT, DESTROY) = 30, 34, 35, 36, 38
(BATCH_FORGET, READDIRPLUS, LSEEK) = 42, 44, 46

S_IFDIR, S_IFREG = 0o040000, 0o100000
DT_DIR, DT_REG = 4, 8

FUSE_MAJOR = 7
# The highest minor whose fuse_init_out this build packs. The kernel takes the
# lower of the two, so naming an older revision than it offers is how a fixed
# 64-byte reply stays correct against a kernel whose own struct has grown.
FUSE_MINOR = 31

FUSE_ASYNC_READ = 1 << 0
FUSE_BIG_WRITES = 1 << 5
FUSE_DO_READDIRPLUS = 1 << 13
FUSE_PARALLEL_DIROPS = 1 << 18
FUSE_MAX_PAGES = 1 << 22

FOPEN_KEEP_CACHE = 1 << 1
FOPEN_CACHE_DIR = 1 << 3

# Reads arrive at most this large, which the kernel grants through MAX_PAGES.
# It is the single most effective tuning knob here: at the 128 KiB default a
# 1 GiB sequential read costs 8192 round trips, and at 1 MiB it costs 1024.
MAX_PAGES = 256
MAX_READ = MAX_PAGES * 4096

# Attributes never change under a mount, so the kernel may trust them for as
# long as it likes and skip the lookup and getattr round trips entirely.
FOREVER = (1 << 31) - 1


class FuseError(OSError):
    """Raised when a mount cannot be established."""


# Threads that read /dev/fuse. Two, and the measurement is emphatic: on a
# 1.74 GiB BF16 model with concurrent readers, two threads serve 670 MiB/s
# while four serve 349 and eight 337. This is the same inversion the decode
# path already documents -- a block is a few microseconds of Python around its
# native decode, so a third thread spends its time acquiring the GIL rather
# than decoding -- and it is worse here because the server threads contend
# with the readers as well as each other. A free-threaded interpreter has no
# such point, so the cap lifts there.
GIL_SERVER_THREADS = 2


def default_threads() -> int:
    from .parallel import default_workers, gil_enabled

    if not gil_enabled():
        return default_workers()
    return min(default_workers(), GIL_SERVER_THREADS)


# ------------------------------------------------------------------- the mount


def _mount(point: str, options: str) -> int:
    """Mount `point` via fusermount3 and return the /dev/fuse descriptor.

    fusermount3 is setuid: it performs the mount syscall, then passes the open
    descriptor back through SCM_RIGHTS on a socket named by _FUSE_COMMFD. That
    handshake is the whole reason no privileges are needed here, and it is the
    only part of libfuse's job this module borrows rather than reimplements.
    """
    if not os.path.isdir(point):
        raise FuseError(errno.ENOTDIR, f"{point} is not a directory")
    parent, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        os.set_inheritable(child.fileno(), True)
        env = dict(os.environ, _FUSE_COMMFD=str(child.fileno()))
        try:
            proc = subprocess.Popen(
                ["fusermount3", "-o", options, "--", point],
                env=env, pass_fds=(child.fileno(),))
        except FileNotFoundError:
            raise FuseError(
                errno.ENOENT,
                "fusermount3 not found -- install fuse3 to use `lmz mount`") \
                from None
        _msg, ancillary, _flags, _addr = parent.recvmsg(
            1, socket.CMSG_SPACE(struct.calcsize("i")))
        rc = proc.wait()
        fds = array.array("i")
        for level, kind, data in ancillary:
            if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                fds.frombytes(data[:len(data) - len(data) % fds.itemsize])
        if not fds:
            raise FuseError(errno.EPERM,
                            f"fusermount3 passed no descriptor (exit {rc})")
        return fds[0]
    finally:
        parent.close()
        child.close()


def unmount(point: str) -> None:
    """Detach a mount. Safe to call when nothing is mounted there."""
    subprocess.run(["fusermount3", "-u", "-z", "--", point],
                   check=False, stderr=subprocess.DEVNULL)


# -------------------------------------------------------------- the node tree


class Node:
    """One entry in the tree. Directories hold children; files hold a source.

    `source` is whatever the filesystem's `read` understands -- here a
    (model, member) pair -- so this class stays ignorant of archives.
    """

    __slots__ = ("ino", "name", "mode", "size", "children", "source", "parent")

    def __init__(self, ino: int, name: str, mode: int, size: int = 0,
                 source=None, parent: int = 1):
        self.ino = ino
        self.name = name
        self.mode = mode
        self.size = size
        self.children: dict[str, Node] | None = {} if mode & S_IFDIR else None
        self.source = source
        self.parent = parent

    @property
    def is_dir(self) -> bool:
        return self.children is not None


class Tree:
    """A static directory tree addressed by inode.

    Every path is materialised at mount time. That is affordable because the
    tree is one entry per file in the store -- thousands at most -- and it
    means lookup and readdir are dict hits rather than anything that can fail
    partway through and leave the kernel holding a stale inode.
    """

    def __init__(self):
        self.root = Node(1, "", S_IFDIR | 0o555)
        self.nodes: dict[int, Node] = {1: self.root}
        self._next = 2

    def add_file(self, path: str, size: int, source) -> Node:
        parts = [p for p in path.split("/") if p and p != "."]
        if not parts or ".." in parts:
            raise ValueError(f"refusing suspicious member path: {path!r}")
        node = self.root
        for part in parts[:-1]:
            child = node.children.get(part)
            if child is None:
                child = self._new(part, S_IFDIR | 0o555, parent=node.ino)
                node.children[part] = child
            elif not child.is_dir:
                raise ValueError(f"{part} is both a file and a directory")
            node = child
        leaf = self._new(parts[-1], S_IFREG | 0o444, size, source, node.ino)
        node.children[parts[-1]] = leaf
        return leaf

    def _new(self, name, mode, size=0, source=None, parent=1) -> Node:
        node = Node(self._next, name, mode, size, source, parent)
        self.nodes[self._next] = node
        self._next += 1
        return node

    @property
    def total_size(self) -> int:
        return sum(n.size for n in self.nodes.values() if not n.is_dir)


# ----------------------------------------------------------------- the server


class FuseServer:
    """Serves one mount until it is unmounted.

    Subclasses supply `read_file`; everything else is derived from the tree.
    """

    def __init__(self, tree: Tree, point: str, *, threads: int | None = None,
                 allow_other: bool = False, read_only: bool = True):
        self.tree = tree
        self.point = os.path.abspath(point)
        self.threads = max(1, threads or default_threads())
        self.fd = -1
        self._stop = threading.Event()
        self._opts = ["nosuid", "nodev", "default_permissions",
                      f"max_read={MAX_READ}"]
        if read_only:
            self._opts.append("ro")
        if allow_other:
            self._opts.append("allow_other")
        self.requests = 0
        self.bytes_served = 0
        self._active = 0
        self._tally = threading.Lock()

    # -- what a subclass provides -----------------------------------------
    def read_file(self, node: Node, offset: int, size: int,
                  active: int = 1) -> bytes:
        """Bytes [offset, offset+size) of `node`.

        `active` is how many reads are in flight across the whole server,
        including this one, which is what tells an implementation whether
        there is spare capacity to speculate with.
        """
        raise NotImplementedError

    # -- lifecycle ---------------------------------------------------------
    def mount(self) -> None:
        self.fd = _mount(self.point, ",".join(self._opts))

    def serve(self) -> None:
        """Run until unmounted. Blocks; returns cleanly on DESTROY or ENODEV."""
        if self.fd < 0:
            self.mount()
        workers = [threading.Thread(target=self._loop, name=f"lmz-fuse-{i}",
                                    daemon=True)
                   for i in range(self.threads)]
        for w in workers:
            w.start()
        try:
            while not self._stop.is_set():
                # A plain join() would swallow SIGINT on the main thread.
                self._stop.wait(0.25)
                if not any(w.is_alive() for w in workers):
                    break
        finally:
            self.close()

    def close(self) -> None:
        self._stop.set()
        if self.fd >= 0:
            try:
                os.close(self.fd)
            except OSError:
                pass
            self.fd = -1

    # -- the loop ----------------------------------------------------------
    def _loop(self) -> None:
        # One buffer per thread, large enough for the biggest request the
        # kernel may send: a full write plus its header. Reads must be taken
        # in one os.read of at least this size or the kernel returns EINVAL.
        bufsize = MAX_READ + 65536
        buf = bytearray(bufsize)
        view = memoryview(buf)
        while not self._stop.is_set():
            try:
                n = os.readv(self.fd, (view,))
            except OSError as exc:
                # ENODEV is the ordinary end of a mount; EINTR and EAGAIN are
                # not. ENOENT means the request was interrupted before we
                # answered it, which is normal under load.
                if exc.errno in (errno.ENODEV, errno.EBADF):
                    self._stop.set()
                    return
                if exc.errno in (errno.EINTR, errno.EAGAIN, errno.ENOENT):
                    continue
                raise
            if n < IN_HDR.size:
                continue
            try:
                if self._dispatch(view[:n]) is False:
                    self._stop.set()
                    return
            except Exception:  # a bug here must not wedge the mount
                import traceback
                traceback.print_exc()

    def _reply(self, unique: int, payload=b"") -> None:
        os.writev(self.fd, (OUT_HDR.pack(OUT_HDR.size + len(payload), 0,
                                         unique), payload))

    def _error(self, unique: int, err: int) -> None:
        os.write(self.fd, OUT_HDR.pack(OUT_HDR.size, -err, unique))

    def _attr(self, node: Node) -> bytes:
        nlink = 2 if node.is_dir else 1
        return ATTR.pack(node.ino, node.size, (node.size + 511) // 512,
                         0, 0, 0, 0, 0, 0, node.mode, nlink,
                         os.getuid(), os.getgid(), 0, 4096, 0)

    def _dispatch(self, req: memoryview):
        ln, op, unique, nodeid, _uid, _gid, _pid, _ext, _pad = \
            IN_HDR.unpack_from(req)
        body = req[IN_HDR.size:ln]

        # No reply is permitted for these, and sending one corrupts the
        # stream for every request that follows.
        if op in (FORGET, BATCH_FORGET, INTERRUPT):
            return
        if op == INIT:
            return self._init(unique, body)
        if op == DESTROY:
            self._reply(unique)
            return False

        with self._tally:
            self.requests += 1

        node = self.tree.nodes.get(nodeid)
        if node is None:
            return self._error(unique, errno.ENOENT)

        if op == LOOKUP:
            name = bytes(body).split(b"\0", 1)[0].decode("utf-8", "replace")
            child = node.children.get(name) if node.is_dir else None
            if child is None:
                return self._error(unique, errno.ENOENT)
            return self._reply(unique, self._entry(child))
        if op == GETATTR:
            return self._reply(unique,
                               ATTR_OUT.pack(FOREVER, 0, 0) + self._attr(node))
        if op == OPEN:
            if node.is_dir:
                return self._error(unique, errno.EISDIR)
            return self._reply(unique, OPEN_OUT.pack(0, FOPEN_KEEP_CACHE, 0))
        if op == OPENDIR:
            if not node.is_dir:
                return self._error(unique, errno.ENOTDIR)
            return self._reply(unique, OPEN_OUT.pack(0, FOPEN_CACHE_DIR, 0))
        if op == READ:
            return self._read(unique, node, body)
        if op in (READDIR, READDIRPLUS):
            return self._readdir(unique, node, body, op == READDIRPLUS)
        if op in (RELEASE, RELEASEDIR, FLUSH, FSYNC, FSYNCDIR, ACCESS):
            return self._reply(unique)
        if op == STATFS:
            total = self.tree.total_size
            return self._reply(unique, KSTATFS.pack(
                total // 4096, 0, 0, len(self.tree.nodes), 0,
                4096, 255, 4096, 0, b""))
        if op == LSEEK:
            # SEEK_DATA/SEEK_HOLE: declining leaves the kernel's own handling,
            # which is right for a filesystem with no holes.
            return self._error(unique, errno.ENOSYS)
        if op in (WRITE, SETATTR, MKDIR, MKNOD, CREATE, UNLINK, RMDIR,
                  RENAME, LINK, SYMLINK, SETXATTR, REMOVEXATTR):
            return self._error(unique, errno.EROFS)
        return self._error(unique, errno.ENOSYS)

    def _init(self, unique: int, body: memoryview) -> None:
        major, minor, readahead, flags = struct.unpack_from("<IIII", body)
        if major != FUSE_MAJOR:
            return self._error(unique, errno.EPROTO)
        want = (FUSE_ASYNC_READ | FUSE_BIG_WRITES | FUSE_PARALLEL_DIROPS
                | FUSE_DO_READDIRPLUS | FUSE_MAX_PAGES)
        self._reply(unique, INIT_OUT.pack(
            FUSE_MAJOR, min(minor, FUSE_MINOR), readahead, flags & want,
            12, 10, MAX_READ, 1, MAX_PAGES, 0, b""))

    def _entry(self, node: Node) -> bytes:
        return (ENTRY_OUT.pack(node.ino, 1, FOREVER, FOREVER, 0, 0)
                + self._attr(node))

    def _read(self, unique: int, node: Node, body: memoryview) -> None:
        if node.is_dir:
            return self._error(unique, errno.EISDIR)
        _fh, offset, size, _rf, _owner, _flags, _pad = READ_IN.unpack_from(body)
        if offset >= node.size:
            return self._reply(unique)
        size = min(size, node.size - offset)
        # How many reads are in flight, counting this one. A server with
        # nothing else to do can afford to work ahead of its reader; one that
        # is already saturated cannot, and measured 17% slower for trying.
        with self._tally:
            self._active += 1
            active = self._active
        try:
            data = self.read_file(node, offset, size, active)
        except Exception:
            import traceback
            traceback.print_exc()
            return self._error(unique, errno.EIO)
        finally:
            with self._tally:
                self._active -= 1
        with self._tally:
            self.bytes_served += len(data)
        return self._reply(unique, data)

    def _readdir(self, unique: int, node: Node, body: memoryview,
                 plus: bool) -> None:
        if not node.is_dir:
            return self._error(unique, errno.ENOTDIR)
        _fh, offset, size, *_rest = READ_IN.unpack_from(body)
        parent = self.tree.nodes.get(node.parent, node)
        entries = [(".", node), ("..", parent)]
        entries += sorted(node.children.items())

        out = bytearray()
        for index, (name, child) in enumerate(entries, start=1):
            if index <= offset:
                continue
            raw = name.encode()
            kind = DT_DIR if child.is_dir else DT_REG
            rec = DIRENT.pack(child.ino, index, len(raw), kind) + raw
            rec += b"\0" * (-len(rec) % 8)
            # "." and ".." must not enter the dentry cache under their own
            # names, so readdirplus sends a zeroed entry for them.
            if plus:
                head = (self._entry(child) if name not in (".", "..")
                        else bytes(ENTRY_OUT.size + ATTR.size))
                rec = head + rec
            if len(out) + len(rec) > size:
                break
            out += rec
        return self._reply(unique, bytes(out))


def available() -> tuple[bool, str]:
    """Whether this machine can mount, and why not when it cannot."""
    if not sys.platform.startswith("linux"):
        return False, "FUSE mounts are Linux-only in this build"
    if not os.path.exists("/dev/fuse"):
        return False, "/dev/fuse is missing (no FUSE support in this kernel)"
    if not os.access("/dev/fuse", os.R_OK | os.W_OK):
        return False, "/dev/fuse is not readable and writable by this user"
    from shutil import which
    if which("fusermount3") is None:
        return False, "fusermount3 is not on PATH (install fuse3)"
    return True, "ok"
