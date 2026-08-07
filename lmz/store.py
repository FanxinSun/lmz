"""The on-device model store.

A place where models live compressed and are read without ever being
expanded. Each model is one page-mapped archive, so the store's cost on disk
is what lmz compressed it to, and the cost of using it is a decode of the
blocks a reader actually touches.

The layout is deliberately dull -- a directory of archives plus a JSON index --
because the archive already carries everything that matters. The index exists
so listing the store does not mean opening and parsing every archive in it,
and it is rebuildable from the archives alone if it is ever lost.

    ~/.lmz/
      index.json
      models/<name>.lmz
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass, field

from . import api
from .format import FormatError

INDEX_VERSION = 1
NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def default_root() -> str:
    """Where the store lives: $LMZ_HOME, else XDG-ish, else ~/.lmz."""
    env = os.environ.get("LMZ_HOME")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return os.path.join(os.path.expanduser("~"), ".lmz")


def normalise_name(text: str) -> str:
    """A store name: a filename, not a path, and never empty.

    Slashes become dashes rather than directories so that a Hugging Face
    style `org/model` names one entry instead of nesting the store.
    """
    text = text.strip().strip("/").replace("/", "-").replace("\\", "-")
    text = NAME_RE.sub("-", text).strip("-.")
    if not text or text in (".", ".."):
        raise ValueError(f"cannot use {text!r} as a model name")
    return text


@dataclass
class Entry:
    """One model in the store."""

    name: str
    archive: str          # path, relative to the store root
    original_size: int    # what it restores to
    stored_size: int      # what it costs on disk
    files: int
    source: str = ""
    added: float = 0.0
    members: list = field(default_factory=list)

    @property
    def saved(self) -> float:
        if not self.original_size:
            return 0.0
        return 1.0 - self.stored_size / self.original_size

    def to_json(self) -> dict:
        return {"name": self.name, "archive": self.archive,
                "original_size": self.original_size,
                "stored_size": self.stored_size, "files": self.files,
                "source": self.source, "added": self.added,
                "members": self.members}

    @staticmethod
    def from_json(d: dict) -> "Entry":
        return Entry(name=d["name"], archive=d["archive"],
                     original_size=d.get("original_size", 0),
                     stored_size=d.get("stored_size", 0),
                     files=d.get("files", 0), source=d.get("source", ""),
                     added=d.get("added", 0.0), members=d.get("members", []))


class Store:
    """Models kept compressed, listed cheaply, and opened for random access."""

    def __init__(self, root: str | None = None):
        self.root = os.path.abspath(root or default_root())
        self.models_dir = os.path.join(self.root, "models")
        self.index_path = os.path.join(self.root, "index.json")
        self._lock = threading.Lock()

    # -- the index ---------------------------------------------------------
    def _load(self) -> dict[str, Entry]:
        try:
            with open(self.index_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            raise FormatError(f"store index is unreadable: {exc}") from None
        if data.get("version") != INDEX_VERSION:
            raise FormatError(
                f"store index is version {data.get('version')}, "
                f"this build writes {INDEX_VERSION}")
        return {name: Entry.from_json(d)
                for name, d in data.get("models", {}).items()}

    def _save(self, entries: dict[str, Entry]) -> None:
        os.makedirs(self.root, exist_ok=True)
        payload = {"version": INDEX_VERSION,
                   "models": {n: e.to_json() for n, e in sorted(entries.items())}}
        # Written beside the target and renamed, so an interrupted save leaves
        # the old index intact rather than a half-written one.
        tmp = self.index_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=1)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.index_path)

    # -- queries -----------------------------------------------------------
    def models(self) -> list[Entry]:
        return sorted(self._load().values(), key=lambda e: e.name)

    def get(self, name: str) -> Entry:
        entries = self._load()
        entry = entries.get(name) or entries.get(normalise_name(name))
        if entry is None:
            known = ", ".join(sorted(entries)) or "the store is empty"
            raise KeyError(f"no model named {name!r} ({known})")
        return entry

    def archive_path(self, entry: Entry) -> str:
        return os.path.join(self.root, entry.archive)

    def totals(self) -> tuple[int, int]:
        """(what the store restores to, what it costs on disk)."""
        entries = self._load().values()
        return (sum(e.original_size for e in entries),
                sum(e.stored_size for e in entries))

    # -- mutation ----------------------------------------------------------
    def add(self, src: str, name: str | None = None, *, level: int | None = None,
            workers: int | None = None, force: bool = False,
            block_size: int | None = None, progress=None) -> Entry:
        """Compress `src` into the store and register it.

        The archive is always page-mapped: a stored model exists to be read in
        place, and an 8 MiB block would make reading a 200-byte bias decode
        8 MiB. That costs about a point of ratio against the default and is
        the whole reason the store is usable without expanding anything.
        """
        if not os.path.exists(src):
            raise FileNotFoundError(src)
        name = normalise_name(name or os.path.basename(os.path.abspath(src)))
        with self._lock:
            entries = self._load()
            if name in entries and not force:
                raise ValueError(
                    f"{name} is already in the store (pass force=True to replace)")
        os.makedirs(self.models_dir, exist_ok=True)
        rel = os.path.join("models", f"{name}.lmz")
        dst = os.path.join(self.root, rel)

        stats = api.compress(src, dst, mapped=True, chunk_size=block_size,
                             level=level if level is not None else api.DEFAULT_LEVEL,
                             workers=workers, progress=progress)
        info = api.info(dst)
        entry = Entry(name=name, archive=rel,
                      original_size=stats.input_bytes,
                      stored_size=stats.output_bytes,
                      files=stats.files,
                      source=os.path.abspath(src), added=time.time(),
                      members=[m["path"] for m in info["members"]])
        with self._lock:
            entries = self._load()
            entries[name] = entry
            self._save(entries)
        return entry

    def remove(self, name: str) -> Entry:
        with self._lock:
            entries = self._load()
            entry = entries.get(name) or entries.get(normalise_name(name))
            if entry is None:
                raise KeyError(f"no model named {name!r}")
            del entries[entry.name]
            self._save(entries)
        try:
            os.unlink(self.archive_path(entry))
        except OSError:
            pass
        return entry

    def rebuild(self) -> list[Entry]:
        """Reconstruct the index from the archives actually present."""
        entries: dict[str, Entry] = {}
        if os.path.isdir(self.models_dir):
            for fn in sorted(os.listdir(self.models_dir)):
                if not fn.endswith(".lmz"):
                    continue
                path = os.path.join(self.models_dir, fn)
                try:
                    info = api.info(path)
                except (FormatError, OSError):
                    continue
                name = fn[:-4]
                entries[name] = Entry(
                    name=name, archive=os.path.join("models", fn),
                    original_size=info["original_size"],
                    stored_size=info["archive_size"],
                    files=len(info["members"]),
                    members=[m["path"] for m in info["members"]])
        with self._lock:
            self._save(entries)
        return sorted(entries.values(), key=lambda e: e.name)

    # -- reading -----------------------------------------------------------
    def open(self, name: str, **kwargs):
        """A MappedArchive over one model, for random access without expanding."""
        return api.MappedArchive(self.archive_path(self.get(name)), **kwargs)

    def extract(self, name: str, dst: str, *, overwrite: bool = False,
                workers: int | None = None):
        """Write a stored model back out as ordinary files."""
        entry = self.get(name)
        return api.decompress(self.archive_path(entry), dst, workers=workers,
                              overwrite=overwrite)


# ------------------------------------------------------------------ the mount


# Read ahead only while this many reads or fewer are in flight. Two, not one,
# because the kernel keeps a second request outstanding even for a single
# sequential reader, so gating on one turns the prefetch off exactly when it
# is wanted. Measured on a free-threaded server, MiB/s by reader count:
#
#   gate    1 reader    2      4     16
#      1         773   1400   2368   2763
#      2        1506   1441   2350   2888
#      6        1562   1207   1906   2933
#
# Past two it keeps buying a little for the lone reader and starts taking it
# from everyone else, which is the wrong trade: the lone reader is the case
# that was starved, not the case that matters most.
RA_MAX_ACTIVE = 2


def default_readahead() -> int:
    """Threads that decode ahead of a sequential reader, or 0 for none.

    Set by measurement, and the two interpreters want opposite things. Under
    the GIL there is nowhere for the work to go: a prefetch thread can only
    take the interpreter lock away from the thread whose reply the reader is
    waiting on, measured at 610 MiB/s down to 471, so the answer is not to
    prefetch at all. Without the GIL it is real parallelism on idle cores and
    the same reader goes 698 to 1525.
    """
    from .parallel import default_workers, gil_enabled

    if gil_enabled():
        return 0
    return max(2, default_workers() - 2)


class Readahead:
    """Decode ahead of a reader that is walking a file forwards.

    The kernel will not ask for the next megabyte until this one has been
    answered, so a single sequential reader spends the whole of every decode
    waiting and no amount of server threads helps -- measured at 519 MiB/s
    against 2680 for sixteen concurrent readers on the same mount. The
    request stream is the bottleneck, not the decoder.

    So the reader's next request is predicted rather than waited for. A
    stream is recognised when a read begins where the last one ended, and the
    window beyond it is decoded into the shared block cache while the current
    reply is still being written. The prediction only ever costs work: a
    wrong guess decodes blocks nobody reads, and a right one turns the next
    request into a cache hit.
    """

    # Far enough ahead to cover the decode it is racing, short enough that a
    # wrong guess is cheap, and cut into slices rather than handed to one
    # thread. A single task decoding the whole window is serial, so it
    # finishes a window behind the reader it is supposed to be ahead of and
    # the prefetch never lands -- which is what a first attempt did, at 1.6x
    # slower than no prefetch at all.
    WINDOW = 4 << 20
    SLICE = 512 << 10

    def __init__(self, workers: int, window: int = WINDOW, slice: int = SLICE):
        from concurrent.futures import ThreadPoolExecutor

        self.window = window
        self.slice = slice
        self.workers = workers
        self._pool = ThreadPoolExecutor(workers, thread_name_prefix="lmz-ra")
        self._streams: dict = {}
        self._inflight: set = set()
        self._lock = threading.Lock()
        self.issued = 0
        self.skipped = 0

    def note(self, arc, key, offset: int, size: int) -> None:
        """Record a read, and decode ahead of it when it continues a stream.

        Each stream carries a frontier -- how far ahead it has already been
        asked to decode -- and only the gap between that and the window is
        submitted. Without it every request re-queues the whole window, since
        a slice stops being "in flight" the moment it finishes: a 1.74 GiB
        read issued 114157 prefetches for the 3565 slices it contains, and
        the pool spent its time rediscovering that the work was already done.
        """
        nxt = offset + size
        base = key[1]
        with self._lock:
            expected, frontier = self._streams.get(key, (None, 0))
            if expected is None or offset != expected:
                # A seek, or the first read: start a stream, predict nothing.
                self._streams[key] = (nxt, 0)
                return
            # Both ends are slice-aligned, and the frontier only ever moves
            # forwards. Aligning without that floor lets `start` land behind
            # the frontier and re-queue slices already asked for, which shows
            # up only when the pool happened to finish one first.
            start = max(frontier, nxt - nxt % self.slice)
            stop = nxt + self.window
            self._streams[key] = (nxt, frontier)
            if start >= stop:
                return  # the window is already covered
            if len(self._inflight) >= 4 * self.workers:
                self.skipped += 1
                return
            slices = range(start, stop, self.slice)
            tokens = []
            for at in slices:
                token = (key, at)
                if token not in self._inflight:
                    self._inflight.add(token)
                    tokens.append((token, base + at))
            self._streams[key] = (nxt, max(frontier, slices[-1] + self.slice))
            self.issued += len(tokens)
        for token, at in tokens:
            try:
                self._pool.submit(self._run, arc, token, at)
            except RuntimeError:  # pool already shut down
                with self._lock:
                    self._inflight.discard(token)

    def _run(self, arc, token, at: int) -> None:
        try:
            arc.prefetch(at, self.slice)
        except Exception:
            pass  # a prefetch is a guess; failing one must not fail a read
        finally:
            with self._lock:
                self._inflight.discard(token)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)


class ModelFS:
    """The store, seen as a directory tree of ordinary model files.

    One directory per model, holding exactly the files that were compressed
    into it. A reader walking this tree cannot tell it from the original
    directory: same names, same sizes, same bytes.
    """

    def __init__(self, store: Store, names: list[str] | None = None,
                 *, cache_blocks: int = 512, verify: bool = False,
                 readahead: int | None = None):
        self.store = store
        self.cache_blocks = cache_blocks
        self.verify = verify
        self._archives: dict = {}
        self._open_lock = threading.Lock()
        self._retired = dict.fromkeys(self.COUNTERS, 0)

        from .fuse import Tree

        self.tree = Tree()
        self.entries: dict[str, Entry] = {}
        wanted = set(names) if names else None
        for entry in store.models():
            if wanted is not None and entry.name not in wanted:
                continue
            self.entries[entry.name] = entry
            with api.MappedArchive(store.archive_path(entry),
                                   verify=False, cache_blocks=1) as arc:
                for member in arc.members:
                    self.tree.add_file(f"{entry.name}/{member.path}",
                                       member.size, (entry.name, member.dst))
        if wanted:
            missing = wanted - set(self.entries)
            if missing:
                raise KeyError(f"not in the store: {', '.join(sorted(missing))}")

        if readahead is None:
            readahead = default_readahead()
        self.readahead = Readahead(readahead) if readahead > 0 else None

    def archive(self, model: str):
        """The reader for one model, shared by every thread that wants it.

        Sharing is what makes reading ahead worth anything: a block decoded
        by a prefetch thread has to land somewhere the serving thread will
        look. It is also correct, which per-thread readers hid rather than
        provided -- `read` keeps no state between calls, positional reads need
        no file offset, decode scratch is thread-local and the block cache
        takes its own lock. What per-thread readers really bought was N copies
        of the same cache, so a read straddling two blocks could decode each
        of them twice over.
        """
        arc = self._archives.get(model)
        if arc is not None:
            return arc
        with self._open_lock:
            arc = self._archives.get(model)
            if arc is None:
                # workers=1: the concurrency is the server's threads and the
                # readahead pool, and a third pool inside each reader would
                # oversubscribe the machine several times over.
                arc = api.MappedArchive(
                    self.store.archive_path(self.entries[model]),
                    verify=self.verify, cache_blocks=self.cache_blocks,
                    workers=1)
                self._archives[model] = arc
        return arc

    def read(self, source, offset: int, size: int, active: int = 1) -> bytes:
        """Serve a byte range, reading ahead only if there is capacity spare.

        `active` is the server's in-flight read count. Speculating pays for a
        lone sequential reader, which is otherwise stalled on every decode --
        698 to 1525 MiB/s -- and costs up to 17% once several readers are
        already keeping the decoder busy, because then the prefetch threads
        are competing with real work rather than filling a gap.
        """
        model, base = source
        arc = self.archive(model)
        data = arc.read(base + offset, size)
        if self.readahead is not None and active <= RA_MAX_ACTIVE:
            self.readahead.note(arc, source, offset, size)
        return data

    COUNTERS = ("blocks_decoded", "decoded_bytes", "cache_hits")

    def stats(self) -> dict:
        """What the readers have done, whether or not they are still open.

        Closing folds a reader's counters into a running total first, because
        the server closes itself on the way out of `serve()` and a caller
        asking afterwards -- which is the only time anyone asks -- would
        otherwise be told the mount did nothing at all.
        """
        with self._open_lock:
            readers = list(self._archives.values())
            out = {name: self._retired[name]
                   + sum(getattr(a, name) for a in readers)
                   for name in self.COUNTERS}
        out.update(models=len(self.entries), readers=len(readers),
                   files=sum(1 for n in self.tree.nodes.values()
                             if not n.is_dir))
        if self.readahead is not None:
            out["readahead_issued"] = self.readahead.issued
            out["readahead_skipped"] = self.readahead.skipped
        return out

    def close(self) -> None:
        if self.readahead is not None:
            self.readahead.close()
        with self._open_lock:
            readers, self._archives = list(self._archives.values()), {}
            for arc in readers:
                for name in self.COUNTERS:
                    self._retired[name] += getattr(arc, name)
        for arc in readers:
            try:
                arc.close()
            except OSError:
                pass


def mount(point: str, store: Store | None = None, *, names=None,
          threads: int | None = None, allow_other: bool = False,
          cache_blocks: int = 512, verify: bool = False,
          readahead: int | None = None):
    """Build a server for the store. The caller runs `serve()`."""
    from .fuse import FuseServer

    fs = ModelFS(store or Store(), names, cache_blocks=cache_blocks,
                 verify=verify, readahead=readahead)

    class StoreServer(FuseServer):
        def read_file(self, node, offset, size, active=1):
            return fs.read(node.source, offset, size, active)

        def close(self):
            super().close()
            fs.close()

    server = StoreServer(fs.tree, point, threads=threads,
                         allow_other=allow_other)
    server.fs = fs
    return server
