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


class ModelFS:
    """The store, seen as a directory tree of ordinary model files.

    One directory per model, holding exactly the files that were compressed
    into it. A reader walking this tree cannot tell it from the original
    directory: same names, same sizes, same bytes.
    """

    def __init__(self, store: Store, names: list[str] | None = None,
                 *, cache_blocks: int = 64, verify: bool = False):
        self.store = store
        self.cache_blocks = cache_blocks
        self.verify = verify
        self._local = threading.local()
        self._open: list = []
        self._open_lock = threading.Lock()

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

    def archive(self, model: str):
        """This thread's reader for one model, opened on first use.

        Each thread gets its own descriptor and block cache. The alternative,
        one shared reader behind a lock, would serialise exactly the decodes
        the server threads exist to overlap.
        """
        table = getattr(self._local, "table", None)
        if table is None:
            table = self._local.table = {}
        arc = table.get(model)
        if arc is None:
            # workers=1: the concurrency lives in the server's threads, and a
            # pool per reader per thread would oversubscribe the machine many
            # times over.
            arc = api.MappedArchive(
                self.store.archive_path(self.entries[model]),
                verify=self.verify, cache_blocks=self.cache_blocks, workers=1)
            table[model] = arc
            with self._open_lock:
                self._open.append(arc)
        return arc

    def read(self, source, offset: int, size: int) -> bytes:
        model, base = source
        return self.archive(model).read(base + offset, size)

    def stats(self) -> dict:
        with self._open_lock:
            readers = list(self._open)
        return {"models": len(self.entries),
                "files": sum(1 for n in self.tree.nodes.values() if not n.is_dir),
                "readers": len(readers),
                "blocks_decoded": sum(a.blocks_decoded for a in readers),
                "decoded_bytes": sum(a.decoded_bytes for a in readers),
                "cache_hits": sum(a.cache_hits for a in readers)}

    def close(self) -> None:
        with self._open_lock:
            readers, self._open = self._open, []
        for arc in readers:
            try:
                arc.close()
            except OSError:
                pass


def mount(point: str, store: Store | None = None, *, names=None,
          threads: int | None = None, allow_other: bool = False,
          cache_blocks: int = 64, verify: bool = False):
    """Build a server for the store. The caller runs `serve()`."""
    from .fuse import FuseServer

    fs = ModelFS(store or Store(), names, cache_blocks=cache_blocks,
                 verify=verify)

    class StoreServer(FuseServer):
        def read_file(self, node, offset, size):
            return fs.read(node.source, offset, size)

        def close(self):
            super().close()
            fs.close()

    server = StoreServer(fs.tree, point, threads=threads,
                         allow_other=allow_other)
    server.fs = fs
    return server
