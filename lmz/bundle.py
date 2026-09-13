"""Complete, lossless artifact bundles built on the existing LMZ container.

The bundle layer deliberately lives above the tensor planner.  The archive
continues to be the existing byte-addressed container; a ``bundle`` object in
its tail manifest describes the files and the consumer contract.  This keeps
old archives and readers compatible while giving a delivery consumer one
machine-readable inventory.

No code in a bundle is imported or executed.  Graph parsing here is structural
only, and ONNX external data is treated as a bounded dependency on another
bundle entry.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import io
import json
import os
import posixpath
import re
import shutil
import stat
import tempfile
from dataclasses import dataclass
from typing import Iterable

from . import api
from . import planner
from .format import (CODEC_NAMES, FOOTER,
                     FormatError, ArchiveReader, ArchiveWriter)


BUNDLE_MANIFEST_KEY = "bundle"
BUNDLE_SCHEMA_MAJOR = 1
BUNDLE_SCHEMA_MINOR = 0
BUNDLE_SCHEMA = {"major": BUNDLE_SCHEMA_MAJOR, "minor": BUNDLE_SCHEMA_MINOR}
ENTRY_ROLES = frozenset({
    "graph", "weights", "config", "preprocess", "vocabulary",
    "calibration", "opaque",
})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DRIVE = re.compile(r"^[A-Za-z]:")
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_O_DIRECTORY = getattr(os, "O_DIRECTORY", 0)
_RENAME_NOREPLACE = 1


class BundleError(FormatError):
    """A structured bundle validation or materialization error."""

    def __init__(self, message: str, *, code: str = "invalid_bundle"):
        super().__init__(message)
        self.code = code

    def as_dict(self) -> dict:
        return {"code": self.code, "message": str(self)}


class BundleSourceChanged(BundleError):
    """A source file changed while a bundle snapshot was being collected."""

    def __init__(self, message: str):
        super().__init__(message, code="source_changed")


@dataclass(frozen=True)
class _SourceFile:
    path: str
    staged_path: str
    size: int
    sha256: str
    mode: int


def _copy_json(value):
    """Copy only JSON-shaped input, producing deterministic errors."""
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError) as exc:
        raise BundleError(f"manifest is not JSON-shaped: {exc}",
                          code="manifest_type") from exc


def canonical_manifest_bytes(manifest: dict) -> bytes:
    """Return the deterministic UTF-8 representation used for identity."""
    if not isinstance(manifest, dict):
        raise BundleError("bundle manifest must be an object",
                          code="manifest_type")
    return json.dumps(manifest, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def manifest_sha256(manifest: dict) -> str:
    """Digest canonical manifest bytes; this is equality evidence, not trust."""
    return hashlib.sha256(canonical_manifest_bytes(manifest)).hexdigest()


def _nonnegative_int(value, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BundleError(f"{what} must be a non-negative integer",
                          code="invalid_integer")
    return value


def _safe_rel_path(path: str, *, what: str = "path") -> str:
    """Accept only an already-canonical, portable relative POSIX path."""
    if not isinstance(path, str) or not path:
        raise BundleError(f"{what} must be a non-empty string",
                          code="unsafe_path")
    if "\x00" in path:
        raise BundleError(f"{what} contains NUL", code="unsafe_path")
    if "\\" in path or path.startswith("/") or path.startswith("//"):
        raise BundleError(f"{what} is not a portable relative path: {path!r}",
                          code="unsafe_path")
    if _DRIVE.match(path):
        raise BundleError(f"{what} has a drive-letter path: {path!r}",
                          code="unsafe_path")
    parts = path.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise BundleError(f"{what} is not canonical: {path!r}",
                          code="unsafe_path")
    if posixpath.normpath(path) != path:
        raise BundleError(f"{what} is not canonical: {path!r}",
                          code="unsafe_path")
    return path


def _entry_id(path: str) -> str:
    return "entry-" + hashlib.sha256(path.encode("utf-8")).hexdigest()[:32]


def _default_role(path: str) -> str:
    lower = path.lower()
    if lower.endswith((".onnx", ".tflite", ".mlir", ".pb")):
        return "graph"
    if lower.endswith((".safetensors", ".gguf", ".pt", ".pth", ".bin",
                       ".weights")):
        return "weights"
    name = posixpath.basename(lower)
    if "vocab" in name or "label" in name or name.endswith(".labels"):
        return "vocabulary"
    if "calib" in name or "scale" in name:
        return "calibration"
    if "preprocess" in name or "normaliz" in name:
        return "preprocess"
    if lower.endswith((".json", ".yaml", ".yml", ".toml")):
        return "config"
    return "opaque"


def _default_bundle(source_root: str) -> dict:
    name = posixpath.basename(os.path.abspath(source_root).rstrip(os.sep))
    return {
        "schema": dict(BUNDLE_SCHEMA),
        "id": name or "bundle",
        "version": "1",
        "source": {"uri": None, "revision": "unknown"},
        "license": {"classification": "unknown",
                     "redistribution": "unknown"},
        "entry_point": None,
        "consumer": {"engine": "unspecified", "backend": "unspecified",
                      "version_range": "*", "required_operators": [],
                      "extensions": [], "export_provenance": "unknown"},
        "io": {"inputs": [], "outputs": []},
        "preprocess": {"identity": "unspecified", "parameters": {}},
        "temporal_state": {
            "inputs": [], "outputs": [], "initialization": "unspecified",
            "reset": "unspecified", "cadence": "unspecified",
            "discontinuity": "unspecified", "max_state_bytes": 0,
        },
        "resources": {
            "decode_workspace": {"bytes": 0, "measurement": "estimated",
                                  "method": "not measured"},
            "materialized_bytes": 0,
        },
        "evaluation": {
            "input_identity": None, "expected_output_identity": None,
            "metric": None, "tolerance": None,
        },
        "extensions": {},
        "entries": [],
    }


def _merge_manifest(source_root: str, supplied: dict | None) -> dict:
    base = _default_bundle(source_root)
    if supplied is not None:
        incoming = _copy_json(supplied)
        if not isinstance(incoming, dict):
            raise BundleError("bundle manifest must be an object",
                              code="manifest_type")
        for key, value in incoming.items():
            base[key] = value
    return base


def _walk_regular_files(root: str) -> Iterable[tuple[str, str, os.stat_result]]:
    root_stat = os.lstat(root)
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise BundleError("bundle source root must be a real directory",
                          code="unsafe_source")

    def visit(directory: str, relative: str):
        try:
            entries = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise BundleError(f"cannot inspect source directory {directory}: {exc}",
                              code="source_read") from exc
        for item in entries:
            rel = item.name if not relative else relative + "/" + item.name
            st = item.stat(follow_symlinks=False)
            if stat.S_ISLNK(st.st_mode):
                raise BundleError(f"source symlink is not allowed: {rel}",
                                  code="unsafe_source")
            if stat.S_ISDIR(st.st_mode):
                yield from visit(item.path, rel)
            elif stat.S_ISREG(st.st_mode):
                yield rel, item.path, st
            else:
                raise BundleError(f"source entry is not a regular file: {rel}",
                                  code="unsafe_source")

    yield from visit(root, "")


def _snapshot_file(source: str, destination: str, before: os.stat_result,
                   mode: int) -> tuple[int, str]:
    flags = os.O_RDONLY | _O_NOFOLLOW
    try:
        fd = os.open(source, flags)
    except OSError as exc:
        raise BundleError(f"cannot open source {source}: {exc}",
                          code="source_open") from exc
    try:
        first = os.fstat(fd)
        if (first.st_dev, first.st_ino, first.st_size) != \
                (before.st_dev, before.st_ino, before.st_size):
            raise BundleSourceChanged(f"source identity changed before read: {source}")
        h = hashlib.sha256()
        total = 0
        parent = os.path.dirname(destination)
        os.makedirs(parent, exist_ok=True)
        out_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                          mode or 0o644)
        try:
            while True:
                block = os.read(fd, 1 << 20)
                if not block:
                    break
                view = memoryview(block)
                while view:
                    written = os.write(out_fd, view)
                    if written <= 0:
                        raise OSError("short write while staging source")
                    view = view[written:]
                h.update(block)
                total += len(block)
            os.fsync(out_fd)
        finally:
            os.close(out_fd)
        after = os.fstat(fd)
        if ((after.st_dev, after.st_ino, after.st_size) !=
                (first.st_dev, first.st_ino, first.st_size) or
                after.st_mtime_ns != first.st_mtime_ns or
                after.st_ctime_ns != first.st_ctime_ns):
            raise BundleSourceChanged(f"source changed during read: {source}")
        if total != after.st_size:
            raise BundleSourceChanged(f"source size changed during read: {source}")
        return total, h.hexdigest()
    except BundleError:
        raise
    except OSError as exc:
        raise BundleError(f"cannot snapshot source {source}: {exc}",
                          code="source_read") from exc
    finally:
        os.close(fd)


def _snapshot_source(root: str, stage: str) -> dict[str, _SourceFile]:
    files: dict[str, _SourceFile] = {}
    for rel, source, st in _walk_regular_files(root):
        destination = os.path.join(stage, *rel.split("/"))
        size, sha = _snapshot_file(source, destination, st, st.st_mode & 0o777)
        files[rel] = _SourceFile(rel, destination, size, sha,
                                 st.st_mode & 0o777)
    if not files:
        raise BundleError("bundle source contains no regular files",
                          code="empty_source")
    return files


def _normalise_dependency(dep, *, entries_by_path: dict,
                          entries_by_id: dict, owner: str) -> dict:
    if isinstance(dep, str):
        target = dep
        item = {"path": target}
    elif isinstance(dep, dict):
        item = dict(dep)
        target = item.get("path") or item.get("id")
        if not isinstance(target, str):
            raise BundleError(f"dependency of {owner} lacks path or id",
                              code="missing_dependency")
    else:
        raise BundleError(f"dependency of {owner} must be a string or object",
                          code="invalid_dependency")
    by_path = entries_by_path.get(target)
    by_id = entries_by_id.get(target)
    if by_path is None and by_id is None:
        raise BundleError(f"dependency of {owner} is missing: {target!r}",
                          code="missing_dependency")
    target_entry = by_path or by_id
    if "path" in item and item["path"] != target_entry["path"]:
        raise BundleError(f"dependency of {owner} has conflicting path/id",
                          code="ambiguous_dependency")
    if "id" in item and item["id"] != target_entry["id"]:
        raise BundleError(f"dependency of {owner} has conflicting path/id",
                          code="ambiguous_dependency")
    out = {"id": target_entry["id"], "path": target_entry["path"]}
    if "offset" in item:
        out["offset"] = _nonnegative_int(item["offset"],
                                          f"dependency offset of {owner}")
    if "length" in item:
        out["length"] = _nonnegative_int(item["length"],
                                          f"dependency length of {owner}")
    return out


def _normalise_entries(bundle: dict, *, paths: Iterable[str] | None = None) -> list[dict]:
    raw = bundle.get("entries")
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise BundleError("bundle entries must be a list", code="manifest_type")
    seen_paths: set[str] = set()
    seen_ids: set[str] = set()
    entries: list[dict] = []
    for raw_entry in raw:
        if not isinstance(raw_entry, dict):
            raise BundleError("each bundle entry must be an object",
                              code="manifest_type")
        entry = dict(raw_entry)
        path = _safe_rel_path(entry.get("path"), what="entry path")
        role = entry.get("role")
        if role not in ENTRY_ROLES:
            raise BundleError(f"entry {path} has invalid role {role!r}",
                              code="invalid_role")
        ident = entry.get("id") or _entry_id(path)
        if not isinstance(ident, str) or not ident or "\x00" in ident:
            raise BundleError(f"entry {path} has invalid identity",
                              code="invalid_identity")
        if path in seen_paths:
            raise BundleError(f"duplicate entry path: {path}",
                              code="duplicate_path")
        if ident in seen_ids:
            raise BundleError(f"duplicate entry identity: {ident}",
                              code="duplicate_identity")
        seen_paths.add(path)
        seen_ids.add(ident)
        entry["path"] = path
        entry["id"] = ident
        entry["role"] = role
        constraints = entry.get("consumer")
        if constraints is not None:
            if not isinstance(constraints, dict):
                raise BundleError(f"entry {path} consumer constraints must be an object",
                                  code="consumer")
            for key in ("engine", "backend", "version_range"):
                if not isinstance(constraints.get(key), str):
                    raise BundleError(
                        f"entry {path} consumer constraint {key} must be a string",
                        code="consumer")
            for key in ("required_operators", "extensions"):
                if key in constraints and (
                        not isinstance(constraints[key], list) or
                        not all(isinstance(x, str) for x in constraints[key])):
                    raise BundleError(
                        f"entry {path} consumer constraint {key} must be a string list",
                        code="consumer")
        entries.append(entry)

    if paths is not None:
        source_paths = set(paths)
        declared_paths = {entry["path"] for entry in entries}
        missing = sorted(source_paths - declared_paths)
        for path in missing:
            entries.append({"id": _entry_id(path), "path": path,
                            "role": _default_role(path)})
            seen_paths.add(path)
            seen_ids.add(entries[-1]["id"])
        absent = sorted(declared_paths - source_paths)
        if absent:
            raise BundleError(f"manifest names absent source files: {absent}",
                              code="missing_source")

    entries.sort(key=lambda item: item["path"])
    paths_sorted = [entry["path"] for entry in entries]
    for i, path in enumerate(paths_sorted):
        prefix = path + "/"
        if any(other.startswith(prefix) for other in paths_sorted[i + 1:]):
            raise BundleError(f"file/directory entry conflict at {path}",
                              code="path_conflict")
    return entries


def _normalise_bundle(bundle: dict, *, paths: Iterable[str] | None = None,
                      require_artifact_fields: bool = True) -> dict:
    out = _copy_json(bundle)
    schema = out.get("schema", dict(BUNDLE_SCHEMA))
    if not isinstance(schema, dict):
        raise BundleError("bundle schema must be an object", code="schema")
    major = schema.get("major")
    minor = schema.get("minor", 0)
    if isinstance(major, bool) or not isinstance(major, int):
        raise BundleError("bundle schema major must be an integer", code="schema")
    if major != BUNDLE_SCHEMA_MAJOR:
        raise BundleError(f"unsupported bundle schema major: {major}",
                          code="unsupported_major")
    if isinstance(minor, bool) or not isinstance(minor, int) or minor < 0:
        raise BundleError("bundle schema minor must be a non-negative integer",
                          code="schema")
    out["schema"] = {"major": major, "minor": minor}
    for key in ("id", "version"):
        if not isinstance(out.get(key), str) or not out[key]:
            raise BundleError(f"bundle {key} must be a non-empty string",
                              code="manifest_type")

    source = out.get("source")
    if not isinstance(source, dict) or not isinstance(source.get("revision"), str):
        raise BundleError("bundle source must declare a revision",
                          code="provenance")
    license_info = out.get("license")
    if (not isinstance(license_info, dict) or
            not isinstance(license_info.get("classification"), str) or
            not isinstance(license_info.get("redistribution"), str)):
        raise BundleError("bundle license must declare classification and redistribution",
                          code="license")

    entries = _normalise_entries(out, paths=paths)
    by_path = {entry["path"]: entry for entry in entries}
    by_id = {entry["id"]: entry for entry in entries}
    for entry in entries:
        if require_artifact_fields:
            size = entry.get("decoded_bytes", entry.get("size"))
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise BundleError(f"entry {entry['path']} lacks decoded byte length",
                                  code="entry_size")
            digest = entry.get("sha256")
            if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
                raise BundleError(f"entry {entry['path']} lacks sha256",
                                  code="entry_digest")
            entry["decoded_bytes"] = size
            entry["size"] = size
        deps = entry.get("dependencies", [])
        if not isinstance(deps, list):
            raise BundleError(f"entry {entry['path']} dependencies must be a list",
                              code="invalid_dependency")
        normal = []
        seen = set()
        for dep in deps:
            item = _normalise_dependency(dep, entries_by_path=by_path,
                                         entries_by_id=by_id,
                                         owner=entry["path"])
            key = (item["id"], item.get("offset"), item.get("length"))
            if key in seen:
                raise BundleError(f"duplicate dependency of {entry['path']}",
                                  code="duplicate_dependency")
            seen.add(key)
            target = by_id[item["id"]]
            offset = item.get("offset", 0)
            target_size = target.get("decoded_bytes", target.get("size"))
            if target_size is not None and offset > target_size:
                raise BundleError(f"dependency offset past {target['path']}",
                                  code="invalid_range")
            if "length" in item and target_size is not None:
                end = offset + item["length"]
                if end < offset or end > target_size:
                    raise BundleError(f"dependency range past {target['path']}",
                                      code="invalid_range")
            normal.append(item)
        entry["dependencies"] = normal

    # Every dependency edge must be acyclic, including ordinary config edges.
    state: dict[str, int] = {}

    def visit(ident: str):
        mark = state.get(ident, 0)
        if mark == 1:
            raise BundleError("bundle dependency cycle detected",
                              code="dependency_cycle")
        if mark == 2:
            return
        state[ident] = 1
        for dep in by_id[ident].get("dependencies", []):
            visit(dep["id"])
        state[ident] = 2

    for entry in entries:
        visit(entry["id"])

    point = out.get("entry_point")
    if isinstance(point, str):
        point = {"path": point}
    if point is None:
        candidates = [e for e in entries if e["role"] in ("graph", "opaque")]
        if not candidates:
            raise BundleError("bundle needs a graph or opaque entry point",
                              code="entry_point")
        point = {"path": candidates[0]["path"]}
    if not isinstance(point, dict) or not isinstance(point.get("path"), str):
        raise BundleError("bundle entry_point must name a path",
                          code="entry_point")
    point_path = _safe_rel_path(point["path"], what="entry point")
    if point_path not in by_path:
        raise BundleError(f"entry point is missing: {point_path}",
                          code="entry_point")
    if by_path[point_path]["role"] not in ("graph", "opaque"):
        raise BundleError("entry point must be a graph or opaque entry",
                          code="entry_point")
    point = dict(point)
    point["path"] = point_path
    point.setdefault("kind", by_path[point_path]["role"])
    out["entry_point"] = point

    consumer = out.get("consumer")
    if not isinstance(consumer, dict):
        raise BundleError("bundle consumer must be an object", code="consumer")
    for key in ("engine", "backend", "version_range"):
        if not isinstance(consumer.get(key), str):
            raise BundleError(f"consumer {key} must be a string", code="consumer")
    for key in ("required_operators", "extensions"):
        if key in consumer and (not isinstance(consumer[key], list) or
                                not all(isinstance(x, str) for x in consumer[key])):
            raise BundleError(f"consumer {key} must be a string list",
                              code="consumer")
    state_desc = out.get("temporal_state")
    if not isinstance(state_desc, dict):
        raise BundleError("temporal_state must be an object", code="state_schema")
    if "max_state_bytes" in state_desc:
        _nonnegative_int(state_desc["max_state_bytes"], "max_state_bytes")
    resources = out.get("resources")
    if not isinstance(resources, dict):
        raise BundleError("resources must be an object", code="resources")
    workspace = resources.get("decode_workspace", {})
    if not isinstance(workspace, dict):
        raise BundleError("decode_workspace must be an object", code="resources")
    if "bytes" in workspace:
        _nonnegative_int(workspace["bytes"], "decode workspace bytes")
    if "measurement" in workspace and workspace["measurement"] not in \
            ("measured", "estimated", "unknown"):
        raise BundleError("decode workspace measurement is invalid",
                          code="resources")
    return out


def _entry_map(bundle: dict) -> tuple[dict, dict]:
    entries = bundle["entries"]
    return ({entry["path"]: entry for entry in entries},
            {entry["id"]: entry for entry in entries})


def _read_pb_string(f, a: int, b: int) -> str:
    if b - a > (1 << 20):
        raise BundleError("ONNX metadata string is too large", code="onnx_metadata")
    f.seek(a)
    try:
        return f.read(b - a).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BundleError("ONNX metadata is not UTF-8", code="onnx_metadata") from exc


def _parse_external_entry(data: bytes) -> tuple[str, str]:
    f = io.BytesIO(data)
    key = value = None
    try:
        fields = planner._pb_scan_file(f, 0, len(data))
        for field, wire, a, b in fields:
            if field == 1 and wire == 2:
                key = _read_pb_string(f, a, b)
            elif field == 2 and wire == 2:
                value = _read_pb_string(f, a, b)
    except (ValueError, OSError) as exc:
        raise BundleError(f"malformed ONNX external_data entry: {exc}",
                          code="onnx_external") from exc
    if key is None or value is None:
        raise BundleError("ONNX external_data entry lacks key/value",
                          code="onnx_external")
    return key, value


def _parse_external_dependencies(data: bytes) -> list[dict]:
    """Return ONNX external locations and ranges, without an engine import."""
    if not data or data[:1] != b"\x08":
        return []
    f = io.BytesIO(data)
    try:
        graph = planner._pb_find_from_file(f, 0, len(data), 7)
        if graph is None:
            raise BundleError("ONNX graph field is missing", code="onnx_structure")
        found: list[dict] = []
        for field, wire, a, b in planner._pb_scan_file(f, graph[0], graph[1]):
            if field != 5 or wire != 2:
                continue
            location = {}
            data_location = None
            raw_present = False
            for tf, tw, ta, tb in planner._pb_scan_file(f, a, b):
                if tf == 9 and tw == 2:
                    raw_present = True
                elif tf == 13 and tw == 2:
                    key, value = _parse_external_entry(data[ta:tb])
                    if key in location and location[key] != value:
                        raise BundleError("conflicting ONNX external_data key",
                                          code="onnx_external")
                    location[key] = value
                elif tf == 14 and tw == 0:
                    data_location = ta
            if not location and data_location != 1:
                continue
            if raw_present:
                raise BundleError("ONNX initializer mixes raw_data and external_data",
                                  code="onnx_external")
            loc = location.get("location")
            if not isinstance(loc, str) or not loc:
                raise BundleError("ONNX external_data lacks location",
                                  code="onnx_external")
            _safe_rel_path(loc, what="ONNX external location")
            def decimal(name: str, default=None):
                value = location.get(name)
                if value is None:
                    return default
                if not isinstance(value, str) or not value.isdigit():
                    raise BundleError(f"ONNX external {name} is not a non-negative decimal",
                                      code="onnx_range")
                return int(value, 10)
            found.append({"location": loc, "offset": decimal("offset", 0),
                          "length": decimal("length", None)})
        return found
    except BundleError:
        raise
    except (ValueError, OSError) as exc:
        raise BundleError(f"malformed ONNX graph: {exc}",
                          code="onnx_structure") from exc


def _resolve_external(graph_path: str, dep: dict, sizes: dict[str, int]) -> dict:
    location = dep["location"]
    graph_dir = posixpath.dirname(graph_path)
    resolved = posixpath.join(graph_dir, location) if graph_dir else location
    _safe_rel_path(resolved, what="resolved ONNX external location")
    if resolved not in sizes:
        raise BundleError(f"ONNX sidecar is missing: {resolved}",
                          code="missing_dependency")
    side_size = sizes[resolved]
    offset = dep["offset"]
    if offset > side_size:
        raise BundleError(f"ONNX offset past sidecar: {resolved}",
                          code="invalid_range")
    length = side_size - offset if dep["length"] is None else dep["length"]
    if length < 0 or offset + length < offset or offset + length > side_size:
        raise BundleError(f"ONNX range past sidecar: {resolved}",
                          code="invalid_range")
    return {"path": resolved, "offset": offset, "length": length}


def _external_requirements(files: dict[str, _SourceFile], entries: dict[str, dict]) -> dict[str, list[dict]]:
    requirements: dict[str, list[dict]] = {}
    sizes = {path: item.size for path, item in files.items()}
    for path, entry in entries.items():
        if entry["role"] != "graph" or not path.lower().endswith(".onnx"):
            continue
        with open(files[path].staged_path, "rb") as fh:
            data = fh.read()
        found = _parse_external_dependencies(data)
        requirements[path] = [_resolve_external(path, dep, sizes) for dep in found]
    return requirements


def _merge_external_dependencies(bundle: dict, requirements: dict[str, list[dict]]):
    by_path, _by_id = _entry_map(bundle)
    for graph_path, found in requirements.items():
        entry = by_path[graph_path]
        declared = entry.get("dependencies", [])
        for requirement in found:
            matching = []
            for dep in declared:
                if isinstance(dep, str) and dep == requirement["path"]:
                    matching.append(dep)
                elif isinstance(dep, dict) and (dep.get("path") == requirement["path"] or
                                                dep.get("id") == by_path[requirement["path"]]["id"]):
                    matching.append(dep)
            if matching:
                item = matching[0]
                if isinstance(item, str):
                    declared[declared.index(item)] = {
                        "path": requirement["path"],
                        "id": by_path[requirement["path"]]["id"],
                        "offset": requirement["offset"],
                        "length": requirement["length"],
                    }
                elif (item.get("offset", 0) != requirement["offset"] or
                      item.get("length") != requirement["length"]):
                    raise BundleError(
                        f"manifest dependency range disagrees with ONNX graph: {graph_path}",
                        code="dependency_mismatch")
            else:
                declared.append({"path": requirement["path"],
                                 "id": by_path[requirement["path"]]["id"],
                                 "offset": requirement["offset"],
                                 "length": requirement["length"]})
        entry["dependencies"] = declared


def _member_chunks(reader: ArchiveReader, member) -> list:
    chunks = []
    end = member.dst + member.size
    for chunk in reader.chunks:
        if member.dst <= chunk.dst < end or (member.size == 0 and chunk.dst == member.dst):
            if chunk.dst < member.dst or chunk.dst + chunk.rlen > end:
                raise BundleError(f"archive chunk crosses member {member.path}",
                                  code="archive_coverage")
            chunks.append(chunk)
    chunks.sort(key=lambda c: c.dst)
    pos = member.dst
    for chunk in chunks:
        if chunk.dst != pos:
            raise BundleError(f"archive member has a gap or overlap: {member.path}",
                              code="archive_coverage")
        pos += chunk.rlen
    if pos != end:
        raise BundleError(f"archive member is not fully covered: {member.path}",
                          code="archive_coverage")
    return chunks


def _union_payload_bytes(chunks: list) -> int:
    ranges = sorted((c.off, c.off + c.clen) for c in chunks if c.clen)
    total = 0
    end = -1
    for start, stop in ranges:
        if start > end:
            total += stop - start
            end = stop
        elif stop > end:
            total += stop - end
            end = stop
    return total


def _actual_representation(chunks: list) -> dict:
    names = [CODEC_NAMES.get(c.codec, f"codec-{c.codec}") for c in chunks]
    counts = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1
    codecs = sorted(counts)
    if not codecs or set(codecs) == {"stored"}:
        mode = "plain"
    elif set(codecs) == {"ref"}:
        mode = "reference"
    else:
        mode = "codec"
    stored = sum(c.clen for c in chunks)
    return {
        "mode": mode,
        "codec": codecs[0] if len(codecs) == 1 else "mixed",
        "codecs": codecs,
        "codec_counts": counts,
        "stored_bytes": stored,
        "physical_payload_bytes": _union_payload_bytes(chunks),
        "chunk_count": len(chunks),
    }


def _tail_lengths(fh, reader: ArchiveReader) -> tuple[int, int, int]:
    fh.seek(reader.file_size - FOOTER.size)
    table_off, table_len, manifest_off, manifest_len, _tail = FOOTER.unpack(
        fh.read(FOOTER.size))
    return table_len, manifest_len, reader.file_size - reader.payload_end


def _actual_entries(archive: str, reader: ArchiveReader) -> tuple[list[dict], dict]:
    actual = []
    by_path = {}
    with api.MappedArchive(archive, verify=True) as mapped:
        for member in reader.members:
            chunks = _member_chunks(reader, member)
            data = mapped.read_member(member.path)
            digest = hashlib.sha256(data).hexdigest()
            representation = _actual_representation(chunks)
            # The complete-cost figure intentionally includes one record and a
            # proportional metadata allowance. It is a decision aid, never a
            # claim that container metadata disappeared.
            complete_cost = representation["stored_bytes"] + 32
            item = {
                "path": member.path,
                "size": member.size,
                "decoded_bytes": member.size,
                "sha256": digest,
                "representation": representation,
                "complete_stored_cost": complete_cost,
                "compression_smaller": complete_cost < member.size,
                "mode": member.mode,
                "kind": member.kind,
            }
            actual.append(item)
            by_path[member.path] = item
    return actual, by_path


def _compare_representation(declared: dict, actual: dict, path: str):
    rep = declared.get("representation")
    if not isinstance(rep, dict):
        raise BundleError(f"entry {path} lacks actual representation metadata",
                          code="representation")
    for key in ("stored_bytes", "physical_payload_bytes", "chunk_count"):
        if key in rep and rep[key] != actual[key]:
            raise BundleError(f"entry {path} representation {key} disagrees with archive",
                              code="representation")
    if "codec" in rep and rep["codec"] != actual["codec"]:
        raise BundleError(f"entry {path} representation codec disagrees with archive",
                          code="representation")


def _verify_external_manifest(archive: str, bundle: dict,
                              actual_by_path: dict[str, dict]):
    sizes = {path: item["decoded_bytes"] for path, item in actual_by_path.items()}
    by_path, _by_id = _entry_map(bundle)
    with api.MappedArchive(archive, verify=True) as mapped:
        for graph_path, entry in by_path.items():
            if entry["role"] != "graph" or not graph_path.lower().endswith(".onnx"):
                continue
            data = mapped.read_member(graph_path)
            found = _parse_external_dependencies(data)
            required = _resolve_external_requirements_for_bundle(graph_path, found,
                                                                 sizes)
            declared = {(dep["path"], dep.get("offset", 0),
                         dep.get("length", sizes[dep["path"]] - dep.get("offset", 0)))
                        for dep in entry.get("dependencies", [])}
            for item in required:
                key = (item["path"], item["offset"], item["length"])
                if key not in declared:
                    raise BundleError(f"ONNX dependency is not declared: {graph_path} -> {key}",
                                      code="dependency_mismatch")


def _resolve_external_requirements_for_bundle(graph_path: str, found: list[dict],
                                              sizes: dict[str, int]) -> list[dict]:
    out = []
    for item in found:
        location = item["location"]
        graph_dir = posixpath.dirname(graph_path)
        resolved = posixpath.join(graph_dir, location) if graph_dir else location
        _safe_rel_path(resolved, what="resolved ONNX external location")
        if resolved not in sizes:
            raise BundleError(f"ONNX sidecar is missing: {resolved}",
                              code="missing_dependency")
        offset = item["offset"]
        length = sizes[resolved] - offset if item["length"] is None else item["length"]
        if offset > sizes[resolved] or length < 0 or offset + length < offset or \
                offset + length > sizes[resolved]:
            raise BundleError(f"ONNX external range is invalid: {resolved}",
                              code="invalid_range")
        out.append({"path": resolved, "offset": offset, "length": length})
    return out


def _inspect(archive: str, *, expected_manifest_sha256: str | None = None,
             expected_bundle_sha256: str | None = None) -> dict:
    with open(archive, "rb") as fh:
        reader = ArchiveReader(fh)
        bundle_raw = reader.manifest.get(BUNDLE_MANIFEST_KEY)
        if bundle_raw is None:
            raise BundleError("archive has no bundle manifest", code="not_bundle")
        bundle = _normalise_bundle(bundle_raw)
    actual, actual_by_path = _actual_entries(archive, reader)
    paths = {item["path"] for item in actual}
    declared_paths = {item["path"] for item in bundle["entries"]}
    if paths != declared_paths:
        raise BundleError("bundle entries do not match archive members",
                          code="entry_mismatch")
    declared_by_path = {item["path"]: item for item in bundle["entries"]}
    for index, item in enumerate(actual):
        declared = declared_by_path[item["path"]]
        if declared["size"] != item["size"] or declared["sha256"] != item["sha256"]:
            raise BundleError(f"entry identity mismatch: {item['path']}",
                              code="digest_mismatch")
        _compare_representation(declared, item["representation"], item["path"])
        # Inventory is the consumer-facing join of manifest semantics and
        # archive facts. Keep role, identity, dependencies and opaque
        # constraints alongside the measured representation.
        joined = dict(declared)
        joined.update(item)
        actual[index] = joined
        actual_by_path[item["path"]] = joined
    _verify_external_manifest(archive, bundle, actual_by_path)
    canonical_digest = manifest_sha256(bundle)
    for expected, label in ((expected_manifest_sha256, "manifest"),
                            (expected_bundle_sha256, "bundle")):
        if expected is not None:
            if not isinstance(expected, str) or not _SHA256.fullmatch(expected):
                raise BundleError(f"expected {label} digest is not sha256",
                                  code="expected_digest")
            if expected != canonical_digest:
                raise BundleError(f"expected {label} digest does not match",
                                  code="expected_digest")
    with open(archive, "rb") as fh:
        table_len, manifest_len, tail_len = _tail_lengths(fh, reader)
    # A ref or delta entry can name payload bytes already used by another
    # member. Archive accounting therefore uses the union of all physical
    # chunk ranges; per-entry representation retains its own physical cost.
    payload_bytes = _union_payload_bytes(list(reader.chunks))
    decoded_bytes = sum(item["decoded_bytes"] for item in actual)
    result = {
        "status": "verified",
        "validation": {
            "status": "verified",
            "reason": None,
            "expected_digest": (expected_manifest_sha256 or expected_bundle_sha256),
            "expected_digest_match": (expected_manifest_sha256 is not None or
                                       expected_bundle_sha256 is not None),
            "publisher_authenticated": False,
        },
        "archive": archive,
        "archive_bytes": reader.file_size,
        "bundle": bundle,
        "bundle_manifest_sha256": canonical_digest,
        "entries": actual,
        "entry_point": bundle["entry_point"],
        "consumer": bundle["consumer"],
        "archive_accounting": {
            "decoded_bytes": decoded_bytes,
            "materialized_bytes": decoded_bytes,
            "unique_payload_bytes": payload_bytes,
            "chunk_table_stored_bytes": table_len,
            "container_manifest_stored_bytes": manifest_len,
            "fixed_archive_overhead_bytes": reader.file_size - payload_bytes,
            "tail_bytes": tail_len,
        },
        "workspace": bundle["resources"].get("decode_workspace", {
            "bytes": 0, "measurement": "unknown"}),
        "materialization": {"status": "not_run", "path": None,
                            "failure": None},
    }
    return result


def _rewrite_bundle_manifest(archive: str, bundle: dict) -> None:
    with open(archive, "r+b") as fh:
        reader = ArchiveReader(fh)
        manifest = dict(reader.manifest)
        manifest[BUNDLE_MANIFEST_KEY] = bundle
        writer = ArchiveWriter(fh, manifest, flags=reader.flags,
                               resume_at=reader.table_off,
                               chunks=list(reader.chunks))
        writer.close(reader.original_size)
        fh.truncate(writer.end)
        fh.flush()
        os.fsync(fh.fileno())


def create_bundle(source_root: str, archive: str, manifest: dict | None = None,
                  *, bundle: dict | None = None, level: int = api.DEFAULT_LEVEL,
                  workers: int | None = None, checksum: bool = True,
                  dedup: bool = True, delta: bool = True,
                  mapped: bool = False, align: bool = False,
                  shared_tables: bool = False, overwrite: bool = False) -> dict:
    """Snapshot a regular-file tree and create a complete bundle archive.

    ``manifest`` and the keyword alias ``bundle`` are the semantic bundle
    description.  Missing file entries are added as opaque/config/etc. based
    on their extension so the archive remains complete; supplied size/digest
    values are treated as expected source identity and must match.
    """
    if manifest is not None and bundle is not None:
        raise BundleError("pass either manifest or bundle, not both",
                          code="manifest_type")
    if bundle is not None:
        manifest = bundle
    if not os.path.isdir(source_root) or os.path.islink(source_root):
        raise BundleError("bundle source root must be a real directory",
                          code="unsafe_source")
    if os.path.lexists(archive) and not overwrite:
        raise FileExistsError(archive)
    parent = os.path.dirname(os.path.abspath(archive)) or "."
    os.makedirs(parent, exist_ok=True)
    created = False
    with tempfile.TemporaryDirectory(prefix=".lmz-bundle-source-",
                                      dir=parent) as stage:
        files = _snapshot_source(source_root, stage)
        working = _merge_manifest(source_root, manifest)
        entries = _normalise_entries(working, paths=files.keys())
        working["entries"] = entries
        working = _normalise_bundle(working, paths=files.keys(),
                                    require_artifact_fields=False)
        by_path, _by_id = _entry_map(working)
        for path, entry in by_path.items():
            expected_size = entry.get("size", entry.get("decoded_bytes"))
            expected_sha = entry.get("sha256")
            if expected_size is not None and expected_size != files[path].size:
                raise BundleError(f"source size disagrees for {path}",
                                  code="source_identity")
            if expected_sha is not None and expected_sha != files[path].sha256:
                raise BundleError(f"source digest disagrees for {path}",
                                  code="source_identity")
            entry["size"] = files[path].size
            entry["decoded_bytes"] = files[path].size
            entry["sha256"] = files[path].sha256
            entry.setdefault("mode", files[path].mode)
        requirements = _external_requirements(files, by_path)
        _merge_external_dependencies(working, requirements)
        # Validate the source-side dependency graph before any archive bytes
        # are published. Archive representations are filled after compression.
        working = _normalise_bundle(working, paths=files.keys(),
                                    require_artifact_fields=True)
        working["resources"] = dict(working["resources"])
        working["resources"]["materialized_bytes"] = sum(
            item["size"] for item in working["entries"])
        # A temporary sidecar lets the normal compressor preserve all existing
        # tensor planning and codecs.  It is removed with the source snapshot.
        api.compress(stage, archive, level=level, workers=workers,
                     checksum=checksum, dedup=dedup, delta=delta,
                     mapped=mapped, align=align, shared_tables=shared_tables)
        created = True
        with open(archive, "rb") as fh:
            reader = ArchiveReader(fh)
        actual, actual_by_path = _actual_entries(archive, reader)
        actual_map = {item["path"]: item for item in actual}
        for entry in working["entries"]:
            actual_item = actual_map[entry["path"]]
            entry.update({
                "size": actual_item["size"],
                "decoded_bytes": actual_item["decoded_bytes"],
                "sha256": actual_item["sha256"],
                "representation": actual_item["representation"],
                "complete_stored_cost": actual_item["complete_stored_cost"],
                "compression_smaller": actual_item["compression_smaller"],
            })
            entry["mode"] = actual_item["mode"]
        working = _normalise_bundle(working, paths=files.keys())
        _rewrite_bundle_manifest(archive, working)
    try:
        return inventory_bundle(archive)
    except BaseException:
        if created:
            try:
                os.unlink(archive)
            except OSError:
                pass
        raise


def validate_bundle(archive: str, *, expected_manifest_sha256: str | None = None,
                    expected_bundle_sha256: str | None = None) -> dict:
    """Verify structure, every entry digest, dependencies and ONNX ranges."""
    return _inspect(archive, expected_manifest_sha256=expected_manifest_sha256,
                    expected_bundle_sha256=expected_bundle_sha256)


def inventory_bundle(archive: str, *, expected_manifest_sha256: str | None = None,
                     expected_bundle_sha256: str | None = None,
                     strict: bool = True) -> dict:
    """Return the machine-readable inventory lmsluice can consume."""
    try:
        return _inspect(archive, expected_manifest_sha256=expected_manifest_sha256,
                        expected_bundle_sha256=expected_bundle_sha256)
    except BundleError as exc:
        if strict:
            raise
        return {
            "status": "invalid",
            "validation": {"status": "failed", "reason": exc.as_dict(),
                            "publisher_authenticated": False},
            "archive": archive,
            "materialization": {"status": "not_run", "path": None,
                                "failure": exc.as_dict()},
        }
    except FormatError as exc:
        reason = {"code": "archive_invalid", "message": str(exc)}
        if strict:
            raise
        return {
            "status": "invalid",
            "validation": {"status": "failed", "reason": reason,
                            "publisher_authenticated": False},
            "archive": archive,
            "materialization": {"status": "not_run", "path": None,
                                "failure": reason},
        }


def _ensure_directory_chain(path: str):
    """Create/check every directory component without following symlinks."""
    if not _O_NOFOLLOW or not _O_DIRECTORY:
        raise BundleError("no-follow directory creation is unavailable",
                          code="unsafe_destination")
    absolute = os.path.abspath(path)
    root_fd = os.open(os.path.sep, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW)
    final_fd = -1
    try:
        parts = [part for part in absolute.split(os.path.sep) if part]
        final_fd = _mkdir_chain(root_fd, parts)
    except OSError as exc:
        if exc.errno in (errno.ELOOP, errno.ENOTDIR):
            raise BundleError(
                f"destination parent contains a symlink or non-directory: {path}",
                code="unsafe_destination") from exc
        raise
    finally:
        if final_fd >= 0:
            os.close(final_fd)
        os.close(root_fd)


def _mkdir_chain(root_fd: int, parts: list[str]) -> int:
    fd = os.dup(root_fd)
    try:
        for part in parts:
            try:
                os.mkdir(part, 0o755, dir_fd=fd)
            except FileExistsError:
                pass
            nxt = os.open(part, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW,
                          dir_fd=fd)
            os.close(fd)
            fd = nxt
        return fd
    except BaseException:
        os.close(fd)
        raise


def _write_relative(root_fd: int, relative: str, data: bytes, mode: int):
    parts = relative.split("/")
    parent_fd = _mkdir_chain(root_fd, parts[:-1])
    try:
        fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                     _O_NOFOLLOW, mode or 0o644, dir_fd=parent_fd)
        try:
            view = memoryview(data)
            while view:
                n = os.write(fd, view)
                if n <= 0:
                    raise OSError("short write")
                view = view[n:]
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)


def _publish_noreplace(staging: str, destination: str):
    """Atomically rename a staging directory without replacing a destination."""
    parent = os.path.dirname(destination)
    staging_name = os.path.basename(staging)
    destination_name = os.path.basename(destination)
    parent_fd = os.open(parent, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW)
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        fn = getattr(libc, "renameat2", None)
        if fn is not None:
            fn.restype = ctypes.c_int
            fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int,
                           ctypes.c_char_p, ctypes.c_uint]
            result = fn(parent_fd, os.fsencode(staging_name), parent_fd,
                        os.fsencode(destination_name), _RENAME_NOREPLACE)
        else:
            syscall = getattr(libc, "syscall", None)
            numbers = {"x86_64": 316, "amd64": 316, "aarch64": 276,
                       "arm64": 276, "ppc64le": 357, "riscv64": 276}
            number = numbers.get(os.uname().machine)
            if syscall is None or number is None:
                raise BundleError(
                    "atomic no-replace directory publish is unavailable",
                    code="atomic_publish_unavailable")
            syscall.restype = ctypes.c_long
            result = syscall(number, parent_fd, os.fsencode(staging_name),
                             parent_fd, os.fsencode(destination_name),
                             _RENAME_NOREPLACE)
        if result != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(destination)
            if error in (errno.ENOSYS, errno.ENOTSUP, errno.EINVAL):
                raise BundleError(
                    "atomic no-replace directory publish is unavailable",
                    code="atomic_publish_unavailable")
            raise OSError(error, os.strerror(error), destination)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def materialize_bundle(archive: str, destination: str, *,
                       expected_manifest_sha256: str | None = None,
                       expected_bundle_sha256: str | None = None) -> dict:
    """Materialize a verified bundle into a new destination atomically."""
    inventory = validate_bundle(
        archive, expected_manifest_sha256=expected_manifest_sha256,
        expected_bundle_sha256=expected_bundle_sha256)
    destination = os.path.abspath(destination)
    _ensure_directory_chain(os.path.dirname(destination))
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    parent = os.path.dirname(destination)
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    staging = tempfile.mkdtemp(prefix=".lmz-materialize-", dir=parent)
    root_fd = -1
    try:
        root_fd = os.open(staging, os.O_RDONLY | _O_DIRECTORY | _O_NOFOLLOW)
        with api.MappedArchive(archive, verify=True) as mapped:
            for entry in sorted(inventory["entries"], key=lambda item: item["path"]):
                data = mapped.read_member(entry["path"])
                if len(data) != entry["decoded_bytes"] or \
                        hashlib.sha256(data).hexdigest() != entry["sha256"]:
                    raise BundleError(f"entry changed while materializing: {entry['path']}",
                                      code="digest_mismatch")
                _write_relative(root_fd, entry["path"], data, entry.get("mode", 0o644))
        os.fsync(root_fd)
        os.close(root_fd)
        root_fd = -1
        _publish_noreplace(staging, destination)
        staging = None
        inventory["materialization"] = {
            "status": "complete", "path": destination, "failure": None,
        }
        inventory["status"] = "complete"
        return inventory
    except BaseException as exc:
        if root_fd >= 0:
            os.close(root_fd)
            root_fd = -1
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        if isinstance(exc, BundleError):
            raise
        raise


# Short public aliases keep the surface easy to discover while the explicit
# names make call sites self-documenting.
create = create_bundle
validate = validate_bundle
inventory = inventory_bundle
materialize = materialize_bundle


__all__ = [
    "BUNDLE_MANIFEST_KEY", "BUNDLE_SCHEMA_MAJOR", "BUNDLE_SCHEMA_MINOR",
    "BundleError", "BundleSourceChanged", "canonical_manifest_bytes",
    "manifest_sha256", "create_bundle", "validate_bundle", "inventory_bundle",
    "materialize_bundle", "create", "validate", "inventory", "materialize",
]
