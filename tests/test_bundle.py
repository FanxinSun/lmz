"""Generated-fixture tests for complete, safe LMZ artifact bundles."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lmz  # noqa: E402
from lmz import bundle  # noqa: E402


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def _varint(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _field_int(field, value):
    return _varint(field << 3) + _varint(value)


def _field_bytes(field, value):
    return _varint((field << 3) | 2) + _varint(len(value)) + value


def _external_onnx(location, offset, length):
    def pair(key, value):
        return _field_bytes(1, key.encode()) + _field_bytes(2, value.encode())
    tensor = (_field_int(2, 1) + _field_bytes(8, b"weights") +
              _field_bytes(13, pair("location", location)) +
              _field_bytes(13, pair("offset", str(offset))) +
              _field_bytes(13, pair("length", str(length))) +
              _field_int(14, 1))
    graph = _field_bytes(5, tensor) + _field_bytes(2, b"main")
    return _field_int(1, 8) + _field_bytes(7, graph) + _field_bytes(8, b"test")


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def test_mixed_bundle_roundtrip_inventory_and_materialization():
    with tempfile.TemporaryDirectory() as d:
        source = os.path.join(d, "source")
        os.makedirs(source)
        _write(os.path.join(source, "graph.onnx"), b"generated graph bytes")
        _write(os.path.join(source, "weights.bin"), bytes(range(251)) * 20)
        _write(os.path.join(source, "config.json"), b'{"sample_rate":16000}')
        _write(os.path.join(source, "preprocess.json"), b'{"layout":"NCHW"}')
        _write(os.path.join(source, "vocabulary.txt"), b"silence\nhello\n")
        _write(os.path.join(source, "calibration.bin"), b"calibration")
        _write(os.path.join(source, "backend.payload"), b"opaque-backend")
        archive = os.path.join(d, "artifact.lmz")
        manifest = {
            "id": "observer-fixture",
            "version": "2026.09",
            "source": {"uri": "fixture://observer", "revision": "gen-1"},
            "license": {"classification": "generated",
                         "redistribution": "allowed"},
            "entry_point": {"path": "graph.onnx", "kind": "graph"},
            "consumer": {"engine": "fixture-engine", "backend": "cpu",
                          "version_range": ">=0", "required_operators": [],
                          "extensions": [], "export_provenance": "generated"},
            "io": {"inputs": [{"name": "frame", "dtype": "U8",
                                 "shape": [1, 4]}],
                   "outputs": [{"name": "label", "dtype": "U8"}]},
            "preprocess": {"identity": "fixture-normalize", "parameters":
                           {"layout": "NCHW"}},
            "temporal_state": {"inputs": [{"name": "state", "bytes": 8}],
                                "outputs": [{"name": "state", "bytes": 8}],
                                "initialization": "zero", "reset": "on discontinuity",
                                "cadence": "one frame", "discontinuity": "reset",
                                "max_state_bytes": 8},
            "resources": {"decode_workspace": {"bytes": 4096,
                                                  "measurement": "estimated",
                                                  "method": "fixture estimate"}},
            "evaluation": {"input_identity": "fixture-input-1",
                           "expected_output_identity": "fixture-output-1",
                           "metric": "exact", "tolerance": 0},
            "entries": [
                {"path": "graph.onnx", "role": "graph"},
                {"path": "weights.bin", "role": "weights"},
                {"path": "config.json", "role": "config"},
                {"path": "preprocess.json", "role": "preprocess"},
                {"path": "vocabulary.txt", "role": "vocabulary"},
                {"path": "calibration.bin", "role": "calibration"},
                {"path": "backend.payload", "role": "opaque",
                 "consumer": {"engine": "fixture-engine",
                               "backend": "opaque-cpu", "version_range": "==1"}},
            ],
        }
        result = lmz.create_bundle(source, archive, manifest, workers=1)
        assert result["status"] == "verified"
        assert result["entry_point"]["path"] == "graph.onnx"
        assert result["archive_accounting"]["decoded_bytes"] == sum(
            os.path.getsize(os.path.join(source, name)) for name in (
                "graph.onnx", "weights.bin", "config.json", "preprocess.json",
                "vocabulary.txt", "calibration.bin", "backend.payload"))
        assert result["archive_accounting"]["fixed_archive_overhead_bytes"] > 0
        assert {e["path"] for e in result["entries"]} == {
            "graph.onnx", "weights.bin", "config.json", "preprocess.json",
            "vocabulary.txt", "calibration.bin", "backend.payload"}
        backend = next(e for e in result["entries"]
                       if e["path"] == "backend.payload")
        assert backend["consumer"]["version_range"] == "==1"
        expected = result["bundle_manifest_sha256"]
        assert lmz.validate_bundle(archive,
                                   expected_manifest_sha256=expected)["status"] == "verified"
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cli = os.path.join(repo_root, "lmz-cli")
        inventory = subprocess.run(
            [sys.executable, cli, "bundle", "inventory", archive,
             "--expected-manifest-sha256", expected, "--json"],
            check=False, capture_output=True, text=True)
        assert inventory.returncode == 0, inventory.stderr
        assert json.loads(inventory.stdout)["status"] == "verified"
        manifest_path = os.path.join(d, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh)
        cli_archive = os.path.join(d, "cli-artifact.lmz")
        created = subprocess.run(
            [sys.executable, cli, "bundle", "create", source, cli_archive,
             "--manifest", manifest_path, "--threads", "1", "--json"],
            check=False, capture_output=True, text=True)
        assert created.returncode == 0, created.stderr
        cli_result = json.loads(created.stdout)
        cli_out = os.path.join(d, "cli-materialized")
        materialized_cli = subprocess.run(
            [sys.executable, cli, "bundle", "materialize", cli_archive,
             cli_out, "--expected-manifest-sha256",
             cli_result["bundle_manifest_sha256"], "--json"],
            check=False, capture_output=True, text=True)
        assert materialized_cli.returncode == 0, materialized_cli.stderr
        assert json.loads(materialized_cli.stdout)["status"] == "complete"
        destination = os.path.join(d, "materialized")
        materialized = lmz.materialize_bundle(archive, destination)
        assert materialized["status"] == "complete"
        for root, _dirs, names in os.walk(source):
            for name in names:
                src = os.path.join(root, name)
                rel = os.path.relpath(src, source)
                assert digest(src) == digest(os.path.join(destination, rel))


def test_external_data_closure_ranges_and_missing_sidecar():
    with tempfile.TemporaryDirectory() as d:
        source = os.path.join(d, "src")
        _write(os.path.join(source, "graphs", "model.onnx"),
               _external_onnx("weights.bin", 3, 7))
        _write(os.path.join(source, "graphs", "model2.onnx"),
               _external_onnx("weights.bin", 3, 7))
        _write(os.path.join(source, "graphs", "weights.bin"), b"xxPAYLOADyy")
        archive = os.path.join(d, "external.lmz")
        result = lmz.create_bundle(source, archive, {
            "id": "external", "version": "1",
            "source": {"revision": "fixture"},
            "license": {"classification": "generated", "redistribution": "allowed"},
            "entry_point": "graphs/model.onnx",
            "consumer": {"engine": "none", "backend": "cpu", "version_range": "*"},
            "entries": [
                {"path": "graphs/model.onnx", "role": "graph",
                 "dependencies": ["graphs/weights.bin"]},
                {"path": "graphs/model2.onnx", "role": "graph"},
                {"path": "graphs/weights.bin", "role": "weights"},
            ],
        }, workers=1)
        graph = next(e for e in result["bundle"]["entries"]
                     if e["path"] == "graphs/model.onnx")
        assert graph["dependencies"] == [{
            "id": next(e["id"] for e in result["bundle"]["entries"]
                        if e["path"] == "graphs/weights.bin"),
            "path": "graphs/weights.bin", "offset": 3, "length": 7,
        }]
        graph2 = next(e for e in result["bundle"]["entries"]
                      if e["path"] == "graphs/model2.onnx")
        assert graph2["dependencies"] == graph["dependencies"]
        assert lmz.validate_bundle(archive)["status"] == "verified"
        out = os.path.join(d, "out")
        lmz.materialize_bundle(archive, out)
        assert open(os.path.join(out, "graphs", "weights.bin"), "rb").read() == b"xxPAYLOADyy"

        corrupt_sidecar = os.path.join(d, "corrupt-sidecar.lmz")
        shutil.copyfile(archive, corrupt_sidecar)
        with open(corrupt_sidecar, "r+b") as fh:
            reader = bundle.ArchiveReader(fh)
            member = next(m for m in reader.members
                          if m.path == "graphs/weights.bin")
            chunk = next(c for c in reader.chunks if c.dst == member.dst)
            fh.seek(chunk.off)
            original = fh.read(1)
            fh.seek(chunk.off)
            fh.write(bytes([original[0] ^ 0x01]))
        invalid_sidecar = lmz.inventory_bundle(corrupt_sidecar, strict=False)
        assert invalid_sidecar["status"] == "invalid"
        assert invalid_sidecar["validation"]["reason"]["code"] == "archive_invalid"

        bad_range = os.path.join(d, "bad-range")
        _write(os.path.join(bad_range, "model.onnx"),
               _external_onnx("weights.bin", 0, 99))
        _write(os.path.join(bad_range, "weights.bin"), b"short")
        try:
            lmz.create_bundle(bad_range, os.path.join(d, "bad-range.lmz"),
                              workers=1)
        except lmz.BundleError as exc:
            assert exc.code == "invalid_range"
        else:
            raise AssertionError("out-of-bounds ONNX range was accepted")

        missing = os.path.join(d, "missing")
        _write(os.path.join(missing, "model.onnx"),
               _external_onnx("weights.bin", 0, 2))
        try:
            lmz.create_bundle(missing, os.path.join(d, "missing.lmz"), workers=1)
        except lmz.BundleError as exc:
            assert exc.code == "missing_dependency"
        else:
            raise AssertionError("missing ONNX sidecar was accepted")


def test_hostile_paths_links_destination_and_cleanup():
    with tempfile.TemporaryDirectory() as d:
        source = os.path.join(d, "source")
        _write(os.path.join(source, "tiny.payload"), b"abc")
        for hostile in ("/absolute", "C:/drive", "\\\\server\\share",
                        "mixed\\separator", "a/../b", "a//b", "a/./b"):
            try:
                bundle._safe_rel_path(hostile)
            except lmz.BundleError as exc:
                assert exc.code == "unsafe_path"
            else:
                raise AssertionError(f"hostile path was accepted: {hostile!r}")
        archive = os.path.join(d, "tiny.lmz")
        try:
            lmz.create_bundle(source, archive, {
                "id": "bad", "version": "1", "source": {"revision": "x"},
                "license": {"classification": "unknown", "redistribution": "unknown"},
                "entries": [{"path": "../escape", "role": "opaque"}],
            })
        except lmz.BundleError as exc:
            assert exc.code == "unsafe_path"
        else:
            raise AssertionError("traversal path was accepted")

        linked = os.path.join(d, "linked")
        os.symlink(source, linked)
        try:
            lmz.create_bundle(linked, os.path.join(d, "linked.lmz"))
        except lmz.BundleError as exc:
            assert exc.code == "unsafe_source"
        else:
            raise AssertionError("source symlink was accepted")

        linked_file = os.path.join(source, "linked.payload")
        os.symlink(os.path.join(source, "tiny.payload"), linked_file)
        try:
            lmz.create_bundle(source, os.path.join(d, "linked-file.lmz"))
        except lmz.BundleError as exc:
            assert exc.code == "unsafe_source"
        else:
            raise AssertionError("source file symlink was accepted")
        os.unlink(linked_file)

        if hasattr(os, "mkfifo"):
            fifo = os.path.join(source, "pipe")
            os.mkfifo(fifo)
            try:
                lmz.create_bundle(source, os.path.join(d, "fifo.lmz"))
            except lmz.BundleError as exc:
                assert exc.code == "unsafe_source"
            else:
                raise AssertionError("source FIFO was accepted")
            os.unlink(fifo)

        good = os.path.join(d, "good.lmz")
        lmz.create_bundle(source, good, workers=1)
        existing = os.path.join(d, "existing")
        os.mkdir(existing)
        try:
            lmz.materialize_bundle(good, existing)
        except FileExistsError:
            pass
        else:
            raise AssertionError("existing destination was overwritten")

        original_publish = bundle._publish_noreplace
        try:
            def fail_publish(_staging, _destination):
                raise RuntimeError("fixture publish failure")
            bundle._publish_noreplace = fail_publish
            failed = os.path.join(d, "failed")
            try:
                lmz.materialize_bundle(good, failed)
            except RuntimeError:
                pass
            else:
                raise AssertionError("publish failure was swallowed")
            assert not os.path.exists(failed)
            assert not [name for name in os.listdir(d)
                        if name.startswith(".lmz-materialize-")]
        finally:
            bundle._publish_noreplace = original_publish

        appeared = os.path.join(d, "appeared")
        try:
            def appear_then_publish(staging, destination):
                os.mkdir(destination)
                original_publish(staging, destination)
            bundle._publish_noreplace = appear_then_publish
            try:
                lmz.materialize_bundle(good, appeared)
            except FileExistsError:
                pass
            else:
                raise AssertionError("destination appearance race was accepted")
            assert os.path.isdir(appeared)
            assert not [name for name in os.listdir(d)
                        if name.startswith(".lmz-materialize-")]
        finally:
            bundle._publish_noreplace = original_publish

        real_parent = os.path.join(d, "real-parent")
        os.mkdir(real_parent)
        linked_parent = os.path.join(d, "linked-parent")
        os.symlink(real_parent, linked_parent)
        try:
            lmz.materialize_bundle(good, os.path.join(linked_parent, "out"))
        except lmz.BundleError as exc:
            assert exc.code == "unsafe_destination"
        else:
            raise AssertionError("symlinked destination parent was followed")

        original = os.path.join(d, "race-source")
        replacement = os.path.join(d, "replacement")
        _write(original, b"before")
        _write(replacement, b"after")
        before = os.stat(original)
        os.replace(replacement, original)
        try:
            bundle._snapshot_file(original, os.path.join(d, "snapshot"),
                                  before, 0o644)
        except lmz.BundleSourceChanged as exc:
            assert exc.code == "source_changed"
        else:
            raise AssertionError("source identity change was accepted")


def test_invalid_manifest_cycle_and_corrupt_archive_are_reported():
    with tempfile.TemporaryDirectory() as d:
        source = os.path.join(d, "source")
        _write(os.path.join(source, "a.payload"), b"a")
        _write(os.path.join(source, "b.payload"), b"b")
        cycle = {
            "id": "cycle", "version": "1", "source": {"revision": "fixture"},
            "license": {"classification": "generated", "redistribution": "allowed"},
            "entry_point": "a.payload",
            "consumer": {"engine": "opaque", "backend": "cpu", "version_range": "*"},
            "entries": [
                {"path": "a.payload", "role": "opaque",
                 "dependencies": ["b.payload"]},
                {"path": "b.payload", "role": "opaque",
                 "dependencies": ["a.payload"]},
            ],
        }
        try:
            lmz.create_bundle(source, os.path.join(d, "cycle.lmz"), cycle,
                              workers=1)
        except lmz.BundleError as exc:
            assert exc.code == "dependency_cycle"
        else:
            raise AssertionError("dependency cycle was accepted")

        good = os.path.join(d, "good.lmz")
        result = lmz.create_bundle(source, good, workers=1)
        unsupported = os.path.join(d, "unsupported.lmz")
        shutil.copyfile(good, unsupported)
        with open(unsupported, "r+b") as fh:
            reader = bundle.ArchiveReader(fh)
            bad_manifest = dict(reader.manifest["bundle"])
            bad_manifest["schema"] = {"major": 2, "minor": 0}
            bundle._rewrite_bundle_manifest(unsupported, bad_manifest)
        invalid = lmz.inventory_bundle(unsupported, strict=False)
        assert invalid["status"] == "invalid"
        assert invalid["validation"]["reason"]["code"] == "unsupported_major"

        corrupt = os.path.join(d, "corrupt.lmz")
        shutil.copyfile(good, corrupt)
        with open(corrupt, "r+b") as fh:
            reader = bundle.ArchiveReader(fh)
            chunk = reader.chunks[0]
            fh.seek(chunk.off)
            original = fh.read(1)
            fh.seek(chunk.off)
            fh.write(bytes([original[0] ^ 0x01]))
        invalid = lmz.inventory_bundle(corrupt, strict=False)
        assert invalid["status"] == "invalid"
        assert invalid["validation"]["reason"]["code"] == "archive_invalid"
        try:
            lmz.validate_bundle(corrupt)
        except Exception as exc:
            assert "corrupt" in str(exc).lower() or "checksum" in str(exc).lower()
        else:
            raise AssertionError("corrupt archive was accepted")
        try:
            lmz.validate_bundle(good, expected_manifest_sha256="0" * 64)
        except lmz.BundleError as exc:
            assert exc.code == "expected_digest"
        else:
            raise AssertionError("wrong expected manifest digest was accepted")
        assert result["status"] == "verified"


def test_manifest_shape_rejections_are_structured():
    base = {
        "id": "shape", "version": "1", "source": {"revision": "fixture"},
        "license": {"classification": "generated", "redistribution": "allowed"},
        "entry_point": "a.payload",
        "consumer": {"engine": "fixture", "backend": "cpu", "version_range": "*"},
        "temporal_state": {}, "resources": {},
    }

    def expect(entries, code):
        manifest = dict(base)
        manifest["entries"] = entries
        try:
            bundle._normalise_bundle(manifest, require_artifact_fields=False)
        except lmz.BundleError as exc:
            assert exc.code == code, (code, exc.code)
        else:
            raise AssertionError(f"manifest rejection missing: {code}")

    expect([{"path": "a.payload", "role": "opaque"},
            {"path": "a.payload", "role": "opaque"}], "duplicate_path")
    expect([{"id": "same", "path": "a.payload", "role": "opaque"},
            {"id": "same", "path": "b.payload", "role": "opaque"}],
           "duplicate_identity")
    expect([{"path": "a", "role": "opaque"},
            {"path": "a/b", "role": "opaque"}], "path_conflict")
    expect([{"path": "a.payload", "role": "opaque",
             "dependencies": ["missing.payload"]}], "missing_dependency")
    expect([{"path": "a.payload", "role": "opaque", "size": 1,
             "dependencies": [{"path": "b.payload", "offset": 0,
                                "length": 10}]},
            {"path": "b.payload", "role": "opaque", "size": 1}],
           "invalid_range")


def test_plain_small_artifact_is_not_claimed_as_a_saving():
    with tempfile.TemporaryDirectory() as d:
        source = os.path.join(d, "source")
        _write(os.path.join(source, "random.payload"), bytes(range(251)))
        archive = os.path.join(d, "plain.lmz")
        result = lmz.create_bundle(source, archive, workers=1)
        entry = result["entries"][0]
        assert entry["representation"]["mode"] in ("plain", "codec")
        assert entry["compression_smaller"] is False
        assert result["archive_accounting"]["materialized_bytes"] == 251


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:
            failures.append(name)
            print(f"  FAIL  {name}: {type(exc).__name__}: {exc}")
            import traceback
            traceback.print_exc()
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
