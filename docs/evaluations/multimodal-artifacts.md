# Complete multimodal artifact bundles

This document records the first bounded MM-LMZ-01 implementation. It covers
lossless packaging and safe reconstruction of a deployable artifact tree. The
fixtures are generated mechanics evidence; they are not trained acoustic or
visual models, quality evidence, inference results, target-device evidence, or
energy measurements.

## Capability and gap matrix

| Area | Existing LMZ behavior | Bundle unit | Compatibility consequence |
|---|---|---|---|
| Byte archive | Version-7 byte-addressed members, tensor-aware codecs, refs and deltas | Reuses the existing archive and stores a `bundle` object in the existing tail manifest | Existing archives and APIs remain readable without a bundle object |
| File collection | Regular files are compressed from a directory | Bundle snapshots regular files through stable descriptors, rejects source links and special files, and records digest/length | No source link is recreated; hard-linked inputs become independent files |
| Metadata | Container member metadata and codecs | Deterministic schema 1 manifest with provenance, license, consumer, I/O, preprocessing, temporal state, resources, evaluation, entries and dependencies | Additive Python and CLI surface; no codec opcode or container major revision |
| Inventory | `lmz info` reports container members and codecs | `inventory_bundle` reports verified identity, dependencies, actual representation/stored bytes, archive accounting, workspace, entry point and consumer constraints | No inference engine import is needed |
| Materialization | Ordinary decompression creates files directly | `materialize_bundle` validates first, writes an owned no-follow staging tree, fsyncs, and publishes with atomic no-replace rename | Existing destinations are refused; incomplete staging is removed |
| ONNX | Self-contained `raw_data` initializers are typed by the planner | Bundle validation reads external-data declarations structurally, resolves safe bundle-relative sidecars, checks byte ranges and manifest closure | ONNX Runtime remains optional and is not imported |
| Small/opaque assets | General codec can choose stored chunks | Each entry records actual codec/representation and complete stored-cost decision; opaque files retain bytes without tensor claims | A plain or already-compact result is reported without a false saving claim |

## Public surface

```python
import lmz

result = lmz.create_bundle("fixture-tree", "fixture.lmz", manifest)
verified = lmz.validate_bundle(
    "fixture.lmz", expected_manifest_sha256=result["bundle_manifest_sha256"])
inventory = lmz.inventory_bundle("fixture.lmz")
materialized = lmz.materialize_bundle("fixture.lmz", "materialized-tree")
```

The short names `lmz.bundle.create`, `validate`, `inventory`, and
`materialize` are aliases. `BundleError.code` and its `as_dict()` result make
failure handling machine-readable. A digest supplied to validation proves
equality to the caller's expected canonical manifest; it does not authenticate
the publisher.

The command-line equivalents are:

```text
lmz bundle create SOURCE ARCHIVE [--manifest MANIFEST.json] [--json]
lmz bundle inventory ARCHIVE [--expected-manifest-sha256 DIGEST] --json
lmz bundle materialize ARCHIVE DESTINATION [--expected-manifest-sha256 DIGEST]
```

## Schema 1 example

The `bundle` object is stored in the existing archive tail manifest. Its
canonical identity is the SHA-256 of deterministic UTF-8 JSON with sorted keys
and compact separators. The implementation supplies explicit unknown values so
an omitted license, engine or measurement is not mistaken for a claim.

```json
{
  "schema": {"major": 1, "minor": 0},
  "id": "observer-fixture",
  "version": "2026.09",
  "source": {"uri": "fixture://observer", "revision": "generated-1"},
  "license": {"classification": "generated", "redistribution": "allowed"},
  "entry_point": {"path": "graphs/model.onnx", "kind": "graph"},
  "consumer": {
    "engine": "onnxruntime",
    "backend": "cpu",
    "version_range": ">=1.0,<2",
    "required_operators": ["Add"],
    "extensions": [],
    "export_provenance": "fixture"
  },
  "io": {"inputs": [], "outputs": []},
  "preprocess": {"identity": "fixed-normalize", "parameters": {}},
  "temporal_state": {
    "inputs": [], "outputs": [], "initialization": "zero",
    "reset": "on discontinuity", "cadence": "one frame",
    "discontinuity": "reset", "max_state_bytes": 0
  },
  "resources": {
    "decode_workspace": {"bytes": 4096, "measurement": "estimated",
                          "method": "fixture estimate"},
    "materialized_bytes": 0
  },
  "evaluation": {
    "input_identity": "fixture-input", "expected_output_identity": "fixture-output",
    "metric": "exact", "tolerance": 0
  },
  "entries": [
    {
      "id": "entry-…",
      "path": "graphs/model.onnx",
      "role": "graph",
      "decoded_bytes": 123,
      "sha256": "…",
      "dependencies": [
        {"id": "entry-…", "path": "graphs/weights.bin",
         "offset": 3, "length": 7}
      ],
      "consumer": {
        "engine": "fixture-engine", "backend": "opaque-cpu",
        "version_range": ">=1"
      },
      "representation": {
        "mode": "plain", "codec": "stored", "codecs": ["stored"],
        "codec_counts": {"stored": 1}, "stored_bytes": 123,
        "physical_payload_bytes": 123, "chunk_count": 1
      },
      "complete_stored_cost": 155,
      "compression_smaller": false
    }
  ]
}
```

Entry roles are `graph`, `weights`, `config`, `preprocess`, `vocabulary`,
`calibration`, and `opaque`. Unknown or compiled payloads use `opaque` and do
not receive invented tensor metadata. An entry may carry a `consumer` constraint
object with engine, backend, version range, and optional operator/extension
lists; these fields are part of canonical manifest identity and are descriptive
requirements, not execution permission. Dependencies are identity/path pairs
and may carry a non-negative bounded byte range. Duplicate identities,
duplicate paths, prefix conflicts, missing dependencies, cycles, unsupported
major versions, malformed ranges, and non-canonical portable paths are
rejected.

## ONNX external data

For a graph at `graphs/model.onnx`, an ONNX external location `weights.bin`
resolves only to `graphs/weights.bin`. Absolute paths, drive-letter paths,
UNC paths, backslashes, empty components, `.`/`..` traversal, missing sidecars,
conflicting metadata, overflow, truncation and ranges past the verified
sidecar are rejected. The graph declaration and manifest dependency range must
agree. Structural parsing does not prove that an inference engine can execute
the graph; no engine is imported by lmz.

## Safe collection and materialization

Collection copies regular files into an operation-owned snapshot through an
open descriptor, hashes while reading, and compares identity, size, mtime and
ctime afterwards. Source symlinks, devices, sockets, FIFOs and other non-regular
entries fail before an archive is published. Hard-linked source files are
copied as independent regular entries.

Materialization validates all archive bytes and digests before completion. It
creates a fresh same-filesystem staging directory, opens every output
component with no-follow flags, refuses pre-existing destinations, fsyncs the
files and staging directory, and uses a Linux atomic no-replace directory
publication operation. Any failure removes only the staging directory owned by
that invocation. A platform without the required atomic no-replace primitive
fails closed rather than claiming a weaker guarantee.

`unique_payload_bytes` is the union of physical chunk ranges across the whole
archive, so references shared by multiple entries are counted once. Each
entry's `representation.physical_payload_bytes` remains its own range cost;
`fixed_archive_overhead_bytes` includes the chunk table, manifest, footer, and
any gaps after subtracting the unique payload union.

## Generated fixture inventory and accounting

The reproducible fixture tests are in `tests/test_bundle.py` and generate all
bytes locally. They cover a mixed graph/weights/config/preprocess/vocabulary/
calibration/opaque tree, ONNX external data with an explicit offset and
length, a stateful acoustic-shaped declaration, a visual preprocessing
declaration, an opaque backend payload, a missing sidecar, hostile paths and
links, an existing destination, owned-staging cleanup, deterministic manifest
identity, and an already-compact plain result. The test runner reports the
per-entry SHA-256, decoded bytes, actual codec representation, stored bytes,
archive overhead and materialized bytes in the returned inventory; the final
handover report records the exact run values and hashes.

No runtime engine, real model, private recording, credential, sensor, device,
energy meter or WSL model weight is used by these fixtures. Engine opening and
output equivalence are therefore explicitly **NOT_RUN** for this unit.
