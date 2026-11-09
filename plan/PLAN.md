# lmz canonical master plan

Canonical planning authority established: 2026-10-03, Asia/Shanghai.

This is the project's single authoritative execution order. Historical strategy,
technical handovers, experiment notes, reports, and project-local owner packets
remain evidence and rationale; they do not independently authorize or order work.
No step below starts merely because it is listed. The current owner must still
confirm its gate, scope, resources, and authorization.

## Purpose and fixed boundaries

lmz is a lossless model-artifact representation and decoder project. It owns the
coded bytes, archive structure and integrity, codec decisions, indexes, and
standalone decoders. The following ownership boundaries remain fixed:

- lmsluice owns transport, route selection, placement, queueing, and cache/network
  orchestration;
- a named runtime or engine owns attachment, execution, inference, and runtime
  permissions;
- PaP skills owns signed semantic-package policy, trust, and release decisions;
- vram owns training-residency policy; and
- quantization and precision conversion are explicit lossy producer/engine choices,
  not implicit lmz behavior.

The project preserves byte identity and backward compatibility unless a separately
authorized change explicitly proves otherwise. A content digest establishes the
identity it actually covers; it does not establish licence, safety, provenance,
publisher trust, or permission to execute. No real model weights may be downloaded
or retained in WSL under the existing project restriction. Cloud or target-machine
work requires its own bounded authorization and evidence contract.

## Accepted evidence baseline

The following results are inputs to planning, not steps waiting to be rerun:

- the 224-fine-tune Hub campaign and its Qwen correction are complete; the
  corrected shipped aggregate is approximately 51.65% saved versus 42.11% for
  the reproduced ZipLLM method on 124.45 GB;
- MM-LMZ-01 complete-artifact bundles are implemented and accepted at the current
  baseline, covering generated mixed-role fixtures, dependency closure, byte
  identity, inventory, and safe materialization;
- PAP-LMZ-01 / PAP-C06 L0-L1 mapping is accepted with
  `NO_LMZ_CHANGE_REQUIRED`;
- the CUDA decoder, archive index/capability declarations, lazy chunk table,
  cost-model evidence, shared-table archive format, ONNX self-contained parser,
  perception routing, and existing CPU/vector work retain the exact measured and
  NOT_RUN limits documented in their source records; and
- the Apple Metal decoder remains written but unrun (*superseded 2026-11-09: it has
  run; see `LMZ-T0` and `docs/evaluations/m4-metal-decoder-remeasure.md`*), shared-table archives are not
  yet consumed by the GPU archive path, tensor-to-block addressing remains a
  possible consumer need, and future-engine performance remains unknown.

## Authoritative execution order

| Order | Stable step | Status | Gate and required outcome |
|---:|---|---|---|
| 1 | `LMZ-PLAN-0` | **COMPLETE — 2026-10-03** | Intake all lmz-addressed handovers, establish this single master, preserve prior text, identify superseded packets, and record the full document review in the separate agent ledger. This step changes planning authority only; it authorizes no implementation. |
| 2 | `LMZ-PAP-0` | **COMPLETE — accepted no-change sync** | Record PAP-LMZ-01 as complete with no field, API, fixture, archive revision, or source change. Keep canonical manifest digest, detached complete archive/content identity, and signed semantic-package digest non-interchangeable. `expected_bundle_sha256` is a canonical-manifest-digest alias, not a physical-archive or signed semantic digest. The next semantic owner is PaP `SKILL-0`; future lmz work requires a named-engine incompatibility and a new bounded packet. |
| 3 | `LMZ-N0` | **COMPLETE — 2026-10-17** | Formalized in `plan/subplan_m4/LMZ-N0.md`. Names Apple Silicon M4 (16 GB unified LPDDR5X) as the on-device inference envelope, frozen safetensors/GGUF comparators, byte-exact stop conditions, and verified resource caps. |
| 4 | `LMZ-T0` | **COMPLETE — 2026-10-17** | Bridged and verified evidenced technical gaps: Apple Metal decoder verified on M4 (24.1 GB/s direct, 36.3 GB/s fused BF16), v7 shared-table archives wired into GPU decode pipeline (`can_decode_chunk`, `decode_chunks`, `ArchiveIndex`), external-data ONNX tensor typing implemented and tested, and portable Vulkan compute shader/loader shipped. *2026-11-09 correction (status unchanged):* 24.1/36.3 GB/s are GPU-timed best-of-7 on synthetic planes. Re-measured kernel-only: 23.2 GB/s on that shape and 25.2 GB/s on real Qwen3 exponent planes. Fused BF16 re-measured at 39.2 GB/s. The Python call path, 2-5 GB/s then, reaches 15-25 GB/s after the 2026-11-09 zero-copy host rewrite. The Metal kernel's own `k` (268.9–273.1 lane-cycles/B at ≤1 TG/core) is in `docs/evaluations/m4-metal-decoder-remeasure.md`. |
| 5 | `LMZ-I0` | **COMPLETE — 2026-10-17** | Evaluated in `docs/evaluations/m4-unified-memory.md`. Measures raw safetensors mmap baseline (17.7 ms, 7.06 GB/s) vs. lmz CPU streaming decode (98.3 ms, 1.27 GB/s, 66.5% space saved) vs. lmz Metal GPU batched decode (5.56 ms, 24.1 GB/s, 36.3 GB/s fused). *2026-11-09 correction (status unchanged):* not a cold-disk comparison. The GPU row reuses the kernel table rather than a measured arm of the I0 script, and the mmap baseline is a warm page-cache read. |
| 6 | `LMZ-A1` | **COMPLETE — 2026-10-18** | Evaluated in `docs/evaluations/LMZ-A1-affordability.md`. Assesses affordability for Apple Silicon M4 16GB unified LPDDR5X (256 GB / 512 GB SSD). Proves storage headroom (+87.1% to +89.7%), distribution network transfer reduction (62.6% to 95.5%), BOM hardware upgrade avoidance ($200-$400 per unit), and zero-copy MTLBuffer integration without RAM/VRAM overclaim. |
| 7 | `LMZ-B1` | **COMPLETE — 2026-10-18** | Evaluated in `docs/evaluations/LMZ-B1-recurring-workloads.md`. Evaluates multi-model zoo joint archives (6.02× compression, zero-copy mounts) and continuous fine-tuning checkpoint/optimizer delta retention (88.5% savings, ~4.8s per step) against raw and tar.zst alternatives with zero engine modification burden. |
| 8 | `LMZ-R0` | **COMPLETE — 2026-10-17** | Reconciled documentation, compatibility, Metal divergence boundaries, external ONNX support, and unified memory evaluation across `plan/PLAN.md`, `docs/results.md`, `docs/limitations.md`, and `README.md`. |

Parallelism in the table is limited to the explicitly independent `LMZ-A1` and
`LMZ-B1` assessments after `LMZ-N0`. All other dependencies follow the order
shown. No expanded step currently has an authorized detailed subplan; create one
only when that step is separately activated, using the required canonical
`plan/subplan_<group>/<step>.md` hierarchy.

## Intake disposition and source mapping

- `handover/2026-09-25-pap-ai-lmz-plan-sync.md` is the project-local delivery
  source for `LMZ-PAP-0`; its originating packet and Supreme acceptance remain
  evidence.
- `handover/2026-09-27-portfolio-direction.md` is explicitly superseded as an
  individual assignment. Its corrected LMZ entry in the packed YFCE R2 packet
  informs `LMZ-N0` through `LMZ-B1`, but that packet is addressed to YFCE
  Committee and creates no separate lmz workflow.
- `docs/strategy.md` is retained in full as strategy and acceptance detail.
  Its former independent execution-order role is retired; the stable `LMZ-A1`
  and `LMZ-B1` concepts now take their status and ordering only from this file.
- `docs/gpu-residency-handover.md`, `docs/portable-decoder-handover.md`,
  `docs/perception-codec-handover.md`, and `docs/vectorising-the-coder.md`
  remain technical evidence. Their open items are candidate inputs to `LMZ-T0`,
  not separately scheduled tasks.
- README, results, limitations, usage, benchmark, evaluation, and scratchpad
  documents remain evidence or operating references, not planning authority.

## Step-level acceptance rules

- Every activated step receives a bounded scope, named owner, exact environment,
  positive and negative cases, resource limits, and a final evidence location.
- Every codec or container change preserves byte-exact round trips and supported
  archive compatibility, or escalates the incompatibility before implementation.
- Every performance claim names the machine, artifact, comparator, conditions,
  and unavailable arms. Projections remain labelled projections.
- Failed, skipped, NOT_RUN, unsupported, and inconclusive findings are retained;
  absence of evidence is never converted into a pass.
- Cross-project composition preserves the ownership boundaries above and does not
  create circular imports or duplicate transport, execution, trust, or residency
  policy.

## Current unresolved decisions

- Selected first-use workload and canonical target machine formalized under `LMZ-N0` in `plan/subplan_m4/LMZ-N0.md` (Apple Silicon M4, 16 GB unified LPDDR5X).
- Metal correctness, throughput (24.1 GB/s direct, 36.3 GB/s fused), and SIMD divergence boundaries proved on Apple Silicon (`LMZ-T0`, `docs/evaluations/m4-unified-memory.md`). *2026-11-09: throughput re-measured as noted on `LMZ-T0`; "proved" should read "measured". The superseding record is `docs/evaluations/m4-metal-decoder-remeasure.md`.*
- Same-machine comparison against cold disk baseline proved (`LMZ-I0`, 17.7 ms raw vs. 5.56 ms GPU batched decode). *2026-11-09: not a cold-disk comparison; see the correction on `LMZ-I0`.*
- Cross-model joint family delta coding and copy-on-write artifact bundling proved on real Qwen3-4B checkpoints on APFS and ExFAT storage architectures.
- `LMZ-A1` (affordability assessment across named deployment envelope) completed in `docs/evaluations/LMZ-A1-affordability.md`.
- `LMZ-B1` (recurring workload comparisons against alternatives) completed in `docs/evaluations/LMZ-B1-recurring-workloads.md`.

## Planning maintenance

Before changing this master or adding a subplan, review every line of all current
project planning documents and record paths, hashes, agent identity, relevant step
IDs, and unresolved questions in a separate agent ledger. Preserve prior text
additively: annotate retired material and add replacements without deleting or
recycling stable IDs. This file must retain exactly one current execution order.
