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
- the Apple Metal decoder remains written but unrun, shared-table archives are not
  yet consumed by the GPU archive path, tensor-to-block addressing remains a
  possible consumer need, and future-engine performance remains unknown.

## Authoritative execution order

| Order | Stable step | Status | Gate and required outcome |
|---:|---|---|---|
| 1 | `LMZ-PLAN-0` | **COMPLETE — 2026-10-03** | Intake all lmz-addressed handovers, establish this single master, preserve prior text, identify superseded packets, and record the full document review in the separate agent ledger. This step changes planning authority only; it authorizes no implementation. |
| 2 | `LMZ-PAP-0` | **COMPLETE — accepted no-change sync** | Record PAP-LMZ-01 as complete with no field, API, fixture, archive revision, or source change. Keep canonical manifest digest, detached complete archive/content identity, and signed semantic-package digest non-interchangeable. `expected_bundle_sha256` is a canonical-manifest-digest alias, not a physical-archive or signed semantic digest. The next semantic owner is PaP `SKILL-0`; future lmz work requires a named-engine incompatibility and a new bounded packet. |
| 3 | `LMZ-N0` | **NEXT DECISION; HOLD — NOT AUTHORIZED** | Before any new campaign or implementation, name one recurring workload or consumer, its target machine, artifact forms, frozen comparator, success/failure measures, first-use path, resource cap, stop conditions, and evidence owner. A generic corpus demonstration is insufficient because the completed corpus already establishes broad compression value. No device, engine, niche, customer, price, or schedule is selected here. |
| 4 | `LMZ-T0` | **CONDITIONAL; HOLD — NOT AUTHORIZED** | After `LMZ-N0`, determine whether current lmz already meets the named need. If not, authorize exactly the smallest evidenced technical gap. Candidate evidence—not preselected work—includes running/repairing the Metal decoder for an Apple target, wiring shared-table archives into the GPU decoder, tensor-to-block addressing for a named consumer, external-data ONNX typing, or the measured vector-decoder prototype. Preserve whole-entry/plain materialization as a valid baseline and reject unrelated format expansion. |
| 5 | `LMZ-I0` | **CONDITIONAL; HOLD — NOT AUTHORIZED** | With a named consumer and any required `LMZ-T0` result, compare the same artifact through plain and lmz-backed paths. Measure byte identity, ratio, access/recovery cost, throughput, peak memory, copies/synchronization, and integration cost on a labelled target. Runtime output equivalence and task quality belong to the consuming engine; lmz must not claim them from archive evidence. |
| 6 | `LMZ-A1` | **CONDITIONAL INDEPENDENT TRACK; HOLD — NOT AUTHORIZED** | Assess affordability for a named deployment envelope using exact deployed language, acoustic, visual, and combined artifacts. Separate archive, deduplication, related-version, storage-headroom, transfer-cost, and actual BOM outcomes. This track may proceed independently of `LMZ-B1` after `LMZ-N0`; it must not infer RAM/VRAM or device-price savings from archive percentage. |
| 7 | `LMZ-B1` | **CONDITIONAL INDEPENDENT TRACK; HOLD — NOT AUTHORIZED** | Compare at least two accessible recurring workloads—model-family archives and checkpoint/optimizer retention are evidence-backed candidates—against their real alternatives and integration burden. Record demand evidence or its absence. This track may proceed independently of `LMZ-A1` after `LMZ-N0`; benchmark advantage alone is not customer value. |
| 8 | `LMZ-R0` | **CONDITIONAL; HOLD — NOT AUTHORIZED** | Only after the relevant preceding step is accepted, reconcile documentation, compatibility, negative results, target-labelled claims, and release evidence. A release, tag, publication, major format change, merge, or broad campaign requires separate authority; no such action is granted by this plan. |

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

- No recurring first-use workload or named consumer has been selected for
  `LMZ-N0`.
- No target machine, resource budget, effort cap, customer-access path, or
  acceptable integration cost has been selected.
- Metal correctness and throughput remain unproved on Apple silicon.
- Same-machine comparisons with current alternatives, practical recovery/access
  cost, future-engine compatibility, end-to-end task quality, demand, and product
  affordability remain unproved or inconclusive.
- None of these unknowns authorizes work; they are inputs to the next bounded
  decision.

## Planning maintenance

Before changing this master or adding a subplan, review every line of all current
project planning documents and record paths, hashes, agent identity, relevant step
IDs, and unresolved questions in a separate agent ledger. Preserve prior text
additively: annotate retired material and add replacements without deleting or
recycling stable IDs. This file must retain exactly one current execution order.
