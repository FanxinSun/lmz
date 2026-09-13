# Development plan — multimodal artifacts and independent A/B tracks

Updated 2026-09-13: the September 10 independent-track decision now includes
the user's acoustic, visual, physical/spatial and language/multimodal direction.
This is the
current product-development direction, not a claim that the experiments below
have run. It supplements the existing technical specifications and measured
results; it does not change the archive contract or retire standalone users.

## 1. Decision and scope

Keep two independent assessments open:

- **A — affordability:** can lmz help deliver the broad YFCE experience at
  lower complete-system cost, without silently narrowing its capabilities?
- **B — niche positioning:** where does byte-exact model compression solve
  a particularly valuable customer problem? A niche may justify expensive
  hardware and does not have to meet Track A's price target.

Neither track is a prerequisite for the other. An affordable niche device is
an optional intersection, not the selected product. No niche, retail price,
memory capacity or hardware platform has been selected by this decision.

lmz remains an independently useful, lossless compressor/decompressor. YFCE
is one consumer, not a reason to make the format device- or industry-specific.
Keep one shared implementation; do not create affordability and niche forks.
Existing compatibility, correctness and supported-user obligations continue.

### Ownership stays unchanged

lmz owns representation, container/index layout, ref/delta semantics, codec
capabilities and decoders. Machine probing, route selection, queueing and
placement belong to lmsluice; runtime residency policy belongs above the codec.
Do not put device-cost policy, customer workflows or fleet management in lmz.
The companion repository's `docs/boundary.md` remains the detailed ownership
record; this plan does not move its boundaries.

## 2. Starting evidence and limits

- [Existing results](../README.md#what-it-saves) demonstrate byte-exact
  compression, directory deduplication and related-model savings. They do not
  demonstrate a cheaper finished device.
- The recorded 34.7% BF16 saving and 64.6% whole-directory saving have different
  denominators. The latter includes duplicate content; do not apply it to an
  already-deduplicated deployment bundle.
- The recorded Q4_K_M example saves 5.1%. Quantized language, speech and vision
  artifacts have different distributions: measure the deployment forms rather
  than transferring the BF16 result to them. See [limitations](limitations.md).
- Smaller storage or transfer representation does not automatically reduce
  steady-state inference RAM/VRAM. An ordinary decode still reconstructs the
  original tensors. Any running-memory claim needs a consuming runtime and
  end-to-end memory/performance evidence.
- Delta sources currently live in the same archive. This is not a general
  installed-base patch or over-the-air update protocol.
- CPU/CUDA interfaces and perception-container work already exist. External-data
  ONNX typing and GPU consumption of shared-table archives remain technical
  candidates, not automatic priorities. Consult their current implementation
  before treating old design notes as unfinished work.

## 2a. Next development — complete deployable artifacts

**Planning status:** direction updated; implementation and handover acceptance
are not established by this edit. Finish existing accepted work and its final
return before starting a newly dispatched unit. The completed Hub campaign and
its Qwen follow-up retain their own scope and evidence; do not restart them.

Support the four workload families by representing their complete deployment
inputs, not by turning lmz into an inference engine:

| Family | Artifacts to preserve | Important distinction |
|---|---|---|
| Acoustic | Graph/weights, sample-rate and feature-normalization settings, vocabulary/tokenizer, voice assets, streaming-state descriptors | Live PCM and recurrent state values belong to the runtime |
| Visual | Graph/weights, external data, input layout/color/normalization, labels, optional calibration | File size does not establish activation/workspace fit or frame deadlines |
| Physical/spatial | Estimator/policy artifacts where applicable, units/frames, calibration and observation/state schema | Passive estimation can use deterministic code; no actuator support is implied |
| Language/multimodal | Reasoning weights, tokenizer/config, encoders/projectors and dependency identities | Perception extends reasoning; it does not remove the language path |

### MM-LMZ-01 — first bounded implementation unit

1. Inventory existing container/planner capabilities before coding. Add an
   additive versioned bundle manifest: entry path/role, bytes/digest,
   plain-or-coded representation, dependencies, source/revision/license,
   consumer/backend constraints and input/output/preprocessing/state schema.
   Preserve opaque backend artifacts without inventing tensor semantics.
2. Cover ONNX graph plus **all** external-data sidecars and ancillary assets.
   Keep decode pure and file-level packaging separate from transport policy.
   No compulsory inference framework, hardware-specific format, new training,
   or TFLite/CoreML conversion program in this first unit.
3. Validate path containment, unsafe links/path races, duplicate entries,
   missing/truncated data, offset/length bounds and content identity. Use owned
   destinations and atomic completion. A digest is not publisher authority.
4. Preserve existing archives/APIs. Add tiny generated plain, mixed-assets,
   external-data and opaque fixtures, with byte-exact round trips and rejection
   tests. Escalate a breaking format change before implementation.
5. Report manifest/fixed overhead, payload savings, decode workspace and
   materialized bytes separately. Small/already compact models may stay plain;
   route selection remains outside lmz.

**Exit:** existing compatibility checks plus new bundle/corruption/path tests;
complete dependency reconstruction; concrete interoperable manifest examples;
documented optional engine checks and every unavailable arm. A generated graph
proves mechanics, not trained-model accuracy. Result destination:
`docs/evaluations/multimodal-artifacts.md` with linked fixtures/raw evidence.

### Following integration gates

After this unit is finished, reviewed and the executor is ready, use the same
approved trained ASR and visual artifacts with normal/plain/coded loading in a
named consumer. Check exact artifact restoration **and** pinned-engine output
equivalence/task metrics. Extend later to TTS/reasoning and passive spatial
state. Compare actual quantized deployment forms; BF16 ratios do not transfer.
These are subsequent scoped units, not automatic campaign continuation.

The existing project restriction remains: real-model work on authorized Colab
ephemeral storage, **no WSL model-weight downloads or retention**. Local first
unit fixtures are tiny/generated. Preserve environments and scientific limits.
Shared YFCE coordination lives in `docs/multimodal-development-plan.md` in the
parent project; this local plan remains readable without that checkout.

Bundle mechanics support both evaluations below, but A and B remain independent
decisions. Byte savings alone cannot prove a cheaper RAM tier or customer value.

## 3. Track A — affordability without a niche prerequisite

### Question

Can lmz reduce storage provision or distribution cost for a broad device
workload while preserving the exact deployed model bytes and the application's
capability/latency baseline?

### Next development work: LMZ-A1 — deployment-bundle assessment

1. Freeze a reproducible manifest of representative deployed language, speech
   and vision artifacts, including quantized forms and a combined bundle.
   Record revisions, hashes, dtypes, containers, unique tensor bytes, duplicate
   content and required ancillary files. Reuse local artifacts first; a small
   proxy is acceptable if explicitly labelled, not represented as the full
   broad-device acceptance test.
2. Compare the exact same artifacts as their ordinary deployment form, with
   a relevant general-purpose compressor, and with current lmz. Separate
   per-file coding, directory deduplication and related-version savings. Include
   an already-deduplicated baseline rather than only redundant Hub directories.
3. Record archive bytes including metadata, encode/decode time, peak working
   memory, decoder availability and CPU cost. Record energy where measurable;
   otherwise mark it unmeasured. Coordinate end-to-end loading measurements
   with lmsluice without absorbing its scheduling responsibilities.
4. Cover relevant resource envelopes: no CUDA, limited RAM, slower storage or
   links, and unified memory where accessible. A throttled workstation is a
   simulation, not a measured low-cost CPU. Preserve machine parameters and
   distinguish measured, projected and unavailable results.
5. Translate savings into separate outcomes: **BOM reduction**, **storage
   headroom**, **transfer/operating cost reduction**, or **no useful saving**.
   A BOM claim must identify an actually cheaper storage/system configuration
   at a stated volume and dated price basis, while retaining update space and
   meeting the same application requirements. Archive percentage alone cannot
   establish that claim.

### Acceptance and next decision

Write the evidence to `docs/evaluations/affordability.md`, with a reproducible
manifest and linked raw results. Require byte-identical round trips for every
tested artifact, explicit baseline/decode settings, full cost boundaries and
negative as well as positive results. Hardware cost includes any required host;
cloud savings/costs and runtime memory are separate from archive size.

Report Track A independently as supported for a named envelope, unsupported,
or inconclusive. No BOM reduction is a valid result even when storage or
transfer savings remain useful. Select subsequent codec/format optimization
only against a demonstrated bottleneck; do not substitute a narrower task set
or a different quantization level and count that saving as lossless compression.

## 4. Track B — niche value without an affordability ceiling

### Question

Which repeatable customer workload benefits enough from exact restoration,
related-model storage or distribution efficiency to adopt lmz?

### Next development work: LMZ-B1 — customer/workload comparison

1. Compare at least two candidate workloads before selecting a first lane:
   specialized multi-model releases, collections of customer/model variants,
   or checkpoint retention are starting hypotheses, not chosen markets.
   Record the user/buyer, existing workflow, alternative, recurring pain,
   integration effort and access to representative evidence.
2. For an accessible candidate, reproduce the complete artifact lifecycle:
   initial collection, another version/variant, and restoring one required
   member. Count base dependencies, retained versions, reconstruction memory
   and total bytes that must be available or transferred. Do not describe the
   compressed delta alone as the complete deliverable.
3. Compare with the customer's real alternative, including existing dedup or
   delta storage where applicable. For a Hub-oriented case, chunk-deduplicated
   transfer is a baseline to assess, not a capability assumed absent.
4. Evaluate repeatability and adoption cost separately from compression ratio:
   how often the problem occurs, who controls publication, deployment effort,
   support burden and evidence of customer value. A benchmark is not proof of
   demand; unavailable customer access remains an explicit evidence gap.

### Acceptance and next decision

Write a separate `docs/evaluations/niche.md` with the candidate comparison,
baseline, measured benefit, failure cases, source-dependency constraints and
customer evidence or its absence. Record pursue/defer/reject per candidate.
Track B may succeed on premium hardware even if Track A yields no BOM saving.
No container support, distribution protocol or vertical-specific integration
is selected merely because it appears on the candidate list.

## 5. How the next development cycle uses this plan

1. Check current code and release state before reopening older technical
   backlogs. Fix correctness/compatibility regressions when required; this
   strategy does not postpone those obligations.
   For the multimodal extension, use Section 2a's bounded bundle unit first;
   do not substitute a fresh large checkpoint campaign for the missing interface.
2. Define the A1 artifact manifest and B1 candidate comparison independently.
   Either can progress while the other's hardware or customer evidence is
   unavailable. A scheduling order is not a strategic dependency.
3. Use existing interfaces for the first measurements. If a missing metric or
   decoder capability blocks one, specify the smallest required change with
   its owning repository, reproducer and validation before implementing it.
4. Keep separate conclusions and next actions for A and B, even if they share
   raw data. Choose subsequent engineering from those findings, not from GPU
   headline throughput or assumptions about an unannounced competing device.

For any resulting code change, retain byte-exact behavior and archive
compatibility, run the relevant existing tests and add a focused regression
test. GPU-only evidence cannot establish CPU/unified-memory support. Do not
change existing environments, fetch large model sets, purchase hardware or
contact customers implicitly through this roadmap; use the user's standing
environment, network and authorization rules. This plan updates direction;
it does not start a benchmark campaign or authorize unrelated changes.

Any resulting user-operated validation must follow the installed all-in-one-probe
skill: an actual no-argument `.sh` with complete approved tracked inputs and
internal preparation/logging/collection, published to the established task Git
remote/branch, verified from a clean checkout and exact remote tree. The owning
executor runs the completion checker before giving launch instructions. No
manual download/copy or pasted-script substitute; existing Colab/WSL and model
licensing limits still control. Report all failures, skips and inconclusive
results; ordinary documentation checks are not model or target validation.
