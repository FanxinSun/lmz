# Perception models — what lmz needs for the machine's eyes and ears

*The pieces of work in lmz between an LLM compressor and the perception set
of blueprint D15, why each is load-bearing, what is already measured about
it, and where the boundary is between lmz and the things that consume it.
Both expected ratio items dissolved under measurement and are written down
so they stay dissolved; what turned out to be real is a coder-routing
default with the GPU decoder on the other end of it, a container, and a
fixed cost the GPU handover already wants removed. Written for whoever
picks this up next.*

**Status.** Three of the four items have landed, and the fourth changed
shape under measurement. What is done: **int8 now routes to rANS** (§1), so
the GPU decoder can read it; **ONNX is parsed** (§2), worth 4.3 measured
points; the `limitations.md` INT8 claim is **corrected**, and `lmz compress`
now says when zstd is missing. What remains is the shared table (§4), whose
design the measurements changed — see the note there. Original measurements
are E5, `experiments/e5_perception_codec/`, in the parent repository — probe,
conditions, raw JSON; the numbers added while implementing are marked as
such.

## Why this exists

The AI PC grew eyes and ears (blueprint D15/D16): a detector that watches,
a vision encoder that looks, streaming ASR that listens, acoustic-event
models always on. E4 measured that set running beside decode; what it did
not touch is how those models are **stored, shipped, swapped and updated**
— and all of that is the archive's job. Perception models are app-class
residents (M12): blocks in the standard archive, demand-paged, swapped by
descriptor when a watch is granted, delta-updated when a variant lands.
The residency layer's swap path and the appliance's descriptor ring both
read the same container, which is lmz's.

What changes for the codec is the population. lmz's ratios and its
treatment decisions were calibrated on LLM checkpoints — BF16 weights,
GGUF block quants. Perception arrives as **fp32 that serves as fp16 or
per-channel int8**, in conv-and-transformer bodies, increasingly inside
**ONNX** containers. E5 put real members of each class through lmz exactly
as it ships and measured the entropy of every treatment a
perception-aware codec might add. The verdict: the ratios are already
almost all collected; the work is routing int8 onto the coder the GPU can
decode, parsing the container perception ships in, and landing the shared
table that small archives now need for a second reason.

## What already works, measured — and should be said out loud

| model (native form) | size | lmz as shipped | speed |
|---|---|---|---|
| GLM-4V vision tower slice, **bf16** (579M params) | 1159 MB | **31.8%** | 993 MB/s |
| whisper-tiny, ASR, **fp32 as published** | 151 MB | **57.3%** | 555 MB/s |
| SSDLite320 detector, fp32 | 14 MB | 14.4% | 154 MB/s |
| MobileNetV3-small (sentinel class), fp32 | 10 MB | 17.0% | 121 MB/s |

Two rows are the LLM story repeating on new data, and they are worth a line
in lmz's own README. Trained **BF16 vision weights** take the field-split-
plus-conditioning path built for Llama and land within three points of the
Llama number — the home path works on vision out of the box. And
**whisper-tiny ships fp16-trained values in an fp32 container**, which is
bge-m3's case again: two byte planes collapse and more than half the file
vanishes, byte-exact. Audio checkpoints in the wild are very often exactly
this shape. No work item here — only the claim, which is now measured.

**int8 ratio belongs on this list too**, which was not the expectation.
`limitations.md` says raw INT8 is "detected and **stored unchanged**"; the
tree does no such thing — every chunk under 7.98 bits/byte is priced and
coded, and real per-channel int8 checkpoints through the shipping CLI land
at their entropy bound (E5b, round-trips verified):

| per-channel int8 file | order-0 bound | lmz | zstd −19 |
|---|---|---|---|
| GLM-4V vision slice, 580 MB | 25.7% | **25.2%** @ 1.2 GB/s | 25.7% @ 6 MB/s |
| whisper-tiny, 38 MB | 16.2% | **16.5%** | 16.5% |
| SSDLite detector, 3.7 MB | 17.3% | **18.6%** | 18.9% |

The planes are peaked (5.9–7.2 bits/weight) because a per-channel scale
maps each filter's maximum to 127 and leaves the Gaussian body
concentrated — unlike GGUF Q8_0's genuinely flat 7.64, where the "nothing
left to code" reading was made. The `limitations.md` line is stale and
should be corrected: the second time a claim there has trailed the tree,
the GPU path being the first.

## 1. int8 — not a ratio item, a routing item: zstd strands it off the GPU — **done**

The file numbers above hide **which coder got chosen**, and it matters to
exactly one consumer — the one this whole programme is for. A 1-byte
element has no plane split, so an int8 tensor takes the general path:
zstd runs first, and lmz's own rANS is priced only if zstd failed to dent
the stream by 1/32 (`codec.py:376`). On int8 weights zstd never fails that
badly, so rANS is never tried. On one 56 MB int8 plane, lmz's own kernels
over identical bytes (E5c):

| coder | saved | speed |
|---|---|---|
| the general path's choice — **METHOD_ZSTD** | 11.3% | 801 MB/s |
| lmz's rANS, called directly | **12.1%** | 663 MB/s |
| zstd −19 | 11.3% | 6 MB/s |

The 0.8 points are pleasant and not the point. **`lmz.gpu` decodes rANS
streams only.** An int8 chunk coded METHOD_ZSTD can never ride the
111–418 GB/s batch decoder or a fused path — it decodes on a CPU at
~2 GB/s forever. Coded-in-VRAM residency for a perception model — the
watch tier's swap path, and the coded tier `vram/` measured at 948 GB/s
on BF16 — quietly stops existing for the one dtype every deployed
perception model uses.

The fix is one routing change, not a coder: for int8 weight planes, price
rANS **alongside** zstd instead of behind its failure gate, and keep the
winner with a preference for rANS at equal size — the way delta coding
already decides by encoding a sample both ways. On this evidence it costs
nothing at encode time and pays twice: slightly smaller, and
GPU-decodable. Per-channel scale vectors ride along as their own small
plane, exactly as k-quant scale fields already do.

**Landed, and the gate is the cost floor rather than a threshold.** rANS is
now priced whenever `_rans_cost` — a floor taken from the histogram already
in hand — comes in under zstd's actual output, and ties go to rANS since the
bytes are equal and one of them is GPU-decodable. The general path is not
handed a histogram, so one is counted there; it is a single pass, and it
buys skipping a second full compression wherever the floor rules rANS out.

On a real 38 MB per-channel int8 whisper checkpoint, page-mapped: **272 of
576 chunks and 47% of payload bytes** now land on rANS, where none did
before, byte-exact both ways. The remaining zstd chunks are honest wins —
274 of them the floor never priced, and on the 30 contested ones zstd was
better by 0.058%, which is real repetition a symbol coder cannot see. Where
lmz *could* be pushed further is forcing rANS on those too: it would cost a
fraction of a point of ratio to make the whole archive GPU-decodable. That
is a residency-layer policy call rather than a compressor default, and it is
not implemented.

## 2. ONNX — the container perception actually ships in — **done**

lmz parses safetensors, GGUF and PyTorch zip. Perception deploys through
**ONNX**, which lmz does not parse, so every initializer would travel as
opaque 1-byte regions today. E5's control measures what that costs: the
same detector bytes save **11.0% inside a parsed container and 7.9% as a
raw blob** — layout awareness is worth 3.1 points before anything clever
happens, because without the element width the byte-position planes cannot
phase-align, and fp16 exponents smear across positions.

The parse is small and safe by the standard lmz already set for PyTorch
`.bin` (opcode scan, nothing executed): an ONNX file is one protobuf walk —
`graph.initializer[]`, each with dtype, dims and `raw_data` at a known
offset; the external-data variant stores tensors in a flat sidecar file
that is even simpler. The layout parser feeds the existing splitter
machinery; no coder changes. TFLite is the same idea one format later, and
worth doing only if a consumer shows up with one; CoreML stays out.

This item also buys what §5 needs: a container lmz can parse is a container
whose tensors the chunk table can *address*.

**Landed, and worth more than the control suggested: 4.3 points, not 3.1.**
Re-measured on a real fp16 detector (the ssdlite weights exported as ONNX,
7.0 MB, 406 initializers) against the identical bytes with the parser
switched off: **11.90% parsed against 7.60% opaque**. BF16 initializers
reach `KIND_BF16`, so ONNX gets the conditioned bf16 path for free.

Two things worth carrying forward. The walk goes **against the file, not a
buffer** — an ONNX model with internal weights is one protobuf over the
whole file, so reading it in to plan it would hold a multi-gigabyte model in
memory; `raw_data` is located by its length prefix and seeked over. And
because **ONNX has no magic number**, this parser is offered every file the
others declined and `probe` runs on everything compressed, so every
malformed shape has to degrade to opaque bytes rather than raise: truncation,
length prefixes past the file, the deprecated group wire type, unknown
dtypes, payloads that are not a whole number of elements. All are tested.

**External-data ONNX is still opaque**, and that is the one piece of this
item left undone. Those initializers carry no `raw_data`, so there is
nothing in the file to type; the sidecar is flat and codes well as raw
regions, but typing it needs the sidecar name resolved against the model's
directory — a multi-file question rather than a parser one.

## 3. fp16 — the item that dissolved, written down so it stays dissolved

fp16 is perception's serving dtype, F16 chunks take the plain byte-plane
path (`planner.py:229` → KIND_BYTES), and the obvious move is the BF16
treatment: split on the float's own fields. **Measured, it buys nothing:**

| fp16 treatment, bits/element | ssdlite | whisper-tiny | mobilenet |
|---|---|---|---|
| byte planes, order-0 — today's path | 13.80 | 13.39 | 13.43 |
| field split (1+5+10), order-0 | 13.80 | 13.47 | 13.45 |
| either split, conditioned | 13.41 | 13.35 | 13.20 |
| joint bound, saved | 16.2% | 16.6% | 17.5% |

The mechanism: BF16's byte boundary slices through its 8-bit exponent —
that is what made the field split worth 0.065 bits there. **fp16's 5-bit
exponent sits entirely inside the high byte**; the boundary slices two
mantissa bits instead, and mantissa is noise on either side of any cut. The
splits measure identical to 0.01 bits on every model. There is no fp16
field-split item.

What survives is smaller and cheaper: **conditioning** the low byte on the
high — the treatment GGUF fp16 *scale fields* already get (GRP_COND) —
is worth 0.3–2.4 points at the bound on whole-fp16 tensors, and lmz's real
archives sit 0.9–5.2 points off the bound today. Most of that gap on small
files is overhead, which is item 4's, not a coder's. The honest ceiling on
lossless fp16 is 16–18% total; a third of a Q4 LLM's story, and worth
having mostly because perception models are small enough that the work is
too.

## 4. Small archives — the fixed costs the sentinel classes cannot amortise

A sentinel model is 3–10 MB; a watch-tier detector 14–90. At those sizes
the archive's fixed costs stop hiding:

- the **64 KiB page-mapped form** — the one the residency layer actually
  reads — costs **4.6 points** on a 10 MB fp32 model (17.0% → 12.4%) and
  2.7–3.4 points on the int8 files: alignment padding plus a 516-byte
  table per chunk;
- the plain form on a 7 MB fp16 file sits 5.2 points off its entropy bound
  for the same reason.

The fix is already specified: the **shared frequency table**, item 1 of
`gpu-residency-handover.md`, which was justified there as 3.8× of GPU
decode speed and shown to make files *smaller* on real planes. Perception
adds the second justification: on small archives the per-chunk table is a
material fraction of the payload, and sharing it returns most of these
points. One format option now carries both arguments; it should land
first. What needs **no** work: tensor coalescing already sweeps a
detector's hundreds of sub-64 KiB tensors into shared chunks (476 tensors,
440 under 64 KiB, 1.5 MB — handled), and should not be touched.

### The coder primitives are landed; the format is not, and its design changed

`lmz_rans_table`, `lmz_rans_encode_shared` and `lmz_rans_decode_shared` are
in the kernel with Python bindings, verified, and exposed as
`kernels.rans_table` / `rans_encode_shared` / `rans_decode_shared`. The
shared form is the ordinary form with the header lifted off, so **prepending
the table to a headerless stream yields a stream `lmz_rans_decode` reads
unchanged** — that is the oracle the tests use, and it means there is no
second decoder to keep in step. Encoding refuses a symbol the table gives
zero frequency rather than emitting a stream that decodes to different
bytes; both directions refuse a table that does not sum to the probability
scale.

**The measurement that must not be lost: sharing one table across a whole
archive is worse than what lmz does today.** Both handovers sketched the
table as an archive-wide or per-chunk object. Measured on real perception
models at 64 KiB blocks:

| one table for… | mobilenet fp32 | ssdlite fp32 |
|---|---|---|
| per stream — today | 13.93% | 11.97% |
| the whole archive | 7.57% (**−6.36**) | 5.81% (**−6.17**) |
| one chunk's planes | 7.04% (**−6.89**) | 6.01% (**−5.97**) |
| **one plane kind, across chunks** | **16.45% (+2.52)** | **13.49% (+1.52)** |

fp32 planes have unrelated distributions — an exponent plane and a mantissa
plane share nothing — so one table fits none of them and every stream pays
for the mismatch. The win is real only when the table is shared **across
chunks within a plane kind**; whisper fp32 gains +2.24 points the same way.

So the format work is **archive-level tables in the manifest, keyed by plane
kind**, not a per-chunk or per-archive table. That means `decode_chunk`,
which is deliberately context-free today, has to be handed the archive's
table set — five call sites in `api.py`, `codec.py` and `cli.py`. It is the
one piece of this handover that is a real format change, and it is the
reason it did not land alongside the other three.

## 5. What the residency layer will ask of a perception archive

Mostly re-affirmations of the GPU handover, with the perception numbers
attached:

- **Tensor-level addressing** (item 4 there) is the watch tier's swap path:
  a grant arrives, the runtime asks "where are this detector's blocks",
  and today the answer routes through a mounted filesystem. The manifest
  the compressor already knows at encode time is the answer.
- **The batch ABI fits small models.** A 43 MB int8 detector is ~650
  chunks × 8 states ≈ 5,000 lanes — a real batch, decoded in ~0.4 ms at
  the shipped kernel's 111 GB/s and ~0.1 ms with the shared table. The
  watch tier's arming latency is not codec-bound, and `_dev`'s
  caller-provided output pointer is exactly what a CUDA-graph consumer
  (E4's single-launch contract) needs. Keep both as they are.
- **Delta coding is the update path.** Task-tuned variants of one detector
  base — the app-bundle case — are delta's home ground; the
  same-archive constraint suits bundles that travel as one file. Nothing
  new; worth a sentence in lmz's docs when perception archives exist.
- **Per-chunk checksums stay on.** A silently wrong LLM weight says
  something strange; a silently wrong detector weight *acts* (D16). The
  residency engine's worst outcome has not changed, and neither should
  this default.

## Speed, so it is on record

"High speed" was the ask and it is already met; the items above are where
the work is. CPU compress ran at 60 MB/s–1.2 GB/s wall in E5 (startup
included; the small files pay the startup, the 580 MB int8 file runs at
1.2 GB/s where zstd −19 manages 6). CPU decode at 1.4–1.9 GiB/s beats
every storage tier the sentinel classes live on; the GPU batch kernel's
111–418 GB/s makes any perception swap sub-millisecond against a
28.8 GB/s link — *provided the chunks are rANS*, which is item 1's point.

**One trap found while measuring, worth a line of code.** An interpreter
with no zstd binding silently degrades every general-path chunk to
deflate — this box's ML env (python 3.11, no `zstandard`, no stdlib zstd)
did exactly that, and only `lmz doctor` says so. A perception box is very
likely to run lmz from such an env. A one-line notice at compress time
when the entropy backend is deflate would have saved a re-measurement
here, and will save someone a mysteriously fat archive later.

**Done.** `lmz compress` now prints it under the ratio, which is where it
will be read — `doctor` is not run by someone who does not yet know
anything is wrong. On the int8 whisper file the gap it explains is 16.7%
against 16.0%.

## Where the boundary is

| | whose |
|---|---|
| quantisation itself — PTQ, QAT, which models go int8 | the toolchain (M2's lossy half); lmz codes what it emits, losslessly |
| when a model loads, priority lanes, graph capture | the runtime — E4's contract, not the codec's |
| sentinel/watch policy, event classes | `os/spec/perception.md` |
| the coded stream, its planes and tables, the container, addressing | **lmz** |
| the int8 decision rule, the ONNX walk, the shared table | **lmz** — items 1, 2, 4 above |

The dividing line from the GPU handover holds unchanged: lmz produces
bytes and turns them back into bytes as fast as anyone; it does not decide
when, and it does not decide what got quantised.

## How to measure

`experiments/e5_perception_codec/` re-runs everything against a scratch
directory of checkpoints: `probe.py` (ship forms, fp16 entropy),
`int8_probe.py` (real I8 safetensors through the CLI, zstd beside),
`int8_coder.py` (lmz's own kernels priced on one plane), `fetch2.py`
(rebuilds the checkpoint set — ~175 MB of downloads, cached at
`~/.cache/yfce-e5-ckpt/`, plus a free slice of the GLM-4V vision tower
already on this disk; sized for a metered connection, and the one gap
worth closing on an unmetered one is a real audio-trained conv model in
place of MobileNet as the AED stand-in). Conditions and the departures
that matter — fp16 by cast, int8 quantised here not by a vendor tool,
tables uncharged in entropy figures, and the interpreter trap above — are
`CONDITIONS.md` there; every number in this file traces to
`results.json`, `results_int8.json` or `results_int8_coder.json`, or is
labelled as the earlier RT-DETR pass.
