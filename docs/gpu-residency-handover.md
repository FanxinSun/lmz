# GPU residency — what lmz has to ship for it

*The pieces of work in lmz between a fast kernel and a residency layer, why
each one is load-bearing, what is already measured about it, and where the
boundary is between lmz and the thing that consumes it. Three of the four
have landed; this says what they cost and what they left. Written for whoever picks
this up next, so that none of it has to be rediscovered.*

[← back to the README](../README.md)

## Why this exists

The decoder is already written. `scratchpad/gpu/cuda/gpu_fused.cu` decodes
lmz's own rANS at **399–418 GB/s on an RTX 5080** — whole-BF16 in one kernel,
exponent rANS plus raw sign/mantissa plus merge — verified byte-identical to
`lmz_rans_decode` over 936 MB and 1.87 GB of real planes, and 1.46× faster
than plain safetensors end to end from cold disk to VRAM. `gpu_rans10.cu`
carries the whole V0..V5 ladder and the ablations that attributed the time.

The kernel is now also *shipped*: `lmz.gpu` is in the package, and item 2
below records what that took. The index can now be opened at the size a
residency layer opens it at — item 3, which was the precondition the rest
waited on. What still does not exist is a layer that can put the kernel to
work: the format option the fast path needs is now written (item 1), but
nothing can yet ask an archive where a tensor's blocks are. That is one piece
of work, and it is not research.

The consumer is a residency engine that keeps weights coded in VRAM, in
page-locked host RAM and on NVMe, and decodes them on the GPU. Its charter,
its roofline and the measurements behind it are in the parent repository at
`vram/`. **The one number it turns on:** a GPU rANS decoder runs at 399 GB/s
and the PCIe link into it at 28.8, so compression on that path is free by
14×, and every point of lmz's ratio becomes a point of bandwidth.

That engine is not lmz's job. This is the part that is.

## 1. The shared frequency table, as a format option — **done**

**Measured, and it is free in both directions.** A table shared across chunks
instead of one per chunk is **1.7× of the GPU decode speed** — a per-stream
16 KiB table means 32 lanes in a warp thrash 512 KiB of L1, a shared table
lives in shared memory — and on real data it also makes the file *smaller*:
325.0 MB against 336.1 MB over 936 MB of exponent plane, because the
516-byte-per-chunk header costs more than the cross-entropy the shared table
gives up.

That 1.7× was the ladder's, against a per-stream table read from global
memory. Against the shipped kernel, which does the best that can be done with
a per-chunk table — a 5.6 KiB narrow LUT per group, in shared memory — it is
**3.8×: 111 GB/s against 418** on the same 936 MB. The gap got wider, not
narrower, because the narrow table buys its size with a second dependent
shared load on the critical path. Item 2 shipped without this and is useful
without it; this is what it is leaving on the table.

**And it is already backward compatible.** `scratchpad/gpu/prep_shared.py`
proves the property that makes this cheap: prepending the shared header turns
a shared-table stream back into an ordinary lmz stream, so the *existing*
decoder reads shared-table data with no change at all. The encoder is
`scratchpad/gpu/shared_enc.c` — lmz's own coder with the per-stream header
lifted out.

What is left is promotion, not invention: a format flag, the encoder path
choosing between per-chunk and shared on measured cross-entropy rather than
by assumption, and the round-trip in the test suite. This is the one item
that touches the format, and it is the one that should land first, because
everything else can be built against a stream that already exists.

**The format side has now landed too, and what it cost is in
[portable-decoder-handover.md](portable-decoder-handover.md) §1** — including
three things this section did not anticipate: sharing has to be decided per
*plane* rather than per chunk, the decision has to exclude planes the encoder
stores as noise (counting them predicted +1.9 points on a real detector and
delivered nothing), and a shared-table chunk needs its own native plane
decoder, because `lmz_decode_planes` reads a per-stream header that is no
longer there. `CODEC_SPLIT_ST` is the byte-plane split in that form;
`--shared-tables` writes it. **The GPU kernel does not read it yet**, which
is where the 3.8x below still sits.

**The coder side is now promoted, and the format side has a design it did
not have here.** `lmz_rans_table`, `lmz_rans_encode_shared` and
`lmz_rans_decode_shared` are in `lmzcore.c` with Python bindings, sharing
the decode body with `lmz_rans_decode` rather than copying it, and refusing
the two ways a shared table can silently corrupt: a symbol the table cannot
represent, and a table that does not sum to the probability scale.

What changed is the *scope* of the sharing. This section assumed one table
per archive. Measured on real models, that is **six points worse** than a
table per stream, because fp32 planes have unrelated distributions and one
table fits none of them; a chunk-local table loses the same six. Sharing
within a **plane kind, across chunks** is what wins — +1.5 to +2.5 points on
perception models. So the manifest needs a table *set* keyed by plane kind,
and `decode_chunk` needs it threaded in. The numbers are in
[perception-codec-handover.md](perception-codec-handover.md).

## 1b. What the decoder costs, published — **done**

A downstream consumer asked for the decoder's cost constants, because it has
to decide whether the coded route beats the plain one *on its machine* and
could only do that by guessing at lmz's kernel. The request noted that one
point on one device — 418 GB/s, shared table, RTX 5080 — fits two readings
that agree here and differ by 8× on a 2-CU integrated GPU: 84% of a
bandwidth bound, or k = 128 cycles per byte if compute binds.

**Measured, and neither reading was right.** The separator does not need a
second device: the block size is a compute knob at *fixed byte traffic*.
Every row below decodes the same 936.4 MB from the same 325.2 MB of coded
bytes; only the number of resident lanes changes.

| threads/block | GB/s | DRAM traffic | % of ~960 GB/s peak |
|---|---|---|---|
| 64 | 245.7 | 331 GB/s | 34% |
| 96 | 262.9 | 354 | 37% |
| 128 | 324.0 | 437 | 45% |
| 160 | 397.5 | 536 | 56% |
| **192** | **414.4** | 558 | 58% |
| 256 | 404.6 | 545 | 57% |
| 384 | 417.3 | 562 | 59% |

Two runs, agreeing within 1% at every point, on an otherwise idle card.

**Below 192 threads compute binds** — the rate tracks resident lanes, a
1.70× spread bought by occupancy alone. **At and above 192 it saturates**,
and adding 2× more lanes (192 → 384) buys nothing. So the ceiling is
bandwidth, but at **59% of peak DRAM, not 84%**: the earlier figure counted
only the 936 MB written and forgot the 325 MB read, and traffic per decoded
byte is 1 + 1/2.88 = 1.347, not 1.

**k is 230–330 lane-cycles per decoded byte**, from the rows below
saturation, using each row's resident lanes against the kernel's own shared
memory request (a 16 KiB LUT plus 640 B a group, against the device's 99 KiB
opt-in block) and a 2.66 GHz sustained clock. It is published as an interval
because it is one: occupancy hides part of the cost and the amount hidden
varies with the block size.

**And an interval is only a bracket where the occupancy holds**, which is why
`cost_model()` publishes the shared-memory layout as numbers rather than
prose: `shmem_lut_bytes` (16384), `shmem_per_group_bytes` (640) and
`blocks_per_unit_at_measurement` (3, 4). A caller divides its own device's
per-unit shared-memory budget by the block request to learn how many blocks
it can hold, and compares. **Hold at least as many as the measurement did and
the interval brackets that device; hold fewer and less of the dependent-load
chain is hidden, so 330 is a floor with no published ceiling.** That is the
case on a small integrated adapter — 128 KiB per unit gives four resident
blocks and a two-sided bracket, 64 KiB gives two and the row collapses to a
floor — and it is exactly the case the interval must not be quoted as a
bracket for. The three numbers exist so a consumer can decide that
automatically instead of a reader remembering the caveat.

**It is latency-bound, not throughput-bound**, which is the part that
travels to other devices. The inner loop is a dependent shared-memory load
feeding a state update feeding the next load; the cost is a pointer chase,
not arithmetic. So **k must not be scaled by a device's FP32 rate** — the
consumer's instinct that FP32 FMA is the wrong unit for integer rANS is
correct, and this says why.

`lmz.gpu.cost_model()` publishes all of it with provenance. What is still
missing is the low-FLOP-per-byte device the request asked for: this box's
iGPU is not reachable from WSL2 (no `/dev/dri`, no Vulkan, no OpenCL) and
locking clocks needs privileges this session does not have, so the interval
is from one high-bandwidth card and is marked that way in `provenance`.

## 2. The GPU decoder as a library, not a benchmark — **done**

`lmz/gpu/` is the kernel promoted out of its benchmark. The entry point is a
batch rather than the single stream this document originally sketched:

    lmz_gpu_decode_batch_dev(hdr, streams, offsets, nstr, plane, out, stream, tpb)

**A batch, because one stream is 8 lanes of work.** `lmz_gpu_decode(stream,
coded, ...)` was the wrong shape and it is worth saying why: an lmz rANS
stream has exactly 8 interleaved states, so it has exactly 8 lanes of
parallelism in it no matter how many megabytes it holds. A single-stream call
would occupy a quarter of one warp of one SM. What fills a GPU is many
streams at once, which is a property of the format and not of the kernel, so
the ABI has to say so. `lmz_gpu_decode_batch` is the host-memory convenience
form; it is PCIe-bound by construction and exists for tests and for callers
who want the bytes back. The `_dev` form takes device pointers and a stream
and does not synchronise, which is the one a residency engine wants.

**Both table layouts, so the format decision stays open.** `hdr` non-NULL is
a table shared by the batch, and NULL means each stream carries its own —
which is what an archive written today holds. They decode the same coded
bytes, because sharing a table is a question of where 516 bytes live and not
of how a symbol is coded. Both verified byte-identical to `lmz_rans_decode`
over the full 936 MB.

**What the per-chunk table costs, now measured on the shipped kernel:**

| | GB/s |
|---|---|
| one table per chunk, narrow LUT in shared memory | **111** |
| one table for the batch, wide LUT in shared memory | **418** |

A per-chunk table cannot use the 16 KiB wide LUT — sixteen groups of it is
256 KiB and a block may opt into 99 — so that path packs the table into
4 KiB of slot→symbol plus 1 KiB of symbol→(start, freq) and pays a second
dependent shared load on the critical path. 5.6 KiB per group is what lets a
block hold sixteen of them. 111 GB/s is still 3.9× the PCIe link and 50× the
CPU decoder, so **the no-format-change path is already useful**; the shared
table is worth 3.8× on top and is why item 1 comes first.

**A 16-byte `cp.async` needs a 16-byte-aligned source.** The benchmark never
met this because `prep_shared.py` padded every stream to 16 and the table was
lifted out. A real per-chunk stream puts its coded bytes at 516 + 32 = 548,
which never is, and the failure is `misaligned address` from the *next*
synchronise with no line number in it. The fix costs one add at setup: drop
the cursor to the boundary below and start the byte count at the difference,
since ring slot is source offset mod BUFB either way.

**CUDA stays optional, and this is load-bearing.** The wheel carries a `.cu`
and no CUDA, stays `py3-none-any`, and `pip install lmzip` needs no toolkit.
`lmz/gpu/build.py` mirrors `lmz/native/build.py`: nvcc *if one is present*,
built into the package directory, nothing installed system-wide, never
raises. Two things it has to do that the C build does not — it asks the
driver for the device's compute capability and puts it in the artifact name,
because `-arch` bakes the target in and `platform.machine()` does not change
when the card does; and it tries every nvcc it can find newest-first, because
a box very often has an old toolkit on PATH beside a new one in
`/usr/local`, and CUDA 12.4 answers `-arch=sm_120` with "not defined for
option gpu-architecture". `-cudart static` so the result depends on the
driver alone and not on a toolkit that may move. `lmz doctor` reports it.

**Never hardcode an architecture list.** The first version of that build
carried `("70", "80", "86", "89")` and it was wrong in both directions at
once: CUDA 12.4 cannot target sm_120, and **CUDA 13 has dropped sm_70
entirely** — one gencode nvcc refuses fails the whole compile, so on a modern
toolkit the multi-architecture fallback did not build at all. `nvcc
--list-gpu-arch` answers the question, and the floor is Turing: `__ballot_sync`
and `__syncwarp` want Volta's independent thread scheduling, `cp.async` is
Ampere's and degrades to a synchronous copy below it, and CUDA 13 will not
compile for anything older anyway. A card below the floor is declined with
its compute capability in the message rather than handed to nvcc to reject.
The fallback also emits PTX for the newest architecture it knows, because a
card newer than the toolkit has nothing to JIT from otherwise.

**A driver can fault on load, and no return code says so.** This one was
found the hard way, when a driver update landed underneath a running session:
`libcuda.so.1` stays on disk with an initialiser that segfaults, `nvidia-smi`
comes back on PATH but answers nothing, and a bare `ctypes.CDLL` takes the
interpreter down. Checking return codes cannot help, because the process is
already gone. So the first load happens in a child that is allowed to die,
and the parent turns a signal into a sentence. It costs one interpreter start
and one CUDA context, once, and only on a machine that has both a toolkit and
something that looks like a driver — everywhere else the build has already
declined and none of it runs. The hardware gates also sit *ahead* of the
"already built" check, because an artifact built when the driver worked is
still sitting there after it stops.

**The pipeline needs four slots, not two.** A two-slot `cp.async` buffer
verifies byte-identical on the exponent plane and *races anyway*; the failure
only appears on the sign+mantissa plane, whose 3× higher refill rate reaches
the boundary sooner. This is written down because it is the exact shape of a
bug that passes its test.

**The lane trick is the format's, and it should be documented as such.** The
7.3× that made any of this possible needs no format change: the 8 interleaved
rANS states share one input cursor and refill in strict order, but whether a
state needs a refill depends only on that state — so the 8 states map onto 8
lanes and one `__ballot_sync` plus a popcount prefix sum reconstructs every
lane's byte offset. Anything that changes the interleave breaks it.

**What is left of this item.** Nothing in the archive path calls the decoder;
`lmz decompress` is unchanged and deliberately so, because the useful thing
to do with a GPU decode is to leave the result in VRAM and lmz does not
decide when. The fused whole-BF16 kernel is not promoted either — it is one
plane's rANS plus a raw plane plus a merge, and the merge is the archive's
business rather than the coder's. A batch must be uniform: every stream
decodes to the same `plane`, and `plane` must be a multiple of 128, which the
64 KiB block store gives and a ragged tail does not.

`scratchpad/gpu/metal/` is the Apple silicon port, **written but never run**.
It is worth exactly what an unrun kernel is worth.

## 3. The chunk table, which was the one that blocked a residency layer — **done**

`limitations.md` used to call this "the next thing to fix and it is a real
limit today": the chunk table was parsed into one Python object per chunk at
open, which for a 70B checkpoint is **0.51 GB and about 5 s** before a byte is
read. For a compressor that is an annoyance. For a residency engine it is
fatal: the archive is opened on every process start, a placement solver needs
every extent before it can decide anything, and half a gigabyte of Python
objects is a meaningful fraction of the host tier it is trying to manage.

**It needed no format change, because the records were already an index.**
The table is fixed-width `RECORD`s, so the decompressed bytes can be held as
they are and a `Chunk` unpacked when something asks for one. `ChunkTable` is
that, and it is a sequence, so the nine call sites that indexed, iterated,
sorted or `len`'d a list kept working untouched.

| 70B index, 2.2M chunks | before | after |
|---|---|---|
| open | 5.1 s | 54 µs |
| resident | 0.51 GB | the 70 MB table |
| one lookup | attribute load | ~5 µs |

**The measurement that decided the design.** Of the 5 s, the zstd decode of
those 70 MB is 0.07 s; everything else was building objects. So keeping the
flat buffer captures essentially all of the win, and mmapping the file — the
obvious move, and what this section used to ask for — would have needed the
table stored uncompressed to work at all. It buys 0.07 s and costs a format
change. Not worth it.

**Two traps found while building it, both measured rather than reasoned.**
Sorting the packed `dst` column and then pulling records out by index is the
natural instinct and is *slower*: a per-record `unpack_from` loop cost 2.08 s
against 1.13 s for one `iter_unpack` pass, which more than spends what the
cheaper sort saves. And the aggregate paths (`info`, the coverage check) do
want the column — `sum(c.clen for c in chunks)` built 2.2M objects to add one
integer, and reading the column instead is about twice as fast *and* allocates
nothing. `order_by_dst()` ends up 1.7× faster than the old sort because it
also skips sorting entirely when the table is already ascending, which is the
common case.

What is still true: a caller that walks the whole table repeatedly should
hoist it into a list, and `verify` and `mount` do, since they were going to
materialise everything anyway.

## 4. Tensor-level addressing

A residency manager asks one question — *where are the blocks for this
tensor?* — and today the only way to answer it is to go through a filesystem:
mount the archive, parse the safetensors or GGUF header, map offsets to
chunks. That works, and it is three layers too many for something in the
per-layer path.

What is wanted is a manifest carried in the archive: tensor name → dtype,
shape, and the block range that holds it, written at compress time when the
layout parser already knows all of it. The block splitters for safetensors
and for GGUF's ggml struct layout already recover exactly this information
and then throw it away.

This changes no bytes of any coded stream. It is metadata, and it is what
turns `lmz` from something a filesystem can read into something a scheduler
can address.

## Where the boundary is

Not lmz's, and it should stay that way. This table had two columns when
there were two parties; there are now three, and the middle one is where
most of the ambiguity lived — a transport layer that decides *when* bytes
move is neither the codec nor the residency engine:

| | whose |
|---|---|
| the placement solver, the tier budgets, eviction | the engine |
| adapters for PyTorch, vLLM, llama.cpp | the engine |
| a *fused* decode-into-GEMV kernel | the engine — it owns the matrix kernel |
| prefetch, queue depth, stream and event choreography | **the transport layer** |
| page-locked staging buffers, the copy/compute overlap | **the transport layer** |
| measuring the machine and choosing the route | **the transport layer** |
| the coded stream, its table, its blocks, its index | **lmz** |
| a standalone GPU decoder with a stable ABI | **lmz** |
| decoding all the way to plaintext, planes merged | **lmz** |
| the decoder's own cost constants | **lmz** |
| tensor → block addressing | **lmz** |

The dividing line that keeps holding: lmz produces bytes and can turn them
back into bytes as fast as anyone. It does not decide *when*.

**The purity rule that follows, stated where a contributor will hit it.**
lmz's decode path is bytes in, bytes out. A new entry point that opens a
file, starts a thread pool, or measures the machine in order to choose a
policy belongs to the caller instead — there is one rate model in the stack
and it is the transport layer's. `cost_model()` is the shape that stays on
lmz's side: it publishes what the *kernel* costs, with provenance and with
an interval where the value is not pinned, and says nothing about the
machine it is called on. `Store`'s fixed readahead is the one place inside
lmz that looks like policy; it is a constant with no probe and no
adaptation, which keeps it on the right side of the line. The day it wants
to time a disk, it should take the model from downstream rather than grow a
second one.

**Interrogation is not measurement, and the line runs between them.**
`compress(workers=None)` falls back to `default_workers()`, which asks the OS
for a CPU count honouring affinity and cgroups. That is a *question* — it
times nothing, probes no device, adapts to no observed throughput — and it
stays. What must never happen is a default derived from a *rate*: nothing
timed, fitted, adapted from measured throughput, or cached as a machine
profile, because that is a second rate model and the stack has one.

`encode_options()` publishes the distinction so a caller can act on it. Every
keyword `compress` accepts is declared `format`, `schedule` or `observe`;
`format` decides what the coded bytes are and is lmz's to choose, `schedule`
decides only how the work is spread, and `observe` watches without changing
anything. **A scheduling default is a fallback, never a decision** — it was
chosen knowing nothing about the caller's machine or workload, which is
precisely the fact a transport layer needs and could not otherwise discover.
`workers` is the one today, and its default is deliberately a weak proxy:
compression goes memory-bus-bound near four threads on real weights, so a
core count is a starting point rather than an answer.

The split is a verified property rather than a naming convention — the same
input at 1, 2, 8 and 16 workers produces byte-identical archives, and the
suite checks both that and the declaration against `compress`'s own signature
by introspection, so a new keyword cannot be added without classifying it.

## What is already right and should not be touched

**64 KiB, page-aligned.** Measured in the consumer's probe on a real disk:
64 KiB random reads at queue depth 64 reach 6.34 GB/s, which *beats* a
single-threaded sequential 4 MiB loop at 3.59. The access pattern was never
the problem — depth was, and at depth-1 the same reads get 0.34 GB/s. So the
block size is right, random access against it costs nothing, and there is no
reason to relayout anything to be more sequential.

**Per-chunk checksums.** A residency engine's worst outcome is a silently
wrong weight. Keep them, keep them cheap, and keep them on by default.

**The 8-state interleave.** It is what let 8 lanes replace 8 chains. See above.

## Three traps, carried forward

The first two are from `vectorising-the-coder.md` and apply here unchanged;
the third was found in this kernel.

**Do not benchmark one stream twice.** The two-chunk test in `coderbench`
originally decoded the same stream into two buffers and reported a clean 2×
that did not exist. On a GPU this is easier to do and harder to see, because
a second pass over the same coded bytes finds them in L2.

**Do not ship a kernel change on one machine's evidence.** The CPU version of
this cost a regression on Apple silicon. The GPU version has a sharper edge:
two of the ablations in `scratchpad/gpu/README.md` contradicted a confident
guess outright — coalescing the refill loads gained *exactly nothing*,
because the cost was latency on the dependency chain and not transaction
count.

**A tuning knob measured only where it does not bind will be tuned wrong.**
`pick_tpb` chose the widest block that fits, which is the wrong quantity for a
latency-bound kernel -- what covers a dependent load is lanes resident on a
unit, so a block twice as wide that therefore fits only once is a loss. It
also read only the per-block cap and never the per-unit pool
(`sharedMemPerMultiprocessor`, 100 KiB here against the 99 KiB a block may
opt into), so it could not have reasoned about residency even in principle.

The reason it survived a full occupancy sweep is the trap: **this card
saturates on bandwidth at 192 threads**, so residency past that buys nothing
here and no measurement taken on it could see the defect. It bites only where
the kernel is compute-bound, which is every small device the "decode where it
lands" argument is aimed at and none of the ones it was measured on. It was
caught by a downstream consumer's arithmetic about a 2-CU adapter nobody in
this project owns.

The rule that follows: when a knob's optimum is reached on the development
box, that is evidence the box cannot distinguish the choices -- not evidence
the choice is right. Reason about what the knob *means* on a device where the
binding resource is different.

## How to measure

`scratchpad/gpu/prep_shared.py` and `prep_fused.py` regenerate the inputs
from a safetensors checkpoint; the data files are large (325 MB + 936 MB +
1.87 GB) and live under `~/.cache/`, not in the repo. Build with

    /usr/local/cuda-13.2/bin/nvcc -O3 -arch=sm_120 -o gpu_fused gpu_fused.cu

An `nvcc` older than 12.8 cannot target Blackwell. `gpu_rans10.cu` carries the
whole V0..V5 ladder plus the ablations that attributed the time; some kernels
in it deliberately produce wrong output and are labelled timing-only.

`scratchpad/gpu/cuda/libbench.cu` times what the *package* builds against the
same data — it `#include`s `lmz/gpu/lmzgpu.cu` rather than linking it, so
there is no second copy of the kernel to drift — and sweeps the block size.
It is the source of the 111 and 418 above, and takes a directory so it can be
pointed at `prep_synth.py`'s output instead of this machine's cache. Both
numbers were re-measured after a driver upgrade from 580 to 610.88 (CUDA UMD
13.3) and did not move: 111.1 and 417.4 GB/s.

Verification is as easy here as it was for the vector decoder: a decoder
either reproduces the plaintext or it does not, and `prep_fused.py` takes its
reference from the model file rather than re-deriving it from the planes, so
the test is not circular. The suite carries the same check without the cached
data: `test_gpu_decode_matches_cpu` builds streams with lmz's own encoder and
compares against `lmz_rans_decode`, and skips where there is no GPU.

**One card is one card, and the doc's own trap applies here.** What has been
run on an RTX 5080 cannot be run on a Turing or a Hopper without one — so the
evidence was widened in every direction that does not need the hardware, and
there turned out to be more of those than expected. A free T4 then closed the
one direction that did need it; the whole sequence is below, in the order it
happened, because the cheap steps are what made the last one a confirmation
rather than a gamble:

**And the decoder checks itself before it is used.** The first thing it ever
does on a machine is decode a stream that machine just encoded and compare
against `lmz_rans_decode`; a device that disagrees is not used at all, and
`lmz doctor` says so by name. This is the mitigation that actually covers
silicon nobody has run on, because a silently wrong decoder is far worse
than an absent one — the caller gets weights back rather than an error. It
costs one launch, after a CUDA context that was being created anyway: 385 ms
for the whole first probe, context and the out-of-process driver check
included, and 2 µs for every call after it. It was 281 ms before that check
was added, so surviving a broken driver costs about 100 ms once per process.

    compute-sanitizer --tool=memcheck   python3 -c ...   # 0 errors
    compute-sanitizer --tool=racecheck  python3 -c ...   # 0 hazards
    compute-sanitizer --tool=synccheck  python3 -c ...   # 0 errors

clean on both kernels over real planes — which is the check that speaks
directly to "a two-slot `cp.async` buffer verifies byte-identical and races
anyway" — and the kernel compiles for every architecture CUDA 13 supports at
or above the floor, sm_75 through sm_121.

**And the untested architectures are not equal, which narrows the problem to
one.** Counting `LDGSTS` in the generated SASS says which of them is running
the code that was actually verified:

| | `LDGSTS` |
|---|---|
| sm_75 | **0** |
| sm_80, sm_86, sm_89 | 38 |
| sm_90, sm_120 | 41 |

Turing has no `cp.async` instruction, so `__pipeline_memcpy_async` falls back
to a synchronous copy and sm_75 is *different generated code*. Everything at
sm_80 and above runs the algorithm that was verified and sanitized here and
differs only in scheduling, so above the Turing line the open question was a
throughput number, and at it the open question was correctness. That made a
free Colab T4 worth more than a rented A100, which is why the notebook exists
and why it asks for the card it does.

**A code path can be run on silicon it was not compiled for, which is how the
sm_75 question got answered without a Turing.** `-arch=compute_75` emits PTX
and no cubin, so `__CUDA_ARCH__` is 750 while the header picks its scalar
branch, and the driver JITs the result onto whatever card is present. That
works for every architecture, not just Turing, so each variant lmz can emit
was run on a Blackwell over the same 936 MB:

| codegen | `LDGSTS` | per-chunk | shared |
|---|---|---|---|
| sm_120, native | 41 | 110.2 GB/s | 417.4 GB/s |
| sm_90 | 41 | 104.4 GB/s | — |
| sm_80 / 86 / 89 | 38 | 104.4 GB/s | — |
| sm_75 | 0 | 86.7 GB/s | 391.8 GB/s |

Every one byte-identical, at all seven block sizes. Both kernels under sm_75
are also clean under `memcheck`, `racecheck` and `synccheck` — the shared one
too, which the first pass missed by sanitizing only `k_perstream`.

**This does not measure a T4.** It is each architecture's code on the wrong
silicon, so the 21% sm_75 gives up is what dropping `cp.async` costs a
Blackwell, not what Turing does. What it settles is that every variant lmz can
emit decodes and does not race, which was the whole *correctness* column. The
JIT'd rows sitting 5% under native is itself expected — the driver's JIT is
not `ptxas -O3` on the exact target.

**And then a real T4 ran it, which is the thing none of the above could
substitute for.** Via the Colab notebook, on a stock runtime with nothing
built by hand:

    lmz 1.1.3 GPU verification
      device   Tesla T4 sm_75 40 SMs
      shapes   30 decoded byte-identically to the CPU decoder
      verdict  OK

CUDA 12.8, driver 580.82.07, compute capability 7.5. Every step ran as
shipped: `pip install lmzip` fetched a pure-Python wheel with no CUDA in it,
`build.py` found the toolkit and targeted `sm_75`, the out-of-process probe
cleared the driver, the self-test agreed with the CPU coder, and thirty
distributions and batch shapes came back byte-identical.

**Two predictions this turned into measurements.** `pick_tpb` was expected to
step the per-chunk kernel down to 64 threads inside Turing's 64 KiB, and the
64 KiB figure came from a table rather than from a T4; both held. And the
`_ARCH_FLOOR` of 7.5 is a boundary the only sm_75 card sits exactly on — an
off-by-one there would have declined every Turing with a plausible-sounding
message instead of decoding.

**What is still open is now only Ampere through Hopper, and only as a
number.** sm_75 and sm_120 are the two ends of the supported range, they are
the two that generate *different* code, and both have now been verified on
real silicon. sm_80/86/89/90 sit between them, share the 38/41 `LDGSTS` path,
and have been JIT-run and sanitized here — so a report from an A100, A10G, L4
or H100 buys a throughput figure, not an answer to whether it works.

**A device that cannot fit the tables is refused, and that is arithmetic over
one number, so no rented card is needed to check it either.**
`scratchpad/gpu/cuda/shmfit.cu` runs the shipped `pick_tpb` against every
architecture's `sharedMemPerBlockOptin` and re-derives the local card's row
from the driver, so the table is checked against hardware at least once:

| | optin | per-chunk | shared |
|---|---|---|---|
| sm_75 T4, RTX 2080 | 64 K | 64 thr, 45 K | 384 thr, 46 K |
| sm_80 A100 | 163 K | 128 thr, 90 K | 384 thr, 46 K |
| sm_86 / 89 A10G, L4, RTX 4090 | 99 K | 128 thr, 90 K | 384 thr, 46 K |
| sm_90 / 100 H100, B200 | 227 K | 128 thr, 90 K | 384 thr, 46 K |
| sm_120 RTX 5080 | 99 K | 128 thr, 90 K | 384 thr, 46 K |

Nothing declines. Turing is the only architecture the per-chunk picker steps
down for, and it steps down to 64 threads — which measured *faster* than 128
here, so the narrower block Turing is forced into costs it nothing.

**A benchmark nobody else can run is not evidence.** `prep_synth.py` writes
libbench's input from lmz's own encoder with no checkpoint and no 1.8 GB of
cache to reproduce first, bisecting the decay to land on the entropy real
exponents have — 2.78 bits a symbol, because refill rate is a direct function
of it and a flatter buffer would measure a different machine. It reproduces
the real planes within 2.4% at the shipped default (107.6 against 110.2 GB/s)
and reproduces the 96-thread occupancy cliff too, and it leans the right way:
340.5 MB coded against the real 336.1 MB, so it asks for slightly more refills
per byte than real weights do. Twenty minutes on a rented card is now enough
to produce a number comparable with this one.

`lmz doctor --gpu-verify` is the asking. It builds streams with lmz's own
encoder, decodes thirty distributions and batch shapes, checks lmz's own CPU
decoder agrees with every byte, and prints a block worth pasting. No data
files and no network: the oracle travels with the question. That is what a
stock Colab T4 ran to produce the sm_75 result above, so the mechanism is not
theoretical — the same one click now wants an Ampere or an Ada.
