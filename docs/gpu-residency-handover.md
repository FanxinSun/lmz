# GPU residency — what lmz has to ship for it

*The pieces of work in lmz between a fast kernel and a residency layer, why
each one is load-bearing, what is already measured about it, and where the
boundary is between lmz and the thing that consumes it. One of the four has
landed; this says what it cost and what it left. Written for whoever picks
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
below records what that took. What still does not exist is a layer that can
put it to work. The fast path needs a format option that is still an
experiment, the archive index cannot be opened at the size a residency layer
opens it at, and nothing can ask an archive where a tensor's blocks are. That
is three pieces of work, and none of them is research.

The consumer is a residency engine that keeps weights coded in VRAM, in
page-locked host RAM and on NVMe, and decodes them on the GPU. Its charter,
its roofline and the measurements behind it are in the parent repository at
`vram/`. **The one number it turns on:** a GPU rANS decoder runs at 399 GB/s
and the PCIe link into it at 28.8, so compression on that path is free by
14×, and every point of lmz's ratio becomes a point of bandwidth.

That engine is not lmz's job. This is the part that is.

## 1. The shared frequency table, as a format option

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

## 3. The chunk table, which is the one that blocks a residency layer

`limitations.md` already calls this "the next thing to fix and it is a real
limit today":

> A large store costs memory to mount. The chunk table is parsed into Python
> objects at open: 305 bytes per 64 KiB block, which is 8 MiB for the
> 1.74 GiB model here and extrapolates to **~0.6 GiB for a 70B one, taking
> about 11 s**.

For a compressor that is an annoyance. For a residency engine it is fatal:
the archive is opened on every process start, a placement solver needs every
extent before it can decide anything, and 0.6 GB of Python objects is a
meaningful fraction of the host tier it is trying to manage.

What it needs is an on-disk index that is **mmapped and used in place** —
fixed-width records, sorted, binary-searched or directly indexed, with no
Python objects between the file and a block address. Opening a 70B archive
should cost a page fault, not eleven seconds. Everything else in this
handover is an improvement; this one is a precondition.

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

Not lmz's, and it should stay that way:

| | whose |
|---|---|
| the placement solver, the tier budgets, eviction | the engine |
| prefetch, queue depth, stream and event choreography | the engine |
| page-locked staging buffers, the copy/compute overlap | the engine |
| adapters for PyTorch, vLLM, llama.cpp | the engine |
| a *fused* decode-into-GEMV kernel | the engine — it owns the matrix kernel |
| the coded stream, its table, its blocks, its index | **lmz** |
| a standalone GPU decoder with a stable ABI | **lmz** |
| tensor → block addressing | **lmz** |

The dividing line that keeps holding: lmz produces bytes and can turn them
back into bytes as fast as anyone. It does not decide *when*.

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

## Two traps, carried forward

Both are from `vectorising-the-coder.md` and both apply here unchanged.

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
there turned out to be more of those than expected:

**And the decoder checks itself before it is used.** The first thing it ever
does on a machine is decode a stream that machine just encoded and compare
against `lmz_rans_decode`; a device that disagrees is not used at all, and
`lmz doctor` says so by name. This is the mitigation that actually covers an
architecture nobody has run, because a silently wrong decoder is far worse
than an absent one — the caller gets weights back rather than an error. It
costs one launch, after a CUDA context that was being created anyway: 281 ms
for the whole first probe, context included.

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
throughput number, and at it the open question was correctness.

**A code path can be run on silicon it was not compiled for, which is how the
sm_75 question got answered without a Turing.** `-arch=compute_75` emits PTX
and no cubin, so `__CUDA_ARCH__` is 750 while the header picks its scalar
branch, and the driver JITs the result onto whatever card is present. Turing's
generated code, run on a Blackwell over the same 936 MB:

| | per-chunk tables |
|---|---|
| sm_120 codegen, 41 `LDGSTS` | 110.2 GB/s, byte-identical |
| sm_75 codegen, 0 `LDGSTS` | 86.7 GB/s, byte-identical |

byte-identical at all seven block sizes, and clean under `memcheck`,
`racecheck` and `synccheck`. **This does not measure a T4** — it is Turing's
code on the wrong silicon, so the 21% is what dropping `cp.async` costs a
Blackwell. What it does settle is that the fallback decodes and does not race,
which was the only *correctness* gap in the matrix. A real T4 is now wanted for
its number and its scheduler rather than to find out whether it works.

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
files and no network: the oracle travels with the question.
