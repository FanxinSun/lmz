# Vectorising the coder

*Two pieces of work that are open, what is already known about them, and the
six things that were tried and do not work. Written for whoever picks this up
next, so that none of it has to be rediscovered.*

[← back to the README](../README.md)

## Where the time goes

Per 64 MiB of BF16, single-threaded, AMD Ryzen 7 9800X3D:

| | encode — 81.0 ms | decode — 103 ms |
|---|---|---|
| rANS | 28.8 ms (36%) | 71 ms (**70%**) |
| partition / unpartition | 21.1 (26%) | 19.4 (19%) |
| histogram | 17.2 (21%) | 7.8 (8%) |
| split / merge | 9.6 (12%) | 3.4 (3%) |

Against `zstd -1` on the same data: encode is about 1.4× behind on x86 and
2.0× on arm64; decode is 2.4× behind on x86 and 1.1× on arm64.

The encoder is vectorised on x86 and nowhere else. The decoder is vectorised
nowhere. Those are the two projects, and they are not equally well understood:
the first is a port of something that works, the second is an open question
with a large prize and no guarantee.

## Project one — the NEON encoder

arm64 gets nothing from the AVX2 encoder, and arm64 is where models are loaded.
It is the widest single gap in the project.

**What to copy.** `rans_enc_body_avx2` in `lmz/native/lmzcore.c`. Eight 32-bit
states are exactly one AVX2 register and eight is what the format already
interleaves, so the vector encoder is not a new stream layout — it is the same
arithmetic in parallel, and it writes byte-identical output. That property is
what makes the whole thing safe, and NEON should keep it.

**Three things the scalar step needs replacing.**

*The divide* becomes a fixed-point reciprocal, after Giesen. It is exact only
while states stay below 2^31, and states reach `freq << 20`, so the two meet at
half the probability scale — see `RANS_SIMD_MAX_FREQ`. The caller checks the
frequencies and hands anything above it to the scalar loop, whose hardware
divide is exact everywhere. Frequencies below two need `bias = start +
RANS_PROB_SCALE - 1` to make the reciprocal's off-by-one come out right, which
keeps the lane step free of any special case.

*The emission.* Lanes renormalise independently, so their words must be made
contiguous before they can be written. On AVX2 they are packed down, compacted
to the right by a shuffle chosen from an 8-bit emission mask, and stored
unconditionally sixteen bytes below the cursor — which is scratch, because
encoding runs backwards, so only the cursor moves by the number of lanes that
really emitted. NEON has `vqtbl1q_u8`, which is the shuffle. What it does not
have is `movemask`: narrowing with `vshrn_n_u16` and extracting a lane, or a
paired-add against a bit-weight vector, are the usual substitutes and neither
has been measured here.

*The table lookups.* Do not reach for a gather. On x86 three gathers reached
1.56×, two reached 1.90×, and eight ordinary scalar loads filling the vectors
by hand reached 1.94× — `vpgatherdd` was the slow one. NEON has no gather at
all, which for once costs nothing.

**What to expect.** AVX2 measured 1.94× over the scalar loop, and within 3% of
that same loop with its table lookups removed entirely — so what remains there
is one dependency chain and not anything still on the table. NEON is 128-bit,
so eight states is two registers rather than one, and the compaction has to
happen twice per group. Expect less than 1.94×.

**How to know it is right.** `test_rans_vector_path_writes_the_same_bytes`
already does the whole job: nine lengths straddling the group boundary and the
length the vector path starts at, across five distributions, three of which take
the vector path and two of which the frequency guard turns away. It compares
against `kernels.rans_encode(data, portable=True)`, which forces the scalar
body through `lmz_rans_encode_portable`. If that passes, the kernel is correct.

**Where to do it.** Not from a machine that cannot compile it. An arm64 Linux
guest on Apple silicon runs natively under Parallels and gives a compile-run
loop of seconds; CI gives timings on arm64 but a three-minute round trip, which
is not a development loop for SIMD.

## Project two — the vector decoder

**The prize is large and it is measured.** Replay the table indices from a
recorded run instead of computing them — same loads, same store, same
arithmetic, same refill, one dependency removed — and the loop runs **21×
faster on x86 and 16× on Apple silicon** on an exponent plane, 4–6× on a
near-uniform one. The decoder is not short of work. It is waiting on its own
state and leaving most of the machine idle. `decode, chain cut` in
`scratchpad/coderbench.c` measures this anywhere.

**Why no scalar arrangement collects it.** Every way of handing the machine
more independent chains was tried and every one lost — see the table below. The
constraint is the general-purpose register file and the length of the cursor
chain. Eight rANS states and their eight table entries already fill sixteen
registers; stepping sixteen states as two groups of eight recovers to *exactly*
1.00× and not one point further, which is the shape of a spill and nothing
else. arm64's thirty-one registers buy about a tenth and then stop.

**So the design is the vector register file, and it needs no format change.**
Eight lanes per chunk, one register per chunk, several chunks at once —
thirty-two chains across four registers, where the scalar file could not hold
sixteen. Each chunk keeps its own cursor and its own table, which matters:
N chunks is N chains of eight, while sixteen states in one stream is one chain
of sixteen, and that arrangement is the one that loses hardest.

**The hard part is the refill, and it is not the encoder's problem mirrored.**
Encoding compacts — a variable number of lanes each contribute a word to a
contiguous run. Decoding *expands* — a variable number of lanes each take a
word from a contiguous run, in lane order. AVX-512 has `vpexpandd` for exactly
this; AVX2 needs a 256-entry shuffle table, the inverse of the encoder's; NEON
needs the table plus a `movemask` substitute, as above.

**Honest expectation.** Somewhere between 1.5× and 2.5× on 70% of decode, and
that is a guess rather than a measurement. A gather adds latency to every
chain, and latency is precisely what is scarce here — so the same finding that
makes the project attractive also threatens it. Prototype before committing:
build it standalone in `scratchpad/`, on generated planes, and measure against
the scalar decoder before touching `lmzcore.c`.

Verification is at least easy. A decoder either reproduces the plaintext or it
does not.

## Measured and closed — do not re-tread

Each of these looked right beforehand. Ratios are against the shipped scalar
path on the plane named; two figures mean exponent plane then near-uniform.

| | x86-64 | Apple silicon | why |
|---|---|---|---|
| 16 or 32 interleaved states | 0.83× / 0.53× | 0.41× / 0.54× | one cursor advanced sixteen times, and the states spill |
| Interleaving 2–4 chunks, scalar | 0.96×, 0.87×, 0.88× | 1.03×–1.13×, flat past two | register file; arm64 buys a tenth and stops |
| Decode table 16 KiB → 4+1 KiB | 1.07× / 0.99× | — | and it changes interleaving by *nothing*: 32 KiB and 10 KiB of tables behave identically, so L1 footprint was never the constraint |
| AVX2 decoder over the existing 8 lanes | — | — | eight chains in one register is still eight chains |
| Histogram with 8 or 16 sub-tables | 1.10× / 0.95× | — | counting is already ~1 cycle/byte |
| Joint counts fused into the partition | 0.98× | — | a histogram costs its increment, not its load; the partition goes 1.37 → 1.89 cyc/B and gives back exactly the pass it saves |

The 16-state format also costs the scalar *encoder* 0.97× on x86 and 0.88× on
Apple silicon, and buys the AVX2 encoder 3.50×. It is dead on the decode side
alone, but it is worth knowing that widening the stream only ever helps where
there is a vector unit to feed.

## How to measure

`scratchpad/coderbench.c` is self-contained: no dependencies, no input files,
no libm. It generates both planes from a fixed seed and prints their digests,
because a comparison of different bytes is worse than no comparison. It builds
with `cc -O3 -o coderbench scratchpad/coderbench.c` and takes the core clock in
GHz as an optional argument for cycles per byte.

Push a branch named `bench…` and `.github/workflows/bench.yml` runs the whole
pipeline on x86, arm64 and macOS runners. `LMZ_NO_NATIVE=1` forces the numpy
and pure-Python fallbacks. `lmz bench <file> --methods` shows which coder
actually earned the bytes.

## Two traps, both of which caught this work

**Do not benchmark one stream twice.** The two-chunk test in `coderbench`
originally decoded the same stream into two buffers, so the second pass read
cache lines the first had just pulled in and a table it had just warmed. It
reported two streams in 1.01× the time of one — a clean 2× that does not exist.
Measured with two different streams and a frequency table each, it is 0.94×.
A whole roadmap was built on that number before it was caught.

**Do not ship a kernel change on one machine's evidence.** Deciding all eight
refills before any of their addresses is worth +11% on Zen 5 and −12% on Apple
silicon, where the serial cursor is the fastest decoder of the three machines
measured. It shipped on the x86 number alone and had to be made conditional
afterwards. The `bench…` branch trigger makes checking both architectures
nearly free; the cost of not checking is a regression on the platform people
actually run models on.
