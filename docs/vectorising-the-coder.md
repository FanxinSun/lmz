# Vectorising the coder

*One piece of work that is open, what is already known about it, the six things
that were tried and do not work, and how the encoder got to arm64. Written for
whoever picks this up next, so that none of it has to be rediscovered.*

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
1.5× on arm64; decode is 2.4× behind on x86 and 1.1× on arm64.

The encoder is now vectorised on x86 and on arm64. The decoder is vectorised
nowhere, and that is the work that is left. The two were never equally well
understood: the encoder was a port of something that already worked, and the
decoder is an open question with a large prize and no guarantee.

## The NEON encoder

`rans_enc_body_neon` in `lmz/native/lmzcore.c`. Eight 32-bit states are what
the format already interleaves, so this is not a new stream layout — it is the
same arithmetic in parallel, and it writes byte-identical output. That property
is what makes it a speed change rather than a format change, and everything
below is arranged around keeping it.

**Three things the scalar step needs replacing, and only one of them was the
problem x86 also had.**

*The divide* becomes a fixed-point reciprocal, after Giesen — the same one the
AVX2 body uses, exact only while states stay below 2^31, and states reach
`freq << 20`, so the two meet at half the probability scale. The caller checks
the frequencies and hands anything above `RANS_SIMD_MAX_FREQ` to the scalar
loop, whose hardware divide is exact everywhere. Frequencies below two need
`bias = start + RANS_PROB_SCALE - 1` to make the reciprocal's off-by-one come
out right, which keeps the lane step free of any special case. Both bodies now
read that table from one place, because it is the part that is easy to get
wrong.

*The emission mask*, because NEON has no `movemask`. The lane masks are
unzipped down to one 16-bit element each and summed against a weight vector,
which is a single `ADDV` — and the weights are `256 + (1 << j)` rather than
`1 << j`, so one sum carries the shuffle's index in its low byte and the number
of lanes that emitted above it. A mask cannot reach 256 and eight lanes cannot
reach 65536, so the two never meet. That matters because both answers are on
short chains that the rest of the loop waits on: the store waits for the index,
the cursor waits for the count, and neither waits for a second reduction.

*The variable shift*, which does not exist on NEON as such: `USHL` by a
negative count is a right shift, so the shift is negated in a vector and
applied left.

**Two things are cheaper here than on x86.** Packing the eight low halves into
one register is a single `UZP1`, where AVX2 needs a shuffle and a cross-lane
permute. And the per-symbol table is one array of 64-bit entries rather than
two of 32-bit, so a group of eight is eight loads landing straight in a
register, where AVX2 fills its two vectors by writing sixteen words to the
stack and reading them back.

**What it is worth.** 16 MiB planes, `scratchpad/encbench.c`, best of seven,
against the scalar loop on the same machine in the same process:

| | scalar | vector | |
|---|---|---|---|
| **GitHub arm64, exponent plane** | 455 MiB/s | 911 MiB/s | **2.00×** |
| **GitHub arm64, near-uniform** | 463 | 895 | **1.93×** |
| **Apple silicon, exponent plane** | 1063 | 1340 | **1.26×** |
| **Apple silicon, near-uniform** | 1040 | 1301 | **1.25×** |
| GitHub x86, exponent plane (AVX2) | 395 | 1266 | 3.20× |
| Zen 5, exponent plane (AVX2) | 811 | 1566 | 1.93× |

Which is the same kernel varying by a factor of two across two arm64 machines,
and the same AVX2 kernel varying by 1.7× across two x86 ones. The reason is
visible in the scalar column: **a machine with a fast integer divider has less
to win**, because the divide is what the vector path removes. Apple silicon
runs the scalar encoder at 1063 MiB/s, faster than any other machine here runs
it, and takes the smallest speedup as a result. The old estimate in this
document — "expect less than 1.94×" — was wrong on one machine and right on the
other, for a reason that has nothing to do with NEON being 128-bit.

On the whole compress pipeline, 64 MiB of BF16 through `lmz bench`, arm64:
**330 MiB/s before, 455 after**, with `zstd -1` measured in the same process at
666 both times — so that is the pipeline gaining 38%, not the runner having a
good day. Decompression is untouched at 468 → 471. Apple silicon gains about
30% on the same measurement, more noisily.

**One arrangement was measured and kept.** `x + bias + q * cmpl` was three
instructions deep behind the quotient, and the quotient is what everything
waits for. `x + bias` does not need the quotient, so it goes in the accumulator
and `MLA` does the rest, which leaves one instruction on the chain instead of
three. Worth nothing on the GitHub arm64 runner (2.00× either way) and a great
deal on Apple silicon, where the near-uniform plane went from **1.04× to
1.25×** — that is, from a wash to a real speedup. A latency-bound machine and a
throughput-bound one do not respond to the same change, and there is no way to
find that out except on both.

**What was not separately measured.** The 64-bit table and the single-`ADDV`
mask were reasoned about and then shipped inside a kernel that was measured as
a whole; neither was held against the alternative on hardware. If the decoder
work makes either look wrong, they are worth pricing on their own.

## How the arm64 kernel is checked without an arm64 machine

`scratchpad/neonshim/arm_neon.h` is an `arm_neon.h` written in plain C —
only the intrinsics `lmzcore.c` actually names, each one written to the
definition the ARM intrinsics reference gives it. `-DLMZ_FORCE_NEON` makes the
kernel include that instead of the real one:

```
cc -O2 -DLMZ_FORCE_NEON -Iscratchpad/neonshim -o encbench-neon \
    scratchpad/encbench.c && ./encbench-neon
```

The source under test is the shipped file, unmodified, compiled down its arm64
path. What comes out is the real byte stream — lane order, emission mask,
compaction and arithmetic are all decided by the source — so the comparison
against the scalar loop is the real comparison. Only the timings are
meaningless, and the tool says so on the line it prints.

The same trick runs the *whole Python suite* against the arm64 kernel. Build
the library by hand into the name `lmz/native/build.py` would give it, and it
is loaded rather than built:

```
cc -O2 -fPIC -shared -DLMZ_FORCE_NEON -Iscratchpad/neonshim \
    -o "$(python -c 'from lmz.native import build; print(build.library_path())')" \
    lmz/native/lmzcore.c
python -c "from lmz import kernels; print(kernels.backend())"   # native:neon
python tests/test_lmz.py                                        # 91/91
```

Do that in a copy of the tree. The library's name is a hash of the source, so a
shim build and a real build claim the same file, and one left behind is slow,
correct, and very hard to notice.

None of this replaces an arm64 machine for *tuning* — the two arrangements
above were separated by a CI round trip each, and that is a bad development
loop for a third or a tenth. It replaces it for correctness, which is the part
that has to hold everywhere.

## The project — the vector decoder

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
this; AVX2 needs a 256-entry shuffle table, the inverse of the encoder's
`rans_compact_table`; NEON needs that table and `vqtbl1q_u8`, whose
out-of-range-selects-zero rule lets the same 0x80 fill serve both ISAs.

What the encoder settles for it: the mask is no longer an open question. `UZP1`
down to 16-bit lanes and one weighted `ADDV` produces both the table index and
the population count, and on the encoder that chain was never what the loop
waited on. The decoder will lean on it harder — there it is per group, on the
critical path, and the vector-to-general-register transfer is the part to
watch.

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

`scratchpad/encbench.c` asks the two questions a vector encoder has to answer —
does it write the scalar loop's bytes, and is it faster — and it compiles the
shipped kernel itself rather than a copy, so it cannot drift from it. It uses
the same generator and the same seed as `coderbench`, so the two print the same
plane digests. It also counts how many of its inputs were *eligible* for the
vector body, because two encoders agreeing while both run the scalar loop is
not evidence of anything, and that is a quiet way for a check like this to pass
forever.

Every CI job runs it twice, natively and through the shim, so the arm64 encoder
is checked on every machine in the matrix rather than on the two that are
arm64. Push a branch named `bench…` and `.github/workflows/bench.yml` runs the
whole pipeline, and `encbench` natively, on x86, arm64 and macOS runners —
which is where the numbers above come from. `LMZ_NO_NATIVE=1` forces the numpy
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
afterwards.

The NEON encoder is the same lesson from the other side. The same kernel is
worth 2.00× on the GitHub arm64 runner and 1.26× on Apple silicon; the one
instruction it was tuned by is worth nothing on the first and 20% on the
second. Neither machine would have predicted the other, and neither number is
wrong. The `bench…` branch trigger makes checking both nearly free; the cost of
not checking is a regression on the platform people actually run models on.
