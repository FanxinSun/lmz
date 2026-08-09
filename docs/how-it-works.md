# How lmz works

*The coder, from why a float array defeats a general-purpose compressor to the exact bit-level choices that get within 0.3 points of the entropy bound.*

[← back to the README](../README.md)

## Why weights resist ordinary compression

A float array looks like noise byte by byte. It isn't — but the structure sits
on *bit* boundaries that don't line up with bytes. A bfloat16 value is
1 sign bit, 8 exponent bits, 7 mantissa bits, and the useful skew lives
entirely in the exponent: trained weights cluster near zero, so on real Llama
weights the exponent takes **32 distinct values at 2.58 bits**, while the
mantissa is near-uniform noise at 6.89 of a possible 7 bits.

Cutting a BF16 value in half at the byte boundary slices *through* the
exponent, welding its last bit onto the mantissa. lmz cuts on the float's own
field boundaries instead, so the whole exponent stays in one alphabet. On real
weights that regrouping alone is worth 0.065 bits per element.

Then each plane is entropy coded with a static order-0 **rANS** coder rather
than a general-purpose compressor. This matters more than it sounds. An
exponent plane has no repeats for an LZ matcher to find, so zstd's match
search wastes time *and* dilutes its own Huffman stage — measured at **11%
above the order-0 entropy bound**. Huffman alone still gives up 1.6%, because
it must round every symbol to a whole bit. rANS pays fractional bits and lands
**within 0.16% of entropy**.

The last step is conditioning: the mantissa is *not quite* independent of the
exponent, and dealing the sign+mantissa plane into eight exponent buckets,
each with its own table, collects that remaining 0.075 bits per element.
Together the three choices are worth 3.1 points over the byte-splitting
approach lmz started with, and land within 0.3 points of the bound no
lossless coder of any kind can pass:

| encoder on real Llama BF16 | saved |
|---|---|
| byte split + zstd (where this project began) | 31.6% |
| byte split + rANS on both planes | 34.0% |
| field split + rANS | 34.4% |
| **field split + exponent-conditioned rANS** | **34.7%** |
| theoretical joint-entropy bound | 35.0% |

## How it works

**Planning.** The tensor index is read from the container (safetensors, GGUF
and PyTorch's `.bin` zip checkpoints are all understood) to recover the dtype
layout, so every chunk holds elements of a single known width. For a `.bin`
the storage classes are read from `data.pkl` with `pickletools`, which only
scans opcodes — nothing is unpickled or executed. Headers, padding and
unrecognised formats become 1-byte-element regions, which still compress,
just without the split. Runs of adjacent same-dtype tensors are coalesced
first, so that a model's thousands of tiny bias and norm tensors don't each
become an undersized chunk.

**Deduplication.** Checkpoints genuinely repeat themselves: Ministral ships a
`consolidated.safetensors` alongside the same weights in HF shards, and tied
embeddings get stored twice. Tensors are grouped by size, then by a sampled
digest, and only still-colliding groups are hashed in full (BLAKE2b-256), so
the extra reads stay proportional to data that really is duplicated. A
duplicate becomes a *ref chunk*: eight bytes naming the byte range it equals.
Refs resolve by decoding the source's own chunks straight from the archive,
so decompression stays a bag of independent jobs with no ordering.

**Growing an archive.** The savings that matter most are between files, not
inside them — duplicated tensors, and checkpoints that differ slightly — so
they only appear when related files share an archive. But a training run
produces checkpoints over hours, and recompressing every earlier one to add
the next is not a workflow anybody would use. `lmz append` codes a new file
against what the archive already holds, reading the base back by *decoding*
rather than from the original files, which may be long gone. Only the tail is
rewritten: payloads go where the old chunk table began, and the table,
manifest and footer are rebuilt after them. Growing a four-checkpoint series
one file at a time lands within a byte of compressing all four together
(11 358 532 against 11 358 531, 69.19% either way).

`lmz extract` pulls one member out without expanding the rest, which is the
other half of making an archive a place to keep things rather than a thing to
unpack.

One limit is worth stating plainly: a delta may only name a plain chunk, so
that resolving one stays a single hop. In a checkpoint series that leaves the
first checkpoint as the only base, and the difference grows with the distance
— measured 12.9 points at a 1000-step gap against 11.6 at 2000 — so later
checkpoints gain slightly less than a fresh archive would give them. Chaining
each to its predecessor would recover that at the cost of resolution walking
the chain.

**Delta coding.** Checkpoints from one training run are not duplicates, and
not independent either: every weight is rewritten and almost every one moves a
little, so dedup finds nothing while the difference is nearly all zeros. A
*delta chunk* is a ref chunk with data behind it — the same eight bytes naming
a source range, plus the XOR against it, coded by the ordinary path so it gets
the same plane split and per-plane adaptivity as anything else. Candidates are
tensors matching an earlier file by name, dtype and size, and each is decided
by encoding a megabyte of it both ways, so a pair that has drifted too far
declines by measurement rather than by rule. Sources are always the earliest
file holding the tensor, so a delta never points at another delta and
resolving one stays a single hop. The checksum is of the reconstructed bytes,
not of the difference.

**Splitting.** BF16 chunks are cut on the float's own field boundaries: one
plane takes the whole 8-bit exponent, the other takes the sign bit above 7
mantissa bits. GGUF block-quantised chunks are cut on their *struct* layout
instead — ggml's own field offsets, so a Q4_K block becomes `d`, `dmin`, the
packed sub-scales and the nibbles rather than 144 anonymous bytes. Each field
then picks its own treatment from exact histograms: one stream for a wide
quant array (128 near-identical tables would only cost overhead), one stream
per byte position for a narrow packed field (measured 15.6% recoverable
against 10.9% merged), and for a 2-byte fp16 the low byte coded per bucket of
the high byte. A Q4_K's quants take a fourth route: its eight sub-blocks each
carry their own 6-bit scale and min, which ggml packs across byte boundaries
and interleaves two-to-a-byte, so the kernel undoes both and codes each quant
against its own sub-block's class. Which way each field went is written into
the payload, so decoding needs no GGUF type table at all. Everything else is cut on byte
boundaries, one plane per byte position, which still separates exponents from
mantissas for FP16/FP32/FP64 and is what makes an FP32-upcast-from-BF16
checkpoint collapse.

**Conditioning.** The sign+mantissa plane is not quite independent of the
exponent: on real Llama weights the joint 16-bit entropy sits 0.075 bits per
element below the sum of the planes' own entropies. So the mantissa plane is
dealt into eight equal-mass exponent buckets, each entropy coded with its own
table — measured to capture essentially all of the dependence (a full 256-way
context gains nothing further). The bucket map is a pure function of the
exponent histogram and is never stored: the decoder recovers the exponent
plane first and rebuilds the identical map. The whole scheme is decided per
chunk from exact histograms and declines automatically (small chunks, no
correlation), which keeps it from ever costing bytes.

**Per-plane adaptivity.** Every plane is judged on its own: entropy is
estimated from a sample, genuine noise is stored without ever reaching the
coder, and a plane that cannot beat its own header is kept raw. If nothing in
a chunk compressed, the original bytes are stored so decoding needs no merge
at all. This is what lets one codec handle BF16, an FP32 upcast and INT8
without being told which is which.

**Entropy coding.** Planes go to a static order-0 rANS coder (12-bit
probabilities, 8 interleaved states, 16-bit renormalisation). Non-plane data —
JSON headers, config files, unrecognised formats — goes to zstd, where LZ
matching genuinely pays. A stream that zstd barely dents gets a second look
from rANS, which is what catches quantised INT8 tensors.

**Page-mapped archives.** The default 8 MiB chunk is right for compressing a
model once and restoring it once, and wrong for everything else: reading a
200-byte bias out of one decodes 8.39 MB, 32768 times what was asked for.
`--mapped` cuts the archive into 64 KiB blocks instead, which the chunk table
already indexes by destination offset, so `MappedArchive` answers any byte
range by decoding the one or two blocks it touches. 64 KiB is where the two
curves cross — 33.1% against 34.3% at 8 MiB, one point, while a block decodes
in 74 us instead of 14 ms. Below it the ratio falls away fast, and at 4 KiB
every block is stored raw because the frequency tables cost more than the data
saves. That last fact is not lmz's alone: a filesystem cannot compress a 4 KiB
cluster either, because it allocates whole 4 KiB blocks, so 4 KiB is the floor
for any compressed storage scheme, in the kernel or out of it.

`--align` additionally starts every block on a 4 KiB boundary. It costs the
padding it writes — measured 1.6 points on real BF16 weights — and repays it
only where reads bypass the page cache, so it is off by default. Both flags
are backward compatible: padding sits between payloads that the chunk table
addresses explicitly, so an older build reads these archives unchanged.

**Parallelism.** Each chunk records where it belongs in the output, so
decompression is a set of wholly independent jobs: N threads decode N chunks
and place them with positional writes, with no ordering and no locks.
Compression is the mirror image, with results appended in order so archives
are byte-identical regardless of thread count.

**Kernels.** The byte deinterleave is SIMD (AVX2/SSE2 on x86-64, NEON on ARM),
loaded via ctypes so it needs no Python headers and releases the GIL. Element
sizes above 2 are handled by repeated 2-byte deinterleaving, so a single
well-tuned kernel covers 2, 4 and 8-byte elements. It runs at about 3 GB/s per
core, comfortably ahead of the entropy coder. Without a compiler, numpy or
plain extended slicing take over; the tests check every backend agrees.

Findings that shaped the implementation, each of which cost more than it
looked like it would:

- **A general-purpose compressor cannot reach this data's entropy.** zstd -1
  lands 11% above the order-0 bound on exponent planes, and no setting closes
  it: levels 2 and 3 are *worse* (the LZ stage dilutes the Huffman stage), and
  the best variant found still sat 9.3% above. Replacing it with rANS was the
  single largest win.
- **A quantised block is a struct, and treating it as an array is what made
  lmz useless on quantised models.** Every published lossless weight
  compressor, and lmz through v0.3, hands a k-quant tensor to a coder as flat
  bytes. On real Q4_K that leaves 2.5% — indistinguishable from zstd -19's
  2.5%, which is what "already at entropy" looked like. The entropy is there;
  it is just spread across fields whose alphabets cancel when merged. Cutting
  on ggml's own struct offsets takes Q4_K to 5.4% and Q6_K to 3.9%, and the
  scales, not the quants, are where nearly all of it lives.
- **Q8_0 really is finished, and it took four independent probes to be sure.**
  Its quant payload — 32 of every 34 bytes — measures **7.64 of 8 bits** on
  real llama.cpp files. `xz -9e` scores 7.73 and `bzip2 -9` 7.90, both *worse*
  than a plain order-0 model, so there is no LZ or high-order structure to
  find. A full order-2 context over 65 536 tables recovers 0.057 bits and
  would spend 33 MB of tables doing it. Every context a decoder could actually
  rebuild — position in block, previous quant, block-scale class, column
  class, and their products — gains **≤0.022 bits**. Add the best of them to a
  perfect scale model and the ceiling is ~7.2%; lmz gets 7.0%. The only
  untaken structure is ggml's own invariant that `d = amax/127`, so every
  block holds a quant at ±127 (true in 100.00% of blocks), worth 0.638 bits
  per block — 0.234% of the file, and it needs sequential decoding with a
  shrinking alphabet to collect. Halving a Q8_0 file losslessly would mean
  3.94 bits per quant against a measured 7.64; it is not a coder problem.
- **A k-quant super-block is eight distributions, not one, and the min matters
  more than the scale.** Q4_K stores `d*q - m` per 32-weight sub-block, so a
  quant's alphabet depends on which sub-block produced it — and both
  parameters are decoded before the quants, making them a context that costs
  nothing to send. Two things hide it: ggml packs the parameters six bits at a
  time with four of each straddling a byte, and it interleaves two sub-blocks
  per quant byte, so every byte plane mixes two alphabets. Undoing both is
  worth 9.7 bits per block. Four scale classes by four min classes was the
  best buy once frequency tables are paid for; at a fixed sixteen streams,
  splitting the min four ways beats ignoring it and splitting the scale
  sixteen ways by half a point of the whole file. Finer contexts model better
  and lose anyway — eight by eight reaches 977.9 bits per block against
  979.5, and the 64 tables cost more than the 1.7 bits return.
- **XOR beat the clever difference.** IEEE bit patterns of same-sign floats
  are monotonically ordered as integers, so integer subtraction should track a
  small change *across* exponent boundaries where XOR flips high bits — and an
  order-preserving map (`w ^ 0x8000` if positive else `~w`) should extend that
  across zero. Both lost, on every fine-tune and every checkpoint pair
  measured. A borrow propagates into the high byte and destroys exactly what
  is being exploited: that the high byte usually does not change at all.
- **Adam's two moments could not be more different, and the recurrences say
  so before any data does.** `m = 0.9m + 0.1g` has a half-life of about 6.6
  steps; `v = 0.999v + 0.001g²` has a time constant of a thousand. So across a
  500-step gap the first moment is entirely refreshed and the second is mostly
  itself, and on a real AdamW state that is exactly what happens: `v` has
  57.7% of its bytes unchanged and deltas from 27.0% to **65.4%**, while `m`
  has 1.29% unchanged and deltas from 13.5% to **6.2%** — the difference of two
  independent values is *worse* than either alone, because XOR destroys what
  little structure each had. Averaged together they would have read as a
  mediocre win. Judged per tensor, which the encoder already does, lmz takes
  every `v` and declines every `m`: 384 chunks delta-coded, 39.0 MiB down to
  15.0 MiB, and the checkpoint pair goes from 15.8% to 26.1%. This is the
  reason the decision is made by measurement per tensor and not by a rule
  about what a file contains.
- **Quantised optimizer state deltas *better* than full-precision state**,
  which is the opposite of the intuition that precision is what a difference
  eats. No public run publishes fp32 Adam at two consecutive steps, so the
  optimizer was run here instead — exact Adam arithmetic over two million real
  fp32 parameters lifted out of a Pythia checkpoint. At a 1000-step gap `v`
  goes from 20.1% to 29.5% and `m` from 16.4% to 15.1%, so the shape matches
  the real 8-bit run exactly while the size of the win does not: 24.6% of `v`'s
  bytes are unchanged in fp32 against 57.7% in 8-bit. Full precision keeps
  moving every low mantissa bit every step; quantisation rounds those moves
  away entirely, and a byte that never changes is a byte a difference codes
  for nothing. The same sweep dates the first moment precisely — at a gap of
  one step delta still wins on `m` (17.8% against 16.4%), and by ten steps it
  has already lost, which is the 6.6-step half-life showing up on the clock.
- **The delta idea's home is checkpoints, not fine-tunes.** Measured against
  the right baseline — dedup+lmz, not lmz alone, or the delta takes credit for
  identical tensors that already cost eight bytes — a full fine-tune of
  Llama-3.2-1B gives 38.6% against 33.2%, and an SFT that leaves 23% of bytes
  untouched gives 49.1% against dedup's own 48.5%. Only a *surgical* edit
  reaches 70.8%. Between consecutive training checkpoints, though, not one
  tensor is byte-identical — dedup gets literally nothing — and the delta is
  worth 12.9 points, decaying only to 11.6 at twice the step gap. One
  measurement also found a 2.47 GB "fine-tune" on the hub that is byte-for-byte
  its own base model, which dedup already stored for free.
- **Pairing nibbles beat splitting them.** Coding each 4-bit quant as its own
  symbol gives the best entropy but doubles the symbols through the coder.
  Re-pairing two nibbles *of the same sub-block* into one byte keeps a
  256-entry table and byte-rate throughput, and measured *better*: 979.53
  bits per block against 979.90, because adjacent quants are independent so
  the pairing costs nothing.
- **Q5_K has the identical field layout and gains exactly nothing.** Its `qs`
  holds only the low four bits of a five-bit quant — the fifth is in `qh` —
  and the low bits of a peaked distribution are near-uniform whatever the
  sub-block does. Measured 0.000 points on real Llama Q5_K while the estimate
  alone cost 13% of encode time, so it is deliberately not registered.
  Measuring that was the difference between a feature and a tax.
- **Order-1 context modelling is worthless here** — 2.6444 bits versus 2.6449
  order-0, a 0.02% gain. Measuring that first saved building it. The same
  measurement pass killed two more tempting contexts on real Llama weights:
  conditioning an exponent on the one directly above it in the same column
  gains 0.0000 bits, and per-column-group tables gain 0.0002. The only
  structure that survives measurement is *within* the element — exponent to
  mantissa — worth 0.075 bits, which is what the bucket conditioning collects.
- **The bucket map must not be stored.** Deriving it from the exponent
  histogram on both sides costs one 256-entry integer scan and saves having
  any map bytes or versioning at all; eight equal-mass buckets capture the
  full 256-context conditional entropy to four decimal places.
- **The fixed-point reciprocal in the encoder was wrong, and 60 GB of
  weights never noticed.** The classic rANS reciprocal (exact below 2^31)
  was built for a coder whose renormalisation keeps states under that line;
  this one's 16-bit renormalisation lets states reach 2^20 x freq, so any
  symbol past 50% frequency can push a quotient one too high and write a
  neighbouring symbol's slot. Float planes never have a majority byte value,
  so every BF16 model round-tripped clean — the first Q8_0 file failed
  verification within seconds, on a norm plane that is 97.6% one value with
  the minority scattered (contiguous runs happen to dodge the bad states).
  The quotient is now a hardware division: exact everywhere, byte-identical
  output wherever the old path was right, and invisible in throughput at
  every thread count. Verification catching it is the system working; the
  regression test pins four seeds proven to break the old encoder.
- **Preallocating the output is worth ~9x.** Writing to a sparse file faults
  in and clears a page per write. `fallocate` costs 15 ms for 2 GiB on ext4
  and repays it many times over — but on tmpfs a "block" is a page of RAM, so
  the same call spends 1.3 s allocating memory to speed up writes that were
  already cheap. lmz reserves space on disk filesystems and leaves RAM-backed
  ones sparse.
- **Waiting on a future set is O(n) per completion.** Draining with
  `FIRST_COMPLETED` re-installs a waiter on every outstanding future each time
  one finishes; at a few dozen in flight that alone held decompression to
  roughly single-threaded speed. Completions now arrive on a queue.
- **Threading a decoder written in two languages goes backwards.** Decoding a
  block is 48 us of work against 1 us of pread, so a read is compute-bound and
  the rANS kernel releases the GIL. It still got *slower* with threads: 0.81
  GB/s on one, 1.60 on two, 0.70 on four. `decode_chunk` crossed into the
  kernel once per plane and again to merge, so a 64 KiB block handed the GIL
  round three times, and past two threads the interpreter spent its time on
  the handoff rather than on decoding. That glue is only 8.6% of a block --
  what mattered was not its cost but that it was *serialised*. Folding a whole
  chunk into one crossing (`lmz_decode_planes`) removed the inversion and is
  worth 9% single-threaded, but it did not buy a third or fourth thread: two
  is still the ceiling, at 0.90 / 1.73 / 1.69 / 0.85 GB/s on one, two, four
  and eight. Four *does* pay under `sys.setswitchinterval(50us)` against the
  5 ms default -- 1.69 to 2.05 -- which says the rest is handoff latency, not
  work. That setting is process-wide, so it belongs to whoever embeds lmz.
- **The lock came back as the next bottleneck, one level down.** With a chunk
  decoding in ~70 us, taking the block cache's lock twice per block put four
  threads *behind* two again. Reading it once for a whole run and writing it
  once at the end fixed that; a bulk read now touches the lock twice, not 256
  times.
- **How work reaches the pool mattered more than the pool.** Handing one
  future per 64 KiB block costs about what decoding the block costs, so the
  first attempt at parallel reads landed at 0.36 GB/s against 0.68
  single-threaded -- a 2x pessimisation dressed as an optimisation. Giving
  each thread a contiguous *run* of blocks instead turned the same two
  threads into 1.13 GB/s. Also worth 45%: a single-block read now slices the
  cached block directly, where assembling an output buffer for it had copied
  every byte twice for nothing.
- **The "obvious next step" was worth nothing, and measuring it cost less
  than building it.** This list used to end by saying that the shared output
  cursor serialises the eight interleaved states, and that giving each its own
  byte stream was the obvious fix. It is not. An isolated loop with one cursor
  per state runs at 4.89 cycles per symbol against 4.71 for the shared one --
  a hair *worse* -- and deleting the refill machinery altogether only reaches
  3.40, so the entire mechanism is a 1.4x ceiling rather than the 3-6x the
  argument assumed. That would have been a breaking change to the stream
  format in exchange for nothing. Two neighbouring ideas failed the same way:
  splitting the 16 KiB decode table into 4 KiB of slot-to-symbol plus 1 KiB of
  symbol-to-frequency costs 0.72x, because the second dependent load is dearer
  than the cache footprint it saves, and writing the output with non-temporal
  stores costs 0.94x, because the output is not what evicts the table.
- **The GIL was the whole ceiling, and a free-threaded interpreter removes it
  entirely.** Every thread cap in lmz exists because decoding is native work
  with a little Python between the calls, and that little is serialised. On
  CPython 3.14 free-threaded it simply is not. Cold 64 KiB blocks, one to
  sixteen threads:

  | | 1 | 2 | 4 | 8 | 16 |
  |---|---|---|---|---|---|
  | with the GIL | 0.90 | 1.71 | 1.59 | 0.83 | 0.83 |
  | free-threaded | 0.88 | 1.73 | **3.08** | **5.37** | **8.84** |

  Near-linear to sixteen: 9.8x over one thread and 5.1x over the best the GIL
  allows. It carries through the real paths -- `MappedArchive` reads go from
  0.69 to 2.87 GB/s, and a page-mapped archive decompresses at 2.24 against
  0.95 -- and all 58 tests pass there, so the caps lift themselves when
  `sys._is_gil_enabled()` says they can. This also puts the earlier dead ends
  in proportion: SIMD and per-state streams were chasing about 2x on one core
  while the interpreter was giving away 10x across the machine.
- **Two fixes that each did nothing, and together were worth 3x.** Small
  blocks made decompression six times slower -- 16k chunks through the
  pipeline where the default has 119. Grouping them into 4 MiB tasks looked
  like the obvious answer and moved nothing, because the interpreter's share
  is per *chunk*, not per task. Capping the worker count, once that was
  understood, also moved almost nothing on its own. On a 64 KiB-block archive,
  best of three at twelve threads:

  | | one chunk per task | batched into 4 MiB |
  |---|---|---|
  | **-j12** | 0.33 GB/s | 0.34 |
  | **capped at 2** | 0.40 | **1.04** |

  Batching gives each thread a run long enough to be worth holding the GIL
  for; the cap stops the threads fighting over it. Either alone leaves the
  other's bottleneck in place, which is why the first two attempts each read
  as a failure. Encoding has the same shape and takes the same pair --
  compressing to 64 KiB blocks went from 0.17 to 0.41 GB/s at twelve threads,
  while the 8 MiB default is untouched at 1.35.
- **The right thread count is a property of the chunk size, and the boundary
  is sharp.** A chunk costs a few microseconds of interpreter around whatever
  native decoding it carries, so at 8 MiB the Python is a rounding error and
  threads scale, while at 64 KiB it is a few percent and they collapse. The
  worker count now comes from the average chunk rather than from `-j`. Capped,
  256 KiB and 1 MiB chunks run at 1.31 and 1.39 GB/s; uncapped, at 1.23 and
  **4.29**. The first attempt put the threshold at 1 MiB and so capped the
  size that scales best, costing exactly the 3x it was meant to win. Single
  runs hid it: this measurement swings 30% run to run, and only best-of-three
  is stable enough to place a boundary with.
- **A decoder that looks three times faster is usually measuring cache.** The
  same kernel reports 4.65 cycles per symbol on a 32 KiB plane and 13.2 on a
  2 MiB one, which reads like a cliff worth chasing. It is not: the small case
  re-decodes one resident buffer thousands of times. Cold, on data streamed
  once -- which is the only case a decompressor ever sees -- it is flat at
  0.37-0.39 GB/s from 8 KiB to 4 MiB. This measurement fooled two separate
  attempts here before the cold version settled it.
- **Making rANS renormalisation branchless made decoding *slower*.** The
  explanation offered here for a long time was that the shared output cursor
  serialises each stream's refill behind the previous one. That explanation
  was wrong, as the entry above records; the refill is not where the cycles
  go. What remains true is only the measurement: the branchless variant lost.

## The LMZ protocol

The container is a specification, not an implementation detail: a versioned
format with numbered codecs, where every chunk records its own codec, element
width and destination, so any reader that understands the codec ids can
decode an archive without knowing which version wrote it. That is what makes
it worth calling a protocol rather than a file format — the same bytes are
read by the CLI, by `MappedArchive`, by the FUSE mount and by the filesystem,
and a v1 archive still opens today.

## Archive format

```
[32-byte header]  magic, version, original size
[chunk payloads]  in destination order
[chunk table]     32 bytes per chunk, zstd
[manifest]        JSON, zstd: members, tensor index, settings
[40-byte footer]  offsets + trailing magic
```

Each chunk record holds its archive offset, its destination offset, both
lengths, a crc32, the codec, the element size, and a 2-bit method per plane.
The table and manifest sit at the tail so writing streams in one pass, and
both are read up front so decompression can start anywhere.

Format v2 adds two codecs: `ref` (the payload is an 8-byte offset naming an
identical earlier range; integrity rides on the source chunks' checksums) and
`bf16-cond` (field split whose sign+mantissa plane is coded per exponent
bucket, methods and lengths self-described in the payload). v3 added
`q8-block`, a Q8_0-only block split. v4 replaces it with `blk-split`, whose
payload carries its own field table — block period, each field's offset and
width, and how each was coded — so one codec covers every GGUF quantisation
and a decoder needs no layout knowledge. v5 adds a fourth field mode, which
codes a k-quant's quants per sub-block class and records alongside them where
its context field is, how the block divides and how finely each parameter was
bucketed — so that too needs no ggml type table to reverse. v6 adds `delta`,
which names a source range like a ref and carries the coded difference from
it, with the inner codec written into the payload. This build still reads v1
to v5.

Integrity is checked at three levels: chunks must tile the output exactly with
no gap or overlap, each chunk carries a crc32 of its decoded bytes, and decoder
failures are reported as corruption rather than surfacing as backend errors.
Member paths are validated on extraction, so an archive cannot write outside
its destination directory.
