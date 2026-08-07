# lmz

**Smaller checkpoints. Read in place. Byte for byte.**

Fast lossless compression for large model weights — and a store you can read
a model out of without ever expanding it.

On Llama-3.1-8B-Instruct's BF16 shards, lmz removes **34.7%** — better than
any general-purpose compressor and better than the published state of the art
for model weights. On the model *directory* as Hugging Face actually ships it,
lmz removes **64.6%**, because the directory carries the same weights twice —
HF shards plus Meta's `original/` checkpoint — and lmz reads both containers
and stores each tensor once. Output is byte-for-byte identical to the input:
no quantisation, no approximation.

| | saved on Llama-3.1-8B (BF16 shards) |
|---|---|
| **lmz** | **34.7%** |
| [ZipNN](https://github.com/zipnn/zipnn) (published, same model) | 33.6% |
| [DFloat11](https://arxiv.org/pdf/2504.11651) (published) | ~30% |
| bzip2 -9 | 30.7% |
| xz -6 | 29.9% |
| zstd -19 | 23.6% |
| zstd -1 | 22.7% |
| gzip -6 | 21.2% |

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

## Quick start

No installation, no dependencies:

```
./lmz-cli compress  model.safetensors            # also .gguf, .bin, .pth
./lmz-cli decompress model.safetensors.lmz       # -> model.safetensors
./lmz-cli compress  ./my-model/                  # a whole directory; duplicate
                                                 #    tensors are stored once
```

Or keep the model compressed and read it where it lies:

```
./lmz-cli add   ./my-model/                      # into the store
./lmz-cli mount ~/models                         # ordinary files, decoded on read
```

Python 3.9+ is the only requirement. On a **free-threaded build** (3.13+ with
the GIL disabled) lmz lifts its own thread caps and decoding scales across the
whole machine — measured 8.84 GB/s against 1.73 on sixteen threads, verified
on CPython 3.14.7t with the full test suite. Nothing needs configuring; it
asks the interpreter. On Python 3.14+ zstd comes from the
standard library; on older versions `pip install zstandard` gets it, and
without either the tool falls back to deflate. A C compiler, if present, is
used once to build the SIMD kernel into the package directory — nothing is
installed system-wide and nothing outside the project is touched.

Check what is active with `./lmz-cli doctor`.

## Results

Measured on real checkpoints, AMD Ryzen 7 9800X3D, Python 3.14, zstd 1.5.7.

**Whole models**, compressed end to end, every roundtrip verified
byte-identical:

| model | size | compressed | saved |
|---|---|---|---|
| Llama-3.1-8B-Instruct, 4 BF16 shards | 16.06 GB | 10.49 GB | **34.70%** |
| Llama-3.1-8B-Instruct, whole directory | 32.13 GB | 11.38 GB | **64.60%** |
| Ministral-8B-Instruct, whole directory | 32.11 GB | 11.55 GB | **64.02%** |
| GLM-4V-9B (BF16, 15 shards) | 27.82 GB | 18.46 GB | **33.64%** |
| bge-m3 directory (FP32 `.bin` + onnx) | 4.59 GB | 2.45 GB | **46.48%** |
| Llama-3-8B NF4-quantised (bitsandbytes) | 5.70 GB | 4.87 GB | **14.55%** |
| Llama-3.1-8B-Instruct Q8_0 GGUF | 8.54 GB | 7.97 GB | **6.66%** |
| Pythia-160m, 2 consecutive checkpoints | 1.21 GiB | 450.8 MiB | **63.60%** |
| Pythia-160m, 3 consecutive checkpoints | 1.81 GiB | 644.0 MiB | **65.30%** |
| 2 consecutive AdamW optimizer states | 161.3 MiB | 119.3 MiB | **26.10%** |

The last three rows are v0.6's delta coding, and they are the case where
nothing else helps: between two checkpoints of one training run **not a single
tensor is byte-identical**, so dedup finds nothing and each checkpoint alone
codes at 57.5%. Coding the second as its difference from the first takes it to
69.7%. The optimizer row is a real 8-bit AdamW state from a public SFT run,
500 steps apart, and it is the one to read alongside the note on Adam's two
moments below — half of that state deltas superbly and the other half must be
refused, which is why the choice is made per tensor.

Delta coding is not a general win, and the same measurements say where it
stops. Between fine-tunes it is largely redundant with dedup, and against
LoRA adapters 500 steps into an SFT run it is worth only 16.5% → 19.9%,
because an adapter early in training moves far more per step than a converged
pretrained model does.

**Quantised GGUF**, where a general-purpose compressor gets almost nothing,
because the quantiser already removed what it could reach. The Q8_0 row is a
llama.cpp file; the Q4_K and Q6_K rows are built here from Llama-3.1-8B's real
BF16 weights through ggml's own block layouts, with a plain min/max scale
search rather than llama.cpp's error-minimising one — so the block structure
and the weight statistics are real, and only the scale search differs. All
three round-trip byte-identical:

| tensor type | lmz v0.3 (opaque bytes) | **lmz v0.4 (block split)** | zstd -19 | gzip -6 |
|---|---|---|---|---|
| Q8_0 | 6.66% | **6.66%** | 5.5% | 5.4% |
| Q4_K | 2.49% | **5.43%** | 2.52% | 2.17% |
| Q6_K | 0.54% | **3.94%** | 0.51% | 0.27% |

v0.5 adds sub-block conditioning, measured end to end on llama.cpp's own
GGUF builds of Llama-3.2-1B-Instruct rather than on blocks built here:

| file | v0.4 | **v0.5** | note |
|---|---|---|---|
| Q4_K_M, Q4_K tensors alone | 4.94% | **5.67%** | +0.73 points |
| Q4_K_M, whole file | 4.7% | **5.1%** | 61% of it is Q4_K, the rest Q6_K |
| Q6_K | 3.3% | 3.3% | nothing left to take |
| Q8_0 | 7.0% | 7.0% | at the entropy floor, see below |
| Qwen2.5-0.5B Q8_0 | 26.6% | **26.6%** | tied embeddings shipped twice |

That last row is the one to read twice. It is a quantised model at 26.6%, and
21 of those points are dedup: Qwen ties its embeddings, so `token_embd` and
`output` are byte-identical and the GGUF stores both. Entropy coding is not
where the large numbers live.

Each of those numbers has a reason. The two directory halvings are dedup:
Llama's repo ships HF shards *plus* Meta's `original/consolidated.00.pth`,
Ministral ships shards plus `consolidated.safetensors`, and in both cases
~14.5 GB of tensors are byte-identical across the two containers (attention
q/k differ — the HF conversion permutes them for its rope convention).
bge-m3's `pytorch_model.bin` holds FP32 that was trained in FP16: two of its
four byte planes are zeros and a third holds 3.0 bits, so the weights payload
alone drops 57.5%. The NF4 checkpoint is mostly *already* at entropy — its
BF16 embeddings compress 34%, its 4-bit payload barely at all.

The quantised GGUF rows are all block structure, not weight structure. A
quantised block is a *struct*: Q4_K packs two fp16 scales, twelve bytes of
6-bit sub-scales and 128 bytes of nibble pairs into 144 bytes. Handed to any
coder as a flat byte stream those alphabets land in one histogram and erase
each other, which is why zstd -19 stops at 2.5% — and why lmz stopped there
too until v0.4. Splitting the block on its own field boundaries separates
them, and the scales hold most of what is left: on Q4_K the fp16 `d`/`dmin`
planes are 46–51% recoverable and the packed sub-scales 16%, against 3.5% for
the quants that are 89% of the file.

**Throughput**, 1.09 GiB Llama shard, RAM-backed filesystem, including all I/O
and per-chunk checksums, best of 3:

| threads | compress | decompress |
|---|---|---|
| 1 | 0.31 GiB/s | 0.35 GiB/s |
| 4 | 1.18 GiB/s | 1.18 GiB/s |
| 8 | **1.92 GiB/s** | **1.76 GiB/s** |
| 16 | 1.71 GiB/s | 1.64 GiB/s |

Exponent conditioning costs ~20% single-threaded next to v0.1's plain field
split; at 8 threads the job is memory-bound and the difference disappears.
For scale, ZipNN publishes 1.15 GB/s compress and 1.65 GB/s decompress on this
model — so lmz is ahead on ratio and on both speeds, though that comparison
crosses different hardware and should be read loosely.

Throughput peaks at 8 threads and falls off after: with source, archive and
output all in flight the job goes memory-bandwidth bound. On ordinary disks
the limit arrives sooner still — writing to ext4 with an fsync runs entirely
I/O bound, which is the point. The archive is a third smaller, so a
storage-bound load moves a third fewer bytes.

**Other dtypes** (synthetic checkpoints, `./lmz-cli bench <file>` reproduces
any row):

| weights | lmz | zstd -1 | gzip -6 |
|---|---|---|---|
| BF16 | **31.0%** → see above for real weights | 21.1% | 20.0% |
| FP16 | **14.1%** | 7.2% | 7.0% |
| FP32 | **15.5%** | 7.0% | 6.9% |
| FP32 upcast from BF16 | **65.5%** | 51.4% | 52.7% |
| INT8 quantised | 0.0% (stored) | 0.0% | 0.0% |

Two of the dtype rows below deserve comment. A checkpoint saved as FP32 but trained in
bfloat16 has 16 zero mantissa bits in every value; lmz finds two entirely
empty planes and removes two thirds of the file. Already-quantised INT8
weights have no structure left to exploit — lmz detects that from a sample,
stores them unchanged, and spends almost no time proving it, rather than
running them through a coder that cannot help.

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

## Reading a model without expanding it

Compressing a model is only half of an answer. The other half is that
something has to *read* it, and until now that meant expanding the archive
back onto the disk you were trying to save — paying the full size again,
transiently, before the first token.

`lmz mount` removes that step. Models live in a store, compressed; the mount
presents them as the ordinary files a runtime expects, and decodes the blocks
a reader actually touches on the way out.

```
lmz add ./Llama-3.1-8B-Instruct        # compressed once, into the store
lmz models                             # what is there, and what it costs
lmz mount ~/models                     # served as ordinary files
llama-cli -m ~/models/Llama-3.1-8B-Instruct/model.gguf
```

Nothing is patched and nothing links against lmz. `llama.cpp`, vLLM and
`transformers` call `open()`, `read()` and `mmap()` and get the original
bytes, because that is all the mount is: a filesystem the kernel talks to.

There is no libfuse and nothing to install. `fusermount3` performs the
privileged mount and passes back a descriptor over a Unix socket; after that
the kernel's FUSE protocol is a sequence of fixed structs, so the whole
server is `struct.pack` and `os.read` — the same standard-library-only rule
the rest of the package follows.

### What it costs

Measured on the 1.74 GiB BF16 model above, AMD Ryzen 7 9800X3D, reading
through the mount by concurrent readers.

| server | 1 reader | 2 | 4 | 16 |
|---|---|---|---|---|
| CPython 3.14 (GIL) | 602 | 689 | 694 | 813 |
| **free-threaded 3.14t** | **1476** | **1478** | **2490** | **3000** |
| plain uncompressed file | 5492 | 43571 | 47104 | 23624 |
| lmz mount, page cache warm | 24135 | | | |

**The mount is still not faster than reading an uncompressed file on this
machine, and the table says so.** But it is now 2.8x faster than it was for a
lone sequential reader and 4.4x for concurrent ones, and 3.0 GB/s is past the
point where storage is usually the thing you are waiting for. The crossover
the random-access reader already documents still applies: decoding beats
phone flash, eMMC, an SD card and a network filesystem, and it loses to a
fast local SSD.

The `plain file` row cannot be measured honestly here and should not be
quoted: under WSL2 `posix_fadvise(DONTNEED)` does not reach the Windows
host's own cache, and the same file measured 1843 and 5846 MiB/s on
successive "cold" runs.

What the mount wins outright is the part that is not a throughput number:

| | ready to read | disk needed |
|---|---|---|
| `lmz mount` | 15 ms | 1.19 GiB |
| `lmz decompress` first | 2062 ms | 2.93 GiB |

A model that is never expanded never costs its expanded size, and the second
read of a tensor costs nothing at all: the kernel caches the *decoded* pages,
which is where the 24 GB/s row comes from. Repeated loads — the normal case
for a model you actually use — are served from that cache without decoding
anything.

### Where the speed came from

Three things, each of which had to be measured before it was believed.

**Free-threading is most of it.** The decode path is native work with a
little Python between the calls, and it is that Python which serialises. On a
free-threaded build the caps lift by themselves — `lmz` asks the interpreter
rather than being told — and the same mount goes from 813 to 3000 MiB/s.
Under the GIL the server is best at **two** threads, and the measurement is
emphatic: two serve 670 MiB/s where four serve 349 and eight 337, the same
inversion decompression already documents.

**A single sequential reader cannot be helped by threads at all**, because
the kernel will not ask for the next block until this one is answered. Worse,
buffered reads arrive at 128 KiB however large the caller's read is: the
readahead window is capped at `VM_MAX_READAHEAD`, and `process_init_reply`
takes the *smaller* of that and whatever a filesystem asks for, so a mount can
only lower it. A 1.74 GiB read is 14299 round trips no matter what.

So the mount predicts instead of waiting. A stream is recognised when a read
begins where the last one ended, and the window beyond it is decoded on idle
cores while the current reply is being written — 687 to 1476 MiB/s. Two
details decide whether that works at all, and getting either wrong made it
*slower* than no prefetch:

- The window is cut into slices across the pool. One task decoding 4 MiB is
  serial, so it finishes a window behind the reader it is meant to be ahead
  of — measured 1.6x slower than not prefetching.
- A block already being decoded is waited on, not decoded again. Without
  that, the prefetch and the reader race for the same blocks and the
  speculation costs exactly what it saves. With it, serving the whole model
  decodes 28579 blocks where the file contains 28577.

**Reading ahead is switched off when the server is busy.** Speculation is
only free while cores are idle; with several readers already saturating the
decoder it took 17% away. The mount reads ahead only while two or fewer
requests are in flight, which is exactly the lone-reader case that was
starved, and leaves concurrent readers alone.

Bigger blocks were tried and rejected. They compress slightly better, but a
1 MiB request straddles two 1 MiB blocks and decodes twice what it serves:

| block | saved | 1 reader | 16 readers | 4 KiB read amplification |
|---|---|---|---|---|
| **64 KiB** | 32.02% | **519** | **2680** | 16x |
| 256 KiB | 32.54% | 263 | 1898 | 64x |
| 1 MiB | 32.66% | 92 | 750 | 256x |

## Command line

```
lmz compress    <input> [output]   -l LEVEL  -j N  --chunk-size N  --no-checksum
                                   --no-dedup  --no-delta  --mapped  --align  -f
lmz append      <archive> <input>  -l LEVEL  -j N  --no-checksum  --no-delta
lmz extract     <archive> <member> <output>  -f
lmz decompress  <input> [output]   -j N  --no-verify  -f
lmz verify      <archive>          -j N
lmz info        <archive>          --tensors  --json  --limit N
lmz cat         <archive> <tensor> -o FILE  --member FILE
lmz bench       <file>             --bytes N
lmz doctor

lmz add         <path>             --name N  -l LEVEL  -j N  -f   --store DIR
lmz models                         --json                        --store DIR
lmz rm          <name>                                           --store DIR
lmz mount       <mountpoint>       --model N  -d  --allow-other  --store DIR
                                   --cache-blocks N  --verify  -j N  --readahead N
lmz unmount     <mountpoint>
```

The store lives at `$LMZ_HOME`, else `~/.lmz`: a directory of page-mapped
archives plus a JSON index. The index exists only so that listing the store
does not mean opening every archive in it, and `Store.rebuild()` reconstructs
it from the archives alone if it is lost.

`info` reports what the codec actually did, which is the quickest way to see
why a file compressed the way it did:

```
chunk codecs
  bf16-split   140 chunks    1.09 GiB -> 730.89 MiB  1.524x
  entropy        2 chunks   12.54 KiB ->   1.82 KiB  6.885x
  stored         1 chunks       568 B ->      568 B  1.000x
```

`cat` pulls a single tensor out of an archive by decoding only the chunks it
overlaps, without expanding the rest.

## Python API

```python
import lmz

stats = lmz.compress("model.safetensors", "model.lmz")
print(f"{stats.ratio:.3f}x, {stats.saved:.1%} smaller, {stats.seconds:.1f}s")

lmz.decompress("model.lmz", "restored.safetensors")
lmz.verify("model.lmz")

meta = lmz.info("model.lmz")                       # members, tensors, codecs
dtype, shape, raw = lmz.read_tensor("model.lmz", "model.embed_tokens.weight")
```

`MappedArchive` is the random-access reader:

```python
with lmz.MappedArchive("model.lmz") as arc:      # written with mapped=True
    head = arc.read(0, 4096)                     # any byte range
    dtype, shape, raw = arc.tensor("model.embed_tokens.weight")
    print(arc.decoded_bytes)                     # what it actually expanded
```

On a 942 MiB BF16 model, 200 random 4 KiB reads take 20 ms against 3.9 s from
an 8 MiB-chunk archive, and expand 17x what was asked for rather than 2025x.
Reading runs at 1.22 GB/s, which is the honest ceiling and worth
being precise about: it beats phone flash, and it does not beat an NVMe drive.
Compression buys a cold load a third fewer bytes to move; it does not make
inference faster, and a plain mmap of an uncompressed file still wins on
random access.

```python
lmz.append("run.lmz", "checkpoint-9000.safetensors")   # code against what is there
lmz.extract("run.lmz", "checkpoint-3000.safetensors", "ck3000.safetensors")
```

The store is the same reader with a name attached, plus somewhere to put it:

```python
store = lmz.Store()                              # $LMZ_HOME, else ~/.lmz
store.add("./Llama-3.1-8B-Instruct")             # compressed, page-mapped
for e in store.models():
    print(e.name, e.stored_size, f"{e.saved:.1%}")

with store.open("Llama-3.1-8B-Instruct") as arc:   # no mount, no expansion
    dtype, shape, raw = arc.tensor("model.embed_tokens.weight")
```

`mount` builds a server the caller runs; it must be its own process, because
a thread that page-faults on its own mount blocks in the kernel while holding
the GIL, and the thread that would answer the fault can never run:

```python
server = lmz.mount("/home/me/models")            # optionally names=[...]
server.serve()                                   # blocks until unmounted
```

`Store` takes `root`; `add` takes `name`, `level`, `workers`, `force`,
`block_size` and `progress`; `mount` takes `store`, `names`, `threads`,
`allow_other`, `cache_blocks`, `verify` and `readahead` (0 disables it).

`MappedArchive.prefetch(offset, length)` decodes a range into the cache and
returns how many blocks it had to decode, which is what the mount's readahead
is built on. A reader that knows where it is going next can use it directly:

```python
with lmz.MappedArchive("model.lmz", cache_blocks=256) as arc:
    arc.prefetch(offset, 4 << 20)        # on another thread, ahead of the read
    data = arc.read(offset, 1 << 20)     # now a cache hit
```

It is safe to call from any thread against a shared reader: a block already
being decoded elsewhere is waited on rather than decoded twice.

`compress` takes `level`, `workers`, `chunk_size`, `checksum`, `dedup`,
`delta`, `mapped`, `align` and `progress`; `append` takes `level`, `workers`,
`checksum`, `delta` and `progress`; `decompress` takes `workers`, `verify_checksums`, `overwrite` and
`progress`. `progress` is called with `(bytes_done, total)`.

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

## Limitations

- **Lossless only.** Ratios are bounded by the real entropy of the weights.
  On BF16 the sign and mantissa are measured at 7.92 of 8 possible bits on
  real Llama weights — genuine noise — which caps *any* lossless method near
  35%, and lmz's conditioned coder sits within ~0.2 points of that joint
  bound. Halving a BF16 checkpoint losslessly is information-theoretically
  impossible; the halvings in the table above come from data that really is
  redundant (duplicated tensors, FP32 containers holding 16-bit values). If
  you need 4x on the weights themselves, you need quantisation, which is a
  different and lossy tool.
- **rANS archives need the native kernel.** It builds automatically wherever
  a C compiler exists. Without one, compression falls back to zstd (~31%
  instead of ~34%) and archives already written with rANS cannot be read —
  `./lmz-cli doctor` reports which backend is live.
- **Already-quantised weights are mostly entropy.** Raw INT8/FP8 checkpoints
  are detected and stored unchanged. Block-quantised GGUF keeps a few percent
  in its scale and sub-scale fields, which the block codec collects, but 5–7%
  is the honest ceiling on the weights themselves and it is measured, not
  estimated: Q8_0's quant payload sits at 7.64 of 8 bits, and lmz lands within
  0.2 points of what any lossless coder could reach. Quantisation *is* lossy
  compression, and lmz only codes what it left behind. Large numbers on
  quantised models come from duplication instead — 26.6% on a Q8_0 Qwen2.5
  whose tied embeddings ship twice — so it is worth pointing lmz at the whole
  repository rather than one file.
- **Delta coding needs the source in the same archive.** A checkpoint stored
  as a difference cannot be restored without the file it was coded against, so
  the whole series has to travel together. That suits a training run or a
  repository and does not suit shipping one model to one user. It also needs
  matching tensor names, dtype and size: a fine-tune that switches from BF16
  to F16 has no byte-level relationship to its base at all, and gets nothing.
- **No GPU path.** DFloat11 and NeuZip decompress on the GPU so a model that
  doesn't fit in VRAM can still run; lmz only produces files and bytes. For
  that use case they win regardless of ratio.
- **Decompressing to a file is I/O bound** on real storage. The gain there is
  that there are a third fewer bytes to move.
- **The mount is slower than an uncompressed file on fast local storage.**
  1.5–3.0 GB/s of decoding free-threaded, and 0.6–0.8 under the GIL, against
  several GB/s of NVMe. It wins on the disk it saves, on never expanding
  anything, and on repeat reads out of the page cache — not on cold
  sequential throughput. Where the storage is slower than the decoder, which
  is most on-device storage, it wins there too.
- **Under the GIL the mount is roughly a quarter of its free-threaded speed**
  (813 against 3000 MiB/s) and reading ahead is disabled, because a prefetch
  thread can only take the interpreter lock away from the thread whose reply
  the reader is waiting on. This is the one place where the interpreter build
  changes what lmz can do rather than just how fast it does it.
- **Mounting is Linux-only and needs `fuse3`.** `fusermount3` must be on PATH
  and `/dev/fuse` readable; `lmz doctor` reports whether it is. The store and
  its random-access reader work everywhere, mount or no mount. Serving mmap
  page faults directly through `userfaultfd` would avoid FUSE entirely, but
  it needs `vm.unprivileged_userfaultfd=1`, which is off on most distributions.
- **The mount must be its own process.** A thread that page-faults on an mmap
  of its own mount blocks inside the kernel while still holding the GIL, so
  the thread that would answer the fault never runs. `lmz mount` is a separate
  process by construction; embedding the server in a process that also reads
  the mount will deadlock.
- **A large store costs memory to mount.** The chunk table is parsed into
  Python objects at open: 305 bytes per 64 KiB block, which is 8 MiB for the
  1.74 GiB model here and extrapolates to ~0.6 GiB for a 70B one, taking
  about 11 s. That is the next thing to fix and it is a real limit today.
- GGUF block-quantised tensors are split on their ggml struct layout, which
  covers Q4_0 through Q8_1, every k-quant, the IQ types and the ternary ones.
  A layout this build does not recognise falls back to opaque bytes.

## Tests

```
python3 tests/test_lmz.py          # also runs under pytest
```

76 tests covering kernel equivalence across all backends, element sizes and
block periods,
rANS round-trips over adversarial distributions (including single-symbol
streams, which exposed a frequency-field overflow), rANS landing within 2% of
measured entropy, BF16 field-split reversibility, bucket partition
reversibility and native/Python agreement, conditioned-BF16 round-trips (and
that conditioning declines on uncorrelated data), codec round-trips per
dtype, block round-trips for Q8_0 and every k-quant with all four field
modes exercised, that every declared GGUF field layout tiles its block and
agrees with the ggml type table, that the sub-block kernels invert themselves
and reproduce ggml's own `get_scale_min_k4` value for value, that sub-block
conditioning is taken only when it wins and declines when the quants owe
nothing to their sub-block, that a damaged sub-block descriptor is rejected
rather than acted on, that structureless blocks decline the split,
that v3's Q8_0-only block payload still decodes,
safetensors/GGUF/PyTorch-zip/raw layout detection, tensor dedup within
and across files, delta coding against a near-copy (chosen, reversed, and
declined when the files are unrelated), hostile ref and delta chunks
(self-referencing, out of range, damaged difference), v1
archive compatibility, file and directory round-trips compared by hash,
deterministic output across thread counts, tensor extraction, corruption and
truncation detection (damage to a block payload is either rejected or lands
on a byte no decoder consults — never silently wrong output),
page-mapped random access (every byte range matching the original, a one-byte
read expanding no more than one block, and the cache never changing what is
returned), that aligned archives really do start every block on a page
boundary and still decompress and verify by the ordinary paths,
growing an archive with `append` (matching a one-shot compression, refusing a
name already present, leaving a rejected append's bytes untouched, and keeping
a page-mapped archive page-mapped), single-member `extract`,
that coalescing a run of block payloads into one read returns what reading
them one at a time returns, that prefetching changes no byte of what is
subsequently read and costs nothing when the blocks are already cached,
that eight threads sharing one reader over deliberately overlapping ranges
against a cache far too small never deadlock and never disagree with the
original, that the readahead predicts only genuinely sequential streams and
its frontier never re-queues a slice it has already asked for,
that a mount serves identical bytes with readahead on and off,
the store (add, list, remove, a refused overwrite leaving the first copy
intact, name normalisation rejecting empty and traversing names, rebuilding a
lost index from the archives, and reading a tensor out of a stored model),
the mount's node tree over nested member paths and its refusal of `..`,
and the mount itself, served from a separate process and compared against the
source directory file by file — every member's sha256, random reads at page
boundaries and at the end of a file, serving only the named models, agreement
with a full `decompress`, and that writes, unlinks and mkdirs are all refused.
Mount tests skip themselves where FUSE is unavailable rather than failing.
Also path-traversal rejection, and the CLI.

`tests/make_model.py` generates synthetic checkpoints with per-channel
lognormal scaling, which reproduces the exponent skew of trained weights —
measurements against uniform random floats would not.

## Buy me a coffee

lmz is free and unfunded. If it saved you disk or bandwidth —
[**☕ Buy me a coffee**](https://buymeacoffee.com/fanxinsun), or
[Alipay](assets/alipay.jpg) (打开支付宝，扫一扫).

