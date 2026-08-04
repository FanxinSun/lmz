# lmz

Fast lossless compression for large model weights.

On Llama-3.1-8B-Instruct, lmz removes **34.2%** — better than any
general-purpose compressor and better than the published state of the art for
model weights. Output is byte-for-byte identical to the input: no
quantisation, no approximation.

| | saved on Llama-3.1-8B (BF16) |
|---|---|
| **lmz** | **34.2%** |
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

That combination is the whole trick, and it is worth 2.6 points over the
byte-splitting approach lmz started with:

| encoder on real Llama BF16 | saved |
|---|---|
| byte split + zstd (where this project began) | 31.6% |
| byte split + rANS on both planes | 34.0% |
| **field split + rANS** | **34.4%** |
| theoretical joint-entropy bound | 35.0% |

## Quick start

No installation, no dependencies:

```
./lmz-cli compress  model.safetensors            # -> model.safetensors.lmz
./lmz-cli decompress model.safetensors.lmz       # -> model.safetensors
./lmz-cli compress  ./my-model/                  # a whole model directory
```

Python 3.9+ is the only requirement. On Python 3.14+ zstd comes from the
standard library; on older versions `pip install zstandard` gets it, and
without either the tool falls back to deflate. A C compiler, if present, is
used once to build the SIMD kernel into the package directory — nothing is
installed system-wide and nothing outside the project is touched.

Check what is active with `./lmz-cli doctor`.

## Results

Measured on real checkpoints, AMD Ryzen 7 9800X3D, Python 3.14, zstd 1.5.7.

**Whole models**, all shards, compressed end to end:

| model | size | compressed | saved |
|---|---|---|---|
| Llama-3.1-8B-Instruct (BF16) | 16.06 GB | 10.57 GB | **34.18%** |
| Ministral-8B-Instruct (BF16) | 16.04 GB | 10.60 GB | **33.89%** |
| GLM-4V-9B (BF16, 5 shards) | 9.47 GB | 6.27 GB | **33.80%** |

**Throughput**, 1.09 GiB Llama shard, RAM-backed filesystem, including all I/O
and per-chunk checksums:

| threads | compress | decompress |
|---|---|---|
| 1 | 0.40 GB/s | 0.40 GB/s |
| 4 | 1.54 GB/s | 1.35 GB/s |
| 8 | **1.92 GB/s** | **1.83 GB/s** |
| 16 | 1.63 GB/s | 1.56 GB/s |

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

Two of those deserve comment. A checkpoint saved as FP32 but trained in
bfloat16 has 16 zero mantissa bits in every value; lmz finds two entirely
empty planes and removes two thirds of the file. Already-quantised INT8
weights have no structure left to exploit — lmz detects that from a sample,
stores them unchanged, and spends almost no time proving it, rather than
running them through a coder that cannot help.

## How it works

**Planning.** The tensor index is read from the container (safetensors and
GGUF are both understood) to recover the dtype layout, so every chunk holds
elements of a single known width. Headers, padding and unrecognised formats
become 1-byte-element regions, which still compress, just without the split.
Runs of adjacent same-dtype tensors are coalesced first, so that a model's
thousands of tiny bias and norm tensors don't each become an undersized chunk.

**Splitting.** BF16 chunks are cut on the float's own field boundaries: one
plane takes the whole 8-bit exponent, the other takes the sign bit above 7
mantissa bits. Everything else is cut on byte boundaries, one plane per byte
position, which still separates exponents from mantissas for FP16/FP32/FP64
and is what makes an FP32-upcast-from-BF16 checkpoint collapse.

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
- **Order-1 context modelling is worthless here** — 2.6444 bits versus 2.6449
  order-0, a 0.02% gain. Measuring that first saved building it.
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
- **Two plausible rANS optimisations did not pay.** Making renormalisation
  branchless made decoding *slower*; the real serialisation is the shared
  output cursor, which forces each stream's refill load to wait on the
  previous one. Giving each state its own byte stream would fix it and is the
  obvious next step.

## Command line

```
lmz compress    <input> [output]   -l LEVEL  -j N  --chunk-size N  --no-checksum  -f
lmz decompress  <input> [output]   -j N  --no-verify  -f
lmz verify      <archive>          -j N
lmz info        <archive>          --tensors  --json  --limit N
lmz cat         <archive> <tensor> -o FILE  --member FILE
lmz bench       <file>             --bytes N
lmz doctor
```

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

`compress` takes `level`, `workers`, `chunk_size`, `checksum` and `progress`;
`decompress` takes `workers`, `verify_checksums`, `overwrite` and `progress`.
`progress` is called with `(bytes_done, total)`.

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

Integrity is checked at three levels: chunks must tile the output exactly with
no gap or overlap, each chunk carries a crc32 of its decoded bytes, and decoder
failures are reported as corruption rather than surfacing as backend errors.
Member paths are validated on extraction, so an archive cannot write outside
its destination directory.

## Limitations

- **Lossless only.** Ratios are bounded by the real entropy of the weights.
  34.4% on BF16 is within 0.6 points of the joint-entropy bound of 35.0%, so
  there is very little left on the table for *any* method of this kind. If you
  need 4x, you need quantisation, which is a different and lossy tool.
- **rANS archives need the native kernel.** It builds automatically wherever
  a C compiler exists. Without one, compression falls back to zstd (~31%
  instead of ~34%) and archives already written with rANS cannot be read —
  `./lmz-cli doctor` reports which backend is live.
- **Already-quantised weights don't compress.** INT8/FP8 checkpoints are
  detected and stored unchanged. This is the correct outcome, not a failure,
  but it means lmz has little to offer them.
- **No GPU path.** DFloat11 and NeuZip decompress on the GPU so a model that
  doesn't fit in VRAM can still run; lmz only produces files and bytes. For
  that use case they win regardless of ratio.
- **Decompressing to a file is I/O bound** on real storage. The gain there is
  that there are a third fewer bytes to move.
- GGUF is parsed for its dtype layout; quantised GGUF blocks are treated as
  opaque bytes, which is where their entropy already is.

## Tests

```
python3 tests/test_lmz.py          # also runs under pytest
```

24 tests covering kernel equivalence across all backends and element sizes,
rANS round-trips over adversarial distributions (including single-symbol
streams, which exposed a frequency-field overflow), rANS landing within 2% of
measured entropy, BF16 field-split reversibility, codec round-trips per dtype,
safetensors/GGUF/raw layout detection, file and directory round-trips compared
by hash, deterministic output across thread counts, tensor extraction,
corruption and truncation detection, path-traversal rejection, and the CLI.

`tests/make_model.py` generates synthetic checkpoints with per-channel
lognormal scaling, which reproduces the exponent skew of trained weights —
measurements against uniform random floats would not.
