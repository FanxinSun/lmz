# Limitations, and the tests

*Where lmz is not worth using, stated plainly, and what the suite actually checks.*

[← back to the README](../README.md)

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
- **Already-quantised weights are mostly entropy — with one exception.**
  Raw INT8 checkpoints are detected and stored unchanged. **FP8 is not one of
  them**: `F8_E4M3` is a float, 1 sign + 4 exponent + 3 mantissa, and trained
  weights skew its exponent just as they skew BF16's. Measured on a real
  Qwen3.6-27B-FP8 shard (99.7% `F8_E4M3`), lmz removes **17.1%**, byte-exact.
  There is almost nothing left after that: the order-0 bound is 17.7%, and
  zstd -3 and zstd -19 both reach 17.1% too — the same ratio, but lmz takes
  0.7s where zstd -19 takes 57.7s. Context does not rescue it either; block
  magnitude class is worth 1.0 point, position within a block exactly zero,
  and the previous code 0.1. So FP8 is worth compressing and lmz holds no
  real advantage in doing it. Block-quantised GGUF keeps a few percent
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
- **The GPU decoder ships, but nothing routes to it yet.** `lmz.gpu` is in
  the package and builds itself with `nvcc` on first use, the same bargain
  `lmz.native` makes with a C compiler: the wheel carries a `.cu` and no
  CUDA, installing needs no toolkit, and a machine with neither nvcc nor a
  card decodes on the CPU exactly as before. `lmz doctor` says which. What it
  gives you is one call — `lmz.gpu.decode_batch`, a batch of lmz rANS streams
  in, their plaintext out, verified byte-identical to `lmz_rans_decode` over
  936 MB of real planes. What it does not give you is a faster
  `lmz decompress`: nothing in the archive path calls it, because the useful
  thing to do with a GPU decode is leave the result in VRAM, and deciding
  when to do that belongs to the layer above. Two further limits are real.
  **One stream is 8 lanes of work**, so a GPU is filled by many streams at
  once and never by one, however large — that is the format's 8-state
  interleave, not the kernel's. And **an archive written today codes a
  frequency table per chunk**, which costs 3.8× against a table shared across
  chunks: 111 GB/s versus 418 on the same 936 MB. The shared table is a
  format option that has not landed; it is item 1 of
  [gpu-residency-handover.md](gpu-residency-handover.md). The Apple silicon
  port under `scratchpad/gpu/metal/` is still written and never run.
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
- **CPython's bundled zstd is slow on macOS.** Measured on a GitHub
  `macos-latest` runner: 103 MiB/s compressing a 64 MiB BF16 sample through
  `compression.zstd`, against 767 MiB/s for the same work through the
  `zstandard` package on an Apple Silicon Mac, and 673 MiB/s for the same
  stdlib module on Linux. lmz hands libzstd whole multi-megabyte chunks, so
  binding overhead cannot explain a gap that size. On Darwin lmz therefore
  prefers the `zstandard` package when it is installed, and falls back to the
  stdlib when it is not. This affects generic data and `lmz fs`, which code
  everything with zstd; it does not affect model weights, where the plane split
  sends the compressible bytes to rANS and zstd sees close to none of them.
  `lmz doctor` prints which binding is in use.

## Tests

```
python3 tests/test_lmz.py          # also runs under pytest
```

99 tests covering kernel equivalence across all backends, element sizes and
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
and the read-write filesystem: that a model written through it comes back
byte for byte and really is stored compressed, that writing and immediately
reading a file wins the race against the kernel's asynchronous release,
that modes, sizes, directories, rename and unlink all survive, that
incompressible and tiny files are stored raw instead of being wrapped in a
container, and that a rewrite replaces the contents and leaves exactly one
backing form behind.
Mount tests skip themselves where FUSE is unavailable rather than failing.
Also path-traversal rejection, and the CLI.

The GPU decoder is checked the only way a decoder can be: that it reproduces
what `lmz_rans_decode` produces, over streams built by lmz's own encoder and
packed the way an archive packs them -- unpadded and unaligned, which is what
catches a `cp.async` source alignment bug. Also that it declines rather than
guesses on a shape it cannot take, that corruption raises instead of decoding
anyway, and that asking `backends()` a question does not run a compiler as a
side effect. The distributions are the ones that break tables rather than the
one the kernel was tuned on -- a single symbol at the full probability scale,
two symbols, near-uniform, and one dominant with a tail -- against batch sizes
that leave a partial block. These skip themselves where there is no GPU; the
one that does not is that a card below the kernel's floor is declined with its
compute capability in the message rather than handed to a compiler that will
reject it. The decoder also checks itself: the first thing it does on any
machine is decode a stream that machine just encoded and compare against the
CPU decoder, and a device that disagrees is not used. `lmz doctor --gpu-verify`
runs the whole set on demand and prints a report, because the only evidence
that will ever exist for an architecture nobody here owns is somebody else's.
And that loading the CUDA library cannot kill the caller: a driver that is
mid-upgrade faults inside `ctypes.CDLL` with no return code involved, so the
first load happens in a child process, and the test kills one to prove the
crash is reported rather than propagated.

`tests/make_model.py` generates synthetic checkpoints with per-channel
lognormal scaling, which reproduces the exponent skew of trained weights —
measurements against uniform random floats would not.
