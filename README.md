# lmz

[![tests](https://github.com/FanxinSun/lmz/actions/workflows/tests.yml/badge.svg)](https://github.com/FanxinSun/lmz/actions/workflows/tests.yml)
[![Python](https://img.shields.io/pypi/pyversions/lmzip?cacheSeconds=3600)](https://pypi.org/project/lmzip/)
[![PyPI](https://img.shields.io/pypi/v/lmzip?cacheSeconds=3600)](https://pypi.org/project/lmzip/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Smaller checkpoints. Byte for byte.**

Lossless compression built for model weights. `zstd -1` takes 22.7% off a
Llama-3.1-8B BF16 checkpoint. lmz takes **34.7%** — and **64.6%** off the
directory as Hugging Face actually ships it, which is **13 GB more than zstd
on one 8B model**.

Nothing is approximated. Every byte comes back.

And the decoder now runs on the GPU — **111 GB/s** from an ordinary archive
against a 28.8 GB/s PCIe link, shipped in the wheel. [Jump to it](#on-a-gpu).

```
pip install lmzip
lmz compress ./Llama-3.1-8B-Instruct/
```

## What it saves

Real checkpoints, every round-trip verified byte-identical.

| | size | after | saved | zstd -1 |
|---|---|---|---|---|
| **Llama-3.1-8B, whole HF directory** | 32.13 GB | 11.38 GB | **64.6%** | 22.7% |
| **Llama-3.1-8B, 4 BF16 shards** | 16.06 GB | 10.49 GB | **34.7%** | 22.7% |
| Ministral-8B, whole directory | 32.11 GB | 11.55 GB | **64.0%** | 22.7% |
| Pythia-160m, 3 training checkpoints | 1.81 GiB | 644 MiB | **65.3%** | 22.7% |
| bge-m3 directory (FP32 container) | 4.59 GB | 2.45 GB | **46.5%** | — |
| 8-bit AdamW optimizer state ×2 | 161 MiB | 119 MiB | **26.1%** | — |

On BF16 weights lmz beats the published state of the art, and sits 0.3 points
off the bound no lossless coder of any kind can pass:

| on real Llama BF16 | saved |
|---|---|
| **lmz** | **34.7%** |
| [ZipNN](https://github.com/zipnn/zipnn) (published, same model) | 33.6% |
| [DFloat11](https://arxiv.org/pdf/2504.11651) (published) | ~30% |
| bzip2 -9 | 30.7% |
| xz -6 | 29.9% |
| zstd -19 | 23.6% |
| theoretical joint-entropy bound | 35.0% |

**Three things a general compressor structurally cannot do**, which is where
most of the margin comes from: store a tensor once when a directory ships it
twice, code a checkpoint as the difference from the one before it, and split
on a float's own bit-fields instead of byte boundaries.

## What it costs to do that

A 1.12 GiB BF16 shard, RAM-backed, including all I/O and per-chunk checksums:

| threads | compress | decompress |
|---|---|---|
| 1 | 0.58 GiB/s | 0.40 GiB/s |
| 4 | **2.00 GiB/s** | 1.44 GiB/s |
| 8 | 1.87 GiB/s | **1.88 GiB/s** |

`zstd -1` still compresses about 1.5× faster than lmz does, and saves twelve
points less. Past four threads lmz is not CPU-bound any more: it runs into the
memory bus at about 2 GiB/s, and it reaches that at four threads where it used
to need eight. On real storage the disk arrives before either of them, and the
archive is a third smaller, so a storage-bound load moves a third fewer bytes.

## On a GPU

`pip install lmzip` ships a CUDA decoder. On an RTX 5080 it decodes lmz's own
rANS at **111 GB/s** out of an archive written today, and **418 GB/s** when
the frequency table is shared across chunks — both verified byte-identical to
the CPU decoder over 936 MB of real BF16 planes.

**The ratio to the link is the point, and it is a ratio you can recompute.**
On this machine PCIe Gen4 x16 delivers 28.8 GB/s, so a decoder running at 111
is 3.9× faster than the link it feeds — and once decoding is faster than the
link, compression on the path into VRAM is free, with every point of lmz's
ratio becoming a point of load bandwidth. Both halves are machine-dependent: a
Gen5 host doubles the link, a unified-memory Mac has no such link at all, and
a phone changes both numbers. What travels is the comparison, not the 3.9×.

The 418 GB/s is a bandwidth ceiling rather than a compute one — 59% of this
card's ~960 GB/s peak, because decoding moves 1.347 bytes of DRAM traffic per
decoded byte: the plaintext out plus the coded bytes in. The fused whole-BF16
kernel in `scratchpad/gpu/` measures the end-to-end consequence: cold disk to
VRAM, plain safetensors 0.373 s against lmz's 0.256 — **1.46× faster**,
converting 98% of the ratio into load speed.

```python
from lmz import gpu

gpu.available()                                    # (True, '') -- or why not
gpu.decode_batch(streams, offsets, nstr, plane)    # a batch in, plaintext out
```

A batch, not a stream: lmz's 8 interleaved rANS states are 8 lanes of work, so
one stream never fills a GPU however large it is, and many streams at once do.

## Building on lmz

New in 1.3.0, and the reason to upgrade if you are writing the layer above:
lmz stopped requiring anyone to reach inside it. A runtime that wants to
schedule reads, drive its own device, or predict lmz's cost on hardware nobody
here owns can now ask, instead of copying a constant out of the source and
hoping it does not move.

**The archive says who can read it.** Not plumbing — a footgun with a fix.
Once a chunk is big enough and conditioning wins, lmz emits `CODEC_BF16C`,
whose per-bucket segments have unequal lengths — and **the GPU decoder cannot
read it**, because a batch decode needs equal-length streams. Nothing used to
say so.

```python
lmz.capabilities("model.lmz")
# {'batch_decodable': False, 'blockers': {'bf16-cond': 224, 'entropy': 1},
#  'blocker_bytes': 1872871856, 'batch_decodable_bytes': 0.0, ...}
```

`lmz info` prints the same thing as a `batch` line. If it says no and you
wanted a GPU-readable archive, compress with `--mapped`: 64 KiB blocks stay
under the conditioning threshold, so that codec is never chosen. Measured on a
1.87 GB BF16 checkpoint on this box, that moved 224 blocking chunks and
1.87 GB down to one 12 KB chunk, and cost **0.89 points** of ratio (32.91% →
32.02%).

Read the bytes and not only the boolean. That last 12 KB chunk is the
safetensors JSON header — one general-purpose stream, enough on its own to
make `batch_decodable` False while 99.999% of the output is still
batch-decodable. `blocker_bytes` is there so you can tell one header from two
hundred weight chunks and route the bulk on the device either way.

**Decode bytes you already have.** `ArchiveIndex` exists because of a coupling
worth naming, since it is not specific to lmz: `decompress` reads and decodes
inside one worker, so a single thread count sets both decode throughput *and*
read queue depth — and those want opposite things. An NVMe device reaches its
rated rate only with about a dozen requests in flight; past two threads lmz's
own interpreter overhead between native calls becomes GIL contention. You
cannot tune your way out of that from outside. You can only step around it, by
fetching the payload yourself and handing lmz the bytes.

```python
idx = lmz.ArchiveIndex("model.lmz")
for c in idx.chunks():                    # where every payload is
    plain = idx.decode(c, my_reader.pread(c.off, c.clen))   # no I/O, no threads
```

Ref and delta chunks raise `NeedsSource` instead: they are *defined* as a
difference from an earlier range, so decoding one needs bytes from elsewhere in
the archive — and lmz will not reach for the file behind your back. Use
`decompress` when you want it to handle that.

**The caller owns the device.** `gpu.decode_batch_dev()` takes device pointers
as plain integers plus your stream, allocates nothing, creates no context, and
does not synchronise. Your allocator, your stream, your context; lmz does not
build a second one behind them and does not decide when you need the result.

**The kernel publishes its constants, not just its speed.** A rate measured on
a 5080 tells you about a 5080. `gpu.cost_model()` gives you the mechanism
instead, so you can compute the answer for your own device:

```python
gpu.cost_model()
# k_cycles_per_byte: (230, 330)   -- an interval, because it is one
# shmem_lut_bytes / shmem_per_group_bytes / blocks_per_unit_at_measurement
# expansion, bound, and provenance for every number
```

`k` is published as a range rather than a midpoint because occupancy hides part
of the cost and how much varies with the block size — and it is **latency-bound
on a dependent shared-memory load chain, not throughput-bound**, so scaling it
by a device's FP32 rate is the wrong arithmetic. The shared-memory numbers let
you check how many blocks your device holds and decide whether that interval
brackets you or is only a floor.

**Every encode keyword is declared.** `encode_options()` labels all ten as
`format` (eight — they decide what the coded bytes are), `schedule` (one:
`workers`) or `observe` (one: `progress`). The distinction is the useful part:
**a scheduling default is a fallback, never a decision.** lmz picks `workers`
from a CPU count without knowing your machine or your workload — compression
goes memory-bus-bound near four threads on real weights — so it is yours to
override, and the declaration is how you can tell that from a considered
answer.

**One format choice worth knowing about.** A rANS stream normally carries its
own 516-byte frequency table. `lmz compress --shared-tables` lifts it out and
carries one per plane kind in the manifest instead, which is smaller on
archives whose chunks are small enough for those headers to matter and is the
layout a GPU wants, since a warp can then share one table in shared memory. It
is **off by default** because it writes a v7 archive an older lmz cannot read,
and the current GPU kernel does not read it yet — so today it is a size and
format decision, not a speed one. Sharing is decided per plane and only where
it measured better: **+0.93 points** on a 151 MB fp32 whisper checkpoint in
64 KiB blocks, and on a BF16 Llama shard it declined outright and wrote an
ordinary v1 archive, because there it does not pay.

The first thing it does on any machine is decode a stream that machine just
encoded and check the CPU decoder agrees; a device that disagrees is not used,
and `lmz doctor` names it. The kernel is clean under `compute-sanitizer` and
compiles for sm_75 through sm_121. It has been *run* on two architectures —
an RTX 5080 and a Tesla T4 — which are the two ends of the range and the two
that generate different code, so it verifies rather than assumes.

**If you have a GPU, this is worth thirty seconds:**

```
lmz doctor --gpu-verify
```

It decodes thirty awkward distributions and batch shapes and checks lmz's own
CPU decoder agrees with every byte — no data file, no network, no login: the
streams are built by lmz's own encoder, so the oracle travels with the
question. Paste the block into
[an issue](https://github.com/FanxinSun/lmz/issues). A pass is evidence too,
and an **Ampere, Ada or Hopper** is the gap now: those are the cards nobody
has run.

**No GPU? A free Colab T4 takes one click** —
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/FanxinSun/lmz/blob/main/docs/verify-on-colab.ipynb)

Turing runs *different generated code*: it has no `cp.async` instruction, so
the intrinsic falls back to a synchronous copy. Counting them says which
architectures share which path:

| | `LDGSTS` | run on real silicon |
|---|---|---|
| sm_75, Turing | **0** | **yes — Tesla T4, 30/30 byte-identical** |
| sm_80 / 86 / 89 | 38 | not yet |
| sm_90 / 120 | 41 | **yes — RTX 5080, 936 MB byte-identical** |

The synchronous fallback is not a guess: a Tesla T4 decodes all thirty shapes
byte-identically, and separately, `compute_75` built as PTX with no cubin —
so the driver must JIT it — decodes 936 MB byte-identically at every block
size on both kernels and is clean under `memcheck`, `racecheck` and
`synccheck`. Ampere through Hopper sit *between* two architectures that have
both been verified on hardware, and their generated code has been JIT-run and
sanitized too, so what is open there is a throughput number rather than a
question of whether it works. Shared memory excludes nobody either: a T4's
64 KiB holds the per-chunk tables at 64 threads a block.

**CUDA is optional in every direction.** The wheel is pure Python, carries a
`.cu` and no CUDA, and installing needs no toolkit. `nvcc`, if it is there, is
used once to build the decoder into the package directory — the same bargain
the SIMD kernel already makes with a C compiler, and nothing is installed
system-wide. No nvcc or no card means the CPU path, unchanged. `lmz doctor`
says which you have.

That holds even when the driver itself is broken. A CUDA driver that is
half-removed or mid-upgrade leaves `libcuda.so.1` on disk with an initialiser
that faults, and loading it takes the whole process down with no return code
involved — so lmz does its first load in a child process that is allowed to
die, and reports it. A segmentation fault in your program is not a fallback.

Nothing in `lmz decompress` routes to it yet, deliberately: the useful thing
to do with a GPU decode is to leave the result in VRAM, and deciding when
belongs to the layer above. That has stopped being a deferral. The layer above
now exists, and what lmz owes it is published on both sides of the line —
where the bytes are and what can read them, a decode that touches no file, a
device entry point that takes your pointers and your stream, and the kernel's
own cost constants so it can predict lmz on hardware nobody here has. lmz
turns bytes into bytes as fast as the hardware allows and says what that
costs; it does not decide when. The
[GPU residency handover](docs/gpu-residency-handover.md) is where that boundary
is written down.

## Where it is not worth it

Stated plainly, because a compressor that only advertises its wins should not
be believed:

| | lmz | best alternative | verdict |
|---|---|---|---|
| Quantised GGUF (Q8_0 / Q4_K_M) | 6.7% / 5.1% | 5.5% / 2.5% | the quantiser already took it |
| FP8 safetensors | 17.14% | zstd -3, 17.11% | **just use zstd**, the gap is 0.03 points |
| Text, code, JSON, binaries | = zstd | zstd | lmz *is* zstd here, by design |
| Read speed | slower | a plain file | see below |

Reading a compressed file transparently can never beat a plain one by more
than `1/(1−saved)` — you still have to read the archive. That is 1.5× on BF16
and 1.05× on Q4_K, so on a fast SSD the mount is *slower*. It buys disk, not
speed.

## Also included

```
lmz add ./my-model/ && lmz mount ~/models   # read a compressed model as ordinary
                                            #   files; llama.cpp needs no patch
lmz fs ~/.lmz/data ~/data                   # a read-write compressed filesystem;
                                            #   32.1% where btrfs+zstd gets 18.9%
```

## Buy me a coffee

lmz is free, MIT-licensed and unfunded. If it saved you disk or bandwidth —

### [☕ **Buy me a coffee**](https://buymeacoffee.com/fanxinsun)

or [Alipay](assets/alipay.jpg) (打开支付宝，扫一扫). Thank you.

## Documentation

- [**How it works**](docs/how-it-works.md) — why a float array defeats a
  general-purpose compressor, and the bit-level choices that close the gap
- [**Measured results**](docs/results.md) — every number, with its conditions
- [**Using lmz**](docs/usage.md) — command line, Python API, the mount and the
  filesystem
- [**Limitations**](docs/limitations.md) — where it does not pay, and what the
  120 tests check
- [**Vectorising the coder**](docs/vectorising-the-coder.md) — how the encoder
  reached arm64, the one piece of work still open, and the six that were tried
  and measured out flat
- [**GPU residency handover**](docs/gpu-residency-handover.md) — what the
  decoder costs and why, with the occupancy sweep that separated a compute
  ceiling from a bandwidth one; what shipped as `lmz.gpu`, the interfaces the
  layer above asked for, the two measurements still open, and where lmz's job
  ends
- [**Portable decoder handover**](docs/portable-decoder-handover.md) — the
  shared frequency table as a format option, and the three things its design
  did not survive contact with
- [**Perception codec handover**](docs/perception-codec-handover.md) — what
  vision and audio models need that LLM checkpoints did not: int8 routed to
  the coder the GPU can read, ONNX parsed, and the two expected ratio items
  that dissolved under measurement and should stay dissolved

Python 3.10+, no runtime dependencies. zstd comes from the standard library on
3.14+; a C compiler, if present, is used once to build the SIMD kernel into the
package directory, and `nvcc`, if present, does the same once for the CUDA
decoder — nothing is installed system-wide and neither is required. Runs
straight from a checkout with `./lmz-cli` if you would rather not install it
at all.

Check what is active with `lmz doctor`.
