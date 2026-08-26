# A decoder off CUDA — what lmz has to ship for the machines most people have

*Why lmz currently makes loading **slower** on every machine without an
NVIDIA GPU, what the fix is, the order it has to happen in and why, and what
the first item cost when it landed. One of three items is done; this says
what it left. Written for whoever picks this up next, so that none of it has
to be rediscovered.*

[← back to the README](../README.md)

## Why this exists

`gpu-residency-handover.md` records a good result: the CUDA decoder runs at
111 GB/s from an archive written today and 418 with a shared table, against a
28.8 GB/s PCIe link, so compression on the path into VRAM is free by 14× and
every point of ratio becomes a point of load bandwidth. Cold disk to VRAM,
plain safetensors 0.373 s against lmz's 0.256 — **1.46× faster**.

The same paragraph records the other half, which has not been treated as a
work item: *"the same comparison with the CPU decoder is 2.85× slower."*

That is the whole of this document. On a machine with no CUDA, lmz today
costs about **2.85× on load time** in exchange for 34.7% less disk. Not a
missing optimisation — a bad trade, taken by default, on the majority of the
machines lmz is installed on. The arithmetic behind it is not subtle:

| | GB/s |
|---|---|
| lmz CPU decoder, 4–8 threads | **2.0** |
| NVMe, 64 KiB reads at queue depth 64 | **6.34** |
| lmz CUDA decoder, per-chunk table | 111 |
| lmz CUDA decoder, shared table | 418 |

The CPU decoder is three times slower than the disk it is feeding from, and
`README.md` already says why it will not improve: *"past four threads lmz is
not CPU-bound any more: it runs into the memory bus at about 2 GiB/s."* More
cores will not fix this. A different processor might.

## The device that is already in the box

Integrated graphics is the obvious candidate and the obvious objection is
that an iGPU is not a real GPU. Measured on this machine — Ryzen 7 9800X3D,
2 RDNA2 CUs, DDR5-6000, through a D3D11 compute benchmark written for the
purpose:

| | GFLOP/s FP32 | GB/s | FLOP per byte |
|---|---|---|---|
| RTX 5080 | 53,691 | 822 | 65 |
| Radeon iGPU, 2 CU | 567 | 59.4 | 9.5 |
| CPU, 16 threads | — | 54.3 | — |

Two things in that table are worth carrying forward. The iGPU pulls **59.4
GB/s flat from 64 MB to 2 GB of working set** — 62% of the DDR5 bus, with no
cliff at the 486 MB UMA carve-out — which is *more* than the 16-thread CPU
manages out of cache. And it is the smallest iGPU AMD ships: a display
adapter, not an APU. A Radeon 780M has roughly seven times the compute at
similar bandwidth, which puts it in the same FLOP-per-byte regime as the
5080.

**Two measurements this replaced, both of which were wrong.** Before EXPO was
enabled on this box the same iGPU measured 22 GB/s past 384 MB and looked
like it had a hard carve-out cliff. It does not; it was running at
DDR5-4800 JEDEC defaults, and at the rated 6000 the cliff disappears
entirely and the number nearly triples. Any bandwidth figure taken from this
machine before 2026-08-26 was taken at the wrong memory clock.

**What this does *not* argue for.** Running the model on the iGPU. `vram/`'s
roofline already settles that: inference decode does one multiply-add per
byte of BF16 and needs about 4,132, *"short by a factor of four thousand, and
no scheduling recovers it."* The iGPU should decode the archive, not run the
network. That is a load-path job, and the load path is where lmz is currently
losing.

## 1. The shared frequency table, as a format option — **done**

`gpu-residency-handover.md` item 1 and `perception-codec-handover.md` item 4
both ask for this and both say it should land first. It has. What follows is
what it cost, because roughly none of it was the part either document
anticipated.

**The shape is per plane kind, per plane, and the manifest is the
authority.** `lmz/freqs.py` carries a `TableSet` keyed `"{codec}:{esize}:{k}"`
and serialised into the manifest; `CODEC_SPLIT_ST` (9) marks a byte-plane
split whose rANS streams are headerless. A separate codec id rather than a
flag bit because a split chunk's 16 flag bits are already two per plane for
eight planes with nothing spare — and because a reader that does not know
about shared tables must refuse the chunk rather than misread it, which an
unknown codec id gets for free.

**Sharing is decided per plane, not per chunk, and that was not the first
design.** The first version required a table for *every* plane of a chunk or
fell back wholesale. On the fixture it was written against, it produced
archives byte-identical in size to the per-chunk form, and the reason is the
interesting part: an fp32 checkpoint's low byte planes pool beautifully
while its **high byte carries a per-tensor scale and does not**. One plane
declining took the other three down with it. Nothing in the record has to
say which planes are shared — the decoder consults the identical table set,
so a kind with no table simply carries its own header as it always did.

**The decision is a measurement, and its first version measured the wrong
thing.** `Primer` compares, per plane kind, the cross-entropy of the pooled
counts against what the archive pays today. Cross-entropy against a fixed
table is linear in the counts, so the pooled side needs the pooled histogram
alone and no per-plane histograms have to be kept.

The bug: the first version counted **every** plane, including the ones the
encoder stores raw as noise. An fp32 detector's low mantissa planes are
genuine noise at 8 bits a byte and are never coded, so they never pay for a
frequency table — but the comparison credited a shared table with saving the
516 bytes each of them "would have" spent on one. On `ssdlite320` it
predicted **+1.9 points and delivered nothing**, which is how it was found.
The fix is one line: a plane at or above `NOISE_BITS` is stored under either
scheme and belongs in neither column.

Two guards followed from the same class of error. A kind with **one stream**
is not sharing anything, and a table has to **pay for its own passage** —
it is 516 bytes in the manifest that the per-stream form never spends twice.
Without both, `ssdlite320` shipped eight tables no chunk ever used.

**Measured, on the three models `perception-codec-handover.md` used.** Every
archive round-tripped byte-identical; decode is the median of nine runs.

| | own tables | shared | | decode |
|---|---|---|---|---|
| mobilenet_v3_small, `--mapped` | 15.66% | **16.61%** | +0.96 | 1.4–1.6× |
| mobilenet_v3_small, `--mapped --align` | 12.43% | 12.39% | −0.04 | 1.6× |
| ssdlite320, either | 14.57% | 14.57% | 0.00 | 1.0× |
| whisper_tiny, `--mapped` | 55.00% | **55.93%** | +0.93 | 1.13× |
| whisper_tiny, `--mapped --align` | 50.01% | **53.11%** | **+3.10** | 1.12× |

**These are smaller than the handover predicted, and the difference is the
measurement basis rather than a regression.** That document's table (+2.52
mobilenet, +1.52 ssdlite, +2.24 whisper) is entropy with tables uncharged,
per its own `CONDITIONS.md`; the column above is what the shipping encoder
actually writes to disk, thresholds and stored planes included. Where the
two can be compared directly they agree in sign and in which planes pool.

**The largest win is in the aligned form and the model does not predict
it.** `Primer` expects +0.27 points on whisper and delivers +3.10 under
`--align`. Coding a plane that was previously stored does not only save its
bits — it changes how many payloads there are and therefore how much 4 KiB
alignment padding the archive carries. The estimator is blind to that, so it
under-predicts, which is the safe direction: it keeps a table only when it is
confident, and reality does better than it expects.

**Decoding it cost 3× before it cost 1.2×, and both causes are worth
knowing.** A shared-table chunk cannot use `lmz_decode_planes`, which reads a
per-stream header that is not there, so the first version fell back to the
Python per-plane loop and measured **2.5–3× slower decode**. Two fixes:

- `_freqs_from_table` unpacked a 516-byte header into a ctypes array with a
  **256-step Python loop, once per plane per chunk**. Copied wholesale and
  cached on the table bytes instead. Roughly half the loss, and it was pure
  interpreter — an archive has a handful of tables and decodes them across
  thousands of planes.
- `lmz_decode_planes_shared` in `lmzcore.c` — the existing body, refactored
  to take an optional flat table array plus a have-bitmask, with the old
  entry point calling it with NULL. A plane whose bit is clear decodes
  exactly as it always did.

What is left is 1.0–1.6×, and the residue is real work rather than overhead:
the shared archive codes planes the per-chunk archive stored, so there is
more rANS to do.

**The version stamp is a requirement on the reader, not a build number.**
Bumping `FORMAT_VERSION` to 7 unconditionally made every ordinary archive
this build writes unreadable to lmz 1.2.0, which can decode all of it. The
writer now stamps `BASE_VERSION` (6) and only a chunk that genuinely needs a
newer reader raises it. `ssdlite320` keeps no tables and is still a v6
archive.

**What is left of this item.** It is `--shared-tables`, off by default, and
should stay off until a GPU decoder can read it — the CPU-side ratio is
worth under a point outside the aligned case, and the decode cost is real.
`CODEC_BF16` and the conditioned and block codecs are untouched: the measured
wins were all fp32, `CODEC_BF16C` has a different sub-stream structure, and
scope was held to the case with evidence behind it. Delta chunks deliberately
keep per-stream tables. And **the CUDA kernel has not been taught to read a
v7 archive** — `lmz_gpu_decode_batch_dev` already takes a shared header in
`hdr`, so this is wiring rather than invention, but until it happens the 3.8×
this format change exists for is not collected.

## 2. Run the Metal port

`scratchpad/gpu/metal/` is an Apple silicon port, **written but never run**,
and `gpu-residency-handover.md` says what that is worth. It is also the
cheapest possible test of whether the kernel design survives leaving CUDA,
and it wants hardware that is already to hand.

Two questions it answers, and neither can be answered from a table:

**Does the lane trick port?** The 7.3× that made the whole GPU decoder
possible is eight rANS states mapped onto eight lanes with `__ballot_sync`
plus a popcount prefix sum. Metal's `simd_ballot` and `simd_prefix_exclusive_sum`
look like the same primitives. "Look like" is not a measurement.

**Does the table fit?** This is the one that should be checked first, because
it may change the kernel rather than the port. `scratchpad/gpu/cuda/shmfit.cu`
records what the shipped picker asks for: **90 KiB** per block for the
per-chunk path and **46 KiB** for the shared one. Apple silicon's threadgroup
memory limit is far below both — 32 KiB on current parts, though that number
should come off the device rather than out of a table, since it is the number
that decides the port. If it is 32 KiB then **Metal cannot run the per-chunk
kernel at all**, and the shared-table path needs fewer groups per block than
CUDA gives it.

That is the concrete reason item 1 came first. A portable kernel written
before the shared table exists gets written against the path that cannot fit.

## 3. Then Vulkan

Broadest reach and the most work: AMD and Intel iGPUs, Windows and Linux, and
no toolkit at all, since Vulkan compute is driver-level. That last point fits
lmz's existing constraint better than CUDA does — `lmz/gpu/build.py` exists
because nvcc has to be found, versioned and survived, and a Vulkan backend
needs none of that.

It should not start until 1 and 2 are done. `subgroupBallot` and
`subgroupExclusiveAdd` are the load-bearing assumption and Metal tests the
same assumption for a fraction of the effort; `cp.async` has no Vulkan
equivalent, so the pipeline degrades to the synchronous copy that sm_75
already takes — measured at **21% off** the Blackwell figure, which is a
known cost rather than an open question.

## Where the boundary is

Unchanged from `gpu-residency-handover.md`, which should stay the single
statement of it. What this document adds is that **the decoder line is
per-backend, not per-vendor**: "a standalone GPU decoder with a stable ABI"
is lmz's whether the device is NVIDIA, Apple or AMD, and the engine's side of
the line does not move because the device changed.

One row is worth restating because this document is about the load path and
that is where it gets tested: lmz produces bytes and turns them back into
bytes as fast as anyone. It does not decide *when*.

## Traps, carried forward

**Do not benchmark one stream twice**, and **do not ship a kernel change on
one machine's evidence.** Both are from `vectorising-the-coder.md` and both
apply here unchanged. The second one earned a fresh example on this machine:
every bandwidth number taken here before EXPO was enabled was taken at
DDR5-4800 against a 6000-rated kit, and it moved the iGPU figure by 2.7×.

**A predicted ratio is not a measured one.** `Primer` under-predicts by an
order of magnitude on the aligned case and predicted a win on `ssdlite320`
that did not exist. Every number in the table above is a file on disk with a
verified round trip behind it.

**A plane that is stored is not a plane that is coded.** The one-line noise
gate in `Primer.add` is the whole of the ssdlite bug, and the same mistake is
available anywhere the archive's costs are modelled rather than observed.

## How to measure

    python3 tests/test_lmz.py            # 114 tests, five of them this item's

The five: a page-mapped archive that must come out smaller, a mixed file
where some planes pool and some do not, a v7 chunk that must refuse to decode
without its manifest, the version stamp following the codecs actually used,
and a primer that must be able to say no.

Ratios and decode times above come from a scratch script over
`~/.cache/yfce-e5-ckpt/`, the checkpoint set `experiments/e5_perception_codec/`
builds; `fetch2.py` there rebuilds it. Compare `--mapped` and
`--mapped --align` both ways and take the median of several decodes — a
single run on this box varies by 40%.

The iGPU and CPU bandwidth figures come from a self-contained D3D11 compute
benchmark (`igpu-bench.ps1`, PowerShell plus inline C#, no toolkit) that
enumerates every DXGI adapter and measures streaming read bandwidth, FP32 FMA
throughput and a CPU baseline. It is not in this repository; the numbers it
produced are in the table above, and the FLOP-per-byte column is the gate for
whether decoding on a given device can pay at all.
