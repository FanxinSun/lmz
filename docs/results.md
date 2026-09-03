# Measured results

*Every number lmz claims, on real checkpoints, with the conditions each was measured under.*

[← back to the README](../README.md)

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
| Qwen3.6-27B-FP8, one shard (99.7% F8_E4M3) | 366.1 MiB | 303.3 MiB | **17.10%** |
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

**Against ZipLLM's method**, on one corpus, measured the same way. ZipLLM
publishes 54.1% across 3,048 models and 43.19 TB of hub, 99.64% of it
fine-tunes. That is not a number lmz can be compared to directly — different
corpus, different composition — so their pipeline was reproduced here instead:
tensor dedup, a BitX XOR delta against the base, then a general-purpose coder.
Only the last step differs between the rows.

The corpus is Qwen2.5-0.5B and four full fine-tunes of it — `-Instruct`,
`Coder-0.5B`, `numind/NuExtract-1.5-tiny`, `alamios/DeepSeek-R1-DRAFT` — 4.941
GB, all BF16, weights only.

| | archive | saved | time |
|---|---|---|---|
| ZipLLM's method, zstd -1 | 2.9239 GB | 40.82% | 27 s |
| ZipLLM's method, zstd -3 | 2.9353 GB | 40.59% | — |
| ZipLLM's method, zstd -19 | 2.8479 GB | 42.36% | 1143 s |
| lmz's coder in that same pipeline | 2.4168 GB | 51.08% | 28 s |
| **lmz as shipped** | **2.3940 GB** | **51.55%** | **5.6 s** |

The first four rows are one single-threaded harness that does nothing but code,
and they are comparable to each other. The last is the tool: threaded, writing a
real archive, checksumming every chunk — 5.6 s to compress and 8.8 s to
decompress, and not comparable to the rows above it. Each coder was timed on its
own; a run that codes two of them and is then divided by a run that codes two
others gives a ratio of nothing in particular, which is how this table came to
say "37 times" before it was measured properly.

zstd -19 is their strongest setting: it buys 1.5 points over -1 and costs about
forty times as long. The gap is the coder: on the XOR residuals themselves lmz beats
zstd by 23–54% across update sizes from 1e-4 to 1e-1, against -1 and -19 both.
The exponent field of `a XOR b` is not an exponent, which looked like it should
cost lmz its field split — but those bits are then almost always zero, 0.00 to
0.70 bits per byte, so the split still pays.

Three things this does not say. **It is not a claim about 54.1%**: on this
corpus ZipLLM's own method reaches 40.8%, and the comparable sentence is the
one the table makes. **Dedup found 0.00% here** — 95 tensors, norms and biases,
0.1 MB — where their hub gets 8.3% from it, so this corpus is *less* favourable
to them than their own on that component; dedup is coder-independent and would
lift both rows. And **four fine-tunes of one base is more favourable to delta
than a real hub is**, which cuts the other way.

Worth recording because it was found here: until 2026-09-03 the number in that
last row was 41.83%, and which one you got depended on what your files were
called. The delta source was "the earliest member holding this tensor", which
is the lowest member index, which is directory order — so on this corpus every
fine-tune was subtracted from `-Instruct` rather than from the base they share,
because a hyphen sorts before a dot. Across orderings that was 17.6 points of
spread, 34.0% to 51.6%, and at the bottom of that range lmz lost to the method
it beats. Sources are now chosen by measuring candidates with the real coder,
the same way the decision to delta at all was already made, and the archive is
invariant under member order.

**Throughput**, 1.119 GiB BF16 shard, RAM-backed filesystem, including all I/O
and per-chunk checksums, best of 3:

| threads | compress | decompress |
|---|---|---|
| 1 | 0.58 GiB/s | 0.40 GiB/s |
| 2 | 1.28 GiB/s | 0.77 GiB/s |
| 4 | **2.00 GiB/s** | 1.44 GiB/s |
| 6 | 1.95 GiB/s | 1.65 GiB/s |
| 8 | 1.87 GiB/s | **1.88 GiB/s** |
| 12 | 1.81 GiB/s | 1.67 GiB/s |
| 16 | 1.70 GiB/s | 1.53 GiB/s |

A synthetic shard here rather than the Llama one the ratio rows use, so that
the conditions are reproducible by anyone:
`tests/make_model.py --dtype bf16 --layers 7 --hidden 2048 --seed 7` builds it,
and it codes at 32.9% against the real shard's 34.7%. Timed in-process, because
starting an interpreter costs more than a tenth of a second and the fast rows
here take less than a second in total.

Compress peaks at four threads and decompress at eight; both used to peak at
eight. The ceiling has not moved — about 2 GiB/s either way, with source,
archive and output all in flight and the job memory-bandwidth bound. What moved
is the work: a thread spends about half the cycles per byte it used to, so half
as many threads reach the same wall, and the ones past it cost a little rather
than earning anything.

For scale, ZipNN publishes 1.15 GB/s compress and 1.65 GB/s decompress on this
model, though that comparison crosses different hardware and should be read
loosely. On ordinary disks the limit arrives sooner than any row above: writing
to ext4 with an fsync runs entirely I/O bound, which is the point. The archive
is a third smaller, so a storage-bound load moves a third fewer bytes.

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
