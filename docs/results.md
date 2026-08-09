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
