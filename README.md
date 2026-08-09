# lmz

[![tests](https://github.com/FanxinSun/lmz/actions/workflows/tests.yml/badge.svg)](https://github.com/FanxinSun/lmz/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/lmzip)](https://pypi.org/project/lmzip/)
[![Python](https://img.shields.io/pypi/pyversions/lmzip)](https://pypi.org/project/lmzip/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Smaller checkpoints. Byte for byte.**

Lossless compression built for model weights. `zstd -1` takes 22.7% off a
Llama-3.1-8B BF16 checkpoint. lmz takes **34.7%** — and **64.6%** off the
directory as Hugging Face actually ships it, which is **13 GB more than zstd
on one 8B model**.

Nothing is approximated. Every byte comes back.

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
  81 tests check

Python 3.10+, no runtime dependencies. zstd comes from the standard library on
3.14+; a C compiler, if present, is used once to build the SIMD kernel into the
package directory — nothing is installed system-wide. Runs straight from a
checkout with `./lmz-cli` if you would rather not install it at all.

Check what is active with `lmz doctor`.
