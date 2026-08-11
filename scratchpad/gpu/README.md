# GPU decoding for lmz

Experiments, not shipped code. Nothing here is imported by the `lmz` package and
none of it changes the archive format. It exists to answer one question: lmz
compresses better than anything published, but every competing system since 2024
decodes *inside a GPU kernel* while lmz decodes on the CPU — so what would lmz
get on a GPU, and what would it cost?

Measured on an RTX 5080 (sm_120, 84 SMs, PCIe Gen4 x16), CUDA 13.2, against the
BF16 planes of a real Qwen2.5 checkpoint. Every kernel is verified byte-identical
to lmz's own `lmz_rans_decode` over the full 936 MB / 1.87 GB workload.

## The result

| | GB/s |
|---|---|
| CPU, one thread (AVX2) | 0.48 |
| lmz whole archive, all cores, free-threaded | 2.21 |
| GPU, one thread per stream, per-stream table in global memory | 9.4 |
| GPU, one shared table | 33.6 |
| GPU, **8 lanes per stream, ballot prefix-sum cursor** | 245 |
| + output staged in shared memory, flushed as `uint4` | 280 |
| + stream prefetched into shared memory | 402 |
| + `cp.async` 4-slot pipeline | **418** |
| **fused whole-BF16** (exponent rANS + raw sign/mantissa + merge) | **399** |

End to end, cold disk (`O_DIRECT`) to VRAM with BF16 resident: plain safetensors
0.373 s, lmz 0.256 s — **1.46x faster**, converting 98% of a 1.485x compression
ratio into load speed. The same comparison with the CPU decoder is 2.85x
*slower*, so the decoder is a 4.2x end-to-end swing.

## What made it fast

lmz's 8 interleaved rANS states share one input cursor and refill in strict
order, which looks inherently serial. It is not: whether a state needs a refill
depends only on that state, never on the cursor. So the 8 states map onto 8
**lanes**, and one `__ballot_sync` plus a `popcount` prefix sum reconstructs
every lane's byte offset. That single change is 7.3x, and it needs no format
change at all.

The other 1.7x needs one format change: **a frequency table shared across
chunks instead of one per chunk**. A per-stream 16 KiB table means 32 lanes in a
warp thrash 512 KiB of L1; a shared table lives in shared memory. Measured on
real data, sharing also makes the file *smaller* — 325.0 MB against 336.1 MB
over 936 MB of exponent plane, because the 516-byte-per-chunk header costs more
than the cross-entropy the shared table gives up.

Two diagnoses worth remembering, both of which contradicted a confident guess:
coalescing the refill loads gained exactly nothing (the cost was latency on the
dependency chain, not transaction count), and a two-slot `cp.async` buffer races
even though it verifies byte-identical on the exponent plane — it needs four
slots, and the failure only appears on the sign+mantissa plane, whose 3x higher
refill rate reaches the boundary sooner.

## Layout

```
prep_shared.py     build streams coded against ONE shared table, and prove they
                   round-trip through lmz's unmodified decoder
prep_fused.py      the BF16 reference, taken from the model file rather than
                   re-derived from the planes, so the test is not circular
shared_enc.c       the shared-table encoder: lmz's coder with the per-stream
                   header lifted out. Prepending the shared header turns a
                   shared-table stream back into an ordinary lmz stream
cuda/gpu_rans10.cu the full V0..V5 ladder plus the ablations that attributed
                   the time. Some kernels deliberately produce wrong output and
                   are labelled as timing-only
cuda/gpu_fused.cu  the final decoder: whole-BF16 in one kernel, plus the
                   end-to-end O_DIRECT load measurement
metal/             the Apple silicon port -- WRITTEN BUT NEVER RUN, see its
                   own README
```

Build: `/usr/local/cuda-13.2/bin/nvcc -O3 -arch=sm_120 -o gpu_fused gpu_fused.cu`.
A `nvcc` older than 12.8 cannot target Blackwell.

The data files these read are large (325 MB + 936 MB + 1.87 GB) and live under
`~/.cache/`, not in the repo. `prep_shared.py` and `prep_fused.py` regenerate
them from a safetensors checkpoint.
