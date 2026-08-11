# lmz rANS decode on Apple silicon

Port of the CUDA decoder that was verified byte-identical on an RTX 5080. It is
**not yet run on any Apple GPU** — there is no Metal toolchain on the Linux box
it was written on, so treat every number as unknown and the correctness as
unproven until the harness prints `byte-identical`.

## What it is

Same trick as the CUDA version, which is what made it fast. lmz's 8 interleaved
rANS states share one input cursor and refill in strict order, but whether a
state needs a refill depends only on that state, never on the cursor. So the 8
states map onto 8 **lanes**, and one `simd_ballot` + `popcount` prefix sum
replaces the serial cursor walk.

Three kernels:

| kernel | what it does |
|---|---|
| `lmz_decode_plane_direct` | refills read straight from device memory. No divergent barrier anywhere, so it is correct by construction — **this is the oracle** |
| `lmz_decode_plane_prefetch` | stages the stream in threadgroup memory first. Worth 1.44× on the RTX |
| `lmz_decode_bf16` | fused: decode the exponent plane, read the stored-raw sign+mantissa plane, write bit-interleaved BF16 |

## Running it

```sh
swiftc -O bench.swift -o lmzmetal
./lmzmetal <dir>                          # plane decode + verify
./lmzmetal <dir> smplane.bin bf16.bin      # also the fused BF16 decode
```

`<dir>` needs `streams.bin` and `ref.bin` as written by
`scratchpad/prep_shared.py`. Either copy them across (325 MB + 936 MB for the
Qwen2.5 set) or regenerate on the Mac:

```sh
cd ..                                       # the prep scripts live one level up
gcc -O3 -fPIC -shared -o libshared_enc.so shared_enc.c
python3 prep_shared.py                      # KIND=exp (default) or KIND=sm
python3 prep_fused.py                       # writes bf16.bin
```

That path also builds lmz's own C kernel for arm64, which is worth doing on its
own — `lmzcore.c` already has a NEON path (`split2_neon` / `merge2_neon`,
`lmz_isa()` returns `"neon"`) that has never been benchmarked on real hardware.

## Three things to check first

1. **`simdgroup_barrier` under divergence.** The prefetch kernel refills
   per-group, so different 8-lane groups inside one SIMD-group can take
   different branches. That is safe only because Apple GPUs run a SIMD-group in
   lockstep and `simdgroup_barrier` is a memory fence rather than an execution
   barrier. If `lmz_decode_plane_prefetch` mismatches while
   `lmz_decode_plane_direct` verifies, this is why — and the direct kernel is
   still a usable answer.
2. **Threadgroup memory is 32 KiB on M1.** The threadgroup is therefore fixed at
   128 threads: LUT 16384 + starts 1024 + stage 2048 + inbuf 4096 = 23.5 KiB.
   256 threads would need ~44 KiB and the pipeline will refuse to build. The
   harness reports `maxThreadgroupMemoryLength`, so check it.
3. **Expect bandwidth, not magic.** On the RTX the decoder ended up at ~60% of
   peak memory bandwidth. An M1 Pro at ~200 GB/s would land near 120 GB/s of
   BF16, an M1 Max at ~400 GB/s near 240. If it comes out far below that, the
   first suspect is the LUT: 4096 random `threadgroup` reads per lane hit bank
   conflicts, and that was the single biggest effect on the RTX (3.6×).

## Why Apple silicon changes the story

Every load-path number measured on the RTX is dominated by disk → PCIe → VRAM,
and the H2D ceiling (28 GB/s) is what made a GPU decoder worth having at all.
There is no such hop here: `.storageModeShared` means the GPU reads the same
pages the CPU wrote. The interesting question is not whether decode is fast
enough to beat PCIe — there is no PCIe — but whether decoding into unified
memory beats reading the uncompressed file at all, which is the same crossover
the RTX work put at 0.71 GB/s of storage bandwidth.
