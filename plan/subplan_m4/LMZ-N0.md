# Subplan: LMZ-N0 — First-Use Workload & Target Machine Selection

**Status**: ACTIVE & FORMALIZED  
**Date**: 2026-10-17  
**Parent Authority**: [`../PLAN.md`](../PLAN.md) § Authoritative execution order (`LMZ-N0`)

---

## 1. Selected First-Use Workload & Consumer

- **Workload**: Lossless model weight delivery and streaming residency for transformer and multimodal architectures (LLM, vision, ASR).
- **Target Consumer**: On-device Apple Silicon runtime (Python `lmz.MappedArchive` / `ArchiveIndex`, interoperating with PyTorch, MLX, and `llama.cpp` runtimes).
- **Artifact Forms**:
  1. `safetensors` BF16/FP16/FP32 weights.
  2. GGUF block-quantized weights (Q4_K, Q8_0).
  3. Complete artifact bundles (manifest, tokenizer, config, model weights).
  4. v7 shared frequency table archives (`--shared-tables`, `CODEC_SPLIT_ST`).

---

## 2. Target Machine & Hardware Envelope

- **Hardware**: Apple Mac mini (2024 / M4).
- **SoC**: Apple M4 (10-core CPU: 4 performance + 6 efficiency; 10-core GPU with hardware ray tracing / Metal 3 support).
- **Memory**: 16 GB Unified Memory (LPDDR5X, ~120 GB/s peak bandwidth).
- **OS**: macOS Sequoia 15.4.1 (Darwin arm64).
- **Compute Backends Available**:
  - CPU: ARM64 NEON with hardware dot-product and vector FP/integer units.
  - GPU: Apple M4 Metal unified architecture (`MTLStorageModeShared` zero-copy between CPU and GPU).

---

## 3. Frozen Comparators

1. **Uncompressed Raw SafeTensors / GGUF**: Standard direct file mapping (`mmap`).
2. **Standard Compression Baseline**: `zstd -1` and `zstd -19` standalone and in tarballs.
3. **Internal LMZ Baselines**:
   - Scalar C rANS (`lmzcore.c` scalar).
   - CPU 4-chunk NEON vector rANS.
   - GPU Metal batch rANS decoder (`lmz_rans.metal`).

---

## 4. Success and Stop Conditions

- **Correctness**: 100% byte-exact identity across all test fixtures and archives.
- **Speedup**:
  - GPU Metal batch decoder exceeds CPU single-thread decode by >5×.
  - 4-chunk NEON vector decoder provides >1.5× speedup over scalar rANS on Apple Silicon.
- **Residency**: Unified memory zero-copy (`MTLStorageModeShared`) avoids duplicate memory copies between host RAM and GPU VRAM.
- **Compatibility**: Backward compatible with v1 through v7 archives, pure-Python fallbacks retained.
