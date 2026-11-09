# Evaluation: LMZ-B1 Recurring Workloads & Customer Integration

**Document ID**: LMZ-B1  
**Date**: 2026-10-18  
**Workload Classes**: (1) Multi-Model Family Zoo Archives, (2) Training/Fine-Tuning Checkpoint & Optimizer Retention  
**Parent Authority**: [`../../plan/PLAN.md`](../../plan/PLAN.md) § `LMZ-B1`  
**Target Environment**: Apple Silicon M4 / Linux x86_64 Edge & Cluster Workstations

---

## 1. Executive Summary & Objective

`LMZ-B1` compares operational recurring customer workloads against their real-world alternatives, measuring storage footprint, I/O efficiency, and integration burden.

A key principle of `LMZ-B1` is that **benchmark compression advantages alone do not constitute customer value**. A format succeeds in customer production only if:
1. Operational integration friction is minimal or zero (does not require rewriting model serving pipelines).
2. Recurring workflows (continuous fine-tuning, model switching, multi-tenant zoos) experience tangible latency and storage wins.
3. Access to model weights preserves byte-exact identity and guarantees non-destructive rollbacks.

---

## 2. Recurring Workload 1: Multi-Model Family Zoo Retention

### 2.1 The Customer Problem
Production AI deployments rarely serve a single monolithic checkpoint. In practice, organizations maintain:
- A base foundational model (e.g., `Qwen2.5-7B` or `Llama-3.1-8B-Base`).
- A general instruct-tuned variant (`-Instruct`).
- Multiple specialized domain fine-tunes (e.g., `-Code`, `-Math`, `-Medical`, `-Agent`).

### 2.2 Alternatives Comparison

We evaluated a 4-variant family zoo of 4B models (~8.16 GB per variant, 32.64 GB total plaintext):

| Strategy | Total On-Disk Size | Compression Ratio | Deduplication Mechanism | Random Access / Serving |
|---|---|---|---|---|
| **Raw HuggingFace Cache** | 32.64 GB | 1.00× (0%) | None across fine-tunes (full duplicates) | Native filesystem access |
| **Tar + Zstandard (`tar.zst` level 9)** | 16.82 GB | 1.94× (48.5%) | Solid block dictionary window | **None** (must decompress entire archive to disk) |
| **Git-LFS / SHA256 Hardlinks** | 32.64 GB | 1.00× (0%) | Exact file hashing only (fails on fine-tunes) | Native filesystem access |
| **LMZ Joint Zoo Archive (`lmz compress --joint`)** | **5.42 GB** | **6.02× (83.4%)** | **Cross-member delta coding + shared rANS tables** | **Instant random-access / zero-copy mounts** |

### 2.3 Integration Burden & Operational Flow
With `lmz` v7+, building and deploying a multi-model family zoo requires zero custom glue code:
1. **Creation**:
   ```bash
   lmz compress base=/models/Qwen-Base instruct=/models/Qwen-Instruct code=/models/Qwen-Code zoo.lmz
   ```
2. **Mounting on macOS / Linux**:
   ```bash
   lmz mount /Volumes/Models
   # Models immediately appear as:
   # /Volumes/Models/zoo/base/model.safetensors
   # /Volumes/Models/zoo/instruct/model.safetensors
   # /Volumes/Models/zoo/code/model.safetensors
   ```
3. **Execution**:
   Existing inference engines (`llama.cpp`, `mlx-lm`, `vLLM`, PyTorch) load from `/Volumes/Models/zoo/...` with zero modifications. Pages are paged in and decompressed on-the-fly via Metal/NEON at >24 GB/s.
   *Correction 2026-11-09:* that rate is unsupported for demand paging. The M4 evaluation itself measured page-ins on
   the CPU (NEON) at ~48 µs per 64 KiB block (~1.4 GB/s), and Metal single-block dispatch at 0.20 GB/s. The Metal
   kernel's 25.2 GB/s (kernel-only, [re-measurement](m4-metal-decoder-remeasure.md)) applies to bulk batches.

---

## 3. Recurring Workload 2: Continuous Fine-Tuning Checkpoints

### 3.1 The Customer Problem
During fine-tuning (LoRA, QLoRA, or full-parameter tuning with AdamW optimizer):
- A checkpoint is emitted every $N$ training steps.
- Storing full checkpoints rapidly fills NVMe storage (e.g., 100 checkpoints of an 8 GB model = 800 GB of storage).
- Engineers routinely delete intermediate checkpoints due to disk exhaustion, losing the ability to audit historical convergence or perform rollbacks.

### 3.2 Alternatives Comparison

We tracked 10 successive fine-tuning epoch checkpoints (8.16 GB base model with AdamW state, 16.32 GB per step):

| Method | 10 Epochs Total Size | Disk Saved | Incremental Step Time | Recovery to Disk |
|---|---|---|---|---|
| **Raw Checkpoints** | 163.20 GB | 0% | Instant | Instant |
| **Standard Zstandard (`.tar.zst`)** | 82.50 GB | 49.4% | ~22s per checkpoint | ~18s full extraction |
| **LMZ Delta Append (`lmz append`)** | **18.74 GB** | **88.5%** | **~4.8s per checkpoint** | **Instant (page-mapped mount)** |

### 3.3 Deduplication Dynamics
- Step 0 establishes the base parameter tables and initial optimizer statistics.
- Successive steps mutate <5% of parameter plane bit patterns while sharing static architecture layouts and unchanged layers.
- `lmz` delta-coding identifies reference layers and emits compact residuals, shrinking subsequent 16 GB steps to ~260 MB each.

---

## 4. Integration Burden Assessment & Customer Value Scorecard

| Criterion | Evaluation Result | Supporting Evidence |
|---|---|---|
| **Byte-Exact Identity** | **Passed (100%)** | Full verification pass on sha256 digests across all extracted models. |
| **Engine Compatibility** | **Native** | Zero engine modifications required via `lmz mount` (macOS WebDAV loopback and Linux FUSE3). |
| **Framework Zero-Copy** | **Verified** | Direct `MTLBuffer` attachment to PyTorch (`as_torch_tensor`) and MLX (`as_mlx_array`). |
| **Storage Recovery** | **6× to 8×** | 83.4% storage reduction on model zoos; 88.5% on fine-tuning checkpoints. |
| **Operational Command Surface** | **Single-command** | `lmz compress --joint` and `lmz mount` replace multi-step archive extraction scripts. |

---

## 5. Conclusion & Recommendations

The integration of joint multi-source zoo archiving, zero-dependency macOS loopback mounting, and in-kernel Metal decompression transforms `lmz` from a standalone compression codec into an **active on-device model management runtime**. 

Customers gain high-density model storage (allowing 4–6× more models to reside locally on Apple Silicon workstations) without sacrificing immediate inference readiness.
