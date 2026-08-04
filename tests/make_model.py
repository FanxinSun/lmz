"""Generate synthetic model files that behave like real checkpoints.

Real weights are not uniform noise: magnitudes vary per output channel, which
is what gives the exponent plane its skew. Sampling from a lognormal-scaled
normal reproduces that, so measurements here track what real checkpoints do
rather than flattering the codec.
"""

from __future__ import annotations

import argparse
import json
import os
import struct

import numpy as np

DTYPES = {
    "bf16": ("BF16", 2), "f16": ("F16", 2), "f32": ("F32", 4),
    # A checkpoint saved as F32 but trained in bfloat16: every value has 16
    # zero mantissa bits. Common in the wild, and worth measuring separately.
    "f32u": ("F32", 4),
    "i8": ("I8", 1), "i64": ("I64", 8),
}


def to_bf16_bytes(x: np.ndarray) -> np.ndarray:
    """Round float32 to bfloat16, ties to even, returning the raw uint16 words."""
    u = x.astype(np.float32).view(np.uint32)
    rounded = (u.astype(np.uint64) + (((u >> 16) & 1).astype(np.uint64) + 0x7FFF))
    return (rounded >> 16).astype(np.uint16)


def make_tensor(rng, shape, kind: str) -> np.ndarray:
    rows = shape[0]
    # float32 throughout: lognormal returns float64, and letting that promote
    # the product would silently write 8-byte values under a 4-byte dtype.
    scale = (rng.lognormal(0.0, 0.5, size=(rows,) + (1,) * (len(shape) - 1))
             * 0.02).astype(np.float32)
    if kind == "bf16":
        return to_bf16_bytes(rng.standard_normal(shape, dtype=np.float32) * scale)
    if kind == "f16":
        return (rng.standard_normal(shape, dtype=np.float32) * scale).astype(np.float16)
    if kind == "f32":
        return (rng.standard_normal(shape, dtype=np.float32) * scale).astype(np.float32)
    if kind == "f32u":
        x = (rng.standard_normal(shape, dtype=np.float32) * scale).astype(np.float32)
        return (x.view(np.uint32) & 0xFFFF0000).view(np.float32)
    if kind == "i8":
        return rng.integers(-127, 128, shape, dtype=np.int8)
    if kind == "i64":
        return rng.integers(0, 32000, shape, dtype=np.int64)
    raise ValueError(kind)


def write_safetensors(path: str, tensors: dict, metadata: dict | None = None) -> int:
    header, offset, blobs = {}, 0, []
    for name, (dtype, shape, arr) in tensors.items():
        raw = arr.tobytes()
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    if metadata:
        header["__metadata__"] = metadata
    blob = json.dumps(header, separators=(",", ":")).encode()
    pad = (-len(blob)) % 8  # safetensors aligns the data section to 8 bytes
    blob += b" " * pad
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for raw in blobs:
            fh.write(raw)
    return os.path.getsize(path)


def build(path: str, layers: int = 8, hidden: int = 2048, kind: str = "bf16",
          seed: int = 0) -> int:
    """A transformer-shaped checkpoint: attention and MLP blocks plus norms."""
    rng = np.random.default_rng(seed)
    dtype, _ = DTYPES[kind]
    tensors: dict = {}
    ffn = hidden * 4

    def add(name, shape):
        tensors[name] = (dtype, shape, make_tensor(rng, shape, kind))

    add("model.embed_tokens.weight", (32000, hidden))
    for i in range(layers):
        p = f"model.layers.{i}"
        for proj, shape in (("q_proj", (hidden, hidden)), ("k_proj", (hidden, hidden)),
                            ("v_proj", (hidden, hidden)), ("o_proj", (hidden, hidden)),
                            ("gate_proj", (ffn, hidden)), ("up_proj", (ffn, hidden)),
                            ("down_proj", (hidden, ffn))):
            sub = "self_attn" if proj.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) \
                else "mlp"
            add(f"{p}.{sub}.{proj}.weight", shape)
        add(f"{p}.input_layernorm.weight", (hidden,))
        add(f"{p}.post_attention_layernorm.weight", (hidden,))
    add("model.norm.weight", (hidden,))
    add("lm_head.weight", (32000, hidden))
    return write_safetensors(path, tensors, {"format": "pt", "generator": "lmz-tests"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--dtype", default="bf16", choices=sorted(DTYPES))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    size = build(a.path, a.layers, a.hidden, a.dtype, a.seed)
    print(f"{a.path}: {size / (1 << 20):.1f} MiB")
