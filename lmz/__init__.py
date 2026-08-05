"""lmz - fast lossless compression for large model weights.

Model weights compress poorly with general-purpose tools because a float
array looks like noise byte for byte. It is not: in a BF16 tensor the
sign-and-exponent byte of every element takes only a few dozen distinct
values, while the mantissa byte really is close to random. Separating those
into distinct planes lets the exponents be entropy coded properly and the
mantissas be passed through untouched, which is both smaller and faster than
compressing the interleaved bytes.

    import lmz
    lmz.compress("model.safetensors", "model.lmz")
    lmz.decompress("model.lmz", "restored.safetensors")

Output is byte-for-byte identical to the input.
"""

__version__ = "0.3.0"

from .api import (DEFAULT_CHUNK_SIZE, DEFAULT_LEVEL, Stats, backends, compress,
                  decompress, info, read_tensor, verify)
from .format import FormatError

__all__ = [
    "compress", "decompress", "verify", "info", "read_tensor", "backends",
    "Stats", "FormatError", "DEFAULT_LEVEL", "DEFAULT_CHUNK_SIZE", "__version__",
]
