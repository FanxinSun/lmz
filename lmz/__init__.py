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

A model can also be kept in the store and read in place, so nothing is ever
expanded onto disk:

    import lmz
    lmz.Store().add("./my-model", "my-model")   # compressed, once

    lmz mount ~/models                          # every runtime reads it as
                                                #   ordinary model files
"""

__version__ = "1.1.2"

from .api import (DEFAULT_CHUNK_SIZE, DEFAULT_LEVEL, Stats, backends, compress,
                  decompress, info, read_tensor, verify)
from .api import MappedArchive, append, extract  # noqa: F401
from .format import FormatError
from .lmzfs import LmzFS  # noqa: F401
from .store import Store, mount  # noqa: F401

__all__ = [
    "compress", "decompress", "verify", "info", "read_tensor", "backends",
    "MappedArchive", "append", "extract", "Store", "mount", "LmzFS",
    "Stats", "FormatError", "DEFAULT_LEVEL", "DEFAULT_CHUNK_SIZE", "__version__",
]
