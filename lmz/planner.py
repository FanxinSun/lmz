"""Working out where the tensors are, so each chunk holds one dtype.

Byte-plane splitting only helps when a chunk contains elements of a single
known width, so the planner reads the container's own index to recover the
dtype layout. Anything it cannot identify -- headers, padding, unknown
formats -- becomes a 1-byte-element region, which still gets compressed,
just without the split.

Both readers are strictly best-effort: a file that does not parse is treated
as opaque bytes rather than rejected.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

# safetensors dtype -> element size in bytes. Unlisted names fall back to 1,
# which is always safe: it just forgoes the split.
ST_ESIZE = {
    "BOOL": 1, "U8": 1, "I8": 1,
    "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1, "F4": 1, "F6_E2M3": 1, "F6_E3M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}

# ggml type id -> (name, block_size, bytes_per_block, element size for splitting)
GGML_TYPES = {
    0: ("F32", 1, 4, 4), 1: ("F16", 1, 2, 2), 2: ("Q4_0", 32, 18, 1),
    3: ("Q4_1", 32, 20, 1), 6: ("Q5_0", 32, 22, 1), 7: ("Q5_1", 32, 24, 1),
    8: ("Q8_0", 32, 34, 34), 9: ("Q8_1", 32, 36, 1), 10: ("Q2_K", 256, 84, 1),
    11: ("Q3_K", 256, 110, 1), 12: ("Q4_K", 256, 144, 1), 13: ("Q5_K", 256, 176, 1),
    14: ("Q6_K", 256, 210, 1), 15: ("Q8_K", 256, 292, 1), 16: ("IQ2_XXS", 256, 66, 1),
    17: ("IQ2_XS", 256, 74, 1), 18: ("IQ3_XXS", 256, 98, 1), 19: ("IQ1_S", 256, 50, 1),
    20: ("IQ4_NL", 32, 18, 1), 21: ("IQ3_S", 256, 110, 1), 22: ("IQ2_S", 256, 82, 1),
    23: ("IQ4_XS", 256, 136, 1), 24: ("I8", 1, 1, 1), 25: ("I16", 1, 2, 2),
    26: ("I32", 1, 4, 4), 27: ("I64", 1, 8, 8), 28: ("F64", 1, 8, 8),
    29: ("IQ1_M", 256, 56, 1), 30: ("BF16", 1, 2, 2), 34: ("TQ1_0", 256, 54, 1),
    35: ("TQ2_0", 256, 66, 1),
}

# ggml type id -> (block bytes, field groups as (start, width)).
#
# A quantised block is a struct, not an array: a Q4_K block is two fp16
# scales, twelve bytes of packed 6-bit sub-scales and 128 bytes of nibble
# pairs. Coding all of that in one alphabet is the mistake that made lmz
# useless on quantised models -- the fp16 exponent bytes and the near-uniform
# quants land in one histogram and erase each other. Field order and widths
# are ggml-common.h's struct layouts; every row sums to the block size in
# GGML_TYPES above, which is what catches a typo here.
#
# A boundary that is wrong costs ratio, never correctness: the split is
# reversed by byte position and knows nothing about what a field means.
BLOCK_LAYOUTS = {
    2:  (18,  ((0, 2), (2, 16))),                               # Q4_0    d, qs
    3:  (20,  ((0, 2), (2, 2), (4, 16))),                       # Q4_1    d, m, qs
    6:  (22,  ((0, 2), (2, 4), (6, 16))),                       # Q5_0    d, qh, qs
    7:  (24,  ((0, 2), (2, 2), (4, 4), (8, 16))),               # Q5_1    d, m, qh, qs
    8:  (34,  ((0, 2), (2, 32))),                               # Q8_0    d, qs
    9:  (36,  ((0, 2), (2, 2), (4, 32))),                       # Q8_1    d, s, qs
    10: (84,  ((0, 16), (16, 64), (80, 2), (82, 2))),           # Q2_K    scales, qs, d, dmin
    11: (110, ((0, 32), (32, 64), (96, 12), (108, 2))),         # Q3_K    hmask, qs, scales, d
    12: (144, ((0, 2), (2, 2), (4, 12), (16, 128))),            # Q4_K    d, dmin, scales, qs
    13: (176, ((0, 2), (2, 2), (4, 12), (16, 32), (48, 128))),  # Q5_K    d, dmin, scales, qh, qs
    14: (210, ((0, 128), (128, 64), (192, 16), (208, 2))),      # Q6_K    ql, qh, scales, d
    16: (66,  ((0, 2), (2, 64))),                               # IQ2_XXS d, qs
    17: (74,  ((0, 2), (2, 64), (66, 8))),                      # IQ2_XS  d, qs, scales
    18: (98,  ((0, 2), (2, 96))),                               # IQ3_XXS d, qs
    19: (50,  ((0, 2), (2, 32), (34, 16))),                     # IQ1_S   d, qs, qh
    20: (18,  ((0, 2), (2, 16))),                               # IQ4_NL  d, qs
    21: (110, ((0, 2), (2, 64), (66, 8), (74, 32), (106, 4))),  # IQ3_S   d, qs, qh, signs, scales
    22: (82,  ((0, 2), (2, 64), (66, 8), (74, 8))),             # IQ2_S   d, qs, qh, scales
    23: (136, ((0, 2), (2, 2), (4, 4), (8, 128))),              # IQ4_XS  d, scales_h, scales_l, qs
    29: (56,  ((0, 32), (32, 16), (48, 8))),                    # IQ1_M   qs, qh, scales
    34: (54,  ((0, 48), (48, 4), (52, 2))),                     # TQ1_0   qs, qh, d
    35: (66,  ((0, 64), (64, 2))),                              # TQ2_0   qs, d
}

# Q8_K (292 bytes) is deliberately absent: ggml only builds it as a dot-product
# intermediate, it never reaches a file, and it would not fit the chunk
# record's one-byte element size.
MAX_BLOCK_PERIOD = 255

# ggml type -> (quant field start, scales field start, packing kind).
#
# A k-quant super-block is not one distribution but eight: each sub-block has
# its own 6-bit scale and 6-bit min, and stores `d*q - m`, so a quant's
# alphabet depends on which sub-block it came from. Those parameters sit
# earlier in the block than the quants, so the decoder already holds them --
# a context that costs nothing to transmit. It is worth 9.7 bits per block on
# real Llama Q4_K weights, against 0.02 bits for anything available inside a
# Q8_0 block, whose one scale covers the whole thing.
#
# Only listed here when the quants really are 8 sub-blocks of 32 nibbles over
# ggml's K_SCALE_SIZE packing; the codec measures per chunk and declines when
# the conditioning does not pay, so a listing can cost ratio but not
# correctness.
#
# Q5_K has exactly the same layout for these two fields and is deliberately
# absent. Its qs holds only the low four bits of a five-bit quant -- the fifth
# lives in qh -- and the low bits of a peaked distribution are near-uniform
# whatever the sub-block does. Measured on real Llama Q5_K it gains 0.000
# points while the estimate itself costs 13% of encode time, so the listing
# would be all cost. The payload describes the layout it used, so registering
# it later needs no format change.
SUB_K4 = 0
SUBBLOCK_CTX = {
    12: (16, 4, SUB_K4),  # Q4_K  qs[128] conditioned on scales[12]
}


def _check_subblock_ctx():
    """Both fields named must be real groups of the block they belong to."""
    for tid, (qstart, cstart, kind) in SUBBLOCK_CTX.items():
        groups = BLOCK_LAYOUTS[tid][1]
        assert kind == SUB_K4, tid
        assert (qstart, 128) in groups, tid
        assert (cstart, 12) in groups, tid
        assert cstart + 12 <= qstart, tid  # context must decode first


_check_subblock_ctx()


def _check_block_layouts():
    """Every group list must tile its block exactly and agree with GGML_TYPES."""
    for tid, (period, groups) in BLOCK_LAYOUTS.items():
        assert period == GGML_TYPES[tid][2] <= MAX_BLOCK_PERIOD, tid
        pos = 0
        for start, width in groups:
            assert start == pos and width > 0, tid
            pos += width
        assert pos == period, tid


_check_block_layouts()

MAX_HEADER = 256 << 20  # refuse absurd declared header sizes

# The longest tensor name or packed dims list worth reading out of an ONNX
# message. Anything past this is not a name, and the parse should not be led
# into a large read by a length prefix it has not verified.
MAX_NAME = 1 << 16


# Element layouts that get their own treatment beyond the element width.
KIND_BYTES = 0
KIND_BF16 = 1
KIND_REF = 2  # bytes duplicate an earlier output range; src is its offset
KIND_BLOCK = 3  # GGUF quantised blocks; esize is the block period
# Bytes are an earlier output range XOR a coded difference. Unlike a ref, the
# difference has the same element structure as the data it came from, so the
# region keeps its width and `ikind` remembers how to code it.
KIND_DELTA = 4

# torch storage class -> (dtype name, element size, kind). Complex dtypes use
# the width of one component pair's half so the split still lines up on a
# power-of-two period.
TORCH_STORAGE = {
    "DoubleStorage": ("F64", 8, KIND_BYTES),
    "FloatStorage": ("F32", 4, KIND_BYTES),
    "HalfStorage": ("F16", 2, KIND_BYTES),
    "BFloat16Storage": ("BF16", 2, KIND_BF16),
    "LongStorage": ("I64", 8, KIND_BYTES),
    "IntStorage": ("I32", 4, KIND_BYTES),
    "ShortStorage": ("I16", 2, KIND_BYTES),
    "CharStorage": ("I8", 1, KIND_BYTES),
    "ByteStorage": ("U8", 1, KIND_BYTES),
    "BoolStorage": ("BOOL", 1, KIND_BYTES),
    "ComplexFloatStorage": ("C64", 8, KIND_BYTES),
    "ComplexDoubleStorage": ("C128", 8, KIND_BYTES),
}


@dataclass(slots=True)
class Region:
    """A byte range of the file whose elements all have the same width."""

    start: int
    end: int
    esize: int
    kind: int = KIND_BYTES
    src: int = -1  # for KIND_REF/KIND_DELTA: the source range's output offset
    btype: int = -1  # for KIND_BLOCK: the ggml type, naming its field layout
    ikind: int = KIND_BYTES  # for KIND_DELTA: how to code the difference

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(slots=True)
class Layout:
    kind: str
    regions: list[Region]
    tensors: dict | None = None


def parse_safetensors(f, size: int) -> Layout | None:
    f.seek(0)
    head = f.read(8)
    if len(head) < 8:
        return None
    hlen = int.from_bytes(head, "little")
    if hlen <= 0 or hlen > MAX_HEADER or 8 + hlen > size:
        return None
    try:
        meta = json.loads(f.read(hlen))
    except (ValueError, UnicodeDecodeError):
        return None
    if not isinstance(meta, dict):
        return None

    base = 8 + hlen
    regions: list[Region] = []
    tensors: dict = {}
    for name, info in meta.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict):
            return None
        offsets = info.get("data_offsets")
        if not (isinstance(offsets, list) and len(offsets) == 2):
            return None
        start, end = offsets
        if not (isinstance(start, int) and isinstance(end, int)):
            return None
        if not (0 <= start <= end) or base + end > size:
            return None
        dtype = info.get("dtype", "")
        esize = ST_ESIZE.get(dtype, 1)
        kind = KIND_BF16 if dtype == "BF16" else KIND_BYTES
        if end > start:
            regions.append(Region(base + start, base + end, esize, kind))
        tensors[name] = {"dtype": dtype, "shape": info.get("shape", []),
                         "offsets": [base + start, base + end]}
    if not tensors:
        return None
    return Layout("safetensors", regions, tensors)


class _Cursor:
    """Little-endian reader over the GGUF header."""

    def __init__(self, f):
        self.f = f

    def u8(self):
        return self.f.read(1)[0]

    def u32(self):
        return struct.unpack("<I", self.f.read(4))[0]

    def i32(self):
        return struct.unpack("<i", self.f.read(4))[0]

    def u64(self):
        return struct.unpack("<Q", self.f.read(8))[0]

    def string(self):
        n = self.u64()
        if n > MAX_HEADER:
            raise ValueError("implausible GGUF string length")
        return self.f.read(n).decode("utf-8", "replace")

    def value(self, vtype):
        if vtype == 0:
            return self.u8()
        if vtype == 1:
            return struct.unpack("<b", self.f.read(1))[0]
        if vtype == 2:
            return struct.unpack("<H", self.f.read(2))[0]
        if vtype == 3:
            return struct.unpack("<h", self.f.read(2))[0]
        if vtype == 4:
            return self.u32()
        if vtype == 5:
            return self.i32()
        if vtype == 6:
            return struct.unpack("<f", self.f.read(4))[0]
        if vtype == 7:
            return bool(self.u8())
        if vtype == 8:
            return self.string()
        if vtype == 9:
            elem = self.u32()
            count = self.u64()
            if elem == 9 or count > (1 << 28):
                raise ValueError("unsupported GGUF array")
            return [self.value(elem) for _ in range(count)]
        if vtype == 10:
            return self.u64()
        if vtype == 11:
            return struct.unpack("<q", self.f.read(8))[0]
        if vtype == 12:
            return struct.unpack("<d", self.f.read(8))[0]
        raise ValueError(f"unknown GGUF value type {vtype}")


def parse_gguf(f, size: int) -> Layout | None:
    f.seek(0)
    if f.read(4) != b"GGUF":
        return None
    try:
        c = _Cursor(f)
        version = c.u32()
        if version not in (2, 3):
            return None
        n_tensors = c.u64()
        n_kv = c.u64()
        if n_tensors > (1 << 24) or n_kv > (1 << 20):
            return None

        alignment = 32
        for _ in range(n_kv):
            key = c.string()
            val = c.value(c.u32())
            if key == "general.alignment" and isinstance(val, int) and val > 0:
                alignment = val

        infos = []
        for _ in range(n_tensors):
            name = c.string()
            ndim = c.u32()
            if ndim > 8:
                return None
            dims = [c.u64() for _ in range(ndim)]
            ttype = c.u32()
            offset = c.u64()
            infos.append((name, dims, ttype, offset))

        pos = f.tell()
        data_start = (pos + alignment - 1) // alignment * alignment
        if data_start > size:
            return None
    except (ValueError, struct.error, IndexError, UnicodeDecodeError):
        return None

    # Sizes come from the ggml type table, but are clamped to the gap before
    # the next tensor so an unfamiliar quantisation cannot overrun.
    infos.sort(key=lambda t: t[3])
    regions: list[Region] = []
    tensors: dict = {}
    for i, (name, dims, ttype, offset) in enumerate(infos):
        start = data_start + offset
        limit = data_start + infos[i + 1][3] if i + 1 < len(infos) else size
        if not (data_start <= start <= limit <= size):
            return None
        info = GGML_TYPES.get(ttype)
        nbytes = limit - start
        if info:
            tname, blk, per_blk, esize = info
            nelem = 1
            for d in dims:
                nelem *= d
            if blk and nelem % blk == 0:
                nbytes = min(nbytes, nelem // blk * per_blk)
        else:
            tname, esize = f"TYPE_{ttype}", 1
        btype = -1
        if ttype == 30:
            kind = KIND_BF16
        elif ttype in BLOCK_LAYOUTS:
            period = BLOCK_LAYOUTS[ttype][0]
            # The block structure only holds if the tensor really is whole
            # blocks; a clamped or odd size falls back to plain bytes.
            if nbytes % period == 0:
                esize, kind, btype = period, KIND_BLOCK, ttype
            else:
                esize, kind = 1, KIND_BYTES
        else:
            kind = KIND_BYTES
        if nbytes > 0:
            regions.append(Region(start, start + nbytes, esize, kind, -1, btype))
        tensors[name] = {"dtype": tname, "shape": dims,
                         "offsets": [start, start + nbytes]}
    if not tensors:
        return None
    return Layout("gguf", regions, tensors)


def _pickle_storage_types(pkl: bytes) -> dict:
    """Map torch storage keys to storage class names, without unpickling.

    torch's persistent ids are tuples of ('storage', StorageClass, key,
    location, numel). In the opcode stream the class arrives as a GLOBAL
    (protocol 2) or STACK_GLOBAL (4+), possibly through a memo reference,
    and the very next string pushed is the key. pickletools only reads the
    stream, so nothing here executes.
    """
    import pickletools

    memo: dict = {}
    strings: list = []  # the last two string pushes, for STACK_GLOBAL
    last = None  # what the latest op left on top: ("global", name) or ("str", s)
    pending = None
    out: dict = {}
    for op, arg, _pos in pickletools.genops(pkl):
        name = op.name
        if name in ("SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
                    "STRING", "BINSTRING", "SHORT_BINSTRING"):
            if pending is not None:
                out[str(arg)] = pending
                pending = None
            last = ("str", arg)
            strings.append(arg)
            del strings[:-2]
        elif name == "GLOBAL":
            g = str(arg).split()[-1]
            last = ("global", g)
            if g.endswith("Storage"):
                pending = g
        elif name == "STACK_GLOBAL":
            g = str(strings[-1]) if strings else ""
            last = ("global", g)
            if g.endswith("Storage"):
                pending = g
        elif name in ("BINPUT", "LONG_BINPUT"):
            memo[arg] = last
        elif name == "MEMOIZE":
            memo[len(memo)] = last
        elif name in ("BINGET", "LONG_BINGET"):
            last = memo.get(arg)
            if last and last[0] == "global" and last[1].endswith("Storage"):
                pending = last[1]
            elif last and last[0] == "str":
                strings.append(last[1])
                del strings[:-2]
        else:
            last = None
    return out


def parse_pytorch_zip(f, size: int) -> Layout | None:
    """The torch.save zip container: data.pkl plus one file per storage.

    Storages are stored uncompressed, so their payload ranges can be typed
    from the pickled storage classes and split like any other tensor data.
    Anything unexpected -- compressed entries, an unreadable pickle, keys
    with no known dtype -- degrades to untyped bytes rather than failing.
    """
    import zipfile

    f.seek(0)
    if f.read(4) != b"PK\x03\x04":
        return None
    try:
        zf = zipfile.ZipFile(f)
        entries = zf.infolist()
    except Exception:
        return None
    pkl_entry = next((zi for zi in entries
                      if zi.filename.endswith("data.pkl")
                      and zi.file_size <= MAX_HEADER), None)
    if pkl_entry is None:
        return None
    prefix = pkl_entry.filename[:-len("data.pkl")]
    try:
        dtypes = _pickle_storage_types(zf.read(pkl_entry))
    except Exception:
        dtypes = {}

    regions: list[Region] = []
    tensors: dict = {}
    for zi in entries:
        if not zi.filename.startswith(prefix + "data/"):
            continue
        key = zi.filename.rsplit("/", 1)[1]
        if zi.compress_type != zipfile.ZIP_STORED or zi.file_size == 0:
            continue
        # The central directory's extra field can differ from the local one,
        # so the payload offset comes from the local header itself.
        f.seek(zi.header_offset)
        local = f.read(30)
        if len(local) < 30 or local[:4] != b"PK\x03\x04":
            continue
        nlen, elen = struct.unpack("<HH", local[26:30])
        start = zi.header_offset + 30 + nlen + elen
        end = start + zi.file_size
        if end > size:
            continue
        dtype, esize, kind = TORCH_STORAGE.get(dtypes.get(key, ""),
                                               ("U8", 1, KIND_BYTES))
        if zi.file_size % esize:
            dtype, esize, kind = "U8", 1, KIND_BYTES
        regions.append(Region(start, end, esize, kind))
        tensors[f"data/{key}"] = {"dtype": dtype, "shape": [],
                                  "offsets": [start, end]}
    if not tensors:
        return None
    return Layout("pytorch", regions, tensors)


# ONNX TensorProto.DataType -> (name, element size, split kind). The numbering
# is fixed by the format; only the types that carry weights are listed, and
# anything else falls back to untyped bytes rather than being guessed at.
ONNX_DTYPES = {
    1: ("F32", 4), 2: ("U8", 1), 3: ("I8", 1), 4: ("U16", 2), 5: ("I16", 2),
    6: ("I32", 4), 7: ("I64", 8), 9: ("BOOL", 1), 10: ("F16", 2),
    11: ("F64", 8), 12: ("U32", 4), 13: ("U64", 8), 16: ("BF16", 2),
    17: ("F8_E4M3", 1), 18: ("F8_E5M2", 1),
}

# Field numbers inside the protobuf messages this walks. ONNX is a stable
# published schema, so these are constants of the format rather than guesses.
_PB_MODEL_GRAPH = 7        # ModelProto.graph
_PB_GRAPH_INITIALIZER = 5  # GraphProto.initializer
_PB_TENSOR_DIMS = 1        # TensorProto.dims
_PB_TENSOR_DTYPE = 2       # TensorProto.data_type
_PB_TENSOR_NAME = 8        # TensorProto.name
_PB_TENSOR_RAW = 9         # TensorProto.raw_data


def _pb_varint(buf, i: int):
    """One protobuf varint. Returns (value, next index)."""
    val = shift = 0
    while True:
        if i >= len(buf) or shift > 63:
            raise ValueError("bad varint")
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if not b & 0x80:
            return val, i
        shift += 7


def _pb_read_varint(f, limit: int):
    """One varint read from a file. Returns (value, bytes consumed)."""
    val = shift = used = 0
    while True:
        if used >= 10 or f.tell() >= limit:
            raise ValueError("bad varint")
        b = f.read(1)
        if not b:
            raise ValueError("bad varint")
        used += 1
        val |= (b[0] & 0x7F) << shift
        if not b[0] & 0x80:
            return val, used
        shift += 7


def _pb_scan_file(f, start: int, end: int):
    """Walk a protobuf message in a file, yielding (field, wire, a, b).

    The same walk as `_pb_fields`, against a file rather than a buffer, so a
    multi-gigabyte ONNX model can be traversed by seeking over its weight
    payloads instead of reading them. Ranges are absolute file offsets.
    """
    i = start
    while i < end:
        f.seek(i)
        key, used = _pb_read_varint(f, end)
        i += used
        field, wire = key >> 3, key & 7
        if wire == 0:
            f.seek(i)
            val, used = _pb_read_varint(f, end)
            i += used
            yield field, wire, val, val
        elif wire == 1:
            i += 8
        elif wire == 2:
            f.seek(i)
            n, used = _pb_read_varint(f, end)
            i += used
            if n < 0 or i + n > end:
                raise ValueError("length-delimited field runs past its message")
            yield field, wire, i, i + n
            i += n
        elif wire == 5:
            i += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        if i > end:
            raise ValueError("field runs past its message")


def _pb_find_from_file(f, start: int, end: int, want: int):
    """The range of the first length-delimited field `want`, or None."""
    for field, wire, a, b in _pb_scan_file(f, start, end):
        if field == want and wire == 2:
            return a, b
    return None


def _pb_fields(buf, start: int, end: int):
    """Walk one protobuf message, yielding (field number, wire type, a, b).

    For a length-delimited field, (a, b) is the payload's absolute range; for
    a varint it is (value, value). Groups are not emitted -- they are a
    deprecated wire type that ONNX does not use, and skipping one correctly
    needs a matching end marker, so hitting one abandons the parse.
    """
    i = start
    while i < end:
        key, i = _pb_varint(buf, i)
        field, wire = key >> 3, key & 7
        if wire == 0:
            val, i = _pb_varint(buf, i)
            yield field, wire, val, val
        elif wire == 1:
            i += 8
        elif wire == 2:
            n, i = _pb_varint(buf, i)
            if n < 0 or i + n > end:
                raise ValueError("length-delimited field runs past its message")
            yield field, wire, i, i + n
            i += n
        elif wire == 5:
            i += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        if i > end:
            raise ValueError("field runs past its message")


def parse_onnx(f, size: int) -> Layout | None:
    """The ONNX container: a protobuf whose initializers hold the weights.

    Perception models ship as ONNX, and without a parse every initializer
    travels as opaque bytes -- which costs real ratio, because the byte-position
    planes cannot phase-align without knowing the element width, so an fp16
    exponent smears across positions. Measured worth about 3 points on a
    detector against the same bytes as a blob.

    This reads the field numbers it needs and skips everything else: the graph,
    its initializers, and each one's dtype, dims and `raw_data` range. Nothing
    is executed and no protobuf library is required -- the same bargain
    `parse_pytorch_zip` makes with the pickle, where an opcode scan replaces
    unpickling.

    Returns None for the external-data form, where `raw_data` is empty and the
    tensors live in a sidecar file. Those files are flat, so lmz already codes
    them well as raw regions; typing them needs the sidecar's own name, which
    is a separate step.
    """
    if size < 8:
        return None
    f.seek(0)
    # ONNX has no magic number. ir_version is field 1, a varint, so a real
    # model starts with 0x08 -- the cheapest way to decline a file that is not
    # this protobuf before reading any more of it.
    if f.read(1) != b"\x08":
        return None

    # An ONNX file with internal weights is one protobuf over the whole file,
    # so "the header" is the file. Only the structure is needed, never the
    # weight bytes, and the structure is a walk of length prefixes -- so the
    # top level is walked against the file and only the message that actually
    # has to be inspected is read in.
    graph = _pb_find_from_file(f, 0, size, _PB_MODEL_GRAPH)
    if graph is None:
        return None

    regions: list[Region] = []
    tensors: dict = {}
    # Each initializer is walked against the file too. Its metadata -- name,
    # dtype, dims -- is small and comes before the weights, but `raw_data` is
    # the tensor itself, so it is located and skipped rather than read.
    for field, wire, a, b in _pb_scan_file(f, graph[0], graph[1]):
        if field != _PB_GRAPH_INITIALIZER or wire != 2:
            continue
        name, dtype_id, dims, raw = "", 0, [], None
        for tf, tw, ta, tb in _pb_scan_file(f, a, b):
            if tf == _PB_TENSOR_RAW and tw == 2:
                raw = (ta, tb)          # located, never read
            elif tf == _PB_TENSOR_DTYPE and tw == 0:
                dtype_id = ta
            elif tf == _PB_TENSOR_DIMS and tw == 0:
                dims.append(ta)
            elif tf == _PB_TENSOR_NAME and tw == 2 and tb - ta <= MAX_NAME:
                f.seek(ta)
                name = f.read(tb - ta).decode("utf-8", "replace")
            elif tf == _PB_TENSOR_DIMS and tw == 2 and tb - ta <= MAX_NAME:
                # dims is `repeated int64`, so a writer may pack it.
                f.seek(ta)
                packed = f.read(tb - ta)
                j = 0
                while j < len(packed):
                    v, j = _pb_varint(packed, j)
                    dims.append(v)
        if raw is None or raw[1] <= raw[0] or raw[1] > size:
            continue
        dtype, esize = ONNX_DTYPES.get(dtype_id, ("U8", 1))
        if (raw[1] - raw[0]) % esize:
            dtype, esize = "U8", 1
        kind = KIND_BF16 if dtype == "BF16" else KIND_BYTES
        regions.append(Region(raw[0], raw[1], esize, kind))
        tensors[name or f"initializer_{len(tensors)}"] = {
            "dtype": dtype, "shape": dims, "offsets": [raw[0], raw[1]]}

    if not tensors:
        return None
    return Layout("onnx", regions, tensors)


def probe(f, size: int) -> Layout:
    """Identify the file's tensor layout, falling back to opaque bytes."""
    for parser in (parse_safetensors, parse_gguf, parse_pytorch_zip,
                   parse_onnx):
        try:
            layout = parser(f, size)
        except Exception:
            layout = None
        if layout is not None:
            return layout
    return Layout("raw", [], None)


def _carve_refs(regions: list[Region], refs, size: int) -> list[Region]:
    """Cut duplicate ranges out of the region list and insert them as refs.

    `refs` is a list of (start, end, source_output_offset). Anything
    malformed -- overlapping or out of bounds -- drops the whole dedup rather
    than risking the layout: refs are an optimisation, never a requirement.
    """
    from bisect import bisect_right

    refs = sorted(refs)
    pos = 0
    for s, e, _src in refs:
        if s < pos or e <= s or e > size:
            return regions
        pos = e
    ref_starts = [s for s, _e, _src in refs]
    out: list[Region] = []
    for r in regions:
        cur = r.start
        k = bisect_right(ref_starts, cur)
        if k and refs[k - 1][1] > cur:
            k -= 1
        while k < len(refs) and refs[k][0] < r.end:
            s, e, _src = refs[k]
            a, b = max(s, r.start), min(e, r.end)
            if b > cur:
                if a > cur:
                    out.append(Region(cur, a, r.esize, r.kind, -1, r.btype))
                cur = b
            k += 1
        if cur < r.end:
            out.append(Region(cur, r.end, r.esize, r.kind, -1, r.btype))
    out.extend(Region(s, e, 1, KIND_REF, src) for s, e, src in refs)
    out.sort(key=lambda r: r.start)
    return out


def _carve_deltas(regions: list[Region], deltas, size: int) -> list[Region]:
    """Retype ranges that will be coded as a difference from an earlier one.

    Unlike a ref, a delta still carries data, so it keeps the covering
    region's element width and remembers that region's kind for the encoder.
    A range spanning several regions is cut at their boundaries, so each piece
    is coded the way its own bytes deserve.
    """
    from bisect import bisect_right

    deltas = sorted(deltas)
    pos = 0
    for s, e, _src in deltas:
        if s < pos or e <= s or e > size:
            return regions  # malformed: deltas are an optimisation, never a need
        pos = e
    starts = [s for s, _e, _src in deltas]

    out: list[Region] = []
    for r in regions:
        if r.kind in (KIND_REF, KIND_DELTA):
            out.append(r)
            continue
        cur = r.start
        k = bisect_right(starts, cur)
        if k and deltas[k - 1][1] > cur:
            k -= 1
        while k < len(deltas) and deltas[k][0] < r.end:
            s, e, src = deltas[k]
            a, b = max(s, r.start), min(e, r.end)
            if b > cur:
                if a > cur:
                    out.append(Region(cur, a, r.esize, r.kind, -1, r.btype))
                out.append(Region(a, b, r.esize, KIND_DELTA, src + (a - s),
                                  r.btype, r.kind))
                cur = b
            k += 1
        if cur < r.end:
            out.append(Region(cur, r.end, r.esize, r.kind, -1, r.btype))
    return out


def chunkify(layout: Layout, size: int, chunk_size: int, refs=None,
             deltas=None) -> list[tuple[int, int, int, int, int, int, int]]:
    """Cover [0, size) with (start, end, esize, kind, src, btype, ikind) chunks.

    Regions are coalesced before slicing so that runs of small same-dtype
    tensors -- a model has thousands of biases and norm weights -- do not each
    become their own undersized chunk. `refs` marks byte ranges that duplicate
    an earlier output range; those become KIND_REF chunks carrying the source
    offset in `src` (-1 everywhere else). `deltas` marks ranges that are merely
    *close* to an earlier range, which become KIND_DELTA chunks carrying both
    the source offset and the width to code the difference at. `btype` names a
    KIND_BLOCK chunk's GGUF field layout, and is -1 for every other kind.
    """
    regions = sorted(layout.regions, key=lambda r: r.start)
    if refs:
        regions = _carve_refs(regions, refs, size)
    if deltas:
        regions = _carve_deltas(regions, deltas, size)

    # Fill the gaps (header, padding, anything unclaimed) with 1-byte elements.
    covered: list[Region] = []
    pos = 0
    for r in regions:
        if r.start < pos:  # overlapping or out-of-order index: give up on splitting
            return _slice_region(0, size, 1, KIND_BYTES, chunk_size)
        if r.start > pos:
            covered.append(Region(pos, r.start, 1))
        covered.append(r)
        pos = r.end
    if pos < size:
        covered.append(Region(pos, size, 1))

    merged: list[Region] = []
    for r in covered:
        prev = merged[-1] if merged else None
        sourced = r.kind in (KIND_REF, KIND_DELTA)
        joins = (prev is not None and prev.end == r.start and prev.kind == r.kind
                 and prev.btype == r.btype and prev.ikind == r.ikind
                 and (prev.src + prev.length == r.src if sourced else True)
                 and (prev.esize == r.esize or r.kind == KIND_REF))
        if joins:
            merged[-1] = Region(prev.start, r.end, prev.esize, prev.kind,
                                prev.src, prev.btype, prev.ikind)
        else:
            merged.append(r)

    out: list[tuple[int, int, int, int, int, int, int]] = []
    for r in merged:
        out.extend(_slice_region(r.start, r.end, r.esize, r.kind, chunk_size,
                                 r.src, r.btype, r.ikind))
    return out


def _slice_region(start: int, end: int, esize: int, kind: int, chunk_size: int,
                  src: int = -1, btype: int = -1, ikind: int = KIND_BYTES):
    """Cut [start, end) into chunks whose lengths are multiples of esize."""
    step = max(chunk_size - chunk_size % esize, esize)
    out = []
    pos = start
    while pos < end:
        stop = min(pos + step, end)
        # A trailing piece shorter than one element cannot be split; the
        # region boundary already guarantees alignment, so this only guards
        # against malformed inputs.
        if kind != KIND_REF and (stop - pos) % esize and stop == end:
            out.append((pos, stop, 1, KIND_BYTES, -1, -1, KIND_BYTES))
        else:
            out.append((pos, stop, esize, kind,
                        src + (pos - start) if src >= 0 else -1, btype, ikind))
        pos = stop
    return out
