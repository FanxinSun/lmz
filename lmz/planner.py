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

MAX_HEADER = 256 << 20  # refuse absurd declared header sizes


# Element layouts that get their own treatment beyond the element width.
KIND_BYTES = 0
KIND_BF16 = 1
KIND_REF = 2  # bytes duplicate an earlier output range; src is its offset
KIND_Q80 = 3  # GGUF Q8_0 blocks: 2-byte fp16 scale + 32 int8 quants

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
    src: int = -1  # for KIND_REF: the duplicated range's output offset

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
        if ttype == 30:
            kind = KIND_BF16
        elif ttype == 8:
            kind = KIND_Q80
            # The block structure only holds if the tensor really is whole
            # 34-byte blocks; a clamped or odd size falls back to plain bytes.
            if nbytes % 34:
                esize, kind = 1, KIND_BYTES
        else:
            kind = KIND_BYTES
        if nbytes > 0:
            regions.append(Region(start, start + nbytes, esize, kind))
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


def probe(f, size: int) -> Layout:
    """Identify the file's tensor layout, falling back to opaque bytes."""
    for parser in (parse_safetensors, parse_gguf, parse_pytorch_zip):
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
                    out.append(Region(cur, a, r.esize, r.kind))
                cur = b
            k += 1
        if cur < r.end:
            out.append(Region(cur, r.end, r.esize, r.kind))
    out.extend(Region(s, e, 1, KIND_REF, src) for s, e, src in refs)
    out.sort(key=lambda r: r.start)
    return out


def chunkify(layout: Layout, size: int, chunk_size: int,
             refs=None) -> list[tuple[int, int, int, int, int]]:
    """Cover [0, size) with (start, end, esize, kind, src) chunks.

    Regions are coalesced before slicing so that runs of small same-dtype
    tensors -- a model has thousands of biases and norm weights -- do not each
    become their own undersized chunk. `refs` marks byte ranges that duplicate
    an earlier output range; those become KIND_REF chunks carrying the source
    offset in `src` (-1 everywhere else).
    """
    regions = sorted(layout.regions, key=lambda r: r.start)
    if refs:
        regions = _carve_refs(regions, refs, size)

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
        joins = (prev is not None and prev.end == r.start and prev.kind == r.kind
                 and (prev.src + prev.length == r.src if r.kind == KIND_REF
                      else prev.esize == r.esize))
        if joins:
            merged[-1] = Region(prev.start, r.end, prev.esize, prev.kind, prev.src)
        else:
            merged.append(r)

    out: list[tuple[int, int, int, int, int]] = []
    for r in merged:
        out.extend(_slice_region(r.start, r.end, r.esize, r.kind, chunk_size, r.src))
    return out


def _slice_region(start: int, end: int, esize: int, kind: int, chunk_size: int,
                  src: int = -1):
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
            out.append((pos, stop, 1, KIND_BYTES, -1))
        else:
            out.append((pos, stop, esize, kind,
                        src + (pos - start) if src >= 0 else -1))
        pos = stop
    return out
