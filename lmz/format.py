"""The .lmz container.

Layout:

    [32-byte header]
    [chunk payloads, in destination order]
    [chunk table, zstd]
    [manifest JSON, zstd]
    [40-byte footer]

Every chunk records where it belongs in the reconstructed output, so
decompression is a set of wholly independent writes: N threads decode N
chunks and place them at known offsets with no coordination and no locks.
Putting the table and manifest at the tail lets compression stream in a
single pass, since chunk sizes are only known once they are compressed.

A multi-file archive treats its members as one virtual byte range laid end to
end; each chunk's destination is an offset in that space, which keeps the
single-file and directory cases on exactly the same path.
"""

from __future__ import annotations

import array
import io
import json
import struct
import sys
from dataclasses import dataclass, field
from itertools import islice

MAGIC = b"LMZ\x01"
TAIL = b"LMZTAIL\x01"
# v2 added ref chunks (cross-file tensor dedup) and conditioned BF16 chunks;
# v3 added block-split chunks for GGUF Q8_0; v4 replaces those with a block
# split that carries its own field layout, which covers every GGUF
# quantisation rather than one; v5 adds a field mode that codes a k-quant's
# quants per sub-block class, described in the payload alongside the rest;
# v6 adds delta chunks, which name an earlier output range the way a ref does
# and carry the coded difference from it; v7 adds a byte-plane split whose
# rANS streams are coded against a table carried once in the manifest instead
# of once per stream. Earlier archives contain none of the newer codecs, so
# this build reads every version listed.
FORMAT_VERSION = 7
READABLE_VERSIONS = (1, 2, 3, 4, 5, 6, 7)

# What an archive is stamped with when it uses none of the newest codecs.
# The stamp is a *requirement on the reader*, not a build number: writing
# FORMAT_VERSION unconditionally would make every ordinary archive this build
# produces unreadable to the previous release, which can decode all of it. So
# the writer stamps the lowest version that can read what it actually wrote,
# and only a chunk below raises it.
BASE_VERSION = 6
CODEC_MIN_VERSION = {}

HEADER = struct.Struct("<4sHHQQQ")  # magic, version, flags, original_size, 2x reserved
HEADER_SIZE = 32
assert HEADER.size == HEADER_SIZE

FOOTER = struct.Struct("<QQQQ8s")  # table off/len, manifest off/len, tail
FOOTER_SIZE = 40
assert FOOTER.size == FOOTER_SIZE

# off, dst, clen, rlen, crc, codec, esize, plane flags
RECORD = struct.Struct("<QQIIIBBH")
RECORD_SIZE = 32
assert RECORD.size == RECORD_SIZE

CODEC_STORED = 0  # payload is the raw bytes
CODEC_ZSTD = 1  # payload is one entropy-coded stream
CODEC_SPLIT = 2  # per-plane lengths, then per-plane data, split on byte bounds
CODEC_BF16 = 3  # as CODEC_SPLIT, but split on bfloat16's own field bounds
CODEC_REF = 4  # payload is a u64 offset: bytes equal an earlier output range
CODEC_BF16C = 5  # BF16 field split; sign+mantissa coded per exponent bucket
CODEC_BLK = 6  # v3 Q8_0 block split; still decoded, no longer written
CODEC_GBLK = 7  # block split whose field grouping is described in the payload
CODEC_DELTA = 8  # payload names an earlier output range plus a coded difference
# As CODEC_SPLIT, but every rANS plane is headerless and is decoded against
# the archive's shared table for that plane kind. A separate codec id rather
# than a flag bit because a split chunk's 16 flag bits are already two per
# plane for eight planes, with nothing spare -- and because a reader that does
# not know about shared tables must refuse the chunk rather than misread it,
# which an unknown codec id gets for free.
CODEC_SPLIT_ST = 9
CODEC_MIN_VERSION[CODEC_SPLIT_ST] = 7

# Short names for the codec ids, for anything that reports what an archive
# holds. Here rather than in a caller so that adding a codec updates every
# report at once -- the previous inline copy silently printed "?" for
# CODEC_SPLIT_ST from the day it was added.
CODEC_NAMES = {
    CODEC_STORED: "stored",
    CODEC_ZSTD: "entropy",
    CODEC_SPLIT: "split",
    CODEC_BF16: "bf16-split",
    CODEC_REF: "ref",
    CODEC_BF16C: "bf16-cond",
    CODEC_BLK: "q8-block",
    CODEC_GBLK: "blk-split",
    CODEC_DELTA: "delta",
    CODEC_SPLIT_ST: "split-shared",
}

# Which codecs a batch decoder can read, and which only the CPU can.
#
# lmz's GPU decoder takes a *batch of equal-length rANS streams* -- that is
# the shape that fills a device, since one stream is only 8 lanes of work. A
# codec whose payload is that shape can ride it; one whose payload is a single
# stream, a pointer, or segments of unequal length cannot, and no amount of
# effort on the decoder's side changes that. It is a property of the chunk
# layout, so it is declared here rather than inferred by a consumer from a
# chunk size or a dtype.
#
# The one that matters is CODEC_BF16C: it codes sign+mantissa per exponent
# bucket, so its segments have unequal lengths by construction. A consumer
# that wants an archive its GPU can read has to make lmz not emit it, and
# until this table existed the only way to know that was to reproduce lmz's
# own encoder threshold.
#
# `False` here does not mean slow or unsupported -- every codec decodes
# correctly on the CPU. It means the batch decoder cannot take this chunk.
BATCH_DECODABLE = {
    CODEC_STORED: False,   # not coded at all
    CODEC_ZSTD: False,     # one stream, whatever coder produced it
    CODEC_SPLIT: True,     # esize equal-length planes
    CODEC_BF16: True,      # two equal-length planes
    CODEC_REF: False,      # a pointer; decoding is a copy from elsewhere
    CODEC_BF16C: False,    # per-bucket segments, unequal by construction
    CODEC_BLK: True,       # equal-length stride lanes
    CODEC_GBLK: False,     # field groups of differing widths
    CODEC_DELTA: False,    # a source range plus a difference
    CODEC_SPLIT_ST: True,  # equal-length planes; table comes from the manifest
}

# Header flags. Both are advisory: the chunk table already addresses every
# payload by explicit (offset, length), so a reader that ignores these still
# decodes the archive correctly. They exist so a reader can tell whether
# random access will be cheap before it starts.
FLAG_PAGE_MAPPED = 1 << 0  # blocks are small enough to decode one at a time
FLAG_ALIGNED = 1 << 1  # payloads start on PAGE_ALIGN boundaries

# What a page-mapped archive aligns to, when it aligns at all. Storage below
# this is indivisible: filesystems allocate whole 4 KiB blocks and flash reads
# a page at a time, so a payload that starts mid-page makes the device fetch a
# page nobody asked for.
PAGE_ALIGN = 4096


class FormatError(ValueError):
    """Raised when an archive is malformed, truncated, or not an archive."""


class CorruptArchive(FormatError):
    """Raised when archive data fails to decode or does not match its checksum."""


@dataclass(slots=True)
class Chunk:
    off: int
    dst: int
    clen: int
    rlen: int
    crc: int
    codec: int
    esize: int
    flags: int

    def pack(self) -> bytes:
        return RECORD.pack(self.off, self.dst, self.clen, self.rlen, self.crc,
                           self.codec, self.esize, self.flags)


class ChunkTable:
    """The chunk table, read in place instead of expanded into objects.

    The table is fixed-width records, so the decompressed bytes *are* the
    index and a `Chunk` can be built from any record on demand. Building all
    of them up front is what used to dominate opening an archive: for a 70B
    checkpoint the table is 70 MB of records that become 0.51 GB of Python
    objects and take about 7 s, and the zstd decode of those 70 MB is only
    0.07 s of it. Every process paid that before reading a single byte.

    This holds the flat buffer and unpacks per access, which is a trade rather
    than a free win: one record costs a few microseconds, so code that walks
    the whole table repeatedly should hoist it into a list. The aggregate
    columns below exist so the callers that only want a total do not have to.

    A sequence, so anything that indexes, iterates, sorts or len()s a list of
    chunks keeps working unchanged.
    """

    __slots__ = ("_buf", "_n")

    def __init__(self, buf):
        if len(buf) % RECORD_SIZE:
            raise FormatError("chunk table has a partial record")
        self._buf = buf
        self._n = len(buf) // RECORD_SIZE

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, i):
        if isinstance(i, slice):
            return [self[k] for k in range(*i.indices(self._n))]
        if i < 0:
            i += self._n
        if not 0 <= i < self._n:
            raise IndexError("chunk index out of range")
        return Chunk(*RECORD.unpack_from(self._buf, i * RECORD_SIZE))

    def __iter__(self):
        for rec in RECORD.iter_unpack(self._buf):
            yield Chunk(*rec)

    def column(self, name: str) -> "array.array":
        """One field of every record, without building any Chunk.

        `sum(c.clen for c in chunks)` over a 70B table builds 2.2M objects to
        add one integer. Reading the column straight out of the buffer is
        about a hundred times less work and is what `info` and the coverage
        check want.
        """
        width, start = _COLUMNS[name]
        code = _TYPECODE.get(width)
        if code is None:                       # no native type of that width
            return array.array("Q", [getattr(c, name) for c in self])
        col = array.array(code)
        col.frombytes(self._buf)
        if sys.byteorder != "little":          # records are little-endian
            col.byteswap()
        stride = RECORD_SIZE // width
        return col[start::stride]

    def order_by_dst(self) -> list["Chunk"]:
        """Every chunk, in destination order.

        Chunks are appended as their workers finish, so the table is *nearly*
        sorted but not reliably so, and the callers that need an order cannot
        assume one.

        This deliberately materialises in one `iter_unpack` pass and sorts the
        objects, rather than sorting the packed column and then picking
        records out by index: a per-record `unpack_from` loop measured almost
        twice the cost of the single pass, which more than spends whatever the
        cheaper sort saves. Sorting a column is the right instinct and the
        wrong trade here, so it is written down rather than re-attempted.
        """
        out = list(self)
        # islice rather than dst[1:], which would copy the whole column just
        # to look at it once.
        dst = self.column("dst")
        if all(a <= b for a, b in zip(dst, islice(dst, 1, None))):
            return out                  # already ordered: skip the sort
        out.sort(key=_dst_of)
        return out


def _dst_of(c: "Chunk") -> int:
    """Sort key for chunk order. A module-level function rather than a lambda
    so the hot sort does not rebuild a closure per call site."""
    return c.dst


# field -> (width in bytes, index of the field's first element within a record
# measured in units of that width). RECORD is "<QQIIIBBH": two u64 then three
# u32 then two u8 and a u16, in 32 bytes.
_COLUMNS = {
    "off": (8, 0),
    "dst": (8, 1),
    "clen": (4, 4),
    "rlen": (4, 5),
    "crc": (4, 6),
}

# array's typecode widths are platform-defined rather than fixed, so they are
# discovered instead of assumed; a build where nothing is 4 or 8 bytes falls
# back to unpacking, which is slower but still correct.
_TYPECODE = {array.array(c).itemsize: c for c in ("I", "L", "Q")
             if array.array(c).itemsize in (4, 8)}


@dataclass(slots=True)
class Member:
    """One file inside the archive."""

    path: str
    size: int
    dst: int  # base offset in the virtual concatenated space
    kind: str = "raw"  # safetensors | gguf | raw
    mode: int = 0o644
    crc: int = 0
    tensors: dict | None = field(default=None)

    def to_json(self) -> dict:
        d = {"path": self.path, "size": self.size, "dst": self.dst,
             "kind": self.kind, "mode": self.mode, "crc": self.crc}
        if self.tensors:
            d["tensors"] = self.tensors
        return d

    @staticmethod
    def from_json(d: dict) -> "Member":
        return Member(path=d["path"], size=d["size"], dst=d["dst"],
                      kind=d.get("kind", "raw"), mode=d.get("mode", 0o644),
                      crc=d.get("crc", 0), tensors=d.get("tensors"))


def _zstd_compress(data: bytes, level: int = 3) -> bytes:
    from .entropy import compress

    return compress(data, level)


def _zstd_decompress(data: bytes) -> bytes:
    from .entropy import decompress

    return decompress(data)


class ArchiveWriter:
    """Streams chunk payloads out, then writes the table, manifest and footer."""

    def __init__(self, fileobj, manifest: dict, flags: int = 0, align: int = 0,
                 resume_at: int | None = None, chunks: list | None = None):
        """Start a new archive, or carry on writing an existing one.

        `resume_at` is where the old chunk table began: everything from there
        on -- table, manifest, footer -- is rebuilt, so appending costs the
        tail of the file rather than the whole of it. The header is already
        present in that case and is rewritten in place by close().
        """
        self.f = fileobj
        self.manifest = manifest
        self.chunks: list[Chunk] = list(chunks or ())
        self.version = max((CODEC_MIN_VERSION.get(c.codec, BASE_VERSION)
                            for c in self.chunks), default=BASE_VERSION)
        self.flags = flags
        self.align = align
        self.padding = 0
        if resume_at is None:
            self.offset = HEADER_SIZE
            self.f.write(b"\0" * HEADER_SIZE)  # rewritten by close()
        else:
            self.offset = resume_at
            self.f.seek(resume_at)

    def append(self, parts, dst: int, rlen: int, crc: int,
               codec: int, esize: int, flags: int) -> None:
        """Append one chunk. `parts` is a sequence of buffers written in order."""
        if self.align:
            # Pad ahead of the payload, not after it, so the recorded offset is
            # the aligned one. The gap is addressed by nothing and skipped by
            # every reader, which is what keeps this backward compatible.
            pad = -self.offset % self.align
            if pad:
                self.f.write(b"\0" * pad)
                self.offset += pad
                self.padding += pad
        clen = 0
        for part in parts:
            self.f.write(part)
            clen += len(part)
        self.chunks.append(Chunk(self.offset, dst, clen, rlen, crc,
                                 codec, esize, flags))
        need = CODEC_MIN_VERSION.get(codec)
        if need is not None and need > self.version:
            self.version = need
        self.offset += clen

    def close(self, original_size: int) -> None:
        table = b"".join(c.pack() for c in self.chunks)
        table_c = _zstd_compress(table, 3)
        table_off = self.offset
        self.f.write(table_c)
        self.offset += len(table_c)

        self.manifest["chunks"] = len(self.chunks)
        self.manifest["table_raw_len"] = len(table)
        man = json.dumps(self.manifest, separators=(",", ":")).encode()
        man_c = _zstd_compress(man, 3)
        man_off = self.offset
        self.f.write(man_c)
        self.offset += len(man_c)
        self.f.write(FOOTER.pack(table_off, len(table_c), man_off, len(man_c), TAIL))

        # Where the archive really ends, recorded before the seek below moves
        # the file position back to the header. An append truncates to this.
        self.end = self.offset + FOOTER_SIZE

        self.f.seek(0)
        self.f.write(HEADER.pack(MAGIC, self.version, self.flags,
                                 original_size, 0, 0))
        self.f.flush()


class ArchiveReader:
    """Reads the header, manifest and chunk table. Payloads are read on demand."""

    def __init__(self, fileobj):
        self.f = fileobj
        self.f.seek(0, io.SEEK_END)
        self.file_size = self.f.tell()
        if self.file_size < HEADER_SIZE + FOOTER_SIZE:
            raise FormatError("file is too small to be an lmz archive")

        self.f.seek(0)
        magic, version, self.flags, self.original_size, _, _ = HEADER.unpack(
            self.f.read(HEADER_SIZE))
        if magic != MAGIC:
            raise FormatError("not an lmz archive (bad magic)")
        if version not in READABLE_VERSIONS:
            raise FormatError(
                f"archive format v{version}, this build understands "
                f"v{' and v'.join(str(v) for v in READABLE_VERSIONS)}")

        self.f.seek(self.file_size - FOOTER_SIZE)
        table_off, table_clen, man_off, man_clen, tail = FOOTER.unpack(
            self.f.read(FOOTER_SIZE))
        if tail != TAIL:
            raise FormatError("archive is truncated or corrupt (bad footer)")
        for off, ln, what in ((table_off, table_clen, "chunk table"),
                              (man_off, man_clen, "manifest")):
            if off < HEADER_SIZE or off + ln > self.file_size:
                raise FormatError(f"{what} points outside the file")

        # Where the payloads stop and the tail begins. An append rewrites
        # everything from here on, so it must come from the footer rather than
        # be inferred from the chunk records.
        self.table_off = table_off

        self.f.seek(man_off)
        self.manifest = json.loads(_zstd_decompress(self.f.read(man_clen)))
        self.f.seek(table_off)
        table = _zstd_decompress(self.f.read(table_clen))
        self.chunks = ChunkTable(table)

        self.members = [Member.from_json(m) for m in self.manifest.get("members", [])]

    @property
    def payload_end(self) -> int:
        """First byte after the chunk payloads."""
        off = self.chunks.column("off")
        return min(off) if off else self.file_size

    @property
    def page_mapped(self) -> bool:
        """Whether blocks are small enough that random access is cheap."""
        return bool(self.flags & FLAG_PAGE_MAPPED)

    @property
    def aligned(self) -> bool:
        return bool(self.flags & FLAG_ALIGNED)
