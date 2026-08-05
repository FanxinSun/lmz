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

import io
import json
import struct
from dataclasses import dataclass, field

MAGIC = b"LMZ\x01"
TAIL = b"LMZTAIL\x01"
# v2 added ref chunks (cross-file tensor dedup) and conditioned BF16 chunks;
# v3 adds block-split chunks for GGUF Q8_0. Earlier archives contain none of
# the newer codecs, so this build reads every version listed.
FORMAT_VERSION = 3
READABLE_VERSIONS = (1, 2, 3)

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
CODEC_BLK = 6  # fixed-period block split: scale planes apart from quant bytes


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

    def __init__(self, fileobj, manifest: dict):
        self.f = fileobj
        self.manifest = manifest
        self.chunks: list[Chunk] = []
        self.offset = HEADER_SIZE
        self.f.write(b"\0" * HEADER_SIZE)  # rewritten by close()

    def append(self, parts, dst: int, rlen: int, crc: int,
               codec: int, esize: int, flags: int) -> None:
        """Append one chunk. `parts` is a sequence of buffers written in order."""
        clen = 0
        for part in parts:
            self.f.write(part)
            clen += len(part)
        self.chunks.append(Chunk(self.offset, dst, clen, rlen, crc,
                                 codec, esize, flags))
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

        self.f.seek(0)
        self.f.write(HEADER.pack(MAGIC, FORMAT_VERSION, 0, original_size, 0, 0))
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

        self.f.seek(man_off)
        self.manifest = json.loads(_zstd_decompress(self.f.read(man_clen)))
        self.f.seek(table_off)
        table = _zstd_decompress(self.f.read(table_clen))
        if len(table) % RECORD_SIZE:
            raise FormatError("chunk table has a partial record")
        self.chunks = [Chunk(*rec) for rec in RECORD.iter_unpack(table)]

        self.members = [Member.from_json(m) for m in self.manifest.get("members", [])]

    @property
    def payload_end(self) -> int:
        """First byte after the chunk payloads."""
        return min((c.off for c in self.chunks), default=self.file_size)
