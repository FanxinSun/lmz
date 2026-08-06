"""Command line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import __version__, api
from .format import FormatError
from .parallel import default_workers


def human(n: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} TiB"


def rate(bytes_per_s: float) -> str:
    return f"{bytes_per_s / (1 << 20):.0f} MiB/s"


def parse_size(text: str) -> int:
    units = {"k": 1 << 10, "kb": 1 << 10, "kib": 1 << 10,
             "m": 1 << 20, "mb": 1 << 20, "mib": 1 << 20,
             "g": 1 << 30, "gb": 1 << 30, "gib": 1 << 30}
    s = text.strip().lower()
    for suffix, mult in sorted(units.items(), key=lambda kv: -len(kv[0])):
        if s.endswith(suffix):
            return int(float(s[:-len(suffix)]) * mult)
    return int(s)


class Progress:
    """A single rewriting status line; silent when stderr is not a terminal."""

    def __init__(self, label: str, enabled: bool = True):
        self.label = label
        self.enabled = enabled and sys.stderr.isatty()
        self.start = time.perf_counter()
        self.last = 0.0
        self.width = 0

    def __call__(self, done: int, total: int) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        if now - self.last < 0.05 and done < total:
            return
        self.last = now
        elapsed = max(now - self.start, 1e-9)
        frac = done / total if total else 1.0
        filled = int(frac * 24)
        bar = "#" * filled + "-" * (24 - filled)
        line = (f"\r{self.label} [{bar}] {frac * 100:5.1f}%  "
                f"{human(done)} / {human(total)}  {rate(done / elapsed)}")
        self.width = max(self.width, len(line))
        sys.stderr.write(line.ljust(self.width))
        sys.stderr.flush()

    def done(self) -> None:
        if self.enabled and self.width:
            sys.stderr.write("\r" + " " * self.width + "\r")
            sys.stderr.flush()


def cmd_compress(args) -> int:
    src = args.input
    dst = args.output or (src.rstrip("/\\") + ".lmz")
    if os.path.exists(dst) and not args.force:
        print(f"lmz: {dst} exists (use -f to overwrite)", file=sys.stderr)
        return 1
    bar = Progress("compressing", not args.quiet)
    stats = api.compress(src, dst, level=args.level, workers=args.threads,
                         chunk_size=args.chunk_size, checksum=not args.no_checksum,
                         dedup=not args.no_dedup, delta=not args.no_delta,
                         progress=bar)
    bar.done()
    if not args.quiet:
        print(f"{src} -> {dst}")
        print(f"  {human(stats.input_bytes)} -> {human(stats.output_bytes)}  "
              f"{stats.ratio:.3f}x  ({stats.saved * 100:.1f}% smaller)")
        print(f"  {stats.seconds:.2f}s  {rate(stats.throughput)}  "
              f"{stats.files} file(s), {stats.chunks} chunks")
        deduped = stats.detail.get("dedup_bytes", 0)
        if deduped:
            print(f"  {human(deduped)} of duplicate tensors stored once")
        delta = stats.detail.get("delta_bytes", 0)
        if delta:
            print(f"  {human(delta)} coded as differences from an earlier file")
    return 0


def cmd_decompress(args) -> int:
    src = args.input
    dst = args.output
    if dst is None:
        dst = src[:-4] if src.endswith(".lmz") else src + ".out"
    bar = Progress("decompressing", not args.quiet)
    stats = api.decompress(src, dst, workers=args.threads,
                           verify_checksums=not args.no_verify,
                           progress=bar, overwrite=args.force)
    bar.done()
    if not args.quiet:
        print(f"{src} -> {dst}")
        print(f"  {human(stats.output_bytes)} -> {human(stats.input_bytes)}  "
              f"{stats.seconds:.2f}s  {rate(stats.throughput)}")
    return 0


def cmd_verify(args) -> int:
    bar = Progress("verifying", not args.quiet)
    stats = api.verify(args.input, workers=args.threads, progress=bar)
    bar.done()
    print(f"{args.input}: OK  ({stats.chunks} chunks, {human(stats.input_bytes)} "
          f"original, {stats.seconds:.2f}s, {rate(stats.throughput)})")
    return 0


def cmd_info(args) -> int:
    data = api.info(args.input)
    if args.json:
        print(json.dumps(data, indent=2))
        return 0

    man = data["manifest"]
    orig = data["original_size"]
    size = data["archive_size"]
    print(f"archive     {data['path']}")
    print(f"created by  {man.get('tool', '?')}   entropy: {man.get('entropy', '?')}")
    print(f"original    {human(orig)}")
    print(f"compressed  {human(size)}   {orig / size if size else 0:.3f}x  "
          f"({(1 - size / orig) * 100 if orig else 0:.1f}% smaller)")
    print(f"chunks      {data['chunks']} of {human(man.get('chunk_size', 0))}  "
          f"level {man.get('level', '?')}  checksum {man.get('checksum', '?')}")
    if man.get("dedup_bytes"):
        print(f"dedup       {human(man['dedup_bytes'])} of duplicate tensors stored once")
    if data["codecs"]:
        print("chunk codecs")
        for name, (count, rlen, clen) in sorted(data["codecs"].items()):
            r = rlen / clen if clen else 0
            print(f"  {name:10s} {count:7d} chunks  {human(rlen):>10s} -> "
                  f"{human(clen):>10s}  {r:.3f}x")
    print(f"members     {len(data['members'])}")
    for m in data["members"][:args.limit]:
        ntensors = len(m.get("tensors") or {})
        extra = f", {ntensors} tensors" if ntensors else ""
        print(f"  {m['path']}  {human(m['size'])}  [{m['kind']}{extra}]")
    if len(data["members"]) > args.limit:
        print(f"  ... {len(data['members']) - args.limit} more")

    if args.tensors:
        print("tensors")
        shown = 0
        for m in data["members"]:
            for name, meta in (m.get("tensors") or {}).items():
                if shown >= args.limit:
                    break
                lo, hi = meta["offsets"]
                print(f"  {name:<52s} {meta['dtype']:<8s} "
                      f"{str(meta['shape']):<20s} {human(hi - lo)}")
                shown += 1
    return 0


def cmd_cat(args) -> int:
    dtype, shape, raw = api.read_tensor(args.input, args.tensor, args.member)
    if args.output:
        with open(args.output, "wb") as fh:
            fh.write(raw)
        print(f"{args.tensor}: {dtype} {shape}  {human(len(raw))} -> {args.output}",
              file=sys.stderr)
    else:
        sys.stdout.buffer.write(raw)
    return 0


def cmd_doctor(args) -> int:
    b = api.backends()
    print(f"lmz {__version__}")
    print(f"  python    {sys.version.split()[0]}")
    print(f"  kernel    {b['kernel']}")
    print(f"  entropy   {b['entropy']}")
    print(f"  threads   {b['workers']} (detected)")
    if not b["kernel"].startswith("native"):
        print("  note: native kernel unavailable; using a slower fallback.")
        print("        a C compiler (cc/gcc/clang) enables the SIMD path.")
    return 0


def cmd_bench(args) -> int:
    """Compare the dtype-aware codec against plain general-purpose compression."""
    import zlib

    from . import codec as _codec
    from . import entropy as _entropy
    from .planner import chunkify, probe

    path = args.input
    size = os.path.getsize(path)
    limit = args.bytes or (256 << 20)
    with open(path, "rb") as fh:
        layout = probe(fh, size)
        data = []
        taken = 0
        for start, end, esize, kind, _src, btype, _ikind in chunkify(
                layout, size, args.chunk_size):
            if taken >= limit:
                break
            fh.seek(start)
            data.append((fh.read(end - start), esize, kind, btype))
            taken += end - start

    total = sum(len(d) for d, _, _, _ in data)
    print(f"{path}: {layout.kind}, sampling {human(total)} in {len(data)} chunks\n")
    print(f"  {'codec':<22s} {'ratio':>8s} {'saved':>8s} {'compress':>12s} "
          f"{'decompress':>12s}")

    def fmt(bps: float) -> str:
        mib = bps / (1 << 20)
        # Data left uncompressed decodes in no measurable time; a six-digit
        # rate there would say more about the clock than the codec.
        return "  (no work)" if mib > 100000 else f"{mib:.0f} MiB/s"

    def report(name, enc, dec, note_fn=None):
        """enc returns (encoded_size, opaque); dec takes the opaque back."""
        t = time.perf_counter()
        encoded = [enc(d, e, k, b) for d, e, k, b in data]
        tc = time.perf_counter() - t
        out = sum(size for size, _ in encoded)
        t = time.perf_counter()
        for (_, obj), (d, e, k, b) in zip(encoded, data):
            dec(obj, d, e, k, b)
        td = time.perf_counter() - t
        note = note_fn() if note_fn else ""
        print(f"  {name:<22s} {total / out:>7.3f}x {(1 - out / total) * 100:>7.1f}% "
              f"{fmt(total / tc):>12s} {fmt(total / td):>12s}  {note}")

    def enc_lmz(d, e, k, b, lvl, tally):
        parts, cid, flags, _ = _codec.encode_chunk(d, e, lvl, False, kind=k, btype=b)
        payload = b"".join(bytes(p) for p in parts)
        tally[cid] = tally.get(cid, 0) + 1
        return len(payload), (payload, cid, flags)

    def dec_lmz(obj, d, e, k, b):
        payload, cid, flags = obj
        return _codec.decode_chunk(payload, cid, e, flags, len(d), 0, False)

    for lvl in (1, 3):
        tally: dict = {}
        report(f"lmz (level {lvl})",
               lambda d, e, k, b, lvl=lvl, t=tally: enc_lmz(d, e, k, b, lvl, t),
               dec_lmz,
               note_fn=lambda t=tally: ("stored unchanged: no compressible structure"
                                        if set(t) == {0} else ""))
    if _entropy.HAVE_ZSTD:
        for lvl in (1, 3):
            def enc_zstd(d, e, k, b, lvl=lvl):
                c = _entropy.compress(d, lvl, _entropy.METHOD_ZSTD)
                return len(c), c
            report(f"zstd -{lvl} (plain)", enc_zstd,
                   lambda obj, d, e, k, b: _entropy.decompress(
                       obj, _entropy.METHOD_ZSTD))

    def enc_gzip(d, e, k, b):
        c = zlib.compress(d, 6)
        return len(c), c

    report("gzip -6 (plain)", enc_gzip, lambda obj, d, e, k, b: zlib.decompress(obj))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lmz",
        description="Fast lossless compression for large model weights.")
    p.add_argument("--version", action="version", version=f"lmz {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, threads=True):
        if threads:
            sp.add_argument("-j", "--threads", type=int, default=None,
                            metavar="N",
                            help=f"worker threads (default: {default_workers()})")
        sp.add_argument("-q", "--quiet", action="store_true")

    c = sub.add_parser("compress", aliases=["c"], help="compress a file or directory")
    c.add_argument("input")
    c.add_argument("output", nargs="?", help="archive path (default: <input>.lmz)")
    c.add_argument("-l", "--level", type=int, default=api.DEFAULT_LEVEL,
                   help="entropy coder level (default: 1, the fastest and, on "
                        "weight data, usually the smallest too)")
    c.add_argument("--chunk-size", type=parse_size, default=api.DEFAULT_CHUNK_SIZE,
                   metavar="N", help="chunk size (default: 8MiB)")
    c.add_argument("--no-checksum", action="store_true",
                   help="skip per-chunk crc32")
    c.add_argument("--no-dedup", action="store_true",
                   help="skip duplicate-tensor detection")
    c.add_argument("--no-delta", action="store_true",
                   help="skip delta coding against matching tensors in an "
                        "earlier file")
    c.add_argument("-f", "--force", action="store_true", help="overwrite output")
    common(c)
    c.set_defaults(func=cmd_compress)

    d = sub.add_parser("decompress", aliases=["d", "x"], help="restore an archive")
    d.add_argument("input")
    d.add_argument("output", nargs="?", help="destination (default: input minus .lmz)")
    d.add_argument("--no-verify", action="store_true", help="skip checksum checks")
    d.add_argument("-f", "--force", action="store_true", help="overwrite existing files")
    common(d)
    d.set_defaults(func=cmd_decompress)

    v = sub.add_parser("verify", help="check an archive without writing output")
    v.add_argument("input")
    common(v)
    v.set_defaults(func=cmd_verify)

    i = sub.add_parser("info", aliases=["ls"], help="show archive contents")
    i.add_argument("input")
    i.add_argument("--tensors", action="store_true", help="list tensors")
    i.add_argument("--json", action="store_true", help="machine-readable output")
    i.add_argument("--limit", type=int, default=25, help="max rows to print")
    i.set_defaults(func=cmd_info)

    t = sub.add_parser("cat", help="extract one tensor without full decompression")
    t.add_argument("input")
    t.add_argument("tensor")
    t.add_argument("-o", "--output", help="write to a file instead of stdout")
    t.add_argument("--member", help="restrict to one file inside the archive")
    t.set_defaults(func=cmd_cat)

    b = sub.add_parser("bench", help="compare against general-purpose compressors")
    b.add_argument("input")
    b.add_argument("--bytes", type=parse_size, default=None,
                   help="how much to sample (default: 256MiB)")
    b.add_argument("--chunk-size", type=parse_size, default=api.DEFAULT_CHUNK_SIZE)
    b.set_defaults(func=cmd_bench)

    doc = sub.add_parser("doctor", help="report the active backends")
    doc.set_defaults(func=cmd_doctor)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FormatError, KeyError, ValueError) as exc:
        print(f"lmz: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        # Out of space, permissions, a vanished file: report the path.
        where = f" ({exc.filename})" if getattr(exc, "filename", None) else ""
        print(f"lmz: {exc.strerror or exc}{where}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nlmz: interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
