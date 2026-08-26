"""Command line interface."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import __version__, api, entropy
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
                         mapped=args.mapped, align=args.align, progress=bar,
                         shared_tables=args.shared_tables)
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
        # An interpreter with no zstd binding silently codes every general
        # chunk with deflate, which costs several points and looks like
        # nothing at all -- until `doctor` is run, which nobody does after a
        # ratio they were not expecting. Say it where the ratio is printed.
        if not entropy.HAVE_ZSTD:
            print("  note: no zstd backend, so general chunks used deflate "
                  "and this archive is larger than lmz would normally make "
                  "it; `lmz doctor` says what is available")
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
    # The codec above is the framing; this is the coder that did the work. A
    # bf16-split chunk says nothing about whether rANS or zstd earned its bytes,
    # and that is the question worth asking of a hand-written entropy coder.
    if data.get("methods"):
        total_raw = sum(m["raw"] for m in data["methods"].values())
        print("entropy coders")
        for name, m in sorted(data["methods"].items(),
                              key=lambda kv: -kv[1]["raw"]):
            r = m["raw"] / m["coded"] if m["coded"] else 0
            share = m["raw"] / total_raw * 100 if total_raw else 0
            print(f"  {name:10s} {m['streams']:7d} streams {human(m['raw']):>10s} -> "
                  f"{human(m['coded']):>10s}  {r:.3f}x  {share:5.1f}% of input")
            if m.get("contested"):
                alt = m["contested_alt"]
                by = (1 - m["contested_coded"] / alt) * 100 if alt else 0
                print(f"  {'':10s} {m['contested']:7d} of those ran against the "
                      f"other coder and won by {by:.1f}%")
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


def cmd_append(args) -> int:
    bar = Progress("appending", not args.quiet)
    stats = api.append(args.archive, args.input, level=args.level,
                       workers=args.threads, checksum=not args.no_checksum,
                       delta=not args.no_delta, progress=bar)
    bar.done()
    if not args.quiet:
        print(f"{args.input} -> {args.archive}")
        print(f"  added {human(stats.input_bytes)}, archive now "
              f"{human(stats.output_bytes)}  {stats.seconds:.2f}s")
        d = stats.detail.get("delta_bytes", 0)
        if d:
            print(f"  {human(d)} coded as differences from what was already there")
    return 0


def cmd_extract(args) -> int:
    stats = api.extract(args.archive, args.member, args.output,
                        overwrite=args.force, workers=args.threads)
    if not args.quiet:
        print(f"{args.member} -> {args.output}  {human(stats.input_bytes)}  "
              f"{stats.seconds:.2f}s")
    return 0


def cmd_doctor(args) -> int:
    from . import fuse, gpu

    if getattr(args, "gpu_verify", False):
        return _gpu_verify(gpu)

    b = api.backends()
    print(f"lmz {__version__}")
    print(f"  python    {sys.version.split()[0]}")
    print(f"  kernel    {b['kernel']}")
    print(f"  entropy   {b['entropy']}")
    print(f"  threads   {b['workers']} (detected)")
    ok, why = fuse.available()
    print(f"  mount     {'available' if ok else 'unavailable -- ' + why}")
    # Unlike backends(), doctor is the place that is *supposed* to go and look:
    # this builds the CUDA decoder if nvcc and a device are both present.
    gok, gwhy = gpu.available()
    print(f"  gpu       {gpu.backend() if gok else 'unavailable -- ' + gwhy}")
    from .store import Store
    store = Store()
    print(f"  store     {store.root}"
          f"{'' if os.path.isdir(store.root) else ' (not created yet)'}")
    if not b["kernel"].startswith("native"):
        print("  note: native kernel unavailable; using a slower fallback.")
        print("        a C compiler (cc/gcc/clang) enables the SIMD path.")
    if not gok:
        print("  note: the GPU decoder is optional and nothing needs it. It")
        print("        decodes batches of lmz streams straight into VRAM; the")
        print("        CPU path is unaffected either way.")
    return 0


def _gpu_verify(gpu) -> int:
    """Prove the GPU decoder on this machine, in a form worth pasting.

    The kernel has been run on one architecture. Turing in particular gets
    different generated code -- cp.async has no instruction there and the
    intrinsic becomes a synchronous copy -- so a report from a card that is
    not a Blackwell is worth more than any amount of further testing here.
    """
    r = gpu.verify()
    print(f"lmz {r['lmz']} GPU verification")
    if r["device"] is None:
        print(f"  no GPU decoder -- {r['why']}")
        return 1
    print(f"  device   {r['device']}")
    print(f"  shapes   {r['checked']} decoded byte-identically to the CPU decoder"
          if not r["failures"] else
          f"  shapes   {r['checked']} checked, {len(r['failures'])} FAILED")
    for f in r["failures"]:
        print(f"    FAIL   {f}")
    if r["gbps"] is not None:
        print(f"  batch    {r['gbps']} GB/s decoding a 67 MB batch, host round "
              f"trip.")
        print("           That includes PCIe in and out and is not the kernel's "
              "own rate;")
        print("           see docs/gpu-residency-handover.md for the resident "
              "numbers.")
    print(f"  verdict  {'OK' if r['ok'] else 'MISMATCH'}")
    if not r["ok"]:
        print("\n  Please open an issue with this block:")
        print("  https://github.com/FanxinSun/lmz/issues")
    return 0 if r["ok"] else 1


# ------------------------------------------------------------------- the store


def _store(args):
    from .store import Store

    return Store(getattr(args, "store", None))


def cmd_add(args) -> int:
    store = _store(args)
    bar = Progress("compressing", not args.quiet)
    entry = store.add(args.input, args.name, level=args.level,
                      workers=args.threads, force=args.force, progress=bar)
    bar.done()
    if not args.quiet:
        print(f"{args.input} -> {store.archive_path(entry)}")
        print(f"  {entry.name}: {human(entry.original_size)} -> "
              f"{human(entry.stored_size)}  ({entry.saved * 100:.1f}% smaller, "
              f"{entry.files} file(s))")
        print(f"  read it without expanding: lmz mount <dir>")
    return 0


def cmd_models(args) -> int:
    store = _store(args)
    entries = store.models()
    if args.json:
        print(json.dumps([e.to_json() for e in entries], indent=2))
        return 0
    if not entries:
        print(f"no models in {store.root}\n  add one:  lmz add ./my-model")
        return 0
    print(f"{'NAME':<28s} {'ON DISK':>10s} {'RESTORES TO':>12s} {'SAVED':>7s}  FILES")
    for e in entries:
        print(f"{e.name:<28s} {human(e.stored_size):>10s} "
              f"{human(e.original_size):>12s} {e.saved * 100:>6.1f}%  {e.files}")
    original, stored = store.totals()
    print(f"{'':<28s} {human(stored):>10s} {human(original):>12s} "
          f"{(1 - stored / original) * 100 if original else 0:>6.1f}%")
    return 0


def cmd_rm(args) -> int:
    store = _store(args)
    entry = store.remove(args.name)
    if not args.quiet:
        print(f"removed {entry.name} ({human(entry.stored_size)} freed)")
    return 0


def cmd_mount(args) -> int:
    from . import fuse
    from .store import mount

    ok, why = fuse.available()
    if not ok:
        print(f"lmz: cannot mount: {why}", file=sys.stderr)
        return 1
    os.makedirs(args.point, exist_ok=True)
    if os.listdir(args.point):
        print(f"lmz: {args.point} is not empty", file=sys.stderr)
        return 1

    server = mount(args.point, _store(args), names=args.model or None,
                   threads=args.threads, allow_other=args.allow_other,
                   cache_blocks=args.cache_blocks, verify=args.verify,
                   readahead=args.readahead)
    if not server.fs.entries:
        print("lmz: the store is empty; add a model first with `lmz add`",
              file=sys.stderr)
        return 1
    try:
        server.mount()
    except OSError as exc:
        print(f"lmz: mount failed: {exc.strerror or exc}", file=sys.stderr)
        return 1

    if args.daemon:
        # The mount is already live, so the fork below cannot lose a request:
        # the kernel queues them on the descriptor until a thread reads it.
        if os.fork():
            print(f"{args.point}: serving {len(server.fs.entries)} model(s) "
                  f"in the background")
            print(f"  stop with: lmz unmount {args.point}")
            # _exit skips the atexit and stdio teardown that would otherwise
            # run twice over a forked pair of processes, so the flush is ours.
            sys.stdout.flush()
            os._exit(0)
        os.setsid()
        devnull = os.open(os.devnull, os.O_RDWR)
        for fd in (0, 1, 2):
            os.dup2(devnull, fd)
    elif not args.quiet:
        total = sum(e.original_size for e in server.fs.entries.values())
        stored = sum(e.stored_size for e in server.fs.entries.values())
        print(f"{args.point}: {len(server.fs.entries)} model(s), "
              f"{human(total)} of files served from {human(stored)} on disk")
        for name in sorted(server.fs.entries):
            print(f"  {args.point.rstrip('/')}/{name}/")
        print("press ctrl-c to unmount")
    s = {}
    try:
        server.serve()
    except KeyboardInterrupt:
        pass
    finally:
        # Read before close(), which drops the readers the counters live on.
        s = server.fs.stats()
        server.close()
        fuse.unmount(args.point)
    if not args.quiet and not args.daemon:
        print(f"\nunmounted: {server.requests} requests, "
              f"{human(server.bytes_served)} served, "
              f"{human(s['decoded_bytes'])} decoded in "
              f"{s['blocks_decoded']} blocks, {s['cache_hits']} cache hits"
              + (f", readahead {s['readahead_issued']} issued / "
                 f"{s['readahead_skipped']} skipped"
                 if 'readahead_issued' in s else ""))
    return 0


def cmd_fs(args) -> int:
    from . import fuse
    from .lmzfs import LmzFS

    ok, why = fuse.available()
    if not ok:
        print(f"lmz: cannot mount: {why}", file=sys.stderr)
        return 1
    os.makedirs(args.point, exist_ok=True)
    if os.listdir(args.point):
        print(f"lmz: {args.point} is not empty", file=sys.stderr)
        return 1

    server = LmzFS(args.backing, args.point, threads=args.threads,
                   allow_other=args.allow_other, level=args.level,
                   block_size=args.block_size)
    try:
        server.mount()
    except OSError as exc:
        print(f"lmz: mount failed: {exc.strerror or exc}", file=sys.stderr)
        return 1

    if args.daemon:
        if os.fork():
            print(f"{args.backing} -> {args.point}  (compressing, background)")
            print(f"  stop with: lmz unmount {args.point}")
            sys.stdout.flush()
            os._exit(0)
        os.setsid()
        devnull = os.open(os.devnull, os.O_RDWR)
        for fd in (0, 1, 2):
            os.dup2(devnull, fd)
    elif not args.quiet:
        print(f"{args.backing} -> {args.point}")
        print("  files written here are compressed; reads decode transparently")
        print("  press ctrl-c to unmount")

    s = {}
    try:
        server.serve()
    except KeyboardInterrupt:
        pass
    finally:
        s = server.stats()
        server.close()
        fuse.unmount(args.point)
    if not args.quiet and not args.daemon and s.get("files_written"):
        print(f"\nunmounted: {s['files_written']} file(s) written, "
              f"{human(s['bytes_in'])} -> {human(s['bytes_out'])} "
              f"({s['saved'] * 100:.1f}% smaller, {s['stored_raw']} stored raw)")
    return 0


def cmd_unmount(args) -> int:
    from . import fuse

    fuse.unmount(args.point)
    if not args.quiet:
        print(f"unmounted {args.point}")
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

    if args.methods:
        # A second pass, deliberately untimed. `measure_alt` makes the plane
        # path also run the general-purpose coder it normally skips, which is
        # the only way to see what rANS is winning by -- the archive records
        # the winner's size and nothing about the loser. It roughly doubles the
        # coding work, which is why it stays out of the timings above.
        mt = _codec.MethodTally(measure_alt=True)
        for d, e, k, b in data:
            _codec.encode_chunk(d, e, 1, False, kind=k, btype=b, tally=mt)
        rows = mt.to_json()
        if rows:
            raw_total = sum(r["raw"] for r in rows.values())
            print(f"\n  {'entropy coder':<22s} {'streams':>8s} {'in':>10s} "
                  f"{'out':>10s} {'ratio':>8s} {'share':>7s}  vs the other coder")
            for name, r in sorted(rows.items(), key=lambda kv: -kv[1]["raw"]):
                ratio = r["raw"] / r["coded"] if r["coded"] else 0
                share = r["raw"] / raw_total * 100 if raw_total else 0
                if r.get("contested"):
                    alt = r["contested_alt"]
                    by = (1 - r["contested_coded"] / alt) * 100 if alt else 0
                    versus = f"{by:+.1f}% over {r['contested']} streams"
                else:
                    versus = "not contested"
                print(f"  {name:<22s} {r['streams']:>8d} {human(r['raw']):>10s} "
                      f"{human(r['coded']):>10s} {ratio:>7.3f}x {share:>6.1f}%  "
                      f"{versus}")
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
    c.add_argument("--chunk-size", type=parse_size, default=None,
                   metavar="N",
                   help="chunk size (default: 8MiB, or 64KiB with --mapped)")
    c.add_argument("--no-checksum", action="store_true",
                   help="skip per-chunk crc32")
    c.add_argument("--no-dedup", action="store_true",
                   help="skip duplicate-tensor detection")
    c.add_argument("--no-delta", action="store_true",
                   help="skip delta coding against matching tensors in an "
                        "earlier file")
    c.add_argument("--mapped", action="store_true",
                   help="small blocks, so any byte range can be decoded on its "
                        "own (default 64KiB; costs ~1 point of ratio)")
    c.add_argument("--align", action="store_true",
                   help="with --mapped, start every block on a 4KiB boundary")
    c.add_argument("--shared-tables", action="store_true",
                   help="fit one rANS table per plane kind and carry it in the "
                        "manifest instead of in every stream; writes a v7 "
                        "archive older builds cannot read")
    c.add_argument("-f", "--force", action="store_true", help="overwrite output")
    common(c)
    c.set_defaults(func=cmd_compress)

    ap_ = sub.add_parser("append", help="add files to an existing archive")
    ap_.add_argument("archive")
    ap_.add_argument("input")
    ap_.add_argument("-l", "--level", type=int, default=api.DEFAULT_LEVEL)
    ap_.add_argument("--no-checksum", action="store_true")
    ap_.add_argument("--no-delta", action="store_true",
                     help="skip delta coding against what is already there")
    common(ap_)
    ap_.set_defaults(func=cmd_append)

    e = sub.add_parser("extract", help="write one member out on its own")
    e.add_argument("archive")
    e.add_argument("member")
    e.add_argument("output")
    e.add_argument("-f", "--force", action="store_true", help="overwrite output")
    common(e)
    e.set_defaults(func=cmd_extract)

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
    b.add_argument("--methods", action="store_true",
                   help="break the sample down by entropy coder, and code every "
                        "plane both ways to show what the winner won by")
    b.set_defaults(func=cmd_bench)

    doc = sub.add_parser("doctor", help="report the active backends")
    doc.add_argument("--gpu-verify", action="store_true",
                     help="decode every awkward shape on this GPU and check "
                          "the CPU decoder agrees")
    doc.set_defaults(func=cmd_doctor)

    # -- the store ---------------------------------------------------------
    def with_store(sp):
        sp.add_argument("--store", metavar="DIR",
                        help="store root (default: $LMZ_HOME, else ~/.lmz)")
        return sp

    a = with_store(sub.add_parser(
        "add", help="compress a model into the store, ready to be mounted"))
    a.add_argument("input")
    a.add_argument("--name", help="name in the store (default: the basename)")
    a.add_argument("-l", "--level", type=int, default=api.DEFAULT_LEVEL)
    a.add_argument("-f", "--force", action="store_true",
                   help="replace a model of the same name")
    common(a)
    a.set_defaults(func=cmd_add)

    m = with_store(sub.add_parser("models", help="list models in the store"))
    m.add_argument("--json", action="store_true")
    m.set_defaults(func=cmd_models)

    r = with_store(sub.add_parser("rm", help="remove a model from the store"))
    r.add_argument("name")
    common(r, threads=False)
    r.set_defaults(func=cmd_rm)

    mo = with_store(sub.add_parser(
        "mount", help="serve the store as ordinary model files"))
    mo.add_argument("point", metavar="mountpoint")
    mo.add_argument("--model", action="append", metavar="NAME",
                    help="serve only this model (repeatable)")
    mo.add_argument("-d", "--daemon", action="store_true",
                    help="detach and serve in the background")
    mo.add_argument("--allow-other", action="store_true",
                    help="let other users read the mount (needs user_allow_other)")
    mo.add_argument("--cache-blocks", type=int, default=512, metavar="N",
                    help="decoded blocks held per model (default: 512, 32MiB)")
    mo.add_argument("--readahead", type=int, default=None, metavar="N",
                    help="threads decoding ahead of a sequential reader "
                         "(0 disables; default: 1 under the GIL, more without)")
    mo.add_argument("--verify", action="store_true",
                    help="check every block's crc32 as it is read")
    mo.add_argument("-j", "--threads", type=int, default=None, metavar="N",
                    help="server threads (default: 2 under the GIL, where "
                         "more measure slower; the whole machine without it)")
    mo.add_argument("-q", "--quiet", action="store_true")
    mo.set_defaults(func=cmd_mount)

    fs = sub.add_parser(
        "fs", help="mount a compressed read-write filesystem over a directory")
    fs.add_argument("backing", help="directory holding the compressed form")
    fs.add_argument("point", metavar="mountpoint")
    fs.add_argument("-d", "--daemon", action="store_true",
                    help="detach and serve in the background")
    fs.add_argument("--allow-other", action="store_true")
    fs.add_argument("-l", "--level", type=int, default=api.DEFAULT_LEVEL)
    fs.add_argument("--block-size", type=parse_size, default=64 << 10,
                    metavar="N", help="block size for stored files "
                                      "(default: 64KiB)")
    fs.add_argument("-j", "--threads", type=int, default=None, metavar="N")
    fs.add_argument("-q", "--quiet", action="store_true")
    fs.set_defaults(func=cmd_fs)

    um = sub.add_parser("unmount", aliases=["umount"], help="detach a mount")
    um.add_argument("point", metavar="mountpoint")
    common(um, threads=False)
    um.set_defaults(func=cmd_unmount)
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
