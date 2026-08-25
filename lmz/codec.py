"""Per-chunk encode and decode.

A chunk is split into byte planes and each plane is judged on its own. That
adaptivity is where the compression actually comes from, and it is what a
fixed codec cannot do: in BF16 weights the exponent plane shrinks by ~2.6x
while the mantissa plane is incompressible noise, and in an FP32 checkpoint
that was upcast from BF16 two planes are entirely zero. Measuring each plane
handles all of those without knowing which case it is in.

Planes that look random are stored without ever being handed to the entropy
coder, which saves most of the time that would be spent proving they are
incompressible.
"""

from __future__ import annotations

import struct
import threading
import zlib
from math import log2

from . import entropy, kernels
from .format import (CODEC_BF16, CODEC_BF16C, CODEC_BLK, CODEC_DELTA,
                     CODEC_GBLK, CODEC_REF, CODEC_SPLIT, CODEC_STORED,
                     CODEC_ZSTD, CorruptArchive)
from .planner import (BLOCK_LAYOUTS, KIND_BF16, KIND_BLOCK, KIND_BYTES,
                      SUB_K4, SUBBLOCK_CTX)

# Above this many bits per byte a stream is genuinely noise and is stored.
# The threshold sits this close to 8 because rANS can profitably code a
# mantissa plane at ~7.90 bits, which anything LZ-based has to turn down.
NOISE_BITS = 7.98

# Bytes sampled when estimating entropy, taken from three places so a
# non-uniform tensor is not judged by its opening bytes alone.
SAMPLE_BYTES = 3 * 65536

# Exponent-conditioned coding of the sign+mantissa plane. Eight equal-mass
# exponent buckets were measured to capture the whole usable dependence on
# real weights (the full 256-way context gains nothing further), and below
# ~1M elements the extra frequency tables cost more than conditioning saves,
# so small chunks skip the attempt entirely.
COND_BUCKETS = 8
COND_MIN_ELEMS = 1 << 20

# What one rANS stream costs beyond its coded bytes: the 516-byte frequency
# header plus the flushed states.
RANS_OVERHEAD = 516 + 4 * 8

_bf16c_hdr = struct.Struct(f"<BB{COND_BUCKETS}BI{COND_BUCKETS}I")

# A GGUF Q8_0 block: 2-byte fp16 scale then 32 int8 quants. Coding scale
# bytes and quant bytes in one alphabet was measured to waste 2% of the
# payload. v3 archives hold this shape and no other; the generalised block
# codec below writes everything now.
Q80_PERIOD = 34

_q80_hdr = struct.Struct(f"<BB{COND_BUCKETS}BBI{COND_BUCKETS}II")

# Generalised block split. A quantised GGUF block is a struct of fields, and
# each field is judged separately:
#
#   GRP_CONCAT  one stream for the whole field. A 128-byte quant array is one
#               alphabet repeated, so one table beats 128 near-identical ones.
#   GRP_PLANES  one stream per byte position. The two halves of an fp16 scale,
#               or the twelve bytes of Q4_K's packed 6-bit sub-scales, hold
#               genuinely different alphabets -- measured 15.6% recoverable
#               split by position against 10.9% merged.
#   GRP_COND    a 2-byte field is an fp16: code the high byte plain and the low
#               byte per high-byte bucket, the same conditioning the Q8_0 scale
#               already used. Worth another 0.19% of a Q4_K file.
#
# The choice is made per field from exact histograms and written into the
# payload, so no per-quantisation tuning is compiled in and a GGUF type this
# build has never seen still decodes on any other build.
#   GRP_SUB     a k-quant's quants, coded per sub-block class. See below.
GRP_CONCAT = 0
GRP_PLANES = 1
GRP_COND = 2
GRP_SUB = 3

# Sub-block conditioning for k-quants.
#
# A Q4_K super-block is eight sub-blocks, each with its own 6-bit scale and
# 6-bit min, storing `d*q - m`. A quant's alphabet therefore depends on which
# sub-block produced it, and both parameters sit earlier in the block, so the
# decoder holds them before it needs them. Two things stand in the way and
# both are undone in the kernel: the parameters are packed six bits at a time
# with four of each straddling a byte boundary, and ggml interleaves two
# sub-blocks per quant byte, so every byte plane mixes two alphabets.
#
# Four scale classes by four min classes was measured best on real Llama Q4_K
# weights once the frequency tables are paid for. The min carries most of it:
# at a fixed sixteen streams, splitting the min four ways and the scale four
# ways beats splitting the scale sixteen ways and ignoring the min by half a
# point of the whole file.
#
#   quants, one shared table (v4)     989.28 bits/block
#   conditioned, 16 streams           979.53 bits/block   -> +0.74% of the file
#
# Finer contexts model better and lose anyway: eight by eight reaches 977.86
# bits but needs 64 tables, which costs more than the extra 1.7 bits returns.
SUB_KS = 4
SUB_KM = 4
SUB_BUCKETS = SUB_KS * SUB_KM

# Below this the per-stream frequency tables cost more than the split saves.
# The estimator declines on its own well before here; this only avoids doing
# the arithmetic on tiny chunks.
BLOCK_MIN_BLOCKS = 1 << 12

_gblk_hdr = struct.Struct("<HB")   # block period, field count
_gblk_grp = struct.Struct("<HHB")  # start, width, mode
_gblk_str = struct.Struct("<BI")   # method, payload length
# Follows a GRP_SUB group only: where its context field is, how the block
# divides, and how finely each parameter is bucketed. Written out so the
# payload stays self-describing -- a decoder needs no ggml type table.
_gblk_sub = struct.Struct("<HHBBBB")  # ctx start, ctx width, nsub, kind, ks, km

_lut_cache: dict[int, bytes] = {}


def _ident_lut(k: int) -> bytes:
    """Bucket map for a context plane that already holds bucket indices."""
    lut = _lut_cache.get(k)
    if lut is None:
        lut = _lut_cache[k] = bytes(min(i, k - 1) for i in range(256))
    return lut


def _sub_classes(ctx_planes, nblocks: int, ks: int, km: int) -> bytes:
    """One context class per (sub-block, block), from the packed parameters.

    Returns 8 planes of `nblocks` bytes, each value in [0, ks*km). Every input
    is a decoded byte and both bucket maps are pure functions of their
    histograms, so the decoder rebuilds this exactly and none of it is stored.
    """
    sc, mn = kernels.k4_scales(ctx_planes, nblocks)
    lut_s = kernels.bucket_lut(kernels.histogram(sc), ks)
    lut_m = kernels.bucket_lut(kernels.histogram(mn), km)
    # Pre-scale the scale map by km so the two compose by addition, then add
    # the planes as one big integer. Every byte sum is at most ks*km-1, so no
    # carry ever crosses a byte and this stays an elementwise add at C speed.
    a = bytes(sc).translate(bytes(min(x, ks - 1) * km for x in lut_s))
    b = bytes(mn).translate(bytes(min(x, km - 1) for x in lut_m))
    if not a:
        return b""
    return (int.from_bytes(a, "big") + int.from_bytes(b, "big")).to_bytes(
        len(a), "big")


def _sub_segments(mv, nblocks: int, start: int, width: int, sub):
    """Deal a k-quant's quants into one segment per sub-block class.

    Returns (segments, classes). Segment b concatenates, over sub-blocks in
    order, every packed byte whose sub-block fell in class b -- so one
    frequency table serves the whole field for that class.
    """
    ctx_start, ctx_width, nsub, _kind, ks, km = sub
    k = ks * km
    cls = _sub_classes(mv[ctx_start * nblocks:(ctx_start + ctx_width) * nblocks],
                       nblocks, ks, km)
    packed = kernels.k4_pack(mv[start * nblocks:(start + width) * nblocks],
                             nblocks)
    pmv = memoryview(packed)
    lut = _ident_lut(k)
    per = width // nsub
    segs = [[] for _ in range(k)]
    for s in range(nsub):
        ctx = cls[s * nblocks:(s + 1) * nblocks] * per
        part, counts = kernels.bucket_partition(
            ctx, pmv[s * per * nblocks:(s + 1) * per * nblocks], lut, k)
        part = memoryview(part)
        pos = 0
        for b in range(k):
            if counts[b]:
                segs[b].append(bytes(part[pos:pos + counts[b]]))
                pos += counts[b]
    return [b"".join(x) for x in segs], cls

_hdr_cache: dict[int, struct.Struct] = {}


def _plane_header(nplanes: int) -> struct.Struct:
    s = _hdr_cache.get(nplanes)
    if s is None:
        s = _hdr_cache[nplanes] = struct.Struct(f"<{nplanes}I")
    return s


def _sample(buf) -> bytes:
    n = len(buf)
    if n <= SAMPLE_BYTES:
        return bytes(buf)
    w = SAMPLE_BYTES // 3
    mid = (n - w) // 2
    mv = memoryview(buf)
    return b"".join((mv[:w], mv[mid:mid + w], mv[n - w:]))


def estimate_entropy(buf) -> float:
    """Order-0 entropy in bits per byte, from a sample of the buffer."""
    s = _sample(buf)
    if not s:
        return 0.0
    counts = kernels.histogram(s)
    total = len(s)
    h = 0.0
    for c in counts:
        if c:
            p = c / total
            h -= p * log2(p)
    return h


class _Sink:
    """One encode attempt's worth of per-stream records, not yet committed.

    An attempt can lose -- a block split that fails to beat its own input, a
    conditioned BF16 chunk that the plain split undercuts -- and the streams it
    coded never reach the archive. Recording into a sink and committing only the
    attempt that won is what keeps the tally a description of the file rather
    than of the search.
    """

    __slots__ = ("rows", "measure_alt")

    def __init__(self, measure_alt: bool = False):
        self.rows: list[tuple[int, int, int, int | None]] = []
        self.measure_alt = measure_alt

    def add(self, method: int, raw: int, coded: int, alt: int | None = None):
        self.rows.append((method, raw, coded, alt))


class MethodTally:
    """Which entropy coder actually did the work, and on how much of the file.

    The chunk table records the codec that framed each chunk, never the coder
    that compressed the planes inside it, so a BF16 split chunk says nothing
    about whether rANS or zstd earned the bytes. Every coded stream in every
    codec passes through `_encode_stream`, which makes that the one place the
    question can be answered for the whole archive.

    `contested` counts only the streams where both coders were actually run, so
    `alt` is a real measured size rather than a guess. On the plane path only
    rANS runs, so those streams are contested solely when `measure_alt` asks for
    the loser to be computed too -- which doubles the coding work and is
    therefore never on by default.
    """

    __slots__ = ("_by_method", "_lock", "measure_alt")

    def __init__(self, measure_alt: bool = False):
        # method -> [streams, raw, coded, contested, contested_coded, alt]
        self._by_method: dict[int, list[int]] = {}
        self._lock = threading.Lock()
        self.measure_alt = measure_alt

    def sink(self) -> _Sink:
        return _Sink(self.measure_alt)

    def commit(self, sink: _Sink | None) -> None:
        if not sink or not sink.rows:
            return
        with self._lock:
            for method, raw, coded, alt in sink.rows:
                slot = self._by_method.setdefault(method, [0, 0, 0, 0, 0, 0])
                slot[0] += 1
                slot[1] += raw
                slot[2] += coded
                if alt is not None:
                    slot[3] += 1
                    slot[4] += coded
                    slot[5] += alt

    def stored(self, raw: int) -> None:
        """A whole chunk kept as-is, whatever its discarded attempts coded."""
        sink = _Sink()
        sink.add(entropy.METHOD_STORED, raw, raw)
        self.commit(sink)

    def to_json(self) -> dict:
        out = {}
        with self._lock:
            for method, slot in self._by_method.items():
                name = entropy.METHOD_NAMES.get(method, f"method-{method}")
                row = {"streams": slot[0], "raw": slot[1], "coded": slot[2]}
                if slot[3]:
                    row["contested"] = slot[3]
                    row["contested_coded"] = slot[4]
                    row["contested_alt"] = slot[5]
                out[name] = row
        return out


def merge_methods(a: dict | None, b: dict | None) -> dict:
    """Sum two method tables, for an archive that grew by being appended to."""
    out = {name: dict(row) for name, row in (a or {}).items()}
    for name, row in (b or {}).items():
        slot = out.setdefault(name, {})
        for key, value in row.items():
            slot[key] = slot.get(key, 0) + value
    return out


def _encode_stream(buf, level: int, method: int, plane: bool = False, sink=None,
                   hist=None):
    """Compress one stream unless doing so is not worth it.

    Returns (method_used, payload). `sink` records what was chosen and what it
    cost; see MethodTally for why the record is provisional.

    Planes get the order-0 rANS coder, which is what the data actually calls
    for. A mantissa plane sits at about 7.90 bits per byte -- a real 1.3%
    of structure, but so little that a general-purpose coder's framing and
    match search eat the whole gain and then some. rANS charges fractional
    bits and almost no overhead, so it collects that 1.3% where zstd could
    only decline the job.

    `hist` is this stream's exact histogram when the caller has already taken
    one, and it changes both decisions here. Whether to code at all is settled
    by pricing the stream against the coder's own frequencies rather than by
    sampling it, and `_rans_cost` is a floor on that price -- so a stream it
    cannot bring under the threshold will not get there by being coded. Those
    are the streams that used to be coded in full and then discarded. What is
    coded is coded from the counts already in hand rather than counting them
    a second time.
    """
    n = len(buf)
    if n == 0:
        return entropy.METHOD_STORED, b""
    if hist is None and plane and kernels.have_rans():
        # No caller-supplied counts, so the sample decides whether this is
        # noise -- and noise is stored here without ever being counted, which
        # is most of what the sample is for. Anything that survives it is going
        # to be counted anyway, by the coder if not here, so counting it now
        # costs nothing and buys the exact price below. Streams that cannot pay
        # then never reach the coder at all.
        if estimate_entropy(buf) >= NOISE_BITS:
            if sink is not None:
                sink.add(entropy.METHOD_STORED, n, n)
            return entropy.METHOD_STORED, buf
        hist = kernels.histogram(buf)
    cost = _rans_cost(hist, n) if hist is not None else None
    if cost is not None:
        hopeless = cost + _min_gain(n) >= n
    else:
        hopeless = estimate_entropy(buf) >= NOISE_BITS
    if hopeless:
        if sink is not None:
            sink.add(entropy.METHOD_STORED, n, n)
        return entropy.METHOD_STORED, buf

    best_method, best, alt_size = entropy.METHOD_STORED, None, None
    if plane and kernels.have_rans():
        best = kernels.rans_encode(buf, hist)
        if best is not None:
            best_method = entropy.METHOD_RANS
        if sink is not None and sink.measure_alt:
            # What the general-purpose coder would have charged for the same
            # bytes. Only ever computed on request: it is a second full
            # compression of a stream that is already coded.
            try:
                alt_size = len(entropy.compress(buf, level, method))
            except Exception:
                alt_size = None
    else:
        try:
            best = entropy.compress(buf, level, method)
            best_method = method
        except entropy.UnsupportedMethod:
            # The caller named a coder this interpreter does not have -- zstd
            # on anything before 3.14 without the package. Fall back to the
            # one it does have rather than leaving `best` empty, which would
            # hand the stream to rANS below without ever pricing it against a
            # general-purpose coder.
            try:
                best = entropy.compress(buf, level, entropy.DEFAULT_METHOD)
                best_method = entropy.DEFAULT_METHOD
            except Exception:
                best = None
        except Exception:
            best = None
        # A stream the general-purpose coder barely dented may still be a
        # plain symbol distribution -- quantised INT8 weights, for one.
        #
        # This used to run only where zstd had failed to dent the stream by
        # 1/32, which is a threshold int8 weights never reach: a per-channel
        # scale maps each filter's maximum to 127 and leaves the body
        # concentrated, so zstd saves a comfortable 10-11% and rANS was never
        # priced. It is worth pricing anyway, and not only for the ~0.8 points
        # it wins there. `lmz.gpu` decodes rANS and nothing else, so a chunk
        # that lands on METHOD_ZSTD can never ride the batch decoder -- for
        # int8, the dtype every deployed perception model uses, choosing zstd
        # by default strands the weights on the CPU permanently.
        #
        # `cost` is a floor on what rANS will charge. The general path is not
        # given a histogram, so one is counted here -- a single pass over a
        # stream zstd has already read, against a second full compression it
        # saves wherever the floor says rANS cannot win.
        if kernels.have_rans() and cost is None:
            hist = kernels.histogram(buf)
            cost = _rans_cost(hist, n)
        if kernels.have_rans() and (best is None or cost is None
                                    or cost < len(best)):
            alt = kernels.rans_encode(buf, hist)
            if alt is not None:
                if best is None:
                    best_method, best = entropy.METHOD_RANS, alt
                elif len(alt) <= len(best):
                    # Ties go to rANS: same bytes on disk, and the GPU can
                    # decode it.
                    alt_size = len(best)   # both ran, so the loser is measured
                    best_method, best = entropy.METHOD_RANS, alt
                else:
                    alt_size = len(alt)

    if best is None or len(best) + _min_gain(n) >= n:
        if sink is not None:
            sink.add(entropy.METHOD_STORED, n, n)
        return entropy.METHOD_STORED, buf
    if sink is not None:
        sink.add(best_method, n, len(best), alt_size)
    return best_method, best


def _min_gain(n: int) -> int:
    """Smallest saving worth encoding for: the coder's own header, plus a bit."""
    return max(1024, n >> 8)


# How far the cost below may sit above the stream the coder actually writes.
# The two differ by where the interleaved states happen to fall when they are
# flushed, which measures within a byte or two either way on every
# distribution tested; this is that slack with room to spare. Subtracting it
# is what makes the figure a floor, which is the only form a caller can skip
# work on: below the floor there is nothing to find.
RANS_COST_SLACK = 64


def _rans_cost(hist, total: int) -> int | None:
    """A floor on what rANS will charge for these counts, in bytes.

    Priced against the frequencies the coder will really use, not against the
    histogram's own entropy. The two part company precisely where the decision
    is close: quantising to twelve bits costs a few tenths of a percent on a
    near-uniform alphabet, which is the same order as the saving being weighed,
    so the raw-entropy estimate says yes to streams that end up stored.

    None when the native coder cannot say, or when the counts are not this
    stream's -- the kernels that take a histogram check the same total, but
    they only guard the coding, and a decision made from another stream's
    counts would skip work that was worth doing. Both ends check.
    """
    if sum(hist) != total:
        return None
    freqs = kernels.rans_freqs(hist)
    if not freqs:
        return None
    scale = sum(freqs)
    if not scale:
        return None
    bits = 0.0
    for c, f in zip(hist, freqs):
        if c:
            bits -= c * log2(f / scale)
    return RANS_OVERHEAD + int(bits / 8) - RANS_COST_SLACK


def _hist_entropy(counts, total: int) -> float:
    if not total:
        return 0.0
    h = 0.0
    for c in counts:
        if c:
            p = c / total
            h -= p * log2(p)
    return h


def _est_stream(total: int, h: float) -> int:
    """Predicted rANS stream size; a stream is never stored above raw size."""
    if total == 0:
        return 0
    return min(total, int(total * h / 8) + RANS_OVERHEAD)


def _encode_bf16_cond(exp, sm, nelem: int, level: int, method: int, sink=None):
    """Code the sign+mantissa plane with one rANS table per exponent bucket.

    Decides from exact histograms whether conditioning beats one shared
    table, and only encodes the winning side, so declining costs one
    histogram pass and a partition rather than a wasted encode. Returns
    (parts, flags) or None to fall back to the plain field split.

    The bucket map is derived from the exponent histogram, which the decoder
    reconstructs from the decoded exponent plane -- nothing about the map is
    stored in the archive.
    """
    ehist = kernels.histogram(exp)
    lut = kernels.bucket_lut(ehist, COND_BUCKETS)
    part, counts = kernels.bucket_partition(exp, sm, lut, COND_BUCKETS,
                                            hist=ehist)
    pmv = memoryview(part)

    # Each segment's histogram is what decides this, and it is also what the
    # coder would otherwise take for itself, so it is kept rather than dropped.
    #
    # These counts look like they should be free. They are the joint counts of
    # (bucket, sign+mantissa byte), and the partition above has just read both
    # planes and already knows the bucket, so it could take them on the way
    # past instead of this pass reading the segments back. Measured, that costs
    # the partition exactly what it saves here -- 1.37 to 1.89 cycles a byte
    # against a pass that runs at about one -- because what a histogram costs is
    # the increment, not the load, and the segments are still warm. Widening
    # lmz_hist from four sub-tables to eight or sixteen is the same story: 1.10x
    # on a skewed plane, 0.95x on a flat one. Counting is near its floor, and
    # every byte counted here is needed exactly, since the bucket map has to be
    # one the decoder can rebuild and the coder's frequencies have to match.
    est_cond = 0
    seg_hists = []
    pos = 0
    for c in counts:
        if not c:
            seg_hists.append(None)
            continue
        hseg = kernels.histogram(pmv[pos:pos + c])
        seg_hists.append(hseg)
        est_cond += _est_stream(c, _hist_entropy(hseg, c))
        pos += c
    # The partition permutes the plane, so its histogram is the sum of the
    # segments' -- the same numbers, and no second pass over the plane to get
    # them. This is the baseline the conditioned side has to beat.
    taken = [h for h in seg_hists if h]
    shist = [sum(x) for x in zip(*taken)] if taken else [0] * 256
    est_plain = _est_stream(nelem, _hist_entropy(shist, nelem))
    # The conditional side pays its larger header and one table per bucket;
    # demand a clear win so estimate noise cannot flip the choice.
    if est_cond + (_bf16c_hdr.size - _plane_header(2).size) + 512 >= est_plain:
        return None

    exp_used, exp_payload = _encode_stream(exp, level, method, plane=True,
                                           sink=sink, hist=ehist)
    sm_methods = []
    seg_lens = []
    parts = []
    pos = 0
    for c, hseg in zip(counts, seg_hists):
        if c == 0:
            sm_methods.append(entropy.METHOD_STORED)
            seg_lens.append(0)
            continue
        used, payload = _encode_stream(pmv[pos:pos + c], level, method,
                                       plane=True, sink=sink, hist=hseg)
        sm_methods.append(used)
        seg_lens.append(len(payload))
        parts.append(payload)
        pos += c
    header = _bf16c_hdr.pack(COND_BUCKETS, exp_used, *sm_methods,
                             len(exp_payload), *seg_lens)
    return [header, exp_payload] + parts, 0


def _group_streams(width: int, mode: int, sub=None) -> int:
    """How many streams a field contributes, from its width and mode alone."""
    if mode == GRP_CONCAT:
        return 1
    if mode == GRP_PLANES:
        return width
    if mode == GRP_SUB:
        return sub[4] * sub[5]
    return 1 + COND_BUCKETS


def _choose_group(mv, nblocks: int, start: int, width: int, sub=None):
    """Pick how one field is coded, from exact histograms of its planes.

    Returns (mode, estimated bytes, conditioning work) where the last item is
    the already-computed partition when GRP_COND wins, so the encode does not
    repeat it.
    """
    hists = [kernels.histogram(mv[(start + j) * nblocks:(start + j + 1) * nblocks])
             for j in range(width)]
    est = _est_stream(nblocks * width, _hist_entropy(
        [sum(h[s] for h in hists) for s in range(256)], nblocks * width))
    mode, best = GRP_CONCAT, est + _gblk_str.size

    if width > 1:
        est = sum(_est_stream(nblocks, _hist_entropy(h, nblocks)) for h in hists)
        est += width * _gblk_str.size
        if est < best:
            mode, best = GRP_PLANES, est

    if sub is not None:
        # A k-quant's quants, coded per sub-block class. The partition is the
        # expensive part and the encode needs it too, so it is handed back
        # rather than recomputed.
        segs, _cls = _sub_segments(mv, nblocks, start, width, sub)
        est = sum(_est_stream(len(s), _hist_entropy(kernels.histogram(s), len(s)))
                  for s in segs)
        est += len(segs) * _gblk_str.size + _gblk_sub.size
        # Demand a clear win so estimate noise cannot flip the choice.
        if est + 512 < best:
            return GRP_SUB, est, segs

    if width != 2 or nblocks == 0:
        return mode, best, None

    # An fp16 field: the low byte is not independent of the high byte, which
    # carries the exponent. Partitioning is O(nblocks) on two planes, cheap
    # enough to measure rather than guess.
    lo = mv[start * nblocks:(start + 1) * nblocks]
    hi = mv[(start + 1) * nblocks:(start + 2) * nblocks]
    lut = kernels.bucket_lut(hists[1], COND_BUCKETS)
    part, counts = kernels.bucket_partition(hi, lo, lut, COND_BUCKETS)
    pmv = memoryview(part)
    est = _est_stream(nblocks, _hist_entropy(hists[1], nblocks))
    pos = 0
    for c in counts:
        if c:
            est += _est_stream(c, _hist_entropy(kernels.histogram(
                pmv[pos:pos + c]), c))
            pos += c
    est += (1 + COND_BUCKETS) * _gblk_str.size
    # Demand a clear win so estimate noise cannot flip the choice.
    if est + 512 < best:
        return GRP_COND, est, (part, counts)
    return mode, best, None


def _encode_gblk(data, nblocks: int, period: int, groups, level: int, method: int,
                 subctx=None, sink=None):
    """Split blocks on their field boundaries and code each field its own way.

    Every field records its own mode and stream lengths in the payload, so the
    decoder needs no table of GGUF layouts to reverse this. Returns
    (parts, flags).
    """
    planes = kernels.split_stride(data, period)
    mv = memoryview(planes)
    descs = []
    subs = {}
    streams = []  # (method, payload) in the order the decoder reads them

    for start, width in groups:
        sub = subctx[1] if (subctx is not None and start == subctx[0]) else None
        mode, _est, cond = _choose_group(mv, nblocks, start, width, sub)
        descs.append((start, width, mode))
        if mode == GRP_SUB:
            subs[start] = sub
            for seg in cond:
                streams.append(_encode_stream(seg, level, method, plane=True,
                                              sink=sink))
        elif mode == GRP_CONCAT:
            streams.append(_encode_stream(
                mv[start * nblocks:(start + width) * nblocks], level, method,
                plane=True, sink=sink))
        elif mode == GRP_PLANES:
            for j in range(width):
                streams.append(_encode_stream(
                    mv[(start + j) * nblocks:(start + j + 1) * nblocks], level,
                    method, plane=True, sink=sink))
        else:
            part, counts = cond
            pmv = memoryview(part)
            streams.append(_encode_stream(
                mv[(start + 1) * nblocks:(start + 2) * nblocks], level, method,
                plane=True, sink=sink))
            pos = 0
            for c in counts:
                if c == 0:
                    streams.append((entropy.METHOD_STORED, b""))
                    continue
                streams.append(_encode_stream(pmv[pos:pos + c], level, method,
                                              plane=True, sink=sink))
                pos += c

    parts = [_gblk_hdr.pack(period, len(descs))]
    for start, width, mode in descs:
        parts.append(_gblk_grp.pack(start, width, mode))
        if mode == GRP_SUB:
            parts.append(_gblk_sub.pack(*subs[start]))
    parts += [_gblk_str.pack(m, len(p)) for m, p in streams]
    return [b"".join(parts)] + [p for _m, p in streams], 0


def _check_sub(sub, start: int, width: int) -> None:
    """Validate a sub-block descriptor read from the payload.

    These bytes are not covered by the chunk record, so anything outside what
    the one defined packing can mean is damage rather than a future format.
    """
    ctx_start, ctx_width, nsub, kind, ks, km = sub
    if kind != SUB_K4:
        raise CorruptArchive(f"block field names sub-block packing {kind}")
    if (ctx_width != kernels.K4_SCALE_BYTES or nsub != kernels.K4_SUBBLOCKS
            or width != kernels.K4_QUANT_BYTES):
        raise CorruptArchive("sub-block conditioning has the wrong field sizes")
    if ctx_start + ctx_width > start:
        # The context has to be decoded before the field it conditions.
        raise CorruptArchive("sub-block context does not precede its quants")
    if not (1 <= ks <= 32 and 1 <= km <= 32 and ks * km <= 32):
        raise CorruptArchive(f"sub-block conditioning declares {ks}x{km} classes")


def _decode_sub(lanes, nblocks: int, start: int, width: int, sub, take, skip):
    """Rebuild a k-quant's quant planes from its per-class streams.

    The context field is already decoded, so its bucket maps and every
    segment length follow from bytes both sides agree on; the payload carries
    no map and no counts.
    """
    ctx_start, ctx_width, nsub, _kind, ks, km = sub
    k = ks * km
    per = width // nsub

    gathered = bytearray(ctx_width * nblocks)
    for j in range(ctx_width):
        obj, off = lanes[ctx_start + j]
        gathered[j * nblocks:(j + 1) * nblocks] = memoryview(obj)[off:off + nblocks]
    cls = _sub_classes(gathered, nblocks, ks, km)

    hists = [kernels.histogram(cls[s * nblocks:(s + 1) * nblocks])
             for s in range(nsub)]
    counts = [sum(h[b] for h in hists) * per for b in range(k)]

    streams = []
    for b in range(k):
        if counts[b] == 0:
            skip(f"sub-block class {b}")
            streams.append([b"", 0])
        else:
            obj, off = take(counts[b], f"sub-block class {b}")
            streams.append([obj, off])

    lut = _ident_lut(k)
    packed = bytearray(width * nblocks)
    pv = memoryview(packed)
    for s in range(nsub):
        ctx = cls[s * nblocks:(s + 1) * nblocks] * per
        kernels.bucket_unpartition(ctx, [tuple(x) for x in streams], lut, k,
                                   pv[s * per * nblocks:(s + 1) * per * nblocks])
        for b in range(k):
            streams[b][1] += hists[s][b] * per
    return kernels.k4_unpack(packed, nblocks)


def _decode_gblk(payload, rlen: int, out=None):
    """Decode a field-split block chunk back to interleaved blocks.

    Everything needed is in the payload: the block period, where each field
    starts, how it was coded, and how long each stream is. The fields must
    tile the block exactly, which is the check that catches a damaged header
    before any of it is used as a length.
    """
    if len(payload) < _gblk_hdr.size:
        raise CorruptArchive("block chunk is truncated")
    period, ngroups = _gblk_hdr.unpack_from(payload, 0)
    if period == 0 or ngroups == 0 or rlen % period:
        raise CorruptArchive(
            f"block chunk declares a {period}-byte block for {rlen} bytes")
    nblocks = rlen // period
    pos = _gblk_hdr.size

    descs = []
    covered = 0
    nstreams = 0
    for _ in range(ngroups):
        if pos + _gblk_grp.size > len(payload):
            raise CorruptArchive("block chunk is truncated")
        start, width, mode = _gblk_grp.unpack_from(payload, pos)
        pos += _gblk_grp.size
        if start != covered or width == 0 or start + width > period:
            raise CorruptArchive("block fields do not tile the block")
        if mode not in (GRP_CONCAT, GRP_PLANES, GRP_COND, GRP_SUB):
            raise CorruptArchive(f"block field names mode {mode}")
        if mode == GRP_COND and width != 2:
            raise CorruptArchive("a conditioned block field must be 2 bytes")
        sub = None
        if mode == GRP_SUB:
            if pos + _gblk_sub.size > len(payload):
                raise CorruptArchive("block chunk is truncated")
            sub = _gblk_sub.unpack_from(payload, pos)
            pos += _gblk_sub.size
            _check_sub(sub, start, width)
        covered += width
        nstreams += _group_streams(width, mode, sub)
        descs.append((start, width, mode, sub))
    if covered != period:
        raise CorruptArchive("block fields do not tile the block")

    if pos + nstreams * _gblk_str.size > len(payload):
        raise CorruptArchive("block chunk is truncated")
    sdesc = []
    for _ in range(nstreams):
        m, ln = _gblk_str.unpack_from(payload, pos)
        pos += _gblk_str.size
        if m not in entropy.METHOD_NAMES:
            raise CorruptArchive(f"block chunk names method {m}")
        sdesc.append((m, ln))

    mv = memoryview(payload)
    idx = 0

    def take(want: int, what: str):
        nonlocal pos, idx
        m, ln = sdesc[idx]
        idx += 1
        if pos + ln > len(payload):
            raise CorruptArchive("block chunk is truncated")
        if m == entropy.METHOD_STORED:
            if ln != want:
                raise CorruptArchive(f"{what} holds {ln} bytes, expected {want}")
            block = (payload, pos)
        else:
            data = _decode_stream(mv[pos:pos + ln], m, what, want)
            if len(data) != want:
                raise CorruptArchive(
                    f"{what} decoded to {len(data)} bytes, expected {want}")
            block = (data, 0)
        pos += ln
        return block

    def skip(what: str):
        """Consume the stream slot of a context class that holds no elements."""
        nonlocal idx
        if sdesc[idx][1]:
            raise CorruptArchive(f"{what} should be empty but holds data")
        idx += 1

    lanes = [None] * period
    for start, width, mode, sub in descs:
        if mode == GRP_CONCAT:
            obj, off = take(width * nblocks, f"block bytes {start}..{start + width}")
            for j in range(width):
                lanes[start + j] = (obj, off + j * nblocks)
        elif mode == GRP_PLANES:
            for j in range(width):
                lanes[start + j] = take(nblocks, f"block byte {start + j}")
        elif mode == GRP_SUB:
            q = _decode_sub(lanes, nblocks, start, width, sub, take, skip)
            for j in range(width):
                lanes[start + j] = (q, j * nblocks)
        else:
            hi_obj, hi_off = take(nblocks, f"block byte {start + 1}")
            hi = bytes(memoryview(hi_obj)[hi_off:hi_off + nblocks])
            hhist = kernels.histogram(hi)
            lut = kernels.bucket_lut(hhist, COND_BUCKETS)
            counts = kernels.bucket_counts(hhist, lut, COND_BUCKETS)
            segs = []
            for b in range(COND_BUCKETS):
                if counts[b] == 0:
                    if sdesc[idx][1]:
                        raise CorruptArchive(
                            f"block bucket {b} should be empty but holds data")
                    idx += 1
                    segs.append((b"", 0))
                else:
                    segs.append(take(counts[b], f"block bucket {b}"))
            lanes[start] = (kernels.bucket_unpartition(hi, segs, lut,
                                                       COND_BUCKETS), 0)
            lanes[start + 1] = (hi, 0)

    if pos != len(payload):
        raise CorruptArchive("block chunk carries trailing bytes")
    buf = out if (out is not None and len(out) >= rlen) else bytearray(rlen)
    return kernels.merge_stride(lanes, nblocks, period, buf)


def _decode_q80(payload, nblocks: int, out=None):
    """Decode a block-split Q8_0 chunk back to interleaved 34-byte blocks."""
    hdr = _q80_hdr
    if len(payload) < hdr.size:
        raise CorruptArchive("block chunk is truncated")
    fields = hdr.unpack_from(payload, 0)
    k = fields[0]
    if k not in (0, COND_BUCKETS):
        raise CorruptArchive(f"block chunk declares {k} scale buckets")
    hi_method = fields[1]
    lo_methods = fields[2:2 + COND_BUCKETS]
    q_method = fields[2 + COND_BUCKETS]
    hi_len = fields[3 + COND_BUCKETS]
    lo_lens = fields[4 + COND_BUCKETS:4 + 2 * COND_BUCKETS]
    q_len = fields[4 + 2 * COND_BUCKETS]
    for m in (hi_method, q_method, *lo_methods):
        if m not in entropy.METHOD_NAMES:
            raise CorruptArchive(f"block chunk names method {m}")

    mv = memoryview(payload)
    pos = hdr.size

    def take(ln, m, what, want):
        nonlocal pos
        if pos + ln > len(payload):
            raise CorruptArchive("block chunk is truncated")
        if m == entropy.METHOD_STORED:
            if ln != want:
                raise CorruptArchive(f"{what} holds {ln} bytes, expected {want}")
            block = (payload, pos)
        else:
            data = _decode_stream(mv[pos:pos + ln], m, what, want)
            if len(data) != want:
                raise CorruptArchive(
                    f"{what} decoded to {len(data)} bytes, expected {want}")
            block = (data, 0)
        pos += ln
        return block

    hi_obj, hi_off = take(hi_len, hi_method, "scale-high plane", nblocks)
    hi = bytes(memoryview(hi_obj)[hi_off:hi_off + nblocks])

    if k == 0:
        lo_obj, lo_off = take(lo_lens[0], lo_methods[0], "scale-low plane",
                              nblocks)
        for b in range(1, COND_BUCKETS):
            if lo_lens[b]:
                raise CorruptArchive("unbucketed block chunk carries bucket data")
        lo = (lo_obj, lo_off)
    else:
        hhist = kernels.histogram(hi)
        lut = kernels.bucket_lut(hhist, COND_BUCKETS)
        counts = kernels.bucket_counts(hhist, lut, COND_BUCKETS)
        streams = []
        for b in range(COND_BUCKETS):
            want = counts[b]
            if want == 0:
                if lo_lens[b]:
                    raise CorruptArchive(
                        f"scale bucket {b} should be empty but holds data")
                streams.append((b"", 0))
            else:
                streams.append(take(lo_lens[b], lo_methods[b],
                                    f"scale bucket {b}", want))
        lo = (kernels.bucket_unpartition(hi, streams, lut, COND_BUCKETS), 0)

    q_obj, q_off = take(q_len, q_method, "quant plane", 32 * nblocks)

    streams = [lo, (hi, 0)]
    streams += [(q_obj, q_off + j * nblocks) for j in range(32)]
    buf = out if (out is not None and len(out) >= nblocks * Q80_PERIOD) else \
        bytearray(nblocks * Q80_PERIOD)
    return kernels.merge_stride(streams, nblocks, Q80_PERIOD, buf)


def encode_chunk(data, esize: int, level: int = 1, checksum: bool = True,
                 method: int = -1, kind: int = KIND_BYTES, btype: int = -1,
                 tally=None):
    """Encode one chunk.

    Returns (payload, codec, flags, crc). `payload` may be a list of buffers,
    which the caller writes in order.

    `tally`, when given, is told about the streams of whichever attempt is
    kept. Each attempt below codes into its own sink, so the ones that lose --
    a block split that cannot beat its own input, a conditioned BF16 chunk the
    plain split undercuts -- leave no trace. Otherwise the tally would describe
    the search rather than the archive, and would credit a coder for bytes that
    were ultimately stored raw.
    """
    if method < 0:
        method = entropy.DEFAULT_METHOD
    crc = zlib.crc32(data) & 0xFFFFFFFF if checksum else 0
    n = len(data)
    new_sink = tally.sink if tally is not None else lambda: None

    layout = BLOCK_LAYOUTS.get(btype) if kind == KIND_BLOCK else None
    if (layout is not None and esize == layout[0] and n % esize == 0
            and n // esize >= BLOCK_MIN_BLOCKS and kernels.have_rans()):
        recipe = SUBBLOCK_CTX.get(btype)
        subctx = None
        if recipe is not None:
            qstart, cstart, skind = recipe
            subctx = (qstart, (cstart, kernels.K4_SCALE_BYTES,
                               kernels.K4_SUBBLOCKS, skind, SUB_KS, SUB_KM))
        sink = new_sink()
        parts, flags = _encode_gblk(data, n // esize, layout[0], layout[1],
                                    level, method, subctx, sink=sink)
        if sum(len(p) for p in parts) + _min_gain(n) < n:
            if tally is not None:
                tally.commit(sink)
            return parts, CODEC_GBLK, flags, crc
        # Otherwise fall through: a block period is not a splittable width, so
        # the chunk takes the generic single-stream path.

    splittable = esize > 1 and n % esize == 0 and esize in kernels.SUPPORTED_ESIZES
    if kind == KIND_BF16 and esize == 2 and n % 2 == 0:
        nplanes, cid = 2, CODEC_BF16
        planes = kernels.split_bf16(data)
        nelem = n // 2
        if nelem >= COND_MIN_ELEMS and kernels.have_rans():
            mv = memoryview(planes)
            sink = new_sink()
            cond = _encode_bf16_cond(mv[:nelem], mv[nelem:], nelem, level, method,
                                     sink=sink)
            if cond is not None:
                parts, flags = cond
                if sum(len(p) for p in parts) + _min_gain(n) < n:
                    if tally is not None:
                        tally.commit(sink)
                    return parts, CODEC_BF16C, flags, crc
    elif splittable:
        nplanes, cid = esize, CODEC_SPLIT
        planes = kernels.split(data, esize)
    else:
        nplanes = 0

    if nplanes:
        nelem = n // nplanes
        mv = memoryview(planes)
        parts, lengths, flags = [], [], 0
        total = 0
        any_compressed = False
        sink = new_sink()
        for k in range(nplanes):
            used, payload = _encode_stream(mv[k * nelem:(k + 1) * nelem], level,
                                           method, plane=True, sink=sink)
            flags |= (used & 0x3) << (2 * k)
            lengths.append(len(payload))
            parts.append(payload)
            total += len(payload)
            if used != entropy.METHOD_STORED:
                any_compressed = True
        # If nothing compressed, the split only shuffled bytes; storing the
        # original avoids paying for a merge on the way back.
        if any_compressed and total + _plane_header(nplanes).size < n:
            if tally is not None:
                tally.commit(sink)
            return ([_plane_header(nplanes).pack(*lengths)] + parts,
                    cid, flags, crc)
        if tally is not None:
            tally.stored(n)   # the coded planes are discarded with the split
        return ([data], CODEC_STORED, 0, crc)

    sink = new_sink()
    used, payload = _encode_stream(data, level, method, sink=sink)
    if tally is not None:
        tally.commit(sink)
    if used == entropy.METHOD_STORED:
        return ([data], CODEC_STORED, 0, crc)
    return ([payload], CODEC_ZSTD, used & 0x3, crc)


# A delta chunk: the output offset it builds on, then which codec was used on
# the difference. The difference is coded by the ordinary path, so it gets the
# plane split and the same per-plane adaptivity as anything else.
_delta_hdr = struct.Struct("<QB")


def encode_delta(data, base, esize: int, level: int = 1, checksum: bool = True,
                 kind: int = KIND_BYTES, btype: int = -1, src: int = 0,
                 tally=None):
    """Code a chunk as its difference from an earlier output range.

    The checksum is of the original bytes, not the difference, so it verifies
    the reconstruction rather than the intermediate. Returns (parts, flags,
    crc); the codec id is always CODEC_DELTA.
    """
    crc = zlib.crc32(data) & 0xFFFFFFFF if checksum else 0
    diff = kernels.xor_bytes(data, base)
    parts, cid, flags, _ = encode_chunk(bytes(diff), esize, level, False,
                                        kind=kind, btype=btype, tally=tally)
    return [_delta_hdr.pack(src, cid)] + list(parts), flags, crc


def delta_source(payload):
    """Where a delta chunk's source range starts, and where its data does."""
    if len(payload) < _delta_hdr.size:
        raise CorruptArchive("delta chunk is truncated")
    src, inner = _delta_hdr.unpack_from(payload, 0)
    if inner in (CODEC_REF, CODEC_DELTA):
        raise CorruptArchive(
            f"delta chunk names codec {inner} for its own difference")
    return src, inner, _delta_hdr.size


def decode_delta(payload, esize: int, flags: int, rlen: int, base, crc: int = 0,
                 verify: bool = True):
    """Rebuild a delta chunk from its source bytes and its coded difference."""
    _src, inner, off = delta_source(payload)
    if len(base) != rlen:
        raise CorruptArchive(
            f"delta source resolved to {len(base)} bytes, expected {rlen}")
    diff = decode_chunk(memoryview(payload)[off:], inner, esize, flags, rlen,
                        0, False)
    result = kernels.xor_bytes(base, diff)
    if verify and crc and (zlib.crc32(result) & 0xFFFFFFFF) != crc:
        raise CorruptArchive("chunk failed its checksum; the archive is corrupt")
    return result


def _decode_stream(raw, method: int, what: str, out_len: int | None = None):
    """Decompress one stream, reporting backend failures as corruption.

    Damaged data usually surfaces as an error from the entropy coder rather
    than a checksum mismatch, because decoding fails before there is anything
    to checksum. Callers should see one recognisable error either way.
    """
    try:
        return entropy.decompress(raw, method, out_len)
    except entropy.UnsupportedMethod:
        raise
    except Exception as exc:
        raise CorruptArchive(f"{what} failed to decode: {exc}") from exc


def _decode_bf16c(payload, nelem: int, out=None):
    """Decode a conditioned bf16 chunk back to interleaved bytes.

    The exponent plane comes out first; its histogram rebuilds the bucket
    map, which yields each bucket's expected length, and only then can the
    sign+mantissa segments be decoded and dealt back into element order.
    """
    hdr = _bf16c_hdr
    if len(payload) < hdr.size:
        raise CorruptArchive("conditioned bf16 chunk is truncated")
    fields = hdr.unpack_from(payload, 0)
    k = fields[0]
    if k != COND_BUCKETS:
        raise CorruptArchive(f"conditioned bf16 chunk declares {k} buckets")
    exp_method = fields[1]
    sm_methods = fields[2:2 + k]
    exp_len = fields[2 + k]
    seg_lens = fields[3 + k:3 + k + k]
    # These bytes came from the payload, not the validated chunk record; a
    # value outside the method alphabet is damage, not a future format.
    for m in (exp_method, *sm_methods):
        if m not in entropy.METHOD_NAMES:
            raise CorruptArchive(f"conditioned bf16 chunk names method {m}")

    mv = memoryview(payload)
    pos = hdr.size
    if pos + exp_len > len(payload):
        raise CorruptArchive("conditioned bf16 chunk is truncated")
    if exp_method == entropy.METHOD_STORED:
        if exp_len != nelem:
            raise CorruptArchive(
                f"exponent plane holds {exp_len} bytes, expected {nelem}")
        exp = bytes(mv[pos:pos + exp_len])
    else:
        exp = _decode_stream(mv[pos:pos + exp_len], exp_method,
                             "exponent plane", nelem)
        if len(exp) != nelem:
            raise CorruptArchive(
                f"exponent plane decoded to {len(exp)} bytes, expected {nelem}")
    pos += exp_len

    ehist = kernels.histogram(exp)
    lut = kernels.bucket_lut(ehist, k)
    counts = kernels.bucket_counts(ehist, lut, k)

    streams = []
    for b in range(k):
        ln = seg_lens[b]
        want = counts[b]
        if pos + ln > len(payload):
            raise CorruptArchive("conditioned bf16 chunk is truncated")
        m = sm_methods[b]
        if want == 0:
            if ln:
                raise CorruptArchive(f"bucket {b} should be empty but holds {ln} bytes")
            streams.append((b"", 0))
        elif m == entropy.METHOD_STORED:
            if ln != want:
                raise CorruptArchive(
                    f"bucket {b} holds {ln} bytes, expected {want}")
            streams.append((payload, pos))
        else:
            block = _decode_stream(mv[pos:pos + ln], m, f"bucket {b}", want)
            if len(block) != want:
                raise CorruptArchive(
                    f"bucket {b} decoded to {len(block)} bytes, expected {want}")
            streams.append((block, 0))
        pos += ln

    sm = kernels.bucket_unpartition(exp, streams, lut, k)
    buf = out if (out is not None and len(out) >= nelem * 2) else bytearray(nelem * 2)
    return kernels.merge_bf16([(exp, 0), (sm, 0)], nelem, buf)


def decode_chunk(payload, codec: int, esize: int, flags: int, rlen: int,
                 crc: int = 0, verify: bool = True, out=None):
    """Decode one chunk back to its exact original bytes.

    `out` is an optional scratch buffer to decode into. Callers that hand one
    over must finish with the returned view before reusing the buffer.
    """
    if codec == CODEC_STORED:
        result = payload
        if len(result) != rlen:
            raise ValueError(f"stored chunk is {len(result)} bytes, expected {rlen}")
    elif codec == CODEC_BF16C:
        if esize != 2 or rlen % 2:
            raise ValueError("a conditioned bf16 chunk must have 2-byte elements")
        result = _decode_bf16c(payload, rlen // 2, out)
    elif codec == CODEC_BLK:
        if esize != Q80_PERIOD or rlen % Q80_PERIOD:
            raise ValueError(
                f"a block chunk must be whole {Q80_PERIOD}-byte blocks")
        result = _decode_q80(payload, rlen // Q80_PERIOD, out)
    elif codec == CODEC_GBLK:
        result = _decode_gblk(payload, rlen, out)
    elif codec == CODEC_ZSTD:
        result = _decode_stream(payload, flags & 0x3, "chunk", rlen)
        if len(result) != rlen:
            raise CorruptArchive(
                f"chunk decoded to {len(result)} bytes, expected {rlen}")
    elif codec in (CODEC_SPLIT, CODEC_BF16):
        if esize not in kernels.SUPPORTED_ESIZES or esize < 2 or rlen % esize:
            raise ValueError(f"split chunk has invalid element size {esize}")
        if codec == CODEC_BF16 and esize != 2:
            raise ValueError("a bf16 chunk must have 2-byte elements")
        nplanes = 2 if codec == CODEC_BF16 else esize
        nelem = rlen // nplanes
        # One crossing for the whole chunk. Identical arithmetic to the path
        # below, which still runs whenever the kernel declines -- a plane
        # coded with zstd, a damaged payload, or no native build -- so error
        # reporting and the fallback backends are unchanged.
        fast = kernels.decode_planes(payload, nplanes, nelem, flags,
                                     codec == CODEC_BF16, out)
        if fast is not None:
            result = fast
            if verify and crc:
                if (zlib.crc32(result) & 0xFFFFFFFF) != crc:
                    raise CorruptArchive(
                        "chunk failed its checksum; the archive is corrupt")
            return result
        hdr = _plane_header(nplanes)
        if len(payload) < hdr.size:
            raise ValueError("split chunk is truncated")
        lengths = hdr.unpack_from(payload, 0)
        mv = memoryview(payload)
        pos = hdr.size
        planes = []
        for k, ln in enumerate(lengths):
            if pos + ln > len(payload):
                raise ValueError("split chunk is truncated")
            m = (flags >> (2 * k)) & 0x3
            if m == entropy.METHOD_STORED:
                if ln != nelem:
                    raise ValueError(
                        f"plane {k} holds {ln} bytes, expected {nelem}")
                planes.append((payload, pos))
            else:
                block = _decode_stream(mv[pos:pos + ln], m, f"plane {k}", nelem)
                if len(block) != nelem:
                    raise CorruptArchive(
                        f"plane {k} decoded to {len(block)} bytes, expected {nelem}")
                planes.append((block, 0))
            pos += ln
        buf = out if (out is not None and len(out) >= rlen) else bytearray(rlen)
        if codec == CODEC_BF16:
            result = kernels.merge_bf16(planes, nelem, buf)
        else:
            result = kernels.merge_planes(planes, esize, nelem, buf)
    else:
        raise ValueError(f"unknown codec id {codec}")

    if verify and crc:
        if (zlib.crc32(result) & 0xFFFFFFFF) != crc:
            raise CorruptArchive("chunk failed its checksum; the archive is corrupt")
    return result
