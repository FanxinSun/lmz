"""Archive-level rANS frequency tables, shared across chunks by plane kind.

A rANS stream carries its own 516-byte frequency table. That is the right
default for one stream and the wrong one for an archive: a 10 MB perception
model at 64 KiB blocks is hundreds of chunks, each paying 516 bytes for a
table its neighbours would have been happy with, and on a GPU a per-stream
table is what stops thirty-two lanes of a warp sharing one copy in shared
memory.

The fix is not one table. Measured on real models, sharing a single table
across a whole archive is *six points worse* than what lmz does today,
because an exponent plane and a mantissa plane have nothing in common and
one table fits neither. What wins is sharing **within a plane kind, across
chunks** -- every byte-position-2 plane in the archive against one table --
which is worth +1.5 to +2.5 points on the same models.

So a table set is keyed by plane kind, carried in the manifest, and handed
to the coder on both sides. The streams themselves lose their headers and
nothing else: prepending a table to a headerless stream yields an ordinary
lmz stream, which is why there is no second decoder anywhere in this.

**Every table here is smoothed.** The counts are taken from a bounded sample
of the input rather than from all of it, so a symbol that appears only in an
unsampled chunk would otherwise have zero frequency -- and a coder cannot
represent a symbol it believes impossible, so `rans_encode_shared` would
refuse the chunk. Adding one to every symbol's count costs a fraction of a
bit against millions of real counts and makes the table total, which turns a
correctness cliff into an accounting rounding error.
"""

from __future__ import annotations

import base64
from math import log2

from . import kernels
from .codec import NOISE_BITS, SHARED_OVERHEAD

# How much of the input is read to build the tables. The sample is strided
# across the whole plan rather than taken from the front, because a
# checkpoint's first chunks are embeddings and its last are not, and a table
# fitted to embeddings alone would mis-price everything after them.
#
# 64 MiB is enough that the counts are stable -- a byte plane's distribution
# is settled long before this -- and small enough to disappear beside the
# read the encoder is about to do anyway. Below it, everything is sampled.
PRIME_BUDGET = 64 << 20


def key(cid: int, esize: int, plane: int) -> str:
    """The archive-wide name of one plane kind.

    A string because the manifest is JSON. `cid` is in it so a later codec
    can share the namespace without colliding with this one.
    """
    return f"{cid}:{esize}:{plane}"


class TableSet:
    """The archive's shared tables, by plane kind.

    Empty is the normal state: an archive written without shared tables has
    none, and `get` returning None is how every caller finds that out.
    """

    __slots__ = ("_t",)

    def __init__(self, tables: dict[str, bytes] | None = None):
        self._t = dict(tables or {})

    def __bool__(self) -> bool:
        return bool(self._t)

    def __len__(self) -> int:
        return len(self._t)

    def get(self, cid: int, esize: int, plane: int) -> bytes | None:
        return self._t.get(key(cid, esize, plane))

    def to_json(self) -> dict:
        return {k: base64.b64encode(v).decode("ascii")
                for k, v in sorted(self._t.items())}

    @staticmethod
    def from_json(d: dict | None) -> "TableSet":
        if not d:
            return TableSet()
        out = {}
        for k, v in d.items():
            try:
                raw = base64.b64decode(v, validate=True)
            except Exception:
                raise ValueError(f"shared table {k!r} is not valid base64")
            if len(raw) != kernels.RANS_HEADER:
                raise ValueError(
                    f"shared table {k!r} is {len(raw)} bytes, "
                    f"expected {kernels.RANS_HEADER}")
            out[k] = raw
        return TableSet(out)


class Primer:
    """Accumulates what priming learns, and decides whether sharing wins.

    Two numbers per plane kind, both in bits:

    - what the archive would pay coding every sampled plane against **one
      pooled table**. Cross-entropy against a fixed table is linear in the
      counts, so this is a function of the pooled histogram alone -- there is
      no need to keep a histogram per plane to work it out.
    - what it pays today, coding each plane against **its own** table: the
      plane's own order-0 entropy, plus 516 bytes of header per stream.

    The shared table is kept only where the first is smaller. That is the
    "measured cross-entropy rather than assumption" the handover asks for,
    and it costs nothing at encode time because it is settled here.
    """

    __slots__ = ("_counts", "_own_bits", "_streams")

    def __init__(self):
        self._counts: dict[str, list[int]] = {}
        self._own_bits: dict[str, float] = {}
        self._streams: dict[str, int] = {}

    def add(self, k: str, hist: list[int]) -> None:
        total = sum(hist)
        if not total:
            return
        own = _entropy_bits(hist, total)
        # A plane at the noise threshold is stored raw under either scheme, so
        # it belongs in neither column. Counting it was a real bug and not a
        # rounding one: an fp32 detector's low mantissa planes are genuine
        # noise, and crediting a shared table with the 516 bytes each of them
        # "would have" spent on a header credited it with saving something
        # nobody was paying. It predicted +1.9 points on ssdlite320 and
        # delivered nothing, because those planes were never coded at all.
        if own / total >= NOISE_BITS:
            return
        acc = self._counts.get(k)
        if acc is None:
            self._counts[k] = list(hist)
        else:
            for i, c in enumerate(hist):
                acc[i] += c
        # What the plane really costs today: its own stream and header, or the
        # raw bytes where coding cannot pay for them.
        self._own_bits[k] = self._own_bits.get(k, 0.0) + min(
            own + kernels.RANS_HEADER * 8, total * 8)
        self._streams[k] = self._streams.get(k, 0) + 1

    def build(self) -> "TableSet":
        """The tables worth carrying, smoothed and measured.

        A kind the kernel cannot normalise, or one where pooling loses, is
        simply absent: its chunks take the ordinary per-stream path, which is
        what `TableSet.get` returning None already means everywhere.
        """
        out = {}
        for k, hist in self._counts.items():
            table = kernels.rans_table([c + 1 for c in hist])
            if table is None:
                continue
            if self._streams[k] < 2:
                continue        # a table shared with nobody is not shared
            pooled = (_cross_entropy_bits(hist, table)
                      + self._streams[k] * SHARED_OVERHEAD * 8)
            # The table has to pay for its own passage: it is carried in the
            # manifest, which is 516 bytes the per-stream form never spends
            # twice. Without this, a kind that saves fifty bytes across two
            # streams is kept and makes the archive larger.
            if pooled + kernels.RANS_HEADER * 8 < self._own_bits[k]:
                out[k] = table
        return TableSet(out)

    def report(self) -> dict:
        """Per-kind bits both ways, for `lmz info` and for measuring this."""
        rows = {}
        for k, hist in self._counts.items():
            table = kernels.rans_table([c + 1 for c in hist])
            if table is None:
                continue
            rows[k] = {
                "streams": self._streams[k],
                "shared_bits": (_cross_entropy_bits(hist, table)
                                + self._streams[k] * SHARED_OVERHEAD * 8),
                "own_bits": self._own_bits[k],
            }
        return rows


def _entropy_bits(hist, total: int) -> float:
    """Order-0 entropy of one plane, in bits -- a floor on its own stream."""
    bits = 0.0
    for c in hist:
        if c:
            bits -= c * log2(c / total)
    return bits


def _cross_entropy_bits(hist, table) -> float:
    """Bits to code `hist` against the frequencies packed into `table`."""
    freqs = [table[4 + 2 * s] | (table[5 + 2 * s] << 8) for s in range(256)]
    scale = sum(freqs)
    if not scale:
        return float("inf")
    bits = 0.0
    for c, f in zip(hist, freqs):
        if c:
            if not f:                       # smoothing should make this dead
                return float("inf")
            bits -= c * log2(f / scale)
    return bits


def sample_plan(plan, budget: int = PRIME_BUDGET):
    """Which tasks to read when priming, strided across the whole plan.

    Returns indices into `plan`. A stride rather than a prefix because the
    front of a checkpoint is not representative of it; a prefix would fit the
    tables to the embedding matrix and mis-price every layer after it.
    """
    n = len(plan)
    if n == 0:
        return []
    total = sum(t[2] - t[1] for t in plan)
    if total <= budget:
        return list(range(n))
    # Take every `stride`-th task until the budget is met. Ceiling division so
    # the stride never comes out as zero on a plan of huge chunks.
    stride = max(1, (total + budget - 1) // budget)
    picked, seen = [], 0
    for i in range(0, n, stride):
        picked.append(i)
        seen += plan[i][2] - plan[i][1]
        if seen >= budget:
            break
    return picked
