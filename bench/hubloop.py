"""Run the lmz-versus-ZipLLM hub loop where the link is fast.

The measurement is cheap and the download is not. On the machine this was
written for, Hugging Face arrives at 0.06-0.5 MB/s, so a 268 MB model is
twenty minutes of waiting and about eight seconds of work. Anywhere with a
real link -- Colab, a VPS, a friend's desktop -- the same loop runs hundreds of
times faster, and nothing has to come back except the ledger: kilobytes per
model. On Colab in particular nobody downloads a model at all, because Colab
fetches from Hugging Face cloud to cloud, measures, records and deletes.

Two modes, one file:

    python hubloop.py --state ~/lmz-hub                  # any machine
    python hubloop.py --state /content/drive/MyDrive/lmz-hub --reverse

and in a Colab cell, `--setup` first to install lmz and its two helpers.

WHAT IT MEASURES, which is the whole point of keeping it identical to the loop
it was lifted from: for each fine-tune, both pipelines against the family's
base, differing only at the end. ZipLLM's is dedup by content hash, XOR delta,
zstd -1. lmz's is the shipped tool over the same two files. A fine-tune's own
cost is the pair's archive minus the base's. `verify_against_local()` checks a
pair of these numbers against the local implementation so the two ledgers stay
comparable; run it if you touch anything below the driver.

STATE lives in one directory, which can be Google Drive. Ledger, resume marks,
per-family hashes, the plan and the HOLD marker go there, so a Colab session
ending at the twelve-hour cap or on an idle disconnect loses at most the model
in flight. WEIGHTS never go there: they land in a scratch directory on the
ephemeral disk and are deleted after every model, one model at a time plus the
family's base.

TWO RUNNERS, ONE LEDGER. The local loop keeps going, so the queue is split
deterministically rather than shared: this runner takes families in reverse
plan order, the local loop keeps forward order, every row carries a `runner`
column, and each skips a family the other has already closed (from
`closed.json`, which is small enough to copy between them by hand). They meet
in the middle and stop; no family is measured twice.
"""
from __future__ import annotations

import argparse, csv, hashlib, json, mmap, os, shutil, struct, subprocess, sys
import tempfile, time, urllib.error, urllib.parse, urllib.request

LMZ_GIT = "https://github.com/FanxinSun/lmz.git"
UA = "lmz-hub-loop"
CHUNK = 8 << 20
RESIDUAL_SAMPLE_PCT = 1          # keep the coded residual for 1% of models
STALL_GIVEUP = 200

ESIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4,
         "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}

LEDGER_COLS = ["repo", "family", "base", "raw_bytes", "dedup_bytes",
               "zipllm_zstd1", "lmz_coder", "lmz_shipped", "fetch_s",
               "code_s", "lmz_commit", "in_z19_sample", "family_closed_by",
               "runner", "flags"]

# A re-upload -- every tensor already seen in this family -- that lmz still
# charges more than this for is the open question, not a result. The loop
# flags the row, writes the evidence once per family, and carries on: the
# campaign is worth more than a stopped loop, and a flagged row is a small
# identifiable subset to re-measure when the question is answered.
REUPLOAD_FREE_ENOUGH = 1 << 20

# Which limit closed a family, and so whether it is finished or merely out of
# time. K (the intended number of fine-tunes) and the 5% cap are answers about
# the hub; T (hours at the measured rate) and the probe ceiling are answers
# about the link, and a family closed by either is re-queued when the budget
# or the link grows.
EXTENDABLE = ("T", "probe")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ----------------------------------------------------------------- bootstrap

def resolve_ref(ref="main"):
    """The commit lmz will actually be installed from.

    A version string cannot tell two commits apart, and the whole value of
    this runner is that its ledger is comparable with the one the local loop
    is writing. So the sha is resolved once, installed by sha, and recorded in
    every row -- the same thing `git rev-parse --short HEAD` gives locally.
    """
    url = f"https://api.github.com/repos/FanxinSun/lmz/commits/{ref}"
    try:
        with urllib.request.urlopen(urllib.request.Request(
                url, headers={"User-Agent": UA,
                              "Accept": "application/vnd.github+json"}),
                timeout=30) as r:
            return json.load(r)["sha"][:7]
    except Exception as e:
        log(f"could not resolve {ref} ({e!r}); installing the branch instead")
        return None


def setup(ref="main", quiet=True):
    """Install what is missing. Safe to run twice; installs nothing it has.

    Only two things are needed. `zstandard` is what ZipLLM's half of the
    comparison compresses with before Python 3.14 carries zstd itself.
    huggingface_hub is deliberately NOT installed: nothing here uses it. The
    fetch is plain urllib with a length check, which is the code that has been
    tested against truncated bodies, and at Colab's link speed a faster client
    would save seconds on a job whose cost is compute.
    """
    sha = resolve_ref(ref)
    need = []
    try:
        import lmz  # noqa: F401
    except ImportError:
        need.append(f"git+{LMZ_GIT}@{sha}" if sha else f"git+{LMZ_GIT}")
    try:
        import zstandard  # noqa: F401
    except ImportError:
        try:
            from compression import zstd  # noqa: F401  (3.14 and later)
        except ImportError:
            need.append("zstandard")
    if need:
        log(f"installing: {' '.join(need)}")
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + need
        if quiet:
            cmd.insert(4, "-q")
        subprocess.run(cmd, check=True)
    else:
        log("nothing to install")
        if sha:
            # A warm runtime keeps whatever it already has. Say so, because a
            # pinned ref that silently did not take is how a campaign ends up
            # with rows attributed to a build that never ran: restarting the
            # runtime is what actually changes the version.
            have = lmz_version()
            if not have.startswith(sha[:7]):
                log(f"NOTE: lmz {have} is already installed and was kept; "
                    f"{sha} was requested. Restart the runtime to change it. "
                    f"The ledger records what is installed.")
    return sha


def token():
    """The user's Hugging Face token, from their keystrokes, never from here.

    Colab secrets first, then the environment, then the file huggingface_hub
    writes. It leaves only as an Authorization header: nothing in this file
    prints it, logs it, or writes it anywhere.
    """
    try:
        from google.colab import userdata          # type: ignore
        t = userdata.get("HF_TOKEN")
        if t:
            return t.strip()
    except Exception:
        pass
    t = os.environ.get("HF_TOKEN")
    if t:
        return t.strip()
    for p in ("~/.cache/huggingface/token", "~/.huggingface/token"):
        f = os.path.expanduser(p)
        try:
            if os.path.exists(f):
                s = open(f).read().strip()
                if s:
                    return s
        except OSError:
            pass
    return None


TOKEN = None


def headers():
    h = {"User-Agent": UA}
    if TOKEN:
        h["Authorization"] = f"Bearer {TOKEN}"
    return h


# --------------------------------------------------------------------- fetch

class GaveUp(Exception):
    """The bytes are not coming from this repo -- not that they come slowly."""


class ShortRead(Exception):
    """The body ended before the file did. Resume; do not accept it."""


def api(url, tries=4):
    for i in range(tries):
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers=headers()),
                    timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                return "gated"
            if e.code == 429:
                time.sleep(8 * (i + 1)); continue
            return None
        except Exception:
            time.sleep(4 * (i + 1))
    return None


def weight_files(model):
    """Names and sizes of the weight files, without fetching any of them."""
    tree = api(f"https://huggingface.co/api/models/"
               f"{urllib.parse.quote(model)}/tree/main")
    if tree == "gated":
        return "gated", 0
    if not isinstance(tree, list):
        return None, 0
    want = [f for f in tree if f["path"].endswith(".safetensors")]
    return ([(f["path"], f.get("size", 0)) for f in want],
            sum(f.get("size", 0) for f in want))


def fetch(model, path, dst, expect=0):
    """One file, resuming a partial, insisting on the length.

    urllib does not raise when a body is cut short: HTTPResponse.read returns
    b'' and closes the connection, deliberately, for compatibility. Reading
    until empty and renaming the result therefore accepts half a file as
    whole -- which is exactly what happened on the slow link this runner
    exists to escape, and it was reported as a correctness failure in lmz. So
    the length is checked against what the hub says before the file is
    accepted, and a short body resumes instead.
    """
    url = (f"https://huggingface.co/{model}/resolve/main/"
           f"{urllib.parse.quote(path)}")
    part = dst + ".part"
    t0 = last = time.time()
    barren = 0
    while barren < STALL_GIVEUP:
        before = os.path.getsize(part) if os.path.exists(part) else 0
        h = dict(headers())
        if before:
            h["Range"] = f"bytes={before}-"
        wait = 10
        try:
            with urllib.request.urlopen(
                    urllib.request.Request(url, headers=h), timeout=180) as r:
                resume = bool(before) and r.status == 206
                have = before if resume else 0
                clen = r.headers.get("Content-Length") or ""
                promised = have + int(clen) if clen.isdigit() else 0
                moved = 0
                with open(part, "ab" if resume else "wb") as f:
                    while True:
                        b = r.read(1 << 22)
                        if not b:
                            break
                        f.write(b); moved += len(b)
                        if time.time() - last >= 30:
                            last = time.time()
                            log(f"      {path}: {(have+moved)/1e6:.0f} MB "
                                f"({moved/max(time.time()-t0,1)/1e6:.1f} MB/s)")
            after = os.path.getsize(part)
            need = expect or promised
            if need and after < need:
                raise ShortRead(f"{after:,} of {need:,}")
            os.rename(part, dst)
            return os.path.getsize(dst), time.time() - t0
        except ShortRead as e:
            log(f"      {path}: short read, {e}; resuming"); wait = 5
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 404, 410):
                raise GaveUp(f"HTTP {e.code}")
            if e.code == 416:
                got = os.path.getsize(part) if os.path.exists(part) else 0
                if not expect or got == expect:
                    os.rename(part, dst)
                    return got, time.time() - t0
                os.unlink(part); wait = 5
            else:
                wait = 30 if e.code == 429 else 10
        except Exception:
            wait = 10
        after = os.path.getsize(part) if os.path.exists(part) else 0
        barren = 0 if after > before else barren + 1
        time.sleep(min(wait * (1 + barren // 4), 120))
    raise GaveUp(f"{STALL_GIVEUP} attempts moved no bytes")


def get_model(model, into):
    """Weights and config only. (bytes, seconds), or ('gated', 0)."""
    files, total = weight_files(model)
    if files == "gated":
        return "gated", 0
    if not files:
        return None, 0
    os.makedirs(into, exist_ok=True)
    log(f"    fetching {model}: {total/1e9:.3f} GB in {len(files)} file(s)")
    got, secs = 0, 0.0
    for name, want in files:
        d = os.path.join(into, os.path.basename(name))
        if os.path.exists(d):
            have = os.path.getsize(d)
            if not want or have >= want:
                got += have; continue
            os.replace(d, d + ".part")
        try:
            n, s = fetch(model, name, d, expect=want)
        except GaveUp as e:
            log(f"    {model}: {e}; skipping"); return None, 0
        except Exception as e:
            log(f"    {model}: fetch failed: {e!r}; skipping"); return None, 0
        got += n; secs += s
    try:
        cfg = os.path.join(into, "config.json")
        if not os.path.exists(cfg):
            n, s = fetch(model, "config.json", cfg); got += n; secs += s
    except Exception:
        pass
    return got, secs


# --------------------------------------------------------- the measurement
# Lifted verbatim from the local loop so the two ledgers are comparable.
# verify_against_local() below checks that claim rather than asserting it.

def tensors(path):
    f = open(path, "rb")
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    hlen = struct.unpack("<Q", mm[:8])[0]
    hdr = json.loads(mm[8:8 + hlen]); base = 8 + hlen
    out = {k: (v["dtype"], base + v["data_offsets"][0],
               base + v["data_offsets"][1])
           for k, v in hdr.items() if k != "__metadata__"}
    return f, mm, out


def safetensors_in(d):
    return sorted(os.path.join(d, x) for x in os.listdir(d)
                  if x.endswith(".safetensors"))


def zstd_size(buf, lvl=1):
    from lmz import entropy
    n, out, mv = len(buf), 0, memoryview(buf)
    for off in range(0, n, CHUNK):
        out += len(entropy.compress(bytes(mv[off:off + CHUNK]), lvl,
                                    entropy.METHOD_ZSTD))
    return out


def lmz_coder_size(buf, dtype, keep=None):
    """Bytes lmz's coder needs for this tensor; appends them to `keep` if given."""
    from lmz import codec
    from lmz.planner import KIND_BF16, KIND_BYTES
    es = ESIZE.get(dtype, 1)
    kind = KIND_BF16 if dtype == "BF16" else KIND_BYTES
    step = max(CHUNK - CHUNK % es, es)
    n, out, mv = len(buf), 0, memoryview(buf)
    for off in range(0, n, step):
        parts = codec.encode_chunk(bytes(mv[off:off + step]), es, 1, False,
                                   kind=kind)[0]
        out += sum(len(p) for p in parts)
        if keep is not None:
            keep.extend(parts)
    return out


def both_methods(base_dir, ft_dir, seen_hashes, residual=None):
    """One pass, two coders: dedup by content hash, XOR against the base, then
    zstd -1 for ZipLLM's method and lmz's coder for the comparison."""
    from lmz import kernels
    bt, handles = {}, []
    for p in safetensors_in(base_dir):
        f, mm, t = tensors(p); handles.append((f, mm))
        for k, v in t.items():
            bt[k] = (mm,) + v
    total = coded = lcoded = dedup = 0
    my_hashes = {}
    for p in safetensors_in(ft_dir):
        f, mm, t = tensors(p); handles.append((f, mm))
        for name, (dt, s, e) in t.items():
            raw = e - s
            total += raw
            buf = mm[s:e]
            h = hashlib.blake2b(buf, digest_size=16).hexdigest()
            my_hashes[name] = h
            if h in seen_hashes:
                dedup += raw
                continue
            tgt = buf
            if name in bt:
                bmm, bdt, bs, be = bt[name]
                if bdt == dt and (be - bs) == raw:
                    tgt = kernels.xor_bytes(buf, bmm[bs:be])
            b = bytes(tgt)
            coded += zstd_size(b, 1)
            lcoded += lmz_coder_size(b, dt, keep=residual)
    for f, mm in handles:
        mm.close(); f.close()
    return total, coded, lcoded, dedup, my_hashes


def stage_files(src, scratch, tag):
    """Weight files under names unique across the family, ready to archive."""
    st = os.path.join(scratch, "stage")
    shutil.rmtree(st, ignore_errors=True); os.makedirs(st)
    out = []
    for p in safetensors_in(src):
        dst = os.path.join(st, f"{tag}_{os.path.basename(p)}")
        try:
            os.link(p, dst)          # same filesystem: free
        except OSError:
            shutil.copy2(p, dst)     # a different mount: not
        out.append(dst)
    return st, out


def archive_start(base_dir, arc, scratch):
    """The family archive begins as the base alone."""
    if os.path.exists(arc):
        os.unlink(arc)
    st, _ = stage_files(base_dir, scratch, "000_base")
    subprocess.run([sys.executable, "-m", "lmz", "compress", st, arc, "-q"],
                   check=True)
    shutil.rmtree(st, ignore_errors=True)
    return os.path.getsize(arc)


def archive_append(ft_dir, arc, scratch, tag):
    """Add one model to the family archive; return the archive's new size.

    A fine-tune's cost is what it adds to the archive, not what a two-member
    archive of it and the base costs on its own. The difference is the whole
    comparison: ZipLLM's method is credited with dedup against every model
    already seen in the family, so measuring lmz pairwise denies it the same
    opportunity and charges it in full for a re-upload it would have stored
    once. That is what produced the first HOLD on Qwen3.5-0.8B-Base, where
    four of five fine-tunes were byte-identical re-uploads.

    One file at a time on purpose: `lmz append` deduplicates a new member
    against what the archive already holds, but not against other members of
    the same batch -- appending a directory of two identical files costs
    both -- so a model whose shards match each other would be charged twice.
    """
    st, files = stage_files(ft_dir, scratch, tag)
    for f in files:
        subprocess.run([sys.executable, "-m", "lmz", "append", arc, f, "-q"],
                       check=True)
    shutil.rmtree(st, ignore_errors=True)
    return os.path.getsize(arc)


LMZ_REF = None      # the commit lmz was installed from, once it is known


def check_member(arc, ft_dir, tag, scratch):
    """Prove the model just appended reads back, without re-reading the rest.

    `lmz verify` decodes the whole archive, so its cost grows with the family
    and the last model of a fifty-member family pays for all fifty. Measured
    on Colab that took a family's compute from 43 s a model to 274 s, all of
    it re-checking bytes that were already checked. Extracting the new member
    and comparing it with the file it came from is a stronger claim anyway --
    it proves the round trip, not just that the checksums agree -- and it
    costs one member. The whole archive is verified once, at family close.
    """
    out = os.path.join(scratch, "check")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    try:
        for p in safetensors_in(ft_dir):
            name = f"{tag}_{os.path.basename(p)}"
            dst = os.path.join(out, name)
            subprocess.run([sys.executable, "-m", "lmz", "extract", arc, name,
                            dst], check=True, capture_output=True)
            if os.path.getsize(dst) != os.path.getsize(p):
                raise ValueError(f"{name} came back a different length")
            with open(p, "rb") as a, open(dst, "rb") as b:
                while True:
                    x, y = a.read(1 << 22), b.read(1 << 22)
                    if x != y:
                        raise ValueError(f"{name} did not read back identically")
                    if not x:
                        break
            os.unlink(dst)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def lmz_version():
    """What goes in the ledger's lmz_commit column.

    What is INSTALLED, not what was asked for. `setup` installs nothing when
    lmz is already importable, so on a warm runtime a pinned --lmz-ref would
    otherwise be written into every row while an older build did the work: a
    ledger claiming a provenance it does not have, which is worse than none.
    pip records the commit it resolved a VCS install from in direct_url.json,
    so ask that first; the version string and the requested ref are fallbacks
    for a build with no such record.
    """
    try:
        from importlib.metadata import distribution
        info = json.loads(distribution("lmzip").read_text("direct_url.json"))
        sha = (info.get("vcs_info") or {}).get("commit_id")
        if sha:
            return sha[:7]
    except Exception:
        pass
    try:
        from importlib.metadata import version
        return f"lmzip-{version('lmzip')}"
    except Exception:
        pass
    return LMZ_REF or "lmzip-?"


# ------------------------------------------------------------------- state

class State:
    """Everything that must survive a session ending, in one directory."""

    def __init__(self, root, runner):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.runner = runner
        os.makedirs(self.root, exist_ok=True)
        os.makedirs(os.path.join(self.root, "hashes"), exist_ok=True)
        os.makedirs(os.path.join(self.root, "residuals"), exist_ok=True)
        self.ledger = os.path.join(self.root, "ledger.csv")
        self.resume = os.path.join(self.root, "resume.json")
        self.plan = os.path.join(self.root, "plan.json")
        self.planmeta = os.path.join(self.root, "plan.meta.json")
        self.hold = os.path.join(self.root, "HOLD")
        self.closed = os.path.join(self.root, "closed.json")
        self.st = self._load()

    def _load(self):
        st = {"bytes": 0, "secs": 0.0, "code_s": 0.0, "models": 0,
              "families": 0, "gated": 0, "raw": 0, "zstd": 0, "lmzc": 0,
              "shipped": 0, "dedup": 0, "cur_fam": None, "ft": 0,
              "fam_hist": {}, "fam_closed": {}, "unavailable": [],
              # The family archive this run is growing, and its size before
              # the model in flight: a fine-tune's cost is what it adds.
              "fam_arc": None, "prev": 0,
              # Whole-file digests, so "the same file" is a claim about the
              # bytes on disk rather than about tensor ranges hashed earlier.
              "digests": {}, "fam_flags": {}, "fam_evidence": {}}
        try:
            st.update(json.load(open(self.resume)))
        except Exception:
            pass
        for k, v in (("fam_hist", {}), ("fam_closed", {}),
                     ("unavailable", []), ("cur_fam", None), ("ft", 0),
                     ("fam_arc", None), ("prev", 0), ("digests", {}),
                     ("fam_flags", {}), ("fam_evidence", {})):
            st.setdefault(k, v)
        return st

    def drop_family(self, base):
        """Forget a family entirely: its rows AND its dedup hashes.

        The two must go together. They are the cumulative state of the two
        methods being compared -- the seen-hash set for ZipLLM's, the family
        archive for lmz's -- so leaving one behind scores the two on different
        inputs. Dropping the rows while keeping the hashes makes every model
        look like a duplicate to ZipLLM's method while lmz, starting from an
        empty archive, is charged in full: the exact mirror of the pairwise
        mistake this replaced, and it produced `zstd 0.000` against
        `shipped 1.135` on the first family it touched.
        """
        h = os.path.join(self.root, "hashes", base.replace("/", "__") + ".json")
        if os.path.exists(h):
            os.unlink(h)
        try:
            rows = list(csv.reader(open(self.ledger)))
        except Exception:
            return
        if not rows:
            return
        keep = [rows[0]] + [r for r in rows[1:] if len(r) < 2 or r[1] != base]
        dropped = len(rows) - len(keep)
        if dropped:
            with open(self.ledger + ".tmp", "w", newline="") as fh:
                csv.writer(fh).writerows(keep)
            os.replace(self.ledger + ".tmp", self.ledger)
        log(f"  forgot {base}: {dropped} rows and its hashes")

    def save(self):
        tmp = self.resume + ".tmp"
        json.dump(self.st, open(tmp, "w")); os.replace(tmp, self.resume)
        # closed.json is the small file the two runners exchange by hand.
        json.dump(self.st["fam_closed"], open(self.closed + ".tmp", "w"),
                  indent=1)
        os.replace(self.closed + ".tmp", self.closed)

    def progress(self, planned=0, note=""):
        """A glanceable summary in the state directory, updated every model.

        So the run can be followed by opening one small file in Drive rather
        than by pasting a log back to whoever is watching.
        """
        st = self.st
        pct = lambda a, b: round((1 - a / b) * 100, 2) if b else None
        done = [b for b, w in st["fam_closed"].items()
                if w in ("K", "cap", "exhausted")]
        tl = [b for b, w in st["fam_closed"].items() if w in EXTENDABLE]
        rate = st["bytes"] / max(st["secs"], 1)
        p = {
            "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runner": self.runner,
            "families_done": len(done),
            "families_time_limited": len(tl),
            "families_flagged": sorted(st["fam_flags"]),
            "models_measured": st["models"],
            "gated_or_unavailable": st["gated"],
            "gb_fetched": round(st["bytes"] / 1e9, 2),
            "fetch_mb_s": round(rate / 1e6, 2),
            "code_s_per_model": round(st["code_s"] / max(st["models"], 1), 1),
            "current_family": st["cur_fam"],
            "current_index": st["ft"],
            "zipllm_pct": pct(st["zstd"], st["raw"]),
            "lmz_coder_pct": pct(st["lmzc"], st["raw"]),
            "lmz_shipped_pct": pct(st["shipped"], st["raw"]),
            "holding": os.path.exists(self.hold),
            "note": note,
        }
        if planned and rate > 0:
            p["days_left_at_this_rate"] = round(
                (planned - st["bytes"]) / rate / 86400, 1)
        try:
            tmp = os.path.join(self.root, "progress.json.tmp")
            json.dump(p, open(tmp, "w"), indent=1)
            os.replace(tmp, os.path.join(self.root, "progress.json"))
        except Exception:
            pass

    def seen_load(self, base):
        p = os.path.join(self.root, "hashes", base.replace("/", "__") + ".json")
        try:
            return json.load(open(p))
        except Exception:
            return {}

    def seen_save(self, base, seen):
        p = os.path.join(self.root, "hashes", base.replace("/", "__") + ".json")
        json.dump(seen, open(p + ".tmp", "w")); os.replace(p + ".tmp", p)

    def measured(self):
        try:
            with open(self.ledger) as fh:
                return {r[0] for r in csv.reader(fh)} - {"repo"}
        except Exception:
            return set()

    def row(self, row):
        new = not os.path.exists(self.ledger)
        with open(self.ledger, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(LEDGER_COLS)
            w.writerow(row)


# ------------------------------------------------------------------ driver

def memory_gb():
    try:
        m = {l.split(":")[0]: l.split()[1] for l in open("/proc/meminfo")}
        return round(int(m["MemAvailable"]) / 1e6, 1)
    except Exception:
        return 0.0


def write_reupload_evidence(state, base, repo, arc, twin, digest, ft_dir, cost):
    """Everything needed to explain one expensive re-upload, written once.

    This used to be a diagnostic the user ran in a cell of its own. It belongs
    in the loop: the campaign is measuring anyway, the evidence is kilobytes,
    and it arrives with the ledger instead of costing a round.
    """
    try:
        from lmz.format import ArchiveReader
        with open(arc, "rb") as fh:
            r = ArchiveReader(fh)
        members, chunks = r.members, list(r.chunks)

        def chunks_of(m):
            return [c for c in chunks if m.dst <= c.dst < m.dst + m.size]

        rows = {}
        for m in members[-2:]:
            rows[m.path] = [[c.codec, c.esize, c.flags, c.rlen, c.crc, c.clen,
                             c.off] for c in chunks_of(m)]
        doctor = subprocess.run([sys.executable, "-m", "lmz", "doctor"],
                                capture_output=True, text=True).stdout
        b = safetensors_in(ft_dir)[0]
        ev = {
            "repo": repo, "family": base, "cost": cost,
            # Byte identity on disk, which is not what the dedup column
            # claims. That column compares tensor ranges and never looks at
            # the header, and it can only compare what it hashed. This is a
            # digest of the whole file as it sits on disk now, so it separates
            # "lmz is wrong about a duplicate" from "these are not the
            # duplicates the ledger thinks they are".
            "file_digest": digest,
            "same_file_on_disk_as": twin,
            "identical_on_disk": twin is not None,
            "size": os.path.getsize(b),
            "doctor": doctor, "mem_available_gb": memory_gb(),
            "cpus": os.cpu_count(), "lmz": lmz_version(),
            "members": [m.path for m in members], "chunks": rows,
        }
        p = os.path.join(state.root, "reupload_evidence.json")
        old = []
        if os.path.exists(p):
            try:
                old = json.load(open(p))
            except Exception:
                old = []
        old.append(ev)
        json.dump(old[-8:], open(p + ".tmp", "w"))
        os.replace(p + ".tmp", p)
        log(f"    wrote the re-upload evidence "
            f"(identical on disk: {ev['identical_on_disk']}, "
            f"{ev['mem_available_gb']} GB free)")
    except Exception as e:
        log(f"    could not write the re-upload evidence: {e!r}")


def keeps_residual(repo):
    """A fixed sample by hash, so the same models are kept on every runner."""
    n = int(hashlib.blake2b(repo.encode(), digest_size=8).hexdigest(), 16)
    return n % 100 < RESIDUAL_SAMPLE_PCT


def hold(state, base, criterion, numbers):
    with open(state.hold, "w") as fh:
        fh.write(f"family: {base}\ncriterion: {criterion}\n"
                 f"runner: {state.runner}\n")
        for k, v in numbers.items():
            fh.write(f"{k}: {v}\n")
        fh.write(f"written: {time.strftime('%Y-%m-%d %H:%M')}\n")
        fh.write("the fix belongs in lmz's tree, not here; this runner "
                 "refetches the family when it is told to continue\n")
    log(f"HOLD written: {base} — {criterion}")


def run(state, scratch, reverse, others_closed, max_models=None):
    fams = json.load(open(state.plan))
    try:
        closed_by = json.load(open(state.planmeta)).get("closed_by", {})
    except Exception:
        closed_by = {}
    try:
        planned = json.load(open(state.planmeta)).get("total_bytes", 0)
    except Exception:
        planned = 0
    order = list(fams.items())
    if reverse:
        order.reverse()
    st = state.st
    covered = state.measured() | set(st["unavailable"])
    if st["cur_fam"] not in fams:
        st["cur_fam"], st["ft"] = None, 0
    log(f"runner {state.runner}: {len(order)} families, "
        f"{'reverse' if reverse else 'forward'} order; "
        f"{len(st['fam_closed'])} closed here, {st['models']} models measured")
    done_models = 0
    for base, fts in order:
        if os.path.exists(state.hold):
            log("HOLD present; parked"); return
        if base in others_closed:
            # Reverse meets forward. Stopping rather than skipping is the
            # point: everything past here belongs to the other runner.
            log(f"met the other runner at {base}; stopping. "
                f"Refresh closed.json from it to go further.")
            return
        if st["fam_closed"].get(base) == "base unavailable":
            continue
        if base in st["fam_closed"] and all(f in covered for f in fts):
            continue
        if st["cur_fam"] != base:
            st["cur_fam"], st["ft"] = base, 0
            shutil.rmtree(scratch, ignore_errors=True)
        os.makedirs(scratch, exist_ok=True)
        bdir = os.path.join(scratch, "base")
        got, secs = get_model(base, bdir)
        if got == "gated" or got is None:
            log(f"  base {base} unavailable; skipping the family")
            st["gated"] += 1
            st["fam_closed"][base] = "base unavailable"
            st["cur_fam"] = None; state.save(); continue
        st["bytes"] += got or 0; st["secs"] += secs
        # The family archive lives on the scratch disk, not in the state
        # directory: it is the only large thing here and Drive is small. It
        # therefore does not survive a disconnect, so a family interrupted
        # part-way is measured again from its first fine-tune and its earlier
        # rows are dropped. Families are bounded by K and the time budget --
        # under an hour of Colab at the rates seen so far -- so this costs a
        # family at worst, and it keeps the incremental costs consistent with
        # one archive rather than stitched across two.
        arc = os.path.join(scratch, "family.lmz")
        if st.get("fam_arc") != base or not os.path.exists(arc):
            # A fresh archive means lmz starts from nothing, so ZipLLM's
            # method must too: whatever this family has already recorded is
            # dropped, rows and hashes together, and it is measured again from
            # its first fine-tune. The condition tests the hashes as well as
            # the cursor, because a session that died between families leaves
            # ft at zero with a full hash file behind it.
            if st["ft"] or state.seen_load(base):
                log(f"  the family archive is gone; remeasuring {base} from "
                    f"its first fine-tune")
                state.drop_family(base)
                covered = state.measured() | set(st["unavailable"])
                st["ft"] = 0
                st["fam_hist"].pop(base, None)
            st["prev"] = archive_start(bdir, arc, scratch)
            st["fam_arc"] = base
        seen = state.seen_load(base)
        acc = st["fam_hist"].setdefault(base, {})
        why = closed_by.get(base, "?")
        log(f"family {base}: {len(fts)} fine-tunes, closed by {why}, "
            f"from index {st['ft']}")
        while st["ft"] < len(fts):
            ft = fts[st["ft"]]
            if ft in covered:
                st["ft"] += 1; state.save(); continue
            fdir = os.path.join(scratch, "ft")
            shutil.rmtree(fdir, ignore_errors=True)
            got, secs = get_model(ft, fdir)
            if got == "gated" or got is None:
                st["gated"] += 1
                st["unavailable"] = sorted(set(st["unavailable"]) | {ft})
                covered.add(ft)
                st["ft"] += 1; state.save(); continue
            st["bytes"] += got; st["secs"] += secs
            residual = [] if keeps_residual(ft) else None
            t0 = time.time()
            try:
                raw, z, lc, dd, hs = both_methods(bdir, fdir, seen, residual)
                now = archive_append(fdir, arc, scratch, f"{st['ft']:03d}")
                check_member(arc, fdir, f"{st['ft']:03d}", scratch)
            except Exception as e:
                hold(state, base, "correctness failure",
                     {"model": ft, "error": repr(e)})
                state.save(); return
            code_s = time.time() - t0
            shipped = max(now - st["prev"], 0)
            st["prev"] = now
            if residual:
                p = os.path.join(state.root, "residuals",
                                 ft.replace("/", "__") + ".lmzres")
                with open(p, "wb") as fh:
                    for part in residual:
                        fh.write(part)
                log(f"    kept the coded residual ({sum(map(len, residual))/1e6:.0f} MB)")
            # A whole-file digest, so "these are the same file" is a claim
            # about the bytes on disk now rather than about tensor ranges
            # hashed at some earlier point. One sequential read.
            h = hashlib.blake2b(digest_size=16)
            with open(safetensors_in(fdir)[0], "rb") as fh:
                for blk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(blk)
            digest = h.hexdigest()
            twin = next((r for r, d in st["digests"].items()
                         if d == digest and r != ft), None)
            st["digests"][ft] = digest

            flags = []
            reupload = dd >= raw > 0
            if reupload:
                flags.append("reupload")
                if shipped > REUPLOAD_FREE_ENOUGH:
                    # The open question, recorded rather than stopped on. The
                    # evidence is written once per family, with the twin's own
                    # file if we still have it.
                    flags.append("reupload_not_shared")
                    if not st["fam_evidence"].get(base):
                        st["fam_evidence"][base] = True
                        write_reupload_evidence(state, base, ft, arc, twin,
                                                digest, fdir, shipped)
                    log(f"    flagged: a re-upload cost {shipped/1e9:.3f} GB; "
                        f"same file on disk as {twin or 'nothing seen'}")
            if twin and not reupload:
                flags.append("same_file_not_deduped")
            seen.update({v: k for k, v in hs.items()})
            state.seen_save(base, seen)
            state.row([ft, base, base, raw, dd, z, lc, shipped,
                       f"{secs:.1f}", f"{code_s:.1f}", lmz_version(), 0, why,
                       state.runner, "|".join(flags)])
            for k, tot in (("raw", raw), ("zstd", z), ("lmzc", lc),
                           ("shipped", shipped), ("dedup", dd)):
                st[k] += tot
                acc[k] = acc.get(k, 0) + tot
            acc["n"] = acc.get("n", 0) + 1
            st["code_s"] += code_s
            covered.add(ft)
            st["models"] += 1; st["ft"] += 1
            log(f"  {ft}: raw {raw/1e9:.2f} GB  zstd {z/1e9:.3f}  "
                f"coder {lc/1e9:.3f}  shipped {shipped/1e9:.3f}  "
                f"({secs:.0f}s fetch, {code_s:.0f}s code)")
            shutil.rmtree(fdir, ignore_errors=True)
            state.save()
            state.progress(planned)
            done_models += 1
            if max_models and done_models >= max_models:
                log(f"reached --max-models {max_models}"); return
        # Once per family, not once per model: the whole archive is decoded
        # and checksummed, which is the check the per-model one is a bounded
        # stand-in for.
        try:
            subprocess.run([sys.executable, "-m", "lmz", "verify", arc],
                           check=True, capture_output=True)
        except Exception as e:
            hold(state, base, "family archive failed verification",
                 {"error": repr(e)})
            state.save(); return
        # A disappointing ratio is a result; it is not a reason to stop. HOLD
        # is kept for an answer that would be WRONG -- a failed decode, a
        # failed verify -- because measuring on past one of those is worthless.
        # A family where lmz trails is flagged, recorded, and continued past:
        # the campaign is worth more than a stopped loop, and the flagged rows
        # are a small identifiable subset to re-measure once the open question
        # is answered.
        if acc.get("raw"):
            sh = (1 - acc["shipped"] / acc["raw"]) * 100
            zp = (1 - acc["zstd"] / acc["raw"]) * 100
            lp = (1 - acc["lmzc"] / acc["raw"]) * 100
            fl = []
            if sh < zp:
                fl.append("below_zipllm")
            if lp - sh > 1.0:
                fl.append("below_own_coder")
            if fl:
                st["fam_flags"][base] = {
                    "flags": fl, "shipped_pct": round(sh, 2),
                    "coder_pct": round(lp, 2), "zipllm_pct": round(zp, 2)}
                log(f"  family {base} FLAGGED ({', '.join(fl)}): "
                    f"shipped {sh:.2f}%  coder {lp:.2f}%  zipllm {zp:.2f}% "
                    f"— recorded, continuing")
            else:
                log(f"  family {base}: shipped {sh:.2f}%  coder {lp:.2f}%  "
                    f"zipllm {zp:.2f}%")
        st["families"] += 1
        st["fam_closed"][base] = why
        st["cur_fam"], st["ft"] = None, 0
        shutil.rmtree(scratch, ignore_errors=True)
        state.save()
        state.progress(planned, f"finished {base}")
    log("reached the end of the plan for this runner")


def summary(state):
    st = state.st
    pct = lambda a, b: (1 - a / b) * 100 if b else 0.0
    log(f"--- {state.runner}: {st['models']} models in {st['families']} "
        f"families, {st['bytes']/1e9:.1f} GB fetched")
    if st["raw"]:
        log(f"    ZipLLM's method {pct(st['zstd'], st['raw']):.2f}%   "
            f"lmz's coder {pct(st['lmzc'], st['raw']):.2f}%   "
            f"lmz as shipped {pct(st['shipped'], st['raw']):.2f}%")
        log(f"    {st['secs']/max(st['models'],1):.0f}s fetch and "
            f"{st['code_s']/max(st['models'],1):.0f}s compute per model "
            f"({st['bytes']/max(st['secs'],1)/1e6:.0f} MB/s)")


def verify_against_local(base_dir, ft_dir):
    """Check this file's measurement against the local loop's, same inputs.

    The two implementations are copies, so they can drift; this is the check
    that says they have not. Only runs where the local loop exists.
    """
    sys.path.insert(0, os.path.expanduser("~/hub-loop"))
    import loop as local                                   # noqa: E402
    a = both_methods(base_dir, ft_dir, {})
    b = local.both_methods(base_dir, ft_dir, {})
    same = a[:4] == b[:4] and a[4] == b[4]
    print(("  ok   " if same else "  FAIL ") +
          f"offsite {a[:4]} vs local {b[:4]}")
    return same


def selftest():
    """Run every stage here, with no network, before anything reaches a user.

    The campaign has cost seven Colab rounds, and five of them were spent on
    things that could have been caught on this machine: a token a subprocess
    could not see, a path with a space, a plan already inside the state
    directory, a measurement that scored the two methods on different inputs,
    and the mirror of that measurement. Each was a round of someone else's
    time. This exercises the same stages against bytes already on disk, so
    the next failure of that kind is found here.

    Returns the number of checks that failed.
    """
    bad = []

    def ok(cond, what):
        print(("  ok   " if cond else "  FAIL ") + what)
        if not cond:
            bad.append(what)

    root = tempfile.mkdtemp()
    # A directory with a space in it, because the user's Drive folder is
    # "Colab Notebooks" and an unquoted path failed there once already.
    state_dir = os.path.join(root, "Colab Notebooks", "offsite")
    os.makedirs(state_dir)
    scratch = os.path.join(root, "scratch")

    # 1. The token as a SUBPROCESS sees it. `!python` gets no Colab kernel
    #    channel, so a secret read only in the notebook is invisible here.
    env = dict(os.environ, HF_TOKEN="selftest-not-a-real-token")
    # Imported by whatever this file is actually called: the cell that fetches
    # it chooses the name, and a check that only passes under one of them is
    # not a check.
    here = os.path.abspath(__file__)
    r = subprocess.run(
        [sys.executable, "-c",
         "import sys, importlib; sys.path.insert(0, %r); "
         "m = importlib.import_module(%r); "
         "print('SEEN' if m.token() else 'MISSING')"
         % (os.path.dirname(here),
            os.path.splitext(os.path.basename(here))[0])],
        capture_output=True, text=True, env=env)
    ok("SEEN" in r.stdout, "a subprocess sees HF_TOKEN from the environment")

    # 2. Real bytes. The distilbert base fetched by the local loop if it is
    #    here, else something with the same shape, plus a fine-tune of it and
    #    a byte-identical re-upload of that.
    real = os.path.expanduser("~/hub-loop/raw/base/model.safetensors")
    repos = os.path.join(root, "repos")
    for d in ("base", "ft", "copy"):
        os.makedirs(os.path.join(repos, d))
    if os.path.exists(real):
        shutil.copy(real, os.path.join(repos, "base", "model.safetensors"))
        src = "the real distilbert base on disk"
    else:
        _synth_model(os.path.join(repos, "base", "model.safetensors"), 1)
        src = "a synthetic stand-in (no real base on disk)"
    _nudge(os.path.join(repos, "base", "model.safetensors"),
           os.path.join(repos, "ft", "model.safetensors"))
    shutil.copy(os.path.join(repos, "ft", "model.safetensors"),
                os.path.join(repos, "copy", "model.safetensors"))
    print(f"  using {src}")

    # 3. A plan that lives INSIDE the state directory, which is what a user
    #    naturally does and what once raised SameFileError.
    plan = {"org/base": ["org/ft", "org/copy"]}
    json.dump(plan, open(os.path.join(state_dir, "plan.json"), "w"))
    json.dump({"closed_by": {"org/base": "K"}, "total_bytes": 3 << 30},
              open(os.path.join(state_dir, "plan.meta.json"), "w"))

    # 4. The driver, with the network replaced by a copy from disk.
    global TOKEN
    TOKEN = "selftest"
    real_get = globals()["get_model"]
    where = {"org/base": "base", "org/ft": "ft", "org/copy": "copy"}

    def fake_get(model, into, *a, **k):
        os.makedirs(into, exist_ok=True)
        src_dir = os.path.join(repos, where[model])
        n = 0
        for f in os.listdir(src_dir):
            shutil.copy(os.path.join(src_dir, f), os.path.join(into, f))
            n += os.path.getsize(os.path.join(into, f))
        return n, 0.1

    globals()["get_model"] = fake_get
    try:
        st = State(state_dir, "selftest")
        ok(os.path.exists(st.plan), "the plan is found inside the state dir")
        # Stop after one model, then resume: the interrupt case.
        run(st, scratch, reverse=False, others_closed=set(), max_models=1)
        rows1 = list(csv.reader(open(st.ledger)))
        ok(len(rows1) == 2, f"one model measured and written ({len(rows1)-1})")
        ok(os.path.exists(os.path.join(state_dir, "progress.json")),
           "progress.json is written")

        st2 = State(state_dir, "selftest")          # a fresh process would
        run(st2, scratch, reverse=False, others_closed=set())
        rows2 = list(csv.reader(open(st2.ledger)))
        ok(rows2[0] == LEDGER_COLS,
           f"the ledger has all {len(LEDGER_COLS)} columns")
        ok(len(rows2) == 3, f"resume added the rest without repeating "
                            f"({len(rows2)-1} rows)")
        seen_repos = [r[0] for r in rows2[1:]]
        ok(len(set(seen_repos)) == len(seen_repos), "no row is duplicated")

        by = {r[0]: r for r in rows2[1:]}
        cp = by.get("org/copy")
        ok(cp is not None, "the re-upload was measured")
        if cp:
            i = LEDGER_COLS.index("flags")
            ok("reupload" in cp[i], f"the re-upload is flagged ({cp[i]!r})")
            shipped = int(cp[LEDGER_COLS.index("lmz_shipped")])
            ok(shipped < REUPLOAD_FREE_ENOUGH,
               f"and it costs a table entry, not a copy ({shipped:,} bytes)")
        ok(os.path.exists(st2.closed), "closed.json is written")
        prog = json.load(open(os.path.join(state_dir, "progress.json")))
        ok(prog["models_measured"] == 2,
           f"progress.json counts the models ({prog['models_measured']})")
        ok(not os.path.exists(st2.hold),
           "nothing HOLDs: a ratio is a result, not a stop")
    finally:
        globals()["get_model"] = real_get
        shutil.rmtree(root, ignore_errors=True)
        shutil.rmtree(scratch, ignore_errors=True)

    print(f"\n{'SELFTEST PASSED' if not bad else 'SELFTEST FAILED: ' + '; '.join(bad)}")
    return len(bad)


def _synth_model(path, seed, mb=120, ntensors=105):
    """Only used when no real checkpoint is on disk: a real one is better."""
    import random
    rng = random.Random(seed)
    per = int(mb * 1e6) // ntensors
    hdr, off, blobs = {}, 0, []
    for i in range(ntensors):
        n = 3072 if i % 3 else per          # real files are mostly tiny tensors
        raw = bytes(rng.getrandbits(8) for _ in range(min(n, 4096)))
        raw = (raw * (n // len(raw) + 1))[:n]
        hdr[f"layer.{i}.weight"] = {"dtype": "F32", "shape": [n // 4],
                                    "data_offsets": [off, off + n]}
        blobs.append(raw); off += n
    hdr["__metadata__"] = {"format": "pt"}
    b = json.dumps(hdr, separators=(",", ":")).encode()
    b += b" " * ((-len(b)) % 8)
    open(path, "wb").write(struct.pack("<Q", len(b)) + b + b"".join(blobs))


def _nudge(src, dst):
    """A fine-tune of `src`: its largest tensors moved a little, in place."""
    raw = bytearray(open(src, "rb").read())
    hlen = struct.unpack("<Q", bytes(raw[:8]))[0]
    hdr = json.loads(bytes(raw[8:8 + hlen]))
    body = 8 + hlen
    t = {k: v for k, v in hdr.items() if k != "__metadata__"}
    big = sorted(t.items(), key=lambda kv: -(kv[1]["data_offsets"][1]
                                             - kv[1]["data_offsets"][0]))[:3]
    for _name, meta in big:
        s, e = meta["data_offsets"]
        for i in range(body + s, body + e, 97):
            raw[i] = (raw[i] + 1) & 0xFF
    open(dst, "wb").write(bytes(raw))


def main():
    global TOKEN, LMZ_REF
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--state", required=True,
                   help="directory that survives the session; Drive is fine")
    p.add_argument("--scratch", default=None,
                   help="ephemeral disk for weights (default: /content or tmp)")
    p.add_argument("--runner", default="offsite", help="name for the ledger")
    p.add_argument("--reverse", action="store_true",
                   help="take families in reverse plan order (the offsite half)")
    p.add_argument("--setup", action="store_true", help="install lmz and zstandard")
    p.add_argument("--lmz-ref", default="main",
                   help="branch or sha of lmz to install (default: main)")
    p.add_argument("--plan", default=None, help="seed the state dir with a plan")
    p.add_argument("--others-closed", default=None,
                   help="the other runner's closed.json")
    p.add_argument("--max-models", type=int, default=None)
    p.add_argument("--verify", nargs=2, metavar=("BASE_DIR", "FT_DIR"),
                   help="compare against the local loop and exit")
    p.add_argument("--selftest", action="store_true",
                   help="run every stage here, with no network, and exit")
    a = p.parse_args()

    if a.setup:
        LMZ_REF = setup(a.lmz_ref)
    if a.selftest:
        raise SystemExit(1 if selftest() else 0)
    if a.verify:
        raise SystemExit(0 if verify_against_local(*a.verify) else 1)

    TOKEN = token()
    log("token: " + ("loaded" if TOKEN else "NOT FOUND — gated repos will be "
                     "skipped; put HF_TOKEN in Colab secrets"))
    state = State(a.state, a.runner)
    # The sha is remembered in the state directory, so a session that resumes
    # without --setup still records what it is measuring with rather than
    # falling back to a version string that cannot tell two commits apart.
    reffile = os.path.join(state.root, "lmz_ref.txt")
    if LMZ_REF:
        open(reffile, "w").write(LMZ_REF)
    elif os.path.exists(reffile):
        LMZ_REF = open(reffile).read().strip() or None
    if a.plan:
        # The natural thing to do is drop the plan straight into the state
        # directory and then also pass --plan pointing at it, which asks the
        # copy to write a file onto itself. shutil raises SameFileError for
        # that, so say it is already there and carry on.
        if os.path.abspath(a.plan) == os.path.abspath(state.plan):
            log("the plan is already in the state directory; nothing to seed")
        else:
            shutil.copy(a.plan, state.plan)
            meta = os.path.join(os.path.dirname(a.plan), "plan.meta.json")
            if os.path.exists(meta) and \
                    os.path.abspath(meta) != os.path.abspath(state.planmeta):
                shutil.copy(meta, state.planmeta)
            log(f"plan seeded from {a.plan}")
    if not os.path.exists(state.plan):
        raise SystemExit(f"no plan at {state.plan}; pass --plan once to seed it")
    scratch = a.scratch or ("/content/hubloop-scratch"
                            if os.path.isdir("/content")
                            else os.path.join(tempfile.gettempdir(), "hubloop"))
    others = {}
    if a.others_closed and os.path.exists(a.others_closed):
        others = json.load(open(a.others_closed))
        log(f"the other runner has closed {len(others)} families")
    log(f"state {state.root}   scratch {scratch}   lmz {lmz_version()}")
    try:
        run(state, scratch, a.reverse, set(others), a.max_models)
    finally:
        state.save()
        summary(state)
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
