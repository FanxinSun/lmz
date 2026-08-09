# Using lmz

*The command line, the Python API, reading a model without expanding it, and the compressed filesystem.*

[← back to the README](../README.md)

## Reading a model without expanding it

Compressing a model is only half of an answer. The other half is that
something has to *read* it, and until now that meant expanding the archive
back onto the disk you were trying to save — paying the full size again,
transiently, before the first token.

`lmz mount` removes that step. Models live in a store, compressed; the mount
presents them as the ordinary files a runtime expects, and decodes the blocks
a reader actually touches on the way out.

```
lmz add ./Llama-3.1-8B-Instruct        # compressed once, into the store
lmz models                             # what is there, and what it costs
lmz mount ~/models                     # served as ordinary files
llama-cli -m ~/models/Llama-3.1-8B-Instruct/model.gguf
```

Nothing is patched and nothing links against lmz. `llama.cpp`, vLLM and
`transformers` call `open()`, `read()` and `mmap()` and get the original
bytes, because that is all the mount is: a filesystem the kernel talks to.

There is no libfuse and nothing to install. `fusermount3` performs the
privileged mount and passes back a descriptor over a Unix socket; after that
the kernel's FUSE protocol is a sequence of fixed structs, so the whole
server is `struct.pack` and `os.read` — the same standard-library-only rule
the rest of the package follows.

### What it costs

Measured on the 1.74 GiB BF16 model above, AMD Ryzen 7 9800X3D, reading
through the mount by concurrent readers.

| server | 1 reader | 2 | 4 | 16 |
|---|---|---|---|---|
| CPython 3.14 (GIL) | 602 | 689 | 694 | 813 |
| **free-threaded 3.14t** | **1476** | **1478** | **2490** | **3000** |
| plain uncompressed file | 5492 | 43571 | 47104 | 23624 |
| lmz mount, page cache warm | 24135 | | | |

**The mount is still not faster than reading an uncompressed file on this
machine, and the table says so.** But it is now 2.8x faster than it was for a
lone sequential reader and 4.4x for concurrent ones, and 3.0 GB/s is past the
point where storage is usually the thing you are waiting for. The crossover
the random-access reader already documents still applies: decoding beats
phone flash, eMMC, an SD card and a network filesystem, and it loses to a
fast local SSD.

The `plain file` row cannot be measured honestly here and should not be
quoted: under WSL2 `posix_fadvise(DONTNEED)` does not reach the Windows
host's own cache, and the same file measured 1843 and 5846 MiB/s on
successive "cold" runs.

What the mount wins outright is the part that is not a throughput number:

| | ready to read | disk needed |
|---|---|---|
| `lmz mount` | 15 ms | 1.19 GiB |
| `lmz decompress` first | 2062 ms | 2.93 GiB |

A model that is never expanded never costs its expanded size, and the second
read of a tensor costs nothing at all: the kernel caches the *decoded* pages,
which is where the 24 GB/s row comes from. Repeated loads — the normal case
for a model you actually use — are served from that cache without decoding
anything.

### Where the speed came from

Three things, each of which had to be measured before it was believed.

**Free-threading is most of it.** The decode path is native work with a
little Python between the calls, and it is that Python which serialises. On a
free-threaded build the caps lift by themselves — `lmz` asks the interpreter
rather than being told — and the same mount goes from 813 to 3000 MiB/s.
Under the GIL the server is best at **two** threads, and the measurement is
emphatic: two serve 670 MiB/s where four serve 349 and eight 337, the same
inversion decompression already documents.

**A single sequential reader cannot be helped by threads at all**, because
the kernel will not ask for the next block until this one is answered. Worse,
buffered reads arrive at 128 KiB however large the caller's read is: the
readahead window is capped at `VM_MAX_READAHEAD`, and `process_init_reply`
takes the *smaller* of that and whatever a filesystem asks for, so a mount can
only lower it. A 1.74 GiB read is 14299 round trips no matter what.

So the mount predicts instead of waiting. A stream is recognised when a read
begins where the last one ended, and the window beyond it is decoded on idle
cores while the current reply is being written — 687 to 1476 MiB/s. Two
details decide whether that works at all, and getting either wrong made it
*slower* than no prefetch:

- The window is cut into slices across the pool. One task decoding 4 MiB is
  serial, so it finishes a window behind the reader it is meant to be ahead
  of — measured 1.6x slower than not prefetching.
- A block already being decoded is waited on, not decoded again. Without
  that, the prefetch and the reader race for the same blocks and the
  speculation costs exactly what it saves. With it, serving the whole model
  decodes 28579 blocks where the file contains 28577.

**Reading ahead is switched off when the server is busy.** Speculation is
only free while cores are idle; with several readers already saturating the
decoder it took 17% away. The mount reads ahead only while two or fewer
requests are in flight, which is exactly the lone-reader case that was
starved, and leaves concurrent readers alone.

Bigger blocks were tried and rejected. They compress slightly better, but a
1 MiB request straddles two 1 MiB blocks and decodes twice what it serves:

| block | saved | 1 reader | 16 readers | 4 KiB read amplification |
|---|---|---|---|---|
| **64 KiB** | 32.02% | **519** | **2680** | 16x |
| 256 KiB | 32.54% | 263 | 1898 | 64x |
| 1 MiB | 32.66% | 92 | 750 | 256x |

## A filesystem that compresses what you put in it

`lmz mount` serves a store you filled ahead of time. `lmz fs` is the other
half: a read-write filesystem where anything you write is compressed on the
way down and decoded on the way back up. Nothing has to be registered, and
the files are just files.

```
lmz fs ~/.lmz/data ~/data       # ~/data behaves normally; ~/.lmz/data holds it compressed
cp model.safetensors ~/data/    # 1.74 GiB in, 1.19 GiB on disk
```

The codec is chosen per file by the same planner the command line uses:
safetensors and GGUF get the field-split and block-split coders, everything
else falls through to the generic entropy coder, which is zstd. So the fair
question is whether it loses anywhere, and the answer is measured below.

### Against a compressing filesystem

btrfs and ZFS compress fixed-size extents independently and store whole 4 KiB
blocks, so that — not a single zstd of the whole file — is what they actually
achieve. Same corpus, same machine:

| file | raw | lmzfs | btrfs-style 128 KiB | delta |
|---|---|---|---|---|
| allsrc.py | 241 KB | 68.5% | 67.8% | +0.7 |
| binary.so | 2.19 MB | 51.3% | 50.4% | +0.9 |
| code.c | 38 KB | 67.6% | 67.9% | −0.3 |
| data.json | 1.77 MB | 83.0% | 81.3% | +1.7 |
| **model.safetensors** | 1.87 GB | **32.0%** | 18.8% | **+13.2** |
| text.md | 56 KB | 57.8% | 56.5% | +1.3 |
| **whole corpus** | 1.88 GB | **32.1%** | **18.9%** | **+13.2** |

An f2fs-style 16 KiB extent — the Android default — saves **0.1%** on the same
corpus and **0.0%** on the model, because zstd in a 16 KiB window finds almost
nothing in float weights and block rounding eats the rest. On this corpus
lmzfs writes **237 MiB less than btrfs would**.

The general-file rows are within a point or two either way, which is the
point: lmzfs is not worse at the things zstd is already good at, because on
those files it *is* zstd. Ordinary files are coded at level 3 to match what a
filesystem uses; weights stay at level 1, where the README's own measurements
put the smallest output anyway.

### How it is stored

The backing directory mirrors the mount, with one suffix saying how each file
is held:

```
backing/model.safetensors.lmz    compressed
backing/photo.jpg.lmr            stored raw, compression declined
backing/subdir/                  a directory, as itself
```

The suffix keeps the mapping unambiguous — a file genuinely called `a.lmz`
lands at `a.lmz.lmz` and still reads back under its own name. Nothing needs a
sidecar index: mode, owner and times are the backing file's own, and the
logical size is read from a fixed offset in the archive's 32-byte header, so a
stat costs one short read. Every file is independent, so an interrupted write
can damage one file and never the store.

Files below 4 KiB, and files where compression fails to win 2%, are stored raw
rather than wrapped in a container whose header would cost more than the coder
saves.

### What it is not good at

Writes are buffered: a file opened for writing is materialised into a scratch
copy, written there, and compressed once on the last close. The coder needs
whole chunks to measure a histogram, and no archive can have a byte rewritten
in the middle, so this is the shape rather than a compromise. It suits files
written once and read many times, which is what model files are, and it suits
a database file badly.

Reading is the same trade as the model mount: **572 MiB/s against 2961 for the
plain file** on this NVMe. You are buying a third of the disk back, not speed.

## Command line

```
lmz compress    <input> [output]   -l LEVEL  -j N  --chunk-size N  --no-checksum
                                   --no-dedup  --no-delta  --mapped  --align  -f
lmz append      <archive> <input>  -l LEVEL  -j N  --no-checksum  --no-delta
lmz extract     <archive> <member> <output>  -f
lmz decompress  <input> [output]   -j N  --no-verify  -f
lmz verify      <archive>          -j N
lmz info        <archive>          --tensors  --json  --limit N
lmz cat         <archive> <tensor> -o FILE  --member FILE
lmz bench       <file>             --bytes N
lmz doctor

lmz add         <path>             --name N  -l LEVEL  -j N  -f   --store DIR
lmz models                         --json                        --store DIR
lmz rm          <name>                                           --store DIR
lmz mount       <mountpoint>       --model N  -d  --allow-other  --store DIR
                                   --cache-blocks N  --verify  -j N  --readahead N
lmz fs          <backing> <mountpoint>   -d  -l LEVEL  -j N  --block-size N
                                         --allow-other
lmz unmount     <mountpoint>
```

The store lives at `$LMZ_HOME`, else `~/.lmz`: a directory of page-mapped
archives plus a JSON index. The index exists only so that listing the store
does not mean opening every archive in it, and `Store.rebuild()` reconstructs
it from the archives alone if it is lost.

`info` reports what the codec actually did, which is the quickest way to see
why a file compressed the way it did:

```
chunk codecs
  bf16-split   140 chunks    1.09 GiB -> 730.89 MiB  1.524x
  entropy        2 chunks   12.54 KiB ->   1.82 KiB  6.885x
  stored         1 chunks       568 B ->      568 B  1.000x
```

`cat` pulls a single tensor out of an archive by decoding only the chunks it
overlaps, without expanding the rest.

## Python API

```python
import lmz

stats = lmz.compress("model.safetensors", "model.lmz")
print(f"{stats.ratio:.3f}x, {stats.saved:.1%} smaller, {stats.seconds:.1f}s")

lmz.decompress("model.lmz", "restored.safetensors")
lmz.verify("model.lmz")

meta = lmz.info("model.lmz")                       # members, tensors, codecs
dtype, shape, raw = lmz.read_tensor("model.lmz", "model.embed_tokens.weight")
```

`MappedArchive` is the random-access reader:

```python
with lmz.MappedArchive("model.lmz") as arc:      # written with mapped=True
    head = arc.read(0, 4096)                     # any byte range
    dtype, shape, raw = arc.tensor("model.embed_tokens.weight")
    print(arc.decoded_bytes)                     # what it actually expanded
```

On a 942 MiB BF16 model, 200 random 4 KiB reads take 20 ms against 3.9 s from
an 8 MiB-chunk archive, and expand 17x what was asked for rather than 2025x.
Reading runs at 1.22 GB/s, which is the honest ceiling and worth
being precise about: it beats phone flash, and it does not beat an NVMe drive.
Compression buys a cold load a third fewer bytes to move; it does not make
inference faster, and a plain mmap of an uncompressed file still wins on
random access.

```python
lmz.append("run.lmz", "checkpoint-9000.safetensors")   # code against what is there
lmz.extract("run.lmz", "checkpoint-3000.safetensors", "ck3000.safetensors")
```

The store is the same reader with a name attached, plus somewhere to put it:

```python
store = lmz.Store()                              # $LMZ_HOME, else ~/.lmz
store.add("./Llama-3.1-8B-Instruct")             # compressed, page-mapped
for e in store.models():
    print(e.name, e.stored_size, f"{e.saved:.1%}")

with store.open("Llama-3.1-8B-Instruct") as arc:   # no mount, no expansion
    dtype, shape, raw = arc.tensor("model.embed_tokens.weight")
```

`mount` builds a server the caller runs; it must be its own process, because
a thread that page-faults on its own mount blocks in the kernel while holding
the GIL, and the thread that would answer the fault can never run:

```python
server = lmz.mount("/home/me/models")            # optionally names=[...]
server.serve()                                   # blocks until unmounted
```

`Store` takes `root`; `add` takes `name`, `level`, `workers`, `force`,
`block_size` and `progress`; `mount` takes `store`, `names`, `threads`,
`allow_other`, `cache_blocks`, `verify` and `readahead` (0 disables it).

`MappedArchive.prefetch(offset, length)` decodes a range into the cache and
returns how many blocks it had to decode, which is what the mount's readahead
is built on. A reader that knows where it is going next can use it directly:

```python
with lmz.MappedArchive("model.lmz", cache_blocks=256) as arc:
    arc.prefetch(offset, 4 << 20)        # on another thread, ahead of the read
    data = arc.read(offset, 1 << 20)     # now a cache hit
```

It is safe to call from any thread against a shared reader: a block already
being decoded elsewhere is waited on rather than decoded twice.

`compress` takes `level`, `workers`, `chunk_size`, `checksum`, `dedup`,
`delta`, `mapped`, `align` and `progress`; `append` takes `level`, `workers`,
`checksum`, `delta` and `progress`; `decompress` takes `workers`, `verify_checksums`, `overwrite` and
`progress`. `progress` is called with `(bytes_done, total)`.
