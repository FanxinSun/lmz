# `hubloop.py` — lmz against ZipLLM's method, at hub scale

ZipLLM reports 54.1% on 3,048 Hugging Face models. That corpus is not
published, so this builds one the same way — families of a base model plus the
fine-tunes whose cards name it — and measures both methods over it, model by
model, on the same bytes.

Per fine-tune, two pipelines that differ only at the end:

- **ZipLLM's**: deduplicate by tensor content hash, XOR against the base,
  zstd -1.
- **lmz as shipped**: the tool, growing one archive per family. A fine-tune's
  cost is what it *adds* to that archive, which is the same cumulative
  opportunity the hash set gives the other side.

One model is on disk at a time, plus the family's base. Nothing but the ledger
needs to come back: kilobytes per model.

## Running it

The measurement is cheap and the download is not — 1.7 GB a model — so run it
where the link is fast. On Colab nothing is downloaded to your own machine at
all: Colab fetches from Hugging Face cloud to cloud, measures, records and
deletes.

```
python hubloop.py --state /content/drive/MyDrive/lmz-hub --setup --reverse
```

`--state` is the only thing that must survive the session: ledger, resume
marks, per-family hashes, the plan, and `progress.json`. Google Drive is fine
and is the point. Weights go to `--scratch` on the ephemeral disk and are
deleted after every model. A Hugging Face token is read from Colab secrets
(`HF_TOKEN`), the environment, or `~/.cache/huggingface/token`, and never
appears in the notebook, the ledger, or anything that comes back.

Re-run the same command after a disconnect: it resumes and loses at most the
model in flight.

## Before you change it

```
python hubloop.py --state /tmp/x --selftest
```

Runs every stage with no network — the token as a subprocess sees it, a path
with a space, a plan inside the state directory, both pipelines over a real
checkpoint, the ledger, `closed.json`, and resume after an interrupt. It uses a
real safetensors file if one is on disk, because synthetic models are a poor
stand-in: a real checkpoint is mostly tensors too small to deduplicate, and
tests built on uniform ones miss what those do.

## Two runners, one ledger

`--reverse` takes the plan from the far end. Run one forward and one reverse
and they work toward each other, every row says which produced it, and each
stops at the first family the other has closed.
