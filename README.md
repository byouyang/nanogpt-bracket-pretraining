# nanoGPT Bracket Pretraining

Does marking nested structure (quotes, parentheticals, etc.) with explicit
bracket tokens during pretraining help a language model learn, compared to
training on the same text unmarked? This repo runs that as a small,
from-scratch nanoGPT experiment at a few model sizes, using bits-per-byte
(bpb) on a shared held-out test set as the comparison metric.

Each model size gets its own self-contained directory (`5m_test/`, and by
the same recipe `10m_test/`, `20m_test/`, ...). Everything below describes
the structure inside one of those directories and the pipeline used to
produce results; to add a new size, copy the recipe (see
[Scaling to a new model size](#scaling-to-a-new-model-size) below).

## Setup

Requires a Python environment with a CUDA-capable `torch` (CPU works but
training will be slow). Using [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html):

```
micromamba create -n bracket python=3.11 -y
micromamba activate bracket
```

Install `torch` first, on its own, matching your CUDA version -- the correct
command depends on your GPU/driver, so get it from the
[PyTorch install selector](https://pytorch.org/get-started/locally/). For
example:

```
# CUDA 12.4:
pip install torch --index-url https://download.pytorch.org/whl/cu124
# CPU only:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install everything else:

```
pip install -r requirements.txt
```

`requirements.txt` covers `numpy`, `tokenizers` (HuggingFace's fast BPE
library, used by `train_tokenizer_8000.py` and every `prepare.py`),
`tensorboard` (both writing logs from `train.py` and reading them back in
`plot_eval_loss.py`/`plot_bpb.py` via `EventAccumulator`), and `matplotlib` +
`seaborn` for plotting. No GPU is required just to plot or run `bpb_eval.py`/
`sample.py` against an already-trained checkpoint, but training from scratch
without one will be impractically slow.

## The four experimental arms

Every model size trains four variants side by side, all on the same
underlying documents:

| Arm | Directory | What it trains on |
|---|---|---|
| **control** | `control/` | Clean, unannotated text. Baseline. |
| **bracket** | `bracket/` | The same text with nested spans wrapped in special bracket tokens (e.g. quotes, parentheticals) — see `brackets.py` for the bracket-pair alphabet. |
| **scrambled** | `scrambled/` | Same as `bracket`, but each bracket-pair *occurrence* is randomly reassigned to a different pair type (nesting/balance preserved, identity-to-role mapping destroyed). Controls for "does having *any* extra structural tokens help" vs. the bracket arm's "do the *specific* bracket identities carry information." |
| **bracket_transfer** | `bracket_transfer/` | One model, two-phase curriculum: phase 1 on annotated text, phase 2 continues the same weights/optimizer onto clean text (tags are scaffolding, kicked away after phase 1). Total token budget is matched to `control`'s run. Tests whether bracket-annotation is useful as *pretraining curriculum* even though the final eval (and deployment target) is plain clean text. |

`bracket`, `scrambled`, and `bracket_transfer` all share one BPE vocabulary
(`tokenizer/bracket_tokenizer.json` — trained fresh on the annotated corpus
with the bracket characters registered as special tokens, so they stay atomic)
so their token ids are comparable; `control` has its own separate, smaller
vocabulary (`tokenizer/tokenizer.json`) that has never seen a bracket token.
Both live in `tokenizer/` and are read from there by every arm — there are no
per-arm copies.

## Repository layout

```
5m_test/
  brackets.py                 bracket-pair character alphabet (BRACKETS_DICT / BRACKET_LIST)
  update_brackets.py          converts raw <...>/«...» delimiters to the bracket alphabet
  scramble_brackets.py        produces the scrambled-corpus variant
  split_corpus.py             one shared shuffle -> train/val (per arm) + held-out test (shared)
  val_split.py                shared train/val carve-out + clean-byte bpb denominator, used by every prepare.py/train.py
  train_tokenizer_8000.py     trains both vocabs into tokenizer/
  run_experiments.py          runs one whole split: setup -> prepare -> train, all 4 arms
  split_config.py             SPLIT_SEED: the train/test split -- the axis being varied; names split<N>/
  seed_config.py              single SEED shared by all training runs (not corpus construction); held fixed
  doc.txt                     notes on how the model/hparam config was derived from nanoGPT defaults

  tokenizer/          the two BPE vocabs, shared by all arms
    tokenizer.json            control's vocab (8000, no bracket tokens)
    bracket_tokenizer.json    bracket/scrambled/bracket_transfer's vocab (8000 + brackets)

  control/            arm A: clean text
    data/prepare.py     tokenizes train.txt -> train.bin/val.bin/meta.pkl
    model.py, train.py

  bracket/             arm B: bracket-annotated text
    data/prepare.py, model.py, train.py

  scrambled/           arm C: scrambled-identity bracket-annotated text
    data/prepare.py, model.py, train.py

  bracket_transfer/    arm D: two-phase annotated -> clean curriculum
    data/prepare.py       phase-1 (annotated) data
    prepare_clean.py      phase-2 (clean) data, encoded into the SAME bracket vocab space
    model.py, train.py

  test_data/
    prepare_test_data.py  tokenizes the shared held-out test set into all 4 arms' vocab spaces
    test*.bin              -> scored by bpb_eval.py

  bpb_eval.py          scores all 4 checkpoints on held-out test data, reports bits-per-byte
  plot_eval_loss.py    plots TensorBoard val/loss for all arms on one figure, x-axis = tokens seen
  plot_bpb.py          same, but for val/bpb -- the cross-arm-comparable curve (see below)
  sample.py            generate text from any of the 4 trained checkpoints
  checkpoint_info.py   CLI: param count / config / training state from a ckpt.pt, no GPU needed

  seed1337/, seed1338/  archived run outputs (out/ dirs + tensorboard logs) for each seed
```

## Pipeline

Run once, shared across arms:
1. (optional) `update_brackets.py` — normalize raw delimiters to the bracket alphabet.
2. `scramble_brackets.py` — derive the scrambled corpus from the annotated one.
3. `split_corpus.py` — one shuffle (seeded by `split_config.py`'s
   `SPLIT_SEED`), applied to all corpora at once (they're line-aligned),
   writes each arm's `data/train.txt` and the shared `test_data/test*.txt`.
   Re-run per split; it overwrites both in place.
4. `train_tokenizer_8000.py` — trains `tokenizer/tokenizer.json` (8000-token
   BPE, control) and `tokenizer/bracket_tokenizer.json` (8000 + bracket
   special tokens), and writes each arm's `data/meta.pkl` (vocab size, read
   by `train.py` from its own data dir).

Per arm:
5. `<arm>/data/prepare.py` — tokenizes `train.txt` into `train.bin` /
   `val.bin` (95/5 split). `bracket_transfer` additionally
   needs `prepare_clean.py` run once before training, to produce its
   phase-2 `data_clean/`.
6. `<arm>/train.py` — trains from scratch. Run from inside the arm's
   directory, e.g.:
   ```
   cd control && python train.py
   cd bracket && python train.py
   cd scrambled && python train.py
   cd bracket_transfer && python train.py
   ```
   Checkpoints go to `split<SPLIT_SEED>/<arm>/ckpt.pt`; TensorBoard logs to
   `split<SPLIT_SEED>/<arm>/tensorboard/`.

   TensorBoard scalars: `train/loss`, `val/loss`, `val/bpb`, `lr`, `mfu`.
   `val/loss` is per-token nats and is **only** comparable within one arm —
   the arms tokenize differently. `val/bpb` is the cross-arm one: the same
   val loss converted to bits and divided by the *clean* val slice's byte
   count, the same denominator for every arm (`val_split.clean_val_bytes`).
   It's a constant rescale of `val/loss`, so it adds nothing within an arm;
   its whole purpose is putting the four curves on one axis. Compare with
   `plot_eval_loss.py --tag val/bpb`.

   Caveats: `val/bpb` is estimated from `estimate_loss()`'s random windows
   over *val* (which influenced checkpoint selection), not the deterministic
   full pass `bpb_eval.py` runs over the held-out test set — it's a curve, not
   a reportable number. And it fixes the *units*, not the confound: bracket
   and scrambled are still evaluated on annotated text where the model sees
   markup as context, so a bracket-vs-control gap in `val/bpb` is not the
   clean comparison. `bracket_transfer`'s phase-2 stretch is.

`run_experiments.py` runs steps 3-6 for one split in one go. It also runs
`test_data/prepare_test_data.py` as part of setup (rather than at eval time)
so that split's test bins can be archived before the next split overwrites
them, then copies the test sets and vocabs into `split<SPLIT_SEED>/` alongside
the checkpoints.

Evaluation, after all 4 arms have a checkpoint. Each takes `--split`,
defaulting to `split_config.py`'s `SPLIT_SEED`, and reads only from that
split's own directory:
7. `bpb_eval.py [--split N]` — scores all 4 checkpoints on that split's
   held-out test set, reporting bits-per-byte (the comparable metric across
   differing vocabularies/tokenizations), split into word-token vs.
   bracket-token bpb.
8. `plot_eval_loss.py [--split N]` — training curves; bare arm names resolve
   inside the split, but explicit paths still work, so you can plot the same
   arm across two splits (`plot_eval_loss.py split42/bracket split43/bracket`).
9. `sample.py [--split N]` / `checkpoint_info.py` — spot-checking generations
   and inspecting checkpoint metadata.

## Quickstart: running one split end to end

Everything below runs from inside `5m_test/` (`cd 5m_test` first).

One-time, corpus-level setup (skip if `bracket/`, `scrambled/` etc. already
have their annotated corpora -- these two are split-independent and only need
to be rerun if the source corpus itself changes):

```
python update_brackets.py
python scramble_brackets.py
```

To run (or rerun) one split:

1. Open `split_config.py` and set `SPLIT_SEED` to whatever integer names this
   split, e.g.:
   ```python
   SPLIT_SEED = 47
   ```
   Every output of this run -- checkpoints, TensorBoard logs, the archived
   test sets and vocabs -- will land under `split47/`. Pick a number you
   haven't used yet (check for existing `split*/` directories) so you don't
   overwrite a previous run; reusing a number silently overwrites that
   split's `data/train.txt`, `test_data/`, and `tokenizer/` in place before
   `run_experiments.py` re-archives them under the same `split<N>/`.
2. (Optional) Open `seed_config.py` and set `SEED` if you want a different
   training-run seed (weight init / batch sampling / dropout) for this run,
   e.g.:
   ```python
   SEED = 43
   ```
   Leave it alone if you're only varying the split -- `SEED` is normally
   held fixed while `SPLIT_SEED` is the thing being swept (see below).
   Unlike `SPLIT_SEED`, `SEED` isn't part of any output path, so if you do
   change it, note the value somewhere yourself (e.g. the archive dir name)
   if you'll need to know it later.
3. From `5m_test/`, run:
   ```
   python run_experiments.py
   ```
   This does the shared setup (reshuffle the corpus per `SPLIT_SEED`, train
   both vocabs, tokenize the held-out test set), archives that split's test
   sets/vocabs into `split47/`, then runs each arm's `prepare.py` and
   `train.py` in turn (`control` -> `bracket` -> `scrambled` ->
   `bracket_transfer`), streaming each script's output live. It's
   sequential and can take a while -- all four arms train back-to-back on
   one GPU, not in parallel.
4. Once it finishes, `split47/<arm>/ckpt.pt` and
   `split47/<arm>/tensorboard/` exist for all four arms. Evaluate with:
   ```
   python bpb_eval.py --split 47
   python plot_eval_loss.py --split 47
   python plot_bpb.py --split 47
   ```
   (`--split` defaults to whatever `SPLIT_SEED` currently is in
   `split_config.py`, so it can be omitted right after step 3, but is
   required once you've moved on to a different split.)

## Repeating the comparison

Two independent seeds, and they do different jobs:

- `split_config.py`'s `SPLIT_SEED` reshuffles which documents land in train
  vs. the held-out test set. **This is the axis the experiment varies.** It
  names the output directory (`split<SPLIT_SEED>/`), so splits never clobber
  each other, and `run_experiments.py` archives that split's test sets and
  vocabs into the same directory.
- `seed_config.py`'s `SEED` reseeds every arm's training run (weight init,
  batch sampling, dropout). Currently held **fixed** while `SPLIT_SEED`
  varies.

A split's test set, vocabs, and checkpoints belong together: the test set is
split-specific, so scoring a checkpoint against a *different* split's test set
scores it on documents it trained on. That's why they're archived as a unit.
Likewise the tokenizer is trained on that split's `train.txt`, so it must be
retrained per split — reusing another split's vocab means the BPE merges were
fit on documents in this split's test set.

(`seed1337/`, `seed1338/` are archives from the older scheme, when `SEED` was
the axis being varied and checkpoints lived in `<arm>/out/`.)

## Model sizing

Param count is controlled by `n_layer`, `n_head`, `n_embd` in each arm's
`model.py`/`train.py` (kept identical across the 4 arms at a given size).
Approximate formula (verified against `checkpoint_info.py` output on real
checkpoints):

```
params ≈ 12 * n_layer * n_embd^2       (transformer body)
        + vocab_size * n_embd          (tied token embedding / output head)
```

(`block_size * n_embd` for position embeddings and LayerNorm weights add a
little more, but are negligible at these sizes.)

`5m_test`'s current config (`n_layer=6, n_head=8, n_embd=256`, vocab 8000/8188)
is named for its original ~5M target, but actually measures at **~6.8M
params** per `checkpoint_info.py` — keep that in mind when comparing across
sizes; the directory name is a nominal label, not the exact count.

## Scaling to a new model size

To run the same 4-arm comparison at a new target size (10M, 20M, ...):

1. Copy `5m_test/` to `10m_test/` (or `20m_test/`), keeping the corpus
   files, `brackets.py`, tokenizer training script, and pipeline scripts —
   only `n_layer`/`n_head`/`n_embd` need to change, in `control/`,
   `bracket/`, `scrambled/`, and `bracket_transfer/`'s `train.py` (kept
   identical across all 4 arms, same as `5m_test`).
2. Pick `(n_layer, n_head, n_embd)` from the sizing formula above.
   Suggested starting points (vocab ≈ 8000-8188, `n_head` chosen so it
   divides `n_embd` evenly):

   | Target | n_layer | n_head | n_embd | Approx. params |
   |---|---|---|---|---|
   | 5M (current) | 6 | 8 | 256 | ~6.8M (actual) |
   | 10M | 6 | 8 | 320 | ~9.9-10.0M |
   | 20M | 8 | 8 | 416 | ~19.9-20.0M |

   Recompute and sanity-check with `checkpoint_info.py` against a freshly
   initialized checkpoint before committing to a config — the formula is an
   approximation, as the 5M/6.8M gap above shows.
3. Keep the corpus, tokenizer, `batch_size`, `block_size`, and training
   schedule (`max_iters`, warmup, LR decay) the same as `5m_test` unless
   there's a specific reason to change them — the point of running 5M/10M/20M
   side by side is a clean scaling comparison where model size is the only
   variable. Since the corpus is small and fixed (~2M train tokens), larger
   models are more prone to overfitting it — watch val loss for early
   divergence from train loss, and consider raising `dropout` (currently
   0.2) if it shows up.
4. Re-run the full pipeline steps 5-9 above inside the new `<N>m_test/`
   directory: `data/prepare.py` (+ `bracket_transfer/prepare_clean.py`) for
   each arm, then `train.py` for each arm, then
   `test_data/prepare_test_data.py` and `bpb_eval.py` for the final
   cross-arm bpb comparison.
5. Use `plot_eval_loss.py` across sizes (point it at each size's
   `out/tensorboard` dirs) to compare convergence, and `bpb_eval.py`'s
   output to compare final bpb — both within a size (control vs. bracket vs.
   scrambled vs. bracket_transfer) and across sizes (does the bracket
   effect grow, shrink, or hold steady as params increase).