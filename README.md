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
(`bracket_tokenizer.json`, control vocab + bracket special tokens) so their
token ids are comparable; `control` has its own smaller vocabulary
(`tokenizer.json`) with no bracket tokens.

## Repository layout

```
5m_test/
  brackets.py                 bracket-pair character alphabet (BRACKETS_DICT / BRACKET_LIST)
  update_brackets.py          converts raw <...>/«...» delimiters to the bracket alphabet
  scramble_brackets.py        produces the scrambled-corpus variant
  split_corpus.py             one shared shuffle -> train/val (per arm) + held-out test (shared)
  train_tokenizer_8000.py     trains tokenizer.json (control) and bracket_tokenizer.json (bracket/scrambled/bracket_transfer)
  seed_config.py              single SEED shared by all training runs (not corpus construction)
  doc.txt                     notes on how the model/hparam config was derived from nanoGPT defaults

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
  sample.py            generate text from any of the 4 trained checkpoints
  checkpoint_info.py   CLI: param count / config / training state from a ckpt.pt, no GPU needed

  seed1337/, seed1338/  archived run outputs (out/ dirs + tensorboard logs) for each seed
```

## Pipeline

Run once, shared across arms:
1. `update_brackets.py` — normalize raw delimiters to the bracket alphabet.
2. `scramble_brackets.py` — derive the scrambled corpus from the annotated one.
3. `split_corpus.py` — one shuffle, applied to all corpora at once (they're
   line-aligned), writes each arm's `data/train.txt` and the shared
   `test_data/test*.txt`.
4. `train_tokenizer_8000.py` — trains `control/tokenizer.json` (8000-token
   BPE) and `bracket/bracket_tokenizer.json` (8000 + bracket special
   tokens), and copies the latter into `scrambled/` and `bracket_transfer/` 
   along with the respective `meta.pkl` (tokenizer vocab size meta data).

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
   Checkpoints go to `<arm>/out/ckpt.pt`; TensorBoard logs to
   `<arm>/out/tensorboard/`.

Evaluation, after all 4 arms have a checkpoint:
7. `test_data/prepare_test_data.py` — tokenizes the shared held-out test
   set into all 4 vocab spaces.
8. `bpb_eval.py` — scores all 4 checkpoints on their held-out test set,
   reporting bits-per-byte (the comparable metric across differing
   vocabularies/tokenizations), split into word-token vs. bracket-token bpb.
9. `plot_eval_loss.py` / `sample.py` / `checkpoint_info.py` — supporting
   tools for visualizing training curves, spot-checking generations, and
   inspecting checkpoint metadata.

`seed_config.py`'s `SEED` reseeds every arm's training run at once, for
repeating the whole comparison across seeds to check whether a bpb gap is
real or run-to-run noise (`seed1337/`, `seed1338/` are two such archived
sweeps).

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