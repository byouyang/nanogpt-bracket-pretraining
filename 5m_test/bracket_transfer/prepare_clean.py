"""
Prepare phase-2 ("clean") data for Arm D (bracket_transfer).

Arm D trains one model in two phases: phase 1 on annotated (bracket-tagged)
text, phase 2 on clean text, same weights + optimizer state throughout. For
that to work, phase 2's tokens must live in the *same* vocab space as phase 1
-- i.e. must be encoded with bracket_tokenizer.json (vocab_size 8188), not
control's own tokenizer.json (vocab_size 8000). If we encoded with control's
tokenizer instead, id 4000 could mean a different subword in each phase.

bracket_tokenizer.json is now trained fresh, directly on the annotated
corpus, with bracket characters registered as special tokens before BPE
training (see train_tokenizer_8000.py) -- so it is NOT derived from
control's tokenizer and its word-token ids are not expected to match
control's. That's exactly why this script re-encodes control's clean text
from source with bracket_tokenizer.json rather than reusing
control/data/*.bin: those live in a different, incompatible vocab space.

Source text: control/data/train.txt -- the clean, unannotated counterpart of
bracket/data's documents (split_corpus.py applied one shuffle to
annotated/control/scrambled corpora together, so lines are aligned across
arms). Same 95/5 train/val carve-out convention as bracket/data/prepare.py
and control/data/prepare.py.

The held-out test set (test_data/test.txt, same clean documents) is handled
separately by test_data/prepare_test_data.py -- not this script.
"""
import os
import sys
import numpy as np
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ROOT)
sys.path.insert(0, REPO_ROOT)
from brackets import BRACKET_LIST
from val_split import split_train_val

CLEAN_SOURCE_DIR = os.path.join(REPO_ROOT, 'control', 'data')
data_dir = os.path.join(ROOT, 'data_clean')
tokenizer_path = os.path.join(REPO_ROOT, 'tokenizer', 'bracket_tokenizer.json')

os.makedirs(data_dir, exist_ok=True)
tok = Tokenizer.from_file(tokenizer_path)

with open(os.path.join(CLEAN_SOURCE_DIR, 'train.txt'), 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Same carve-out as every other arm (val_split.py) applied to the clean source
# -- so this phase-2 val slice is the same documents as phase 1's, just
# unannotated, and matches the slice train.py's bpb denominator measures.
train_data, val_data = split_train_val(lines)

train_ids = tok.encode(train_data).ids
val_ids = tok.encode(val_data).ids

# sanity check: clean text should never produce bracket-markup ids. Looked
# up by token string, not a ">= 8000" threshold -- bracket characters are
# registered as special tokens before training now, which puts them at the
# START of the vocab (0..len(BRACKET_LIST)-1), not the end.
bracket_ids = set(tok.token_to_id(c) for c in BRACKET_LIST)
n_bracket_in_train = sum(1 for t in train_ids if t in bracket_ids)
n_bracket_in_val = sum(1 for t in val_ids if t in bracket_ids)
assert n_bracket_in_train == 0, f"found {n_bracket_in_train} bracket-id tokens in supposedly clean train text"
assert n_bracket_in_val == 0, f"found {n_bracket_in_val} bracket-id tokens in supposedly clean val text"

print(f"clean train has {len(train_ids):,} tokens")
print(f"clean val has {len(val_ids):,} tokens")

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))

# the held-out test set lives in test_data/ (shared across arms), not here --
# see test_data/prepare_test_data.py, which encodes test_data/test.txt with
# this same bracket_tokenizer.json to produce test_bracket_transfer.bin
