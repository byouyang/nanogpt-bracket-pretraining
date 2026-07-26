import os
import sys
import numpy as np
from tokenizers import Tokenizer

data_dir = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(data_dir))
sys.path.insert(0, REPO_ROOT)
from val_split import split_train_val

tokenizer_path = os.path.join(REPO_ROOT, 'tokenizer', 'tokenizer.json')

tok = Tokenizer.from_file(tokenizer_path)

with open(os.path.join(data_dir, 'train.txt'), 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 5/95 of train.txt carved off as val -> an overall 90/5/5 train/val/test
# split. Shared with every other arm (and with train.py's bpb denominator) via
# val_split.py, so the same documents land in val everywhere.
train_data, val_data = split_train_val(lines)

train_ids = tok.encode(train_data).ids
val_ids = tok.encode(val_data).ids
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(data_dir, 'train.bin'))
val_ids.tofile(os.path.join(data_dir, 'val.bin'))

# the held-out test set lives in test_data/ (shared across arms), not here --
# see test_data/prepare_test_data.py
