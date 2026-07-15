import os
import numpy as np
from tokenizers import Tokenizer

data_dir = os.path.dirname(__file__)
tokenizer_path = os.path.join(os.path.dirname(data_dir), 'bracket_tokenizer.json')

tok = Tokenizer.from_file(tokenizer_path)

with open(os.path.join(data_dir, 'train.txt'), 'r', encoding='utf-8') as f:
    lines = f.readlines()

# train.txt is 95% of the full dataset (test.txt already holds the other 5%),
# so carving off 5/95 of it as val yields an overall 90/5/5 train/val/test split.
n_val = round(len(lines) * 5 / 95)
train_data = ''.join(lines[:-n_val])
val_data = ''.join(lines[-n_val:])

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
