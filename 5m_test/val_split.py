"""
The train/val carve-out shared by every arm's prepare step, plus the clean byte
denominator that turns a val loss into bits-per-byte.

train.txt is 95% of the full dataset (test_data/ holds the other 5%), so
carving off 5/95 of it as val yields an overall 90/5/5 train/val/test split.
Every arm applies this identical carve-out to its own line-aligned copy of the
corpus, so n_val is the same everywhere and the arms' val slices are the same
underlying documents in different renderings.

clean_val_bytes() measures that val slice on CONTROL's (unannotated) text --
deliberately not on the calling arm's own text. bpb is only comparable across
arms if every arm divides by the same number, and the annotated/scrambled
corpora have more bytes than the clean one purely because of the markup, so an
arm dividing by its own byte count would score a lower bpb for free. Same
convention as bpb_eval.py, which uses clean test.txt bytes as the shared
denominator for all four models.

Both live here rather than inline in each prepare.py so they cannot drift:
train.py derives its bpb scale from clean_val_bytes(), which is only the right
denominator if it measures the exact slice prepare.py carved off. If the two
ever disagreed the bpb would be quietly wrong rather than fail.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
CLEAN_TRAIN_PATH = os.path.join(ROOT, 'control', 'data', 'train.txt')


def n_val_lines(n_lines):
    return round(n_lines * 5 / 95)


def split_train_val(lines):
    """-> (train_text, val_text) for one arm's train.txt lines."""
    n_val = n_val_lines(len(lines))
    return ''.join(lines[:-n_val]), ''.join(lines[-n_val:])


def clean_val_bytes():
    """UTF-8 byte count of the clean val slice: bpb's shared denominator.

    Reads control/data/train.txt regardless of which arm is asking, so every
    arm's bpb is divided by the same number.
    """
    with open(CLEAN_TRAIN_PATH, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    _, val_text = split_train_val(lines)
    return len(val_text.encode('utf-8'))
