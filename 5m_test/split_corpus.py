"""
Split corpus_processed_annotated.txt, corpus_processed_control.txt, and
corpus_processed_annotated_updated_scrambled.txt into train/test sets.

The three files are line-aligned (line i in one corresponds to line i in the
others), so a single shuffle + split is computed once and applied to all of
them, keeping the same lines in train/test for each variant.

train.txt (95% of each corpus) is written into that arm's own data/ folder,
same as before -- it's what that arm's prepare.py tokenizes into train.bin
(and, internally, an additional 5/95 val.bin slice for training-time
monitoring/checkpoint selection).

test.txt (the other 5%, held out here and never touched again until final
evaluation) is written into the shared test_data/ folder instead of any
arm's data/ folder, since it's used across arms (see test_data/README below
/ test_data/prepare_test_data.py).
"""
import os
import random

SEED = 42
VAL_FRACTION = 0.05

ROOT = os.path.dirname(__file__)
ANNOTATED_PATH = os.path.join(ROOT, 'corpus_processed_annotated_updated.txt')
CONTROL_PATH = os.path.join(ROOT, 'corpus_processed_control.txt')
SCRAMBLED_PATH = os.path.join(ROOT, 'corpus_processed_annotated_updated_scrambled.txt')

BRACKET_OUT_DIR = os.path.join(ROOT, 'bracket', 'data')
CONTROL_OUT_DIR = os.path.join(ROOT, 'control', 'data')
SCRAMBLED_OUT_DIR = os.path.join(ROOT, 'scrambled', 'data')
TEST_DATA_DIR = os.path.join(ROOT, 'test_data')


def read_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.readlines()


def write_lines(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.writelines(lines)


def main():
    annotated_lines = read_lines(ANNOTATED_PATH)
    control_lines = read_lines(CONTROL_PATH)
    scrambled_lines = read_lines(SCRAMBLED_PATH)

    assert len(annotated_lines) == len(control_lines), (
        f"line count mismatch: {len(annotated_lines)} annotated vs "
        f"{len(control_lines)} control"
    )
    assert len(annotated_lines) == len(scrambled_lines), (
        f"line count mismatch: {len(annotated_lines)} annotated vs "
        f"{len(scrambled_lines)} scrambled"
    )

    n = len(annotated_lines)
    indices = list(range(n))
    random.Random(SEED).shuffle(indices)

    n_val = int(n * VAL_FRACTION)
    val_idx = set(indices[:n_val])

    os.makedirs(BRACKET_OUT_DIR, exist_ok=True)
    os.makedirs(CONTROL_OUT_DIR, exist_ok=True)
    os.makedirs(SCRAMBLED_OUT_DIR, exist_ok=True)
    os.makedirs(TEST_DATA_DIR, exist_ok=True)

    for name, lines, out_dir, test_filename in [
        ('annotated', annotated_lines, BRACKET_OUT_DIR, 'test_annotated.txt'),
        ('control', control_lines, CONTROL_OUT_DIR, 'test.txt'),
        ('scrambled', scrambled_lines, SCRAMBLED_OUT_DIR, 'test_scrambled.txt'),
    ]:
        train_lines = [line for i, line in enumerate(lines) if i not in val_idx]
        test_lines = [line for i, line in enumerate(lines) if i in val_idx]

        write_lines(os.path.join(out_dir, 'train.txt'), train_lines)
        write_lines(os.path.join(TEST_DATA_DIR, test_filename), test_lines)

        print(f"{name}: {len(train_lines)} train lines -> {out_dir}, "
              f"{len(test_lines)} test lines -> {TEST_DATA_DIR}/{test_filename}")


if __name__ == '__main__':
    main()
