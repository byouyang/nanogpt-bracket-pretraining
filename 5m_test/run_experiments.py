"""
Runs one train/test split end to end (README pipeline steps 3-6): the shared
corpus/vocab setup, then each of the four arms' prepare step and training run,
in order (A -> D). Sequential, not parallel -- the arms are meant to be
compute-matched, and sharing a GPU between them would muddy wall-clock timing.

Set the split in split_config.py, then run this. Everything for that split --
checkpoints, tensorboard, its held-out test sets and its vocabs -- lands under
split<SPLIT_SEED>/, so splits never overwrite each other.

Assumes update_brackets.py and scramble_brackets.py have already produced the
annotated/control/scrambled corpora (steps 1-2); those are split-independent.
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from split_config import SPLIT_SEED

SPLIT_DIR = os.path.join(ROOT, f'split{SPLIT_SEED}')

# Shared across arms, so these run once up front rather than per arm:
# split_corpus.py applies one shuffle to the line-aligned corpora and writes
# each arm's data/train.txt plus the held-out test_data/test*.txt;
# train_tokenizer_8000.py then trains the vocabs those train.txt files get
# encoded against, into the shared tokenizer/ dir (+ each arm's data/meta.pkl);
# prepare_test_data.py encodes the held-out test text into each arm's vocab
# space. The last one is nominally an eval step, but it runs here so this
# split's test bins get archived below before the next split overwrites them.
SETUP = [
    'split_corpus.py',
    'train_tokenizer_8000.py',
    os.path.join('test_data', 'prepare_test_data.py'),
]

# (arm directory, prepare scripts to run in order before train.py).
# bracket_transfer takes two: data/prepare.py builds its phase-1 (annotated)
# bins, prepare_clean.py its phase-2 (clean) bins in the same vocab space.
EXPERIMENTS = [
    ('control', ['data/prepare.py']),
    ('bracket', ['data/prepare.py']),
    ('scrambled', ['data/prepare.py']),
    ('bracket_transfer', ['data/prepare.py', 'prepare_clean.py']),
]

# Everything the setup step produces is specific to SPLIT_SEED, and the next
# split overwrites it in place. Archive this split's copies next to its
# checkpoints so the run stays scoreable after the working dirs move on: the
# test sets because scoring a checkpoint against a different split's test set
# scores it on documents it trained on, and the vocabs because the archived
# test text can't be re-encoded without the vocab it was tokenized in.
ARCHIVE = [
    # source text -- test.txt is also bpb_eval.py's shared byte denominator
    os.path.join('test_data', 'test.txt'),            # clean -- control, and bracket_transfer's eval
    os.path.join('test_data', 'test_annotated.txt'),  # annotated -- bracket
    os.path.join('test_data', 'test_scrambled.txt'),  # scrambled -- scrambled
    # the same text encoded per vocab space -- what bpb_eval.py actually scores
    os.path.join('test_data', 'test.bin'),
    os.path.join('test_data', 'test_annotated.bin'),
    os.path.join('test_data', 'test_scrambled.bin'),
    os.path.join('test_data', 'test_bracket_transfer.bin'),
    os.path.join('tokenizer', 'tokenizer.json'),
    os.path.join('tokenizer', 'bracket_tokenizer.json'),
]


def run(script, cwd):
    print(f"\n=== {os.path.relpath(cwd, ROOT)}: python {script} ===", flush=True)
    # output is streamed rather than captured: tensorboard already has the loss
    # curves, but capturing would swallow the setup logs (vocab size, param
    # count, checkpoint saves) and -- worse -- a crashed child's traceback,
    # since check=True stashes stderr on the exception instead of printing it.
    subprocess.run([sys.executable, script], cwd=cwd, check=True)


def archive_split_inputs():
    for rel in ARCHIVE:
        dest = os.path.join(SPLIT_DIR, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copyfile(os.path.join(ROOT, rel), dest)
    print(f"\narchived split {SPLIT_SEED}'s test sets + vocabs -> {SPLIT_DIR}\n"
          f"score it later with: python bpb_eval.py --split {SPLIT_SEED}", flush=True)


def main():
    for script in SETUP:
        run(script, cwd=ROOT)

    archive_split_inputs()

    # split_corpus.py writes the annotated train.txt only into bracket/data,
    # but arm D trains phase 1 on that same annotated text, so it needs its own
    # copy to tokenize (into its own bracket vocab space, in its own data/).
    shutil.copyfile(
        os.path.join(ROOT, 'bracket', 'data', 'train.txt'),
        os.path.join(ROOT, 'bracket_transfer', 'data', 'train.txt'),
    )

    for arm, prepare_scripts in EXPERIMENTS:
        arm_dir = os.path.join(ROOT, arm)
        for script in prepare_scripts:
            run(script, cwd=arm_dir)
        # train.py resolves `dataset` relative to the cwd, so it has to run
        # from inside the arm's directory.
        run('train.py', cwd=arm_dir)

    print("\ndone")


if __name__ == '__main__':
    main()
