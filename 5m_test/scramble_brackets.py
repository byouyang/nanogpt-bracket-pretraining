"""
Randomly reassign each matched bracket-pair occurrence in the annotated
corpus to a different bracket-pair type from brackets.py, destroying the
mapping between bracket identity and annotated role while keeping every
pair internally consistent (a swapped open always matches its swapped
close, so nesting/balance is preserved).

Each occurrence is shuffled independently with no fixed mapping - the same
original pair type can end up replaced by different pairs at different
points in the corpus.
"""
import argparse
import io
import os
import random

import brackets

ROOT = os.path.dirname(__file__)
DEFAULT_INPUT = os.path.join(ROOT, 'corpus_processed_annotated_updated.txt')
DEFAULT_OUTPUT = os.path.join(ROOT, 'corpus_processed_annotated_updated_scrambled.txt')

PAIRS = list(brackets.BRACKETS_DICT.items())
PAIR_INDEX = {pair: i for i, pair in enumerate(PAIRS)}
OPEN_TO_CLOSE = brackets.BRACKETS_DICT
CLOSE_TO_OPEN = {close: open_ for open_, close in PAIRS}


def random_other_pair(orig_pair, rng):
    """Pick a uniformly random pair from PAIRS, guaranteed different from orig_pair."""
    orig_idx = PAIR_INDEX[orig_pair]
    idx = rng.randrange(len(PAIRS))
    while idx == orig_idx:
        idx = rng.randrange(len(PAIRS))
    return PAIRS[idx]


def scramble_line(line, rng):
    """Find every matched bracket pair in line (stack-based) and independently
    reassign each one to a random different pair type. Returns
    (new_line, n_pairs_swapped, malformed)."""
    stack = []
    matches = []
    malformed = False
    for pos, ch in enumerate(line):
        if ch in OPEN_TO_CLOSE:
            stack.append((ch, pos))
        elif ch in CLOSE_TO_OPEN:
            if not stack or OPEN_TO_CLOSE[stack[-1][0]] != ch:
                malformed = True
                break
            open_ch, open_pos = stack.pop()
            matches.append((open_pos, pos, open_ch, ch))
    if stack:
        malformed = True

    if malformed:
        return line, 0, True

    chars = list(line)
    for open_pos, close_pos, open_ch, close_ch in matches:
        new_open, new_close = random_other_pair((open_ch, close_ch), rng)
        chars[open_pos] = new_open
        chars[close_pos] = new_close
    return ''.join(chars), len(matches), False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default=DEFAULT_INPUT)
    parser.add_argument('--output', default=DEFAULT_OUTPUT)
    parser.add_argument('--seed', type=int, default=42,
                         help='random seed for reproducibility (default: nondeterministic)')
    args = parser.parse_args()

    rng = random.Random(args.seed)

    total_lines = 0
    malformed_lines = 0
    total_pairs = 0

    with io.open(args.input, encoding='utf-8') as fin, \
         io.open(args.output, 'w', encoding='utf-8') as fout:
        for line in fin:
            total_lines += 1
            new_line, n_pairs, malformed = scramble_line(line, rng)
            if malformed:
                malformed_lines += 1
            total_pairs += n_pairs
            fout.write(new_line)

    print(f"wrote scrambled corpus to {args.output}")
    print(f"lines: {total_lines}, pairs swapped: {total_pairs:,}, malformed lines skipped: {malformed_lines}")


if __name__ == '__main__':
    main()
