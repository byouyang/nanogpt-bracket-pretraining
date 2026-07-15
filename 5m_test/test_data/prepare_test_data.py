"""
Tokenize the held-out test set for every arm, all in one place.

test_data/*.txt (written by split_corpus.py) is the global 5% held out
before any arm's train/val split -- never touched by training or checkpoint
selection. This script produces the corresponding *.bin files that
bpb_eval.py scores for a final reported number:

  test.txt            (clean)             -> test.bin              [control tokenizer]
  test_annotated.txt   (bracket-tagged)    -> test_annotated.bin    [bracket tokenizer]
  test_scrambled.txt   (scrambled tags)    -> test_scrambled.bin    [bracket tokenizer]
  test.txt             (clean, again)      -> test_bracket_transfer.bin [bracket tokenizer]

That last one is not a typo: bracket_transfer's model lives in the bracket
tokenizer's vocab space (trained through the annotated phase 1), but is
evaluated on clean text (phase 2's whole point). So the SAME clean test.txt
gets encoded twice, once per vocab space it needs to be compared in --
test.bin (control vocab, for control's own eval) and
test_bracket_transfer.bin (bracket vocab, for bracket_transfer's eval).
scrambled shares the bracket tokenizer too (same bracket alphabet, just
reassigned pair identities), so its test set uses it directly.
"""
import os
import sys
import numpy as np
from tokenizers import Tokenizer

ROOT = os.path.dirname(__file__)
REPO_ROOT = os.path.dirname(ROOT)
sys.path.insert(0, REPO_ROOT)
from brackets import BRACKET_LIST

control_tok = Tokenizer.from_file(os.path.join(REPO_ROOT, 'control', 'tokenizer.json'))
bracket_tok = Tokenizer.from_file(os.path.join(REPO_ROOT, 'bracket', 'bracket_tokenizer.json'))
scrambled_tok = Tokenizer.from_file(os.path.join(REPO_ROOT, 'scrambled', 'bracket_tokenizer.json'))
bracket_transfer_tok = Tokenizer.from_file(os.path.join(REPO_ROOT, 'bracket_transfer', 'bracket_tokenizer.json'))


def tokenize_to_bin(text_filename, tok, bin_filename, assert_no_brackets=False):
    with open(os.path.join(ROOT, text_filename), 'r', encoding='utf-8') as f:
        text = f.read()
    ids = tok.encode(text).ids
    if assert_no_brackets:
        # bracket ids looked up from THIS tok instance, not assumed to match
        # some other arm's copy of the tokenizer file
        bracket_ids = set(tok.token_to_id(c) for c in BRACKET_LIST)
        n_bracket = sum(1 for t in ids if t in bracket_ids)
        assert n_bracket == 0, f"found {n_bracket} bracket-id tokens in supposedly clean {text_filename}"
    print(f"{text_filename} -> {bin_filename}: {len(ids):,} tokens")
    np.array(ids, dtype=np.uint16).tofile(os.path.join(ROOT, bin_filename))


tokenize_to_bin('test.txt', control_tok, 'test.bin')
tokenize_to_bin('test_annotated.txt', bracket_tok, 'test_annotated.bin')
tokenize_to_bin('test_scrambled.txt', scrambled_tok, 'test_scrambled.bin')
tokenize_to_bin('test.txt', bracket_transfer_tok, 'test_bracket_transfer.bin', assert_no_brackets=True)
