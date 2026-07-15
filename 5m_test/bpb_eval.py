"""
Test bench: evaluate all four models -- control, bracket, scrambled, and
bracket_transfer -- on their own held-out TEST sets and report bits-per-byte
(BPB).

Deliberately scores test_data/*.bin, not val.bin. val.bin is the internal
5/95 slice of train.txt that train.py's estimate_loss() checks during
training and best_val_loss uses to decide which checkpoint to save -- it
influenced checkpoint selection, so it's not a blind set. test_data/ holds
the global 5% split_corpus.py held out before any of that, shared
(line-aligned) across all four arms, and never touched by training -- the
right set for a final reported number. See test_data/prepare_test_data.py
for how the *.bin files there are produced.

These models use different tokenizers/vocab sizes (and the bracket/scrambled
test sets contain extra annotation markup), so raw loss-per-token isn't
comparable between them. BPB fixes this: convert each model's total
cross-entropy loss to bits, then divide by the UTF-8 byte count of the
*clean* (unannotated) test text - the same denominator for every model,
since test_data/test.txt, test_data/test_annotated.txt, and
test_data/test_scrambled.txt are line-aligned versions of the same
underlying documents (test_data/test_bracket_transfer.bin is that same clean
text again, just tokenized in the bracket vocab -- see prepare_test_data.py).

Loss is further split by target-token category - "word" tokens vs. "bracket"
tokens (the annotation markup) - each reported as its own bpb against that
same shared byte denominator, so word_bpb + bracket_bpb == the overall bpb.
The bracket/scrambled tokenizer reserves a fixed set of ids for bracket-pair
markup, registered as special tokens before BPE training so they can never
merge into a word token (see brackets.py / train_tokenizer_8000.py); those
ids are looked up from bracket_tokenizer.json at import time (BRACKET_TOKEN_IDS
below) rather than assumed to fall in any particular range. Control's
tokenizer never emits those ids, so its bracket bucket is always empty.
bracket_transfer shares the bracket tokenizer (it's trained through the
annotated phase 1) but is scored on clean text, so its bracket bucket should
also come out empty -- checked below with a warning rather than assumed.

bracket_transfer's checkpoint keeps block_size=768 (phase 1's sequence
length -- see bracket_transfer/train.py), but nanoGPT's GPT.forward doesn't
require using the full block_size: any T <= config.block_size works, using
positions 0..T-1. So bracket_transfer is deliberately evaluated in
EVAL_BLOCK_SIZE (= control's block_size) windows instead of its own
model.config.block_size, giving it the same context length as control --
same tokens in, same tokens out, so the comparison needs no fairness
argument. control/bracket/scrambled are each evaluated at their own native
block_size.
"""
import os
import math
import sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(__file__)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
EVAL_BATCH_SIZE = 64
EVAL_BLOCK_SIZE = 256  # matches control/train.py's block_size -- forced context length for bracket_transfer

sys.path.insert(0, os.path.join(ROOT, 'control'))
from model import GPTConfig, GPT
from brackets import BRACKET_LIST
from tokenizers import Tokenizer

CLEAN_TEST_PATH = os.path.join(ROOT, 'test_data', 'test.txt')
with open(CLEAN_TEST_PATH, 'r', encoding='utf-8') as f:
    clean_test_docs = f.readlines()
TEST_BYTES = sum(len(d.encode('utf-8')) for d in clean_test_docs)

ctx = torch.autocast(device_type='cuda', dtype=DTYPE) if DEVICE == 'cuda' else nullcontext()

# Bracket token ids are looked up from the tokenizer file rather than
# assumed to be ">= 8000" -- train_tokenizer_8000.py now registers bracket
# characters as special tokens BEFORE training (to keep them atomic and
# out of the BPE merge table), which puts them at the START of the vocab
# (0..len(BRACKET_LIST)-1), not the end.
_bracket_tok = Tokenizer.from_file(os.path.join(ROOT, 'bracket', 'bracket_tokenizer.json'))
BRACKET_TOKEN_IDS = torch.tensor(
    sorted(_bracket_tok.token_to_id(c) for c in BRACKET_LIST), device=DEVICE
)

# bracket_ids is None for control: its tokenizer is a separate, independently
# trained vocab that has never seen a bracket character, so ids 0..187 there
# are ordinary word content, not markup -- applying BRACKET_TOKEN_IDS (which
# means something completely different in control's vocab space) would
# misclassify a chunk of real word tokens as "bracket" by numeric coincidence.
# bracket, scrambled, and bracket_transfer all share the actual bracket
# tokenizer, so it's valid for all three.
#
# Each entry: (name, ckpt_path, test_bin_path, bracket_ids, eval_block_size).
# eval_block_size is None for control/bracket/scrambled, meaning "use the
# checkpoint's own model.config.block_size"; bracket_transfer overrides it to
# EVAL_BLOCK_SIZE so it's scored at the same context length as control.
RUNS = [
    ('control', os.path.join(ROOT, 'control', 'out', 'ckpt.pt'), os.path.join(ROOT, 'test_data', 'test.bin'), None, None),
    ('bracket', os.path.join(ROOT, 'bracket', 'out', 'ckpt.pt'), os.path.join(ROOT, 'test_data', 'test_annotated.bin'), BRACKET_TOKEN_IDS, None),
    ('scrambled', os.path.join(ROOT, 'scrambled', 'out', 'ckpt.pt'), os.path.join(ROOT, 'test_data', 'test_scrambled.bin'), BRACKET_TOKEN_IDS, None),
    ('bracket_transfer', os.path.join(ROOT, 'bracket_transfer', 'out', 'ckpt.pt'), os.path.join(ROOT, 'test_data', 'test_bracket_transfer.bin'), BRACKET_TOKEN_IDS, EVAL_BLOCK_SIZE),
]


def load_model(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


@torch.no_grad()
def total_nats(model, test_bin_path, bracket_ids, eval_block_size=None, batch_size=EVAL_BATCH_SIZE):
    block_size = eval_block_size if eval_block_size is not None else model.config.block_size
    data = np.memmap(test_bin_path, dtype=np.uint16, mode='r')
    n_blocks = (len(data) - 1) // block_size

    word_nats = 0.0
    bracket_nats = 0.0
    n_word_tokens = 0
    n_bracket_tokens = 0
    for start in range(0, n_blocks, batch_size):
        block_ids = range(start, min(start + batch_size, n_blocks))
        x = torch.stack([
            torch.from_numpy(data[i * block_size: i * block_size + block_size].astype(np.int64))
            for i in block_ids
        ]).to(DEVICE)
        y = torch.stack([
            torch.from_numpy(data[i * block_size + 1: i * block_size + block_size + 1].astype(np.int64))
            for i in block_ids
        ]).to(DEVICE)
        with ctx:
            logits, _ = model(x, y)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='none')
        if bracket_ids is None:
            is_bracket = torch.zeros_like(y.view(-1), dtype=torch.bool)
        else:
            is_bracket = torch.isin(y.view(-1), bracket_ids)
        bracket_nats += loss[is_bracket].sum().item()
        word_nats += loss[~is_bracket].sum().item()
        n_bracket_tokens += is_bracket.sum().item()
        n_word_tokens += (~is_bracket).sum().item()

    return word_nats, bracket_nats, n_word_tokens, n_bracket_tokens


def main():
    print(f"clean test bytes (shared denominator): {TEST_BYTES:,}")
    results = {}
    for name, ckpt_path, test_bin_path, bracket_ids, eval_block_size in RUNS:
        if not os.path.exists(ckpt_path):
            print(f"{name}: skipped, no checkpoint at {ckpt_path}")
            continue
        if not os.path.exists(test_bin_path):
            print(f"{name}: skipped, no test data at {test_bin_path}")
            continue
        model = load_model(ckpt_path)
        word_nats, bracket_nats, n_word_tokens, n_bracket_tokens = total_nats(
            model, test_bin_path, bracket_ids, eval_block_size
        )
        word_bpb = word_nats / math.log(2) / TEST_BYTES
        bracket_bpb = bracket_nats / math.log(2) / TEST_BYTES
        sum_bpb = word_bpb + bracket_bpb
        results[name] = sum_bpb
        print(f"{name}: word {n_word_tokens:,} tokens -> {word_bpb:.4f} bpb, "
              f"bracket {n_bracket_tokens:,} tokens -> {bracket_bpb:.4f} bpb, "
              f"sum = {sum_bpb:.4f} bpb")
        if name == 'bracket_transfer' and n_bracket_tokens != 0:
            print(f"WARNING: {n_bracket_tokens} bracket-id tokens found in supposedly clean eval "
                  f"data -- test_data/test_bracket_transfer.bin may not have been produced by "
                  f"test_data/prepare_test_data.py")


if __name__ == '__main__':
    main()
