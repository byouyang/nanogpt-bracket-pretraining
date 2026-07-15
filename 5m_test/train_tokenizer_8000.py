import os
import pickle
import shutil

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

from brackets import BRACKET_LIST

ROOT = os.path.dirname(__file__)
CONTROL_TRAIN_PATH = os.path.join(ROOT, 'control', 'data', 'train.txt')
ANNOTATED_TRAIN_PATH = os.path.join(ROOT, 'bracket', 'data', 'train.txt')

CONTROL_TOKENIZER_PATH = os.path.join(ROOT, 'control', 'tokenizer.json')
BRACKET_TOKENIZER_PATH = os.path.join(ROOT, 'bracket', 'bracket_tokenizer.json')
# bracket_tokenizer.json is duplicated into these two dirs too (scrambled
# reuses the same bracket alphabet; bracket_transfer needs its own copy for
# Arm D) -- keep all three in sync so nobody trains against a stale copy.
BRACKET_TOKENIZER_COPIES = [
    os.path.join(ROOT, 'scrambled', 'bracket_tokenizer.json'),
    os.path.join(ROOT, 'bracket_transfer', 'bracket_tokenizer.json'),
]

CONTROL_META_PATH = os.path.join(ROOT, 'control', 'data', 'meta.pkl')
BRACKET_META_PATH = os.path.join(ROOT, 'bracket', 'data', 'meta.pkl')
# same vocab space as BRACKET_META_PATH (all encoded with bracket_tokenizer.json),
# so their meta.pkl is just a copy of bracket's, mirroring BRACKET_TOKENIZER_COPIES
BRACKET_META_COPIES = [
    os.path.join(ROOT, 'scrambled', 'data', 'meta.pkl'),
    os.path.join(ROOT, 'bracket_transfer', 'data', 'meta.pkl'),
    os.path.join(ROOT, 'bracket_transfer', 'data_clean', 'meta.pkl'),
]

# train the base tokenizer on the control (unannotated) train set only
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tok.decoder = decoders.ByteLevel()
tok.train([CONTROL_TRAIN_PATH], trainers.BpeTrainer(vocab_size=8000))
tok.save(CONTROL_TOKENIZER_PATH)

with open(CONTROL_META_PATH, 'wb') as f:
    pickle.dump({'vocab_size': tok.get_vocab_size()}, f)
print(f"control vocab_size = {tok.get_vocab_size()} -> {CONTROL_META_PATH}")

# The bracket tokenizer is trained FRESH, directly on the annotated corpus
# (not derived from the control tokenizer anymore). Registering the bracket
# characters as special tokens BEFORE training -- both via
# add_special_tokens on the tokenizer and special_tokens on the trainer --
# keeps them atomic: BPE can never learn a merge that fuses a bracket byte
# with a word byte, while the merge table still gets to see real words
# sitting immediately against a bracket (no leading space), which the old
# "train on control text, bolt on brackets after" scheme never saw and
# which was fragmenting word tokens in the annotated corpus.
#
# Side effect: because add_special_tokens runs on a still-empty tokenizer,
# the bracket ids land at the START of the vocab (0..len(BRACKET_LIST)-1),
# not the end like before. Nothing downstream should hardcode where they
# are -- bpb_eval.py / prepare_clean.py all now look them up by token string
# via the saved tokenizer file instead of assuming a ">= 8000" threshold.
bracket_tok = Tokenizer(models.BPE())
bracket_tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
bracket_tok.decoder = decoders.ByteLevel()
bracket_tok.add_special_tokens(BRACKET_LIST)
bracket_trainer = trainers.BpeTrainer(
    vocab_size=8000 + len(BRACKET_LIST),  # total budget incl. the reserved bracket ids
    special_tokens=BRACKET_LIST,
)
bracket_tok.train([ANNOTATED_TRAIN_PATH], bracket_trainer)

# verify atomicity: no bracket character may appear inside a longer merged
# token, and every bracket character must have gotten its own token id.
bracket_set = set(BRACKET_LIST)
vocab = bracket_tok.get_vocab()
for token_str, token_id in vocab.items():
    if any(c in bracket_set for c in token_str):
        assert token_str in bracket_set, (
            f"bracket byte leaked into a merged token: {token_str!r} (id {token_id})"
        )
bracket_ids = sorted(bracket_tok.token_to_id(c) for c in BRACKET_LIST)
assert len(bracket_ids) == len(BRACKET_LIST) and len(set(bracket_ids)) == len(BRACKET_LIST), \
    "not every bracket character got its own distinct token id"
print(f"bracket token ids verified atomic: {bracket_ids[0]}..{bracket_ids[-1]} "
      f"({len(bracket_ids)} tokens, contiguous={bracket_ids == list(range(bracket_ids[0], bracket_ids[-1]+1))})")

bracket_tok.save(BRACKET_TOKENIZER_PATH)
for copy_path in BRACKET_TOKENIZER_COPIES:
    os.makedirs(os.path.dirname(copy_path), exist_ok=True)
    shutil.copyfile(BRACKET_TOKENIZER_PATH, copy_path)
    print(f"copied bracket tokenizer -> {copy_path}")

with open(BRACKET_META_PATH, 'wb') as f:
    pickle.dump({'vocab_size': bracket_tok.get_vocab_size()}, f)
print(f"bracket vocab_size = {bracket_tok.get_vocab_size()} -> {BRACKET_META_PATH}")

for copy_path in BRACKET_META_COPIES:
    os.makedirs(os.path.dirname(copy_path), exist_ok=True)
    shutil.copyfile(BRACKET_META_PATH, copy_path)
    print(f"copied bracket meta.pkl -> {copy_path}")
