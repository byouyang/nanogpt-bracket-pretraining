"""
Single source of truth for the training-run seed, shared by control/train.py,
bracket/train.py, scrambled/train.py, and bracket_transfer/train.py. Change
SEED here to reseed every arm's training run (weight init, batch sampling,
dropout) at once -- for e.g. running each arm across multiple seeds to check
whether a bpb gap holds up or is run-to-run noise.

Does NOT affect corpus construction (split_corpus.py's SEED, or
scramble_brackets.py's --seed) -- those control dataset splits/scrambling,
a separate concern from training-run variance.
"""
SEED = 1338
