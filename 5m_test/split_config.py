"""
Single source of truth for the corpus-split seed, shared by split_corpus.py and
every arm's train.py. Change SPLIT_SEED here to reshuffle which documents land
in train vs. the held-out test set -- the axis this experiment varies (with
seed_config.SEED held fixed) to check whether a bpb gap survives a different
train/test draw or was an artifact of one.

SPLIT_SEED also names the directory each arm trains into
(../split<SPLIT_SEED>/<arm>), so two splits can never overwrite each other's
checkpoints, and run_experiments.py archives that split's held-out test sets
into the same directory. That pairing is the point: the test set is
split-specific, so scoring a checkpoint against a *different* split's test set
would be scoring it on documents it trained on -- a leak that looks like a
strong result rather than an error.

Distinct from seed_config.SEED, which controls training-run variance (weight
init, batch sampling, dropout) and does not touch corpus construction.
"""
SPLIT_SEED = 46
