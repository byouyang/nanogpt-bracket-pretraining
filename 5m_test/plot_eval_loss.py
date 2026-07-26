"""
CLI: plot multiple runs' TensorBoard eval loss (val/loss by default) on one
matplotlib figure, x-axis in tokens seen rather than raw iteration, so
different arms (control/bracket/scrambled/bracket_transfer) -- which use
different batch_size/block_size -- can be compared on equal footing.

tokens seen at iteration N = N * batch_size * block_size, read from the
run's checkpoint (ckpt.pt next to the tensorboard log dir), not
hardcoded -- each arm's config differs, and bracket_transfer switches
block_size partway through (phase1_iters/block_size_clean in its config),
so its tokens-seen curve is piecewise, not a single multiply.

Reads TensorBoard event files via EventAccumulator (part of the tensorboard
package, already a repo dependency) -- no need to launch the TensorBoard UI.

Arm names resolve inside one split (split<SPLIT_SEED>/<arm>/tensorboard), but
an explicit path still works, so runs from other splits -- or the older
seed1337/seed1338 archives -- can be passed directly and mixed on one figure.

Usage:
    python plot_eval_loss.py                    # the 4 arms of the split in split_config.py
    python plot_eval_loss.py --split 43         # the 4 arms of split 43
    python plot_eval_loss.py bracket control    # only these two
    python plot_eval_loss.py split42/bracket split43/bracket   # same arm, two splits
    python plot_eval_loss.py --tag train/loss   # a different scalar tag
    python plot_eval_loss.py -o comparison.png  # save instead of showing
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sns.set_theme(context="poster", style="white")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from split_config import SPLIT_SEED

DEFAULT_RUNS = ['control', 'bracket', 'scrambled', 'bracket_transfer']


def has_event_files(d):
    return os.path.isdir(d) and any(f.startswith('events.out.tfevents') for f in os.listdir(d))


def resolve_log_dir(run, split):
    """Accept an arm name (resolved inside `split`), a run root dir (looks for
    tensorboard/ inside it), or a path straight to a tensorboard log dir.
    Checked from most to least specific, and each candidate must actually
    contain event files -- a run's root dir (e.g. "control") is itself a real
    directory but holds no events directly, so a plain os.path.isdir check
    would wrongly short-circuit before ever looking inside it.

    The `run/tensorboard` candidates are what keep explicit paths working:
    an old-layout archive like seed1337/bracket_out resolves through them."""
    candidates = [
        os.path.join(ROOT, f'split{split}', run, 'tensorboard'),
        os.path.join(run, 'tensorboard'),
        os.path.join(ROOT, run, 'tensorboard'),
        run,
    ]
    for c in candidates:
        if has_event_files(c):
            return c
    return None


def load_scalar(log_dir, tag):
    ea = EventAccumulator(log_dir, size_guidance={'scalars': 0})  # 0 = load all events
    ea.Reload()
    if tag not in ea.Tags().get('scalars', []):
        return None
    events = ea.Scalars(tag)
    return [e.step for e in events], [e.value for e in events]


def load_run_config(log_dir):
    """The checkpoint lives next to the tensorboard dir (<run>/tensorboard
    -> <run>/ckpt.pt) and carries the batch_size/block_size (and, for
    bracket_transfer, phase1_iters/block_size_clean) needed to convert
    iterations to tokens seen."""
    ckpt_path = os.path.join(os.path.dirname(log_dir), 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        return None
    ckpt = torch.load(ckpt_path, map_location='cpu')
    return ckpt.get('config')


def iters_to_tokens(steps, config):
    """tokens seen at iteration N = N * batch_size * block_size. For a
    two-phase run (bracket_transfer), block_size changes at phase1_iters, so
    tokens accumulate at one rate up to the boundary and a different rate
    after it -- config has phase1_iters/block_size_clean only in that case."""
    batch_size = config['batch_size']
    block_size = config['block_size']
    phase1_iters = config.get('phase1_iters')
    block_size_clean = config.get('block_size_clean')

    if phase1_iters is None or block_size_clean is None:
        return [it * batch_size * block_size for it in steps]

    phase1_tokens = phase1_iters * batch_size * block_size
    tokens = []
    for it in steps:
        if it <= phase1_iters:
            tokens.append(it * batch_size * block_size)
        else:
            tokens.append(phase1_tokens + (it - phase1_iters) * batch_size * block_size_clean)
    return tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('runs', nargs='*', default=DEFAULT_RUNS,
                         help="arm names / run dirs / tensorboard log dirs to plot (default: the 4 known arms)")
    parser.add_argument('--split', type=int, default=SPLIT_SEED, metavar='SPLIT_SEED',
                         help=f"split to resolve bare arm names in (default: {SPLIT_SEED}, from split_config.py)")
    parser.add_argument('--tag', default='val/loss', help="scalar tag to plot (default: val/loss)")
    parser.add_argument('-o', '--output', help="save figure to this path instead of showing it")
    args = parser.parse_args()

    plt.figure(figsize=(9, 6))
    plotted = 0
    for run in args.runs:
        log_dir = resolve_log_dir(run, args.split)
        if log_dir is None:
            print(f"skipping {run!r}: no tensorboard log dir found")
            continue
        data = load_scalar(log_dir, args.tag)
        if data is None:
            print(f"skipping {run!r}: tag {args.tag!r} not found in {log_dir}")
            continue
        config = load_run_config(log_dir)
        if config is None:
            print(f"skipping {run!r}: no ckpt.pt found next to {log_dir} to read batch_size/block_size from")
            continue
        steps, values = data
        tokens = iters_to_tokens(steps, config)
        plt.plot(tokens, values, label=run, marker='o', markersize=3)
        plotted += 1

    if plotted == 0:
        raise SystemExit("nothing to plot -- no runs had the requested tag")

    plt.xlabel('tokens seen')
    plt.gca().xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x / 1e6:.0f}M'))
    plt.ylabel(args.tag)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if args.output:
        plt.savefig(args.output, dpi=150, transparent=True)
        print(f"saved to {args.output}")
    else:
        plt.show()


if __name__ == '__main__':
    main()
