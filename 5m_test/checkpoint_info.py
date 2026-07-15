"""
CLI: print metadata from a nanoGPT-style .pt checkpoint produced by any of
this repo's train.py variants (control/bracket/scrambled/bracket_transfer).
Loads onto CPU and never reconstructs the model class -- just inspects the
saved dict -- so it works without a GPU and without model.py on the path.

Usage:
    python checkpoint_info.py <path/to/ckpt.pt>
"""
import argparse
import os

import torch


def count_params(state_dict):
    """Sum tensor.numel() across a state_dict, de-duplicating tied weights
    (e.g. wte/lm_head share one Parameter) by underlying storage pointer so
    they aren't double-counted."""
    seen = set()
    total = 0
    for t in state_dict.values():
        storage = t.untyped_storage() if hasattr(t, 'untyped_storage') else t.storage()
        ptr = storage.data_ptr()
        if ptr in seen:
            continue
        seen.add(ptr)
        total += t.numel()
    return total


def main():
    parser = argparse.ArgumentParser(description="Print metadata from a train.py checkpoint (.pt file).")
    parser.add_argument('checkpoint', help="path to ckpt.pt")
    args = parser.parse_args()

    size_mb = os.path.getsize(args.checkpoint) / (1024 ** 2)
    ckpt = torch.load(args.checkpoint, map_location='cpu')

    print(f"checkpoint: {args.checkpoint}")
    print(f"file size:  {size_mb:.1f} MB")
    print()

    print("training state:")
    print(f"  iter_num:       {ckpt.get('iter_num', 'N/A')}")
    print(f"  best_val_loss:  {ckpt.get('best_val_loss', 'N/A')}")
    if 'phase' in ckpt:  # bracket_transfer's two-phase checkpoints only
        print(f"  phase:          {ckpt['phase']}")
        print(f"  phase1_iters:   {ckpt.get('phase1_iters', 'N/A')}")
        print(f"  phase2_iters:   {ckpt.get('phase2_iters', 'N/A')}")
    print()

    model_args = ckpt.get('model_args', {})
    if model_args:
        print("model_args:")
        for k, v in model_args.items():
            print(f"  {k}: {v}")
        print()

    state_dict = ckpt.get('model')
    if state_dict:
        total = count_params(state_dict)
        print(f"parameters: {total:,} ({total / 1e6:.2f}M)")
        wpe_key = next((k for k in state_dict if k.endswith('wpe.weight')), None)
        if wpe_key:
            non_embedding = total - state_dict[wpe_key].numel()
            print(f"parameters, non-embedding (matches train.py's log line): "
                  f"{non_embedding:,} ({non_embedding / 1e6:.2f}M)")
        print()

    config = ckpt.get('config', {})
    if config:
        print("config:")
        for k in sorted(config):
            print(f"  {k}: {config[k]}")


if __name__ == '__main__':
    main()
