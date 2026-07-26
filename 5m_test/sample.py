"""
Sample from each trained model (control, bracket, scrambled, bracket_transfer)
for one train/test split.

Usage:
    python sample.py             # the split in split_config.py
    python sample.py --split 43  # a specific archived split
"""
import argparse
import json
import os
import sys

import torch
from tokenizers import Tokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'control'))
from split_config import SPLIT_SEED
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
start = 'Action was a federal agency established by President Richard Nixon on 1 July 1971. Its intention was to make the service organizations established during the 1960s operate more'
num_samples = 1
max_new_tokens = 10
temperature = 0.8
top_k = 200
seed = 42  # sampling RNG only -- nothing to do with the split seed
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# -----------------------------------------------------------------------------


def build_models(split):
    """Checkpoints and vocabs both come from the split's own directory. The
    working tokenizer/ dir belongs to whichever split ran last, so decoding an
    older split's checkpoint with it would silently emit the wrong text."""
    d = os.path.join(ROOT, f'split{split}')
    control_tokenizer = os.path.join(d, 'tokenizer', 'tokenizer.json')
    bracket_tokenizer = os.path.join(d, 'tokenizer', 'bracket_tokenizer.json')
    return [
        ('control', os.path.join(d, 'control', 'ckpt.pt'), control_tokenizer),
        ('bracket', os.path.join(d, 'bracket', 'ckpt.pt'), bracket_tokenizer),
        ('scrambled', os.path.join(d, 'scrambled', 'ckpt.pt'), bracket_tokenizer),
        ('bracket_transfer', os.path.join(d, 'bracket_transfer', 'ckpt.pt'), bracket_tokenizer),
    ]


def load_bracket_token_ids(tokenizer_path):
    with open(tokenizer_path, encoding='utf-8') as f:
        tokenizer_config = json.load(f)
    return {token['id'] for token in tokenizer_config.get('added_tokens', []) if token.get('special')}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--split', type=int, default=SPLIT_SEED, metavar='SPLIT_SEED',
                        help=f"which split's runs to sample from (default: {SPLIT_SEED}, from split_config.py)")
    args = parser.parse_args()

    torch.manual_seed(seed)

    for name, ckpt_path, tokenizer_path in build_models(args.split):
        print(f"===== {name} (split {args.split}) =====")
        if not os.path.exists(ckpt_path):
            print(f"skipped, no checkpoint at {ckpt_path}")
            continue

        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)

        tok = Tokenizer.from_file(tokenizer_path)
        bracket_token_ids = load_bracket_token_ids(tokenizer_path)
        encode = lambda s: tok.encode(s).ids
        decode = lambda ids: tok.decode(ids, skip_special_tokens=False)

        x = torch.tensor(encode(start), dtype=torch.long, device=device)[None, ...]

        with torch.no_grad():
            for k in range(num_samples):
                generation_max_new_tokens = max_new_tokens * 3 if name in {'bracket', 'scrambled'} else max_new_tokens
                y = model.generate(x, generation_max_new_tokens, temperature=temperature, top_k=top_k)
                ids = [token_id for token_id in y[0].tolist() if token_id not in bracket_token_ids]
                print(decode(ids))
                print('---------------')


if __name__ == '__main__':
    main()
