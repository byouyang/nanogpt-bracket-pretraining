"""
Sample from each trained model (control, bracket, scrambled, bracket_transfer).
"""
import os
import sys
import torch
from tokenizers import Tokenizer

ROOT = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(ROOT, 'control'))
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
start = "An important, if costly, symbolic victory for the Allies during World War II,"
num_samples = 1
max_new_tokens = 200
temperature = 0.8
top_k = 200
seed = 42
device = 'cuda' if torch.cuda.is_available() else 'cpu'

MODELS = [
    ('control', os.path.join(ROOT, 'control', 'out', 'ckpt.pt'), os.path.join(ROOT, 'control', 'tokenizer.json')),
    ('bracket', os.path.join(ROOT, 'bracket', 'out', 'ckpt.pt'), os.path.join(ROOT, 'bracket', 'bracket_tokenizer.json')),
    ('scrambled', os.path.join(ROOT, 'scrambled', 'out', 'ckpt.pt'), os.path.join(ROOT, 'scrambled', 'bracket_tokenizer.json')),
    ('bracket_transfer', os.path.join(ROOT, 'bracket_transfer', 'out', 'ckpt.pt'), os.path.join(ROOT, 'bracket_transfer', 'bracket_tokenizer.json')),
]
# -----------------------------------------------------------------------------

torch.manual_seed(seed)

for name, ckpt_path, tokenizer_path in MODELS:
    print(f"===== {name} =====")

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
    encode = lambda s: tok.encode(s).ids
    decode = lambda ids: tok.decode(ids, skip_special_tokens=False)

    x = torch.tensor(encode(start), dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
