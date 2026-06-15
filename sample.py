"""Generate Dante-style text from a trained checkpoint.

Loads ``out/ckpt.pt`` (produced by ``train.py``) and the vocabulary
metadata (``data/meta.pkl``), then autoregressively samples characters.

Examples::

    # Generate 500 characters starting from scratch:
    python sample.py --num_samples=1 --max_new_tokens=500

    # Prompt the model with a starting string:
    python sample.py --start="Nel mezzo del cammin"

    # Lower temperature -> more conservative, higher -> more creative:
    python sample.py --temperature=0.7 --top_k=40
"""

import argparse
import os
import pickle

import torch

from model import GPT, GPTConfig


def parse_args():
    p = argparse.ArgumentParser(description="Sample from the Dante GPT.")
    p.add_argument("--out_dir", type=str, default="out")
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument(
        "--start", type=str, default="\n", help="Prompt text to start generation from."
    )
    p.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of independent samples to generate.",
    )
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=500,
        help="Number of characters to generate per sample.",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (<1 conservative, >1 creative).",
    )
    p.add_argument(
        "--top_k",
        type=int,
        default=40,
        help="Keep only the top_k most likely tokens (0 = disabled).",
    )
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = args.device

    # Load vocabulary metadata.
    meta_path = os.path.join(args.data_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{meta_path} not found. Run `python data/prepare.py` first."
        )
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]

    def encode(s):
        # Unknown characters are skipped so an arbitrary prompt still works.
        return [stoi[c] for c in s if c in stoi]

    def decode(ids):
        return "".join(itos[int(i)] for i in ids)

    # Load the trained checkpoint.
    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"{ckpt_path} not found. Train a model with `python train.py` first."
        )
    # weights_only=False because our checkpoint is a dict with more than just
    # tensors (optimizer state, config, model_args). PyTorch >= 2.6 defaults to
    # weights_only=True; this file is one we produced ourselves, so it's trusted.
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    # Strip a possible torch.compile prefix.
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # Build the starting context.
    start_ids = encode(args.start)
    if not start_ids:
        start_ids = [stoi.get("\n", 0)]
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    top_k = args.top_k if args.top_k > 0 else None
    print(
        f"Generating {args.num_samples} sample(s) from checkpoint "
        f"(iter {checkpoint.get('iter_num', '?')}, "
        f"val loss {checkpoint.get('best_val_loss', float('nan')):.4f}).\n"
    )

    with torch.no_grad():
        for i in range(args.num_samples):
            y = model.generate(
                x, args.max_new_tokens, temperature=args.temperature, top_k=top_k
            )
            print(f"===== sample {i + 1} =====")
            print(decode(y[0].tolist()))
            print()


if __name__ == "__main__":
    main()
