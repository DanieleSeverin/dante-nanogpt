"""Training loop for the Dante character-level GPT.

Features:
  - loads the encoded data produced by ``data/prepare.py``
  - trains with AdamW and a cosine learning-rate schedule with warmup
  - periodically evaluates train/val loss and prints it
  - periodically generates a short sample so you can watch the model
    learn to write Dante-style verses over time
  - checkpoints the best model to ``out/ckpt.pt``

All hyper-parameters live in the CONFIG section below or can be
overridden from the command line, e.g.::

    python train.py --max_iters=5000 --n_layer=4 --device=cuda

Run ``python data/prepare.py`` first to create the dataset.
"""

import argparse
import os
import pickle
import time

import numpy as np
import torch

from model import GPT, GPTConfig

# -----------------------------------------------------------------------------
# Default configuration (good for a small/quick run). Override via CLI flags.
# -----------------------------------------------------------------------------
DEFAULTS = dict(
    # I/O
    data_dir="data",
    out_dir="out",
    eval_interval=250,  # how often to evaluate + maybe checkpoint
    log_interval=10,  # how often to print the training loss
    eval_iters=100,  # batches used to estimate the loss
    sample_interval=500,  # how often to print a generated sample
    always_save_checkpoint=False,
    # data / model
    block_size=256,
    batch_size=32,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    # optimizer
    learning_rate=3e-4,
    max_iters=5000,
    weight_decay=1e-1,
    beta1=0.9,
    beta2=0.99,
    grad_clip=1.0,
    # learning-rate schedule (cosine decay with warmup)
    decay_lr=True,
    warmup_iters=100,
    min_lr=3e-5,
    # system
    device="cuda" if torch.cuda.is_available() else "cpu",
    seed=1337,
    compile=False,  # set True with PyTorch 2.0 for a speedup
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Dante GPT.")

    def str2bool(x):
        return x.lower() in ("1", "true", "yes")

    for key, val in DEFAULTS.items():
        if isinstance(val, bool):
            parser.add_argument(f"--{key}", type=str2bool, default=val)
        else:
            parser.add_argument(f"--{key}", type=type(val), default=val)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = vars(args)

    torch.manual_seed(cfg["seed"])
    device = cfg["device"]
    device_type = "cuda" if "cuda" in device else "cpu"
    os.makedirs(cfg["out_dir"], exist_ok=True)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------
    data_dir = cfg["data_dir"]
    meta_path = os.path.join(data_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"{meta_path} not found. Run `python data/prepare.py` first."
        )
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    vocab_size = meta["vocab_size"]
    itos = meta["itos"]
    print(f"Loaded vocabulary of {vocab_size} characters.")

    train_data = np.memmap(
        os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")

    def get_batch(split):
        data = train_data if split == "train" else val_data
        ix = torch.randint(len(data) - cfg["block_size"], (cfg["batch_size"],))
        x = torch.stack(
            [
                torch.from_numpy((data[i : i + cfg["block_size"]]).astype(np.int64))
                for i in ix
            ]
        )
        y = torch.stack(
            [
                torch.from_numpy(
                    (data[i + 1 : i + 1 + cfg["block_size"]]).astype(np.int64)
                )
                for i in ix
            ]
        )
        if device_type == "cuda":
            x, y = (
                x.pin_memory().to(device, non_blocking=True),
                y.pin_memory().to(device, non_blocking=True),
            )
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_args = dict(
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        block_size=cfg["block_size"],
        dropout=cfg["dropout"],
        vocab_size=vocab_size,
    )
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    model.to(device)

    optimizer = model.configure_optimizers(
        cfg["weight_decay"],
        cfg["learning_rate"],
        (cfg["beta1"], cfg["beta2"]),
        device_type,
    )

    if cfg["compile"]:
        print("Compiling the model (this may take a minute)...")
        model = torch.compile(model)

    # Mixed precision on CUDA for speed/memory; plain float32 on CPU.
    use_amp = device_type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ctx = (
        torch.autocast(device_type=device_type, dtype=torch.bfloat16)
        if use_amp
        else _nullcontext()
    )

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        for split in ("train", "val"):
            losses = torch.zeros(cfg["eval_iters"])
            for k in range(cfg["eval_iters"]):
                X, Y = get_batch(split)
                with ctx:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean().item()
        model.train()
        return out

    def get_lr(it):
        # 1) linear warmup
        if it < cfg["warmup_iters"]:
            return cfg["learning_rate"] * (it + 1) / (cfg["warmup_iters"] + 1)
        # 2) after decay, return the minimum learning rate
        if it > cfg["max_iters"]:
            return cfg["min_lr"]
        # 3) cosine decay in between
        decay_ratio = (it - cfg["warmup_iters"]) / (
            cfg["max_iters"] - cfg["warmup_iters"]
        )
        coeff = 0.5 * (1.0 + np.cos(np.pi * decay_ratio))
        return cfg["min_lr"] + coeff * (cfg["learning_rate"] - cfg["min_lr"])

    def generate_sample(num_tokens=300):
        model.eval()
        start = torch.zeros((1, 1), dtype=torch.long, device=device)
        with torch.no_grad(), ctx:
            raw = model.generate(start, num_tokens, temperature=0.8, top_k=40)
        text = "".join(itos[int(i)] for i in raw[0].tolist())
        model.train()
        return text

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"Starting training on {device} for {cfg['max_iters']} iterations.")
    best_val_loss = float("inf")
    t0 = time.time()

    for it in range(cfg["max_iters"] + 1):
        # Set the learning rate for this iteration.
        lr = get_lr(it) if cfg["decay_lr"] else cfg["learning_rate"]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Periodic evaluation + checkpointing.
        if it % cfg["eval_interval"] == 0:
            losses = estimate_loss()
            print(
                f"step {it}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}, lr {lr:.2e}"
            )
            if losses["val"] < best_val_loss or cfg["always_save_checkpoint"]:
                best_val_loss = min(best_val_loss, losses["val"])
                if it > 0:
                    checkpoint = {
                        "model": (
                            model._orig_mod if hasattr(model, "_orig_mod") else model
                        ).state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "model_args": model_args,
                        "iter_num": it,
                        "best_val_loss": best_val_loss,
                        "config": cfg,
                    }
                    ckpt_path = os.path.join(cfg["out_dir"], "ckpt.pt")
                    torch.save(checkpoint, ckpt_path)
                    print(f"  saved checkpoint to {ckpt_path}")

        # Periodic sample preview.
        if it % cfg["sample_interval"] == 0 and it > 0:
            print("-" * 60)
            print(generate_sample())
            print("-" * 60)

        # Forward / backward / update.
        X, Y = get_batch("train")
        with ctx:
            _, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if cfg["grad_clip"] != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        if it % cfg["log_interval"] == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(
                f"iter {it}: loss {loss.item():.4f}, "
                f"time {dt * 1000 / max(1, cfg['log_interval']):.1f}ms/iter"
            )

    print("Training complete.")
    print(f"Best validation loss: {best_val_loss:.4f}")


class _nullcontext:
    """Tiny no-op context manager for the CPU (float32) path."""

    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False


if __name__ == "__main__":
    main()
