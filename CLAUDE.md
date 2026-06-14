# Project: dante-nanogpt

## Context
Educational project for a fullstack developer learning how LLMs work
internally, with the long-term goal of being able to pretrain small
models and fine-tune (SFT) larger ones. This repo is the first step:
a from-scratch, character-level, nanoGPT-style transformer trained on
Dante's Divine Comedy.

## Current status
All core files exist and pass `python -m py_compile` (syntax-checked).
None of it has been run end-to-end yet (no training, no data download).

- `data/prepare.py` — downloads the Divina Commedia (Italian) from Project
  Gutenberg (ebook #1012), strips the license header/footer, removes
  editorial notes / verse & page numbers, builds a character-level vocab
  and a 90/10 train/val split. Outputs `data/train.bin`, `data/val.bin`,
  `data/meta.pkl`. **Untested against the live download / actual text.**
- `model.py` — minimal configurable GPT (`GPTConfig` with `n_layer`,
  `n_head`, `n_embd`, `block_size`, `dropout`, `bias`). Causal multi-head
  self-attention (uses PyTorch flash attention when available, manual
  masked fallback otherwise), MLP, pre-norm blocks, weight tying,
  AdamW optimizer grouping, and a `generate()` with temperature/top-k.
- `train.py` — training loop: AdamW + cosine LR schedule with warmup,
  periodic train/val loss eval, best-checkpoint saving to `out/ckpt.pt`,
  loss logging, and periodic generated-sample previews. All
  hyperparameters overridable via CLI flags. **Untested (not run).**
- `sample.py` — loads `out/ckpt.pt` + `data/meta.pkl` and generates text,
  with optional `--start` prompt, `--temperature`, `--top_k`,
  `--num_samples`. **Untested (no checkpoint exists yet).**
- `requirements.txt` — `torch`, `numpy`, `requests`.
- `README.md` — English; states the educational goal, how it works,
  repo structure, run instructions, suggested params (CPU/small-GPU quick
  test vs Colab GPU serious run), and a placeholder "Open in Colab" badge.
- `.gitignore` — ignores generated artifacts (`*.bin`, `meta.pkl`,
  `out/`, checkpoints, downloaded text).

Generated artifacts (`data/*.bin`, `data/meta.pkl`, `out/`) are git-ignored
and do **not** exist in the repo until the scripts are run.

## Constraints & conventions
- README and all code comments in English
- Actual training will run on Google Colab (free GPU tier), so any code
  changes must stay Colab-compatible (no exotic dependencies, reasonable
  memory footprint)
- The user works mostly from the Claude Code mobile app: prefer small,
  focused changes over large refactors; avoid long-running commands
  (no real training runs here - CPU smoke tests with tiny params are fine)
- After meaningful changes: commit and push (origin is already configured)
- Keep things simple and well-commented - clarity over cleverness, since
  the goal is learning

## Roadmap (update checkboxes as items are completed)
- [x] Validate data/prepare.py end-to-end, confirm clean train/val splits
      (clean/encode/split logic validated via a smoke test; fixed a bug where
      whole footnote-definition lines like "[1] note text" were not removed.
      The real Gutenberg download could NOT be run in this environment because
      gutenberg.org is not in the network egress allowlist — it will run on
      Colab / locally. Add www.gutenberg.org to the allowlist to run it here.)
- [x] Add a Colab notebook (notebooks/dante_nanogpt.ipynb: clone → install →
      prepare → train → sample) + "Open in Colab" badge wired in README
- [ ] GitHub Action: lint/format (ruff/black) on push
- [ ] GitHub Action: CPU smoke test (few steps, tiny config) on push/PR
- [ ] Real training run on Colab GPU; save sample generations
      (e.g. samples/ folder or README section)
- [ ] (Optional) hyperparameter experiments, document results in README

## Working agreement
- Start each session by reading this file and the roadmap
- Pick up the next unchecked roadmap item unless the user asks for
  something else
- Update the roadmap checkboxes as you go
- If a request isn't covered here, use judgment consistent with the
  "educational project" framing above
