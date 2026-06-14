# Dante nanoGPT 🪶

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DanieleSeverin/dante-nanogpt/blob/main/notebooks/dante_nanogpt.ipynb)
<!-- ⚠️ Placeholder badge: the Colab notebook (notebooks/dante_nanogpt.ipynb) will be added later. -->

A small, from-scratch, **character-level transformer** (nanoGPT-style)
trained on the text of Dante's *Divine Comedy* to generate verses in
Dante's style.

> ## 📚 This is an EDUCATIONAL project
>
> The goal of this repository is **to learn how transformer-based Large
> Language Models (LLMs) work internally**, by building one step by step
> from scratch. It is **not** meant for production use. Every component —
> the data pipeline, the GPT architecture, the training loop, and the
> sampling code — is intentionally small and heavily commented so it can
> be read and understood end to end.

---

## 🎯 Project goal & how it works

We train a tiny GPT (a decoder-only transformer, a few million
parameters) **completely from scratch**:

- **Character-level**: the model works on individual characters, not
  words or sub-word tokens. The vocabulary is just the set of characters
  that appear in the text (letters, punctuation, accented Italian vowels,
  newlines…). This keeps the tokenizer trivial and the focus on the
  transformer itself.
- **From-scratch training**: weights start random. We do **not** fine-tune
  any pretrained model — the network learns Italian spelling, Dante's
  vocabulary, and the rhythm of *terzine* purely from the raw text.
- **Self-supervised next-character prediction**: the only training signal
  is "given the previous characters, predict the next one". From this
  simple objective the model gradually learns to produce Dante-like
  verses.

This is the same core idea behind modern LLMs, just shrunk down so it can
be trained on a laptop or a free Colab GPU and understood in full.

---

## 🗂️ Repository structure

```
dante-nanogpt/
├── data/
│   └── prepare.py      # download + clean the Divine Comedy, build the char encoding & train/val split
├── model.py            # the GPT architecture (attention, MLP, blocks) in PyTorch
├── train.py            # training loop: checkpointing, loss logging, sample previews
├── sample.py           # generate text from a trained checkpoint
├── requirements.txt    # dependencies (torch, numpy, requests)
└── README.md           # this file
```

Files produced at runtime (ignored by git):

```
data/train.bin, data/val.bin, data/meta.pkl   # encoded dataset + vocabulary
out/ckpt.pt                                    # trained model checkpoint
```

---

## ⚙️ Setup

Requires Python 3.9+.

```bash
# (optional) create a virtual environment
python -m venv .venv && source .venv/bin/activate

# install dependencies
pip install -r requirements.txt
```

---

## 1️⃣ Prepare the data

This downloads the Italian text of the *Divina Commedia* from
[Project Gutenberg](https://www.gutenberg.org/), cleans it (removes the
license header/footer, editorial notes, line numbers…), builds the
character-level vocabulary and writes a 90 / 10 train/validation split.

```bash
python data/prepare.py
```

You only need to run this once. It creates `data/train.bin`,
`data/val.bin` and `data/meta.pkl`.

---

## 2️⃣ Train the model

```bash
python train.py
```

During training you'll see the training loss logged periodically, the
train/val loss at every evaluation interval, and — every so often — a
**generated sample** so you can watch the model learn to write like Dante.
The best checkpoint (by validation loss) is saved to `out/ckpt.pt`.

All hyper-parameters can be overridden from the command line, e.g.
`python train.py --max_iters=3000 --n_layer=4 --device=cpu`.

### Suggested parameters

**🐢 Quick test run (CPU or a small GPU)** — just to verify the whole
pipeline works and see the model start to produce Italian-looking text.
This is small and fast, the output won't be very good:

```bash
python train.py \
  --device=cpu \
  --n_layer=4 --n_head=4 --n_embd=128 \
  --block_size=128 --batch_size=16 \
  --max_iters=2000 --eval_interval=250 --sample_interval=500
```

On a CPU this is slow but tractable; on any small GPU it's quick.

**🚀 Serious run (Colab / a proper GPU, e.g. T4)** — produces noticeably
better, more coherent Dante-style verses:

```bash
python train.py \
  --device=cuda \
  --n_layer=6 --n_head=6 --n_embd=384 \
  --block_size=256 --batch_size=64 \
  --max_iters=5000 --eval_interval=250 --sample_interval=500 \
  --learning_rate=3e-4 --compile=True
```

Feel free to push `n_layer`, `n_embd`, `block_size` and `max_iters`
higher if you have more GPU memory and time — that's part of the fun of
the experiment.

---

## 3️⃣ Generate text

Once you have a checkpoint in `out/ckpt.pt`:

```bash
# a few samples starting from scratch
python sample.py --num_samples=3 --max_new_tokens=500

# start from a prompt
python sample.py --start="Nel mezzo del cammin di nostra vita"

# tune creativity: lower temperature = safer, higher = wilder
python sample.py --temperature=0.7 --top_k=40
```

---

## ☁️ Run on Colab

A Colab notebook will be added later under `notebooks/`. The badge at the
top of this README is a **placeholder** for it — once the notebook is in
place, clicking the badge will open the whole prepare → train → sample
workflow on a free GPU.

---

## 🙏 Acknowledgements

- The architecture is closely inspired by Andrej Karpathy's excellent
  [nanoGPT](https://github.com/karpathy/nanoGPT) and his "Let's build GPT"
  lecture.
- The text of the *Divine Comedy* comes from
  [Project Gutenberg](https://www.gutenberg.org/) (public domain).

---

## 📄 License

The code in this repository is provided for educational purposes. The
text of the *Divine Comedy* is in the public domain via Project Gutenberg.
