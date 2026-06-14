"""Prepare the Divine Comedy dataset for character-level training.

This script:
  1. Downloads the Italian text of Dante's "Divina Commedia" from
     Project Gutenberg (gutenberg.org).
  2. Cleans it up: strips the Gutenberg license header/footer, removes
     editorial notes, line numbers and other noise.
  3. Builds a character-level vocabulary (the encoding) and splits the
     data into a 90 / 10 train / validation split.
  4. Saves the encoded data to ``train.bin`` / ``val.bin`` and the
     vocabulary metadata to ``meta.pkl`` next to this file.

Run it from the repo root with::

    python data/prepare.py

It only needs to be run once. The produced ``*.bin`` and ``meta.pkl``
files are what ``train.py`` consumes.
"""

import os
import pickle
import re

import numpy as np
import requests

# Directory of this file, so the script works regardless of the cwd.
HERE = os.path.dirname(os.path.abspath(__file__))

# Project Gutenberg edition of the Divina Commedia (Italian, plain UTF-8).
# Gutenberg ebook #1012 is the complete Divina Commedia by Dante Alighieri.
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/1012/pg1012.txt"

RAW_PATH = os.path.join(HERE, "divina_commedia_raw.txt")
CLEAN_PATH = os.path.join(HERE, "divina_commedia_clean.txt")


def download(url: str, dest: str) -> str:
    """Download ``url`` to ``dest`` (cached: skips if already present)."""
    if os.path.exists(dest):
        print(f"Raw text already downloaded at {dest}, skipping download.")
        with open(dest, encoding="utf-8") as f:
            return f.read()

    print(f"Downloading Divine Comedy text from {url} ...")
    headers = {"User-Agent": "dante-nanogpt/1.0 (educational project)"}
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    # Gutenberg serves UTF-8; be explicit so accented Italian chars survive.
    resp.encoding = "utf-8"
    text = resp.text
    with open(dest, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Saved raw text to {dest} ({len(text):,} characters).")
    return text


def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove the Project Gutenberg license header and footer."""
    start_marker = re.compile(
        r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE
    )
    end_marker = re.compile(
        r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", re.IGNORECASE
    )

    start = start_marker.search(text)
    if start:
        text = text[start.end() :]
    end = end_marker.search(text)
    if end:
        text = text[: end.start()]
    return text


def clean(text: str) -> str:
    """Clean the raw text: drop notes, line numbers and other noise.

    The goal is to keep only Dante's verses so the model learns the
    poetic style rather than editorial apparatus.
    """
    text = strip_gutenberg_boilerplate(text)

    cleaned_lines = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()

        # Remove leading verse/line numbers like "  3" or "12 " that some
        # editions print in the margin, plus trailing standalone numbers.
        line = re.sub(r"^\s*\d+\s+", "", line)
        line = re.sub(r"\s+\d+\s*$", "", line)

        stripped = line.strip()

        # Skip footnote / editorial-note markers such as "[1]" or "(1)".
        if re.fullmatch(r"[\[\(]\d+[\]\)]", stripped):
            continue

        # Skip whole footnote/endnote definition lines, which begin with a
        # marker followed by the note text, e.g. "[1] questa è una nota...".
        if re.match(r"^\s*[\[\(]\d+[\]\)]\s+\S", line):
            continue

        # Skip lines that are only digits (page or verse numbers).
        if re.fullmatch(r"\d+", stripped):
            continue

        # Drop inline footnote references like "word[12]" -> "word".
        line = re.sub(r"\[\d+\]", "", line)

        cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)

    # Collapse 3+ consecutive blank lines into at most two.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip() + "\n"


def main() -> None:
    raw = download(GUTENBERG_URL, RAW_PATH)
    data = clean(raw)

    with open(CLEAN_PATH, "w", encoding="utf-8") as f:
        f.write(data)
    print(f"Saved cleaned text to {CLEAN_PATH} ({len(data):,} characters).")

    # Build the character-level vocabulary.
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size} unique characters.")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s: str):
        return [stoi[c] for c in s]

    # 90 / 10 train / val split.
    n = len(data)
    split = int(n * 0.9)
    train_data = data[:split]
    val_data = data[split:]

    train_ids = np.array(encode(train_data), dtype=np.uint16)
    val_ids = np.array(encode(val_data), dtype=np.uint16)

    train_ids.tofile(os.path.join(HERE, "train.bin"))
    val_ids.tofile(os.path.join(HERE, "val.bin"))
    print(f"train has {len(train_ids):,} tokens")
    print(f"val has {len(val_ids):,} tokens")

    # Save the metadata needed to encode/decode at train and sample time.
    meta = {
        "vocab_size": vocab_size,
        "stoi": stoi,
        "itos": itos,
    }
    with open(os.path.join(HERE, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"Saved vocab metadata to {os.path.join(HERE, 'meta.pkl')}.")


if __name__ == "__main__":
    main()
