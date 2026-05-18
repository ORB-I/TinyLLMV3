#!/usr/bin/env python3
"""
TinyLLM - Preprocessing Script

Converts input.txt to token shards and optionally generates semantic vectors
using the MiniLM model. This script should be run before training.
"""

import json
import struct
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from config import INPUT_FILE, SHARD_SIZE
from data.tokenizer_utils import get_tokenizer

# ============================================================
# CONFIG
# ============================================================

TOKEN_SHARDS_DIR = Path("token_shards")
"""Directory for token shard files."""

SEMVEC_SHARDS_DIR = Path("semvec_shards")
"""Directory for semantic vector shard files."""

SEMANTIC_DIM = 384
"""Dimensionality of semantic vectors from MiniLM."""


def clean_text(text: str) -> str:
    """
    Clean text by replacing problematic UTF-8 artifacts.

    Args:
        text: Raw input text.

    Returns:
        Cleaned text with common encoding issues fixed.
    """
    replacements = {
        "â€™": "'",
        "â€œ": '"',
        "â€ ": '"',
        "â€˜": "'",
        "â€¦": "...",
        "Â": " ",
        "\xa0": " ",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def create_token_shards() -> None:
    """
    Convert input.txt to token shards (.pt files).

    Reads the input file in chunks, tokenizes with GPT-2, and saves
    shards of SHARD_SIZE tokens each. Skips if shards already exist.
    """
    TOKEN_SHARDS_DIR.mkdir(exist_ok=True)

    existing = list(TOKEN_SHARDS_DIR.glob("shard_*.pt"))
    if existing:
        print(f"\nFound {len(existing)} existing token shards")
        return

    print("\nCreating token shards...")
    tokenizer = get_tokenizer()

    shard_index = 0
    current_tokens = []
    chunk_size = 2_000_000

    with open(INPUT_FILE, "r", encoding="latin-1") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break

            chunk = clean_text(chunk)
            tokens = tokenizer.encode(chunk, add_special_tokens=False)
            current_tokens.extend(tokens)

            while len(current_tokens) >= SHARD_SIZE:
                shard_tokens = current_tokens[:SHARD_SIZE]
                current_tokens = current_tokens[SHARD_SIZE:]
                shard_path = TOKEN_SHARDS_DIR / f"shard_{shard_index}.pt"
                torch.save(torch.tensor(shard_tokens, dtype=torch.int32), shard_path)
                print(f"   Saved shard {shard_index} ({len(shard_tokens):,} tokens)")
                shard_index += 1

    if current_tokens:
        shard_path = TOKEN_SHARDS_DIR / f"shard_{shard_index}.pt"
        torch.save(torch.tensor(current_tokens, dtype=torch.int32), shard_path)
        print(f"   Saved final shard {shard_index} ({len(current_tokens):,} tokens)")

    print(f"\nCreated {shard_index + 1} token shards")


def create_semantic_vectors(
    device: str = "cuda", max_tokens_per_shard: int = 5000
) -> None:
    """
    Create semantic vectors for unique tokens using MiniLM.

    This function collects all unique tokens from the token shards, then
    generates 384-dimensional semantic vectors using the MiniLM model.
    Results are saved as .semvec shard files.

    Args:
        device: Device to run the MiniLM model on ('cuda' or 'cpu').
        max_tokens_per_shard: Maximum number of tokens per .semvec shard.
    """
    from transformers import AutoModel, AutoTokenizer

    SEMVEC_SHARDS_DIR.mkdir(exist_ok=True)

    existing = list(SEMVEC_SHARDS_DIR.glob("semvec_shard_*.semvec"))
    if existing:
        print(f"\nFound {len(existing)} existing semantic shards")
        return

    print("\nCreating semantic vectors...")

    # Load tokenizers
    gpt2_tokenizer = get_tokenizer()
    sem_tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    sem_encoder = AutoModel.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2"
    ).to(device)
    sem_encoder.eval()

    # First pass: collect unique tokens from all shards
    print("\n   First pass: collecting unique tokens...")
    unique_tokens = set()

    for shard_path in TOKEN_SHARDS_DIR.glob("shard_*.pt"):
        tokens = torch.load(shard_path, map_location="cpu", weights_only=True)
        unique_tokens.update(tokens.tolist())

    unique_tokens = sorted(unique_tokens)
    print(f"   Found {len(unique_tokens):,} unique tokens")

    @torch.no_grad()
    def get_semantic_batch(texts: list, batch_size: int = 32) -> np.ndarray:
        """
        Generate semantic embeddings for a batch of texts.

        Args:
            texts: List of text strings.
            batch_size: Number of texts per batch.

        Returns:
            Array of semantic vectors.
        """
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            inputs = sem_tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding=True,
            ).to(device)
            outputs = sem_encoder(**inputs)
            batch_vectors = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            vectors.extend(batch_vectors)
        return np.array(vectors)

    print(f"\n   Generating semantic vectors for {len(unique_tokens):,} tokens...")

    # Process in shards
    for i in range(0, len(unique_tokens), max_tokens_per_shard):
        shard_tokens = unique_tokens[i : i + max_tokens_per_shard]
        shard_id = i // max_tokens_per_shard

        # Get token strings
        token_strings = [gpt2_tokenizer.decode([tid]) for tid in shard_tokens]
        token_strings = [s.replace("\x00", "") or " " for s in token_strings]

        # Generate vectors
        vectors = get_semantic_batch(token_strings)

        # Write shard
        shard_path = SEMVEC_SHARDS_DIR / f"semvec_shard_{shard_id:04d}.semvec"

        with open(shard_path, "wb") as f:
            f.write(b"SEMV")
            f.write(struct.pack("<I", 1))
            f.write(struct.pack("<I", gpt2_tokenizer.vocab_size))
            f.write(struct.pack("<I", SEMANTIC_DIM))
            f.write(struct.pack("<I", len(shard_tokens)))

            for token_id, token_str, vec in zip(shard_tokens, token_strings, vectors):
                token_bytes = token_str.encode("utf-8")
                f.write(struct.pack("<I", token_id))
                f.write(struct.pack("<H", len(token_bytes)))
                f.write(token_bytes)
                f.write(vec.astype(np.float32).tobytes())

            checksum = 0
            for tid in shard_tokens:
                checksum ^= tid
            f.write(struct.pack("<I", checksum))

        print(
            f"   Saved shard {shard_id} ({len(shard_tokens):,} tokens, {shard_path.stat().st_size / 1e6:.2f} MB)"
        )

    # Create index
    index = {
        "version": 1,
        "gpt2_vocab_size": gpt2_tokenizer.vocab_size,
        "semantic_dim": SEMANTIC_DIM,
        "num_shards": (len(unique_tokens) + max_tokens_per_shard - 1)
        // max_tokens_per_shard,
        "shard_size": max_tokens_per_shard,
        "total_unique_tokens": len(unique_tokens),
    }

    with open(SEMVEC_SHARDS_DIR / "index.json", "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nCreated semantic vector shards")
    print(f"   Total unique tokens: {len(unique_tokens):,}")
    print(f"   Total shards: {index['num_shards']}")


def main() -> None:
    """Main preprocessing entry point."""
    print("=" * 60)
    print("TinyLLM - Preprocessing Pipeline")
    print("=" * 60)

    # Step 1: Create token shards from input.txt
    create_token_shards()

    # Step 2 (optional): Create semantic vectors
    # Uncomment if you have the dependencies and want semantics
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # create_semantic_vectors(device=device)

    print("Preprocessing complete!")
    print("\nNext step: python train.py")


if __name__ == "__main__":
    main()
