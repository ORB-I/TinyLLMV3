#!/usr/bin/env python3
"""
Generate semantic vectors from token shards
"""

import torch

from data.tokenizer_utils import get_tokenizer
from preprocess import create_semantic_vectors


def main():
    print("=" * 60)
    print("TinyLLM - Semantic Vector Generation")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}")
        print(f"VRAM: {props.total_memory / 1e9:.2f} GB")

    # This will take ~10-20 minutes on CPU, ~5-10 on GPU
    create_semantic_vectors(device=device, max_tokens_per_shard=5000)

    print("\nSemantic vectors generated!")
    print("   Ready for training with")


if __name__ == "__main__":
    main()
