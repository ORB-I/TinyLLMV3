"""
.semvec file loader
"""

import json
import struct
from pathlib import Path

import numpy as np
import torch


def load_semantic_vectors(semvec_dir, vocab_size, semantic_dim, device):
    """
    Load semantic vectors from .semvec shards.
    Returns tensor of shape (vocab_size, semantic_dim).
    """

    semvec_dir = Path(semvec_dir)
    index_path = semvec_dir / "index.json"

    if not semvec_dir.exists() or not index_path.exists():
        print("\n📚 No semantic vectors found.")
        return None

    print("\n📚 Loading semantic vectors...")

    with open(index_path, "r") as f:
        semvec_index = json.load(f)

    tensor = torch.zeros(vocab_size, semantic_dim, dtype=torch.float32)

    for shard_id in range(semvec_index["num_shards"]):
        shard_path = semvec_dir / f"semvec_shard_{shard_id:04d}.semvec"
        print(f"   Loading shard {shard_id}...")

        with open(shard_path, "rb") as f:
            magic = f.read(4)
            version = struct.unpack("<I", f.read(4))[0]
            vocab_size_file = struct.unpack("<I", f.read(4))[0]
            embed_dim = struct.unpack("<I", f.read(4))[0]
            num_tokens = struct.unpack("<I", f.read(4))[0]

            for _ in range(num_tokens):
                token_id = struct.unpack("<I", f.read(4))[0]
                token_len = struct.unpack("<H", f.read(2))[0]
                token_str = f.read(token_len).decode("utf-8")
                vec = np.frombuffer(f.read(embed_dim * 4), dtype=np.float32)
                tensor[token_id] = torch.tensor(vec)

    print(f"   Loaded {len(tensor.nonzero()):,} semantic vectors")
    print(f"   Tensor shape: {tensor.shape}")

    return tensor.to(device)
