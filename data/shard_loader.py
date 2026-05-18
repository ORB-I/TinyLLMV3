"""
Token shard management

This module provides a dataset class that loads token shards randomly during
training, enabling efficient memory usage and random access to the full dataset.
"""

import random
from pathlib import Path

import torch


class ShardDataset:
    """
    Loads token shards randomly during training.

    Instead of loading the entire dataset into memory, this class loads one
    shard at a time and randomly rotates through shards, enabling efficient
    training on large datasets with limited memory.

    Attributes:
        shards (list): List of paths to shard files.
        block_size (int): Length of context window.
        current_shard_idx (int): Index of currently loaded shard.
        tokens (torch.Tensor): Currently loaded token tensor.
    """

    def __init__(self, shards: list, block_size: int):
        """
        Initialize the shard dataset.

        Args:
            shards: List of paths to .pt shard files.
            block_size: Length of context window for batches.
        """
        self.shards = shards
        self.block_size = block_size
        self.current_shard_idx = -1
        self.tokens = None
        self.load_random_shard()

    def load_random_shard(self) -> None:
        """
        Load a random shard from disk.

        Skips reloading the currently loaded shard. Prints progress information.
        """
        new_idx = random.randint(0, len(self.shards) - 1)
        if new_idx == self.current_shard_idx:
            return
        self.current_shard_idx = new_idx
        print(f"\nLoading shard {new_idx}...")
        self.tokens = torch.load(
            self.shards[new_idx], map_location="cpu", weights_only=True
        )
        print(f"Loaded {len(self.tokens):,} tokens")

    def get_batch(self, batch_size: int) -> tuple:
        """
        Get a random batch of token sequences from the current shard.

        Args:
            batch_size: Number of sequences to return.

        Returns:
            Tuple of (input_ids, target_ids) tensors.
        """
        max_idx = len(self.tokens) - self.block_size - 1
        starts = torch.randint(0, max_idx, (batch_size,))
        indices = starts.unsqueeze(1) + torch.arange(self.block_size, device="cpu")
        x = self.tokens[indices]
        y = self.tokens[indices + 1]
        return x.long(), y.long()
