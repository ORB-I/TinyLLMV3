"""
Base transformer components
"""

import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    """Learnable positional embeddings"""

    def __init__(self, block_size, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(block_size, embed_dim)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        return self.embed(pos)


class CausalMask:
    """Causal attention mask generator"""

    @staticmethod
    def create(T, device):
        return torch.triu(torch.ones(T, T, device=device) * float("-inf"), diagonal=1)


class TransformerBlock(nn.Module):
    """Single transformer encoder block"""

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, mask):
        x = x + self.attn(self.norm1(x), self.norm1(x), x, attn_mask=mask)[0]
        x = x + self.ff(self.norm2(x))
        return x
