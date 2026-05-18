"""
Semantic augmented TinyLLM model with per-layer gating.

This module uses GPT-2 subword BPE tokenization (50,257 vocab) with semantic
augmentation from MiniLM (384-dim vectors). Each token gets both:
- Token embedding (syntax) from the GPT-2 token ID
- Semantic vector (meaning) from precomputed MiniLM embeddings

The model learns to balance syntax and semantics through per-layer gates.
"""

import torch
import torch.nn as nn

from config import (
    EMBED_DIM,
    NUM_HEADS,
    NUM_LAYERS,
    SEM_CURRENT,
    SEM_NEXT,
    SEM_PREV,
    SEMANTIC_DIM,
    SEMANTIC_DROPOUT,
)
from models.semantic_gate import PerLayerSemanticGate
from models.transformer import CausalMask, PositionalEmbedding, TransformerBlock


class TinyLLMSemantic(nn.Module):
    """
    TinyLLM with semantic augmentation, context blending, and per-layer learnable gates.

    This model extends a standard transformer by adding semantic vectors from a
    precomputed encoder (MiniLM) to the token embeddings. It includes:
    - Context blending: local semantic context from neighboring tokens
    - Per-layer gates: each transformer layer learns its own semantic weight
    - Semantic dropout: prevents over-reliance on semantics during training

    Attributes:
        vocab_size (int): Size of the token vocabulary.
        use_semantics (bool): Whether to use semantic augmentation.
        block_size (int): Maximum context length.
        token_emb (nn.Embedding): Token embedding layer.
        pos_emb (PositionalEmbedding): Positional embedding layer.
        semantic_proj (nn.Linear): Projects semantic vectors to embed space.
        semantic_gates (PerLayerSemanticGate): Per-layer learnable gates.
        blocks (nn.ModuleList): Transformer encoder blocks.
        norm (nn.LayerNorm): Final layer normalization.
        head (nn.Linear): Output projection to vocabulary.
        semantic_tensor (torch.Tensor): Cached semantic tensor for generation.
    """

    def __init__(
        self, vocab_size: int, use_semantics: bool = True, block_size: int = 128
    ):
        """
        Initialize the TinyLLMSemantic model.

        Args:
            vocab_size: Size of the token vocabulary.
            use_semantics: Whether to enable semantic augmentation. Defaults to True.
            block_size: Maximum context length. Defaults to 128.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.use_semantics = use_semantics
        self.block_size = block_size

        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, EMBED_DIM)

        # Positional embedding
        self.pos_emb = PositionalEmbedding(block_size, EMBED_DIM)

        # Semantic projection (if semantics enabled)
        if use_semantics:
            self.semantic_proj = nn.Linear(SEMANTIC_DIM, EMBED_DIM)
            self.semantic_gates = PerLayerSemanticGate(NUM_LAYERS, init_gate=0.3)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(EMBED_DIM, NUM_HEADS) for _ in range(NUM_LAYERS)]
        )

        # Output head
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, vocab_size)

        # Cache for context blending
        self.semantic_tensor = None

    def set_semantic_tensor(self, tensor: torch.Tensor) -> None:
        """
        Set the semantic tensor for generation.

        Args:
            tensor: Semantic tensor of shape (vocab_size, SEMANTIC_DIM).
        """
        self.semantic_tensor = tensor

    def blend_context_semantics(self, sem: torch.Tensor) -> torch.Tensor:
        """
        Blend semantic vectors with neighboring tokens for local context.

        The blending uses configurable weights (SEM_CURRENT, SEM_PREV, SEM_NEXT)
        to incorporate semantic information from adjacent tokens.

        Args:
            sem: Semantic tensor of shape (batch_size, seq_len, SEMANTIC_DIM).

        Returns:
            Blended semantic tensor of the same shape.
        """
        sem_prev = torch.roll(sem, 1, dims=1)
        sem_next = torch.roll(sem, -1, dims=1)
        sem_prev[:, 0, :] = sem[:, 0, :]
        sem_next[:, -1, :] = sem[:, -1, :]
        return SEM_CURRENT * sem + SEM_PREV * sem_prev + SEM_NEXT * sem_next

    def semantic_dropout(self, x_semantic: torch.Tensor) -> torch.Tensor:
        """
        Randomly disable semantic influence during training.

        This prevents the model from becoming over-reliant on semantic vectors.

        Args:
            x_semantic: Projected semantic vectors.

        Returns:
            The same tensor with random elements zeroed out during training.
        """
        if not self.training or not self.use_semantics:
            return x_semantic
        mask = (torch.rand_like(x_semantic[..., :1]) > SEMANTIC_DROPOUT).float()
        return x_semantic * mask

    def get_gates(self) -> list:
        """
        Return list of current per-layer gate values.

        Returns:
            List of gate values for each layer (0-1 range).
        """
        if self.use_semantics:
            return self.semantic_gates.get_all_gates()
        return [0.0] * NUM_LAYERS

    def set_gate(self, layer_idx: int, value: float) -> None:
        """
        Set the semantic gate for a specific layer.

        Args:
            layer_idx: Index of the layer to modify.
            value: New gate value (clamped to 0.01-0.99).
        """
        if self.use_semantics:
            self.semantic_gates.set_gate(layer_idx, value)

    def forward(
        self,
        idx: torch.Tensor,
        semantic_vectors: torch.Tensor = None,
        targets: torch.Tensor = None,
    ) -> tuple:
        """
        Forward pass through the model.

        Args:
            idx: Input token indices of shape (batch_size, seq_len).
            semantic_vectors: Semantic vectors for each token, shape (batch_size, seq_len, SEMANTIC_DIM).
            targets: Target token indices for loss calculation, shape (batch_size, seq_len).

        Returns:
            Tuple of (logits, loss) where logits has shape (batch_size, seq_len, vocab_size)
            and loss is the cross-entropy loss if targets provided, else None.
        """
        B, T = idx.shape

        # Token embedding
        x = self.token_emb(idx)

        # Positional embedding
        x = x + self.pos_emb(idx)

        # Semantic augmentation (project once, apply per-layer gates)
        if self.use_semantics and semantic_vectors is not None:
            x_semantic = self.semantic_proj(semantic_vectors)
            x_semantic = self.semantic_dropout(x_semantic)
            gates = self.semantic_gates()
        else:
            x_semantic = None
            gates = None

        # Causal mask
        mask = CausalMask.create(T, idx.device)

        # Transformer blocks with per-layer semantic gating
        for i, block in enumerate(self.blocks):
            if x_semantic is not None and gates is not None:
                gate = gates[i].view(1, 1, 1)
                x = x + gate * x_semantic
            x = block(x, mask)

        # Output head
        x = self.norm(x)
        logits = self.head(x)

        # Loss calculation
        loss = None
        if targets is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size), targets.view(-1)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new: int = 100,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.

        Args:
            idx: Starting token indices of shape (1, seq_len).
            max_new: Maximum number of new tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_k: Top-k sampling parameter (filter to top k logits).

        Returns:
            Tensor of generated token indices.
        """
        self.eval()

        for _ in range(max_new):
            idx_cond = idx[:, -self.block_size :]

            # Get semantics if available
            if self.use_semantics and self.semantic_tensor is not None:
                sem_cond = self.semantic_tensor[idx_cond]
                sem_cond = self.blend_context_semantics(sem_cond)
                logits, _ = self(idx_cond, sem_cond)
            else:
                logits, _ = self(idx_cond)

            # Sampling
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
