"""
Configuration for TinyLLM

This module contains all hyperparameters and configuration settings for the TinyLLM
training pipeline, including model architecture, training parameters, semantic
augmentation settings, and device configuration.
"""

import torch

# ============================================================
# MODEL ARCHITECTURE
# ============================================================

BATCH_SIZE = 4
"""Number of sequences processed in parallel during training."""

BLOCK_SIZE = 128
"""Maximum context length (number of tokens the model can attend to)."""

EMBED_DIM = 256
"""Dimensionality of token embeddings."""

SEMANTIC_DIM = 384
"""Dimensionality of semantic vectors from MiniLM."""

NUM_LAYERS = 4
"""Number of transformer encoder layers."""

NUM_HEADS = 8
"""Number of attention heads per transformer layer."""

# ============================================================
# TRAINING
# ============================================================

LEARNING_RATE = 3e-4
"""Initial learning rate for AdamW optimizer."""

MIN_LR = 1e-5
"""Minimum learning rate after cosine decay."""

MAX_ITERS = 50000
"""Total number of training iterations."""

WARMUP_STEPS = 500
"""Number of linear warmup steps before cosine decay."""

GRADIENT_CLIP = 1.0
"""Maximum norm for gradient clipping."""

WEIGHT_DECAY = 0.01
"""Weight decay coefficient for AdamW optimizer."""

# ============================================================
# SEMANTIC AUGMENTATION
# ============================================================

USE_SEMANTICS = True
"""Enable or disable semantic vector augmentation."""

SEM_GATE_INIT = 0.3
"""Initial value for the learnable semantic gate."""

SEM_GATE_MAX = 0.85
"""Maximum allowed semantic gate value."""

SEM_GATE_MIN = 0.15
"""Minimum allowed semantic gate value."""

SEMANTIC_DROPOUT = 0.15
"""Dropout rate for semantic vectors during training."""

# ============================================================
# CONTEXT BLENDING (Stage 1)
# ============================================================

SEM_CURRENT = 0.7
"""Weight for current token's semantic vector in context blending."""

SEM_PREV = 0.15
"""Weight for previous token's semantic vector in context blending."""

SEM_NEXT = 0.15
"""Weight for next token's semantic vector in context blending."""

# ============================================================
# ADAPTIVE SEMANTICS (Stage 3)
# ============================================================

ENABLE_ADAPTIVE_SEMANTICS = True
"""Enable automatic semantic gate adjustment based on output quality."""

COLLAPSE_UNIQUE_RATIO = 0.30
"""Minimum unique token ratio before collapse detection."""

COLLAPSE_REPEAT_RATIO = 0.45
"""Maximum repetition ratio before collapse detection."""

COLLAPSE_MIN_LENGTH = 12
"""Minimum token count before collapse detection."""

SEMANTIC_DECAY = 0.92
"""Multiplier for gate reduction during semantic collapse."""

SEMANTIC_RECOVERY = 1.01
"""Multiplier for gate increase during healthy output."""

# ============================================================
# DATA
# ============================================================

SHARD_SIZE = 1_000_000
"""Number of tokens per shard file."""

INPUT_FILE = "input.txt"
"""Path to the input text file for training."""

# ============================================================
# CHECKPOINTS
# ============================================================

EVAL_INTERVAL = 500
"""Number of steps between evaluation runs."""

SAVE_EVERY = 1000
"""Number of steps between checkpoint saves."""

# ============================================================
# GENERATION
# ============================================================

GEN_TEMPERATURE = 0.9
"""Default sampling temperature for text generation."""

GEN_TOP_K = 40
"""Default top-k sampling parameter for generation."""

GEN_MAX_NEW = 120
"""Default maximum number of new tokens to generate."""

# ============================================================
# DEVICE
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
"""Computation device ('cuda' or 'cpu')."""

# ============================================================
# VOCAB_SIZE (set after tokenizer loads)
# ============================================================

VOCAB_SIZE = None
"""Vocabulary size from GPT-2 tokenizer. Set dynamically at runtime."""


def set_vocab_size(size: int) -> None:
    """
    Set the vocabulary size after tokenizer initialization.

    Args:
        size: The vocabulary size from the tokenizer.
    """
    global VOCAB_SIZE
    VOCAB_SIZE = size
