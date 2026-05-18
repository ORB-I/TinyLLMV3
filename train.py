#!/usr/bin/env python3
"""
TinyLLM - Main Training Script

This script orchestrates the complete training pipeline for TinyLLM, including
data loading, model initialization, training loop, checkpointing, and
adaptive semantic regulation.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from config import *
from data.semantic_loader import load_semantic_vectors
from data.shard_loader import ShardDataset
from data.tokenizer_utils import VOCAB_SIZE, get_tokenizer
from generation.sampler import generate
from models.tinyllm_semantic import TinyLLMSemantic
from training.adaptive_semantics import (
    analyze_output_quality,
    get_gate_summary,
    regulate_semantics,
)
from training.checkpoint import load_checkpoint, save_checkpoint
from training.scheduler import get_lr


def main() -> None:
    """Main training entry point."""
    print(f"\nUsing device: {DEVICE}")
    if DEVICE == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}")
        print(f"VRAM: {props.total_memory / 1e9:.2f} GB")

    # Tokenizer
    tokenizer = get_tokenizer()

    # Semantic vectors
    semantic_tensor = None
    use_semantics = USE_SEMANTICS and os.path.exists("semvec_shards")

    if use_semantics:
        semantic_tensor = load_semantic_vectors(
            "semvec_shards", VOCAB_SIZE, SEMANTIC_DIM, DEVICE
        )
        if semantic_tensor is None:
            use_semantics = False

    # Create token shards if needed
    DATA_DIR = Path("token_shards")
    DATA_DIR.mkdir(exist_ok=True)

    if not list(DATA_DIR.glob("shard_*.pt")):
        print("\n❌ No token shards found!")
        print("   Please run: python preprocess.py")
        exit(1)

    shards = sorted(DATA_DIR.glob("shard_*.pt"))
    print(f"\nTotal shards: {len(shards)}")

    # Dataset
    dataset = ShardDataset(shards, BLOCK_SIZE)

    # Get vocab size
    vocab_size = tokenizer.vocab_size

    # Model
    model = TinyLLMSemantic(vocab_size=vocab_size, use_semantics=use_semantics).to(
        DEVICE
    )
    if use_semantics:
        model.set_semantic_tensor(semantic_tensor)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n Model parameters: {total_params:,}")
    if use_semantics:
        print(f"   Per-layer semantic gates (learnable):")
        print(f"      Layer 0: ████████░░░░░░░░░░░░ 0.30 (initial)")
        print(f"      Layer 1: ████████░░░░░░░░░░░░ 0.30 (initial)")
        print(f"      Layer 2: ████████░░░░░░░░░░░░ 0.30 (initial)")
        print(f"      Layer 3: ████████░░░░░░░░░░░░ 0.30 (initial)")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    # Load checkpoint
    start_step, best_val_loss = load_checkpoint(
        model, optimizer, "checkpoint.pt", DEVICE
    )

    # Training loop
    print("\n Starting training...\n")

    for step in range(start_step, MAX_ITERS):
        # Random shard rotation
        if step % 2000 == 0 and step > 0:
            dataset.load_random_shard()

        # Learning rate
        lr = get_lr(step, WARMUP_STEPS, MAX_ITERS, LEARNING_RATE, MIN_LR)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Get batch
        xb, yb = dataset.get_batch(BATCH_SIZE)
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)

        # Get semantics for batch if enabled
        sem = None
        if use_semantics:
            x_flat = xb.flatten()
            sem_flat = semantic_tensor[x_flat]
            sem = sem_flat.view(BATCH_SIZE, BLOCK_SIZE, -1)

        # Forward pass
        _, loss = model(xb, sem, yb)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()

        # Print progress
        if step % 10 == 0:
            if use_semantics:
                gates = model.get_gates()
                gate_str = (
                    f"[{gates[0]:.2f}|{gates[1]:.2f}|{gates[2]:.2f}|{gates[3]:.2f}]"
                )
            else:
                gate_str = "N/A"
            print(
                f"step {step:6d}/{MAX_ITERS} | loss {loss.item():.4f} | lr {lr:.6f} | gates {gate_str}"
            )

        # Evaluation and generation
        if step % EVAL_INTERVAL == 0 and step > 0:
            # Evaluate
            model.eval()
            eval_losses = []
            for _ in range(10):
                xb_eval, yb_eval = dataset.get_batch(BATCH_SIZE)
                xb_eval, yb_eval = xb_eval.to(DEVICE), yb_eval.to(DEVICE)
                if use_semantics:
                    sem_eval = semantic_tensor[xb_eval.flatten()].view(
                        BATCH_SIZE, BLOCK_SIZE, -1
                    )
                    _, loss_eval = model(xb_eval, sem_eval, yb_eval)
                else:
                    _, loss_eval = model(xb_eval, None, yb_eval)
                eval_losses.append(loss_eval.item())
            avg_loss = np.mean(eval_losses)

            print(f"\n--- Step {step} ---")
            print(f"Train loss: {loss.item():.4f} | Val loss: {avg_loss:.4f}")

            if avg_loss < best_val_loss:
                best_val_loss = avg_loss
                torch.save(model.state_dict(), "best_model.pt")
                print(f"✓ Saved best model (val loss: {best_val_loss:.4f})")

            # Generate sample
            context = torch.tensor(
                [[tokenizer.eos_token_id]], dtype=torch.long, device=DEVICE
            )
            output = generate(
                model,
                context,
                max_new=GEN_MAX_NEW,
                temperature=GEN_TEMPERATURE,
                top_k=GEN_TOP_K,
            )
            sample = tokenizer.decode(output[0].tolist())
            print(f"\nSample:\n{sample}\n")

            # Show gate summary
            if use_semantics:
                print(f" Per-layer gates:\n{get_gate_summary(model)}")

            # Adaptive semantics
            if ENABLE_ADAPTIVE_SEMANTICS and use_semantics:
                metrics = analyze_output_quality(output[0].tolist(), tokenizer)
                regulate_semantics(model, metrics)

            model.train()

        # Save checkpoint
        if step % SAVE_EVERY == 0 and step > 0:
            save_checkpoint(model, optimizer, step, best_val_loss, "checkpoint.pt")

    print("\n✓ Training complete!")
    torch.save(model.state_dict(), "final_model.pt")
    print("Saved final_model.pt")


if __name__ == "__main__":
    main()
