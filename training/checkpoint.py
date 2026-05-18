"""
Checkpoint saving and loading
"""

import os

import torch


def save_checkpoint(model, optimizer, step, best_val_loss, filename="checkpoint.pt"):
    """Save full training state"""
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_val_loss": best_val_loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at step {step}")


def load_checkpoint(model, optimizer, filename="checkpoint.pt", device="cpu"):
    """Load training state if exists"""
    if os.path.exists(filename):
        print("\nLoading checkpoint...")
        checkpoint = torch.load(filename, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = checkpoint["step"]
        best_val_loss = checkpoint["best_val_loss"]
        print(f"Resumed from step {step}")
        return step, best_val_loss
    return 0, float("inf")
