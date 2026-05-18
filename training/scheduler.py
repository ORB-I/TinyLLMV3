"""
Learning rate scheduler with warmup and cosine decay
"""

import math


def get_lr(step, warmup_steps, max_iters, learning_rate, min_lr):
    """Cosine decay with warmup"""
    if step < warmup_steps:
        return learning_rate * step / warmup_steps

    progress = (step - warmup_steps) / (max_iters - warmup_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + cosine_decay * (learning_rate - min_lr)
