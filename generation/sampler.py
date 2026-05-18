"""
Token sampling utilities
"""

import torch


def sample_top_k(logits, top_k=40):
    """Top-k sampling"""
    if top_k is not None:
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = -float("Inf")
    return logits


def generate(model, context, max_new=100, temperature=0.8, top_k=40):
    """
    Generate tokens using the model.
    """
    model.eval()
    idx = context.clone()

    for _ in range(max_new):
        logits, _ = model(idx)
        logits = logits[:, -1, :] / temperature
        logits = sample_top_k(logits, top_k)
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
