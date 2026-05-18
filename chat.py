#!/usr/bin/env python3
"""
TinyLLM - Chat Interface

Loads your trained model and provides an interactive chat interface.
Supports commands for adjusting temperature, max length, and semantic gates.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import *
from data.semantic_loader import load_semantic_vectors
from data.tokenizer_utils import VOCAB_SIZE, get_tokenizer
from models.tinyllm_semantic import TinyLLMSemantic

# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "final_model.pt"
"""Path to the trained model checkpoint."""

USE_SEMANTICS = True
"""Whether to use semantic augmentation during inference."""


@torch.no_grad()
def generate_response(
    prompt: str, max_new: int = 200, temperature: float = 0.8, top_k: int = 40
) -> str:
    """
    Generate a response from the creature for a given prompt.

    Args:
        prompt: The user's input text.
        max_new: Maximum number of new tokens to generate.
        temperature: Sampling temperature (higher = more creative).
        top_k: Top-k sampling parameter.

    Returns:
        The generated response as a string.
    """
    # Tokenize prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    context = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)

    # Truncate if too long
    if context.shape[1] > BLOCK_SIZE:
        context = context[:, -BLOCK_SIZE:]

    # Generate
    output = model.generate(
        context, max_new=max_new, temperature=temperature, top_k=top_k
    )

    # Decode response (remove prompt)
    response = tokenizer.decode(output[0].tolist()[len(input_ids) :])

    return response.strip()


def main() -> None:
    """Main chat interface entry point."""
    global tokenizer, model

    print("\n" + "=" * 60)
    print(" TinyLLM - Chat Interface")
    print("=" * 60)

    # Load tokenizer
    tokenizer = get_tokenizer()

    # Load semantic vectors if available
    semantic_tensor = None
    use_semantics = USE_SEMANTICS

    if use_semantics:
        print("\n Loading semantic vectors...")
        semantic_tensor = load_semantic_vectors(
            "semvec_shards", VOCAB_SIZE, SEMANTIC_DIM, DEVICE
        )
        if semantic_tensor is None:
            use_semantics = False
            print("   No semantic vectors found. Running in token-only mode.")
        else:
            print(f"   ✓ Loaded {semantic_tensor.shape[0]:,} semantic vectors")

    # Load model
    print("\n Loading model...")
    model = TinyLLMSemantic(vocab_size=VOCAB_SIZE, use_semantics=use_semantics).to(
        DEVICE
    )

    # Load trained weights
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint)
    model.eval()

    if use_semantics:
        model.set_semantic_tensor(semantic_tensor)

    total_params = sum(p.numel() for p in model.parameters())
    gate_value = model.semantic_gate.value() if use_semantics else None

    print(f"\n✓ Model loaded: {total_params:,} parameters")
    if use_semantics:
        print(f"✓ Semantic gate: {gate_value:.3f}")
    print(f"✓ Device: {DEVICE}")

    # Chat loop
    print("\n" + "=" * 60)
    print("Ready to chat! Type your messages below.")
    print("   Commands:")
    print("     /temp [0.5-1.5]  - Adjust temperature (creativity)")
    print("     /max [N]         - Adjust max response length")
    print("     /gate [0-1]      - Set semantic gate (if enabled)")
    print("     /gates           - Show per-layer gate values (if enabled)")
    print("     /reset           - Reset to defaults")
    print("     /quit            - Exit")
    print("=" * 60)

    # Defaults
    temperature = 0.8
    max_new = 200

    while True:
        try:
            user_input = input("\n\033[36mYou:\033[0m ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "/quit":
                print("\nGoodbye! The creature rests.")
                break

            elif user_input.lower() == "/reset":
                temperature = 0.8
                max_new = 200
                print("✓ Reset to defaults (temp=0.8, max=200)")
                continue

            elif user_input.startswith("/temp"):
                try:
                    temperature = float(user_input.split()[1])
                    temperature = max(0.3, min(1.5, temperature))
                    print(f"✓ Temperature set to {temperature}")
                except:
                    print("Usage: /temp 0.8")
                continue

            elif user_input.startswith("/max"):
                try:
                    max_new = int(user_input.split()[1])
                    max_new = max(10, min(500, max_new))
                    print(f"✓ Max length set to {max_new}")
                except:
                    print("Usage: /max 200")
                continue

            elif user_input.startswith("/gate") and use_semantics:
                try:
                    new_gate = float(user_input.split()[1])
                    new_gate = max(0.01, min(0.99, new_gate))
                    model.semantic_gate.set_value(new_gate)
                    print(f"✓ Semantic gate set to {new_gate:.3f}")
                except:
                    print(
                        f"Usage: /gate 0.3 (current: {model.semantic_gate.value():.3f})"
                    )
                continue

            elif user_input.lower() == "/gates" and use_semantics:
                gates = model.get_gates()
                print("\nPer-layer semantic gates:")
                for i, g in enumerate(gates):
                    bar = "█" * int(g * 20) + "░" * (20 - int(g * 20))
                    print(f"   Layer {i}: {bar} {g:.3f}")
                continue

            # Generate response
            print("\n\033[33mCreature:\033[0m ", end="", flush=True)

            response = generate_response(
                user_input, max_new=max_new, temperature=temperature
            )

            print(response)

            # Optional: show gate value after each response
            if use_semantics and temperature > 0.9:
                print(
                    f"\n\033[90m[gate={model.semantic_gate.value():.3f} | temp={temperature} | max={max_new}]\033[0m"
                )

        except KeyboardInterrupt:
            print("\nGoodbye! The creature rests.")
            break
        except Exception as e:
            print(f"\n\033[31mError: {e}\033[0m")


if __name__ == "__main__":
    main()
