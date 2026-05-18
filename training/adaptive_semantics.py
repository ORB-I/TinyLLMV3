"""
Adaptive Semantic Regulation System with Arithmetic Compensation

This module provides automatic detection of semantic collapse and dynamic
adjustment of per-layer semantic gates. It includes arithmetic compensation
to prevent penalizing mathematically dense code.
"""

import re

import torch

# ============================================================
# CONFIG
# ============================================================

MAX_GATE = 0.85
"""Maximum allowed semantic gate value."""

MIN_GATE = 0.15
"""Minimum allowed semantic gate value."""

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

ARITHMETIC_TOLERANCE_MULTIPLIER = 1.5
"""Tolerance multiplier for mathematical code detection."""


def is_mathematical_code(text: str) -> bool:
    """
    Detect if the generated code is mathematical/scientific in nature.

    Mathematical code typically has high operator density, numbers, and equations.

    Args:
        text: The generated text to analyze.

    Returns:
        True if the text appears to be mathematical code, False otherwise.
    """
    math_patterns = [
        r"[+\-*/%=]",  # operators
        r"\d+",  # numbers
        r"\*\*",  # exponentiation
        r"sqrt|log|exp|sin|cos|tan|pi|e\^",  # math functions
        r"Matrix|Tensor|array",  # array/matrix operations
        r"Float|Int|int|float",  # numeric types
        r"==|!=|<=|>=",  # comparisons
        r"lambda|x\^|y\^",  # variables and exponents
        r"ctx\.|mpmath|sympy",  # math library calls
    ]

    score = 0
    for pattern in math_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score += 1

    return score >= 3


def analyze_output_quality(tokens: list, tokenizer) -> dict:
    """
    Detect semantic collapse or oversmoothing in generated output.

    Analyzes token diversity, repetition, length, and structural density.
    Mathematical code receives higher tolerance thresholds.

    Args:
        tokens: List of generated token IDs.
        tokenizer: GPT-2 tokenizer for decoding.

    Returns:
        Dictionary containing:
            - collapse (bool): Whether collapse was detected
            - reasons (list): Reasons for collapse detection
            - token_count (int): Number of tokens
            - unique_ratio (float): Ratio of unique tokens
            - repeat_ratio (float): Ratio of repeated tokens
            - structure_density (float): Ratio of structural characters
            - is_mathematical (bool): Whether code is mathematical
    """
    text = tokenizer.decode(tokens)

    token_count = len(tokens)
    unique_ratio = len(set(tokens)) / max(1, token_count)

    # Repetition metric
    repeats = 0
    for i in range(1, token_count):
        if tokens[i] == tokens[i - 1]:
            repeats += 1
    repeat_ratio = repeats / max(1, token_count)

    # Structural characters
    structural_chars = ["{", "}", "(", ")", "[", "]", ":", ",", ".", "=", "\n"]
    structure_count = sum(text.count(c) for c in structural_chars)
    structure_density = structure_count / max(1, len(text))

    # Detect if this is mathematical code
    is_math = is_mathematical_code(text)

    # Apply compensation factors for mathematical code
    if is_math:
        unique_threshold = COLLAPSE_UNIQUE_RATIO * ARITHMETIC_TOLERANCE_MULTIPLIER
        repeat_threshold = COLLAPSE_REPEAT_RATIO * ARITHMETIC_TOLERANCE_MULTIPLIER
        length_threshold = COLLAPSE_MIN_LENGTH * 0.5
    else:
        unique_threshold = COLLAPSE_UNIQUE_RATIO
        repeat_threshold = COLLAPSE_REPEAT_RATIO
        length_threshold = COLLAPSE_MIN_LENGTH

    # Collapse detection with compensation
    collapse_detected = False
    reasons = []

    if token_count < length_threshold:
        collapse_detected = True
        reasons.append("short_output")

    if unique_ratio < unique_threshold:
        collapse_detected = True
        reasons.append("low_diversity")

    if repeat_ratio > repeat_threshold:
        collapse_detected = True
        reasons.append("high_repetition")

    return {
        "collapse": collapse_detected,
        "reasons": reasons,
        "token_count": token_count,
        "unique_ratio": unique_ratio,
        "repeat_ratio": repeat_ratio,
        "structure_density": structure_density,
        "is_mathematical": is_math,
    }


def regulate_semantics(model, quality_metrics: dict) -> list:
    """
    Dynamically regulate per-layer semantic influence based on output quality.

    Adjusts gate values downward when collapse is detected, and gradually
    recovers them during healthy output. Mathematical code receives gentler
    regulation.

    Args:
        model: The TinyLLMSemantic model instance.
        quality_metrics: Dictionary from analyze_output_quality().

    Returns:
        List of new gate values after regulation.
    """
    # Get current gates (per-layer)
    if hasattr(model, "get_gates"):
        current_gates = model.get_gates()
        num_layers = len(current_gates)
    else:
        return None

    # Collapse response (but be gentler on mathematical code)
    if quality_metrics["collapse"]:
        if quality_metrics["is_mathematical"]:
            decay = SEMANTIC_DECAY * 0.95
            print(f"\n⚠️ Mathematical code - gentle regulation")
        else:
            decay = SEMANTIC_DECAY
            print(f"\n⚠️ Semantic saturation detected")

        print(f"   Reasons: {quality_metrics['reasons']}")
        print(f"   Is mathematical: {quality_metrics['is_mathematical']}")

        new_gates = []
        for i, g in enumerate(current_gates):
            new_g = max(g * decay, MIN_GATE)
            new_gates.append(new_g)
            model.set_gate(i, new_g)

        print(f"   Gates: {[f'{g:.3f}' for g in current_gates]}")
        print(f"        → {[f'{g:.3f}' for g in new_gates]}")
        return new_gates

    # Healthy output recovery (gradually increase gates)
    if any(g < MAX_GATE for g in current_gates):
        recovery = SEMANTIC_RECOVERY
        target_gates = [0.2 + 0.6 * (i / (num_layers - 1)) for i in range(num_layers)]

        new_gates = []
        for i, g in enumerate(current_gates):
            if g < target_gates[i]:
                new_g = min(g * recovery, target_gates[i])
            else:
                new_g = g
            new_gates.append(new_g)
            if new_g != g:
                model.set_gate(i, new_g)

        return new_gates

    return current_gates


def get_gate_summary(model) -> str:
    """
    Get a formatted summary of per-layer gates for display.

    Args:
        model: The TinyLLMSemantic model instance.

    Returns:
        Formatted string with ASCII bar chart of gate values.
    """
    if hasattr(model, "get_gates"):
        gates = model.get_gates()
        summary = []
        for i, g in enumerate(gates):
            bar = "█" * int(g * 20) + "░" * (20 - int(g * 20))
            summary.append(f"Layer {i}: {bar} {g:.3f}")
        return "\n".join(summary)
    return "No per-layer gates available"
