"""
Tokenizer utilities using GPT-2 subword BPE tokenizer.

This module provides a singleton wrapper around the GPT-2 tokenizer,
which uses Byte-Pair Encoding (BPE) subword tokenization. The vocabulary
size is 50,257 tokens, covering common words, subwords, and characters.
"""

from transformers import AutoTokenizer

_tokenizer = None


def get_tokenizer():
    """
    Get the GPT-2 tokenizer singleton.

    The GPT-2 tokenizer uses subword BPE (Byte-Pair Encoding), which splits
    text into common subword units. For example:
    - "unhappiness" -> ["un", "happiness"]
    - "tokenization" -> ["token", "ization"]

    Returns:
        AutoTokenizer: GPT-2 tokenizer with 50,257 vocab size.
    """
    global _tokenizer
    if _tokenizer is None:
        print("\nLoading GPT-2 subword BPE tokenizer...")
        _tokenizer = AutoTokenizer.from_pretrained("gpt2")
        _tokenizer.pad_token = _tokenizer.eos_token
        print(f"   Vocab size: {_tokenizer.vocab_size:,} (subword BPE)")
        print(f"   Example: 'tokenization' -> {_tokenizer.encode('tokenization')}")
    return _tokenizer


VOCAB_SIZE = get_tokenizer().vocab_size
