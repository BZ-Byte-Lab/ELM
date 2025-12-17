#!/usr/bin/env python3
"""Debug generation issues."""

import sys
from pathlib import Path
import torch

# Add paths for modules
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "summary_training_pipeline"))

from summary_training_pipeline.dataset import ELMTrainingDataset, TrainingCollator
from transformers import AutoTokenizer

def debug_tokenization():
    """Check if tokenizer is working correctly."""

    # Load tokenizer
    model_name = "Qwen/Qwen3-4B-Instruct-2507"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add special tokens
    if "<EMB>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<EMB>"]})

    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

    # Test some example texts
    test_texts = [
        "This is a test summary.",
        "The model is generating repetitive characters.",
        "⽗⽗⽗⽗⽗⽗⽗⽗"  # These are the problematic characters
    ]

    for text in test_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"\nOriginal: {text}")
        print(f"Tokens: {tokens}")
        print(f"Decoded: {decoded}")
        print(f"Token by token:")
        for token in tokens[:10]:  # Show first 10
            print(f"  {token} -> '{tokenizer.decode([token])}'")

if __name__ == "__main__":
    debug_tokenization()