#!/usr/bin/env python3
"""Improve generation parameters and add safety checks."""

import json
from pathlib import Path

def create_improved_generation_config():
    """Create a safer generation configuration."""

    config = {
        "max_new_tokens": 100,  # Reduced from 150
        "min_new_tokens": 10,   # Reduced from 20
        "do_sample": True,
        "temperature": 0.8,     # Slightly higher for more diversity
        "top_p": 0.95,          # Higher for more token options
        "top_k": 50,            # Add top-k sampling
        "repetition_penalty": 1.15,  # Reduced from 1.2
        "pad_token_id": 151643,   # Explicit pad token
        "eos_token_id": 151645,   # Explicit EOS token
        "no_repeat_ngram_size": 3,  # Prevent repetition
        "early_stopping": True,      # Stop when EOS is generated
        "bad_words_ids": [[151642]],  # Prevent the problematic "⽗" token
        # Add constraints to keep tokens in valid range
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
    }

    # Save the config
    with open("generation_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print("Improved generation config saved to generation_config.json")
    return config

if __name__ == "__main__":
    config = create_improved_generation_config()
    print("\nRecommended generation parameters:")
    for k, v in config.items():
        print(f"  {k}: {v}")