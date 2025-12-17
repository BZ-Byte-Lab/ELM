#!/usr/bin/env python3
"""Extract embeddings for each split from the full embeddings file."""

import json
import numpy as np
from pathlib import Path
from safetensors.numpy import load_file, save_file

def main():
    # Load mapping info
    with open("data/summary_clean_fixed/mapping_info.json", 'r') as f:
        mapping_info = json.load(f)

    # Load full embeddings
    print("Loading full embeddings...")
    full_embeddings = load_file("data/embeddings/train_embeddings.safetensors")["embeddings"]
    print(f"Full embeddings shape: {full_embeddings.shape}")

    # Create output directory
    output_dir = Path("data/summary_clean_fixed/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract train embeddings
    print("Extracting train embeddings...")
    train_mapping = mapping_info["train_mapping"]
    train_embeddings = np.zeros((len(train_mapping), full_embeddings.shape[1]), dtype=full_embeddings.dtype)

    for original_idx, new_idx in train_mapping.items():
        train_embeddings[new_idx] = full_embeddings[int(original_idx)]

    save_file({"embeddings": train_embeddings}, str(output_dir / "train_embeddings.safetensors"))
    print(f"Saved train embeddings: {train_embeddings.shape}")

    # Extract val embeddings
    print("Extracting val embeddings...")
    val_mapping = mapping_info["val_mapping"]
    val_embeddings = np.zeros((len(val_mapping), full_embeddings.shape[1]), dtype=full_embeddings.dtype)

    for original_idx, new_idx in val_mapping.items():
        val_embeddings[new_idx] = full_embeddings[int(original_idx)]

    save_file({"embeddings": val_embeddings}, str(output_dir / "val_embeddings.safetensors"))
    print(f"Saved val embeddings: {val_embeddings.shape}")

    # Extract test embeddings
    print("Extracting test embeddings...")
    test_mapping = mapping_info["test_mapping"]
    test_embeddings = np.zeros((len(test_mapping), full_embeddings.shape[1]), dtype=full_embeddings.dtype)

    for original_idx, new_idx in test_mapping.items():
        test_embeddings[new_idx] = full_embeddings[int(original_idx)]

    save_file({"embeddings": test_embeddings}, str(output_dir / "test_embeddings.safetensors"))
    print(f"Saved test embeddings: {test_embeddings.shape}")

    print("\nAll embeddings extracted and saved!")

if __name__ == "__main__":
    main()