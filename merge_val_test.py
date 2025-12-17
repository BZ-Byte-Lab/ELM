#!/usr/bin/env python3
"""Merge validation and test sets for larger evaluation."""

import json
from pathlib import Path

def merge_val_test():
    """Merge validation and test sets."""

    data_dir = Path("data/summary_clean_fixed")

    # Load val and test samples
    val_samples = []
    with open(data_dir / "val_synthesis_clean.jsonl") as f:
        for line in f:
            val_samples.append(json.loads(line))

    test_samples = []
    with open(data_dir / "test_synthesis_clean.jsonl") as f:
        for line in f:
            test_samples.append(json.loads(line))

    print(f"Val samples: {len(val_samples)}")
    print(f"Test samples: {len(test_samples)}")

    # Re-index test samples to continue after val
    max_val_idx = max(s["embedding_index"] for s in val_samples)
    for sample in test_samples:
        sample["embedding_index"] += max_val_idx + 1

    # Merge them
    merged_samples = val_samples + test_samples
    print(f"Merged: {len(merged_samples)} samples")

    # Save merged validation set
    with open(data_dir / "val_synthesis_merged.jsonl", 'w') as f:
        for sample in merged_samples:
            f.write(json.dumps(sample) + '\n')

    # Also need to merge the embeddings
    import numpy as np
    from safetensors.numpy import load_file, save_file

    val_emb = load_file(str(data_dir / "embeddings/val_embeddings.safetensors"))["embeddings"]
    test_emb = load_file(str(data_dir / "embeddings/test_embeddings.safetensors"))["embeddings"]

    merged_emb = np.concatenate([val_emb, test_emb], axis=0)
    save_file({"embeddings": merged_emb}, str(data_dir / "embeddings/val_embeddings_merged.safetensors"))

    print(f"Merged embeddings shape: {merged_emb.shape}")
    print("\n✅ Created merged validation set")
    print("Files created:")
    print("  - val_synthesis_merged.jsonl")
    print("  - embeddings/val_embeddings_merged.safetensors")

if __name__ == "__main__":
    merge_val_test()