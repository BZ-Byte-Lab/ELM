#!/usr/bin/env python3
"""Create proper non-overlapping data splits with actual embeddings.

This script identifies all unique embedding indices, creates non-overlapping splits,
and saves the actual embeddings for each split with contiguous indices.
"""

import json
import random
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import argparse
import numpy as np
from safetensors.numpy import save_file, load_file

def load_jsonl_samples(path: Path) -> List[Dict]:
    """Load samples from JSONL file."""
    samples = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))
    return samples

def analyze_task_distribution(samples: List[Dict]) -> Dict[str, int]:
    """Analyze task type distribution."""
    task_counts = Counter(sample["task_type"] for sample in samples)
    return dict(task_counts)

def create_clean_splits_with_embeddings(
    train_samples: List[Dict],
    val_samples: List[Dict],
    test_samples: List[Dict],
    original_embeddings_path: Path,
    output_dir: Path,
    seed: int = 42
):
    """Create clean splits with unique embedding indices and remapped embeddings."""

    # Set random seed for reproducibility
    random.seed(seed)

    # Get all unique embedding indices
    all_samples = train_samples + val_samples + test_samples
    unique_indices = sorted(list(set(sample["embedding_index"] for sample in all_samples)))

    print(f"Found {len(unique_indices)} unique embedding indices")
    print(f"Original embeddings shape: ?")  # We'll load this later

    # Load original embeddings
    print(f"Loading original embeddings from {original_embeddings_path}")
    tensors = load_file(str(original_embeddings_path))
    original_embeddings = tensors["embeddings"]
    print(f"Original embeddings shape: {original_embeddings.shape}")

    # Group all samples by embedding index
    all_samples_by_index = defaultdict(list)
    for sample in all_samples:
        idx = sample["embedding_index"]
        all_samples_by_index[idx].append(sample)

    # Create list of unique indices for splitting
    unique_indices_list = list(all_samples_by_index.keys())
    random.shuffle(unique_indices_list)

    # Calculate split sizes (70% train, 15% val, 15% test)
    n_total = len(unique_indices_list)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    train_original_indices = set(unique_indices_list[:n_train])
    val_original_indices = set(unique_indices_list[n_train:n_train + n_val])
    test_original_indices = set(unique_indices_list[n_train + n_val:])

    # Create new splits and collect embeddings
    new_train = []
    new_val = []
    new_test = []

    train_embeddings = []
    val_embeddings = []
    test_embeddings = []

    # Process train embeddings
    for new_idx, original_idx in enumerate(sorted(train_original_indices)):
        samples_for_idx = all_samples_by_index[original_idx]
        for sample in samples_for_idx:
            sample = sample.copy()
            sample["embedding_index"] = new_idx  # Remap to contiguous index
            new_train.append(sample)
        train_embeddings.append(original_embeddings[original_idx])

    # Process val embeddings
    for new_idx, original_idx in enumerate(sorted(val_original_indices)):
        samples_for_idx = all_samples_by_index[original_idx]
        for sample in samples_for_idx:
            sample = sample.copy()
            sample["embedding_index"] = new_idx  # Remap to contiguous index
            new_val.append(sample)
        val_embeddings.append(original_embeddings[original_idx])

    # Process test embeddings
    for new_idx, original_idx in enumerate(sorted(test_original_indices)):
        samples_for_idx = all_samples_by_index[original_idx]
        for sample in samples_for_idx:
            sample = sample.copy()
            sample["embedding_index"] = new_idx  # Remap to contiguous index
            new_test.append(sample)
        test_embeddings.append(original_embeddings[original_idx])

    # Convert to numpy arrays
    train_embeddings = np.array(train_embeddings)
    val_embeddings = np.array(val_embeddings)
    test_embeddings = np.array(test_embeddings)

    # Shuffle samples within each split
    random.shuffle(new_train)
    random.shuffle(new_val)
    random.shuffle(new_test)

    print(f"\nNew split statistics:")
    print(f"  Train: {len(new_train)} samples, {len(train_embeddings)} unique embeddings")
    print(f"  Val: {len(new_val)} samples, {len(val_embeddings)} unique embeddings")
    print(f"  Test: {len(new_test)} samples, {len(test_embeddings)} unique embeddings")

    # Analyze task distribution
    print("\nTask distribution in new splits:")
    print("  Train:", analyze_task_distribution(new_train))
    print("  Val:", analyze_task_distribution(new_val))
    print("  Test:", analyze_task_distribution(new_test))

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir = output_dir / "embeddings"
    embeddings_dir.mkdir(exist_ok=True)

    # Save new splits
    def save_split(samples, path):
        with open(path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')

    save_split(new_train, output_dir / "train_synthesis_clean.jsonl")
    save_split(new_val, output_dir / "val_synthesis_clean.jsonl")
    save_split(new_test, output_dir / "test_synthesis_clean.jsonl")

    # Save embeddings
    save_file({"embeddings": train_embeddings}, str(embeddings_dir / "train_embeddings.safetensors"))
    save_file({"embeddings": val_embeddings}, str(embeddings_dir / "val_embeddings.safetensors"))
    save_file({"embeddings": test_embeddings}, str(embeddings_dir / "test_embeddings.safetensors"))

    print(f"\nSaved embeddings:")
    print(f"  Train: {train_embeddings.shape}")
    print(f"  Val: {val_embeddings.shape}")
    print(f"  Test: {test_embeddings.shape}")

    # Save split metadata
    metadata = {
        "train_original_indices": sorted(list(train_original_indices)),
        "val_original_indices": sorted(list(val_original_indices)),
        "test_original_indices": sorted(list(test_original_indices)),
        "train_samples": len(new_train),
        "val_samples": len(new_val),
        "test_samples": len(new_test),
        "train_embeddings": len(train_embeddings),
        "val_embeddings": len(val_embeddings),
        "test_embeddings": len(test_embeddings),
    }

    with open(output_dir / "split_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nClean splits with embeddings saved to {output_dir}")
    print("Files created:")
    print("  - train_synthesis_clean.jsonl")
    print("  - val_synthesis_clean.jsonl")
    print("  - test_synthesis_clean.jsonl")
    print("  - embeddings/train_embeddings.safetensors")
    print("  - embeddings/val_embeddings.safetensors")
    print("  - embeddings/test_embeddings.safetensors")
    print("  - split_metadata.json")

def main():
    parser = argparse.ArgumentParser(description="Create clean non-overlapping data splits with embeddings")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/summary_filtered"),
        help="Directory containing current splits"
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        default=Path("data/embeddings/filtered_embeddings.safetensors"),
        help="Path to full embeddings file"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/summary_clean_v2"),
        help="Directory to save clean splits"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    # Load current splits
    print("Loading current splits...")
    train_samples = load_jsonl_samples(args.input_dir / "train_synthesis.jsonl")
    val_samples = load_jsonl_samples(args.input_dir / "val_synthesis.jsonl")
    test_samples = load_jsonl_samples(args.input_dir / "test_synthesis.jsonl")

    print(f"Loaded {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test samples")

    # Create clean splits with embeddings
    create_clean_splits_with_embeddings(
        train_samples,
        val_samples,
        test_samples,
        args.embeddings_path,
        args.output_dir,
        args.seed
    )

if __name__ == "__main__":
    main()