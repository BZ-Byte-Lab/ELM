#!/usr/bin/env python3
"""Create proper non-overlapping data splits for ELM training.

This script identifies all unique embedding indices and creates
non-overlapping train/val/test splits with proper task distribution.
"""

import json
import random
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import argparse

def load_jsonl_samples(path: Path) -> List[Dict]:
    """Load samples from JSONL file."""
    samples = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))
    return samples

def get_all_embedding_indices(samples: List[Dict]) -> Set[int]:
    """Get all unique embedding indices from samples."""
    return set(sample["embedding_index"] for sample in samples)

def analyze_task_distribution(samples: List[Dict]) -> Dict[str, int]:
    """Analyze task type distribution."""
    task_counts = Counter(sample["task_type"] for sample in samples)
    return dict(task_counts)

def create_clean_splits(
    train_samples: List[Dict],
    val_samples: List[Dict],
    test_samples: List[Dict],
    output_dir: Path,
    seed: int = 42
):
    """Create clean splits with unique embedding indices."""

    # Set random seed for reproducibility
    random.seed(seed)

    # Get all unique embedding indices
    train_indices = get_all_embedding_indices(train_samples)
    val_indices = get_all_embedding_indices(val_samples)
    test_indices = get_all_embedding_indices(test_samples)

    print(f"Initial overlap analysis:")
    print(f"  Train unique indices: {len(train_indices)}")
    print(f"  Val unique indices: {len(val_indices)}")
    print(f"  Test unique indices: {len(test_indices)}")
    print(f"  Train/Val overlap: {len(train_indices & val_indices)}")
    print(f"  Train/Test overlap: {len(train_indices & test_indices)}")
    print(f"  Val/Test overlap: {len(val_indices & test_indices)}")

    # Group all samples by embedding index
    all_samples_by_index = defaultdict(list)

    # Process all samples together
    all_samples = train_samples + val_samples + test_samples
    for sample in all_samples:
        idx = sample["embedding_index"]
        all_samples_by_index[idx].append(sample)

    print(f"\nTotal unique embedding indices: {len(all_samples_by_index)}")

    # Create list of unique indices for splitting
    unique_indices = list(all_samples_by_index.keys())
    random.shuffle(unique_indices)

    # Calculate split sizes (70% train, 15% val, 15% test)
    n_total = len(unique_indices)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val  # Ensure all indices are used

    train_indices_set = set(unique_indices[:n_train])
    val_indices_set = set(unique_indices[n_train:n_train + n_val])
    test_indices_set = set(unique_indices[n_train + n_val:])

    # Create new splits
    new_train = []
    new_val = []
    new_test = []

    for idx, samples_for_idx in all_samples_by_index.items():
        if idx in train_indices_set:
            new_train.extend(samples_for_idx)
        elif idx in val_indices_set:
            new_val.extend(samples_for_idx)
        elif idx in test_indices_set:
            new_test.extend(samples_for_idx)

    # Shuffle samples within each split
    random.shuffle(new_train)
    random.shuffle(new_val)
    random.shuffle(new_test)

    # Verify no overlap
    new_train_indices = get_all_embedding_indices(new_train)
    new_val_indices = get_all_embedding_indices(new_val)
    new_test_indices = get_all_embedding_indices(new_test)

    assert len(new_train_indices & new_val_indices) == 0, "Train/Val overlap!"
    assert len(new_train_indices & new_test_indices) == 0, "Train/Test overlap!"
    assert len(new_val_indices & new_test_indices) == 0, "Val/Test overlap!"

    print(f"\nNew split statistics:")
    print(f"  Train: {len(new_train)} samples, {len(new_train_indices)} unique indices")
    print(f"  Val: {len(new_val)} samples, {len(new_val_indices)} unique indices")
    print(f"  Test: {len(new_test)} samples, {len(new_test_indices)} unique indices")

    # Analyze task distribution in new splits
    print("\nTask distribution in new splits:")
    print("  Train:", analyze_task_distribution(new_train))
    print("  Val:", analyze_task_distribution(new_val))
    print("  Test:", analyze_task_distribution(new_test))

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save new splits
    def save_split(samples, path):
        with open(path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')

    save_split(new_train, output_dir / "train_synthesis_clean.jsonl")
    save_split(new_val, output_dir / "val_synthesis_clean.jsonl")
    save_split(new_test, output_dir / "test_synthesis_clean.jsonl")

    # Save split metadata
    metadata = {
        "train_indices": sorted(list(new_train_indices)),
        "val_indices": sorted(list(new_val_indices)),
        "test_indices": sorted(list(new_test_indices)),
        "train_samples": len(new_train),
        "val_samples": len(new_val),
        "test_samples": len(new_test),
        "train_unique_indices": len(new_train_indices),
        "val_unique_indices": len(new_val_indices),
        "test_unique_indices": len(new_test_indices),
    }

    with open(output_dir / "split_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nClean splits saved to {output_dir}")
    print("Files created:")
    print("  - train_synthesis_clean.jsonl")
    print("  - val_synthesis_clean.jsonl")
    print("  - test_synthesis_clean.jsonl")
    print("  - split_metadata.json")

def main():
    parser = argparse.ArgumentParser(description="Create clean non-overlapping data splits")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/summary_filtered"),
        help="Directory containing current splits"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/summary_clean"),
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

    # Create clean splits
    create_clean_splits(train_samples, val_samples, test_samples, args.output_dir, args.seed)

if __name__ == "__main__":
    main()