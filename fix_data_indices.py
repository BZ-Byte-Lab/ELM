#!/usr/bin/env python3
"""Fix the data splits to have contiguous embedding indices."""

import json
from pathlib import Path
from collections import defaultdict

def load_jsonl_samples(path: Path) -> list:
    """Load samples from JSONL file."""
    samples = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))
    return samples

def fix_split_indices(samples: list, start_idx: int = 0) -> tuple:
    """Remap embedding indices to be contiguous starting from start_idx.

    Returns:
        tuple: (fixed_samples, mapping_dict)
    """
    # Group samples by original embedding index
    samples_by_idx = defaultdict(list)
    original_indices = set()

    for sample in samples:
        original_idx = sample["embedding_index"]
        samples_by_idx[original_idx].append(sample)
        original_indices.add(original_idx)

    # Create mapping from original to new indices
    mapping = {}
    for new_idx, original_idx in enumerate(sorted(original_indices)):
        mapping[original_idx] = start_idx + new_idx

    # Create fixed samples with new indices
    fixed_samples = []
    for sample in samples:
        new_sample = sample.copy()
        new_sample["embedding_index"] = mapping[sample["embedding_index"]]
        fixed_samples.append(new_sample)

    return fixed_samples, mapping

def main():
    # Load original splits
    data_dir = Path("data/summary_filtered")
    output_dir = Path("data/summary_clean_fixed")

    print("Loading original splits...")
    train_samples = load_jsonl_samples(data_dir / "train_synthesis.jsonl")
    val_samples = load_jsonl_samples(data_dir / "val_synthesis.jsonl")
    test_samples = load_jsonl_samples(data_dir / "test_synthesis.jsonl")

    print(f"Loaded {len(train_samples)} train, {len(val_samples)} val, {len(test_samples)} test samples")

    # Get unique indices for each split
    train_indices = set(s["embedding_index"] for s in train_samples)
    val_indices = set(s["embedding_index"] for s in val_samples)
    test_indices = set(s["embedding_index"] for s in test_samples)

    print(f"\nUnique indices before fix:")
    print(f"  Train: {len(train_indices)} indices")
    print(f"  Val: {len(val_indices)} indices")
    print(f"  Test: {len(test_indices)} indices")
    print(f"  Overlaps: Train/Val={len(train_indices & val_indices)}, Train/Test={len(train_indices & test_indices)}, Val/Test={len(val_indices & test_indices)}")

    # Create non-overlapping sets first
    all_unique_indices = list(train_indices | val_indices | test_indices)
    all_unique_indices.sort()

    # Randomly shuffle for random assignment
    import random
    random.seed(42)
    random.shuffle(all_unique_indices)

    n_total = len(all_unique_indices)
    n_train = int(0.7 * n_total)
    n_val = int(0.15 * n_total)
    n_test = n_total - n_train - n_val

    train_set = set(all_unique_indices[:n_train])
    val_set = set(all_unique_indices[n_train:n_train + n_val])
    test_set = set(all_unique_indices[n_train + n_val:])

    # Filter samples to only include those in their assigned sets
    new_train = [s for s in train_samples if s["embedding_index"] in train_set]
    new_val = [s for s in val_samples if s["embedding_index"] in val_set]
    new_test = [s for s in test_samples if s["embedding_index"] in test_set]

    # Now remap indices to be contiguous within each split
    fixed_train, train_mapping = fix_split_indices(new_train, 0)
    fixed_val, val_mapping = fix_split_indices(new_val, 0)
    fixed_test, test_mapping = fix_split_indices(new_test, 0)

    print(f"\nAfter fix:")
    print(f"  Train: {len(fixed_train)} samples, indices 0 to {max(train_mapping.values()) if train_mapping else 0}")
    print(f"  Val: {len(fixed_val)} samples, indices 0 to {max(val_mapping.values()) if val_mapping else 0}")
    print(f"  Test: {len(fixed_test)} samples, indices 0 to {max(test_mapping.values()) if test_mapping else 0}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save fixed splits
    def save_split(samples, path):
        with open(path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')

    save_split(fixed_train, output_dir / "train_synthesis_clean.jsonl")
    save_split(fixed_val, output_dir / "val_synthesis_clean.jsonl")
    save_split(fixed_test, output_dir / "test_synthesis_clean.jsonl")

    # Save mapping info
    mapping_info = {
        "train_mapping": train_mapping,
        "val_mapping": val_mapping,
        "test_mapping": test_mapping,
        "train_indices_range": [0, len(train_mapping)],
        "val_indices_range": [0, len(val_mapping)],
        "test_indices_range": [0, len(test_mapping)]
    }

    with open(output_dir / "mapping_info.json", 'w') as f:
        json.dump(mapping_info, f, indent=2)

    print(f"\nFixed splits saved to {output_dir}")
    print("\nNote: Make sure to use embedding files that match the split sizes:")
    print(f"  Train needs embeddings with shape: ({len(train_mapping)}, 2560)")
    print(f"  Val needs embeddings with shape: ({len(val_mapping)}, 2560)")
    print(f"  Test needs embeddings with shape: ({len(test_mapping)}, 2560)")

if __name__ == "__main__":
    main()