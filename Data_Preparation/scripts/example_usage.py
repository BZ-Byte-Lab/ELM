#!/usr/bin/env python3
"""
Example script demonstrating how to use the ELMDataset class.

This script shows various ways to load and use the prepared data.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline import ELMDataset, Config


def main():
    print("=" * 80)
    print("ELM Dataset Usage Examples")
    print("=" * 80)

    # Initialize configuration
    config = Config()

    # Check if data exists
    if not config.get_processed_path("train").exists():
        print("\nError: Processed data not found!")
        print("Please run the pipeline first:")
        print("  python scripts/run_pipeline.py")
        sys.exit(1)

    # Example 1: Load dataset with embeddings
    print("\n" + "=" * 80)
    print("Example 1: Loading dataset with embeddings")
    print("=" * 80)

    train_dataset = ELMDataset(
        data_dir=config.data_dir,
        split="train",
        load_embeddings=True,
        config=config
    )

    print(f"\nDataset size: {len(train_dataset)} samples")

    # Get a single sample
    sample = train_dataset[0]
    print(f"\nSample 0:")
    print(f"  Text: {sample['text'][:100]}...")
    print(f"  Embedding shape: {sample['embedding'].shape}")
    print(f"  Token count: {sample['metadata']['token_count']}")
    print(f"  Char count: {sample['metadata']['char_count']}")

    # Example 2: Dataset statistics
    print("\n" + "=" * 80)
    print("Example 2: Dataset statistics")
    print("=" * 80)

    stats = train_dataset.get_statistics()
    print(f"\nTrain dataset statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")

    # Example 3: Batch loading
    print("\n" + "=" * 80)
    print("Example 3: Loading a batch")
    print("=" * 80)

    batch_indices = [0, 1, 2, 3, 4]
    batch = train_dataset.get_batch(batch_indices)

    print(f"\nBatch of {len(batch_indices)} samples:")
    print(f"  Texts: {len(batch['text'])} items")
    print(f"  Embeddings shape: {batch['embedding'].shape}")
    print(f"  First text (truncated): {batch['text'][0][:80]}...")

    # Example 4: Using PyTorch DataLoader
    print("\n" + "=" * 80)
    print("Example 4: Using PyTorch DataLoader")
    print("=" * 80)

    dataloader = train_dataset.get_dataloader(
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    print(f"\nDataLoader created with {len(dataloader)} batches")

    # Get first batch
    first_batch = next(iter(dataloader))
    print(f"\nFirst batch:")
    print(f"  Number of texts: {len(first_batch['text'])}")
    print(f"  Embeddings shape: {first_batch['embedding'].shape}")
    print(f"  Sample text: {first_batch['text'][0][:80]}...")

    # Example 5: Embedding interpolation
    print("\n" + "=" * 80)
    print("Example 5: Embedding interpolation")
    print("=" * 80)

    idx1, idx2 = 0, 100
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    print(f"\nInterpolating between sample {idx1} and {idx2}:")
    print(f"  Text 1: {train_dataset[idx1]['text'][:60]}...")
    print(f"  Text 2: {train_dataset[idx2]['text'][:60]}...")

    print("\nInterpolated embeddings:")
    for alpha in alphas:
        interpolated = train_dataset.interpolate_embeddings(idx1, idx2, alpha)
        print(f"  alpha={alpha:.2f}: shape={interpolated.shape}, "
              f"norm={np.linalg.norm(interpolated):.4f}")

    # Example 6: Loading without embeddings
    print("\n" + "=" * 80)
    print("Example 6: Loading text only (no embeddings)")
    print("=" * 80)

    text_only_dataset = ELMDataset(
        data_dir=config.data_dir,
        split="val",
        load_embeddings=False,
        config=config
    )

    print(f"\nValidation dataset (text only): {len(text_only_dataset)} samples")

    sample = text_only_dataset[0]
    print(f"\nSample 0:")
    print(f"  Text: {sample['text'][:100]}...")
    print(f"  Has embedding: {'embedding' in sample}")
    print(f"  Metadata: {sample['metadata']}")

    # Example 7: Computing similarity between samples
    print("\n" + "=" * 80)
    print("Example 7: Computing embedding similarity")
    print("=" * 80)

    # Get embeddings for a few samples
    emb1 = train_dataset[0]['embedding']
    emb2 = train_dataset[1]['embedding']
    emb3 = train_dataset[100]['embedding']

    # Compute cosine similarity (embeddings are already normalized)
    sim_01 = np.dot(emb1, emb2)
    sim_02 = np.dot(emb1, emb3)
    sim_12 = np.dot(emb2, emb3)

    print(f"\nCosine similarities:")
    print(f"  Sample 0 vs 1: {sim_01:.4f}")
    print(f"  Sample 0 vs 100: {sim_02:.4f}")
    print(f"  Sample 1 vs 100: {sim_12:.4f}")

    print(f"\nSample 0 text: {train_dataset[0]['text'][:80]}...")
    print(f"Sample 1 text: {train_dataset[1]['text'][:80]}...")
    print(f"Sample 100 text: {train_dataset[100]['text'][:80]}...")

    # Example 8: All splits
    print("\n" + "=" * 80)
    print("Example 8: Loading all splits")
    print("=" * 80)

    splits = {}
    for split_name in ["train", "val", "test"]:
        splits[split_name] = ELMDataset(
            data_dir=config.data_dir,
            split=split_name,
            load_embeddings=True,
            config=config
        )

    print("\nDataset sizes:")
    for split_name, dataset in splits.items():
        print(f"  {split_name}: {len(dataset)} samples")

    total_samples = sum(len(ds) for ds in splits.values())
    print(f"  Total: {total_samples} samples")

    print("\n" + "=" * 80)
    print("Examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()
