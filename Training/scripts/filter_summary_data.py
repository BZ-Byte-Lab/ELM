#!/usr/bin/env python3
"""Filter summary-only data from multi-task synthesis files.

This script processes existing multi-task JSONL files and filters for task_type == "summary",
creating new embedding index mappings and filtered SafeTensors embeddings files for summary-only training.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from safetensors import safe_open
from tqdm import tqdm


def load_embeddings(embeddings_path: Path) -> torch.Tensor:
    """Load embeddings from SafeTensors file."""
    try:
        with safe_open(embeddings_path, framework="pt") as f:
            # Assume first key contains the embeddings
            key = next(iter(f.keys()))
            return f.get_tensor(key)
    except Exception as e:
        raise RuntimeError(f"Failed to load embeddings from {embeddings_path}: {e}")


def filter_summary_data(
    input_jsonl: Path,
    input_embeddings: Path,
    output_jsonl: Path,
    output_embeddings: Path,
    split_name: str
) -> Tuple[int, int, List[int]]:
    """Filter summary data and create new embedding index mapping.

    Returns:
        Tuple of (total_entries, summary_entries, original_indices_kept)
    """
    print(f"Processing {split_name} split...")

    # Load original embeddings
    print(f"Loading embeddings from {input_embeddings}")
    original_embeddings = load_embeddings(input_embeddings)
    print(f"Original embeddings shape: {original_embeddings.shape}")

    # Track summary entries and their embedding indices
    summary_entries = []
    summary_embedding_indices = []

    # First pass: identify summary entries
    print(f"Scanning {input_jsonl} for summary tasks...")
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc=f"Scanning {split_name}")):
            try:
                data = json.loads(line.strip())
                if data.get('task_type') == 'summary':
                    summary_entries.append(data)
                    # Use the embedding_index from the data, not the line number
                    embedding_idx = data.get('embedding_index')
                    if embedding_idx is not None:
                        summary_embedding_indices.append(embedding_idx)
                    else:
                        print(f"Warning: Line {i} has no embedding_index, skipping")
                        continue
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {i}: {e}")
                continue

    total_entries = i + 1
    summary_count = len(summary_entries)

    print(f"Found {summary_count:,} summary entries out of {total_entries:,} total entries "
          f"({summary_count/total_entries*100:.1f}%)")

    if summary_count == 0:
        print("Warning: No summary entries found!")
        return total_entries, 0, []

    # Filter embeddings based on summary embedding indices
    summary_embeddings = original_embeddings[summary_embedding_indices]
    print(f"Summary embeddings shape: {summary_embeddings.shape}")

    # Update embedding indices in the filtered data
    new_embedding_index = 0
    for entry in summary_entries:
        if entry.get('embedding_index') is not None:
            entry['embedding_index'] = new_embedding_index
            new_embedding_index += 1

    # Write filtered JSONL
    print(f"Writing filtered data to {output_jsonl}")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for entry in tqdm(summary_entries, desc="Writing filtered data"):
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # Write filtered embeddings
    print(f"Writing filtered embeddings to {output_embeddings}")
    output_embeddings.parent.mkdir(parents=True, exist_ok=True)

    # Convert to SafeTensors format
    filtered_embeddings_dict = {
        'embeddings': summary_embeddings
    }

    # Save as SafeTensors
    from safetensors.torch import save_file
    save_file(filtered_embeddings_dict, str(output_embeddings))

    return total_entries, summary_count, summary_embedding_indices


def create_train_val_splits(
    summary_data_path: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Create train/val/test splits from summary data.

    Note: This is only used if original data doesn't have proper splits.
    For this implementation, we'll preserve existing splits.
    """
    np.random.seed(random_seed)

    # Load all summary data
    all_summary_data = []
    for split_file in ['train_synthesis.jsonl', 'val_synthesis.jsonl', 'test_synthesis.jsonl']:
        split_path = summary_data_path / split_file
        if split_path.exists():
            with open(split_path, 'r', encoding='utf-8') as f:
                for line in f:
                    all_summary_data.append(json.loads(line.strip()))

    # Shuffle and split
    np.random.shuffle(all_summary_data)
    n = len(all_summary_data)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_data = all_summary_data[:train_end]
    val_data = all_summary_data[train_end:val_end]
    test_data = all_summary_data[val_end:]

    return train_data, val_data, test_data


def analyze_summary_statistics(
    train_path: Path,
    val_path: Path,
    test_path: Path
) -> Dict:
    """Analyze and print statistics about the filtered summary data."""
    stats = {}

    for split_name, path in [('train', train_path), ('val', val_path), ('test', test_path)]:
        if not path.exists():
            continue

        token_counts = []
        variation_counts = {}
        temperature_counts = {}
        top_p_counts = {}

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())

                # Token count statistics
                if 'token_count' in data:
                    token_counts.append(data['token_count'])

                # Variation statistics
                variation = data.get('variation', 0)
                variation_counts[variation] = variation_counts.get(variation, 0) + 1

                # Temperature statistics
                temp = data.get('temperature', 0.0)
                temp_key = f"{temp:.1f}"
                temperature_counts[temp_key] = temperature_counts.get(temp_key, 0) + 1

                # Top-p statistics
                top_p = data.get('top_p', 1.0)
                top_p_key = f"{top_p:.2f}"
                top_p_counts[top_p_key] = top_p_counts.get(top_p_key, 0) + 1

        stats[split_name] = {
            'total_entries': len(token_counts),
            'avg_tokens': np.mean(token_counts) if token_counts else 0,
            'median_tokens': np.median(token_counts) if token_counts else 0,
            'min_tokens': min(token_counts) if token_counts else 0,
            'max_tokens': max(token_counts) if token_counts else 0,
            'variations': variation_counts,
            'temperatures': temperature_counts,
            'top_p_values': top_p_counts
        }

    return stats


def main():
    parser = argparse.ArgumentParser(description="Filter summary data from multi-task files")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/home/benz/coding_project/elm/data/synthesis"),
        help="Directory containing multi-task synthesis JSONL files"
    )
    parser.add_argument(
        "--input-embeddings-dir",
        type=Path,
        default=Path("data/embeddings"),
        help="Directory containing multi-task embeddings SafeTensors files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/summary_filtered/synthesis"),
        help="Directory to write filtered summary synthesis files"
    )
    parser.add_argument(
        "--output-embeddings-dir",
        type=Path,
        default=Path("data/summary_filtered/embeddings"),
        help="Directory to write filtered summary embeddings files"
    )

    args = parser.parse_args()

    # Process each split
    splits = ['train', 'val', 'test']
    total_stats = {}

    for split in splits:
        input_jsonl = args.input_dir / f"{split}_synthesis.jsonl"
        input_embeddings = args.input_embeddings_dir / f"{split}_embeddings.safetensors"
        output_jsonl = args.output_dir / f"{split}_synthesis.jsonl"
        output_embeddings = args.output_embeddings_dir / f"{split}_embeddings.safetensors"

        if not input_jsonl.exists():
            print(f"Warning: {input_jsonl} not found, skipping {split} split")
            continue

        if not input_embeddings.exists():
            print(f"Warning: {input_embeddings} not found, skipping {split} split")
            continue

        total, summary_count, indices = filter_summary_data(
            input_jsonl=input_jsonl,
            input_embeddings=input_embeddings,
            output_jsonl=output_jsonl,
            output_embeddings=output_embeddings,
            split_name=split
        )

        total_stats[split] = {
            'total_entries': total,
            'summary_entries': summary_count,
            'percentage': (summary_count / total * 100) if total > 0 else 0
        }

    # Analyze and print statistics
    print("\n" + "="*60)
    print("SUMMARY DATA FILTERING STATISTICS")
    print("="*60)

    for split, stats in total_stats.items():
        print(f"\n{split.upper()} SPLIT:")
        print(f"  Original entries: {stats['total_entries']:,}")
        print(f"  Summary entries: {stats['summary_entries']:,}")
        print(f"  Percentage: {stats['percentage']:.1f}%")

    # Detailed statistics analysis
    print("\n" + "="*60)
    print("DETAILED SUMMARY DATA ANALYSIS")
    print("="*60)

    stats = analyze_summary_statistics(
        train_path=args.output_dir / "train_synthesis.jsonl",
        val_path=args.output_dir / "val_synthesis.jsonl",
        test_path=args.output_dir / "test_synthesis.jsonl"
    )

    for split_name, split_stats in stats.items():
        print(f"\n{split_name.upper()} SPLIT STATISTICS:")
        print(f"  Total entries: {split_stats['total_entries']:,}")
        print(f"  Token count - Avg: {split_stats['avg_tokens']:.1f}")
        print(f"  Token count - Median: {split_stats['median_tokens']:.1f}")
        print(f"  Token count - Range: {split_stats['min_tokens']} - {split_stats['max_tokens']}")

        if split_stats['variations']:
            print(f"  Variations: {dict(sorted(split_stats['variations'].items()))}")
        if split_stats['temperatures']:
            print(f"  Temperatures: {dict(sorted(split_stats['temperatures'].items()))}")
        if split_stats['top_p_values']:
            print(f"  Top-p values: {dict(sorted(split_stats['top_p_values'].items()))}")

    print("\n" + "="*60)
    print("FILTERING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Filtered data written to: {args.output_dir}")
    print(f"Filtered embeddings written to: {args.output_embeddings_dir}")


if __name__ == "__main__":
    main()