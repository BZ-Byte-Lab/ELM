#!/usr/bin/env python3
"""Verify that clean data files exist and are accessible."""

import sys
from pathlib import Path

# Add paths for modules
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "summary_training_pipeline"))

def verify_files():
    """Verify that all clean data files exist."""
    base_dir = Path("../data/summary_clean")

    files_to_check = [
        "train_synthesis_clean.jsonl",
        "val_synthesis_clean.jsonl",
        "test_synthesis_clean.jsonl",
        "embeddings/train_embeddings.safetensors",
        "embeddings/val_embeddings.safetensors",
        "embeddings/test_embeddings.safetensors"
    ]

    print("Checking clean data files...")
    all_exist = True

    for file_path in files_to_check:
        full_path = base_dir / file_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print(f"✓ {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"✗ {file_path} - NOT FOUND!")
            all_exist = False

    if all_exist:
        print("\n✅ All files found! Ready to train.")
    else:
        print("\n❌ Some files are missing.")

    return all_exist

if __name__ == "__main__":
    verify_files()