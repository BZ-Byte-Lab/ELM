#!/usr/bin/env python3
"""Analyze the training issues and provide recommendations."""

import json
from pathlib import Path

def analyze_training_status():
    """Check training progress and identify issues."""

    print("=" * 60)
    print("ELM Training Analysis")
    print("=" * 60)

    # Check data distribution
    data_dir = Path("data/summary_clean_fixed")

    print("\n1. Data Distribution:")
    for split in ["train", "val", "test"]:
        jsonl_file = data_dir / f"{split}_synthesis_clean.jsonl"
        if jsonl_file.exists():
            with open(jsonl_file) as f:
                count = sum(1 for line in f)
            print(f"  {split}: {count:,} samples")

    # Check embeddings
    print("\n2. Embedding Sizes:")
    emb_dir = data_dir / "embeddings"
    for split in ["train", "val", "test"]:
        emb_file = emb_dir / f"{split}_embeddings.safetensors"
        if emb_file.exists():
            size = emb_file.stat().st_size / (1024 * 1024)  # MB
            print(f"  {split}: {size:.1f} MB")

    # Analyze BERTScore
    print("\n3. BERTScore Analysis:")
    print("  Current F1: 0.0831 (very low)")
    print("  Expected F1: 0.3-0.4 for initial training")
    print("  Target F1: 0.7+ for full training")

    print("\n4. Identified Issues:")
    print("  ❌ Validation set too small (94 samples)")
    print("  ❌ Model may be under-trained")
    print("  ❌ Generating repetitive tokens (⽗⽗⽗⽗)")
    print("  ❌ BERTScore precision is negative (-0.0654)")

    print("\n5. Recommendations:")
    print("  1. Increase training steps:")
    print("     - Current: Likely < 500 steps")
    print("     - Recommended: 2000-3000 steps")
    print("  ")
    print("  2. Learning rate schedule:")
    print("     - Start with 5e-5 (lower than current 2e-4)")
    print("     - Use longer warmup (500 steps)")
    print("  ")
    print("  3. Data improvements:")
    print("     - Use larger validation set from test split")
    print("     - Ensure normalized embeddings")
    print("  ")
    print("  4. Training monitoring:")
    print("     - Watch for loss plateaus")
    print("     - Check sample generations every 100 steps")

    print("\n6. Quick Fixes:")
    print("  • Add --max-steps 2000 to training command")
    print("  • Add --learning-rate 5e-5")
    print("  • Use --data-dir data/summary_clean_fixed")
    print("  • Disable conflicting losses (already done)")

    print("\n7. Expected Timeline:")
    print("  - 0-500 steps: BERTScore 0.05-0.2 (warmup phase)")
    print("  - 500-1500 steps: BERTScore 0.2-0.4 (learning phase)")
    print("  - 1500-3000 steps: BERTScore 0.4-0.7 (improving)")
    print("  - 3000+ steps: BERTScore 0.7+ (converging)")

    # Create a summary file
    summary = {
        "status": "Under-trained",
        "current_bertscore": 0.0831,
        "target_bertscore": 0.7,
        "issues": [
            "Small validation set",
            "Insufficient training steps",
            "Token generation issues"
        ],
        "recommendations": [
            "Increase max_steps to 2000-3000",
            "Lower learning rate to 5e-5",
            "Monitor generation quality"
        ]
    }

    with open("training_analysis.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n✅ Analysis saved to training_analysis.json")

if __name__ == "__main__":
    analyze_training_status()