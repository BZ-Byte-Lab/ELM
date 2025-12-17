#!/usr/bin/env python3
"""Quick test script to verify ELM training setup."""

import sys
from pathlib import Path

# Add paths for modules
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "summary_training_pipeline"))

from summary_training_pipeline.dataset import ELMTrainingDataset, TrainingCollator
from summary_training_pipeline.config import SummaryTrainingConfig

def test_dataset_loading():
    """Test loading clean dataset."""
    print("Testing dataset loading...")

    # Initialize dataset
    dataset = ELMTrainingDataset(
        synthesis_path=Path("../data/summary_clean/train_synthesis_clean.jsonl"),
        embeddings_path=Path("../data/embeddings/filtered_embeddings.safetensors"),
        max_samples=10  # Just test with 10 samples
    )

    print(f"✓ Dataset loaded with {len(dataset)} samples")

    # Test collator
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")

    # Add special tokens if needed
    if "<EMB>" not in tokenizer.get_vocab():
        tokenizer.add_special_tokens({"additional_special_tokens": ["<EMB>"]})

    collator = TrainingCollator(tokenizer)

    # Test batching
    batch = [dataset[i] for i in range(min(3, len(dataset)))]
    collated = collator(batch)

    print(f"✓ Collated batch shape:")
    print(f"  input_ids: {collated['input_ids'].shape}")
    print(f"  embeddings: {collated['embeddings'].shape}")
    print(f"  embedding_positions: {collated['embedding_positions'].shape}")
    print(f"  labels: {collated['labels'].shape}")

def test_config():
    """Test training configuration."""
    print("\nTesting configuration...")

    config = SummaryTrainingConfig(
        model_name="microsoft/DialoGPT-medium",
        data_dir=Path("../data/summary_clean"),
        max_steps=10,
        warmup_steps=2,
        batch_size=2,
        learning_rate=1e-4,
        use_contrastive=False,
        use_drift_loss=False
    )

    print(f"✓ Configuration created")
    print(f"  Model: {config.model_name}")
    print(f"  Data dir: {config.data_dir}")
    print(f"  Use contrastive: {config.use_contrastive}")
    print(f"  Use drift loss: {config.use_drift_loss}")

if __name__ == "__main__":
    print("Testing ELM Training Pipeline Setup")
    print("=" * 50)

    test_dataset_loading()
    test_config()

    print("\n✅ All tests passed! Ready to train.")