"""
Configuration module for ELM data pipeline.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class Config:
    """Configuration for the ELM data preparation pipeline."""

    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)
    embeddings_dir: Path = field(init=False)

    # Dataset parameters
    hf_dataset_name: str = "Salesforce/wikitext"
    hf_dataset_config: str = "wikitext-2-v1"

    # Text filtering parameters
    min_tokens: int = 100
    max_tokens: int = 2000

    # Dataset split ratios
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42

    # Embedding model parameters
    model_name: str = "Qwen/Qwen3-Embedding-4B"
    embedding_dim: int = 2560
    max_length: int = 8192
    batch_size: int = 8  # Optimized for 16GB VRAM

    # Model optimization parameters
    use_flash_attention: bool = True
    use_fp16: bool = True

    def __post_init__(self):
        """Initialize derived paths."""
        self.data_dir = self.base_dir / "data"
        self.processed_dir = self.data_dir / "wikitext2_processed"
        self.embeddings_dir = self.data_dir / "embeddings"

        # Validate ratios
        if not abs(self.train_ratio + self.val_ratio + self.test_ratio - 1.0) < 1e-6:
            raise ValueError("Train, val, and test ratios must sum to 1.0")

    def create_directories(self):
        """Create necessary directories if they don't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

    def get_processed_path(self, split: str) -> Path:
        """Get path to processed data file for a given split.

        Args:
            split: One of 'train', 'val', or 'test'

        Returns:
            Path to the parquet file
        """
        return self.processed_dir / f"{split}.parquet"

    def get_embeddings_path(self, split: str) -> Path:
        """Get path to embeddings file for a given split.

        Args:
            split: One of 'train', 'val', or 'test'

        Returns:
            Path to the safetensors file
        """
        return self.embeddings_dir / f"{split}_embeddings.safetensors"

    def __repr__(self):
        """Custom representation."""
        return (
            f"Config(\n"
            f"  data_dir={self.data_dir},\n"
            f"  model={self.model_name},\n"
            f"  tokens={self.min_tokens}-{self.max_tokens},\n"
            f"  split={self.train_ratio}/{self.val_ratio}/{self.test_ratio},\n"
            f"  batch_size={self.batch_size}\n"
            f")"
        )
