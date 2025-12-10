"""
Configuration module for ELM data synthesis pipeline.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum


class TaskCategory(Enum):
    """Task category enumeration."""
    FACTUAL = "factual"
    DESCRIPTIVE = "descriptive"
    CREATIVE = "creative"
    PAIR_BASED = "pair_based"


@dataclass
class TaskConfig:
    """Configuration for a single task type."""
    name: str
    category: TaskCategory
    temperature: float
    top_p: float
    min_tokens: int
    prompt_template: str
    variations: int = 2
    requires_pair: bool = False
    knn_k: int = 0
    alpha_range: Optional[Tuple[float, float]] = None


@dataclass
class SynthesisConfig:
    """Configuration for the ELM data synthesis pipeline."""

    # Paths - derived from base_dir similar to Data_Preparation
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(init=False)
    embeddings_dir: Path = field(init=False)
    synthesis_dir: Path = field(init=False)
    checkpoints_dir: Path = field(init=False)

    # Data_Preparation integration
    processed_dir: Path = field(init=False)

    # API Configuration
    api_base_url: str = "https://openrouter.ai/api/v1"
    model_name: str = "qwen/qwen3-30b-a3b-instruct-2507"

    # Rate Limiting
    requests_per_minute: int = 60
    requests_per_second: float = 10
    max_concurrent_requests: int = 10  # NEW: Concurrent request limit for async
    max_retries: int = 3
    retry_delay: float = 1.0

    # k-NN Configuration
    knn_k: int = 10
    knn_metric: str = "cosine"
    use_gpu_index: bool = False

    # Generation Parameters
    checkpoint_interval: int = 20
    variations_per_task: int = 2
    min_samples_per_embedding: int = 15
    batch_size: int = 50  # NEW: Batch size for async operations
    use_async: bool = True  # NEW: Toggle async mode

    # Concurrent Batch Processing
    max_concurrent_batches: int = 3  # Number of batches to process in parallel

    # Performance Profiling
    enable_profiling: bool = True  # Track timing metrics for performance analysis

    # Quality Filtering
    max_rejection_rate: float = 0.20
    repetition_threshold: float = 0.5

    # Embedding parameters (from Data_Preparation)
    embedding_dim: int = 2560

    # Random seed for reproducibility
    random_seed: int = 42

    def __post_init__(self):
        """Initialize derived paths."""
        self.data_dir = self.base_dir / "data"
        self.embeddings_dir = self.data_dir / "embeddings"
        self.processed_dir = self.data_dir / "wikitext2_processed"
        self.synthesis_dir = self.data_dir / "synthesis"
        self.checkpoints_dir = self.synthesis_dir / "checkpoints"

    def create_directories(self):
        """Create necessary directories if they don't exist."""
        self.synthesis_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def get_api_key(self) -> str:
        """Get OpenRouter API key from environment.

        Returns:
            API key string

        Raises:
            ValueError: If OPENROUTER_API_KEY not set
        """
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set")
        return api_key

    def get_synthesis_path(self, split: str) -> Path:
        """Get path to synthesis output file for a given split.

        Args:
            split: One of 'train', 'val', or 'test'

        Returns:
            Path to the JSONL output file
        """
        return self.synthesis_dir / f"{split}_synthesis.jsonl"

    def get_checkpoint_path(self, split: str, task_name: str) -> Path:
        """Get path to checkpoint file.

        Args:
            split: Data split name
            task_name: Task type name

        Returns:
            Path to checkpoint JSON file
        """
        return self.checkpoints_dir / f"{split}_{task_name}_checkpoint.json"

    def __repr__(self):
        """Custom representation."""
        return (
            f"SynthesisConfig(\n"
            f"  synthesis_dir={self.synthesis_dir},\n"
            f"  model={self.model_name},\n"
            f"  checkpoint_interval={self.checkpoint_interval},\n"
            f"  knn_k={self.knn_k}\n"
            f")"
        )
