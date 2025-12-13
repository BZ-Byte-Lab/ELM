"""Configuration for ELM adapter training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for the ELM adapter training pipeline."""

    # Paths
    base_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(init=False)
    embeddings_dir: Path = field(init=False)
    synthesis_dir: Path = field(init=False)
    checkpoints_dir: Path = field(init=False)

    # Model Configuration
    llm_model_name: str = "Qwen/Qwen3-4B-Instruct"
    embedding_dim: int = 2560  # Qwen3-Embedding-4B output dimension
    hidden_dim: int = 4096     # Adapter intermediate size

    # Adapter Configuration (EnhancedAdapter only)
    use_residual: bool = True

    # Training Hyperparameters
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Batch Configuration (optimized for 40GB VRAM)
    batch_size: int = 16
    gradient_accumulation_steps: int = 2  # effective batch = 32

    # Training Schedule
    num_epochs: int = 3
    max_steps: Optional[int] = None
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 50

    # Memory Optimization (40GB VRAM)
    use_bf16: bool = True
    use_gradient_checkpointing: bool = True

    # Data Configuration
    max_seq_length: int = 2048
    num_workers: int = 4

    # Special Token
    emb_token: str = "<EMB>"

    # Random Seed
    random_seed: int = 42

    # Weights & Biases
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None

    def __post_init__(self):
        """Initialize derived paths."""
        self.data_dir = self.base_dir / "data"
        self.embeddings_dir = self.data_dir / "embeddings"
        self.synthesis_dir = self.data_dir / "synthesis"
        self.checkpoints_dir = self.data_dir / "checkpoints"

    def create_directories(self):
        """Create necessary directories if they don't exist."""
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def get_synthesis_path(self, split: str) -> Path:
        """Get path to synthesis JSONL file for a given split.

        Args:
            split: One of 'train', 'val', or 'test'

        Returns:
            Path to the JSONL file
        """
        return self.synthesis_dir / f"{split}_synthesis.jsonl"

    def get_embeddings_path(self, split: str) -> Path:
        """Get path to embeddings file for a given split.

        Args:
            split: One of 'train', 'val', or 'test'

        Returns:
            Path to the safetensors file
        """
        return self.embeddings_dir / f"{split}_embeddings.safetensors"

    def get_checkpoint_path(self, step: int) -> Path:
        """Get path for adapter checkpoint at a given step."""
        return self.checkpoints_dir / f"adapter_step_{step}.safetensors"

    def get_training_state_path(self, step: int) -> Path:
        """Get path for full training state at a given step."""
        return self.checkpoints_dir / f"checkpoint_step_{step}.pt"

    def __repr__(self):
        """Custom representation."""
        return (
            f"TrainingConfig(\n"
            f"  llm={self.llm_model_name},\n"
            f"  adapter_dim={self.embedding_dim}->{self.hidden_dim}->{self.embedding_dim},\n"
            f"  batch_size={self.batch_size} (eff={self.batch_size * self.gradient_accumulation_steps}),\n"
            f"  lr={self.learning_rate},\n"
            f"  epochs={self.num_epochs}\n"
            f")"
        )
