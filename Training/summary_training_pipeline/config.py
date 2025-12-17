"""Configuration for ELM summary-only adapter training."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SummaryTrainingConfig:
    """Configuration for the ELM summary-only adapter training pipeline."""

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
    residual_scale: float = 0.1   # Scale residual to prevent dominance
    dropout_rate: float = 0.1     # Dropout for regularization

    # Contrastive Loss Configuration
    use_contrastive_loss: bool = True
    contrastive_weight: float = 0.003      # REDUCED from 0.01 to prevent over-optimization
    contrastive_temperature: float = 0.2  # InfoNCE temperature

    # Text Drift Loss Configuration (for Bayesian Optimization)
    use_text_drift_loss: bool = True         # Always enabled for summary tasks
    text_drift_weight: float = 0.03         # Starting weight (0.01 to 0.1 in BO)
    text_drift_target_similarity: float = 0.75  # Target cosine similarity (0.7 to 0.9 in BO)

    # Training Hyperparameters
    learning_rate: float = 1e-4   # REDUCED from 2e-4 for more stable training
    warmup_steps: int = 200       # INCREASED from 100 for smoother warmup
    weight_decay: float = 0.02    # INCREASED from 0.01 for stronger regularization
    max_grad_norm: float = 1.0

    # Batch Configuration (optimized for 40GB VRAM)
    batch_size: int = 8                          # DECREASED from 16
    gradient_accumulation_steps: int = 4         # INCREASED from 2 (effective batch = 32)

    # Training Schedule (Summary-Only with 2-epoch constraint)
    summary_only: bool = True                    # Always true for summary pipeline
    summary_data_path: str = "data/summary_filtered"
    max_epochs: int = 2                          # Fixed 2-epoch constraint for Bayesian optimization
    max_steps: Optional[int] = None
    eval_steps: int = 250         # DECREASED from 500 for earlier collapse detection
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

    # BERTScore Evaluation Configuration
    bertscore_model: str = "microsoft/deberta-xlarge-mnli"  # High-quality model
    bertscore_batch_size: int = 16          # Batch size for BERTScore computation
    bertscore_rescale: bool = True           # Rescale scores to [0, 1]

    def __post_init__(self):
        """Initialize derived paths."""
        self.data_dir = self.base_dir / "data"
        self.embeddings_dir = Path(self.summary_data_path) / "embeddings"
        self.synthesis_dir = Path(self.summary_data_path)  # Directories are directly in summary_data_path
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
            f"SummaryTrainingConfig(\n"
            f"  llm={self.llm_model_name},\n"
            f"  adapter_dim={self.embedding_dim}->{self.hidden_dim}->{self.embedding_dim},\n"
            f"  batch_size={self.batch_size} (eff={self.batch_size * self.gradient_accumulation_steps}),\n"
            f"  lr={self.learning_rate},\n"
            f"  epochs={self.max_epochs},\n"
            f"  drift_loss={self.use_text_drift_loss},\n"
            f"  drift_weight={self.text_drift_weight},\n"
            f"  summary_only={self.summary_only}\n"
            f")"
        )
