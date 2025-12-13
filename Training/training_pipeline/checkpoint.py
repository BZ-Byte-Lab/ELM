"""Checkpoint management for ELM adapter training."""

import torch
from safetensors.torch import save_file, load_file
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class AdapterCheckpoint:
    """Manages adapter-only checkpoints for efficient storage.

    Saves only the adapter weights (~21M params) in SafeTensors format,
    not the full LLM (~4B params). Training state (optimizer, scheduler)
    is saved separately in PyTorch format.
    """

    def __init__(self, checkpoints_dir: Path):
        """Initialize checkpoint manager.

        Args:
            checkpoints_dir: Directory to save checkpoints
        """
        self.checkpoints_dir = Path(checkpoints_dir)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        adapter: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        global_step: int,
        epoch: int,
        best_val_loss: float,
        config: Optional[Dict] = None,
        final: bool = False,
    ):
        """Save full training checkpoint.

        Saves:
            - adapter_step_X.safetensors: Adapter weights
            - checkpoint_step_X.pt: Training state (optimizer, scheduler, etc.)

        Args:
            adapter: Adapter module
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            global_step: Current global step
            epoch: Current epoch
            best_val_loss: Best validation loss so far
            config: Optional config dict to save
            final: Whether this is the final checkpoint
        """
        step_name = "final" if final else f"{global_step}"

        # Save adapter weights in SafeTensors format
        adapter_path = self.checkpoints_dir / f"adapter_step_{step_name}.safetensors"
        adapter_state = adapter.state_dict()

        # Convert to CPU and contiguous for SafeTensors
        adapter_state_cpu = {
            k: v.cpu().contiguous() for k, v in adapter_state.items()
        }
        save_file(adapter_state_cpu, str(adapter_path))
        logger.info(f"Saved adapter weights to {adapter_path}")

        # Save training state
        state_path = self.checkpoints_dir / f"checkpoint_step_{step_name}.pt"
        checkpoint = {
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_step": global_step,
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "timestamp": datetime.now().isoformat(),
        }

        if config is not None:
            checkpoint["config"] = config

        torch.save(checkpoint, state_path)
        logger.info(f"Saved training state to {state_path}")

    def save_best(
        self,
        adapter: torch.nn.Module,
        val_loss: float,
        global_step: int,
        epoch: int,
    ):
        """Save best model checkpoint.

        Args:
            adapter: Adapter module
            val_loss: Validation loss
            global_step: Current global step
            epoch: Current epoch
        """
        adapter_path = self.checkpoints_dir / "adapter_best.safetensors"
        adapter_state = adapter.state_dict()

        # Convert to CPU and contiguous
        adapter_state_cpu = {
            k: v.cpu().contiguous() for k, v in adapter_state.items()
        }
        save_file(adapter_state_cpu, str(adapter_path))

        # Save metadata
        meta_path = self.checkpoints_dir / "best_model_meta.json"
        metadata = {
            "val_loss": val_loss,
            "global_step": global_step,
            "epoch": epoch,
            "timestamp": datetime.now().isoformat(),
        }
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved best model with val_loss={val_loss:.4f} to {adapter_path}")

    def load(
        self,
        adapter: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        checkpoint_path: Optional[Path] = None,
        device: str = "cuda",
    ) -> Dict[str, Any]:
        """Load checkpoint for resuming training.

        Args:
            adapter: Adapter module to load weights into
            optimizer: Optional optimizer to load state into
            scheduler: Optional scheduler to load state into
            checkpoint_path: Path to checkpoint (if None, loads latest)
            device: Device to load weights to

        Returns:
            Dictionary with loaded checkpoint metadata
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_latest_checkpoint()
            if checkpoint_path is None:
                raise ValueError("No checkpoints found")

        checkpoint_path = Path(checkpoint_path)
        logger.info(f"Loading checkpoint from {checkpoint_path}")

        # Determine paths
        if checkpoint_path.suffix == ".safetensors":
            adapter_path = checkpoint_path
            step_name = checkpoint_path.stem.replace("adapter_step_", "")
            state_path = self.checkpoints_dir / f"checkpoint_step_{step_name}.pt"
        else:
            state_path = checkpoint_path
            step_name = checkpoint_path.stem.replace("checkpoint_step_", "")
            adapter_path = self.checkpoints_dir / f"adapter_step_{step_name}.safetensors"

        # Load adapter weights
        adapter_state = load_file(str(adapter_path), device=device)
        adapter.load_state_dict(adapter_state)
        logger.info("Loaded adapter weights")

        # Load training state
        training_state = torch.load(state_path, map_location=device)

        if optimizer is not None and "optimizer_state_dict" in training_state:
            optimizer.load_state_dict(training_state["optimizer_state_dict"])
            logger.info("Loaded optimizer state")

        if scheduler is not None and "scheduler_state_dict" in training_state:
            scheduler.load_state_dict(training_state["scheduler_state_dict"])
            logger.info("Loaded scheduler state")

        return {
            "global_step": training_state.get("global_step", 0),
            "epoch": training_state.get("epoch", 0),
            "best_val_loss": training_state.get("best_val_loss", float('inf')),
        }

    def load_adapter_only(
        self,
        adapter: torch.nn.Module,
        adapter_path: Path,
        device: str = "cuda",
    ):
        """Load just adapter weights (for inference).

        Args:
            adapter: Adapter module to load weights into
            adapter_path: Path to adapter weights
            device: Device to load weights to
        """
        adapter_state = load_file(str(adapter_path), device=device)
        adapter.load_state_dict(adapter_state)
        logger.info(f"Loaded adapter weights from {adapter_path}")

    def get_latest_checkpoint(self) -> Optional[Path]:
        """Find the latest checkpoint by step number.

        Returns:
            Path to latest checkpoint, or None if no checkpoints found
        """
        checkpoints = list(self.checkpoints_dir.glob("checkpoint_step_*.pt"))
        if not checkpoints:
            return None

        # Sort by step number
        def get_step(p: Path) -> int:
            try:
                stem = p.stem.replace("checkpoint_step_", "")
                return int(stem) if stem != "final" else float('inf')
            except:
                return -1

        latest = max(checkpoints, key=get_step)
        logger.info(f"Found latest checkpoint: {latest}")
        return latest

    def list_checkpoints(self) -> List[Path]:
        """List all available checkpoints.

        Returns:
            List of checkpoint paths, sorted by step number
        """
        checkpoints = list(self.checkpoints_dir.glob("checkpoint_step_*.pt"))

        def get_step(p: Path) -> int:
            try:
                stem = p.stem.replace("checkpoint_step_", "")
                return int(stem) if stem != "final" else float('inf')
            except:
                return -1

        return sorted(checkpoints, key=get_step)
