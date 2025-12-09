"""
Checkpointing and resume functionality for synthesis pipeline.
"""

import asyncio
import json
import hashlib
from pathlib import Path
from typing import Dict, Set, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

from .utils import get_logger
from .config import SynthesisConfig

logger = get_logger("checkpoint")


@dataclass
class CheckpointState:
    """State of a synthesis checkpoint."""
    split: str
    task_name: str
    completed_indices: Set[int]
    total_generations: int
    last_update: str
    rejection_count: int
    metadata: Dict[str, Any]


class CheckpointManager:
    """Manages checkpoints for synthesis pipeline."""

    def __init__(self, config: SynthesisConfig):
        """Initialize checkpoint manager.

        Args:
            config: Synthesis configuration
        """
        self.config = config
        self.output_hashes: Set[str] = set()
        self.lock = asyncio.Lock()  # Thread-safety for concurrent batch processing
        self._load_hash_cache()

    def _load_hash_cache(self) -> None:
        """Load existing output hashes to detect duplicates."""
        hash_cache_path = self.config.checkpoints_dir / "hash_cache.json"
        if hash_cache_path.exists():
            with open(hash_cache_path, 'r') as f:
                self.output_hashes = set(json.load(f))
            logger.info(f"Loaded {len(self.output_hashes)} existing hashes")

    def _save_hash_cache(self) -> None:
        """Save output hashes."""
        hash_cache_path = self.config.checkpoints_dir / "hash_cache.json"
        with open(hash_cache_path, 'w') as f:
            json.dump(list(self.output_hashes), f)

    async def is_duplicate(self, text: str) -> bool:
        """Check if output is duplicate using hash.

        Args:
            text: Generated text

        Returns:
            True if duplicate detected
        """
        async with self.lock:
            text_hash = hashlib.md5(text.encode()).hexdigest()
            if text_hash in self.output_hashes:
                return True
            self.output_hashes.add(text_hash)
            return False

    async def save_checkpoint(
        self,
        split: str,
        task_name: str,
        completed_indices: Set[int],
        total_generations: int,
        rejection_count: int = 0,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Save checkpoint state.

        Args:
            split: Data split name
            task_name: Task type name
            completed_indices: Set of completed embedding indices
            total_generations: Total number of generations so far
            rejection_count: Number of rejections
            metadata: Additional metadata
        """
        async with self.lock:
            state = CheckpointState(
                split=split,
                task_name=task_name,
                completed_indices=completed_indices,
                total_generations=total_generations,
                last_update=datetime.now().isoformat(),
                rejection_count=rejection_count,
                metadata=metadata or {},
            )

            checkpoint_path = self.config.get_checkpoint_path(split, task_name)

            # Convert to serializable dict
            state_dict = asdict(state)
            state_dict["completed_indices"] = list(state.completed_indices)

            with open(checkpoint_path, 'w') as f:
                json.dump(state_dict, f, indent=2)

            # Also save hash cache
            self._save_hash_cache()

            logger.debug(f"Checkpoint saved: {checkpoint_path}")

    def load_checkpoint(
        self,
        split: str,
        task_name: str,
    ) -> Optional[CheckpointState]:
        """Load checkpoint state if exists.

        Args:
            split: Data split name
            task_name: Task type name

        Returns:
            CheckpointState or None if not found
        """
        checkpoint_path = self.config.get_checkpoint_path(split, task_name)

        if not checkpoint_path.exists():
            return None

        with open(checkpoint_path, 'r') as f:
            state_dict = json.load(f)

        # Convert back to CheckpointState
        state = CheckpointState(
            split=state_dict["split"],
            task_name=state_dict["task_name"],
            completed_indices=set(state_dict["completed_indices"]),
            total_generations=state_dict["total_generations"],
            last_update=state_dict["last_update"],
            rejection_count=state_dict.get("rejection_count", 0),
            metadata=state_dict.get("metadata", {}),
        )

        logger.info(f"Loaded checkpoint: {len(state.completed_indices)} completed")
        return state

    def should_checkpoint(self, count: int) -> bool:
        """Check if it's time to save a checkpoint.

        Args:
            count: Current generation count

        Returns:
            True if checkpoint should be saved
        """
        return count > 0 and count % self.config.checkpoint_interval == 0
