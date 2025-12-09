"""
JSONL output writer for synthesis pipeline.
"""

import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime
import jsonlines

from .utils import get_logger
from .config import SynthesisConfig

logger = get_logger("output_writer")


@dataclass
class SynthesisOutput:
    """Single synthesis output record."""
    task_type: str
    input_prompt_template: str
    embedding_index: int
    target_text: str
    metadata: Dict[str, Any]


class OutputWriter:
    """Writes synthesis outputs to JSONL files."""

    def __init__(self, config: SynthesisConfig, split: str):
        """Initialize output writer.

        Args:
            config: Synthesis configuration
            split: Data split name
        """
        self.config = config
        self.split = split
        self.output_path = config.get_synthesis_path(split)
        self.buffer: List[SynthesisOutput] = []
        self.buffer_size = 50
        self.total_written = 0
        self.lock = asyncio.Lock()  # Thread-safety for concurrent batch processing

        # Ensure directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    async def write(self, output: SynthesisOutput) -> None:
        """Write a single output to buffer.

        Args:
            output: SynthesisOutput to write
        """
        async with self.lock:
            self.buffer.append(output)

            if len(self.buffer) >= self.buffer_size:
                self._flush_unlocked()

    async def write_batch(self, outputs: List[SynthesisOutput]) -> None:
        """Write multiple outputs to buffer.

        Args:
            outputs: List of SynthesisOutput objects
        """
        async with self.lock:
            self.buffer.extend(outputs)

            if len(self.buffer) >= self.buffer_size:
                self._flush_unlocked()

    def _flush_unlocked(self) -> None:
        """Flush buffer to disk (assumes lock already held)."""
        if not self.buffer:
            return

        with jsonlines.open(self.output_path, mode='a') as writer:
            for output in self.buffer:
                record = {
                    "task_type": output.task_type,
                    "input_prompt_template": output.input_prompt_template,
                    "embedding_index": output.embedding_index,
                    "target_text": output.target_text,
                    **output.metadata,
                }
                writer.write(record)

        self.total_written += len(self.buffer)
        logger.debug(f"Flushed {len(self.buffer)} records, total: {self.total_written}")
        self.buffer = []

    async def flush(self) -> None:
        """Flush buffer to disk."""
        async with self.lock:
            self._flush_unlocked()

    async def close(self) -> None:
        """Close writer and flush remaining buffer."""
        await self.flush()
        logger.info(f"Wrote {self.total_written} total records to {self.output_path}")


class MetadataWriter:
    """Writes synthesis metadata."""

    def __init__(self, config: SynthesisConfig):
        """Initialize metadata writer.

        Args:
            config: Synthesis configuration
        """
        self.config = config
        self.metadata_path = config.synthesis_dir / "metadata.json"

    def write(
        self,
        split_stats: Dict[str, Dict],
        task_stats: Dict[str, Dict],
        coverage_stats: Dict[str, Any],
    ) -> None:
        """Write comprehensive metadata.

        Args:
            split_stats: Statistics per split
            task_stats: Statistics per task
            coverage_stats: Coverage statistics
        """
        metadata = {
            "model": self.config.model_name,
            "timestamp": datetime.now().isoformat(),
            "splits": split_stats,
            "tasks": task_stats,
            "coverage": coverage_stats,
            "config": {
                "checkpoint_interval": self.config.checkpoint_interval,
                "variations_per_task": self.config.variations_per_task,
                "min_samples_per_embedding": self.config.min_samples_per_embedding,
            },
        }

        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata written to {self.metadata_path}")
