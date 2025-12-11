"""
Unified dataset class for ELM training.
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from torch.utils.data import Dataset, DataLoader
from safetensors.numpy import load_file

from .utils import get_logger, validate_file_exists
from .config import Config

logger = get_logger("dataset")


class ELMDataset(Dataset):
    """Dataset class for ELM that loads text, embeddings, and metadata.

    This dataset stores (text, embedding, metadata) tuples and supports
    efficient loading and batching.
    """

    def __init__(
        self,
        data_dir: Path,
        split: str = "train",
        load_embeddings: bool = True,
        use_normalized_embeddings: bool = False,
        config: Optional[Config] = None
    ):
        """Initialize ELMDataset.

        Args:
            data_dir: Base directory containing processed data
            split: Dataset split ('train', 'val', or 'test')
            load_embeddings: Whether to load embeddings
            use_normalized_embeddings: Whether to load L2-normalized embeddings (for k-NN search)
            config: Optional configuration object
        """
        self.split = split
        self.load_embeddings = load_embeddings

        # Use config if provided, otherwise create default
        if config is None:
            config = Config()
            config.base_dir = data_dir.parent if data_dir.name == "data" else data_dir

        self.config = config

        # Load text data
        self.parquet_path = config.get_processed_path(split)
        validate_file_exists(self.parquet_path, f"{split} data file")

        logger.info(f"Loading {split} data from {self.parquet_path}")
        self.df = pl.read_parquet(self.parquet_path)

        # Extract columns
        self.texts = self.df["text"].to_list()
        self.text_ids = self.df["text_id"].to_list()
        self.token_counts = self.df["token_count"].to_list()
        self.char_counts = self.df["char_count"].to_list()

        logger.info(f"Loaded {len(self.texts)} samples")

        # Load embeddings if requested
        self.embeddings = None
        if load_embeddings:
            if use_normalized_embeddings:
                self.embeddings_path = config.get_normalized_embeddings_path(split)
            else:
                self.embeddings_path = config.get_embeddings_path(split)
            validate_file_exists(self.embeddings_path, f"{split} embeddings file")

            logger.info(f"Loading embeddings from {self.embeddings_path}")
            tensors = load_file(str(self.embeddings_path))
            self.embeddings = tensors["embeddings"]

            logger.info(f"Loaded embeddings shape: {self.embeddings.shape}")

            # Validate dimensions match
            if len(self.embeddings) != len(self.texts):
                raise ValueError(
                    f"Mismatch between number of texts ({len(self.texts)}) "
                    f"and embeddings ({len(self.embeddings)})"
                )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing:
                - text: The text string
                - embedding: The embedding vector (if load_embeddings=True)
                - metadata: Dictionary with text_id, token_count, char_count
        """
        item = {
            "text": self.texts[idx],
            "metadata": {
                "text_id": self.text_ids[idx],
                "token_count": self.token_counts[idx],
                "char_count": self.char_counts[idx],
            }
        }

        if self.embeddings is not None:
            item["embedding"] = self.embeddings[idx]

        return item

    def get_batch(self, indices: List[int]) -> Dict:
        """Get a batch of samples efficiently.

        Args:
            indices: List of sample indices

        Returns:
            Dictionary with batched data
        """
        texts = [self.texts[i] for i in indices]
        metadata = [{
            "text_id": self.text_ids[i],
            "token_count": self.token_counts[i],
            "char_count": self.char_counts[i],
        } for i in indices]

        batch = {
            "text": texts,
            "metadata": metadata,
        }

        if self.embeddings is not None:
            batch["embedding"] = np.stack([self.embeddings[i] for i in indices])

        return batch

    def interpolate_embeddings(
        self,
        idx1: int,
        idx2: int,
        alpha: float
    ) -> np.ndarray:
        """Linear interpolation between two embeddings.

        Args:
            idx1: Index of first embedding
            idx2: Index of second embedding
            alpha: Interpolation factor (0.0 = idx1, 1.0 = idx2)

        Returns:
            Interpolated embedding vector
        """
        if self.embeddings is None:
            raise ValueError("Embeddings not loaded. Set load_embeddings=True")

        if not (0 <= idx1 < len(self) and 0 <= idx2 < len(self)):
            raise ValueError(f"Indices must be in range [0, {len(self)})")

        if not (0.0 <= alpha <= 1.0):
            raise ValueError("Alpha must be in range [0.0, 1.0]")

        emb1 = self.embeddings[idx1]
        emb2 = self.embeddings[idx2]

        # Linear interpolation
        interpolated = (1 - alpha) * emb1 + alpha * emb2

        # Re-normalize to unit length
        norm = np.linalg.norm(interpolated)
        if norm > 0:
            interpolated = interpolated / norm

        return interpolated

    def get_dataloader(
        self,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 0
    ) -> DataLoader:
        """Create a PyTorch DataLoader for this dataset.

        Args:
            batch_size: Batch size
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes

        Returns:
            PyTorch DataLoader
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=ELMCollator(),
        )

    def get_statistics(self) -> Dict:
        """Get dataset statistics.

        Returns:
            Dictionary with statistics
        """
        stats = {
            "split": self.split,
            "num_samples": len(self),
            "avg_token_count": np.mean(self.token_counts),
            "avg_char_count": np.mean(self.char_counts),
            "min_token_count": np.min(self.token_counts),
            "max_token_count": np.max(self.token_counts),
            "min_char_count": np.min(self.char_counts),
            "max_char_count": np.max(self.char_counts),
        }

        if self.embeddings is not None:
            stats["embedding_dim"] = self.embeddings.shape[1]
            stats["embedding_dtype"] = str(self.embeddings.dtype)

        return stats


class ELMCollator:
    """Collate function for batching variable-length texts."""

    def __call__(self, batch: List[Dict]) -> Dict:
        """Collate a batch of samples.

        Args:
            batch: List of samples from ELMDataset

        Returns:
            Batched dictionary
        """
        # Extract fields
        texts = [item["text"] for item in batch]
        metadata = [item["metadata"] for item in batch]

        collated = {
            "text": texts,
            "metadata": metadata,
        }

        # Stack embeddings if present
        if "embedding" in batch[0]:
            embeddings = np.stack([item["embedding"] for item in batch])
            collated["embedding"] = embeddings

        return collated
