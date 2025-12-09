"""
FAISS-based k-NN index for finding similar embeddings.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import faiss

from .utils import get_logger
from .config import SynthesisConfig

logger = get_logger("knn_index")


class KNNIndex:
    """FAISS-based k-NN index for embedding similarity search."""

    def __init__(self, config: SynthesisConfig):
        """Initialize KNNIndex.

        Args:
            config: Synthesis configuration
        """
        self.config = config
        self.index: Optional[faiss.Index] = None
        self.embeddings: Optional[np.ndarray] = None
        self.num_samples: int = 0

    def build_index(self, embeddings: np.ndarray) -> None:
        """Build FAISS index from embeddings.

        Args:
            embeddings: Numpy array of shape (n_samples, embedding_dim)
        """
        logger.info(f"Building k-NN index for {len(embeddings)} embeddings...")

        self.embeddings = embeddings.astype(np.float32)
        self.num_samples = len(embeddings)
        dim = embeddings.shape[1]

        # Create index - use IndexFlatIP for cosine similarity
        # (embeddings are already L2-normalized from Data_Preparation)
        self.index = faiss.IndexFlatIP(dim)

        # Optionally use GPU
        if self.config.use_gpu_index:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                logger.info("Using GPU-accelerated FAISS index")
            except Exception as e:
                logger.warning(f"GPU index failed, using CPU: {e}")

        # Add embeddings to index
        self.index.add(self.embeddings)

        logger.info(f"Index built successfully with {self.index.ntotal} vectors")

    def search(
        self,
        query_idx: int,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors for a given index.

        Args:
            query_idx: Index of query embedding
            k: Number of neighbors to return (excluding self)

        Returns:
            Tuple of (neighbor_indices, similarity_scores)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        # Get query embedding
        query = self.embeddings[query_idx:query_idx+1]

        # Search for k+1 (to exclude self)
        similarities, indices = self.index.search(query, k + 1)

        # Remove self from results
        mask = indices[0] != query_idx
        neighbor_indices = indices[0][mask][:k]
        neighbor_similarities = similarities[0][mask][:k]

        return neighbor_indices, neighbor_similarities

    def get_neighbors_batch(
        self,
        query_indices: List[int],
        k: int = 10
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Find k nearest neighbors for a batch of indices.

        Args:
            query_indices: List of query embedding indices
            k: Number of neighbors per query

        Returns:
            List of (neighbor_indices, similarity_scores) tuples
        """
        results = []
        for idx in query_indices:
            results.append(self.search(idx, k))
        return results

    def interpolate_embeddings(
        self,
        idx1: int,
        idx2: int,
        alpha: float
    ) -> np.ndarray:
        """Interpolate between two embeddings.

        Args:
            idx1: Index of first embedding
            idx2: Index of second embedding
            alpha: Interpolation factor (0.0 = idx1, 1.0 = idx2)

        Returns:
            Interpolated and normalized embedding
        """
        emb1 = self.embeddings[idx1]
        emb2 = self.embeddings[idx2]

        # Linear interpolation
        interpolated = (1 - alpha) * emb1 + alpha * emb2

        # Re-normalize to unit length
        norm = np.linalg.norm(interpolated)
        if norm > 0:
            interpolated = interpolated / norm

        return interpolated

    def save_index(self, path: Path) -> None:
        """Save FAISS index to disk.

        Args:
            path: Path to save index file
        """
        if self.index is not None:
            index_to_save = self.index
            if self.config.use_gpu_index:
                index_to_save = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(index_to_save, str(path))
            logger.info(f"Index saved to {path}")

    def load_index(self, path: Path, embeddings: np.ndarray) -> None:
        """Load FAISS index from disk.

        Args:
            path: Path to index file
            embeddings: Original embeddings array
        """
        self.index = faiss.read_index(str(path))
        self.embeddings = embeddings.astype(np.float32)
        self.num_samples = len(embeddings)
        logger.info(f"Index loaded from {path}")
