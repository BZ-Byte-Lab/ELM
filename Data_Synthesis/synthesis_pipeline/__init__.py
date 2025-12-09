"""
ELM Data Synthesis Pipeline.

This package provides tools for generating synthetic training data
from pre-computed embeddings using LLM-based text generation.
"""

from .config import SynthesisConfig, TaskCategory, TaskConfig
from .task_registry import TaskRegistry
from .knn_index import KNNIndex
from .api_client import OpenRouterClient, GenerationResult
from .quality_filter import QualityFilter
from .checkpoint import CheckpointManager, CheckpointState
from .output_writer import OutputWriter, SynthesisOutput
from .generator import SynthesisGenerator
from .validator import SynthesisValidator

__version__ = "0.1.0"

__all__ = [
    "SynthesisConfig",
    "TaskCategory",
    "TaskConfig",
    "TaskRegistry",
    "KNNIndex",
    "OpenRouterClient",
    "GenerationResult",
    "QualityFilter",
    "CheckpointManager",
    "CheckpointState",
    "OutputWriter",
    "SynthesisOutput",
    "SynthesisGenerator",
    "SynthesisValidator",
]
