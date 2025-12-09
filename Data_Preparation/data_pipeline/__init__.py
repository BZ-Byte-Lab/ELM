"""
ELM Data Pipeline
~~~~~~~~~~~~~~~~~

A data preparation pipeline for Embedding Language Models using WikiText-103 and Qwen3-Embedding-4B.
"""

__version__ = "0.1.0"

from .config import Config
from .dataset import ELMDataset, ELMCollator

__all__ = [
    "Config",
    "ELMDataset",
    "ELMCollator",
]