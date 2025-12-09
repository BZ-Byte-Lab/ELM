"""
Module for downloading WikiText-2 dataset from HuggingFace.
"""

from datasets import load_dataset, DatasetDict
from .utils import get_logger
from .config import Config

logger = get_logger("download")


def load_wikitext2(config: Config) -> DatasetDict:
    """Load WikiText-2 dataset from HuggingFace.

    Args:
        config: Configuration object

    Returns:
        DatasetDict containing train, validation, and test splits

    Raises:
        Exception: If download fails
    """
    logger.info(f"Loading dataset: {config.hf_dataset_name}/{config.hf_dataset_config}")

    try:
        dataset = load_dataset(
            config.hf_dataset_name,
            config.hf_dataset_config,
            trust_remote_code=False
        )

        logger.info("Dataset loaded successfully")
        logger.info(f"  Train samples: {len(dataset['train'])}")
        logger.info(f"  Validation samples: {len(dataset['validation'])}")
        logger.info(f"  Test samples: {len(dataset['test'])}")

        return dataset

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise


def get_dataset_info(dataset: DatasetDict) -> dict:
    """Get basic information about the dataset.

    Args:
        dataset: HuggingFace DatasetDict

    Returns:
        Dictionary with dataset statistics
    """
    info = {
        "splits": list(dataset.keys()),
        "num_samples": {split: len(dataset[split]) for split in dataset.keys()},
        "features": list(dataset[next(iter(dataset.keys()))].features.keys()),
    }

    # Calculate total text size (approximate)
    total_chars = 0
    for split in dataset.keys():
        for item in dataset[split]:
            total_chars += len(item.get("text", ""))

    info["total_chars"] = total_chars
    info["total_mb"] = total_chars / (1024 * 1024)

    return info
