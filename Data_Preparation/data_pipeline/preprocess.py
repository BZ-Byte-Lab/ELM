"""
Module for preprocessing WikiText-103 dataset.
"""

import re
import random
from typing import List, Tuple, Dict
from datasets import DatasetDict
import polars as pl
from tqdm import tqdm

from .utils import get_logger
from .config import Config

logger = get_logger("preprocess")


def extract_paragraphs(dataset: DatasetDict) -> List[str]:
    """Extract paragraphs from HuggingFace dataset.

    WikiText-103 contains one line per dataset item. This function
    merges consecutive non-empty lines into paragraphs.

    Args:
        dataset: HuggingFace DatasetDict

    Returns:
        List of paragraph strings
    """
    logger.info("Extracting paragraphs from dataset...")

    all_paragraphs = []
    current_paragraph = []

    # Process all splits together
    for split in ["train", "validation", "test"]:
        logger.info(f"Processing {split} split...")

        for item in tqdm(dataset[split], desc=f"Extracting from {split}"):
            text = item.get("text", "").strip()

            # Empty line indicates paragraph break
            if not text:
                if current_paragraph:
                    # Join accumulated lines into a paragraph
                    paragraph = " ".join(current_paragraph)
                    all_paragraphs.append(paragraph)
                    current_paragraph = []
            else:
                # Add non-empty line to current paragraph
                current_paragraph.append(text)

        # Don't forget the last paragraph
        if current_paragraph:
            paragraph = " ".join(current_paragraph)
            all_paragraphs.append(paragraph)
            current_paragraph = []

    logger.info(f"Extracted {len(all_paragraphs)} raw paragraphs")
    return all_paragraphs


def clean_text(text: str) -> str:
    """Clean Wikipedia formatting artifacts from text.

    Args:
        text: Raw text string

    Returns:
        Cleaned text string
    """
    # Remove Wikipedia section headers (e.g., " = = Section = = ")
    text = re.sub(r'\s*=\s+=\s+.*?\s+=\s+=\s*', ' ', text)
    text = re.sub(r'\s*=\s+.*?\s+=\s*', ' ', text)

    # Remove Wikipedia formatting artifacts
    text = text.replace("@-@", "-")
    text = text.replace("@.@", ".")
    text = text.replace("@,@", ",")

    # Remove <unk> tokens
    text = text.replace("<unk>", "")

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def is_low_quality(text: str) -> bool:
    """Check if text is low quality and should be filtered out.

    Args:
        text: Text string to check

    Returns:
        True if text is low quality, False otherwise
    """
    if not text:
        return True

    # Calculate character statistics
    total_chars = len(text)
    if total_chars == 0:
        return True

    # Count special characters and digits
    special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
    digits = sum(1 for c in text if c.isdigit())

    # Filter if mostly special characters (>50%)
    if special_chars / total_chars > 0.5:
        return True

    # Filter if mostly digits (>50%)
    if digits / total_chars > 0.5:
        return True

    # Filter very short texts (< 20 characters)
    if total_chars < 20:
        return True

    return False


def filter_paragraphs(
    paragraphs: List[str],
    tokenizer,
    min_tokens: int,
    max_tokens: int
) -> List[Dict[str, any]]:
    """Filter paragraphs by token count and quality.

    Args:
        paragraphs: List of paragraph strings
        tokenizer: HuggingFace tokenizer for counting tokens
        min_tokens: Minimum number of tokens
        max_tokens: Maximum number of tokens

    Returns:
        List of dictionaries with text and metadata
    """
    logger.info(f"Filtering paragraphs (token range: {min_tokens}-{max_tokens})...")

    filtered_data = []

    for text in tqdm(paragraphs, desc="Filtering"):
        # Clean the text
        cleaned_text = clean_text(text)

        # Skip low-quality text
        if is_low_quality(cleaned_text):
            continue

        # Tokenize to count tokens
        tokens = tokenizer.encode(cleaned_text, add_special_tokens=False)
        num_tokens = len(tokens)

        # Filter by token count
        if min_tokens <= num_tokens <= max_tokens:
            filtered_data.append({
                "text": cleaned_text,
                "token_count": num_tokens,
                "char_count": len(cleaned_text),
            })

    logger.info(f"Kept {len(filtered_data)}/{len(paragraphs)} paragraphs "
                f"({100 * len(filtered_data) / len(paragraphs):.1f}%)")

    return filtered_data


def split_dataset(
    data: List[Dict[str, any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """Split data into train, validation, and test sets.

    Args:
        data: List of data dictionaries
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    logger.info(f"Splitting dataset (train/val/test = {train_ratio}/{val_ratio}/{test_ratio})...")

    # Shuffle data with fixed seed
    random.seed(random_seed)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)

    # Calculate split indices
    n_total = len(shuffled_data)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # Split the data
    train_data = shuffled_data[:n_train]
    val_data = shuffled_data[n_train:n_train + n_val]
    test_data = shuffled_data[n_train + n_val:]

    logger.info(f"Train: {len(train_data)} samples")
    logger.info(f"Val: {len(val_data)} samples")
    logger.info(f"Test: {len(test_data)} samples")

    return train_data, val_data, test_data


def save_processed_data(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    config: Config
):
    """Save processed data as parquet files.

    Args:
        train_data: Training data
        val_data: Validation data
        test_data: Test data
        config: Configuration object
    """
    logger.info("Saving processed data as parquet files...")

    # Create output directory
    config.create_directories()

    # Add text IDs to each split
    for idx, item in enumerate(train_data):
        item["text_id"] = f"train_{idx}"

    for idx, item in enumerate(val_data):
        item["text_id"] = f"val_{idx}"

    for idx, item in enumerate(test_data):
        item["text_id"] = f"test_{idx}"

    # Save each split
    splits = {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }

    for split_name, split_data in splits.items():
        output_path = config.get_processed_path(split_name)

        # Convert to Polars DataFrame and save
        df = pl.DataFrame(split_data)
        df.write_parquet(output_path)

        logger.info(f"Saved {split_name}: {output_path} ({len(split_data)} samples)")

    logger.info("All data saved successfully")
