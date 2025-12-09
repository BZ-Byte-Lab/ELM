"""
Module for generating embeddings using Qwen3-Embedding-4B.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch import Tensor
from typing import List, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from safetensors.numpy import save_file
import polars as pl

from .utils import get_logger, count_parameters
from .config import Config

logger = get_logger("embeddings")


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Extract embeddings from the last token position.

    This function handles both left and right padding correctly.

    Args:
        last_hidden_states: Hidden states from model (batch_size, seq_len, hidden_dim)
        attention_mask: Attention mask (batch_size, seq_len)

    Returns:
        Embeddings tensor (batch_size, hidden_dim)
    """
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device),
            sequence_lengths
        ]


def load_embedding_model(
    config: Config
) -> Tuple[AutoModel, AutoTokenizer]:
    """Load Qwen3-Embedding-4B model and tokenizer.

    Args:
        config: Configuration object

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading embedding model: {config.model_name}")

    # Load tokenizer with left padding
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        padding_side='left'
    )

    # Load model
    try:
        if config.use_flash_attention and torch.cuda.is_available():
            logger.info("Loading model with flash_attention_2...")
            model = AutoModel.from_pretrained(
                config.model_name,
                attn_implementation="flash_attention_2",
                torch_dtype=torch.float16 if config.use_fp16 else torch.float32
            )
        else:
            logger.info("Loading model without flash attention...")
            model = AutoModel.from_pretrained(
                config.model_name,
                torch_dtype=torch.float16 if config.use_fp16 else torch.float32
            )
    except Exception as e:
        logger.warning(f"Failed to load with flash_attention_2: {e}")
        logger.info("Falling back to standard attention...")
        model = AutoModel.from_pretrained(
            config.model_name,
            torch_dtype=torch.float16 if config.use_fp16 else torch.float32
        )

    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info(f"Model loaded on GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("CUDA not available, using CPU (this will be slow)")

    # Log model info
    num_params = count_parameters(model)
    logger.info(f"Model parameters: {num_params:,}")

    model.eval()

    return model, tokenizer


@torch.no_grad()
def generate_embeddings_batch(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: List[str],
    config: Config,
    desc: str = "Generating embeddings"
) -> np.ndarray:
    """Generate embeddings for a list of texts.

    Args:
        model: Qwen3-Embedding-4B model
        tokenizer: Tokenizer
        texts: List of text strings
        config: Configuration object
        desc: Description for progress bar

    Returns:
        Numpy array of embeddings (n_texts, embedding_dim)
    """
    logger.info(f"Generating embeddings for {len(texts)} texts...")
    logger.info(f"Batch size: {config.batch_size}, Max length: {config.max_length}")

    all_embeddings = []
    num_batches = (len(texts) + config.batch_size - 1) // config.batch_size

    try:
        for i in tqdm(range(0, len(texts), config.batch_size), total=num_batches, desc=desc):
            batch_texts = texts[i:i + config.batch_size]

            # Tokenize batch
            batch_dict = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )

            # Move to device
            batch_dict = {k: v.to(model.device) for k, v in batch_dict.items()}

            # Forward pass
            outputs = model(**batch_dict)

            # Extract embeddings using last token pooling
            embeddings = last_token_pool(
                outputs.last_hidden_state,
                batch_dict['attention_mask']
            )

            # Normalize embeddings (L2 norm)
            embeddings = F.normalize(embeddings, p=2, dim=1)

            # Move to CPU and convert to numpy
            embeddings_np = embeddings.cpu().float().numpy()
            all_embeddings.append(embeddings_np)

    except RuntimeError as e:
        if "out of memory" in str(e):
            logger.error("CUDA out of memory error!")
            logger.error(f"Current batch size: {config.batch_size}")
            logger.error("Try reducing the batch_size in config.py")
            raise
        else:
            raise

    # Concatenate all batches
    all_embeddings = np.vstack(all_embeddings)

    logger.info(f"Generated embeddings shape: {all_embeddings.shape}")
    logger.info(f"Embedding dtype: {all_embeddings.dtype}")

    return all_embeddings


def save_embeddings(
    embeddings: np.ndarray,
    output_path,
    metadata: dict = None
):
    """Save embeddings as safetensors format.

    Args:
        embeddings: Numpy array of embeddings
        output_path: Path to save file
        metadata: Optional metadata dictionary
    """
    logger.info(f"Saving embeddings to {output_path}...")

    # Prepare tensors dictionary
    tensors = {"embeddings": embeddings}

    # Prepare metadata
    if metadata is None:
        metadata = {}

    metadata.update({
        "shape": str(embeddings.shape),
        "dtype": str(embeddings.dtype),
    })

    # Convert metadata values to strings
    metadata = {k: str(v) for k, v in metadata.items()}

    # Save as safetensors
    save_file(tensors, str(output_path), metadata=metadata)

    logger.info(f"Embeddings saved successfully")


def load_texts_from_parquet(parquet_path) -> List[str]:
    """Load texts from a parquet file.

    Args:
        parquet_path: Path to parquet file

    Returns:
        List of text strings
    """
    logger.info(f"Loading texts from {parquet_path}...")

    df = pl.read_parquet(parquet_path)
    texts = df["text"].to_list()

    logger.info(f"Loaded {len(texts)} texts")

    return texts


def generate_and_save_embeddings(
    split_name: str,
    config: Config,
    model: AutoModel,
    tokenizer: AutoTokenizer
):
    """Generate and save embeddings for a specific split.

    Args:
        split_name: Name of split ('train', 'val', or 'test')
        config: Configuration object
        model: Embedding model
        tokenizer: Tokenizer
    """
    # Load texts
    parquet_path = config.get_processed_path(split_name)
    texts = load_texts_from_parquet(parquet_path)

    # Generate embeddings
    embeddings = generate_embeddings_batch(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        config=config,
        desc=f"Embedding {split_name}"
    )

    # Save embeddings
    output_path = config.get_embeddings_path(split_name)
    save_embeddings(
        embeddings=embeddings,
        output_path=output_path,
        metadata={
            "split": split_name,
            "model": config.model_name,
            "num_texts": len(texts),
        }
    )

    logger.info(f"Completed embeddings for {split_name} split")
