#!/usr/bin/env python3
"""
Main script to run the complete ELM data preparation pipeline.

This script:
1. Downloads WikiText-2 from HuggingFace
2. Preprocesses and filters paragraphs
3. Splits into train/val/test
4. Generates embeddings using Qwen3-Embedding-4B
5. Saves everything in the specified format
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.config import Config
from data_pipeline.utils import setup_logging, format_time
from data_pipeline.download import load_wikitext2
from data_pipeline.preprocess import (
    extract_paragraphs,
    filter_paragraphs,
    split_dataset,
    save_processed_data,
)
from data_pipeline.embeddings import (
    load_embedding_model,
    generate_and_save_embeddings,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run the complete ELM data preparation pipeline"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip downloading dataset (use existing processed data)"
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation (only preprocess text)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for embedding generation"
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=None,
        help="Override minimum tokens per paragraph"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override maximum tokens per paragraph"
    )
    parser.add_argument(
        "--no-flash-attention",
        action="store_true",
        help="Disable flash attention (useful if not available)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (default: logs/pipeline.log)"
    )

    args = parser.parse_args()

    # Initialize configuration
    config = Config()

    # Override config with command-line arguments
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.min_tokens is not None:
        config.min_tokens = args.min_tokens
    if args.max_tokens is not None:
        config.max_tokens = args.max_tokens
    if args.no_flash_attention:
        config.use_flash_attention = False

    # Set up logging
    log_file = None
    if args.log_file:
        log_file = Path(args.log_file)
    else:
        log_file = config.base_dir / "logs" / "pipeline.log"

    logger = setup_logging(log_file=log_file)

    # Print configuration
    logger.info("=" * 80)
    logger.info("ELM Data Preparation Pipeline")
    logger.info("=" * 80)
    logger.info(f"\n{config}\n")

    # Create necessary directories
    config.create_directories()

    # Track total time
    total_start_time = time.time()

    try:
        # Step 1: Download/Load WikiText-2
        if not args.skip_download:
            logger.info("=" * 80)
            logger.info("STEP 1: Loading WikiText-2 dataset")
            logger.info("=" * 80)

            step_start = time.time()
            dataset = load_wikitext2(config)
            logger.info(f"Step completed in {format_time(time.time() - step_start)}\n")

            # Step 2: Extract and preprocess paragraphs
            logger.info("=" * 80)
            logger.info("STEP 2: Extracting and preprocessing paragraphs")
            logger.info("=" * 80)

            step_start = time.time()
            paragraphs = extract_paragraphs(dataset)
            logger.info(f"Step completed in {format_time(time.time() - step_start)}\n")

            # Step 3: Filter paragraphs by token count
            logger.info("=" * 80)
            logger.info("STEP 3: Filtering paragraphs")
            logger.info("=" * 80)

            step_start = time.time()

            # Load tokenizer for filtering
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)

            filtered_data = filter_paragraphs(
                paragraphs=paragraphs,
                tokenizer=tokenizer,
                min_tokens=config.min_tokens,
                max_tokens=config.max_tokens
            )
            logger.info(f"Step completed in {format_time(time.time() - step_start)}\n")

            # Step 4: Split into train/val/test
            logger.info("=" * 80)
            logger.info("STEP 4: Splitting dataset")
            logger.info("=" * 80)

            step_start = time.time()
            train_data, val_data, test_data = split_dataset(
                data=filtered_data,
                train_ratio=config.train_ratio,
                val_ratio=config.val_ratio,
                test_ratio=config.test_ratio,
                random_seed=config.random_seed
            )
            logger.info(f"Step completed in {format_time(time.time() - step_start)}\n")

            # Step 5: Save processed data
            logger.info("=" * 80)
            logger.info("STEP 5: Saving processed data")
            logger.info("=" * 80)

            step_start = time.time()
            save_processed_data(train_data, val_data, test_data, config)
            logger.info(f"Step completed in {format_time(time.time() - step_start)}\n")
        else:
            logger.info("Skipping download and preprocessing (--skip-download)")

        # Step 6: Generate embeddings
        if not args.skip_embeddings:
            logger.info("=" * 80)
            logger.info("STEP 6: Generating embeddings")
            logger.info("=" * 80)

            step_start = time.time()

            # Load embedding model
            model, tokenizer = load_embedding_model(config)

            # Generate embeddings for each split
            for split_name in ["train", "val", "test"]:
                logger.info(f"\nProcessing {split_name} split...")
                generate_and_save_embeddings(
                    split_name=split_name,
                    config=config,
                    model=model,
                    tokenizer=tokenizer
                )

            logger.info(f"\nAll embeddings completed in {format_time(time.time() - step_start)}\n")
        else:
            logger.info("Skipping embedding generation (--skip-embeddings)")

        # Pipeline completed
        total_time = time.time() - total_start_time
        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Total time: {format_time(total_time)}")

        # Print output summary
        logger.info("\nOutput files:")
        logger.info(f"  Processed data: {config.processed_dir}")
        for split in ["train", "val", "test"]:
            parquet_path = config.get_processed_path(split)
            logger.info(f"    - {parquet_path.name}")

        if not args.skip_embeddings:
            logger.info(f"  Embeddings: {config.embeddings_dir}")
            for split in ["train", "val", "test"]:
                emb_path = config.get_embeddings_path(split)
                logger.info(f"    - {emb_path.name}")

        logger.info("\nYou can now use the ELMDataset class to load and use this data!")

    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nPipeline failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
