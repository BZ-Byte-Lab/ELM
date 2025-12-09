#!/usr/bin/env python3
"""
Script to regenerate embeddings from existing processed data.

This is useful when:
- You want to change the embedding model
- You want to adjust embedding parameters
- Previous embedding generation was interrupted
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline.config import Config
from data_pipeline.utils import setup_logging, format_time, validate_file_exists
from data_pipeline.embeddings import (
    load_embedding_model,
    generate_and_save_embeddings,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate embeddings from existing processed data"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Which splits to process (default: all)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size for embedding generation"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Override embedding model name"
    )
    parser.add_argument(
        "--no-flash-attention",
        action="store_true",
        help="Disable flash attention"
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help="Disable FP16 (use FP32 instead)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (default: logs/embeddings.log)"
    )

    args = parser.parse_args()

    # Initialize configuration
    config = Config()

    # Override config with command-line arguments
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.model_name is not None:
        config.model_name = args.model_name
    if args.no_flash_attention:
        config.use_flash_attention = False
    if args.no_fp16:
        config.use_fp16 = False

    # Set up logging
    log_file = None
    if args.log_file:
        log_file = Path(args.log_file)
    else:
        log_file = config.base_dir / "logs" / "embeddings.log"

    logger = setup_logging(log_file=log_file)

    # Print configuration
    logger.info("=" * 80)
    logger.info("ELM Embedding Generation")
    logger.info("=" * 80)
    logger.info(f"\n{config}\n")

    # Validate that processed data exists
    logger.info("Validating processed data files...")
    for split in args.splits:
        parquet_path = config.get_processed_path(split)
        try:
            validate_file_exists(parquet_path, f"{split} data")
            logger.info(f"  ✓ {split}: {parquet_path}")
        except FileNotFoundError as e:
            logger.error(f"  ✗ {split}: {e}")
            logger.error("\nPlease run the full pipeline first (run_pipeline.py)")
            sys.exit(1)

    # Create embeddings directory
    config.create_directories()

    # Track total time
    total_start_time = time.time()

    try:
        # Load embedding model
        logger.info("\n" + "=" * 80)
        logger.info("Loading embedding model")
        logger.info("=" * 80)

        model, tokenizer = load_embedding_model(config)

        # Generate embeddings for each split
        logger.info("\n" + "=" * 80)
        logger.info("Generating embeddings")
        logger.info("=" * 80)

        for split_name in args.splits:
            logger.info(f"\nProcessing {split_name} split...")
            split_start = time.time()

            generate_and_save_embeddings(
                split_name=split_name,
                config=config,
                model=model,
                tokenizer=tokenizer
            )

            logger.info(f"{split_name} completed in {format_time(time.time() - split_start)}")

        # Completed
        total_time = time.time() - total_start_time
        logger.info("\n" + "=" * 80)
        logger.info("EMBEDDING GENERATION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"Total time: {format_time(total_time)}")

        # Print output summary
        logger.info("\nGenerated embeddings:")
        logger.info(f"  Directory: {config.embeddings_dir}")
        for split in args.splits:
            emb_path = config.get_embeddings_path(split)
            logger.info(f"    - {emb_path.name}")

    except KeyboardInterrupt:
        logger.warning("\nEmbedding generation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nEmbedding generation failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
