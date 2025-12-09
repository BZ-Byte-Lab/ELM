#!/usr/bin/env python3
"""
Main script to run the ELM data synthesis pipeline.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from synthesis_pipeline.config import SynthesisConfig
from synthesis_pipeline.generator import SynthesisGenerator
from synthesis_pipeline.validator import SynthesisValidator
from synthesis_pipeline.utils import setup_logging, format_time


def main():
    """Main entry point for synthesis pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the ELM data synthesis pipeline"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Splits to process (default: train val test)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation on existing outputs"
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=20,
        help="Save checkpoint every N generations (default: 20)"
    )
    parser.add_argument(
        "--requests-per-second",
        type=float,
        default=1.0,
        help="API rate limit (default: 1.0)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (default: logs/synthesis.log)"
    )

    args = parser.parse_args()

    # Initialize configuration
    config = SynthesisConfig()

    if args.checkpoint_interval:
        config.checkpoint_interval = args.checkpoint_interval
    if args.requests_per_second:
        config.requests_per_second = args.requests_per_second

    # Set up logging
    log_file = Path(args.log_file) if args.log_file else config.base_dir / "logs" / "synthesis.log"
    logger = setup_logging(log_file=log_file)

    logger.info("=" * 80)
    logger.info("ELM Data Synthesis Pipeline")
    logger.info("=" * 80)

    total_start = time.time()

    try:
        if args.validate_only:
            # Validation only
            logger.info("Running validation only...")
            validator = SynthesisValidator(config)
            results = validator.validate_all(args.splits)
            report = validator.generate_report(results)
            print(report)
        else:
            # Full synthesis
            generator = SynthesisGenerator(config)
            stats = generator.run(args.splits)

            # Run validation
            logger.info("\nRunning post-generation validation...")
            validator = SynthesisValidator(config)
            results = validator.validate_all(args.splits)
            report = validator.generate_report(results)
            print(report)

            # Print summary
            logger.info("\n" + "=" * 80)
            logger.info("SYNTHESIS COMPLETE")
            logger.info("=" * 80)

            for split, split_stats in stats.items():
                logger.info(f"\n{split}:")
                logger.info(f"  Generated: {split_stats['total_generated']}")
                logger.info(f"  Rejected: {split_stats['total_rejected']}")

        total_time = time.time() - total_start
        logger.info(f"\nTotal time: {format_time(total_time)}")

    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nPipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
