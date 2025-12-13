#!/usr/bin/env python
"""Main training script for ELM adapter."""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Training.training_pipeline.config import TrainingConfig
from Training.training_pipeline.trainer import ELMTrainer
from Training.training_pipeline.utils import setup_logging, set_seed


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ELM adapter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--llm-model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Base LLM model name",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=4096,
        help="Adapter intermediate hidden dimension",
    )

    # Training hyperparameters
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=1000,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping",
    )

    # Batch configuration
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size per device (40GB VRAM allows 16)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=2,
        help="Gradient accumulation steps",
    )

    # Training schedule
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of training steps (overrides epochs if set)",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=500,
        help="Evaluate every N steps",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=50,
        help="Log metrics every N steps",
    )

    # Data configuration
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of dataloader workers",
    )

    # Memory optimization
    parser.add_argument(
        "--no-bf16",
        action="store_true",
        help="Disable bfloat16 mixed precision",
    )
    parser.add_argument(
        "--no-grad-checkpoint",
        action="store_true",
        help="Disable gradient checkpointing",
    )

    # Checkpointing
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Paths
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory (defaults to elm/ project root)",
    )

    # Logging
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # Weights & Biases
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="elm-training",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Setup logging
    log_file = Path(args.log_file) if args.log_file else None
    setup_logging(log_file=log_file)

    import logging
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("ELM Adapter Training")
    logger.info("=" * 80)

    # Set random seed
    set_seed(args.seed)
    logger.info(f"Set random seed: {args.seed}")

    # Create configuration
    config = TrainingConfig(
        llm_model_name=args.llm_model,
        hidden_dim=args.hidden_dim,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        max_seq_length=args.max_seq_length,
        num_workers=args.num_workers,
        use_bf16=not args.no_bf16,
        use_gradient_checkpointing=not args.no_grad_checkpoint,
        random_seed=args.seed,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    # Override base_dir if specified
    if args.base_dir:
        config.base_dir = Path(args.base_dir)
        config.__post_init__()

    # Create directories
    config.create_directories()

    # Log configuration
    logger.info(f"\n{config}\n")
    logger.info(f"Data directory: {config.data_dir}")
    logger.info(f"Checkpoints directory: {config.checkpoints_dir}")

    # Validate data files exist
    for split in ["train", "val"]:
        synthesis_path = config.get_synthesis_path(split)
        embeddings_path = config.get_embeddings_path(split)

        if not synthesis_path.exists():
            logger.error(f"Synthesis file not found: {synthesis_path}")
            sys.exit(1)

        if not embeddings_path.exists():
            logger.error(f"Embeddings file not found: {embeddings_path}")
            sys.exit(1)

        logger.info(f"{split.capitalize()} synthesis: {synthesis_path}")
        logger.info(f"{split.capitalize()} embeddings: {embeddings_path}")

    # Create trainer
    logger.info("\n" + "=" * 80)
    logger.info("Initializing Trainer")
    logger.info("=" * 80 + "\n")

    trainer = ELMTrainer(config, use_wandb=args.wandb)

    # Resume path
    resume_path = Path(args.resume) if args.resume else None

    # Train
    logger.info("\n" + "=" * 80)
    logger.info("Starting Training")
    logger.info("=" * 80 + "\n")

    try:
        trainer.train(resume_from=resume_path)
    except KeyboardInterrupt:
        logger.info("\n\nTraining interrupted by user")
        logger.info("Saving checkpoint...")
        trainer.checkpoint_manager.save(
            adapter=trainer.model.adapter,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            global_step=trainer.global_step,
            epoch=trainer.epoch,
            best_val_loss=trainer.best_val_loss,
        )
        logger.info("Checkpoint saved. Exiting...")
        sys.exit(0)

    logger.info("\n" + "=" * 80)
    logger.info("Training Complete!")
    logger.info("=" * 80)
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
    logger.info(f"Final step: {trainer.global_step}")
    logger.info(f"Checkpoints saved to: {config.checkpoints_dir}")


if __name__ == "__main__":
    main()
