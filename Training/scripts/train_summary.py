#!/usr/bin/env python3
"""Summary-only training script for ELM model with BERTScore evaluation.

This script uses the summary_training_pipeline for training only on summary tasks
with enhanced BERTScore evaluation and early stopping based on validation BERTScore.
"""

import argparse
import logging
import sys
from pathlib import Path
import torch
import wandb

# Add paths for modules
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "summary_training_pipeline"))
sys.path.append(str(Path(__file__).parent.parent / "optimization"))

try:
    from summary_training_pipeline.config import SummaryTrainingConfig
    from summary_training_pipeline.trainer import ELMTrainer
    from summary_training_pipeline.dataset import ELMTrainingDataset, TrainingCollator
    from summary_training_pipeline.utils import setup_logging
    from optimization.bertscore_metrics import create_evaluator
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the Training directory:")
    print("cd Training && python scripts/train_summary.py ...")
    sys.exit(1)

# Set up logging
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ELM model on summary tasks only")

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/summary_filtered"),
        help="Directory containing summary-only data"
    )

    # Model configuration
    parser.add_argument(
        "--llm-model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct",
        help="LLM model name"
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=4096,
        help="Adapter hidden dimension"
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=2560,
        help="Embedding dimension"
    )

    # Training hyperparameters
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps"
    )

    # Loss configuration
    parser.add_argument(
        "--use-contrastive",
        action="store_true",
        default=True,
        help="Use contrastive loss"
    )
    parser.add_argument(
        "--contrastive-weight",
        type=float,
        default=0.01,
        help="Weight for contrastive loss"
    )
    parser.add_argument(
        "--use-drift-loss",
        action="store_true",
        default=True,
        help="Use text drift loss"
    )
    parser.add_argument(
        "--drift-weight",
        type=float,
        default=0.03,
        help="Weight for text drift loss"
    )
    parser.add_argument(
        "--drift-target-similarity",
        type=float,
        default=0.75,
        help="Target cosine similarity for drift loss"
    )

    # Adapter configuration
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=0.1,
        help="Residual connection scale"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate"
    )

    # BERTScore configuration
    parser.add_argument(
        "--bertscore-model",
        type=str,
        default="microsoft/deberta-xlarge-mnli",
        help="BERTScore evaluation model"
    )
    parser.add_argument(
        "--bertscore-batch-size",
        type=int,
        default=16,
        help="BERTScore computation batch size"
    )

    # Checkpointing
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("data/checkpoints"),
        help="Checkpoint directory"
    )

    # Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Training device"
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Logging interval steps"
    )

    # Weights & Biases
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="elm-summary",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name"
    )

    # Early stopping
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Early stopping patience epochs"
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=250,
        help="Evaluation interval steps"
    )

    return parser.parse_args()


def create_config(args) -> SummaryTrainingConfig:
    """Create training configuration from arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        SummaryTrainingConfig instance
    """
    config = SummaryTrainingConfig()

    # Update with command line arguments
    config.llm_model_name = args.llm_model
    config.embedding_dim = args.embedding_dim
    config.hidden_dim = args.hidden_dim
    config.learning_rate = args.learning_rate
    config.batch_size = args.batch_size
    config.max_epochs = args.epochs
    config.max_steps = args.max_steps

    # Loss configuration
    config.use_contrastive_loss = args.use_contrastive
    config.contrastive_weight = args.contrastive_weight
    config.use_text_drift_loss = args.use_drift_loss
    config.text_drift_weight = args.drift_weight
    config.text_drift_target_similarity = args.drift_target_similarity

    # Adapter configuration
    config.residual_scale = args.residual_scale
    config.dropout_rate = args.dropout

    # BERTScore configuration
    config.bertscore_model = args.bertscore_model
    config.bertscore_batch_size = args.bertscore_batch_size

    # Data configuration
    # Convert data_dir to absolute path relative to project root
    project_root = Path(__file__).parent.parent.parent
    data_path = project_root / args.data_dir
    config.summary_data_path = str(data_path)
    # Initialize paths after setting summary_data_path
    config.__post_init__()

    # Logging configuration
    config.logging_steps = args.log_interval
    config.eval_steps = args.eval_interval

    # Weights & Biases
    config.use_wandb = args.wandb
    if args.wandb:
        config.wandb_project = args.wandb_project
        config.wandb_run_name = args.wandb_run_name

    # Checkpoint configuration
    config.checkpoints_dir = args.checkpoint_dir

    return config


def evaluate_with_bertscore(
    trainer: ELMTrainer,
    eval_dataset: ELMTrainingDataset,
    evaluator,
    device: str
) -> dict:
    """Evaluate model with BERTScore.

    Args:
        trainer: ELM trainer instance
        eval_dataset: Evaluation dataset
        evaluator: BERTScore evaluator
        device: Device for evaluation

    Returns:
        Dictionary with BERTScore metrics
    """
    logger.info("Running BERTScore evaluation...")

    # Create data loader for evaluation
    collator = TrainingCollator(trainer.model.tokenizer)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=trainer.config.bertscore_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,  # No workers for evaluation
    )

    # Generate predictions
    predictions = []
    references = []

    trainer.model.eval()
    with torch.no_grad():
        for batch in eval_loader:
            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Generate summary
            generated_ids = trainer.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                embeddings=batch["embeddings"],
                embedding_positions=batch["embedding_positions"],
                max_new_tokens=150,  # Reasonable length for summaries
                do_sample=False,     # Deterministic generation
                temperature=1.0,
                top_p=1.0,
                pad_token_id=trainer.model.tokenizer.pad_token_id,
            )

            # Decode predictions and references
            # Filter generated_ids to only include valid token IDs
            vocab_size = trainer.model.tokenizer.vocab_size
            filtered_generated = generated_ids.clone()

            # Filter out invalid token IDs (negative or too large)
            filtered_generated[filtered_generated < 0] = trainer.model.tokenizer.pad_token_id
            filtered_generated[filtered_generated >= vocab_size] = trainer.model.tokenizer.pad_token_id

            batch_predictions = trainer.model.tokenizer.batch_decode(
                filtered_generated, skip_special_tokens=True
            )

            # Also filter labels to prevent any potential issues
            filtered_labels = batch["labels"].clone()

            # Filter out invalid token IDs (negative or too large)
            # Handle -100 (ignore index) and other negative values
            filtered_labels[filtered_labels < 0] = trainer.model.tokenizer.pad_token_id
            filtered_labels[filtered_labels >= vocab_size] = trainer.model.tokenizer.pad_token_id

            batch_references = trainer.model.tokenizer.batch_decode(
                filtered_labels, skip_special_tokens=True
            )

            predictions.extend(batch_predictions)
            references.extend(batch_references)

    # Compute BERTScore
    bertscore_metrics = evaluator.evaluate_batch(predictions, references)

    logger.info(f"BERTScore evaluation completed:")
    logger.info(f"  Precision: {bertscore_metrics['bertscore_precision']:.4f}")
    logger.info(f"  Recall: {bertscore_metrics['bertscore_recall']:.4f}")
    logger.info(f"  F1: {bertscore_metrics['bertscore_f1']:.4f}")
    logger.info(f"  Composite: {bertscore_metrics['bertscore_composite']:.4f}")

    return bertscore_metrics


def main():
    """Main training function."""
    args = parse_args()

    # Set up logging
    setup_logging(args.log_level)
    logger.info("Starting summary-only ELM training")

    # Create configuration
    config = create_config(args)
    logger.info(f"Configuration: {config}")

    # Initialize wandb if requested
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            tags=["summary-only", "bertscore"]
        )

    # Create trainer
    logger.info("Initializing trainer")
    trainer = ELMTrainer(config, use_wandb=args.wandb)

    # Create BERTScore evaluator
    evaluator = create_evaluator(config)

    # Create evaluation dataset
    eval_dataset = ELMTrainingDataset(
        config.get_synthesis_path("val"),
        config.get_embeddings_path("val")
    )

    # Custom training loop with BERTScore evaluation
    logger.info("Starting training with BERTScore evaluation")

    best_bertscore = 0.0
    patience_counter = 0

    try:
        # Load datasets
        train_loader = trainer._create_dataloader("train")
        val_loader = trainer._create_dataloader("val")

        steps_per_epoch = len(train_loader) // config.gradient_accumulation_steps
        total_steps = steps_per_epoch * config.max_epochs

        logger.info(f"Training {steps_per_epoch} steps per epoch for {config.max_epochs} epochs")
        logger.info(f"Total steps: {total_steps}")

        # Create scheduler manually since we're not using trainer.train()
        trainer._create_scheduler(total_steps)

        # Training epochs
        for epoch in range(config.max_epochs):
            logger.info(f"\n{'='*60}\nEpoch {epoch + 1}/{config.max_epochs}\n{'='*60}")

            # Training epoch
            trainer._train_epoch(train_loader, val_loader, scaler=None)

            # BERTScore evaluation
            bertscore_metrics = evaluate_with_bertscore(
                trainer, eval_dataset, evaluator, args.device
            )

            current_bertscore = bertscore_metrics["bertscore_composite"]

            # Log to wandb
            if args.wandb:
                wandb.log({
                    "epoch": epoch + 1,
                    "eval/bertscore_precision": bertscore_metrics["bertscore_precision"],
                    "eval/bertscore_recall": bertscore_metrics["bertscore_recall"],
                    "eval/bertscore_f1": bertscore_metrics["bertscore_f1"],
                    "eval/bertscore_composite": current_bertscore,
                    "eval/num_samples": bertscore_metrics["num_samples"],
                })

            # Early stopping based on BERTScore
            if current_bertscore > best_bertscore:
                best_bertscore = current_bertscore
                patience_counter = 0

                # Save best model
                best_checkpoint_path = config.checkpoints_dir / "best_summary_model.safetensors"
                trainer._save_checkpoint(best_checkpoint_path, epoch, total_steps)
                logger.info(f"New best BERTScore: {best_bertscore:.4f} - saved checkpoint")

            else:
                patience_counter += 1
                logger.info(f"BERTScore: {current_bertscore:.4f} (best: {best_bertscore:.4f})")

                if patience_counter >= args.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {patience_counter} epochs without improvement")
                    break

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        if args.wandb:
            wandb.finish()

    logger.info(f"Training completed. Best BERTScore: {best_bertscore:.4f}")


if __name__ == "__main__":
    main()