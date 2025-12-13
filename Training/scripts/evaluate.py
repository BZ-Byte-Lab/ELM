#!/usr/bin/env python
"""Evaluation script for trained ELM adapter."""

import argparse
import sys
import torch
from pathlib import Path
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from Training.training_pipeline.config import TrainingConfig
from Training.training_pipeline.model import ELMModel
from Training.training_pipeline.dataset import ELMTrainingDataset, TrainingCollator
from Training.training_pipeline.checkpoint import AdapterCheckpoint
from Training.training_pipeline.utils import setup_logging, get_device
from torch.utils.data import DataLoader


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained ELM adapter",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to adapter checkpoint (*.safetensors)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on",
    )

    # Generation parameters
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of samples to generate text for",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling top-p",
    )

    # Paths
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory (defaults to elm/ project root)",
    )

    # Other
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file",
    )

    return parser.parse_args()


@torch.no_grad()
def evaluate_loss(
    model: ELMModel,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    """Evaluate model loss on dataset.

    Args:
        model: ELM model
        dataloader: Dataloader
        device: Device to run on

    Returns:
        Average loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(dataloader, desc="Evaluating loss"):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            embeddings=batch["embeddings"],
            embedding_positions=batch["embedding_positions"],
            labels=batch["labels"],
        )

        total_loss += outputs["loss"].item()
        num_batches += 1

    return total_loss / num_batches if num_batches > 0 else float('inf')


@torch.no_grad()
def generate_samples(
    model: ELMModel,
    dataloader: DataLoader,
    device: torch.device,
    num_samples: int = 10,
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    """Generate text samples from embeddings.

    Args:
        model: ELM model
        dataloader: Dataloader
        device: Device to run on
        num_samples: Number of samples to generate
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling top-p
    """
    model.eval()
    samples_generated = 0

    print("\n" + "=" * 80)
    print("Generated Samples")
    print("=" * 80 + "\n")

    for batch in dataloader:
        if samples_generated >= num_samples:
            break

        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Generate
        generated_ids = model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            embeddings=batch["embeddings"],
            embedding_positions=batch["embedding_positions"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
        )

        # Decode
        generated_texts = model.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )

        # Print samples
        for i in range(min(len(generated_texts), num_samples - samples_generated)):
            print(f"Sample {samples_generated + 1}:")
            print("-" * 80)
            print(generated_texts[i])
            print("\n")
            samples_generated += 1


def main():
    """Main evaluation function."""
    args = parse_args()

    # Setup logging
    log_file = Path(args.log_file) if args.log_file else None
    setup_logging(log_file=log_file)

    import logging
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("ELM Adapter Evaluation")
    logger.info("=" * 80)

    # Get device
    device = get_device()

    # Create configuration
    config = TrainingConfig()
    if args.base_dir:
        config.base_dir = Path(args.base_dir)
        config.__post_init__()

    logger.info(f"Evaluating on {args.split} split")
    logger.info(f"Checkpoint: {args.checkpoint}")

    # Validate files exist
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    synthesis_path = config.get_synthesis_path(args.split)
    embeddings_path = config.get_embeddings_path(args.split)

    if not synthesis_path.exists():
        logger.error(f"Synthesis file not found: {synthesis_path}")
        sys.exit(1)

    if not embeddings_path.exists():
        logger.error(f"Embeddings file not found: {embeddings_path}")
        sys.exit(1)

    # Load model
    logger.info("Loading model...")
    model = ELMModel(config).to(device)

    # Load checkpoint
    logger.info("Loading checkpoint...")
    checkpoint_manager = AdapterCheckpoint(config.checkpoints_dir)
    checkpoint_manager.load_adapter_only(
        adapter=model.adapter,
        adapter_path=checkpoint_path,
        device=str(device),
    )
    logger.info("Checkpoint loaded successfully")

    # Create dataset and dataloader
    logger.info("Loading dataset...")
    dataset = ELMTrainingDataset(
        synthesis_path=synthesis_path,
        embeddings_path=embeddings_path,
    )

    collator = TrainingCollator(
        tokenizer=model.tokenizer,
        max_seq_length=config.max_seq_length,
        emb_token=config.emb_token,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=True,
    )

    logger.info(f"Dataset: {len(dataset)} samples")

    # Evaluate loss
    logger.info("\nEvaluating loss...")
    avg_loss = evaluate_loss(model, dataloader, device)
    logger.info(f"\nAverage Loss: {avg_loss:.4f}")

    # Generate samples
    if args.num_samples > 0:
        logger.info(f"\nGenerating {args.num_samples} samples...")
        generate_samples(
            model=model,
            dataloader=dataloader,
            device=device,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    logger.info("\n" + "=" * 80)
    logger.info("Evaluation Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
