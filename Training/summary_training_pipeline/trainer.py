"""Trainer for ELM adapter."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import logging

from .config import SummaryTrainingConfig
from .model import ELMModel
from .dataset import ELMTrainingDataset, TrainingCollator
from .checkpoint import AdapterCheckpoint
from .utils import get_device, log_gpu_memory

logger = logging.getLogger(__name__)


class ELMTrainer:
    """Trainer for ELM adapter."""

    def __init__(
        self,
        config: SummaryTrainingConfig,
        use_wandb: bool = False,
    ):
        """Initialize trainer.

        Args:
            config: Training configuration
            use_wandb: Whether to use Weights & Biases logging
        """
        self.config = config
        self.device = get_device()

        # Initialize model
        logger.info("Initializing ELM model")
        self.model = ELMModel(config).to(self.device)

        # Initialize optimizer (only adapter parameters)
        logger.info("Initializing optimizer")
        self.optimizer = AdamW(
            self.model.adapter.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Initialize checkpoint manager
        self.checkpoint_manager = AdapterCheckpoint(config.checkpoints_dir)

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.scheduler = None  # Initialized in train()

        # Weights & Biases
        self.use_wandb = use_wandb
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
                logger.info("Weights & Biases logging enabled")
            except ImportError:
                logger.warning("wandb not installed, disabling W&B logging")
                self.use_wandb = False

    def _create_dataloader(self, split: str) -> DataLoader:
        """Create dataloader for a split.

        Args:
            split: 'train' or 'val'

        Returns:
            DataLoader instance
        """
        dataset = ELMTrainingDataset(
            synthesis_path=self.config.get_synthesis_path(split),
            embeddings_path=self.config.get_embeddings_path(split),
        )

        collator = TrainingCollator(
            tokenizer=self.model.tokenizer,
            max_seq_length=self.config.max_seq_length,
            emb_token=self.config.emb_token,
        )

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=(split == "train"),
            num_workers=self.config.num_workers,
            collate_fn=collator,
            pin_memory=True,
        )

    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler with warmup.

        Args:
            num_training_steps: Total number of training steps
        """
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=num_training_steps,
        )
        logger.info(
            f"Created scheduler with {self.config.warmup_steps} warmup steps, "
            f"{num_training_steps} total steps"
        )

    def _contrastive_loss(
        self,
        embeddings: torch.Tensor,
        temperature: float = 0.07
    ) -> torch.Tensor:
        """InfoNCE contrastive loss to prevent adapter collapse.

        Encourages different input embeddings to produce different outputs.

        Args:
            embeddings: Adapter outputs (batch_size, embedding_dim)
            temperature: Temperature parameter for scaling similarities

        Returns:
            Contrastive loss value
        """
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature

        # Mask out diagonal (self-similarity)
        mask = torch.eye(len(embeddings), device=embeddings.device).bool()
        sim_matrix.masked_fill_(mask, -1e9)

        # Create labels (each sample should be dissimilar to others)
        labels = torch.arange(len(embeddings), device=embeddings.device)

        # Cross-entropy loss
        loss = F.cross_entropy(sim_matrix, labels)

        return loss

    def _text_drift_loss(
        self,
        original_embeddings: torch.Tensor,
        last_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Simplified cosine similarity drift loss for summary fidelity.

        Penalizes generated summaries that drift too far from original text content
        using only cosine similarity with mean pooling.

        Args:
            original_embeddings: Original input embeddings (batch_size, embedding_dim)
            last_hidden_states: Last hidden states from LLM (batch_size, seq_len, hidden_dim)
            attention_mask: Attention mask (batch_size, seq_len)

        Returns:
            Cosine similarity drift loss value
        """
        batch_size = original_embeddings.shape[0]

        # Mean pooling of last hidden states
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        pooled_hidden = torch.sum(last_hidden_states * mask_expanded, dim=1)
        pooled_hidden = pooled_hidden / torch.sum(mask_expanded, dim=1).clamp(min=1e-9)

        # L2 normalization
        original_norm = F.normalize(original_embeddings, p=2, dim=1)
        pooled_norm = F.normalize(pooled_hidden, p=2, dim=1)

        # Cosine similarity
        similarity = F.cosine_similarity(original_norm, pooled_norm, dim=1)
        drift_loss = 1 - similarity

        return drift_loss.mean()

    def train(self, resume_from: Optional[Path] = None):
        """Main training loop.

        Args:
            resume_from: Optional checkpoint path to resume from
        """
        # Create dataloaders
        logger.info("Creating dataloaders")
        train_loader = self._create_dataloader("train")
        val_loader = self._create_dataloader("val")

        logger.info(f"Train dataset: {len(train_loader.dataset)} samples")
        logger.info(f"Val dataset: {len(val_loader.dataset)} samples")

        # Calculate total steps
        steps_per_epoch = len(train_loader) // self.config.gradient_accumulation_steps
        total_steps = steps_per_epoch * self.config.max_epochs
        if self.config.max_steps:
            total_steps = min(total_steps, self.config.max_steps)

        # Create scheduler
        self._create_scheduler(total_steps)

        # Resume from checkpoint if specified
        if resume_from:
            logger.info(f"Resuming from checkpoint: {resume_from}")
            checkpoint_info = self.checkpoint_manager.load(
                adapter=self.model.adapter,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                checkpoint_path=resume_from,
                device=str(self.device),
            )
            self.global_step = checkpoint_info["global_step"]
            self.epoch = checkpoint_info["epoch"]
            self.best_val_loss = checkpoint_info["best_val_loss"]
            logger.info(f"Resumed from step {self.global_step}, epoch {self.epoch}")

        # Initialize W&B
        if self.use_wandb:
            self.wandb.init(
                project=self.config.wandb_project or "elm-training",
                name=self.config.wandb_run_name,
                config=vars(self.config),
            )

        # Mixed precision scaler (only needed for FP16, not BF16)
        scaler = None  # BFloat16 doesn't need gradient scaling

        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Total steps: {total_steps}, Steps per epoch: {steps_per_epoch}")
        logger.info(f"Effective batch size: {self.config.batch_size * self.config.gradient_accumulation_steps}")
        log_gpu_memory(logger, "Initial ")

        # Training loop
        for epoch in range(self.epoch, self.config.max_epochs):
            self.epoch = epoch
            logger.info(f"\n{'='*60}\nEpoch {epoch + 1}/{self.config.max_epochs}\n{'='*60}")

            self._train_epoch(train_loader, val_loader, scaler)

            # Check max steps
            if self.config.max_steps and self.global_step >= self.config.max_steps:
                logger.info(f"Reached max_steps ({self.config.max_steps})")
                break

        # Final save
        logger.info("Saving final checkpoint")
        self.checkpoint_manager.save(
            adapter=self.model.adapter,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            global_step=self.global_step,
            epoch=self.epoch,
            best_val_loss=self.best_val_loss,
            final=True,
        )

        if self.use_wandb:
            self.wandb.finish()

        logger.info("Training complete!")

    def _train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        scaler: Optional[torch.cuda.amp.GradScaler],
    ):
        """Train for one epoch.

        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            scaler: Mixed precision scaler (None if not using bf16)
        """
        self.model.train()
        self.optimizer.zero_grad()

        accumulation_loss = 0.0
        original_losses = []  # Track original losses for logging
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch + 1}")

        for step, batch in enumerate(pbar):
            # Move batch to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward pass with mixed precision
            with torch.amp.autocast(
                device_type='cuda',
                dtype=torch.bfloat16,
                enabled=self.config.use_bf16,
            ):
                # Determine if we need hidden states for drift loss
                output_hidden_states = self.config.use_text_drift_loss

                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    embeddings=batch["embeddings"],
                    embedding_positions=batch["embedding_positions"],
                    labels=batch["labels"],
                    output_hidden_states=output_hidden_states,
                )

                # Get adapter outputs for contrastive loss
                adapter_outputs = self.model.get_adapter_outputs(batch["embeddings"])

                # Store original LM loss for logging
                lm_loss = outputs["loss"].item()
                original_losses.append(lm_loss)

                # Start with language modeling loss
                total_loss = outputs["loss"]

                # Add contrastive loss if enabled
                if self.config.use_contrastive_loss:
                    contra_loss = self._contrastive_loss(
                        adapter_outputs,
                        temperature=self.config.contrastive_temperature
                    )
                    total_loss += self.config.contrastive_weight * contra_loss

                # Add text drift loss if enabled
                if self.config.use_text_drift_loss and output_hidden_states:
                    drift_loss = self._text_drift_loss(
                        original_embeddings=batch["embeddings"],
                        last_hidden_states=outputs["hidden_states"],
                        attention_mask=batch["attention_mask"]
                    )
                    total_loss += self.config.text_drift_weight * drift_loss

                # Scale loss for gradient accumulation (backward pass only)
                loss = total_loss / self.config.gradient_accumulation_steps

            # Backward pass
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accumulation_loss += loss.item()

            # Gradient accumulation step
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if scaler:
                    scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.adapter.parameters(),
                    self.config.max_grad_norm,
                )

                # Optimizer step
                if scaler:
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    lr = self.scheduler.get_last_lr()[0]

                    # Calculate average loss from original losses
                    if original_losses:
                        avg_loss = sum(original_losses) / len(original_losses)
                    else:
                        avg_loss = 0.0

                    # Compute adapter variance metrics
                    with torch.no_grad():
                        sample_embeds = batch["embeddings"][:min(8, len(batch["embeddings"]))]
                        # Convert embeddings to adapter dtype to prevent mismatch
                        adapter_dtype = next(self.model.adapter.parameters()).dtype
                        sample_embeds = sample_embeds.to(adapter_dtype)
                        adapter_out = self.model.adapter(sample_embeds)

                        # Normalize and compute similarity
                        adapter_out_norm = F.normalize(adapter_out, dim=1)
                        similarity_matrix = torch.matmul(adapter_out_norm, adapter_out_norm.T)

                        # Average similarity (excluding diagonal)
                        mask = ~torch.eye(len(adapter_out), device=adapter_out.device).bool()
                        avg_similarity = similarity_matrix[mask].mean().item() if len(adapter_out) > 1 else 0.0

                        # Output statistics
                        output_std = adapter_out.std().item()
                        output_norm = adapter_out.norm(dim=1).mean().item()

                    pbar.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'sim': f'{avg_similarity:.3f}',   # Watch for collapse
                        'std': f'{output_std:.2f}',       # Should be > 1.0
                        'lr': f'{lr:.2e}',
                        'step': self.global_step,
                    })

                    if self.use_wandb:
                        self.wandb.log({
                            "train/loss": avg_loss,
                            "train/learning_rate": lr,
                            "train/epoch": self.epoch,
                            "adapter/similarity": avg_similarity,
                            "adapter/output_std": output_std,
                            "adapter/output_norm": output_norm,
                        }, step=self.global_step)

                    # Reset both tracking variables
                    accumulation_loss = 0.0
                    original_losses = []

                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    val_loss = self._evaluate(val_loader)
                    logger.info(
                        f"Step {self.global_step}: val_loss = {val_loss:.4f}, "
                        f"best_val_loss = {self.best_val_loss:.4f}"
                    )

                    if self.use_wandb:
                        self.wandb.log({
                            "val/loss": val_loss,
                        }, step=self.global_step)

                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        logger.info(f"New best model! Saving...")
                        self.checkpoint_manager.save_best(
                            adapter=self.model.adapter,
                            val_loss=val_loss,
                            global_step=self.global_step,
                            epoch=self.epoch,
                        )

                    self.model.train()

                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    logger.info(f"Saving checkpoint at step {self.global_step}")
                    self.checkpoint_manager.save(
                        adapter=self.model.adapter,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        global_step=self.global_step,
                        epoch=self.epoch,
                        best_val_loss=self.best_val_loss,
                    )

                # Max steps check
                if self.config.max_steps and self.global_step >= self.config.max_steps:
                    return

    @torch.no_grad()
    def _evaluate(self, val_loader: DataLoader) -> float:
        """Evaluate on validation set.

        Args:
            val_loader: Validation dataloader

        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(val_loader, desc="Evaluating", leave=False)
        for batch in pbar:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with torch.amp.autocast(
                device_type='cuda',
                dtype=torch.bfloat16,
                enabled=self.config.use_bf16,
            ):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    embeddings=batch["embeddings"],
                    embedding_positions=batch["embedding_positions"],
                    labels=batch["labels"],
                )

            total_loss += outputs["loss"].item()
            num_batches += 1

            pbar.set_postfix({'val_loss': f'{total_loss / num_batches:.4f}'})

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return avg_loss
