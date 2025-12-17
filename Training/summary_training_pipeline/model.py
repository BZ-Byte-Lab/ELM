"""ELM Model combining frozen LLM with trainable adapter."""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, Dict
import logging

from .adapter import EnhancedAdapter
from .config import SummaryTrainingConfig

logger = logging.getLogger(__name__)


class ELMModel(nn.Module):
    """Embedding Language Model with frozen LLM and trainable adapter.

    Architecture:
        - E_0: Frozen token embeddings from Qwen3-4B-Instruct
        - E_A: Trainable MLP adapter for embedding projection
        - M_0: Frozen transformer layers from Qwen3-4B-Instruct

    The model processes sequences with <EMB> placeholders, which are
    replaced by adapter-transformed external embeddings during forward pass.
    """

    def __init__(self, config: SummaryTrainingConfig):
        """Initialize ELM Model.

        Args:
            config: Training configuration
        """
        super().__init__()
        self.config = config

        # Load tokenizer with special <EMB> token
        logger.info(f"Loading tokenizer from {config.llm_model_name}")
        self.tokenizer = self._load_tokenizer()

        # Load frozen LLM
        logger.info(f"Loading LLM from {config.llm_model_name}")
        self.llm = self._load_llm()

        # Create trainable adapter
        logger.info("Creating trainable adapter")
        self.adapter = EnhancedAdapter(
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            use_residual=config.use_residual,
            residual_scale=config.residual_scale,
            dropout_rate=config.dropout_rate,
        )

        # Ensure adapter has same dtype as LLM
        adapter_dtype = torch.bfloat16 if config.use_bf16 else torch.float32
        self.adapter = self.adapter.to(adapter_dtype)

        # Store embedding layer reference
        self.token_embedding = self.llm.model.embed_tokens

        # Get <EMB> token ID
        self.emb_token_id = self.tokenizer.convert_tokens_to_ids(config.emb_token)
        logger.info(f"<EMB> token ID: {self.emb_token_id}")

        # Log parameter counts
        self._log_parameters()

    def _load_tokenizer(self) -> AutoTokenizer:
        """Load and configure tokenizer with special <EMB> token."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.llm_model_name,
            trust_remote_code=True,
        )

        # Add special <EMB> token if not present
        if self.config.emb_token not in tokenizer.get_vocab():
            num_added = tokenizer.add_special_tokens({
                'additional_special_tokens': [self.config.emb_token]
            })
            logger.info(f"Added {num_added} special token(s): {self.config.emb_token}")

        # Ensure pad token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")

        return tokenizer

    def _load_llm(self) -> AutoModelForCausalLM:
        """Load frozen LLM with optimizations."""
        # Determine dtype
        dtype = torch.bfloat16 if self.config.use_bf16 else torch.float32

        # Determine attention implementation
        # Use "eager" for better compatibility with gradient checkpointing
        attn_impl = "eager"

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.llm_model_name,
            dtype=dtype,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )

        # Resize token embeddings to accommodate <EMB> token
        original_vocab_size = model.config.vocab_size
        model.resize_token_embeddings(len(self.tokenizer))
        new_vocab_size = model.config.vocab_size
        if new_vocab_size != original_vocab_size:
            logger.info(f"Resized embeddings: {original_vocab_size} -> {new_vocab_size}")
            # Ensure new embeddings have correct dtype
            if model.get_input_embeddings().weight.dtype != dtype:
                model.get_input_embeddings().weight.data = model.get_input_embeddings().weight.data.to(dtype)

        # Freeze ALL LLM parameters
        for name, param in model.named_parameters():
            param.requires_grad = False
        logger.info("Froze all LLM parameters")

        # Enable gradient checkpointing for memory efficiency
        if self.config.use_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            # Disable use_cache (incompatible with gradient checkpointing)
            model.config.use_cache = False
            logger.info("Enabled gradient checkpointing and disabled use_cache")

        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        embeddings: torch.Tensor,
        embedding_positions: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with embedding injection.

        Args:
            input_ids: Token IDs with <EMB> placeholders, shape (batch, seq_len)
            attention_mask: Attention mask, shape (batch, seq_len)
            embeddings: External embeddings to inject, shape (batch, embedding_dim)
            embedding_positions: Positions of <EMB> tokens, shape (batch,)
            labels: Target token IDs for loss computation, shape (batch, seq_len)
            output_hidden_states: Whether to return hidden states for drift loss

        Returns:
            Dictionary with 'loss', 'logits', and optionally 'hidden_states'
        """
        batch_size, seq_len = input_ids.shape

        # Step 1: Get token embeddings from frozen E_0
        token_embeds = self.token_embedding(input_ids)  # (batch, seq_len, hidden_dim)

        # Step 2: Transform external embeddings through trainable E_A
        # Convert embeddings to adapter dtype to prevent mismatch
        adapter_dtype = next(self.adapter.parameters()).dtype
        embeddings = embeddings.to(adapter_dtype)
        adapted_embeds = self.adapter(embeddings)  # (batch, embedding_dim)

        # Ensure adapted embeddings match token embeddings dtype
        if adapted_embeds.dtype != token_embeds.dtype:
            adapted_embeds = adapted_embeds.to(token_embeds.dtype)

        # Step 3: Inject adapted embeddings at <EMB> positions
        # Note: Qwen3-4B-Instruct hidden_size=2560, same as embedding_dim
        for i in range(batch_size):
            pos = embedding_positions[i].item()
            if 0 <= pos < seq_len:
                token_embeds[i, pos] = adapted_embeds[i]

        # Step 4: Forward through frozen transformer M_0
        outputs = self.llm(
            inputs_embeds=token_embeds,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        result = {
            'loss': outputs.loss if labels is not None else None,
            'logits': outputs.logits,
        }

        if output_hidden_states:
            result['hidden_states'] = outputs.hidden_states[-1]  # Only last layer

        return result

    def get_adapter_outputs(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Get adapter outputs for contrastive loss calculation.

        Args:
            embeddings: Input embeddings (batch_size, embedding_dim)

        Returns:
            Adapter outputs (batch_size, embedding_dim)
        """
        # Convert embeddings to adapter dtype to prevent mismatch
        adapter_dtype = next(self.adapter.parameters()).dtype
        embeddings = embeddings.to(adapter_dtype)
        return self.adapter(embeddings)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        embeddings: torch.Tensor,
        embedding_positions: torch.Tensor,
        max_new_tokens: int = 100,
        **generate_kwargs,
    ) -> torch.Tensor:
        """Generate text given embeddings.

        Args:
            input_ids: Token IDs with <EMB> placeholders
            attention_mask: Attention mask
            embeddings: External embeddings to inject
            embedding_positions: Positions of <EMB> tokens
            max_new_tokens: Maximum number of tokens to generate
            **generate_kwargs: Additional arguments for generation

        Returns:
            Generated token IDs
        """
        batch_size, seq_len = input_ids.shape

        # Get and modify embeddings
        token_embeds = self.token_embedding(input_ids)
        # Convert embeddings to adapter dtype to prevent mismatch
        adapter_dtype = next(self.adapter.parameters()).dtype
        embeddings = embeddings.to(adapter_dtype)
        adapted_embeds = self.adapter(embeddings)

        for i in range(batch_size):
            pos = embedding_positions[i].item()
            if 0 <= pos < seq_len:
                token_embeds[i, pos] = adapted_embeds[i]

        # Generate
        return self.llm.generate(
            inputs_embeds=token_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            **generate_kwargs,
        )

    def get_trainable_parameters(self) -> int:
        """Count trainable parameters (adapter only)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_total_parameters(self) -> int:
        """Count all parameters."""
        return sum(p.numel() for p in self.parameters())

    def _log_parameters(self):
        """Log parameter counts."""
        trainable = self.get_trainable_parameters()
        total = self.get_total_parameters()
        trainable_pct = 100 * trainable / total

        logger.info(f"Trainable parameters: {trainable:,} ({trainable / 1e6:.2f}M)")
        logger.info(f"Total parameters: {total:,} ({total / 1e9:.2f}B)")
        logger.info(f"Trainable percentage: {trainable_pct:.4f}%")
