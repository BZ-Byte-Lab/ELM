"""MLP Adapter for ELM."""

import torch
import torch.nn as nn


class EnhancedAdapter(nn.Module):
    """Enhanced MLP Adapter with bottleneck expansion and residual connection.

    Architecture:
        h = GELU(W1 @ w + b1)           # Up projection to hidden_dim
        z = W2 @ h + b2                  # Down projection to embedding_dim
        output = LayerNorm(z + w)        # Residual + LayerNorm

    Parameters:
        - Up projection: embedding_dim * hidden_dim + hidden_dim
        - Down projection: hidden_dim * embedding_dim + embedding_dim
        - LayerNorm: embedding_dim * 2 (weight + bias)
        Total: ~21M params for embedding_dim=2560, hidden_dim=4096
    """

    def __init__(
        self,
        embedding_dim: int = 2560,
        hidden_dim: int = 4096,
        use_residual: bool = True,
    ):
        """Initialize EnhancedAdapter.

        Args:
            embedding_dim: Dimension of input embeddings (2560 for Qwen3)
            hidden_dim: Intermediate hidden dimension (4096 default)
            use_residual: Whether to use residual connection
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual

        # Layers
        self.up_proj = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.GELU()
        self.down_proj = nn.Linear(hidden_dim, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for near-identity behavior at start.

        Uses small std (0.02) so that adapter output is initially ~0,
        allowing the residual connection to dominate early in training.
        This provides stable gradient flow during warmup.
        """
        std = 0.02
        nn.init.normal_(self.up_proj.weight, std=std)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.normal_(self.down_proj.weight, std=std)
        nn.init.zeros_(self.down_proj.bias)
        # LayerNorm is already initialized to identity transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through adapter.

        Args:
            x: Input embeddings of shape (batch_size, embedding_dim)

        Returns:
            Adapted embeddings of shape (batch_size, embedding_dim)
        """
        residual = x

        # Up projection + activation
        x = self.up_proj(x)
        x = self.activation(x)

        # Down projection
        x = self.down_proj(x)

        # Residual connection
        if self.use_residual:
            x = x + residual

        # Layer normalization
        x = self.layer_norm(x)

        return x

    def get_num_parameters(self) -> int:
        """Get total number of parameters in the adapter."""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters in the adapter."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
