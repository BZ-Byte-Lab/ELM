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
        residual_scale: float = 0.1,
        dropout_rate: float = 0.1,
    ):
        """Initialize EnhancedAdapter.

        Args:
            embedding_dim: Dimension of input embeddings (2560 for Qwen3)
            hidden_dim: Intermediate hidden dimension (4096 default)
            use_residual: Whether to use residual connection
            residual_scale: Scale factor for residual connection (0.1 prevents dominance)
            dropout_rate: Dropout rate for regularization (0.1 standard)
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual
        self.residual_scale = residual_scale

        # Layers
        self.up_proj = nn.Linear(embedding_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout_rate)  # After activation
        self.down_proj = nn.Linear(hidden_dim, embedding_dim)
        self.dropout2 = nn.Dropout(dropout_rate)  # Before residual
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """He initialization amplified 3x to overcome residual dominance.

        He init is theoretically justified for GELU activation.
        3x amplification overcomes the 0.1x residual scaling, ensuring
        adapter contributes significantly from the start.
        """
        # He init for GELU: sqrt(2/fan_in), then amplify 3x
        std_up = (2.0 / self.embedding_dim) ** 0.5 * 3.0      # ≈ 0.084
        std_down = (2.0 / self.hidden_dim) ** 0.5 * 3.0       # ≈ 0.066

        nn.init.normal_(self.up_proj.weight, std=std_up)
        nn.init.zeros_(self.up_proj.bias)
        nn.init.normal_(self.down_proj.weight, std=std_down)
        nn.init.zeros_(self.down_proj.bias)
        # LayerNorm is already initialized to identity transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dropout and scaled residual.

        Args:
            x: Input embeddings of shape (batch_size, embedding_dim)

        Returns:
            Adapted embeddings of shape (batch_size, embedding_dim)
        """
        residual = x

        # Up projection + activation
        x = self.up_proj(x)
        x = self.activation(x)
        x = self.dropout1(x)  # Dropout after activation

        # Down projection
        x = self.down_proj(x)
        x = self.dropout2(x)  # Dropout before residual

        # Scaled residual connection
        if self.use_residual:
            x = x + self.residual_scale * residual  # 0.1x instead of 1.0x

        # Layer normalization
        x = self.layer_norm(x)

        return x

    def get_num_parameters(self) -> int:
        """Get total number of parameters in the adapter."""
        return sum(p.numel() for p in self.parameters())

    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters in the adapter."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
