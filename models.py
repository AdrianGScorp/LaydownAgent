"""
Neural network models for DQN-based warehouse optimization.

This module implements transformer-based neural networks for Q-value estimation.
"""
import torch
import torch.nn as nn


class AttentionQNetwork(nn.Module):
    """Transformer-based Q-value estimator for DQN.
    
    This network uses a transformer architecture to capture relationships
    between items and estimate their Q-values.
    
    Architecture:
        1. Embedding layer: Maps raw features to hidden_dim.
        2. Transformer encoder: Captures item relationships via multi-head attention.
        3. Q-head: Outputs Q-values for each item.
    
    Attributes:
        embedding (nn.Linear): Input embedding layer.
        transformer (nn.TransformerEncoder): Transformer encoder blocks.
        q_head (nn.Linear): Output layer producing Q-values.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = 64, 
                 n_heads: int = 4, n_layers: int = 2):
        """Initialize the Q-network.
        
        Args:
            feature_dim: Dimension of input features per item.
            hidden_dim: Size of transformer hidden states.
            n_heads: Number of attention heads in transformer.
            n_layers: Number of transformer encoder layers.
        """
        super().__init__()
        self.embedding = nn.Linear(feature_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.q_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, num_items, feature_dim).
            
        Returns:
            torch.Tensor: Q-values for each item, shape (batch_size, num_items).
        """
        x = self.embedding(x)
        x = self.transformer(x)
        return self.q_head(x).squeeze(-1)
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters in the network.
        
        Returns:
            int: Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
