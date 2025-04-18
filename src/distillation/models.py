"""
Models for dimensional cascade distillation.

This module provides models for distilling high-dimensional embeddings
to lower dimensions while preserving similarity relationships.
"""

import torch
from torch import nn
import logging
from typing import Dict, List, Optional, Tuple, Any, Union

logger = logging.getLogger(__name__)

class DimensionDistiller(nn.Module):
    """
    Neural network model that distills high-dimensional embeddings to lower dimensions.
    
    This model is trained to preserve similarity relationships between vectors
    when reducing their dimensionality.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: nn.Module = nn.SiLU(),
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_residual: bool = True
    ):
        """
        Initialize the dimension distiller model.
        
        Args:
            input_dim: Dimension of input embeddings
            output_dim: Dimension of output embeddings
            hidden_dims: Dimensions of hidden layers (default: auto-configured)
            activation: Activation function to use
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections where possible
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Auto-configure hidden dimensions if not provided
        if hidden_dims is None:
            # Create a bottleneck architecture
            if input_dim > output_dim:
                # Encoder gradually reduces dimension
                hidden1 = max(output_dim * 4, min(input_dim // 2, 1024))
                hidden2 = max(output_dim * 2, min(hidden1 // 2, 512))
                hidden_dims = [hidden1, hidden2]
            else:
                # Encoder might need to expand first
                hidden1 = max(input_dim * 2, min(output_dim // 2, 1024))
                hidden2 = max(output_dim, min(hidden1 * 2, 2048))
                hidden_dims = [hidden1, hidden2]
        
        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        
        # Build the network
        layers = []
        prev_dim = input_dim
        
        # Hidden layers
        for i, dim in enumerate(hidden_dims):
            # Linear layer
            layers.append(nn.Linear(prev_dim, dim))
            
            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            
            # Activation
            layers.append(activation)
            
            # Dropout for regularization
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"Created DimensionDistiller: {input_dim} → {hidden_dims} → {output_dim}")
    
    def _init_weights(self):
        """Initialize the weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the distiller.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Distilled tensor of shape [batch_size, output_dim]
        """
        # Process through model
        distilled = self.model(x)
        
        # Normalize output vectors
        distilled = nn.functional.normalize(distilled, p=2, dim=1)
        
        return distilled
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration for serialization.
        
        Returns:
            Dictionary with model configuration
        """
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "hidden_dims": self.hidden_dims,
            "use_batch_norm": self.use_batch_norm,
            "use_residual": self.use_residual
        }


class CascadeDistiller(nn.Module):
    """
    Manages multiple dimension distillers for a dimensional cascade.
    
    This model allows distillation from higher to multiple lower dimensions,
    supporting the dimensional cascade search approach.
    """
    
    def __init__(self):
        """Initialize the cascade distiller."""
        super().__init__()
        self.distillers = nn.ModuleDict()  # Source dim -> Distiller
        self.dimension_map = {}  # Maps target_dim -> source_dim
    
    def add_distiller(self, 
                     source_dim: int, 
                     target_dims: List[int],
                     **distiller_kwargs) -> None:
        """
        Add a distiller for a source dimension to multiple target dimensions.
        
        Args:
            source_dim: Source dimension
            target_dims: List of target dimensions (must be <= source_dim)
            **distiller_kwargs: Additional arguments for DimensionDistiller
        """
        # Create a distiller for each target dimension
        for target_dim in sorted(target_dims, reverse=True):
            if target_dim > source_dim:
                logger.warning(f"Target dimension {target_dim} is larger than "
                              f"source dimension {source_dim}. Skipping.")
                continue
            
            # Create the distiller
            distiller = DimensionDistiller(
                input_dim=source_dim,
                output_dim=target_dim,
                **distiller_kwargs
            )
            
            # Add to the module dict by source dimension
            source_key = str(source_dim)
            if source_key not in self.distillers:
                self.distillers[source_key] = nn.ModuleList()
            
            self.distillers[source_key].append(distiller)
            
            # Update dimension map
            self.dimension_map[target_dim] = source_dim
            
            logger.info(f"Added distiller: {source_dim} → {target_dim}")
    
    def distill(self, 
               x: torch.Tensor, 
               target_dim: int) -> Optional[torch.Tensor]:
        """
        Distill embeddings to the target dimension.
        
        Args:
            x: Input tensor of shape [batch_size, dim]
            target_dim: Target dimension to distill to
            
        Returns:
            Distilled tensor of shape [batch_size, target_dim] or None if invalid
        """
        # Check if the input dimension matches any source dimension
        input_dim = x.shape[1]
        
        # If target dimension matches input, return normalized input
        if target_dim == input_dim:
            return nn.functional.normalize(x, p=2, dim=1)
        
        # Check if we have a distiller for this input dimension
        source_key = str(input_dim)
        if source_key not in self.distillers:
            logger.warning(f"No distiller found for input dimension {input_dim}")
            return None
        
        # Find the distiller for the target dimension
        distillers = self.distillers[source_key]
        target_distiller = None
        
        for distiller in distillers:
            if distiller.output_dim == target_dim:
                target_distiller = distiller
                break
        
        if target_distiller is None:
            logger.warning(f"No distiller found for target dimension {target_dim} "
                          f"from source dimension {input_dim}")
            return None
        
        # Distill to the target dimension
        return target_distiller(x)
    
    def forward(self, x: torch.Tensor, target_dim: int) -> torch.Tensor:
        """
        Forward pass through the cascade distiller.
        
        Args:
            x: Input tensor of shape [batch_size, dim]
            target_dim: Target dimension to distill to
            
        Returns:
            Distilled tensor of shape [batch_size, target_dim]
        """
        distilled = self.distill(x, target_dim)
        if distilled is None:
            raise ValueError(f"Cannot distill from dimension {x.shape[1]} to {target_dim}")
        return distilled
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration for serialization.
        
        Returns:
            Dictionary with model configuration
        """
        config = {
            "dimension_map": self.dimension_map,
            "distillers": {}
        }
        
        # Add configuration for each distiller
        for source_dim, distillers in self.distillers.items():
            config["distillers"][source_dim] = []
            for distiller in distillers:
                config["distillers"][source_dim].append(distiller.get_config())
        
        return config
    
    @property
    def supported_dimensions(self) -> List[int]:
        """Get list of supported dimensions for distillation."""
        dimensions = set()
        
        # Add all source dimensions
        for source_dim in self.distillers.keys():
            dimensions.add(int(source_dim))
            
        # Add all target dimensions
        for target_dim in self.dimension_map.keys():
            dimensions.add(target_dim)
            
        return sorted(list(dimensions))
    
    @property
    def target_dimensions(self) -> List[int]:
        """Get list of target dimensions for distillation."""
        return sorted(list(self.dimension_map.keys()))


if __name__ == "__main__":
    # Example usage
    import numpy as np
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a simple distiller
    input_dim = 768
    output_dim = 128
    
    # Example with single distiller
    distiller = DimensionDistiller(input_dim=input_dim, output_dim=output_dim)
    
    # Test with random data
    x = torch.randn(10, input_dim)
    y = distiller(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Example with cascade distiller
    cascade = CascadeDistiller()
    target_dims = [512, 256, 128, 64, 32]
    
    # Add distillers to the cascade
    cascade.add_distiller(768, target_dims)
    
    # Test cascade distiller
    for target_dim in target_dims:
        distilled = cascade.distill(x, target_dim)
        print(f"Distilled to dimension {target_dim}: {distilled.shape}") 