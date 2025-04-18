#!/usr/bin/env python3
"""
Train Dimensional Cascade Model

This script trains a dimensional cascade model by configuring different dimension
reduction layers. It helps optimize the cascade for specific datasets and use cases.
"""
import os
import sys
import argparse
import numpy as np
import json
import time
from typing import List, Dict, Tuple, Optional, Union
import torch
from torch import nn
import logging

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from dimensional_cascade.utils.metrics import time_function
from dimensional_cascade.core import DimensionalCascade

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('dimensional_cascade.train')

try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence_transformers not found. Some features will be unavailable.")


class DimensionReductionNetwork(nn.Module):
    """Neural network for learning optimal dimension reduction transformations."""
    
    def __init__(self, 
                 input_dim: int, 
                 dimensions: List[int],
                 use_linear: bool = True,
                 use_nonlinearity: bool = True):
        """
        Initialize dimension reduction network.
        
        Args:
            input_dim: Input dimension
            dimensions: List of target dimensions to reduce to
            use_linear: Whether to use linear layers
            use_nonlinearity: Whether to use nonlinear activations
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.dimensions = dimensions
        
        # Create dimension reduction layers
        self.reduction_layers = nn.ModuleDict()
        
        # Sort dimensions in descending order
        sorted_dims = sorted(dimensions, reverse=True)
        
        # For each target dimension, create a reduction layer from input_dim
        for dim in sorted_dims:
            if dim >= input_dim:
                # Skip if dimension is greater than or equal to input dimension
                continue
                
            if use_linear:
                # Linear reduction layer
                layer = nn.Sequential()
                layer.add_module(f"linear_{dim}", nn.Linear(input_dim, dim))
                
                if use_nonlinearity:
                    layer.add_module(f"activation_{dim}", nn.Tanh())
            else:
                # Simple projection layer (no training)
                layer = nn.Sequential()
                layer.add_module(f"projection_{dim}", nn.Conv1d(
                    in_channels=1, 
                    out_channels=1, 
                    kernel_size=1, 
                    bias=False
                ))
                
                # Initialize with identity-like projection
                with torch.no_grad():
                    layer[0].weight.data.zero_()
                    for i in range(min(dim, input_dim)):
                        layer[0].weight.data[0, 0, i] = 1.0
            
            self.reduction_layers[str(dim)] = layer
    
    def forward(self, x: torch.Tensor, target_dim: int) -> torch.Tensor:
        """
        Forward pass to reduce dimensionality.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            target_dim: Target dimension to reduce to
            
        Returns:
            Reduced tensor of shape (batch_size, target_dim)
        """
        if target_dim >= self.input_dim:
            # If target dimension is greater than input, return input
            return x
            
        # Get reduction layer for target dimension
        layer = self.reduction_layers[str(target_dim)]
        
        # Apply dimension reduction
        return layer(x)
    
    def reduce_all(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Reduce input to all target dimensions.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Dictionary mapping dimensions to reduced tensors
        """
        results = {}
        
        for dim in self.dimensions:
            if dim >= self.input_dim:
                # If dimension is greater than input, just include the input
                results[dim] = x
            else:
                results[dim] = self.forward(x, dim)
        
        return results


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for training dimension reduction network.
    
    This loss ensures that similar items in the original space remain similar
    in the reduced space, while dissimilar items remain dissimilar.
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        """
        Initialize contrastive loss.
        
        Args:
            margin: Margin for negative pairs
            reduction: Reduction method ('mean', 'sum', or 'none')
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, 
                original_embeddings: torch.Tensor, 
                reduced_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute contrastive loss.
        
        Args:
            original_embeddings: Original embeddings (batch_size, original_dim)
            reduced_embeddings: Reduced embeddings (batch_size, reduced_dim)
            
        Returns:
            Contrastive loss
        """
        batch_size = original_embeddings.size(0)
        
        # Normalize embeddings to unit length
        original_norm = torch.norm(original_embeddings, dim=1, keepdim=True)
        reduced_norm = torch.norm(reduced_embeddings, dim=1, keepdim=True)
        
        original_embeddings = original_embeddings / (original_norm + 1e-8)
        reduced_embeddings = reduced_embeddings / (reduced_norm + 1e-8)
        
        # Compute similarity matrices
        original_sim = torch.matmul(original_embeddings, original_embeddings.t())
        reduced_sim = torch.matmul(reduced_embeddings, reduced_embeddings.t())
        
        # Compute loss - similarity preservation
        loss = torch.mean((original_sim - reduced_sim) ** 2)
        
        if self.reduction == 'sum':
            loss = loss * batch_size
        elif self.reduction == 'none':
            loss = (original_sim - reduced_sim) ** 2
            
        return loss


def parse_arguments():
    """Parse command line arguments for training."""
    parser = argparse.ArgumentParser(description='Train dimensional cascade model')
    
    parser.add_argument(
        '--dims', 
        type=int, 
        nargs='+',
        default=[768, 384, 192, 96, 48, 24],
        help='Dimensions to train for, from highest to lowest'
    )
    
    parser.add_argument(
        '--data-path', 
        type=str, 
        required=True,
        help='Path to numpy file with training embeddings'
    )
    
    parser.add_argument(
        '--validation-path', 
        type=str, 
        default=None,
        help='Path to numpy file with validation embeddings (optional)'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='trained_model',
        help='Directory to save trained model'
    )
    
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=50,
        help='Number of training epochs'
    )
    
    parser.add_argument(
        '--batch-size', 
        type=int, 
        default=64,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--learning-rate', 
        type=float, 
        default=0.001,
        help='Learning rate for optimizer'
    )
    
    parser.add_argument(
        '--use-nonlinearity', 
        action='store_true',
        help='Use nonlinear activations in reduction layers'
    )
    
    parser.add_argument(
        '--use-simple-projection', 
        action='store_true',
        help='Use simple projection instead of learned reduction'
    )
    
    parser.add_argument(
        '--load-checkpoint', 
        type=str, 
        default=None,
        help='Path to checkpoint to resume training'
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default=None,
        help='Device to use for training (cpu, cuda, cuda:0, etc.)'
    )
    
    return parser.parse_args()


def load_data(data_path: str, batch_size: int = 64) -> torch.utils.data.DataLoader:
    """
    Load embedding data for training.
    
    Args:
        data_path: Path to numpy file with embeddings
        batch_size: Batch size for dataloader
        
    Returns:
        DataLoader for training
    """
    logger.info(f"Loading data from {data_path}")
    
    # Load embeddings
    embeddings = np.load(data_path)
    
    # Convert to torch tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    
    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(embeddings_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    logger.info(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    
    return dataloader, embeddings.shape[1]


def train_model(args):
    """Train dimensional cascade model."""
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    
    # Load training data
    train_loader, input_dim = load_data(args.data_path, args.batch_size)
    
    # Load validation data if provided
    val_loader = None
    if args.validation_path:
        val_loader, _ = load_data(args.validation_path, args.batch_size)
    
    # Create model
    model = DimensionReductionNetwork(
        input_dim=input_dim,
        dimensions=args.dims,
        use_linear=not args.use_simple_projection,
        use_nonlinearity=args.use_nonlinearity
    )
    model.to(device)
    
    # Create loss function and optimizer
    criterion = ContrastiveLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.load_checkpoint and os.path.exists(args.load_checkpoint):
        logger.info(f"Loading checkpoint from {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Resuming from epoch {start_epoch}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config for future reference
    config = {
        'input_dim': input_dim,
        'dimensions': args.dims,
        'use_nonlinearity': args.use_nonlinearity,
        'use_simple_projection': args.use_simple_projection,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'device': str(device)
    }
    
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(start_epoch, args.epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        batch_count = 0
        
        for batch in train_loader:
            # Get embeddings
            embeddings = batch[0].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass for each target dimension
            total_loss = 0.0
            
            for dim in args.dims:
                if dim >= input_dim:
                    # Skip if dimension is greater than input
                    continue
                
                # Reduce embeddings to target dimension
                reduced_embeddings = model(embeddings, dim)
                
                # Compute loss
                loss = criterion(embeddings, reduced_embeddings)
                total_loss += loss
            
            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()
            
            # Update statistics
            epoch_loss += total_loss.item()
            batch_count += 1
        
        # Compute average loss
        avg_train_loss = epoch_loss / batch_count
        train_losses.append(avg_train_loss)
        
        # Validation
        val_loss = None
        if val_loader:
            model.eval()
            val_epoch_loss = 0.0
            val_batch_count = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    # Get embeddings
                    embeddings = batch[0].to(device)
                    
                    # Forward pass for each target dimension
                    total_loss = 0.0
                    
                    for dim in args.dims:
                        if dim >= input_dim:
                            # Skip if dimension is greater than input
                            continue
                        
                        # Reduce embeddings to target dimension
                        reduced_embeddings = model(embeddings, dim)
                        
                        # Compute loss
                        loss = criterion(embeddings, reduced_embeddings)
                        total_loss += loss
                    
                    # Update statistics
                    val_epoch_loss += total_loss.item()
                    val_batch_count += 1
            
            # Compute average loss
            avg_val_loss = val_epoch_loss / val_batch_count
            val_losses.append(avg_val_loss)
            val_loss = avg_val_loss
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                }, os.path.join(args.output_dir, 'best_model.pt'))
                logger.info(f"Saved best model with validation loss {best_val_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_train_loss,
            }, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt'))
        
        # Log progress
        if val_loss:
            logger.info(f"Epoch {epoch+1}/{args.epochs}, "
                      f"Train Loss: {avg_train_loss:.6f}, "
                      f"Val Loss: {val_loss:.6f}")
        else:
            logger.info(f"Epoch {epoch+1}/{args.epochs}, "
                      f"Train Loss: {avg_train_loss:.6f}")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': train_losses[-1],
    }, os.path.join(args.output_dir, 'final_model.pt'))
    
    # Save loss history
    loss_history = {
        'train_loss': train_losses,
        'val_loss': val_losses if val_loader else []
    }
    
    with open(os.path.join(args.output_dir, 'loss_history.json'), 'w') as f:
        json.dump(loss_history, f, indent=2)
    
    logger.info("Training completed successfully")
    
    return model


def export_model(model, output_dir: str, input_dim: int, dimensions: List[int]):
    """
    Export trained model as numpy matrices for use in dimensional cascade.
    
    Args:
        model: Trained DimensionReductionNetwork
        output_dir: Directory to save exported matrices
        input_dim: Input dimension
        dimensions: Target dimensions
    """
    logger.info("Exporting model as numpy matrices")
    
    # Create matrices directory
    matrices_dir = os.path.join(output_dir, 'matrices')
    os.makedirs(matrices_dir, exist_ok=True)
    
    # Export matrices for each dimension
    for dim in dimensions:
        if dim >= input_dim:
            # Skip if dimension is greater than input
            continue
        
        # Create random input
        x = torch.randn(1, input_dim)
        
        # Get layer
        layer = model.reduction_layers[str(dim)]
        
        # Extract weights
        if isinstance(layer[0], nn.Linear):
            # Linear layer
            weights = layer[0].weight.detach().cpu().numpy()
            
            # Save weights
            np.save(os.path.join(matrices_dir, f'dim_{dim}_weights.npy'), weights)
            
            # If there's bias
            if layer[0].bias is not None:
                bias = layer[0].weight.detach().cpu().numpy()
                np.save(os.path.join(matrices_dir, f'dim_{dim}_bias.npy'), bias)
        else:
            # Projection layer
            weights = layer[0].weight.detach().cpu().numpy().squeeze()
            np.save(os.path.join(matrices_dir, f'dim_{dim}_weights.npy'), weights)
    
    logger.info(f"Model exported to {matrices_dir}")


def main():
    """Main function for training dimensional cascade model."""
    args = parse_arguments()
    
    # Train model
    model = train_model(args)
    
    # Export model
    dataloader, input_dim = load_data(args.data_path, args.batch_size)
    export_model(model, args.output_dir, input_dim, args.dims)
    
    logger.info("Training and export completed.")


if __name__ == "__main__":
    main() 