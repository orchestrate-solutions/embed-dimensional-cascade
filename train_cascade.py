#!/usr/bin/env python
"""
Train dimensional cascade models from pre-generated embeddings.

This script loads embeddings from a file and trains neural network models
to distill them to lower dimensions while preserving similarity relationships.
"""

import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

class DimensionReducer(nn.Module):
    """Neural network for dimensionality reduction."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None
    ):
        """
        Initialize the dimension reducer.
        
        Args:
            input_dim: Input dimension
            output_dim: Output dimension
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            # Create a reasonable architecture based on dimensions
            hidden_dims = []
            current_dim = input_dim
            
            # Add hidden layers that gradually reduce dimensions
            while current_dim > output_dim * 2:
                next_dim = max(output_dim, current_dim // 2)
                hidden_dims.append(next_dim)
                current_dim = next_dim
        
        # Create layers
        layers = []
        current_dim = input_dim
        
        # Add hidden layers
        for dim in hidden_dims:
            layers.append(nn.Linear(current_dim, dim))
            layers.append(nn.ReLU())
            current_dim = dim
        
        # Add output layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        # Create sequential model
        self.model = nn.Sequential(*layers)
        
        # Store dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor = None) -> torch.Tensor:
    """
    Compute cosine similarity matrix between two sets of vectors.
    
    Args:
        a: First set of vectors
        b: Second set of vectors (if None, use a)
        
    Returns:
        Cosine similarity matrix
    """
    if b is None:
        b = a
    
    # Normalize vectors
    a_norm = nn.functional.normalize(a, p=2, dim=1)
    b_norm = nn.functional.normalize(b, p=2, dim=1)
    
    # Compute similarity matrix
    return torch.mm(a_norm, b_norm.t())

def similarity_preservation_loss(
    original: torch.Tensor,
    reduced: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute similarity preservation loss.
    
    Args:
        original: Original embeddings
        reduced: Reduced embeddings
        normalize: Whether to normalize the embeddings
        
    Returns:
        Loss value
    """
    # Compute similarity matrices
    if normalize:
        original = nn.functional.normalize(original, p=2, dim=1)
        reduced = nn.functional.normalize(reduced, p=2, dim=1)
    
    # Compute similarity matrices
    sim_original = torch.mm(original, original.t())
    sim_reduced = torch.mm(reduced, reduced.t())
    
    # Compute mean squared error
    loss = nn.functional.mse_loss(sim_reduced, sim_original)
    
    return loss

def train_dimension_reducer(
    embeddings: np.ndarray,
    target_dim: int,
    batch_size: int = 64,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    validation_split: float = 0.1,
    device: Optional[str] = None,
    hidden_dims: Optional[List[int]] = None
) -> Tuple[DimensionReducer, Dict]:
    """
    Train a dimension reducer model.
    
    Args:
        embeddings: Input embeddings
        target_dim: Target dimension
        batch_size: Batch size for training
        epochs: Number of epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        validation_split: Validation split ratio
        device: Device to use for training
        hidden_dims: List of hidden layer dimensions
        
    Returns:
        Trained model and training metrics
    """
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    input_dim = embeddings.shape[1]
    model = DimensionReducer(input_dim, target_dim, hidden_dims)
    model.to(device)
    
    # Create dataset
    dataset = TensorDataset(torch.tensor(embeddings, dtype=torch.float32))
    
    # Split dataset
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Initialize metrics
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'best_val_loss': float('inf'),
        'best_epoch': 0
    }
    
    # Train model
    for epoch in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            # Get batch
            x = batch[0].to(device)
            
            # Forward pass
            reduced = model(x)
            
            # Compute loss
            loss = similarity_preservation_loss(x, reduced)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
        
        # Calculate average loss
        train_loss /= len(train_loader)
        metrics['train_losses'].append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                # Get batch
                x = batch[0].to(device)
                
                # Forward pass
                reduced = model(x)
                
                # Compute loss
                loss = similarity_preservation_loss(x, reduced)
                
                # Update metrics
                val_loss += loss.item()
        
        # Calculate average loss
        val_loss /= len(val_loader)
        metrics['val_losses'].append(val_loss)
        
        # Check for improvement
        if val_loss < metrics['best_val_loss']:
            metrics['best_val_loss'] = val_loss
            metrics['best_epoch'] = epoch
        
        # Log progress
        logger.info(f"Epoch {epoch}/{epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
    
    return model, metrics

def reduce_embeddings(
    model: DimensionReducer,
    embeddings: np.ndarray,
    batch_size: int = 64,
    device: Optional[str] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Reduce embeddings using a trained model.
    
    Args:
        model: Trained model
        embeddings: Input embeddings
        batch_size: Batch size for inference
        device: Device to use for inference
        normalize: Whether to normalize the output embeddings
        
    Returns:
        Reduced embeddings
    """
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set model to evaluation mode
    model.to(device)
    model.eval()
    
    # Create dataset and loader
    dataset = TensorDataset(torch.tensor(embeddings, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size)
    
    # Reduce embeddings
    reduced_embeddings = []
    
    with torch.no_grad():
        for batch in loader:
            # Get batch
            x = batch[0].to(device)
            
            # Forward pass
            reduced = model(x)
            
            # Normalize if requested
            if normalize:
                reduced = nn.functional.normalize(reduced, p=2, dim=1)
            
            # Add to list
            reduced_embeddings.append(reduced.cpu().numpy())
    
    # Concatenate embeddings
    return np.vstack(reduced_embeddings)

def generate_synthetic_data(n_samples: int = 10000, n_dim: int = 768) -> np.ndarray:
    """
    Generate synthetic data for testing.
    
    Args:
        n_samples: Number of samples to generate
        n_dim: Dimension of embeddings
        
    Returns:
        Synthetic embeddings
    """
    # Generate random embeddings
    embeddings = np.random.randn(n_samples, n_dim).astype(np.float32)
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    
    return embeddings

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train dimension cascade models")
    
    parser.add_argument("--embeddings", type=str, help="Path to embeddings file (.npy)")
    parser.add_argument("--output-dir", type=str, default="cascade_models", help="Output directory for models and reduced embeddings")
    parser.add_argument("--dimensions", type=str, default="512,256,128,64,32", help="Comma-separated list of target dimensions")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training and inference")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, cpu)")
    parser.add_argument("--save-embeddings", action="store_true", help="Save reduced embeddings")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data for testing")
    parser.add_argument("--n-synthetic", type=int, default=10000, help="Number of synthetic samples to generate")
    parser.add_argument("--input-dim", type=int, default=768, help="Input dimension for synthetic data")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Load or generate embeddings
    if args.synthetic or not args.embeddings:
        logger.info(f"Generating {args.n_synthetic} synthetic embeddings with dimension {args.input_dim}")
        embeddings = generate_synthetic_data(args.n_synthetic, args.input_dim)
    else:
        logger.info(f"Loading embeddings from {args.embeddings}")
        embeddings = np.load(args.embeddings)
    
    input_dim = embeddings.shape[1]
    logger.info(f"Working with {len(embeddings)} embeddings of dimension {input_dim}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse target dimensions
    target_dims = [int(dim) for dim in args.dimensions.split(',')]
    target_dims = sorted(target_dims, reverse=False)  # Sort in ascending order
    logger.info(f"Target dimensions: {target_dims}")
    
    # Train models for each target dimension
    for target_dim in target_dims:
        logger.info(f"Training model for dimension {target_dim}")
        
        # Create model directory
        model_dir = output_dir / f"dim_{target_dim}"
        model_dir.mkdir(exist_ok=True)
        
        # Train model
        model, metrics = train_dimension_reducer(
            embeddings=embeddings,
            target_dim=target_dim,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device
        )
        
        # Save model
        model_path = model_dir / "model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': input_dim,
            'output_dim': target_dim,
            'hidden_dims': model.hidden_dims,
            'metrics': metrics
        }, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save reduced embeddings if requested
        if args.save_embeddings:
            logger.info(f"Reducing embeddings to dimension {target_dim}")
            reduced = reduce_embeddings(
                model=model,
                embeddings=embeddings,
                batch_size=args.batch_size,
                device=args.device
            )
            
            # Save reduced embeddings
            embeddings_path = model_dir / "embeddings.npy"
            np.save(embeddings_path, reduced)
            logger.info(f"Reduced embeddings saved to {embeddings_path}")
    
    logger.info("Done!")

if __name__ == "__main__":
    main() 