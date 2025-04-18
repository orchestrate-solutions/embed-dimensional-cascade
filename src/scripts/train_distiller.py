#!/usr/bin/env python
"""
Dimensional Cascade Distiller Training Script

This script trains a dimension distiller model to compress high-dimensional
embeddings into lower dimensions while preserving similarity relationships.
"""

import os
import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Optional

# Import modules from our package
from src.distillation.models import DimensionDistiller, CascadeDistiller
from src.training.trainer import Trainer, create_dataloaders, create_default_callbacks
from src.utils.data_loader import load_vectors_from_file, reduce_dimensions, generate_synthetic_data

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def parse_dimensions(dimensions_str: str) -> List[int]:
    """Parse comma-separated dimensions string into a list of integers."""
    return sorted([int(d.strip()) for d in dimensions_str.split(',')])

def create_similarity_dataset(
    vectors: np.ndarray,
    target_dim: int,
    n_pairs: int = 10000,
    batch_size: int = 32,
    val_split: float = 0.1,
    random_state: int = 42,
) -> tuple:
    """
    Create a dataset of pairs for similarity-based training.
    
    Args:
        vectors: Input vectors for training
        target_dim: Target dimension for reduction
        n_pairs: Number of pairs to generate
        batch_size: Batch size for training
        val_split: Validation split ratio
        random_state: Random seed
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    logger.info(f"Creating similarity dataset with {n_pairs} pairs for dimension {target_dim}")
    
    # Set random seed
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    
    # Convert to torch tensor
    vectors_tensor = torch.tensor(vectors, dtype=torch.float32)
    
    # Compute PCA for the target dimension for target vectors
    from sklearn.decomposition import PCA
    pca = PCA(n_components=target_dim)
    targets = pca.fit_transform(vectors)
    targets_tensor = torch.tensor(targets, dtype=torch.float32)
    
    # Generate random pairs
    n_vectors = len(vectors)
    indices = np.random.choice(n_vectors, size=(n_pairs, 2), replace=True)
    
    # Create input pairs and target similarities
    X_pairs = []
    y_similarities = []
    
    for i, j in indices:
        # Input is the first vector
        X_pairs.append(vectors_tensor[i])
        
        # Target is the similarity between reduced vectors
        sim = torch.cosine_similarity(
            targets_tensor[i].unsqueeze(0), 
            targets_tensor[j].unsqueeze(0), 
            dim=1
        ).item()
        y_similarities.append(sim)
    
    # Convert to tensors
    X = torch.stack(X_pairs)
    y = torch.tensor(y_similarities, dtype=torch.float32).unsqueeze(1)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        X, y, 
        batch_size=batch_size, 
        val_split=val_split,
        shuffle=True
    )
    
    return train_loader, val_loader

def train_dimension_distiller(
    vectors: np.ndarray,
    target_dim: int,
    hidden_dims: Optional[List[int]] = None,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    n_pairs: int = 10000,
    output_dir: str = "output",
    device: Optional[torch.device] = None,
) -> DimensionDistiller:
    """
    Train a DimensionDistiller model.
    
    Args:
        vectors: Input vectors for training
        target_dim: Target dimension for reduction
        hidden_dims: Hidden layer dimensions (if None, will be automatically determined)
        batch_size: Batch size for training
        epochs: Number of epochs for training
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        n_pairs: Number of pairs to generate for training
        output_dir: Output directory for model and logs
        device: Device to train on (CPU or GPU)
        
    Returns:
        Trained DimensionDistiller model
    """
    input_dim = vectors.shape[1]
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, f"checkpoints_d{target_dim}")
    log_dir = os.path.join(output_dir, f"logs_d{target_dim}")
    
    logger.info(f"Training dimension distiller: {input_dim} -> {target_dim}")
    
    # Create model
    model = DimensionDistiller(
        input_dim=input_dim,
        output_dim=target_dim,
        hidden_dims=hidden_dims,
    )
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=weight_decay
    )
    
    # Using MSE loss for similarity prediction
    loss_fn = nn.MSELoss()
    
    # Create dataloaders
    train_loader, val_loader = create_similarity_dataset(
        vectors=vectors,
        target_dim=target_dim,
        n_pairs=n_pairs,
        batch_size=batch_size,
    )
    
    # Create callbacks
    callbacks = create_default_callbacks(
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        monitor='val_loss',
        patience=10,
        save_best_only=True,
        tensorboard=True,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        callbacks=callbacks,
    )
    
    # Train the model
    history = trainer.fit(
        train_loader=train_loader,
        epochs=epochs,
        val_loader=val_loader,
        verbose=True,
    )
    
    # Save the final model
    model_path = os.path.join(output_dir, f"distiller_d{target_dim}.pt")
    trainer.save_model(model_path, save_optimizer=True)
    logger.info(f"Model saved to {model_path}")
    
    return model

def train_cascade_distiller(
    vectors: np.ndarray,
    dimensions: List[int],
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-5,
    n_pairs: int = 10000,
    output_dir: str = "output",
    strategy: str = "direct",
    device: Optional[torch.device] = None,
) -> CascadeDistiller:
    """
    Train a CascadeDistiller model with multiple dimensions.
    
    Args:
        vectors: Input vectors for training
        dimensions: List of target dimensions for the cascade
        batch_size: Batch size for training
        epochs: Number of epochs for training
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        n_pairs: Number of pairs to generate for training
        output_dir: Output directory for model and logs
        strategy: Distillation strategy ('direct' or 'sequential')
        device: Device to train on (CPU or GPU)
        
    Returns:
        Trained CascadeDistiller model
    """
    input_dim = vectors.shape[1]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Training cascade distiller with dimensions: {dimensions}")
    logger.info(f"Using strategy: {strategy}")
    
    # Create the cascade distiller
    cascade = CascadeDistiller(
        input_dim=input_dim,
        dimensions=dimensions,
        strategy=strategy,
    )
    
    # Train each distiller
    for dim in dimensions:
        logger.info(f"Training distiller for dimension {dim}")
        distiller = train_dimension_distiller(
            vectors=vectors,
            target_dim=dim,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_pairs=n_pairs,
            output_dir=output_dir,
            device=device,
        )
        
        # Add to cascade
        cascade.add_distiller(dim, distiller)
    
    # Save the cascade model
    cascade_path = os.path.join(output_dir, "cascade_distiller.pt")
    torch.save({
        'input_dim': input_dim,
        'dimensions': dimensions,
        'strategy': strategy,
        'state_dict': cascade.state_dict(),
    }, cascade_path)
    logger.info(f"Cascade model saved to {cascade_path}")
    
    return cascade

def main():
    """Main function for training dimension distillers."""
    parser = argparse.ArgumentParser(description="Train a dimensional cascade distiller")
    
    # Input data parameters
    parser.add_argument("--vectors_file", type=str, help="Path to the vectors file")
    parser.add_argument("--n_synthetic", type=int, default=10000, 
                      help="Number of synthetic vectors to generate if no file is provided")
    parser.add_argument("--synthetic_dim", type=int, default=768, 
                      help="Dimension of synthetic vectors")
    
    # Dimension parameters
    parser.add_argument("--dimensions", type=str, default="32,64,128,256",
                      help="Comma-separated list of target dimensions")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument("--n_pairs", type=int, default=10000, help="Number of pairs for training")
    
    # Distillation parameters
    parser.add_argument("--strategy", type=str, default="direct", choices=["direct", "sequential"],
                      help="Distillation strategy")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check if vectors file exists or generate synthetic data
    if args.vectors_file and os.path.exists(args.vectors_file):
        logger.info(f"Loading vectors from {args.vectors_file}")
        vectors = load_vectors_from_file(args.vectors_file)
    else:
        logger.info(f"Generating {args.n_synthetic} synthetic vectors with dimension {args.synthetic_dim}")
        vectors = generate_synthetic_data(
            n_samples=args.n_synthetic,
            n_features=args.synthetic_dim,
            dimensions=[32, 64, 128, 256],
            n_clusters=10,
        )[args.synthetic_dim]
    
    # Parse dimensions
    dimensions = parse_dimensions(args.dimensions)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Train cascade distiller
    cascade = train_cascade_distiller(
        vectors=vectors,
        dimensions=dimensions,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        n_pairs=args.n_pairs,
        output_dir=args.output_dir,
        strategy=args.strategy,
        device=device,
    )
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main() 