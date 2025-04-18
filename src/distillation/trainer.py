"""
Trainer module for dimensional cascade distillation models.

This module provides utilities for training dimension distillation models
that transform embeddings from higher to lower dimensions while preserving
similarity relationships in the dimensional cascade search system.
"""

import os
import time
import logging
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Callable

from src.distillation.models import DimensionDistiller, CascadeDistiller
from src.utils.data_loader import load_vectors_from_file

logger = logging.getLogger(__name__)

class DistillationTrainer:
    """Trainer for dimension distillation models."""
    
    def __init__(
        self,
        model: Union[DimensionDistiller, CascadeDistiller],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        device: Optional[str] = None,
        output_dir: str = "./models",
        experiment_name: Optional[str] = None
    ):
        """
        Initialize the distillation trainer.
        
        Args:
            model: The distillation model to train
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            device: Device to use for training ('cuda', 'mps', or 'cpu')
            output_dir: Directory to save models and results
            experiment_name: Name for the experiment (used in saved files)
        """
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else \
                         'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else \
                         'cpu'
        else:
            self.device = device
            
        logger.info(f"Using device: {self.device}")
        
        self.model = self.model.to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Setup output directories
        self.output_dir = output_dir
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.experiment_name = experiment_name or f"distill_{timestamp}"
        self.experiment_dir = os.path.join(output_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Track metrics
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def cosine_similarity_loss(self, 
                              original_embeddings: torch.Tensor, 
                              distilled_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute cosine similarity loss between original and distilled embeddings.
        
        Args:
            original_embeddings: Original high-dimensional embeddings
            distilled_embeddings: Distilled lower-dimensional embeddings
            
        Returns:
            Cosine similarity loss (higher similarity = lower loss)
        """
        # Normalize embeddings for cosine similarity
        original_norm = torch.nn.functional.normalize(original_embeddings, p=2, dim=1)
        distilled_norm = torch.nn.functional.normalize(distilled_embeddings, p=2, dim=1)
        
        # Compute cosine similarity matrix for original embeddings
        original_sim = torch.mm(original_norm, original_norm.t())
        
        # Compute cosine similarity matrix for distilled embeddings
        distilled_sim = torch.mm(distilled_norm, distilled_norm.t())
        
        # MSE between similarity matrices (preserving similarity relationships)
        loss = torch.nn.functional.mse_loss(distilled_sim, original_sim)
        return loss
    
    def triplet_loss(self,
                    anchor: torch.Tensor,
                    positive: torch.Tensor,
                    negative: torch.Tensor,
                    margin: float = 0.2) -> torch.Tensor:
        """
        Compute triplet loss to maintain relative distances.
        
        Args:
            anchor: Anchor embeddings
            positive: Positive embeddings (same class/similar)
            negative: Negative embeddings (different class/dissimilar)
            margin: Minimum margin between positive and negative distances
            
        Returns:
            Triplet loss value
        """
        # Normalize embeddings
        anchor = torch.nn.functional.normalize(anchor, p=2, dim=1)
        positive = torch.nn.functional.normalize(positive, p=2, dim=1)
        negative = torch.nn.functional.normalize(negative, p=2, dim=1)
        
        # Compute distances
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        
        # Compute triplet loss with margin
        losses = torch.relu(pos_dist - neg_dist + margin)
        return losses.mean()
    
    def prepare_data(self, 
                    vectors: np.ndarray, 
                    batch_size: int = 128,
                    val_split: float = 0.1,
                    seed: int = 42) -> Tuple[DataLoader, Optional[DataLoader]]:
        """
        Prepare data loaders from input vectors.
        
        Args:
            vectors: Input embedding vectors
            batch_size: Batch size for training
            val_split: Validation split ratio
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Convert to PyTorch tensors
        vectors_tensor = torch.tensor(vectors, dtype=torch.float32)
        
        # Split into train/val if needed
        if val_split > 0:
            # Shuffle indices
            rng = np.random.RandomState(seed)
            indices = np.arange(len(vectors))
            rng.shuffle(indices)
            
            # Split indices
            val_size = int(len(vectors) * val_split)
            train_indices = indices[val_size:]
            val_indices = indices[:val_size]
            
            # Create datasets
            train_vectors = vectors_tensor[train_indices]
            val_vectors = vectors_tensor[val_indices]
            
            # Create data loaders
            train_dataset = TensorDataset(train_vectors)
            val_dataset = TensorDataset(val_vectors)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            return train_loader, val_loader
        else:
            # Just create training dataset
            dataset = TensorDataset(vectors_tensor)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            return train_loader, None
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        batch_count = 0
        
        for batch in train_loader:
            # Get batch of vectors
            vectors = batch[0].to(self.device)
            batch_size = vectors.shape[0]
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass depends on model type
            if isinstance(self.model, DimensionDistiller):
                # Single dimension distiller
                distilled = self.model(vectors)
                loss = self.cosine_similarity_loss(vectors, distilled)
            elif isinstance(self.model, CascadeDistiller):
                # Cascade has multiple distillers - compute overall loss
                loss = torch.tensor(0.0).to(self.device)
                
                # Each distiller contributes to loss
                for source_dim, distiller in self.model.distillers.items():
                    for target_dim in distiller.target_dimensions:
                        # Get distilled vectors
                        distilled = self.model.distill(vectors, target_dim)
                        if distilled is not None:
                            sim_loss = self.cosine_similarity_loss(vectors, distilled)
                            loss += sim_loss
            else:
                raise TypeError(f"Unsupported model type: {type(self.model)}")
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            batch_count += 1
            
        # Calculate average loss
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate the model.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Validation loss
        """
        self.model.eval()
        total_loss = 0.0
        batch_count = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Get batch of vectors
                vectors = batch[0].to(self.device)
                
                # Forward pass depends on model type
                if isinstance(self.model, DimensionDistiller):
                    # Single dimension distiller
                    distilled = self.model(vectors)
                    loss = self.cosine_similarity_loss(vectors, distilled)
                elif isinstance(self.model, CascadeDistiller):
                    # Cascade has multiple distillers
                    loss = torch.tensor(0.0).to(self.device)
                    
                    # Each distiller contributes to loss
                    for source_dim, distiller in self.model.distillers.items():
                        for target_dim in distiller.target_dimensions:
                            # Get distilled vectors
                            distilled = self.model.distill(vectors, target_dim)
                            if distilled is not None:
                                sim_loss = self.cosine_similarity_loss(vectors, distilled)
                                loss += sim_loss
                else:
                    raise TypeError(f"Unsupported model type: {type(self.model)}")
                
                # Track metrics
                total_loss += loss.item()
                batch_count += 1
        
        # Calculate average loss
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        return avg_loss
    
    def train(self, 
             train_loader: DataLoader,
             val_loader: Optional[DataLoader] = None,
             num_epochs: int = 100,
             patience: int = 10,
             save_best: bool = True,
             eval_every: int = 1,
             verbose: bool = True) -> Dict:
        """
        Train the model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation
            num_epochs: Number of epochs to train
            patience: Early stopping patience (0 = no early stopping)
            save_best: Whether to save the best model
            eval_every: Validate every N epochs
            verbose: Whether to print progress
            
        Returns:
            Dictionary with training metrics
        """
        best_epoch = 0
        patience_counter = 0
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Calculate ETA
            elapsed = time.time() - start_time
            progress = epoch / num_epochs
            eta_seconds = elapsed / progress * (1 - progress) if progress > 0 else 0
            eta_min = eta_seconds / 60
            
            # Validate if needed
            if val_loader is not None and epoch % eval_every == 0:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
                # Check for improvement
                if val_loss < self.best_val_loss:
                    improvement = (self.best_val_loss - val_loss) / self.best_val_loss * 100
                    self.best_val_loss = val_loss
                    best_epoch = epoch
                    patience_counter = 0
                    
                    # Save best model
                    if save_best:
                        self.save_model("best_model.pt")
                    
                    improvement_msg = f", improved by {improvement:.2f}%"
                else:
                    improvement_msg = ""
                    patience_counter += 1
            else:
                val_loss = None
                improvement_msg = ""
            
            # Print progress
            if verbose and (epoch % eval_every == 0 or epoch == num_epochs):
                elapsed_min = elapsed / 60
                if val_loss is not None:
                    logger.info(f"Epoch {epoch}/{num_epochs}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}{improvement_msg}, "
                              f"time={elapsed_min:.1f}m, ETA={eta_min:.1f}m")
                else:
                    logger.info(f"Epoch {epoch}/{num_epochs}: train_loss={train_loss:.6f}, "
                              f"time={elapsed_min:.1f}m, ETA={eta_min:.1f}m")
            
            # Early stopping
            if patience > 0 and patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
        
        # Calculate training time
        train_time = time.time() - start_time
        
        # Save final model
        self.save_model("final_model.pt")
        
        # Plot and save learning curve
        self.plot_learning_curve()
        
        # Return metrics
        metrics = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses if val_loader is not None else None,
            "best_val_loss": self.best_val_loss if val_loader is not None else None,
            "best_epoch": best_epoch if val_loader is not None else None,
            "train_time": train_time,
            "epochs_completed": epoch
        }
        
        return metrics
    
    def save_model(self, filename: str):
        """Save model to file."""
        filepath = os.path.join(self.experiment_dir, filename)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "model_config": self.model.get_config() if hasattr(self.model, "get_config") else None
        }, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_losses = checkpoint.get("train_losses", [])
        self.val_losses = checkpoint.get("val_losses", [])
        logger.info(f"Model loaded from {filepath}")
    
    def plot_learning_curve(self):
        """Plot and save learning curve."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label="Training Loss")
        if self.val_losses:
            # Plot validation loss at correct intervals
            val_epochs = list(range(0, len(self.train_losses), len(self.train_losses) // len(self.val_losses)))
            if len(val_epochs) > len(self.val_losses):
                val_epochs = val_epochs[:len(self.val_losses)]
            plt.plot(val_epochs, self.val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid(True)
        
        # Save figure
        fig_path = os.path.join(self.experiment_dir, "learning_curve.png")
        plt.savefig(fig_path)
        plt.close()
        logger.info(f"Learning curve saved to {fig_path}")
    
    def evaluate_similarity_preservation(self, test_vectors: np.ndarray) -> Dict:
        """
        Evaluate how well the model preserves similarity relationships.
        
        Args:
            test_vectors: Test vectors to evaluate on
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        results = {}
        
        # Convert to tensor
        test_tensor = torch.tensor(test_vectors, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            # Original similarity matrix
            original_norm = torch.nn.functional.normalize(test_tensor, p=2, dim=1)
            original_sim = torch.mm(original_norm, original_norm.t()).cpu().numpy()
            
            # For each target dimension, evaluate similarity preservation
            if isinstance(self.model, DimensionDistiller):
                # Single distiller
                distilled = self.model(test_tensor)
                distilled_norm = torch.nn.functional.normalize(distilled, p=2, dim=1)
                distilled_sim = torch.mm(distilled_norm, distilled_norm.t()).cpu().numpy()
                
                # Compute metrics
                similarity_mse = np.mean((original_sim - distilled_sim) ** 2)
                results[self.model.output_dim] = {
                    "mse": similarity_mse,
                    "output_dim": self.model.output_dim,
                    "compression_ratio": test_vectors.shape[1] / self.model.output_dim
                }
            elif isinstance(self.model, CascadeDistiller):
                # Evaluate each target dimension
                for source_dim, distiller in self.model.distillers.items():
                    for target_dim in distiller.target_dimensions:
                        # Skip if we can't distill to this dimension
                        if source_dim < target_dim:
                            continue
                            
                        # Get distilled vectors
                        distilled = self.model.distill(test_tensor, target_dim)
                        if distilled is None:
                            continue
                            
                        # Compute similarity matrix
                        distilled_norm = torch.nn.functional.normalize(distilled, p=2, dim=1)
                        distilled_sim = torch.mm(distilled_norm, distilled_norm.t()).cpu().numpy()
                        
                        # Compute metrics
                        similarity_mse = np.mean((original_sim - distilled_sim) ** 2)
                        results[target_dim] = {
                            "mse": similarity_mse,
                            "output_dim": target_dim,
                            "compression_ratio": test_vectors.shape[1] / target_dim
                        }
        
        return results


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Train dimension distillation models')
    parser.add_argument('--vectors_file', type=str, help='Path to vectors file')
    parser.add_argument('--input_dim', type=int, default=768, help='Input dimension')
    parser.add_argument('--output_dims', type=str, default='256,128,64,32', 
                       help='Comma-separated list of output dimensions')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--output_dir', type=str, default='./models', help='Output directory')
    
    args = parser.parse_args()
    
    # Parse output dimensions
    output_dims = [int(dim) for dim in args.output_dims.split(',')]
    
    # Load vectors or create synthetic data
    if args.vectors_file:
        vectors = load_vectors_from_file(args.vectors_file)
        input_dim = vectors.shape[1]
        logger.info(f"Loaded {vectors.shape[0]} vectors with dimension {input_dim}")
    else:
        # Create synthetic data for testing
        input_dim = args.input_dim
        n_samples = 10000
        vectors = np.random.randn(n_samples, input_dim).astype(np.float32)
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        logger.info(f"Created {n_samples} synthetic vectors with dimension {input_dim}")
    
    # Create model - either single distiller or cascade
    if len(output_dims) == 1:
        # Single distiller
        model = DimensionDistiller(input_dim=input_dim, output_dim=output_dims[0])
        logger.info(f"Created DimensionDistiller: {input_dim} → {output_dims[0]}")
    else:
        # Cascade distiller
        model = CascadeDistiller()
        
        # Add distillers to the cascade
        model.add_distiller(input_dim, output_dims)
        logger.info(f"Created CascadeDistiller with dimensions: {input_dim} → {output_dims}")
    
    # Create trainer
    trainer = DistillationTrainer(
        model=model,
        learning_rate=args.lr,
        output_dir=args.output_dir
    )
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(
        vectors=vectors,
        batch_size=args.batch_size,
        val_split=0.1
    )
    
    # Train model
    metrics = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        patience=args.patience
    )
    
    # Evaluate model
    results = trainer.evaluate_similarity_preservation(vectors[:1000])
    
    # Print results
    logger.info("Training complete!")
    logger.info(f"Final training loss: {metrics['train_losses'][-1]:.6f}")
    if metrics['val_losses']:
        logger.info(f"Final validation loss: {metrics['val_losses'][-1]:.6f}")
        logger.info(f"Best validation loss: {metrics['best_val_loss']:.6f} (epoch {metrics['best_epoch']})")
    
    logger.info("Similarity preservation results:")
    for dim, result in results.items():
        logger.info(f"Dimension {dim}: MSE={result['mse']:.6f}, "
                   f"Compression ratio={result['compression_ratio']:.2f}x") 