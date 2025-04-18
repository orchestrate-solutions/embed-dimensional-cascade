"""
Distillation pipeline for dimensional cascade.

This module provides an end-to-end pipeline for dimensional distillation,
from loading high-dimensional embeddings to training and evaluating models
that compress embeddings to various lower dimensions.
"""

import os
import logging
import argparse
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union, Any
import json
import time
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.data_loader import load_vectors_from_file
from src.distillation.model import create_distillation_model, create_distillation_loss
from src.distillation.trainer import DistillationTrainer, train_distillation_model, evaluate_distilled_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DimensionalDistillationPipeline:
    """
    Pipeline for training models to distill high-dimensional embeddings into multiple
    lower dimensions in a cascaded manner.
    
    This pipeline can either:
    1. Train a single model directly from high dimensions to low dimensions
    2. Train a cascade of models, where each step reduces dimensions by a smaller amount
    """
    
    def __init__(
        self,
        input_dim: int,
        target_dims: List[int],
        output_dir: str = "models/distillation",
        device: Optional[torch.device] = None,
        cascade_strategy: str = "direct"
    ):
        """
        Initialize the distillation pipeline.
        
        Args:
            input_dim: Dimension of the input embeddings
            target_dims: List of target dimensions to distill to (in descending order)
            output_dir: Directory to save trained models and results
            device: Device to run training on (default: auto-detect)
            cascade_strategy: Strategy for cascade training:
                - "direct": Train separate models from input_dim to each target_dim
                - "sequential": Train models in sequence, starting from highest dimension
                - "balanced": Train models with similar reduction ratios
        """
        self.input_dim = input_dim
        self.target_dims = sorted(target_dims, reverse=True)  # Sort in descending order
        
        # Ensure target dims are all smaller than input dim
        for dim in self.target_dims:
            if dim >= input_dim:
                raise ValueError(f"Target dimension {dim} must be smaller than input dimension {input_dim}")
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Set up directories
        self.output_dir = Path(output_dir)
        self.model_dir = self.output_dir / "models"
        self.results_dir = self.output_dir / "results"
        self.plots_dir = self.output_dir / "plots"
        
        # Create directories
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Set cascade strategy
        self.cascade_strategy = cascade_strategy
        
        # To store trained models and evaluation results
        self.models = {}
        self.evaluation_results = {}
        
        logger.info(f"Initialized DimensionalDistillationPipeline on {self.device}")
        logger.info(f"Target dimensions: {self.target_dims}")
        logger.info(f"Cascade strategy: {self.cascade_strategy}")
    
    def determine_cascade_path(self) -> List[Tuple[int, int]]:
        """
        Determine the sequence of dimension reductions based on the selected strategy.
        
        Returns:
            List of (source_dim, target_dim) tuples representing the training steps
        """
        if self.cascade_strategy == "direct":
            # Train direct models from input_dim to each target_dim
            return [(self.input_dim, dim) for dim in self.target_dims]
        
        elif self.cascade_strategy == "sequential":
            # Train sequential models, starting from highest dimension
            steps = []
            current_dim = self.input_dim
            
            for dim in self.target_dims:
                steps.append((current_dim, dim))
                current_dim = dim
                
            return steps
        
        elif self.cascade_strategy == "balanced":
            # Compute intermediate dimensions for more balanced reduction ratios
            steps = []
            
            for dim in self.target_dims:
                # Determine if we need intermediate steps
                ratio = dim / self.input_dim
                
                if ratio <= 0.25:  # If the reduction is significant
                    # Add an intermediate step
                    intermediate_dim = int(np.sqrt(self.input_dim * dim))
                    steps.append((self.input_dim, intermediate_dim))
                    steps.append((intermediate_dim, dim))
                else:
                    # Direct step
                    steps.append((self.input_dim, dim))
                    
            return steps
        
        else:
            raise ValueError(f"Unknown cascade strategy: {self.cascade_strategy}")
    
    def train(
        self,
        embeddings: torch.Tensor,
        val_embeddings: Optional[torch.Tensor] = None,
        hidden_dims: Optional[Dict[int, List[int]]] = None,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        epochs: int = 100,
        similarity_weight: float = 1.0,
        ranking_weight: float = 1.0,
        val_split: float = 0.1,
        use_scheduler: bool = True
    ) -> Dict[int, Dict[str, Any]]:
        """
        Train distillation models for all target dimensions.
        
        Args:
            embeddings: Input embeddings to distill
            val_embeddings: Optional separate validation embeddings
            hidden_dims: Optional dictionary mapping target_dim to hidden layer dimensions
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            epochs: Number of training epochs
            similarity_weight: Weight for similarity preservation loss
            ranking_weight: Weight for ranking preservation loss
            val_split: Fraction of data to use for validation if val_embeddings not provided
            use_scheduler: Whether to use a learning rate scheduler
            
        Returns:
            Dictionary mapping target dimensions to training results
        """
        # Determine the cascade path based on the strategy
        cascade_path = self.determine_cascade_path()
        logger.info(f"Cascade path: {cascade_path}")
        
        # Start time
        start_time = time.time()
        
        # Dictionary to store results
        training_results = {}
        
        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = {}
        
        # Dictionary to cache intermediate embeddings
        cached_embeddings = {self.input_dim: embeddings}
        if val_embeddings is not None:
            cached_val_embeddings = {self.input_dim: val_embeddings}
        else:
            cached_val_embeddings = None
        
        # Train models according to the cascade path
        for source_dim, target_dim in cascade_path:
            logger.info(f"Training model for {source_dim}→{target_dim} dimension reduction")
            
            # Get source embeddings
            source_embeddings = cached_embeddings[source_dim]
            
            # Get validation embeddings if available
            if cached_val_embeddings is not None:
                source_val_embeddings = cached_val_embeddings[source_dim]
            else:
                source_val_embeddings = None
            
            # Get hidden dimensions for this model
            model_hidden_dims = hidden_dims.get(target_dim, None)
            if model_hidden_dims is None:
                # Auto-determine reasonable hidden dimensions
                ratio = target_dim / source_dim
                
                if ratio <= 0.25:
                    # Use two hidden layers
                    hidden1 = max(source_dim // 2, target_dim * 2)
                    hidden2 = max(target_dim * 2, hidden1 // 2)
                    model_hidden_dims = [hidden1, hidden2]
                else:
                    # Use one hidden layer
                    hidden = max(source_dim // 2, target_dim * 2)
                    model_hidden_dims = [hidden]
            
            # Set experiment name
            experiment_name = f"{source_dim}to{target_dim}"
            experiment_dir = self.model_dir / experiment_name
            os.makedirs(experiment_dir, exist_ok=True)
            
            # Train the model
            model, history = train_distillation_model(
                teacher_embeddings=source_embeddings,
                output_dim=target_dim,
                val_embeddings=source_val_embeddings,
                hidden_dims=model_hidden_dims,
                batch_size=batch_size,
                learning_rate=learning_rate,
                epochs=epochs,
                similarity_weight=similarity_weight,
                ranking_weight=ranking_weight,
                device=self.device,
                output_dir=str(experiment_dir),
                experiment_name=experiment_name,
                use_scheduler=use_scheduler,
                val_split=val_split
            )
            
            # Store the model
            self.models[(source_dim, target_dim)] = model
            
            # Generate and cache distilled embeddings for next steps
            model.eval()
            with torch.no_grad():
                # Process embeddings in batches
                distilled_embeddings = []
                for i in range(0, len(source_embeddings), batch_size):
                    batch = source_embeddings[i:i+batch_size].to(self.device)
                    output = model(batch).cpu()
                    distilled_embeddings.append(output)
                
                # Concatenate batches
                distilled_embeddings = torch.cat(distilled_embeddings, dim=0)
                cached_embeddings[target_dim] = distilled_embeddings
                
                # Process validation embeddings if available
                if source_val_embeddings is not None:
                    distilled_val_embeddings = []
                    for i in range(0, len(source_val_embeddings), batch_size):
                        batch = source_val_embeddings[i:i+batch_size].to(self.device)
                        output = model(batch).cpu()
                        distilled_val_embeddings.append(output)
                    
                    distilled_val_embeddings = torch.cat(distilled_val_embeddings, dim=0)
                    cached_val_embeddings[target_dim] = distilled_val_embeddings
            
            # Evaluate the distilled embeddings
            eval_results = evaluate_distilled_embeddings(
                model=model,
                teacher_embeddings=source_embeddings,
                batch_size=batch_size,
                device=self.device
            )
            
            # Store training and evaluation results
            training_results[target_dim] = {
                'source_dim': source_dim,
                'target_dim': target_dim,
                'hidden_dims': model_hidden_dims,
                'history': history,
                'evaluation': eval_results,
                'time': time.time() - start_time
            }
            
            # Save results to disk
            with open(self.results_dir / f"{source_dim}to{target_dim}_results.json", 'w') as f:
                # Convert data types for JSON serialization
                serializable_results = {
                    'source_dim': source_dim,
                    'target_dim': target_dim,
                    'hidden_dims': model_hidden_dims,
                    'history': {
                        'train_loss': [float(x) for x in history['train_loss']],
                        'val_loss': [float(x) for x in history['val_loss']]
                    },
                    'evaluation': {k: float(v) for k, v in eval_results.items()},
                    'time': float(time.time() - start_time)
                }
                json.dump(serializable_results, f, indent=2)
        
        # Create summary plot
        self.plot_recall_comparison(training_results)
        
        # Return all results
        return training_results
    
    def plot_recall_comparison(self, results: Dict[int, Dict[str, Any]]) -> str:
        """
        Plot recall comparison for different dimensions.
        
        Args:
            results: Training results dictionary
            
        Returns:
            Path to the saved plot
        """
        plt.figure(figsize=(10, 6))
        
        # Collect data for the plot
        dimensions = []
        recall_at_1 = []
        recall_at_10 = []
        recall_at_100 = []
        rank_correlation = []
        
        for dim in sorted(results.keys()):
            dimensions.append(dim)
            recall_at_1.append(results[dim]['evaluation'].get('recall@1', 0))
            recall_at_10.append(results[dim]['evaluation'].get('recall@10', 0))
            recall_at_100.append(results[dim]['evaluation'].get('recall@100', 0))
            rank_correlation.append(results[dim]['evaluation'].get('rank_correlation', 0))
        
        # Plot metrics
        plt.plot(dimensions, recall_at_1, 'o-', label='Recall@1', linewidth=2)
        plt.plot(dimensions, recall_at_10, 's-', label='Recall@10', linewidth=2)
        plt.plot(dimensions, recall_at_100, '^-', label='Recall@100', linewidth=2)
        plt.plot(dimensions, rank_correlation, 'x-', label='Rank Correlation', linewidth=2)
        
        plt.xlabel('Embedding Dimension')
        plt.ylabel('Metric Value')
        plt.title('Distillation Performance by Dimension')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xscale('log', base=2)
        
        # Add dimension labels on x-axis
        plt.xticks(dimensions, [str(d) for d in dimensions])
        
        # Save the plot
        plot_path = self.plots_dir / 'dimension_comparison.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved dimension comparison plot to {plot_path}")
        return str(plot_path)
    
    def save_models(self) -> None:
        """
        Save all trained models to disk.
        """
        for (source_dim, target_dim), model in self.models.items():
            # Create directory if needed
            model_dir = self.model_dir / f"{source_dim}to{target_dim}"
            os.makedirs(model_dir, exist_ok=True)
            
            # Save the model
            model_path = model_dir / "model.pt"
            torch.save(model.state_dict(), model_path)
            
            # Also save a JSON with model architecture info
            model_info = {
                'source_dim': source_dim,
                'target_dim': target_dim,
                'model_type': model.__class__.__name__,
                'architecture': {
                    'input_dim': source_dim,
                    'output_dim': target_dim,
                    'hidden_layers': []
                }
            }
            
            # Extract hidden layer sizes
            for module in model.modules():
                if isinstance(module, torch.nn.Linear):
                    model_info['architecture']['hidden_layers'].append({
                        'in_features': module.in_features,
                        'out_features': module.out_features
                    })
            
            with open(model_dir / "model_info.json", 'w') as f:
                json.dump(model_info, f, indent=2)
                
            logger.info(f"Saved model for {source_dim}→{target_dim} to {model_path}")
    
    def load_models(self) -> None:
        """
        Load all trained models from disk.
        """
        model_dirs = [d for d in self.model_dir.iterdir() if d.is_dir()]
        
        for model_dir in model_dirs:
            # Try to parse directory name as source_dim to target_dim
            try:
                dir_name = model_dir.name
                if "to" in dir_name:
                    source_dim, target_dim = map(int, dir_name.split("to"))
                    
                    # Check if model file exists
                    model_path = model_dir / "model.pt"
                    if model_path.exists():
                        # Create model with same architecture
                        model = create_distillation_model(
                            input_dim=source_dim,
                            output_dim=target_dim
                        )
                        
                        # Load weights
                        model.load_state_dict(torch.load(model_path, map_location=self.device))
                        model = model.to(self.device)
                        model.eval()
                        
                        # Store model
                        self.models[(source_dim, target_dim)] = model
                        logger.info(f"Loaded model for {source_dim}→{target_dim} from {model_path}")
            except Exception as e:
                logger.warning(f"Failed to load model from {model_dir}: {e}")
    
    def distill(
        self,
        embeddings: torch.Tensor,
        target_dim: int,
        batch_size: int = 64
    ) -> torch.Tensor:
        """
        Distill embeddings to the specified target dimension using trained models.
        
        Args:
            embeddings: Input embeddings to distill
            target_dim: Target dimension to distill to
            batch_size: Batch size for processing
            
        Returns:
            Distilled embeddings
        """
        # Check if target dimension is valid
        if target_dim not in self.target_dims:
            raise ValueError(f"Target dimension {target_dim} not in available dimensions: {self.target_dims}")
        
        # Determine path based on cascade strategy
        source_dim = embeddings.shape[1]
        
        if self.cascade_strategy == "direct":
            # Just use the direct model if available
            if (source_dim, target_dim) in self.models:
                path = [(source_dim, target_dim)]
            else:
                raise ValueError(f"No model available for {source_dim}→{target_dim}")
        
        elif self.cascade_strategy == "sequential":
            # Find sequential path
            path = []
            current_dim = source_dim
            
            while current_dim > target_dim:
                # Find the next step down
                next_dims = [dim for dim in self.target_dims if dim < current_dim and dim >= target_dim]
                
                if not next_dims:
                    # No more steps available
                    break
                
                next_dim = max(next_dims)
                
                if (current_dim, next_dim) in self.models:
                    path.append((current_dim, next_dim))
                    current_dim = next_dim
                else:
                    logger.warning(f"No model available for {current_dim}→{next_dim}, skipping this step")
                    break
            
            if path and path[-1][1] != target_dim:
                raise ValueError(f"No complete path available to target dimension {target_dim}")
        
        else:  # balanced or other strategy
            # Try to find the most direct path
            available_models = list(self.models.keys())
            path = []
            
            # Start from source dimension
            current_dim = source_dim
            
            # Keep going until we reach the target dimension
            while current_dim != target_dim:
                # Find all possible next steps
                possible_steps = [(s, t) for s, t in available_models if s == current_dim]
                
                if not possible_steps:
                    # Try indirect steps
                    possible_steps = [(s, t) for s, t in available_models if s > current_dim]
                    if not possible_steps:
                        raise ValueError(f"No path available from {current_dim} to {target_dim}")
                
                # Choose the step that gets us closest to the target dimension
                next_step = min(possible_steps, key=lambda x: abs(x[1] - target_dim))
                path.append(next_step)
                current_dim = next_step[1]
                
                # If we've gone too low, we need to stop
                if current_dim < target_dim:
                    raise ValueError(f"Went below target dimension {target_dim}")
        
        logger.info(f"Distillation path: {path}")
        
        # Apply models in sequence
        current_embeddings = embeddings
        
        for source_dim, target_dim in path:
            model = self.models[(source_dim, target_dim)]
            model.eval()
            
            # Process in batches
            results = []
            with torch.no_grad():
                for i in range(0, len(current_embeddings), batch_size):
                    batch = current_embeddings[i:i+batch_size].to(self.device)
                    output = model(batch).cpu()
                    results.append(output)
            
            # Update current embeddings
            current_embeddings = torch.cat(results, dim=0)
        
        return current_embeddings


def main():
    """Main entry point for the distillation pipeline."""
    parser = argparse.ArgumentParser(description="Train and evaluate distillation models")
    parser.add_argument("--input_file", required=True, type=str,
                        help="Path to input embeddings file")
    parser.add_argument("--output_dir", default="models/distillation", type=str,
                        help="Directory to save models and results")
    parser.add_argument("--target_dims", default="32,64,128,256", type=str,
                        help="Comma-separated list of target dimensions")
    parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="Learning rate for training")
    parser.add_argument("--epochs", default=100, type=int,
                        help="Number of epochs for training")
    parser.add_argument("--similarity_weight", default=1.0, type=float,
                        help="Weight for similarity preservation loss")
    parser.add_argument("--ranking_weight", default=1.0, type=float,
                        help="Weight for ranking preservation loss")
    parser.add_argument("--val_split", default=0.1, type=float,
                        help="Fraction of data to use for validation")
    parser.add_argument("--cascade_strategy", default="direct", type=str,
                        choices=["direct", "sequential", "balanced"],
                        help="Strategy for cascade training")
    
    args = parser.parse_args()
    
    # Parse target dimensions
    target_dims = [int(dim) for dim in args.target_dims.split(",")]
    
    # Load embeddings
    logger.info(f"Loading embeddings from {args.input_file}")
    embeddings_np = load_vectors_from_file(args.input_file)
    
    # Convert to torch tensor
    embeddings = torch.from_numpy(embeddings_np).float()
    
    # Create pipeline
    input_dim = embeddings.shape[1]
    logger.info(f"Input dimension: {input_dim}")
    
    pipeline = DimensionalDistillationPipeline(
        input_dim=input_dim,
        target_dims=target_dims,
        output_dir=args.output_dir,
        cascade_strategy=args.cascade_strategy
    )
    
    # Train models
    logger.info("Starting training...")
    results = pipeline.train(
        embeddings=embeddings,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        similarity_weight=args.similarity_weight,
        ranking_weight=args.ranking_weight,
        val_split=args.val_split
    )
    
    # Save models
    logger.info("Saving models...")
    pipeline.save_models()
    
    logger.info("Pipeline completed successfully")
    

if __name__ == "__main__":
    main() 