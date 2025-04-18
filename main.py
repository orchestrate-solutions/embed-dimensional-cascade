#!/usr/bin/env python
"""
Dimensional Cascade - Main entry point

This script provides a command-line interface for running the dimensional cascade
pipeline, which includes:
1. Data loading and preprocessing
2. Model training via dimension distillation
3. Evaluation of distilled models
"""

import os
import sys
import logging
import argparse
import numpy as np
import torch
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_parser():
    """Create argument parser for the CLI."""
    parser = argparse.ArgumentParser(description="Dimensional Cascade - Vector Distillation")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Parser for training
    train_parser = subparsers.add_parser("train", help="Train distillation models")
    train_parser.add_argument("--input_file", required=True, type=str,
                        help="Path to input embeddings file")
    train_parser.add_argument("--output_dir", default="models/distillation", type=str,
                        help="Directory to save models and results")
    train_parser.add_argument("--target_dims", default="32,64,128,256", type=str,
                        help="Comma-separated list of target dimensions")
    train_parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size for training")
    train_parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="Learning rate for training")
    train_parser.add_argument("--epochs", default=100, type=int,
                        help="Number of epochs for training")
    train_parser.add_argument("--similarity_weight", default=1.0, type=float,
                        help="Weight for similarity preservation loss")
    train_parser.add_argument("--ranking_weight", default=1.0, type=float,
                        help="Weight for ranking preservation loss")
    train_parser.add_argument("--val_split", default=0.1, type=float,
                        help="Fraction of data to use for validation")
    train_parser.add_argument("--cascade_strategy", default="direct", type=str,
                        choices=["direct", "sequential", "balanced"],
                        help="Strategy for cascade training")
    
    # Parser for distilling embeddings
    distill_parser = subparsers.add_parser("distill", help="Distill embeddings to lower dimensions")
    distill_parser.add_argument("--input_file", required=True, type=str,
                        help="Path to input embeddings file")
    distill_parser.add_argument("--output_file", required=True, type=str,
                        help="Path to save distilled embeddings")
    distill_parser.add_argument("--model_dir", required=True, type=str,
                        help="Directory containing trained models")
    distill_parser.add_argument("--target_dim", required=True, type=int,
                        help="Target dimension for distillation")
    distill_parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size for processing")
    distill_parser.add_argument("--cascade_strategy", default="direct", type=str,
                        choices=["direct", "sequential", "balanced"],
                        help="Strategy for cascade distillation")
    
    # Parser for evaluating models
    evaluate_parser = subparsers.add_parser("evaluate", help="Evaluate distillation models")
    evaluate_parser.add_argument("--input_file", required=True, type=str,
                        help="Path to input embeddings file")
    evaluate_parser.add_argument("--model_dir", required=True, type=str,
                        help="Directory containing trained models")
    evaluate_parser.add_argument("--output_dir", default="evaluation_results", type=str,
                        help="Directory to save evaluation results")
    evaluate_parser.add_argument("--batch_size", default=64, type=int,
                        help="Batch size for processing")
    evaluate_parser.add_argument("--k_values", default="1,10,100", type=str,
                        help="Comma-separated list of k values for recall@k")
    evaluate_parser.add_argument("--num_queries", default=1000, type=int,
                        help="Number of queries for evaluation")
    
    # Parser for generating synthetic data
    synthetic_parser = subparsers.add_parser("generate", help="Generate synthetic data for testing")
    synthetic_parser.add_argument("--output_file", required=True, type=str,
                        help="Path to save synthetic embeddings")
    synthetic_parser.add_argument("--n_samples", default=10000, type=int,
                        help="Number of samples to generate")
    synthetic_parser.add_argument("--n_features", default=768, type=int,
                        help="Number of features/dimensions")
    synthetic_parser.add_argument("--n_clusters", default=10, type=int,
                        help="Number of clusters for synthetic data")
    
    return parser


def train_command(args):
    """Run the training command."""
    from src.distillation.pipeline import DimensionalDistillationPipeline
    from src.utils.data_loader import load_vectors_from_file
    
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


def distill_command(args):
    """Run the distill command."""
    from src.distillation.pipeline import DimensionalDistillationPipeline
    from src.utils.data_loader import load_vectors_from_file
    import numpy as np
    
    # Load embeddings
    logger.info(f"Loading embeddings from {args.input_file}")
    embeddings_np = load_vectors_from_file(args.input_file)
    
    # Convert to torch tensor
    embeddings = torch.from_numpy(embeddings_np).float()
    
    # Create pipeline
    input_dim = embeddings.shape[1]
    
    # Initialize pipeline with target dimension
    pipeline = DimensionalDistillationPipeline(
        input_dim=input_dim,
        target_dims=[args.target_dim],
        output_dir=args.model_dir,
        cascade_strategy=args.cascade_strategy
    )
    
    # Load models
    logger.info("Loading trained models...")
    pipeline.load_models()
    
    # Distill embeddings
    logger.info(f"Distilling embeddings to dimension {args.target_dim}...")
    distilled_embeddings = pipeline.distill(
        embeddings=embeddings,
        target_dim=args.target_dim,
        batch_size=args.batch_size
    )
    
    # Save distilled embeddings
    logger.info(f"Saving distilled embeddings to {args.output_file}")
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save in the appropriate format based on file extension
    ext = output_path.suffix.lower()
    if ext == '.npy':
        np.save(output_path, distilled_embeddings.numpy())
    elif ext == '.npz':
        np.savez_compressed(output_path, embeddings=distilled_embeddings.numpy())
    else:
        # Default to .npy if extension not recognized
        np.save(f"{output_path}.npy", distilled_embeddings.numpy())
    
    logger.info("Distillation completed successfully")


def evaluate_command(args):
    """Run the evaluate command."""
    from src.distillation.pipeline import DimensionalDistillationPipeline
    from src.utils.data_loader import load_vectors_from_file
    from src.evaluation.evaluator import SearchEvaluator
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Load embeddings
    logger.info(f"Loading embeddings from {args.input_file}")
    embeddings_np = load_vectors_from_file(args.input_file)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse k values
    k_values = [int(k) for k in args.k_values.split(",")]
    
    # Create evaluator
    evaluator = SearchEvaluator(
        k_values=k_values,
        metrics=['recall', 'search_time'],
        output_dir=str(output_dir)
    )
    
    # Set ground truth
    logger.info("Computing ground truth...")
    evaluator.set_ground_truth(
        vectors=embeddings_np,
        num_queries=min(args.num_queries, len(embeddings_np))
    )
    
    # Create a function to load and distill embeddings
    def get_distilled_embeddings(dimension):
        model_dir = Path(args.model_dir)
        
        # Find the model for the given dimension
        for model_path in model_dir.glob("*to*/model.pt"):
            # Extract target dimension from directory name
            try:
                dir_name = model_path.parent.name
                if "to" in dir_name:
                    _, target_dim = map(int, dir_name.split("to"))
                    if target_dim == dimension:
                        # Found the right model, now create the pipeline
                        input_dim = embeddings_np.shape[1]
                        pipeline = DimensionalDistillationPipeline(
                            input_dim=input_dim,
                            target_dims=[dimension],
                            output_dir=str(model_dir.parent)
                        )
                        
                        # Load models
                        pipeline.load_models()
                        
                        # Distill embeddings
                        embeddings = torch.from_numpy(embeddings_np).float()
                        distilled = pipeline.distill(
                            embeddings=embeddings,
                            target_dim=dimension,
                            batch_size=args.batch_size
                        )
                        
                        return distilled.numpy()
            except Exception as e:
                logger.warning(f"Error loading model: {e}")
        
        raise ValueError(f"No model found for dimension {dimension}")
    
    # Check what dimensions are available
    model_dir = Path(args.model_dir)
    available_dims = []
    
    for model_path in model_dir.glob("*to*/model.pt"):
        try:
            dir_name = model_path.parent.name
            if "to" in dir_name:
                _, target_dim = map(int, dir_name.split("to"))
                available_dims.append(target_dim)
        except Exception:
            pass
    
    available_dims = sorted(available_dims)
    logger.info(f"Found models for dimensions: {available_dims}")
    
    # Evaluate each dimension
    for dim in available_dims:
        logger.info(f"Evaluating dimension {dim}...")
        
        # Define search function
        def search_func(query, k, vectors):
            # Simple brute force search for evaluation
            similarities = np.dot(vectors, query)
            return np.argsort(-similarities)[:k]
        
        try:
            # Get distilled embeddings
            distilled = get_distilled_embeddings(dim)
            
            # Evaluate search
            logger.info(f"Evaluating search on {dim} dimensions...")
            evaluator.evaluate_search_method(
                vectors=distilled,
                search_func=search_func,
                method_name=f"dim_{dim}",
                dimension=dim
            )
        except Exception as e:
            logger.error(f"Error evaluating dimension {dim}: {e}")
    
    # Plot results
    logger.info("Plotting results...")
    plot_path = output_dir / "dimension_comparison.png"
    evaluator.plot_recall_vs_dimension(
        output_file=str(plot_path)
    )
    
    # Print summary
    evaluator.print_summary()
    
    logger.info(f"Evaluation completed. Results saved to {output_dir}")


def generate_command(args):
    """Run the generate command."""
    from src.utils.data_loader import generate_synthetic_data
    import numpy as np
    
    logger.info(f"Generating {args.n_samples} synthetic embeddings with {args.n_features} dimensions...")
    
    # Generate synthetic data
    embeddings = generate_synthetic_data(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_clusters=args.n_clusters,
        random_state=42
    )
    
    # Save embeddings
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save in the appropriate format based on file extension
    ext = output_path.suffix.lower()
    if ext == '.npy':
        np.save(output_path, embeddings)
    elif ext == '.npz':
        np.savez_compressed(output_path, embeddings=embeddings)
    else:
        # Default to .npy if extension not recognized
        np.save(f"{output_path}.npy", embeddings)
    
    logger.info(f"Saved synthetic embeddings to {args.output_file}")


def main():
    """Main entry point."""
    parser = get_parser()
    args = parser.parse_args()
    
    if args.command == "train":
        train_command(args)
    elif args.command == "distill":
        distill_command(args)
    elif args.command == "evaluate":
        evaluate_command(args)
    elif args.command == "generate":
        generate_command(args)
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 