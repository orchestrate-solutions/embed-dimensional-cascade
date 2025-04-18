#!/usr/bin/env python3
"""
Analyze Precision Loss in Dimensional Cascade

This script evaluates how much precision is lost when reducing dimensions 
in vector embeddings, which is the core principle behind dimensional cascade.
It helps to determine optimal dimensionality reduction steps that balance
performance and accuracy.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from typing import List, Dict, Tuple
import time
import json

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from dimensional_cascade.utils.metrics import (
    evaluate_dimension_performance,
    calculate_cascade_metrics,
    time_function
)
from dimensional_cascade.index import MultiResolutionIndex
from dimensional_cascade.core import DimensionalCascade

# This will be used for synthetic data if no real data is provided
try:
    from sentence_transformers import SentenceTransformer
    HAVE_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAVE_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence_transformers not found. Synthetic data will be used for embeddings.")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Analyze precision loss in dimensional cascade')
    
    parser.add_argument(
        '--dims', 
        type=int, 
        nargs='+',
        default=[768, 384, 192, 96, 48, 24],
        help='Dimensions to evaluate, from highest to lowest'
    )
    
    parser.add_argument(
        '--num-samples', 
        type=int, 
        default=1000,
        help='Number of data samples to use'
    )
    
    parser.add_argument(
        '--num-queries', 
        type=int, 
        default=50,
        help='Number of queries to test'
    )
    
    parser.add_argument(
        '--k', 
        type=int, 
        default=10,
        help='Number of results to consider for precision/recall metrics'
    )
    
    parser.add_argument(
        '--data-path', 
        type=str, 
        default=None,
        help='Path to numpy file with precomputed embeddings'
    )
    
    parser.add_argument(
        '--query-path', 
        type=str, 
        default=None,
        help='Path to text file with query strings (one per line)'
    )
    
    parser.add_argument(
        '--model-name', 
        type=str, 
        default='all-MiniLM-L6-v2',
        help='SentenceTransformer model name to use for generating embeddings'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='precision_analysis.json',
        help='Path to save analysis results'
    )
    
    parser.add_argument(
        '--plot', 
        action='store_true',
        help='Generate and save plots'
    )
    
    parser.add_argument(
        '--cascade-factor', 
        type=int, 
        default=10,
        help='Multiplicative factor for cascade search'
    )
    
    return parser.parse_args()


def generate_synthetic_data(num_samples: int, highest_dim: int) -> np.ndarray:
    """Generate synthetic embedding data."""
    print(f"Generating {num_samples} synthetic embeddings with dimension {highest_dim}")
    embeddings = np.random.randn(num_samples, highest_dim)
    # Normalize to unit length
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / norms
    return embeddings


def generate_synthetic_queries(num_queries: int, highest_dim: int) -> List[np.ndarray]:
    """Generate synthetic query embeddings."""
    print(f"Generating {num_queries} synthetic query embeddings with dimension {highest_dim}")
    query_embeddings = np.random.randn(num_queries, highest_dim)
    # Normalize to unit length
    norms = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    query_embeddings = query_embeddings / norms
    return [query_embeddings[i] for i in range(num_queries)]


def load_or_generate_data(args) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Load data from files or generate synthetic data."""
    highest_dim = args.dims[0]
    
    # Load or generate embeddings
    if args.data_path and os.path.exists(args.data_path):
        print(f"Loading embeddings from {args.data_path}")
        embeddings = np.load(args.data_path)
        if embeddings.shape[1] < highest_dim:
            print(f"Warning: Loaded embeddings have dimension {embeddings.shape[1]}, "
                  f"which is less than requested highest dimension {highest_dim}")
            # Adjust dimensions list to match the data
            args.dims = [d for d in args.dims if d <= embeddings.shape[1]]
            highest_dim = args.dims[0]
        elif embeddings.shape[1] > highest_dim:
            print(f"Truncating loaded embeddings from {embeddings.shape[1]} to {highest_dim}")
            embeddings = embeddings[:, :highest_dim]
    else:
        if args.data_path:
            print(f"Warning: Data path {args.data_path} not found.")
        
        if HAVE_SENTENCE_TRANSFORMERS:
            # Generate real embeddings using SentenceTransformer
            print(f"Generating real embeddings using model {args.model_name}")
            model = SentenceTransformer(args.model_name)
            
            # Generate some random text for embedding
            sentences = [
                f"This is a sample document {i} for testing dimensional cascade."
                for i in range(args.num_samples)
            ]
            
            # Generate embeddings
            embeddings = model.encode(sentences, convert_to_numpy=True)
            
            # Adjust dimensions list to match the model output
            if embeddings.shape[1] != highest_dim:
                print(f"Model produced embeddings with dimension {embeddings.shape[1]}, "
                      f"adjusting from requested {highest_dim}")
                args.dims = [d for d in args.dims if d <= embeddings.shape[1]]
                args.dims.insert(0, embeddings.shape[1])
                highest_dim = args.dims[0]
        else:
            # Generate synthetic embeddings
            embeddings = generate_synthetic_data(args.num_samples, highest_dim)
    
    # Ensure we only use the requested number of samples
    if len(embeddings) > args.num_samples:
        print(f"Using {args.num_samples} samples out of {len(embeddings)} available")
        embeddings = embeddings[:args.num_samples]
    
    # Load or generate query embeddings
    if args.query_path and os.path.exists(args.query_path):
        print(f"Loading queries from {args.query_path}")
        with open(args.query_path, 'r') as f:
            query_texts = [line.strip() for line in f if line.strip()][:args.num_queries]
        
        if HAVE_SENTENCE_TRANSFORMERS:
            model = SentenceTransformer(args.model_name)
            query_embeddings = [model.encode(q, convert_to_numpy=True) for q in query_texts]
            
            # Ensure dimensions match
            query_embeddings = [q[:highest_dim] for q in query_embeddings]
        else:
            print("Warning: sentence_transformers not found but query text provided.")
            print("Generating synthetic query embeddings instead.")
            query_embeddings = generate_synthetic_queries(len(query_texts), highest_dim)
    else:
        if args.query_path:
            print(f"Warning: Query path {args.query_path} not found.")
        
        if HAVE_SENTENCE_TRANSFORMERS:
            # Generate some random queries
            query_texts = [
                f"Query {i} for testing dimensional cascade search"
                for i in range(args.num_queries)
            ]
            
            model = SentenceTransformer(args.model_name)
            query_embeddings = [model.encode(q, convert_to_numpy=True) for q in query_texts]
            
            # Ensure dimensions match
            query_embeddings = [q[:highest_dim] for q in query_embeddings]
        else:
            # Generate synthetic query embeddings
            query_embeddings = generate_synthetic_queries(args.num_queries, highest_dim)
    
    return embeddings, query_embeddings


def create_search_function(embeddings: np.ndarray, dimensions: List[int]):
    """Create a function for searching at different dimensions."""
    # Create multi-resolution index for each dimension
    indices = {
        dim: MultiResolutionIndex(dimension=dim) for dim in dimensions
    }
    
    # Fit indices with embeddings
    for dim, index in indices.items():
        # Use time_function to measure fitting time
        _, fit_time = time_function(
            index.fit, 
            embeddings[:, :dim] if dim < embeddings.shape[1] else embeddings
        )
        print(f"Fitted index for dimension {dim} in {fit_time:.2f} seconds")
        
    def search_function(query_embedding, dimension, k):
        """Search function for a specific dimension."""
        index = indices[dimension]
        query_vec = query_embedding[:dimension] if dimension < len(query_embedding) else query_embedding
        
        # Use time_function to measure search time
        results, search_time = time_function(index.search, query_vec, k)
        
        # Return search results and time
        return results, search_time
    
    return search_function


def create_cascade_search_function(embeddings: np.ndarray, dimensions: List[int], cascade_factor: int):
    """Create a function for cascade search."""
    # Create dimensional cascade
    cascade = DimensionalCascade(dimensions=dimensions, cascade_factor=cascade_factor)
    
    # Fit cascade with embeddings
    _, fit_time = time_function(cascade.fit, embeddings)
    print(f"Fitted cascade in {fit_time:.2f} seconds")
    
    def cascade_search_function(query_embedding, k):
        """Cascade search function."""
        # Use time_function to measure search time
        results, search_time = time_function(cascade.search, query_embedding, k)
        
        # Return search results and time
        return results, search_time
    
    return cascade_search_function


def plot_results(results, dimensions, output_prefix):
    """Generate and save plots from the results."""
    # Prepare data for plotting
    dims = [d for d in dimensions if d != dimensions[0]]  # Skip highest dimension (ground truth)
    
    # Extract metrics
    precision_loss = [results[d]["avg_precision_loss"] for d in dims]
    recall_loss = [results[d]["avg_recall_loss"] for d in dims]
    time_improvement = [results[d]["time_improvement"] for d in dims]
    
    # Create dataframe for easier plotting
    df = pd.DataFrame({
        "Dimension": dims,
        "Precision Loss": precision_loss,
        "Recall Loss": recall_loss,
        "Time Improvement Factor": time_improvement
    })
    
    # Plot precision and recall loss
    plt.figure(figsize=(10, 6))
    plt.plot(df["Dimension"], df["Precision Loss"], 'o-', label="Precision Loss")
    plt.plot(df["Dimension"], df["Recall Loss"], 's-', label="Recall Loss")
    plt.xlabel("Dimension")
    plt.ylabel("Loss (0.0-1.0)")
    plt.title("Precision and Recall Loss by Dimension")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_loss.png")
    
    # Plot time improvement
    plt.figure(figsize=(10, 6))
    plt.plot(df["Dimension"], df["Time Improvement Factor"], 'o-')
    plt.xlabel("Dimension")
    plt.ylabel("Speed Improvement Factor")
    plt.title("Search Speed Improvement by Dimension")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_speed.png")
    
    # Plot trade-off: precision loss vs. time improvement
    plt.figure(figsize=(10, 6))
    plt.scatter(df["Precision Loss"], df["Time Improvement Factor"], s=100)
    
    # Add dimension labels to points
    for i, dim in enumerate(dims):
        plt.annotate(str(dim), (df["Precision Loss"][i], df["Time Improvement Factor"][i]),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.xlabel("Precision Loss")
    plt.ylabel("Speed Improvement Factor")
    plt.title("Trade-off: Precision vs. Speed")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_tradeoff.png")
    
    print(f"Plots saved with prefix {output_prefix}")


def main():
    args = parse_arguments()
    
    # Load or generate data
    embeddings, query_embeddings = load_or_generate_data(args)
    
    print(f"Working with {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    print(f"Testing {len(query_embeddings)} queries")
    print(f"Evaluating dimensions: {args.dims}")
    
    # Create search functions
    search_function = create_search_function(embeddings, args.dims)
    cascade_search_function = create_cascade_search_function(
        embeddings, args.dims, args.cascade_factor
    )
    
    # Evaluate dimension performance
    print("Evaluating dimension performance...")
    dimension_results = evaluate_dimension_performance(
        args.dims, query_embeddings, search_function, args.k
    )
    
    # Calculate cascade metrics
    print("Calculating cascade metrics...")
    cascade_results = calculate_cascade_metrics(
        query_embeddings, args.dims, search_function, cascade_search_function, args.k
    )
    
    # Combine results
    results = {
        "dimensions": args.dims,
        "num_samples": len(embeddings),
        "num_queries": len(query_embeddings),
        "k": args.k,
        "cascade_factor": args.cascade_factor,
        "dimension_results": dimension_results,
        "cascade_results": cascade_results
    }
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Analysis results saved to {args.output}")
    
    # Generate plots if requested
    if args.plot:
        output_prefix = os.path.splitext(args.output)[0]
        plot_results(dimension_results, args.dims, output_prefix)
    
    # Print summary
    print("\nSummary of results:")
    
    # Dimension reduction performance
    print("\nDimension reduction performance:")
    for dim in args.dims[1:]:  # Skip highest dimension (ground truth)
        metrics = dimension_results[dim]
        print(f"  Dimension {dim}:")
        print(f"    Average precision loss: {metrics['avg_precision_loss']:.4f}")
        print(f"    Average recall loss: {metrics['avg_recall_loss']:.4f}")
        print(f"    Average search time: {metrics['avg_search_time']:.6f} seconds")
        print(f"    Time improvement factor: {metrics['time_improvement']:.2f}x")
    
    # Cascade performance
    print("\nCascade performance:")
    print(f"  Precision vs. highest dim: {cascade_results['avg_precision_vs_highest']:.4f}")
    print(f"  Recall vs. highest dim: {cascade_results['avg_recall_vs_highest']:.4f}")
    print(f"  Time speedup vs. highest dim: {cascade_results['avg_time_speedup_vs_highest']:.2f}x")
    print(f"  Time slowdown vs. lowest dim: {cascade_results['avg_time_slowdown_vs_lowest']:.2f}x")


if __name__ == "__main__":
    main() 