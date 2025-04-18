"""
Benchmark script for dimensional cascade search.

This script evaluates different search approaches and dimension configurations
on various datasets to measure performance in terms of recall and query time.
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import logging
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_and_reduce, generate_synthetic_data
from utils.evaluation import get_ground_truth, compare_search_methods, evaluate_dimension_impact, plot_recall_time_tradeoff
from cascade.cascade_search import CascadeSearch

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def setup_parser():
    """Set up command line argument parser."""
    parser = argparse.ArgumentParser(description="Benchmark Dimensional Cascade Search")
    
    # Data options
    parser.add_argument("--vectors", type=str, help="Path to vectors file (if not using synthetic data)")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data")
    parser.add_argument("--n_docs", type=int, default=10000, help="Number of documents for synthetic data")
    parser.add_argument("--n_queries", type=int, default=100, help="Number of queries for synthetic data")
    parser.add_argument("--n_features", type=int, default=768, help="Number of features for synthetic data")
    
    # Dimension options
    parser.add_argument("--dims", type=str, default="32,64,128,256", help="Comma-separated dimensions to evaluate")
    parser.add_argument("--config_file", type=str, help="JSON file with dimension configurations to evaluate")
    
    # Search options
    parser.add_argument("--k", type=int, default=10, help="Number of nearest neighbors")
    parser.add_argument("--filter_ratio", type=int, default=5, help="Filter ratio for cascade search")
    parser.add_argument("--metric", type=str, default="cosine", help="Distance metric to use")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    parser.add_argument("--save_results", action="store_true", help="Save results to CSV")
    
    return parser

def load_or_generate_data(args):
    """Load vectors from file or generate synthetic data."""
    dimensions = [int(d) for d in args.dims.split(",")]
    
    if args.vectors:
        # Load vectors from file
        logger.info(f"Loading vectors from {args.vectors}...")
        doc_vectors_dict = load_and_reduce(
            filepath=args.vectors,
            dimensions=dimensions,
            cache_dir=os.path.join(args.output_dir, "cache")
        )
        
        # Generate synthetic queries
        logger.info(f"Generating {args.n_queries} synthetic query vectors...")
        original_dim = max(doc_vectors_dict.keys())
        
        # Use a small subset of the documents as queries
        doc_vectors = doc_vectors_dict[original_dim]
        n_docs = len(doc_vectors)
        n_queries = min(args.n_queries, n_docs // 10)
        
        query_indices = np.random.choice(n_docs, size=n_queries, replace=False)
        query_vectors_dict = {}
        
        for dim in dimensions:
            query_vectors_dict[dim] = doc_vectors_dict[dim][query_indices]
    else:
        # Generate synthetic data
        logger.info(f"Generating synthetic data with {args.n_docs} documents and {args.n_queries} queries...")
        doc_vectors_dict = generate_synthetic_data(
            n_samples=args.n_docs,
            n_features=args.n_features,
            dimensions=dimensions
        )
        
        query_vectors_dict = generate_synthetic_data(
            n_samples=args.n_queries,
            n_features=args.n_features,
            dimensions=dimensions
        )
    
    return doc_vectors_dict, query_vectors_dict

def load_dimension_configs(args):
    """Load dimension configurations to evaluate."""
    dimensions = [int(d) for d in args.dims.split(",")]
    
    if args.config_file:
        # Load configurations from JSON file
        with open(args.config_file, 'r') as f:
            configs = json.load(f)
        
        # Validate configurations
        for i, config in enumerate(configs):
            if not isinstance(config, list):
                raise ValueError(f"Configuration {i} must be a list of dimensions")
            if not all(isinstance(d, int) for d in config):
                raise ValueError(f"Configuration {i} must contain integers only")
            if not all(d > 0 for d in config):
                raise ValueError(f"Configuration {i} must contain positive dimensions")
            if not all(d1 < d2 for d1, d2 in zip(config[:-1], config[1:])):
                raise ValueError(f"Configuration {i} must have increasing dimensions")
    else:
        # Generate default configurations
        configs = []
        
        # Single dimension configurations
        for dim in dimensions:
            configs.append([dim])
        
        # Multi-dimension configurations
        for i in range(2, len(dimensions) + 1):
            configs.append(dimensions[:i])
    
    return configs

def create_search_methods(doc_vectors_dict, query_vectors_dict, dimension_configs, args):
    """Create search method functions to compare."""
    methods = {}
    
    # Get highest dimension for brute force baseline
    max_dim = max(doc_vectors_dict.keys())
    
    # Brute force baseline
    def brute_force_search(query, k):
        from sklearn.neighbors import NearestNeighbors
        vectors = doc_vectors_dict[max_dim]
        nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric=args.metric)
        nn.fit(vectors)
        _, indices = nn.kneighbors(query.reshape(1, -1))
        return indices[0]
    
    methods["Brute Force"] = brute_force_search
    
    # Create factory for cascade search with different configs
    def create_cascade_search(config):
        cascade = CascadeSearch(
            document_vectors={dim: doc_vectors_dict[dim] for dim in config},
            dimensions=config,
            metric=args.metric,
            algorithm='brute'
        )
        
        def search_func(query, k):
            query_dict = {dim: query for dim in config}
            return cascade.search(query_dict, k=k, filter_ratio=args.filter_ratio)
        
        return search_func
    
    # Add cascade search methods for different configs
    for config in dimension_configs:
        config_name = "->".join([str(d) for d in config])
        if len(config) == 1:
            method_name = f"Flat {config[0]}D"
        else:
            method_name = f"Cascade {config_name}"
        
        methods[method_name] = create_cascade_search(config)
    
    return methods

def run_benchmark(args):
    """Run the benchmark."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load or generate data
    doc_vectors_dict, query_vectors_dict = load_or_generate_data(args)
    
    # Load dimension configurations
    dimension_configs = load_dimension_configs(args)
    
    # Create search methods
    methods = create_search_methods(doc_vectors_dict, query_vectors_dict, dimension_configs, args)
    
    # Get highest dimension for ground truth
    max_dim = max(doc_vectors_dict.keys())
    
    # Compute ground truth
    ground_truth = get_ground_truth(
        vectors=doc_vectors_dict[max_dim],
        query_vectors=query_vectors_dict[max_dim],
        k=args.k,
        metric=args.metric
    )
    
    # Evaluate methods
    results = compare_search_methods(
        methods=methods,
        query_vectors=query_vectors_dict[max_dim],
        ground_truth=ground_truth,
        k=args.k,
        verbose=True
    )
    
    # Generate timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save results
    if args.save_results:
        results_df = pd.DataFrame({
            "Method": [],
            "Recall": [],
            "Time (ms)": [],
            "Dimensions": []
        })
        
        for name, metrics in results.items():
            results_df = results_df.append({
                "Method": name,
                "Recall": metrics["avg_recall"],
                "Time (ms)": metrics["avg_time"] * 1000,
                "Dimensions": name.split(" ")[-1] if " " in name else ""
            }, ignore_index=True)
        
        results_path = os.path.join(args.output_dir, f"benchmark_results_{timestamp}.csv")
        results_df.to_csv(results_path, index=False)
        logger.info(f"Results saved to {results_path}")
    
    # Generate plots
    if args.plot:
        # Recall-time tradeoff
        plot_path = os.path.join(args.output_dir, f"recall_time_tradeoff_{timestamp}.png")
        plot_recall_time_tradeoff(
            results=results,
            title=f"Recall-Time Tradeoff (k={args.k})",
            save_path=plot_path
        )
        
        # Dimension impact
        plt.figure(figsize=(12, 8))
        
        # Extract cascade results
        cascade_results = {name: metrics for name, metrics in results.items() if "Cascade" in name}
        flat_results = {name: metrics for name, metrics in results.items() if "Flat" in name}
        
        # Sort by increasing recall
        cascade_items = sorted(cascade_results.items(), key=lambda x: x[1]["avg_recall"])
        flat_items = sorted(flat_results.items(), key=lambda x: x[1]["avg_recall"])
        
        cascade_names = [name for name, _ in cascade_items]
        cascade_recalls = [metrics["avg_recall"] for _, metrics in cascade_items]
        cascade_times = [metrics["avg_time"] * 1000 for _, metrics in cascade_items]
        
        flat_names = [name for name, _ in flat_items]
        flat_recalls = [metrics["avg_recall"] for _, metrics in flat_items]
        flat_times = [metrics["avg_time"] * 1000 for _, metrics in flat_items]
        
        # Plot dimension impact
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 1, 1)
        plt.bar(cascade_names, cascade_recalls, alpha=0.7)
        plt.xlabel("Dimension Configuration")
        plt.ylabel("Recall@k")
        plt.title("Impact of Dimension Configuration on Recall")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        plt.subplot(2, 1, 2)
        plt.bar(cascade_names, cascade_times, alpha=0.7)
        plt.xlabel("Dimension Configuration")
        plt.ylabel("Average Query Time (ms)")
        plt.title("Impact of Dimension Configuration on Query Time")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        
        # Save plot
        dim_plot_path = os.path.join(args.output_dir, f"dimension_impact_{timestamp}.png")
        plt.savefig(dim_plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Dimension impact plot saved to {dim_plot_path}")
        
        # Flat vs cascade for same final dimension
        plt.figure(figsize=(10, 6))
        
        # Group results by final dimension
        final_dimensions = {}
        for name, metrics in results.items():
            if "Cascade" in name:
                final_dim = int(name.split("->")[-1])
                if final_dim not in final_dimensions:
                    final_dimensions[final_dim] = []
                final_dimensions[final_dim].append((name, metrics))
            elif "Flat" in name:
                final_dim = int(name.split(" ")[-1][:-1])  # Remove 'D' suffix
                if final_dim not in final_dimensions:
                    final_dimensions[final_dim] = []
                final_dimensions[final_dim].append((name, metrics))
        
        # Plot speedup for each final dimension
        dims = sorted(final_dimensions.keys())
        speedups = []
        
        for dim in dims:
            methods = final_dimensions[dim]
            flat_time = None
            cascade_time = None
            
            for name, metrics in methods:
                if "Flat" in name:
                    flat_time = metrics["avg_time"]
                elif "Cascade" in name:
                    # Take the fastest cascade configuration
                    if cascade_time is None or metrics["avg_time"] < cascade_time:
                        cascade_time = metrics["avg_time"]
            
            if flat_time is not None and cascade_time is not None:
                speedup = flat_time / cascade_time
                speedups.append(speedup)
            else:
                speedups.append(0)
        
        plt.bar(dims, speedups, alpha=0.7)
        plt.xlabel("Final Dimension")
        plt.ylabel("Speedup (Flat / Cascade)")
        plt.title("Speedup of Cascade Search vs. Flat Search")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add text labels
        for i, (dim, speedup) in enumerate(zip(dims, speedups)):
            plt.text(i, speedup + 0.1, f"{speedup:.2f}x", ha='center')
        
        # Save plot
        speedup_plot_path = os.path.join(args.output_dir, f"speedup_{timestamp}.png")
        plt.savefig(speedup_plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Speedup plot saved to {speedup_plot_path}")
    
    return results

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    
    run_benchmark(args) 