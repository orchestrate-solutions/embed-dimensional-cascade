#!/usr/bin/env python
"""
Analyze precision loss between different dimensions in the Dimensional Cascade.

This script loads a dataset, creates embeddings at different dimensions,
and measures the precision loss between each dimension level.
"""
import os
import argparse
import random
from typing import List, Dict, Any, Tuple
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from dimensional_cascade.models import ModelHierarchy
from dimensional_cascade.indexing import MultiResolutionIndex
from dimensional_cascade.utils.metrics import calculate_precision_loss, dimension_cascade_analysis
from dimensional_cascade.utils.io import load_jsonl, save_metrics
from dimensional_cascade.core import DimensionalCascade, CascadeConfig


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze precision loss in Dimensional Cascade')
    
    parser.add_argument('--data', type=str, required=True, 
                        help='Path to dataset (JSONL format with "text" field)')
    parser.add_argument('--queries', type=str, required=True,
                        help='Path to queries file (one query per line)')
    parser.add_argument('--model', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                        help='Path to base model or model name')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Output directory for results')
    parser.add_argument('--dimensions', type=str, default='1024,512,256,128,64,32,16',
                        help='Comma-separated list of dimensions to analyze')
    parser.add_argument('--sample-size', type=int, default=1000,
                        help='Number of documents to sample from dataset')
    parser.add_argument('--query-count', type=int, default=50,
                        help='Number of queries to use for evaluation')
    parser.add_argument('--top-k', type=int, default=100,
                        help='Number of top results to consider')
    
    return parser.parse_args()


def load_dataset(file_path: str, sample_size: int = None) -> List[Dict[str, Any]]:
    """Load and optionally sample a dataset.
    
    Args:
        file_path: Path to dataset file
        sample_size: Number of documents to sample (if None, use all)
        
    Returns:
        List of document dictionaries
    """
    documents = load_jsonl(file_path)
    
    # Ensure documents have text field
    documents = [doc for doc in documents if 'text' in doc]
    
    if sample_size and sample_size < len(documents):
        return random.sample(documents, sample_size)
    
    return documents


def load_queries(file_path: str, count: int = None) -> List[str]:
    """Load and optionally sample queries.
    
    Args:
        file_path: Path to queries file
        count: Number of queries to sample (if None, use all)
        
    Returns:
        List of query strings
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]
    
    if count and count < len(queries):
        return random.sample(queries, count)
    
    return queries


def create_cascade(
    model_path: str, 
    dimensions: List[int],
    documents: List[Dict[str, Any]],
    output_dir: str
) -> Tuple[DimensionalCascade, str]:
    """Create and index a dimensional cascade.
    
    Args:
        model_path: Path to model or model name
        dimensions: List of dimensions to use
        documents: List of documents to index
        output_dir: Output directory
        
    Returns:
        Tuple of (cascade, index_path)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    index_path = os.path.join(output_dir, 'index')
    
    # Configure the cascade
    config = CascadeConfig(dimensions=dimensions)
    
    # Create and index the cascade
    cascade = DimensionalCascade(
        model_path=model_path,
        index_path=index_path,
        config=config
    )
    
    # Index documents
    print(f"Indexing {len(documents)} documents at {len(dimensions)} dimension levels...")
    cascade.index_documents(documents, show_progress=True)
    
    # Save the cascade
    cascade.save(index_path)
    
    return cascade, index_path


def search_single_dimension(
    cascade: DimensionalCascade, 
    query: str, 
    dimension: int,
    k: int = 100
) -> List[Tuple[Dict[str, Any], float]]:
    """Search at a single dimension level.
    
    Args:
        cascade: Dimensional cascade
        query: Query text
        dimension: Dimension to search at
        k: Number of results to return
        
    Returns:
        List of (document, score) tuples
    """
    # Generate query embedding
    query_embedding = cascade.model_hierarchy.embed(query, dimension=dimension)
    
    # Search the index directly
    doc_indices, distances = cascade.index.search(query_embedding, dimension, k)
    
    # Convert distances to similarity scores
    similarity_scores = [1.0 / (1.0 + dist) for dist in distances]
    
    # Get documents
    documents = cascade.index.get_documents(doc_indices)
    
    # Return document-score pairs
    return list(zip(documents, similarity_scores))


def analyze_dimensions(
    cascade: DimensionalCascade,
    queries: List[str],
    dimensions: List[int],
    top_k: int = 100
) -> Dict[str, Any]:
    """Analyze precision loss and performance across dimensions.
    
    Args:
        cascade: Dimensional cascade
        queries: List of queries to evaluate
        dimensions: List of dimensions to analyze
        top_k: Number of top results to consider
        
    Returns:
        Dictionary with analysis results
    """
    # Define search functions
    def single_dim_search(query, dimension, k=top_k):
        return search_single_dimension(cascade, query, dimension, k)
    
    def cascade_search(query, k=top_k):
        return cascade.search(query, top_k=k)
    
    # Run analysis
    results = dimension_cascade_analysis(
        dimensions=dimensions,
        queries=queries,
        cascade_search_fn=cascade_search,
        single_dim_search_fn=single_dim_search,
        k=top_k
    )
    
    return results


def calculate_expected_precision_loss(dimensions: List[int]) -> Dict[int, float]:
    """Calculate expected precision loss based on the dimension ratio.
    
    This is a theoretical model based on the documentation.
    
    Args:
        dimensions: List of dimensions to calculate for
        
    Returns:
        Dictionary mapping from dimension to expected precision loss
    """
    max_dim = max(dimensions)
    expected_loss = {}
    
    # These values are from the documentation
    expected_loss_reference = {
        1024: 0.0,   # Reference dimension (no loss)
        512: 0.05,   # 5% loss
        256: 0.10,   # 10% loss
        128: 0.20,   # 20% loss
        64: 0.35,    # 35% loss
        32: 0.55,    # 55% loss
        16: 0.75     # 75% loss
    }
    
    # Use the reference values if available, otherwise interpolate
    for dim in dimensions:
        if dim in expected_loss_reference:
            expected_loss[dim] = expected_loss_reference[dim]
        else:
            # Simple linear interpolation
            # Find nearest reference dimensions
            lower_dim = max([d for d in expected_loss_reference.keys() if d < dim], default=None)
            upper_dim = min([d for d in expected_loss_reference.keys() if d > dim], default=None)
            
            if lower_dim and upper_dim:
                # Interpolate
                lower_loss = expected_loss_reference[lower_dim]
                upper_loss = expected_loss_reference[upper_dim]
                ratio = (dim - lower_dim) / (upper_dim - lower_dim)
                expected_loss[dim] = lower_loss + ratio * (upper_loss - lower_loss)
            elif lower_dim:
                # Extrapolate above max reference
                expected_loss[dim] = expected_loss_reference[lower_dim]
            elif upper_dim:
                # Extrapolate below min reference
                expected_loss[dim] = expected_loss_reference[upper_dim]
            else:
                # Fallback to a simple model if no reference points
                ratio = dim / max_dim
                expected_loss[dim] = 1.0 - ratio
    
    return expected_loss


def plot_results(
    results: Dict[str, Any],
    expected_loss: Dict[int, float],
    output_path: str
):
    """Plot precision loss and latency results.
    
    Args:
        results: Analysis results
        expected_loss: Expected precision loss values
        output_path: Path to save the plot
    """
    # Extract dimensions and metrics
    dimensions = sorted([dim for dim in results['dimension_metrics'].keys()])
    
    # Precision loss
    actual_loss = [results['dimension_metrics'][dim]['avg_precision_loss'] for dim in dimensions]
    expected_loss_values = [expected_loss[dim] for dim in dimensions]
    
    # Latency speedup
    latency_speedup = [results['dimension_metrics'][dim]['latency_speedup'] for dim in dimensions]
    theoretical_speedup = [max(dimensions) / dim for dim in dimensions]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Precision loss plot
    ax1.plot(dimensions, actual_loss, 'bo-', label='Actual Precision Loss')
    ax1.plot(dimensions, expected_loss_values, 'r--', label='Expected Precision Loss')
    ax1.set_xlabel('Dimension')
    ax1.set_ylabel('Precision Loss')
    ax1.set_title('Precision Loss by Dimension')
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(0, 1)
    
    # Latency speedup plot
    ax2.plot(dimensions, latency_speedup, 'go-', label='Actual Speedup')
    ax2.plot(dimensions, theoretical_speedup, 'r--', label='Theoretical Speedup')
    ax2.set_xlabel('Dimension')
    ax2.set_ylabel('Speedup Factor')
    ax2.set_title('Search Speedup by Dimension')
    ax2.legend()
    ax2.grid(True)
    
    # Cascade results if available
    if 'cascade' in results['overall']:
        cascade_loss = results['overall']['cascade']['avg_precision_loss']
        cascade_speedup = results['overall']['cascade']['latency_speedup']
        
        # Add cascade points to plots
        ax1.axhline(y=cascade_loss, color='m', linestyle='-.')
        ax1.text(min(dimensions), cascade_loss, f'Cascade: {cascade_loss:.2f}', va='bottom')
        
        ax2.axhline(y=cascade_speedup, color='m', linestyle='-.')
        ax2.text(min(dimensions), cascade_speedup, f'Cascade: {cascade_speedup:.2f}x', va='bottom')
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main():
    args = parse_args()
    
    # Parse dimensions
    dimensions = [int(dim) for dim in args.dimensions.split(',')]
    dimensions.sort(reverse=True)  # Ensure descending order
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset and queries
    print(f"Loading dataset from {args.data}...")
    documents = load_dataset(args.data, args.sample_size)
    
    print(f"Loading queries from {args.queries}...")
    queries = load_queries(args.queries, args.query_count)
    
    # Create and index the cascade
    cascade, index_path = create_cascade(
        model_path=args.model,
        dimensions=dimensions,
        documents=documents,
        output_dir=args.output_dir
    )
    
    # Analyze dimensions
    print(f"Analyzing precision loss across {len(dimensions)} dimensions...")
    results = analyze_dimensions(
        cascade=cascade,
        queries=queries,
        dimensions=dimensions,
        top_k=args.top_k
    )
    
    # Calculate expected precision loss
    expected_loss = calculate_expected_precision_loss(dimensions)
    
    # Save results
    results_path = os.path.join(args.output_dir, 'precision_analysis.json')
    save_metrics(results, results_path)
    
    # Plot results
    plot_path = os.path.join(args.output_dir, 'precision_analysis.png')
    plot_results(results, expected_loss, plot_path)
    
    print(f"Analysis complete. Results saved to {args.output_dir}")
    
    # Print summary
    print("\nSummary of Precision Loss:")
    print("-------------------------")
    print(f"{'Dimension':<10} {'Actual Loss':<15} {'Expected Loss':<15} {'Speedup':<10}")
    print("-" * 50)
    
    for dim in sorted(dimensions[1:]):  # Skip highest dimension (reference)
        actual = results['dimension_metrics'][dim]['avg_precision_loss']
        expected = expected_loss[dim]
        speedup = results['dimension_metrics'][dim]['latency_speedup']
        print(f"{dim:<10} {actual:.4f} {'':5} {expected:.4f} {'':5} {speedup:.2f}x")
    
    if 'cascade' in results['overall']:
        print("-" * 50)
        cascade_loss = results['overall']['cascade']['avg_precision_loss']
        cascade_speedup = results['overall']['cascade']['latency_speedup']
        print(f"{'Cascade':<10} {cascade_loss:.4f} {'':5} {'N/A':<15} {cascade_speedup:.2f}x")


if __name__ == '__main__':
    main() 