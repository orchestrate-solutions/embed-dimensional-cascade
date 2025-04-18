"""
Evaluation utilities for measuring search performance.

This module provides functions to measure the performance of search methods
in terms of recall, time, and memory usage.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Callable, Optional
import logging
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)

def get_ground_truth(
    vectors: np.ndarray,
    query_vectors: np.ndarray,
    k: int,
    metric: str = "cosine"
) -> List[List[int]]:
    """
    Compute ground truth nearest neighbors for each query vector.
    
    Args:
        vectors: Document vectors to search in, shape (n_docs, n_dims)
        query_vectors: Query vectors, shape (n_queries, n_dims)
        k: Number of nearest neighbors to retrieve
        metric: Distance metric to use
        
    Returns:
        List of lists containing the indices of the k nearest neighbors
        for each query vector.
    """
    logger.info(f"Computing ground truth for {len(query_vectors)} queries (k={k})...")
    
    # Handle case where k is larger than number of vectors
    k_adjusted = min(k, len(vectors) - 1)
    if k_adjusted < k:
        logger.warning(f"Requested k={k} but only {k_adjusted} vectors available")
    
    # Create nearest neighbors index
    nn = NearestNeighbors(n_neighbors=k_adjusted, algorithm='brute', metric=metric)
    nn.fit(vectors)
    
    # Get k nearest neighbors for each query
    _, indices = nn.kneighbors(query_vectors)
    
    return indices.tolist()

def evaluate_recall(results: List[List[int]], ground_truth: List[List[int]]) -> float:
    """
    Calculate recall@k between search results and ground truth.
    
    Args:
        results: List of lists containing search result indices
        ground_truth: List of lists containing ground truth indices
        
    Returns:
        Average recall@k across all queries
    """
    recalls = []
    
    for res, gt in zip(results, ground_truth):
        # Convert to sets for intersection
        res_set = set(res)
        gt_set = set(gt)
        
        # Calculate recall
        if len(gt_set) > 0:
            recall = len(res_set.intersection(gt_set)) / len(gt_set)
            recalls.append(recall)
    
    return np.mean(recalls) if recalls else 0.0

def evaluate_search(
    search_function: Callable,
    query_vectors: np.ndarray,
    ground_truth: List[List[int]],
    k: int,
    n_trials: int = 3,
    verbose: bool = False
) -> Dict[str, float]:
    """
    Evaluate a search function for recall and time.
    
    Args:
        search_function: Function that takes (query, k) and returns indices
        query_vectors: Query vectors to search with
        ground_truth: Ground truth nearest neighbor indices
        k: Number of nearest neighbors to retrieve
        n_trials: Number of trials to average time over
        verbose: Whether to log progress
        
    Returns:
        Dictionary with evaluation metrics
    """
    n_queries = len(query_vectors)
    results = []
    query_times = []
    
    # Evaluate search performance
    iterable = tqdm(range(n_queries), desc="Evaluating queries") if verbose else range(n_queries)
    
    for i in iterable:
        query = query_vectors[i]
        
        # Measure search time (average over multiple trials)
        times = []
        for _ in range(n_trials):
            start_time = time.time()
            indices = search_function(query, k)
            times.append(time.time() - start_time)
        
        # Store results
        query_times.append(np.mean(times))
        results.append(indices)
    
    # Calculate metrics
    recall = evaluate_recall(results, ground_truth)
    avg_time = np.mean(query_times)
    
    return {
        "avg_recall": recall,
        "avg_time": avg_time,
        "min_time": np.min(query_times),
        "max_time": np.max(query_times),
        "std_time": np.std(query_times),
        "n_queries": n_queries,
    }

def compare_search_methods(
    methods: Dict[str, Callable],
    query_vectors: np.ndarray,
    ground_truth: List[List[int]],
    k: int,
    n_trials: int = 3,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple search methods in terms of recall and time.
    
    Args:
        methods: Dictionary mapping method names to search functions
        query_vectors: Query vectors to search with
        ground_truth: Ground truth nearest neighbor indices
        k: Number of nearest neighbors to retrieve
        n_trials: Number of trials to average time over
        verbose: Whether to log progress
        
    Returns:
        Dictionary mapping method names to evaluation metrics
    """
    results = {}
    
    for name, search_function in methods.items():
        if verbose:
            logger.info(f"Evaluating {name}...")
        
        metrics = evaluate_search(
            search_function=search_function,
            query_vectors=query_vectors,
            ground_truth=ground_truth,
            k=k,
            n_trials=n_trials,
            verbose=verbose
        )
        
        results[name] = metrics
        
        if verbose:
            logger.info(f"{name}: Recall={metrics['avg_recall']:.4f}, Time={metrics['avg_time']*1000:.2f}ms")
    
    return results

def evaluate_dimension_impact(
    create_search_function: Callable[[List[int]], Callable],
    dimensions: List[int],
    query_vectors: np.ndarray,
    ground_truth: List[List[int]],
    k: int,
    n_trials: int = 3,
    verbose: bool = True
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate the impact of different dimension configurations on performance.
    
    Args:
        create_search_function: Function to create a search function from dimensions
        dimensions: List of dimensions to evaluate
        query_vectors: Query vectors to search with
        ground_truth: Ground truth nearest neighbor indices
        k: Number of nearest neighbors to retrieve
        n_trials: Number of trials to average time over
        verbose: Whether to log progress
        
    Returns:
        Dictionary mapping dimension configurations to evaluation metrics
    """
    methods = {}
    
    # Single dimension configurations
    for dim in dimensions:
        methods[f"Flat {dim}D"] = create_search_function([dim])
    
    # Multi-dimension configurations
    for i in range(2, len(dimensions) + 1):
        config = dimensions[:i]
        config_name = "->".join([str(d) for d in config])
        methods[f"Cascade {config_name}"] = create_search_function(config)
    
    return compare_search_methods(
        methods=methods,
        query_vectors=query_vectors,
        ground_truth=ground_truth,
        k=k,
        n_trials=n_trials,
        verbose=verbose
    )

def plot_recall_time_tradeoff(
    results: Dict[str, Dict[str, float]],
    title: str = "Recall-Time Tradeoff",
    save_path: Optional[str] = None
) -> None:
    """
    Plot recall vs. time tradeoff for different search methods.
    
    Args:
        results: Dictionary mapping method names to evaluation metrics
        title: Plot title
        save_path: Path to save the plot (if None, plot is displayed)
    """
    plt.figure(figsize=(10, 6))
    
    # Extract cascade and flat methods
    cascade_methods = {name: metrics for name, metrics in results.items() if "Cascade" in name}
    flat_methods = {name: metrics for name, metrics in results.items() if "Flat" in name}
    other_methods = {name: metrics for name, metrics in results.items() 
                     if "Cascade" not in name and "Flat" not in name}
    
    # Plot cascade methods
    for name, metrics in cascade_methods.items():
        plt.scatter(
            metrics["avg_time"] * 1000,  # Convert to ms
            metrics["avg_recall"],
            s=100,
            marker="o",
            alpha=0.7,
            label=name
        )
    
    # Plot flat methods
    for name, metrics in flat_methods.items():
        plt.scatter(
            metrics["avg_time"] * 1000,  # Convert to ms
            metrics["avg_recall"],
            s=100,
            marker="s",
            alpha=0.7,
            label=name
        )
    
    # Plot other methods
    for name, metrics in other_methods.items():
        plt.scatter(
            metrics["avg_time"] * 1000,  # Convert to ms
            metrics["avg_recall"],
            s=100,
            marker="^",
            alpha=0.7,
            label=name
        )
    
    plt.xlabel("Average Query Time (ms)")
    plt.ylabel("Recall@k")
    plt.title(title)
    plt.grid(linestyle="--", alpha=0.7)
    plt.legend(loc="lower right")
    
    # Add text annotations for each point
    for name, metrics in results.items():
        plt.annotate(
            name,
            (metrics["avg_time"] * 1000, metrics["avg_recall"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8
        )
    
    # Save or show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close() 