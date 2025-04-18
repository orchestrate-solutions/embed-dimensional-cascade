"""
Metrics Module for Dimensional Cascade

This module provides metrics to evaluate the performance and precision of dimensional cascade
models. It includes functions to measure precision loss, recall, and other relevant metrics
for comparing embedding performance across dimensions.
"""
import time
import numpy as np
from typing import List, Dict, Tuple, Callable, Any, Union, Optional
import logging
from functools import wraps

logger = logging.getLogger(__name__)

def time_function(func):
    """
    Decorator to measure execution time of a function.
    
    Args:
        func: The function to measure
        
    Returns:
        Wrapped function that logs execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.debug(f"Function {func.__name__} took {elapsed_time:.6f} seconds to run")
        return result, elapsed_time
    
    return wrapper


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity score (-1 to 1)
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return np.dot(a, b) / (norm_a * norm_b)


def batch_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity between batches of vectors.
    
    Args:
        a: First batch of vectors (n x dim)
        b: Second batch of vectors (m x dim)
        
    Returns:
        Similarity matrix (n x m)
    """
    # Normalize vectors
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    
    # Avoid division by zero
    a_norm[a_norm == 0] = 1
    b_norm[b_norm == 0] = 1
    
    a_normalized = a / a_norm
    b_normalized = b / b_norm
    
    # Compute similarity matrix
    sim_matrix = np.dot(a_normalized, b_normalized.T)
    
    return sim_matrix


def compute_recall_at_k(query_results: List[List[int]], 
                        ground_truth: List[List[int]], 
                        k: int) -> float:
    """
    Compute recall@k for search results.
    
    Args:
        query_results: List of lists containing indices of search results for each query
        ground_truth: List of lists containing indices of ground truth results for each query
        k: k value for recall@k
        
    Returns:
        recall@k value (0 to 1)
    """
    if not query_results or not ground_truth:
        return 0.0
    
    if len(query_results) != len(ground_truth):
        raise ValueError("Query results and ground truth must have the same length")
    
    recalls = []
    
    for results, truth in zip(query_results, ground_truth):
        results_at_k = set(results[:k])
        truth_set = set(truth)
        
        if not truth_set:
            continue
        
        recall = len(results_at_k.intersection(truth_set)) / len(truth_set)
        recalls.append(recall)
    
    if not recalls:
        return 0.0
    
    return sum(recalls) / len(recalls)


def compute_precision_at_k(query_results: List[List[int]], 
                          ground_truth: List[List[int]], 
                          k: int) -> float:
    """
    Compute precision@k for search results.
    
    Args:
        query_results: List of lists containing indices of search results for each query
        ground_truth: List of lists containing indices of ground truth results for each query
        k: k value for precision@k
        
    Returns:
        precision@k value (0 to 1)
    """
    if not query_results or not ground_truth:
        return 0.0
    
    if len(query_results) != len(ground_truth):
        raise ValueError("Query results and ground truth must have the same length")
    
    precisions = []
    
    for results, truth in zip(query_results, ground_truth):
        results_at_k = results[:k]
        truth_set = set(truth)
        
        if not results_at_k:
            precisions.append(0.0)
            continue
        
        precision = sum(1 for r in results_at_k if r in truth_set) / len(results_at_k)
        precisions.append(precision)
    
    if not precisions:
        return 0.0
    
    return sum(precisions) / len(precisions)


def mean_average_precision(query_results: List[List[int]], 
                          ground_truth: List[List[int]], 
                          k: int = None) -> float:
    """
    Compute Mean Average Precision (MAP) for search results.
    
    Args:
        query_results: List of lists containing indices of search results for each query
        ground_truth: List of lists containing indices of ground truth results for each query
        k: Optional limit for result consideration
        
    Returns:
        MAP value (0 to 1)
    """
    if not query_results or not ground_truth:
        return 0.0
    
    if len(query_results) != len(ground_truth):
        raise ValueError("Query results and ground truth must have the same length")
    
    aps = []
    
    for results, truth in zip(query_results, ground_truth):
        results_to_consider = results if k is None else results[:k]
        truth_set = set(truth)
        
        if not truth_set:
            continue
        
        precision_sum = 0.0
        num_relevant = 0
        
        for i, result in enumerate(results_to_consider):
            if result in truth_set:
                num_relevant += 1
                precision_at_i = num_relevant / (i + 1)
                precision_sum += precision_at_i
        
        if num_relevant == 0:
            ap = 0.0
        else:
            ap = precision_sum / len(truth_set)
        
        aps.append(ap)
    
    if not aps:
        return 0.0
    
    return sum(aps) / len(aps)


def normalized_discounted_cumulative_gain(query_results: List[List[int]], 
                                         ground_truth: List[List[int]], 
                                         k: int = None) -> float:
    """
    Compute Normalized Discounted Cumulative Gain (NDCG) for search results.
    
    Args:
        query_results: List of lists containing indices of search results for each query
        ground_truth: List of lists containing indices of ground truth results for each query
        k: Optional limit for result consideration
        
    Returns:
        NDCG value (0 to 1)
    """
    if not query_results or not ground_truth:
        return 0.0
    
    if len(query_results) != len(ground_truth):
        raise ValueError("Query results and ground truth must have the same length")
    
    ndcgs = []
    
    for results, truth in zip(query_results, ground_truth):
        results_to_consider = results if k is None else results[:k]
        truth_set = set(truth)
        
        if not truth_set:
            continue
        
        # DCG
        dcg = 0.0
        for i, result in enumerate(results_to_consider):
            if result in truth_set:
                # Using binary relevance (1 if relevant, 0 if not)
                rel = 1
                dcg += rel / np.log2(i + 2)  # +2 because log_2(1) is 0
        
        # IDCG (Ideal DCG)
        idcg = 0.0
        for i in range(min(len(truth_set), len(results_to_consider))):
            idcg += 1 / np.log2(i + 2)
        
        # NDCG
        if idcg == 0:
            ndcg = 0.0
        else:
            ndcg = dcg / idcg
        
        ndcgs.append(ndcg)
    
    if not ndcgs:
        return 0.0
    
    return sum(ndcgs) / len(ndcgs)


def precision_recall_curve(query_results: List[List[int]], 
                           ground_truth: List[List[int]], 
                           k_values: List[int]) -> Tuple[List[float], List[float]]:
    """
    Compute precision-recall curve for different k values.
    
    Args:
        query_results: List of lists containing indices of search results for each query
        ground_truth: List of lists containing indices of ground truth results for each query
        k_values: List of k values to evaluate
        
    Returns:
        Tuple of (precision_values, recall_values)
    """
    precision_values = []
    recall_values = []
    
    for k in k_values:
        precision = compute_precision_at_k(query_results, ground_truth, k)
        recall = compute_recall_at_k(query_results, ground_truth, k)
        
        precision_values.append(precision)
        recall_values.append(recall)
    
    return precision_values, recall_values


def compute_dimension_precision_loss(high_dim_embeddings: np.ndarray, 
                                   low_dim_embeddings: np.ndarray, 
                                   queries: np.ndarray = None,
                                   k: int = 10) -> Dict[str, float]:
    """
    Compute precision loss between high and low dimension embeddings.
    
    Args:
        high_dim_embeddings: High-dimensional embeddings (n_samples x high_dim)
        low_dim_embeddings: Low-dimensional embeddings (n_samples x low_dim)
        queries: Optional query embeddings. If None, uses high_dim_embeddings as queries
        k: Number of nearest neighbors to consider
        
    Returns:
        Dictionary with precision metrics
    """
    if high_dim_embeddings.shape[0] != low_dim_embeddings.shape[0]:
        raise ValueError("High and low dimension embeddings must have the same number of samples")
    
    # If no queries provided, use the embeddings themselves
    if queries is None:
        queries_high = high_dim_embeddings
        queries_low = low_dim_embeddings
    else:
        queries_high = queries
        queries_low = queries
    
    # Compute similarity matrices
    high_sim = batch_cosine_similarity(queries_high, high_dim_embeddings)
    low_sim = batch_cosine_similarity(queries_low, low_dim_embeddings)
    
    # Get top-k indices for each query
    high_indices = np.argsort(-high_sim, axis=1)[:, :k]
    low_indices = np.argsort(-low_sim, axis=1)[:, :k]
    
    # Convert to list format for metric functions
    high_results = [list(indices) for indices in high_indices]
    low_results = [list(indices) for indices in low_indices]
    
    # Compute metrics
    recall_at_k = compute_recall_at_k(low_results, high_results, k)
    precision_at_k = compute_precision_at_k(low_results, high_results, k)
    map_score = mean_average_precision(low_results, high_results, k)
    ndcg_score = normalized_discounted_cumulative_gain(low_results, high_results, k)
    
    # Compute average overlap (Jaccard similarity)
    jaccard_similarities = []
    for high_idx, low_idx in zip(high_indices, low_indices):
        high_set = set(high_idx)
        low_set = set(low_idx)
        
        intersection = len(high_set.intersection(low_set))
        union = len(high_set.union(low_set))
        
        jaccard = intersection / union if union > 0 else 0.0
        jaccard_similarities.append(jaccard)
    
    avg_jaccard = sum(jaccard_similarities) / len(jaccard_similarities)
    
    # Compute rank correlation (Spearman)
    rank_correlations = []
    for high_sim_row, low_sim_row in zip(high_sim, low_sim):
        # Convert similarities to ranks
        high_ranks = np.argsort(np.argsort(-high_sim_row))
        low_ranks = np.argsort(np.argsort(-low_sim_row))
        
        # Compute correlation
        n = len(high_ranks)
        rank_diff_squared = np.sum((high_ranks - low_ranks) ** 2)
        spearman = 1 - (6 * rank_diff_squared) / (n * (n**2 - 1))
        rank_correlations.append(spearman)
    
    avg_rank_correlation = sum(rank_correlations) / len(rank_correlations)
    
    return {
        'recall_at_k': float(recall_at_k),
        'precision_at_k': float(precision_at_k),
        'map': float(map_score),
        'ndcg': float(ndcg_score),
        'jaccard_similarity': float(avg_jaccard),
        'rank_correlation': float(avg_rank_correlation)
    }


def compare_search_performance(reference_search_func: Callable,
                             test_search_func: Callable,
                             queries: List[Any],
                             k: int = 10,
                             num_iterations: int = 5) -> Dict[str, Any]:
    """
    Compare performance of two search functions.
    
    Args:
        reference_search_func: Reference search function (ground truth)
        test_search_func: Test search function to evaluate
        queries: List of queries to search for
        k: Number of results to consider
        num_iterations: Number of times to run each function for timing
        
    Returns:
        Dictionary with performance metrics
    """
    # Run reference search
    reference_results = []
    reference_times = []
    
    for i in range(num_iterations):
        for query in queries:
            start_time = time.time()
            results = reference_search_func(query, k)
            end_time = time.time()
            
            if i == 0:  # Only collect results from first iteration
                reference_results.append(results)
            
            reference_times.append(end_time - start_time)
    
    # Run test search
    test_results = []
    test_times = []
    
    for i in range(num_iterations):
        for query in queries:
            start_time = time.time()
            results = test_search_func(query, k)
            end_time = time.time()
            
            if i == 0:  # Only collect results from first iteration
                test_results.append(results)
            
            test_times.append(end_time - start_time)
    
    # Compute metrics
    recall = compute_recall_at_k(test_results, reference_results, k)
    precision = compute_precision_at_k(test_results, reference_results, k)
    map_score = mean_average_precision(test_results, reference_results, k)
    ndcg = normalized_discounted_cumulative_gain(test_results, reference_results, k)
    
    # Average times
    avg_reference_time = sum(reference_times) / len(reference_times)
    avg_test_time = sum(test_times) / len(test_times)
    speedup = avg_reference_time / avg_test_time if avg_test_time > 0 else float('inf')
    
    return {
        'recall_at_k': float(recall),
        'precision_at_k': float(precision),
        'map': float(map_score),
        'ndcg': float(ndcg),
        'reference_time': float(avg_reference_time),
        'test_time': float(avg_test_time),
        'speedup': float(speedup)
    }


def evaluate_dimension_cascade(cascade_model,
                               reference_embeddings: np.ndarray,
                               queries: np.ndarray,
                               k_values: List[int] = [1, 5, 10, 50, 100],
                               ground_truth: List[List[int]] = None) -> Dict[str, Any]:
    """
    Evaluate a dimensional cascade model across different dimensions.
    
    Args:
        cascade_model: Dimensional cascade model to evaluate
        reference_embeddings: Reference embeddings at full dimension
        queries: Query embeddings
        k_values: List of k values for evaluation
        ground_truth: Optional ground truth results for each query
        
    Returns:
        Dictionary with evaluation metrics
    """
    dimensions = cascade_model.dimensions
    results = {}
    
    # Get ground truth if not provided
    if ground_truth is None:
        # Compute ground truth using reference embeddings
        sim_matrix = batch_cosine_similarity(queries, reference_embeddings)
        ground_truth = [list(np.argsort(-sim_row)[:max(k_values)]) for sim_row in sim_matrix]
    
    # Evaluate each dimension
    for dim in dimensions:
        dim_results = {}
        
        # Get search results for this dimension
        reduced_embeddings = cascade_model.get_embeddings(dim)
        
        # Check if we need to reduce queries
        if cascade_model.transform_queries:
            reduced_queries = cascade_model.transform(queries, target_dim=dim)
        else:
            # Just use original queries
            reduced_queries = queries
        
        # Compute similarity matrix
        sim_matrix = batch_cosine_similarity(reduced_queries, reduced_embeddings)
        
        # Get top-k results for different k values
        for k in k_values:
            dim_results[f'k={k}'] = {}
            
            # Get top-k indices
            top_indices = [list(np.argsort(-sim_row)[:k]) for sim_row in sim_matrix]
            
            # Compute metrics
            recall = compute_recall_at_k(top_indices, ground_truth, k)
            precision = compute_precision_at_k(top_indices, ground_truth, k)
            map_score = mean_average_precision(top_indices, ground_truth, k)
            ndcg = normalized_discounted_cumulative_gain(top_indices, ground_truth, k)
            
            dim_results[f'k={k}']['recall'] = float(recall)
            dim_results[f'k={k}']['precision'] = float(precision)
            dim_results[f'k={k}']['map'] = float(map_score)
            dim_results[f'k={k}']['ndcg'] = float(ndcg)
        
        results[f'dim={dim}'] = dim_results
    
    return results 