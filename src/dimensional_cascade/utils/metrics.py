"""
Metrics for evaluating the Dimensional Cascade performance.
"""
import time
from typing import Dict, List, Tuple, Any, Optional, Set

import numpy as np
from sklearn.metrics import precision_recall_curve, average_precision_score


def calculate_precision_loss(
    high_dim_results: List[Tuple[Dict[str, Any], float]],
    low_dim_results: List[Tuple[Dict[str, Any], float]],
    k: int = 100
) -> Dict[str, float]:
    """Calculate precision loss between high and low dimension results.
    
    Args:
        high_dim_results: Results from higher dimension search (ground truth)
        low_dim_results: Results from lower dimension search to evaluate
        k: Number of top results to consider
        
    Returns:
        Dictionary with precision metrics
    """
    # Limit to top-k results
    high_dim_results = high_dim_results[:k]
    low_dim_results = low_dim_results[:k]
    
    # Extract document IDs
    high_dim_ids = {doc.get('id') for doc, _ in high_dim_results}
    low_dim_ids = {doc.get('id') for doc, _ in low_dim_results}
    
    # Calculate metrics
    intersection = high_dim_ids.intersection(low_dim_ids)
    precision = len(intersection) / len(low_dim_ids) if low_dim_ids else 0
    recall = len(intersection) / len(high_dim_ids) if high_dim_ids else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate relative precision loss
    precision_loss = 1 - precision
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_loss': precision_loss,
        'overlap_count': len(intersection),
        'high_dim_count': len(high_dim_ids),
        'low_dim_count': len(low_dim_ids)
    }


def calculate_precision_recall(
    relevant_docs: Set[int], 
    retrieved_docs: List[Tuple[int, float]]
) -> Dict[str, float]:
    """Calculate precision and recall metrics.
    
    Args:
        relevant_docs: Set of relevant document IDs (ground truth)
        retrieved_docs: List of (doc_id, score) tuples from search results
        
    Returns:
        Dictionary with precision metrics
    """
    # Extract retrieved document IDs
    retrieved_ids = [doc_id for doc_id, _ in retrieved_docs]
    
    # Calculate true/false positives for each k
    precisions = []
    recalls = []
    
    for k in range(1, len(retrieved_ids) + 1):
        top_k = set(retrieved_ids[:k])
        relevant_in_top_k = len(relevant_docs.intersection(top_k))
        
        precision_at_k = relevant_in_top_k / k if k > 0 else 0
        recall_at_k = relevant_in_top_k / len(relevant_docs) if relevant_docs else 0
        
        precisions.append(precision_at_k)
        recalls.append(recall_at_k)
    
    # Calculate average precision (AP)
    ap = 0.0
    for i, precision in enumerate(precisions):
        if i == 0 or retrieved_ids[i-1] in relevant_docs:
            ap += precision
    ap /= len(relevant_docs) if relevant_docs else 1
    
    return {
        'precision@1': precisions[0] if precisions else 0,
        'precision@5': precisions[4] if len(precisions) > 4 else 0,
        'precision@10': precisions[9] if len(precisions) > 9 else 0,
        'recall@10': recalls[9] if len(recalls) > 9 else 0,
        'recall@50': recalls[49] if len(recalls) > 49 else recalls[-1] if recalls else 0,
        'recall@100': recalls[99] if len(recalls) > 99 else recalls[-1] if recalls else 0,
        'average_precision': ap
    }


def calculate_latency(
    dimensions: List[int],
    queries: List[str],
    search_fn
) -> Dict[int, Dict[str, float]]:
    """Measure search latency across different dimensions.
    
    Args:
        dimensions: List of dimensions to test
        queries: List of queries to search for
        search_fn: Function that performs search for a given dimension
        
    Returns:
        Dictionary mapping from dimension to latency metrics
    """
    results = {}
    
    for dim in dimensions:
        latencies = []
        
        for query in queries:
            # Measure search time
            start_time = time.time()
            search_fn(query, dimension=dim)
            elapsed = time.time() - start_time
            
            latencies.append(elapsed)
        
        # Calculate statistics
        results[dim] = {
            'mean_latency': np.mean(latencies),
            'median_latency': np.median(latencies),
            'min_latency': np.min(latencies),
            'max_latency': np.max(latencies),
            'p95_latency': np.percentile(latencies, 95),
            'p99_latency': np.percentile(latencies, 99)
        }
    
    return results


def dimension_cascade_analysis(
    dimensions: List[int],
    queries: List[str],
    cascade_search_fn,
    single_dim_search_fn,
    k: int = 100
) -> Dict[str, Any]:
    """Analyze the dimensional cascade approach vs. single dimension.
    
    Args:
        dimensions: List of dimensions to test (sorted high to low)
        queries: List of queries to search for
        cascade_search_fn: Function that performs cascade search
        single_dim_search_fn: Function that performs search at a single dimension
        k: Number of top results to consider
        
    Returns:
        Dictionary with analysis results
    """
    # Ensure dimensions are sorted (high to low)
    dimensions = sorted(dimensions, reverse=True)
    
    results = {
        'dimension_metrics': {},
        'latency': {},
        'precision_loss': {},
        'overall': {}
    }
    
    # Use highest dimension as ground truth
    highest_dim = dimensions[0]
    
    # Analyze each query
    for query in queries:
        # Get ground truth results
        ground_truth = single_dim_search_fn(query, dimension=highest_dim, k=k)
        ground_truth_ids = {doc.get('id') for doc, _ in ground_truth}
        
        # Test each lower dimension
        for dim in dimensions[1:]:
            # Single dimension search
            start_time = time.time()
            dim_results = single_dim_search_fn(query, dimension=dim, k=k)
            dim_time = time.time() - start_time
            
            # Calculate metrics
            metrics = calculate_precision_loss(ground_truth, dim_results, k)
            
            # Store results
            if dim not in results['dimension_metrics']:
                results['dimension_metrics'][dim] = []
                results['latency'][dim] = []
                results['precision_loss'][dim] = []
            
            results['dimension_metrics'][dim].append(metrics)
            results['latency'][dim].append(dim_time)
            results['precision_loss'][dim].append(metrics['precision_loss'])
        
        # Cascade search
        start_time = time.time()
        cascade_results = cascade_search_fn(query, k=k)
        cascade_time = time.time() - start_time
        
        # Calculate cascade metrics
        cascade_metrics = calculate_precision_loss(ground_truth, cascade_results, k)
        
        # Store cascade results
        if 'cascade' not in results['overall']:
            results['overall']['cascade'] = {
                'metrics': [],
                'latency': []
            }
        
        results['overall']['cascade']['metrics'].append(cascade_metrics)
        results['overall']['cascade']['latency'].append(cascade_time)
    
    # Calculate averages
    for dim in dimensions[1:]:
        if dim in results['dimension_metrics']:
            avg_precision_loss = np.mean([m['precision_loss'] for m in results['dimension_metrics'][dim]])
            avg_latency = np.mean(results['latency'][dim])
            latency_speedup = np.mean(results['latency'][highest_dim]) / avg_latency if highest_dim in results['latency'] else 0
            
            results['dimension_metrics'][dim] = {
                'avg_precision_loss': avg_precision_loss,
                'avg_latency': avg_latency,
                'latency_speedup': latency_speedup
            }
    
    if 'cascade' in results['overall']:
        avg_cascade_precision_loss = np.mean([m['precision_loss'] for m in results['overall']['cascade']['metrics']])
        avg_cascade_latency = np.mean(results['overall']['cascade']['latency'])
        cascade_speedup = np.mean(results['latency'][highest_dim]) / avg_cascade_latency if highest_dim in results['latency'] else 0
        
        results['overall']['cascade'] = {
            'avg_precision_loss': avg_cascade_precision_loss,
            'avg_latency': avg_cascade_latency,
            'latency_speedup': cascade_speedup
        }
    
    return results 