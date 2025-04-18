"""
Evaluation utilities for the Dimensional Cascade project.

This module provides tools for evaluating search performance, including:
- Recall vs search time analysis
- Dimension cascade performance metrics
- Comparison with baseline approaches
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Any, Union, Callable
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SearchEvaluator:
    """
    Evaluation tools for measuring search performance.
    """
    
    def __init__(self, 
                ground_truth: Dict[int, List[int]] = None,
                metrics: List[str] = None,
                save_dir: str = None):
        """
        Initialize the evaluator.
        
        Args:
            ground_truth: Dictionary mapping query IDs to lists of relevant document IDs
            metrics: List of metrics to compute (default: ['recall', 'precision', 'f1', 'time'])
            save_dir: Directory to save evaluation results
        """
        self.ground_truth = ground_truth or {}
        self.metrics = metrics or ['recall', 'precision', 'f1', 'time']
        self.save_dir = save_dir
        
        # Results storage
        self.results = {}
        
    def set_ground_truth(self, ground_truth: Dict[int, List[int]]):
        """
        Set the ground truth for evaluation.
        
        Args:
            ground_truth: Dictionary mapping query IDs to lists of relevant document IDs
        """
        self.ground_truth = ground_truth
        
    def evaluate_search_method(self, 
                            method_name: str,
                            search_fn: Callable[[Any, int, int], Tuple[List[int], float]],
                            query_vectors: np.ndarray,
                            query_ids: List[int],
                            k_values: List[int] = None,
                            num_runs: int = 1) -> Dict[str, Any]:
        """
        Evaluate a search method on the given queries.
        
        Args:
            method_name: Name of the search method
            search_fn: Search function that takes (query_vector, query_id, k) and returns (result_ids, search_time)
            query_vectors: Array of query vectors
            query_ids: List of query IDs corresponding to the vectors
            k_values: List of k values to evaluate (default: [1, 5, 10, 50, 100])
            num_runs: Number of runs for each query (for timing stability)
            
        Returns:
            Dictionary of evaluation results
        """
        if k_values is None:
            k_values = [1, 5, 10, 50, 100]
            
        logger.info(f"Evaluating search method: {method_name}")
        logger.info(f"Number of queries: {len(query_ids)}")
        logger.info(f"k values: {k_values}")
        
        results = {
            'method': method_name,
            'num_queries': len(query_ids),
            'k_values': k_values,
            'metrics': {},
            'query_results': {}
        }
        
        # Initialize metrics
        for k in k_values:
            results['metrics'][k] = {
                'recall': 0.0,
                'precision': 0.0,
                'f1': 0.0,
                'ndcg': 0.0,
                'time': 0.0,
                'time_std': 0.0
            }
            
        # Evaluate each query
        for i, (query_id, query_vec) in enumerate(zip(query_ids, query_vectors)):
            if i > 0 and i % 10 == 0:
                logger.info(f"Processed {i}/{len(query_ids)} queries")
                
            # Skip if no ground truth available
            if query_id not in self.ground_truth:
                logger.warning(f"No ground truth for query ID {query_id}, skipping")
                continue
                
            relevant_docs = set(self.ground_truth[query_id])
            query_result = {'times': {}, 'results': {}}
            
            # Evaluate for each k
            for k in k_values:
                # Run search multiple times for timing stability
                times = []
                results_list = []
                
                for _ in range(num_runs):
                    retrieved_ids, search_time = search_fn(query_vec, query_id, k)
                    times.append(search_time)
                    results_list.append(retrieved_ids)
                    
                # Use results from first run for metrics
                retrieved_ids = results_list[0]
                query_result['results'][k] = retrieved_ids
                
                # Calculate timing stats
                avg_time = np.mean(times)
                std_time = np.std(times)
                query_result['times'][k] = {
                    'avg': avg_time,
                    'std': std_time,
                    'min': min(times),
                    'max': max(times)
                }
                
                # Calculate metrics
                retrieved_set = set(retrieved_ids[:k])
                num_relevant = len(relevant_docs)
                num_retrieved = len(retrieved_set)
                num_relevant_retrieved = len(relevant_docs.intersection(retrieved_set))
                
                # Recall
                recall = num_relevant_retrieved / num_relevant if num_relevant > 0 else 0.0
                
                # Precision
                precision = num_relevant_retrieved / num_retrieved if num_retrieved > 0 else 0.0
                
                # F1
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                # NDCG (Normalized Discounted Cumulative Gain)
                dcg = 0.0
                idcg = 0.0
                
                # Calculate DCG
                for j, doc_id in enumerate(retrieved_ids[:k]):
                    rel = 1 if doc_id in relevant_docs else 0
                    dcg += rel / np.log2(j + 2)  # +2 because j is 0-indexed
                    
                # Calculate IDCG (Ideal DCG)
                num_rel = min(k, len(relevant_docs))
                for j in range(num_rel):
                    idcg += 1.0 / np.log2(j + 2)
                    
                ndcg = dcg / idcg if idcg > 0 else 0.0
                
                # Accumulate metrics
                results['metrics'][k]['recall'] += recall
                results['metrics'][k]['precision'] += precision
                results['metrics'][k]['f1'] += f1
                results['metrics'][k]['ndcg'] += ndcg
                results['metrics'][k]['time'] += avg_time
                results['metrics'][k]['time_std'] = std_time  # Using last query's std for now
                
            # Store per-query results
            results['query_results'][query_id] = query_result
        
        # Calculate averages
        num_queries_with_gt = sum(1 for qid in query_ids if qid in self.ground_truth)
        if num_queries_with_gt > 0:
            for k in k_values:
                results['metrics'][k]['recall'] /= num_queries_with_gt
                results['metrics'][k]['precision'] /= num_queries_with_gt
                results['metrics'][k]['f1'] /= num_queries_with_gt
                results['metrics'][k]['ndcg'] /= num_queries_with_gt
                results['metrics'][k]['time'] /= num_queries_with_gt
        
        # Store results
        self.results[method_name] = results
        logger.info(f"Evaluation completed for {method_name}")
        
        return results
    
    def evaluate_dimension_cascade(self,
                                  method_name: str,
                                  cascade_search_fn: Callable,
                                  query_vectors: Dict[int, np.ndarray],
                                  query_ids: List[int],
                                  dimensions: List[int],
                                  k_values: List[int] = None,
                                  filter_ratios: List[float] = None) -> Dict[str, Any]:
        """
        Evaluate a dimensional cascade search method.
        
        Args:
            method_name: Name of the cascade method
            cascade_search_fn: Function that takes (query_vector_dict, query_id, k, dimensions, filter_ratio)
                              and returns (result_ids, search_time, per_dimension_times)
            query_vectors: Dictionary mapping dimensions to arrays of query vectors
            query_ids: List of query IDs
            dimensions: List of dimensions in the cascade (in increasing order)
            k_values: List of k values to evaluate (default: [10, 50, 100])
            filter_ratios: List of filter ratios to evaluate (default: [2, 5, 10])
            
        Returns:
            Dictionary of evaluation results
        """
        if k_values is None:
            k_values = [10, 50, 100]
            
        if filter_ratios is None:
            filter_ratios = [2, 5, 10]
            
        logger.info(f"Evaluating cascade method: {method_name}")
        logger.info(f"Dimensions: {dimensions}")
        logger.info(f"k values: {k_values}")
        logger.info(f"Filter ratios: {filter_ratios}")
        
        results = {
            'method': method_name,
            'num_queries': len(query_ids),
            'dimensions': dimensions,
            'k_values': k_values,
            'filter_ratios': filter_ratios,
            'metrics': {},
            'query_results': {}
        }
        
        # Initialize metrics
        for k in k_values:
            results['metrics'][k] = {}
            for ratio in filter_ratios:
                results['metrics'][k][ratio] = {
                    'recall': 0.0,
                    'precision': 0.0,
                    'f1': 0.0,
                    'ndcg': 0.0,
                    'time': 0.0,
                    'time_std': 0.0,
                    'dimension_times': {dim: 0.0 for dim in dimensions}
                }
        
        # Evaluate each query
        for i, query_id in enumerate(query_ids):
            if i > 0 and i % 10 == 0:
                logger.info(f"Processed {i}/{len(query_ids)} queries")
                
            # Skip if no ground truth available
            if query_id not in self.ground_truth:
                logger.warning(f"No ground truth for query ID {query_id}, skipping")
                continue
                
            relevant_docs = set(self.ground_truth[query_id])
            query_result = {'times': {}, 'results': {}}
            
            # Get query vectors for this query
            query_vecs = {dim: query_vectors[dim][i] for dim in dimensions}
            
            # Evaluate for each k and filter ratio
            for k in k_values:
                query_result['times'][k] = {}
                query_result['results'][k] = {}
                
                for ratio in filter_ratios:
                    retrieved_ids, search_time, dim_times = cascade_search_fn(
                        query_vecs, query_id, k, dimensions, ratio)
                    
                    query_result['results'][k][ratio] = retrieved_ids
                    query_result['times'][k][ratio] = {
                        'total': search_time,
                        'dimensions': dim_times
                    }
                    
                    # Calculate metrics
                    retrieved_set = set(retrieved_ids[:k])
                    num_relevant = len(relevant_docs)
                    num_retrieved = len(retrieved_set)
                    num_relevant_retrieved = len(relevant_docs.intersection(retrieved_set))
                    
                    # Recall
                    recall = num_relevant_retrieved / num_relevant if num_relevant > 0 else 0.0
                    
                    # Precision
                    precision = num_relevant_retrieved / num_retrieved if num_retrieved > 0 else 0.0
                    
                    # F1
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                    
                    # NDCG
                    dcg = 0.0
                    idcg = 0.0
                    
                    for j, doc_id in enumerate(retrieved_ids[:k]):
                        rel = 1 if doc_id in relevant_docs else 0
                        dcg += rel / np.log2(j + 2)
                        
                    num_rel = min(k, len(relevant_docs))
                    for j in range(num_rel):
                        idcg += 1.0 / np.log2(j + 2)
                        
                    ndcg = dcg / idcg if idcg > 0 else 0.0
                    
                    # Accumulate metrics
                    results['metrics'][k][ratio]['recall'] += recall
                    results['metrics'][k][ratio]['precision'] += precision
                    results['metrics'][k][ratio]['f1'] += f1
                    results['metrics'][k][ratio]['ndcg'] += ndcg
                    results['metrics'][k][ratio]['time'] += search_time
                    
                    # Accumulate dimension times
                    for dim, dim_time in dim_times.items():
                        results['metrics'][k][ratio]['dimension_times'][dim] += dim_time
            
            # Store per-query results
            results['query_results'][query_id] = query_result
        
        # Calculate averages
        num_queries_with_gt = sum(1 for qid in query_ids if qid in self.ground_truth)
        if num_queries_with_gt > 0:
            for k in k_values:
                for ratio in filter_ratios:
                    results['metrics'][k][ratio]['recall'] /= num_queries_with_gt
                    results['metrics'][k][ratio]['precision'] /= num_queries_with_gt
                    results['metrics'][k][ratio]['f1'] /= num_queries_with_gt
                    results['metrics'][k][ratio]['ndcg'] /= num_queries_with_gt
                    results['metrics'][k][ratio]['time'] /= num_queries_with_gt
                    
                    for dim in dimensions:
                        results['metrics'][k][ratio]['dimension_times'][dim] /= num_queries_with_gt
        
        # Store results
        self.results[method_name] = results
        logger.info(f"Evaluation completed for {method_name}")
        
        return results
    
    def plot_results(self, metric: str = 'recall', k: int = 100, save_path: str = None):
        """
        Plot evaluation results.
        
        Args:
            metric: Metric to plot ('recall', 'precision', 'f1', 'ndcg', 'time')
            k: k value to use for the plot
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Plot non-cascade methods
        for method_name, result in self.results.items():
            if 'filter_ratios' not in result:  # Non-cascade method
                metrics = result['metrics']
                if k in metrics:
                    x = [k]
                    y = [metrics[k][metric]]
                    plt.plot(x, y, 'o', label=f"{method_name}")
        
        # Plot cascade methods
        for method_name, result in self.results.items():
            if 'filter_ratios' in result:  # Cascade method
                if k in result['metrics']:
                    x = []
                    y = []
                    for ratio in result['filter_ratios']:
                        x.append(ratio)
                        y.append(result['metrics'][k][ratio][metric])
                    plt.plot(x, y, 'o-', label=f"{method_name}")
        
        plt.xlabel('Filter Ratio' if any('filter_ratios' in r for r in self.results.values()) else 'k')
        plt.ylabel(metric.capitalize())
        plt.title(f'{metric.capitalize()} vs. Filter Ratio (k={k})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
        
    def plot_recall_vs_time(self, k: int = 100, save_path: str = None):
        """
        Plot recall vs. search time.
        
        Args:
            k: k value to use for the plot
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Plot non-cascade methods
        for method_name, result in self.results.items():
            if 'filter_ratios' not in result:  # Non-cascade method
                metrics = result['metrics']
                if k in metrics:
                    x = [metrics[k]['time']]
                    y = [metrics[k]['recall']]
                    plt.plot(x, y, 'o', markersize=8, label=f"{method_name}")
        
        # Plot cascade methods
        for method_name, result in self.results.items():
            if 'filter_ratios' in result:  # Cascade method
                if k in result['metrics']:
                    x = []
                    y = []
                    for ratio in result['filter_ratios']:
                        x.append(result['metrics'][k][ratio]['time'])
                        y.append(result['metrics'][k][ratio]['recall'])
                    plt.plot(x, y, 'o-', label=f"{method_name}")
        
        plt.xlabel('Search Time (s)')
        plt.ylabel('Recall')
        plt.title(f'Recall vs. Search Time (k={k})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
        
    def plot_dimension_times(self, method_name: str, k: int = 100, ratio: int = 5, save_path: str = None):
        """
        Plot search time per dimension.
        
        Args:
            method_name: Name of the cascade method to plot
            k: k value to use for the plot
            ratio: Filter ratio to use for the plot
            save_path: Path to save the plot
        """
        if method_name not in self.results:
            logger.error(f"Method {method_name} not found in results")
            return
            
        result = self.results[method_name]
        if 'filter_ratios' not in result:
            logger.error(f"Method {method_name} is not a cascade method")
            return
            
        if k not in result['metrics'] or ratio not in result['metrics'][k]:
            logger.error(f"k={k} or ratio={ratio} not found in results")
            return
            
        # Get dimension times
        dimensions = result['dimensions']
        times = [result['metrics'][k][ratio]['dimension_times'][dim] for dim in dimensions]
        
        plt.figure(figsize=(10, 6))
        plt.bar(dimensions, times, alpha=0.7)
        plt.xlabel('Dimension')
        plt.ylabel('Search Time (s)')
        plt.title(f'Search Time per Dimension ({method_name}, k={k}, ratio={ratio})')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Plot saved to {save_path}")
        
        plt.tight_layout()
        plt.show()
        
    def print_summary(self, k: int = 100, metric: str = 'recall'):
        """
        Print a summary of the evaluation results.
        
        Args:
            k: k value to use for the summary
            metric: Metric to highlight ('recall', 'precision', 'f1', 'ndcg', 'time')
        """
        print(f"\n===== Evaluation Summary (k={k}, metric={metric}) =====")
        
        # Print non-cascade methods
        print("\nStandard Methods:")
        for method_name, result in self.results.items():
            if 'filter_ratios' not in result:  # Non-cascade method
                metrics = result['metrics']
                if k in metrics:
                    print(f"  {method_name}:")
                    print(f"    Recall: {metrics[k]['recall']:.4f}")
                    print(f"    Precision: {metrics[k]['precision']:.4f}")
                    print(f"    F1: {metrics[k]['f1']:.4f}")
                    print(f"    NDCG: {metrics[k]['ndcg']:.4f}")
                    print(f"    Time: {metrics[k]['time']:.4f} s")
        
        # Print cascade methods
        print("\nCascade Methods:")
        for method_name, result in self.results.items():
            if 'filter_ratios' in result:  # Cascade method
                if k in result['metrics']:
                    print(f"  {method_name}:")
                    
                    for ratio in result['filter_ratios']:
                        metrics = result['metrics'][k][ratio]
                        print(f"    Filter Ratio {ratio}:")
                        print(f"      Recall: {metrics['recall']:.4f}")
                        print(f"      Precision: {metrics['precision']:.4f}")
                        print(f"      F1: {metrics['f1']:.4f}")
                        print(f"      NDCG: {metrics['ndcg']:.4f}")
                        print(f"      Time: {metrics['time']:.4f} s")
                        
                        # Print dimension times
                        print(f"      Time per dimension:")
                        for dim, dim_time in metrics['dimension_times'].items():
                            print(f"        {dim}: {dim_time:.4f} s")
        
        print("\n=================================")


if __name__ == "__main__":
    # Example usage
    import argparse
    from sklearn.neighbors import NearestNeighbors
    
    parser = argparse.ArgumentParser(description="Evaluation demo")
    parser.add_argument("--vectors", type=str, help="Path to vectors file (optional)")
    parser.add_argument("--dims", type=str, default="32,64,128,256", help="Comma-separated dimensions")
    args = parser.parse_args()
    
    # Demo with synthetic data if no vectors provided
    if not args.vectors:
        logger.info("No vectors file provided, using synthetic data for demo")
        dimensions = [int(d) for d in args.dims.split(",")]
        
        # Create synthetic data
        np.random.seed(42)
        n_docs = 1000
        n_queries = 20
        
        # Create vectors for each dimension
        doc_vectors = {}
        query_vectors = {}
        
        for dim in dimensions:
            doc_vectors[dim] = np.random.randn(n_docs, dim).astype(np.float32)
            query_vectors[dim] = np.random.randn(n_queries, dim).astype(np.float32)
            
            # Normalize
            doc_vectors[dim] = doc_vectors[dim] / np.linalg.norm(doc_vectors[dim], axis=1, keepdims=True)
            query_vectors[dim] = query_vectors[dim] / np.linalg.norm(query_vectors[dim], axis=1, keepdims=True)
        
        # Create synthetic ground truth (top 10 nearest neighbors with highest dimension)
        highest_dim = dimensions[-1]
        nbrs = NearestNeighbors(n_neighbors=10, algorithm='brute', metric='cosine').fit(doc_vectors[highest_dim])
        
        ground_truth = {}
        for i in range(n_queries):
            _, indices = nbrs.kneighbors(query_vectors[highest_dim][i:i+1])
            ground_truth[i] = indices[0].tolist()
        
        # Create evaluator
        evaluator = SearchEvaluator(ground_truth=ground_truth)
        
        # Define baseline search function
        def baseline_search(query_vec, query_id, k, dim=highest_dim):
            start_time = time.time()
            nbrs = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='cosine').fit(doc_vectors[dim])
            _, indices = nbrs.kneighbors(query_vec.reshape(1, -1))
            search_time = time.time() - start_time
            return indices[0].tolist(), search_time
        
        # Define cascade search function
        def cascade_search(query_vecs, query_id, k, dimensions, filter_ratio):
            start_time = time.time()
            dim_times = {}
            
            # Start with the smallest dimension
            candidates = None
            
            for i, dim in enumerate(dimensions):
                dim_start_time = time.time()
                
                if i == 0:
                    # First dimension, retrieve filter_ratio * k candidates
                    nbrs = NearestNeighbors(n_neighbors=k * filter_ratio, algorithm='brute', metric='cosine').fit(doc_vectors[dim])
                    _, indices = nbrs.kneighbors(query_vecs[dim].reshape(1, -1))
                    candidates = indices[0].tolist()
                else:
                    # Filter candidates using higher dimension
                    candidate_vectors = np.array([doc_vectors[dim][c] for c in candidates])
                    nbrs = NearestNeighbors(n_neighbors=len(candidates), algorithm='brute', metric='cosine').fit(candidate_vectors)
                    _, indices = nbrs.kneighbors(query_vecs[dim].reshape(1, -1))
                    
                    # Get actual document IDs from candidate indices
                    candidates = [candidates[j] for j in indices[0]]
                    
                    # For the last dimension, only keep the top k
                    if i == len(dimensions) - 1:
                        candidates = candidates[:k]
                
                dim_times[dim] = time.time() - dim_start_time
            
            search_time = time.time() - start_time
            return candidates, search_time, dim_times
        
        # Evaluate baseline methods
        logger.info("Evaluating baseline methods for each dimension...")
        for dim in dimensions:
            evaluator.evaluate_search_method(
                f"baseline_dim{dim}",
                lambda qv, qid, k, d=dim: baseline_search(qv, qid, k, d),
                query_vectors[dim],
                list(range(n_queries)),
                k_values=[10, 50, 100]
            )
        
        # Evaluate cascade method
        logger.info("Evaluating dimensional cascade...")
        evaluator.evaluate_dimension_cascade(
            "cascade",
            cascade_search,
            query_vectors,
            list(range(n_queries)),
            dimensions,
            k_values=[10, 50, 100],
            filter_ratios=[2, 5, 10]
        )
        
        # Print summary
        evaluator.print_summary(k=10)
        
        # Plot results
        evaluator.plot_recall_vs_time(k=10)
        evaluator.plot_dimension_times("cascade", k=10, ratio=5)
    else:
        logger.info(f"Loading vectors from {args.vectors}")
        # Implementation for loading real data would go here 