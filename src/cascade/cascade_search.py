"""
Dimensional Cascade Search implementation.

This module implements the core algorithms for dimensional cascade search,
which progressively filters results using embeddings of increasing dimensionality.
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
import logging
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CascadeSearch:
    """
    Implements dimensional cascade search for efficient approximate nearest neighbor search.
    """
    
    def __init__(self, 
                document_vectors: Dict[int, np.ndarray],
                dimensions: List[int] = None,
                metric: str = 'cosine',
                algorithm: str = 'brute',
                index_params: Dict = None):
        """
        Initialize the cascade search with document vectors for each dimension.
        
        Args:
            document_vectors: Dictionary mapping dimensions to document vectors
            dimensions: List of dimensions to use in the cascade (in increasing order)
            metric: Distance metric to use ('cosine', 'euclidean', etc.)
            algorithm: Nearest neighbor algorithm ('brute', 'ball_tree', 'kd_tree', etc.)
            index_params: Additional parameters for the nearest neighbor algorithm
        """
        self.document_vectors = document_vectors
        self.dimensions = dimensions or sorted(document_vectors.keys())
        self.metric = metric
        self.algorithm = algorithm
        self.index_params = index_params or {}
        
        # Validate that dimensions are in increasing order
        for i in range(1, len(self.dimensions)):
            if self.dimensions[i] <= self.dimensions[i-1]:
                raise ValueError("Dimensions must be in increasing order")
                
        # Store document IDs
        self.doc_ids = list(range(len(document_vectors[self.dimensions[0]])))
        
        # Build indices for each dimension
        self.indices = {}
        self._build_indices()
        
    def _build_indices(self):
        """Build nearest neighbor indices for each dimension."""
        logger.info("Building indices for cascade search...")
        
        for dim in self.dimensions:
            logger.info(f"Building index for dimension {dim}...")
            
            if dim not in self.document_vectors:
                raise ValueError(f"No document vectors found for dimension {dim}")
                
            vectors = self.document_vectors[dim]
            
            # Create and fit index
            index = NearestNeighbors(
                metric=self.metric,
                algorithm=self.algorithm,
                **self.index_params
            )
            index.fit(vectors)
            
            self.indices[dim] = index
            
        logger.info("Finished building indices")
        
    def search(self, 
              query_vectors: Dict[int, np.ndarray],
              k: int = 10,
              filter_ratio: int = 5,
              return_times: bool = False) -> Union[List[int], Tuple[List[int], float, Dict[int, float]]]:
        """
        Perform cascade search for a query.
        
        Args:
            query_vectors: Dictionary mapping dimensions to query vectors
            k: Number of results to return
            filter_ratio: Multiplier for intermediate candidate set sizes
            return_times: Whether to return timing information
            
        Returns:
            If return_times is False: List of document IDs in decreasing order of similarity
            If return_times is True: Tuple of (document IDs, total_time, dimension_times)
        """
        start_time = time.time()
        dim_times = {}
        
        # Validate query vectors
        for dim in self.dimensions:
            if dim not in query_vectors:
                raise ValueError(f"No query vector found for dimension {dim}")
                
        # Start with the smallest dimension
        candidates = None
        
        for i, dim in enumerate(self.dimensions):
            dim_start = time.time()
            
            if i == 0:
                # First dimension, retrieve filter_ratio * k candidates
                n_candidates = min(k * filter_ratio, len(self.doc_ids))
                distances, indices = self.indices[dim].kneighbors(
                    query_vectors[dim].reshape(1, -1), 
                    n_neighbors=n_candidates
                )
                candidates = indices[0].tolist()
            else:
                # Filter candidates using higher dimension
                if len(candidates) <= 1:
                    # No need to filter if we only have 0 or 1 candidates
                    dim_times[dim] = time.time() - dim_start
                    continue
                    
                # Get candidate vectors for this dimension
                candidate_vectors = np.array([self.document_vectors[dim][c] for c in candidates])
                
                # Create a new index for just the candidates
                candidate_index = NearestNeighbors(
                    metric=self.metric,
                    algorithm=self.algorithm,
                    **self.index_params
                )
                candidate_index.fit(candidate_vectors)
                
                # Get the top candidates in this dimension
                n_candidates = min(len(candidates), k if i == len(self.dimensions) - 1 else len(candidates))
                distances, indices = candidate_index.kneighbors(
                    query_vectors[dim].reshape(1, -1),
                    n_neighbors=n_candidates
                )
                
                # Map indices back to original document IDs
                candidates = [candidates[j] for j in indices[0]]
            
            dim_times[dim] = time.time() - dim_start
        
        # Return top k results
        results = candidates[:k]
        total_time = time.time() - start_time
        
        if return_times:
            return results, total_time, dim_times
        else:
            return results
        
    def batch_search(self, 
                    query_vectors: Dict[int, np.ndarray],
                    k: int = 10,
                    filter_ratio: int = 5,
                    return_times: bool = False,
                    show_progress: bool = True) -> List[Union[List[int], Tuple[List[int], float, Dict[int, float]]]]:
        """
        Perform cascade search for multiple queries.
        
        Args:
            query_vectors: Dictionary mapping dimensions to query vector arrays
            k: Number of results to return for each query
            filter_ratio: Multiplier for intermediate candidate set sizes
            return_times: Whether to return timing information
            show_progress: Whether to show progress bar
            
        Returns:
            List of search results (same format as search() method)
        """
        # Validate query vectors
        n_queries = None
        for dim in self.dimensions:
            if dim not in query_vectors:
                raise ValueError(f"No query vectors found for dimension {dim}")
                
            if n_queries is None:
                n_queries = len(query_vectors[dim])
            elif n_queries != len(query_vectors[dim]):
                raise ValueError(f"Inconsistent number of query vectors for dimension {dim}")
        
        # Perform search for each query
        results = []
        iterator = tqdm(range(n_queries), desc="Searching") if show_progress else range(n_queries)
        
        for i in iterator:
            # Extract query vectors for this query
            query_vecs = {dim: query_vectors[dim][i].reshape(1, -1) for dim in self.dimensions}
            
            # Perform search
            result = self.search(query_vecs, k, filter_ratio, return_times)
            results.append(result)
            
        return results


class CascadeSearchFactory:
    """
    Factory for creating and managing cascade search instances.
    """
    
    def __init__(self):
        """Initialize the factory."""
        self.instances = {}
        
    def get_or_create(self, 
                     name: str,
                     document_vectors: Dict[int, np.ndarray] = None,
                     dimensions: List[int] = None,
                     metric: str = 'cosine',
                     algorithm: str = 'brute',
                     index_params: Dict = None) -> CascadeSearch:
        """
        Get an existing cascade search instance or create a new one.
        
        Args:
            name: Name of the cascade search instance
            document_vectors: Dictionary mapping dimensions to document vectors
            dimensions: List of dimensions to use in the cascade (in increasing order)
            metric: Distance metric to use ('cosine', 'euclidean', etc.)
            algorithm: Nearest neighbor algorithm ('brute', 'ball_tree', 'kd_tree', etc.)
            index_params: Additional parameters for the nearest neighbor algorithm
            
        Returns:
            CascadeSearch instance
        """
        if name in self.instances:
            return self.instances[name]
            
        if document_vectors is None:
            raise ValueError("Document vectors must be provided to create a new instance")
            
        instance = CascadeSearch(
            document_vectors=document_vectors,
            dimensions=dimensions,
            metric=metric,
            algorithm=algorithm,
            index_params=index_params
        )
        
        self.instances[name] = instance
        return instance
        
    def remove(self, name: str) -> bool:
        """
        Remove a cascade search instance.
        
        Args:
            name: Name of the instance to remove
            
        Returns:
            True if the instance was removed, False if it didn't exist
        """
        if name in self.instances:
            del self.instances[name]
            return True
        return False


# Singleton factory instance
factory = CascadeSearchFactory()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Cascade Search Demo")
    parser.add_argument("--dims", type=str, default="32,64,128,256", help="Comma-separated dimensions")
    parser.add_argument("--n_docs", type=int, default=1000, help="Number of documents")
    parser.add_argument("--n_queries", type=int, default=10, help="Number of queries")
    parser.add_argument("--k", type=int, default=10, help="Number of results to return")
    parser.add_argument("--filter_ratio", type=int, default=5, help="Filter ratio")
    
    args = parser.parse_args()
    
    # Parse dimensions
    dimensions = [int(d) for d in args.dims.split(",")]
    
    # Create synthetic data
    np.random.seed(42)
    
    # Create document vectors for each dimension
    document_vectors = {}
    for dim in dimensions:
        vectors = np.random.randn(args.n_docs, dim).astype(np.float32)
        # Normalize
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        document_vectors[dim] = vectors
    
    # Create query vectors
    query_vectors = {}
    for dim in dimensions:
        vectors = np.random.randn(args.n_queries, dim).astype(np.float32)
        # Normalize
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        query_vectors[dim] = vectors
    
    # Create cascade search
    cascade = CascadeSearch(
        document_vectors=document_vectors,
        dimensions=dimensions,
        metric='cosine',
        algorithm='brute'
    )
    
    # Perform search for first query
    query_vecs = {dim: query_vectors[dim][0] for dim in dimensions}
    results, total_time, dim_times = cascade.search(query_vecs, k=args.k, filter_ratio=args.filter_ratio, return_times=True)
    
    print(f"Results for query 0: {results}")
    print(f"Total search time: {total_time:.6f} s")
    print("Time per dimension:")
    for dim, time_taken in dim_times.items():
        print(f"  {dim}: {time_taken:.6f} s")
    
    # Perform batch search
    batch_results = cascade.batch_search(query_vectors, k=args.k, filter_ratio=args.filter_ratio, return_times=True)
    
    print(f"\nPerformed batch search for {len(batch_results)} queries")
    
    # Compare with brute force search on highest dimension
    print("\nComparing with brute force search on highest dimension...")
    
    highest_dim = dimensions[-1]
    brute_index = NearestNeighbors(metric='cosine', algorithm='brute')
    brute_index.fit(document_vectors[highest_dim])
    
    start_time = time.time()
    _, brute_indices = brute_index.kneighbors(query_vectors[highest_dim][0].reshape(1, -1), n_neighbors=args.k)
    brute_time = time.time() - start_time
    
    brute_results = brute_indices[0].tolist()
    
    print(f"Brute force results: {brute_results}")
    print(f"Brute force time: {brute_time:.6f} s")
    print(f"Cascade search time: {total_time:.6f} s")
    print(f"Speedup: {brute_time / total_time:.2f}x")
    
    # Calculate recall
    correct = set(results).intersection(set(brute_results))
    recall = len(correct) / len(brute_results)
    
    print(f"Recall@{args.k}: {recall:.4f}") 