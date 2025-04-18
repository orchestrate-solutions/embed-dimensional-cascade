"""
Dimensional Cascade Search implementation.

This module provides the core implementation of dimensional cascade search,
allowing efficient approximate nearest neighbor search by progressively
filtering candidates through multiple dimension projections.
"""

import numpy as np
import time
import logging
from typing import List, Dict, Tuple, Any, Callable, Optional, Union
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
import faiss
import heapq

logger = logging.getLogger(__name__)

class CascadeSearch:
    """
    Dimensional Cascade Search implementation.
    
    This class implements a search algorithm that progressively filters candidates
    through multiple dimension projections, starting from low dimensions and
    refining results in higher dimensions.
    """
    
    def __init__(
        self,
        vector_dict: Dict[int, np.ndarray],
        dimensions: List[int],
        filter_ratio: float = 0.1,
        metric: str = "cosine",
        use_faiss: bool = True,
        id_mapping: Optional[np.ndarray] = None
    ):
        """
        Initialize the cascade search with vectors projected to different dimensions.
        
        Args:
            vector_dict: Dictionary mapping dimensions to document vectors of shape (n_docs, dim)
            dimensions: List of dimensions to use in the cascade, sorted in ascending order
            filter_ratio: Ratio of candidates to keep at each level (multiplied by k)
            metric: Distance metric to use ('cosine', 'l2', etc.)
            use_faiss: Whether to use FAISS for indexing (faster for large datasets)
            id_mapping: Optional mapping from internal IDs to external IDs
        """
        self.vector_dict = vector_dict
        self.dimensions = sorted(dimensions)
        self.filter_ratio = filter_ratio
        self.metric = metric
        self.use_faiss = use_faiss
        
        # Map internal IDs to external IDs if provided
        self.id_mapping = id_mapping
        
        # Create indices for each dimension
        self.indices = self._create_indices()
        
        # Cache for query vectors (to avoid repeated projections)
        self.query_cache = {}
        
        logger.info(f"Initialized CascadeSearch with dimensions {dimensions}, filter_ratio={filter_ratio}, metric={metric}")
    
    def _create_indices(self) -> Dict[int, Any]:
        """
        Create indices for each dimension for efficient search.
        
        Returns:
            Dictionary mapping dimensions to indices
        """
        indices = {}
        
        for dim in self.dimensions:
            vectors = self.vector_dict[dim]
            
            if self.use_faiss and faiss is not None:
                # Use FAISS for fast approximate search
                if self.metric == "cosine":
                    # Normalize vectors for cosine similarity
                    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                    normalized_vectors = vectors / norms
                    
                    # Create FAISS index
                    index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
                    index.add(normalized_vectors.astype(np.float32))
                else:
                    # L2 distance (Euclidean)
                    index = faiss.IndexFlatL2(dim)
                    index.add(vectors.astype(np.float32))
                
                indices[dim] = index
            else:
                # Use scikit-learn for exact search
                nn_algo = 'brute' if len(vectors) < 10000 else 'auto'
                nn = NearestNeighbors(algorithm=nn_algo, metric=self.metric)
                nn.fit(vectors)
                indices[dim] = nn
        
        return indices
    
    def search(
        self,
        query: np.ndarray,
        k: int,
        return_distances: bool = False,
        verbose: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Perform cascade search for the nearest neighbors of a query vector.
        
        Args:
            query: Query vector of shape (dim,) where dim is the highest dimension
            k: Number of nearest neighbors to retrieve
            return_distances: Whether to return distances along with indices
            verbose: Whether to print detailed information during search
            
        Returns:
            Indices of nearest neighbors, or tuple of (indices, distances) if return_distances=True
        """
        start_time = time.time()
        
        # Get the maximum dimension from the dimensions list
        max_dim = self.dimensions[-1]
        
        # Check if the query dimensionality matches the highest dimension
        if len(query.shape) > 1:
            query = query.reshape(-1)
        
        if query.shape[0] != max_dim:
            if query.shape[0] in self.dimensions:
                # If query dimension is lower, we start the cascade from that dimension
                start_dim_idx = self.dimensions.index(query.shape[0])
                working_dimensions = self.dimensions[start_dim_idx:]
            else:
                raise ValueError(f"Query dimension {query.shape[0]} not in available dimensions {self.dimensions}")
        else:
            working_dimensions = self.dimensions
        
        # Project query to each dimension (if not already cached)
        if id(query) not in self.query_cache:
            self.query_cache[id(query)] = {}
            for dim in working_dimensions:
                if dim == max_dim:
                    self.query_cache[id(query)][dim] = query
                else:
                    if dim in self.vector_dict:
                        # Already have pre-projected vectors, so need to project query too
                        self.query_cache[id(query)][dim] = query[:dim]
                    else:
                        # Use full dimension query
                        self.query_cache[id(query)][dim] = query
        
        # Initial candidate set (all document indices)
        n_docs = len(self.vector_dict[working_dimensions[0]])
        candidates = np.arange(n_docs)
        
        # Calculate how many candidates to keep at each level
        k_values = self._calculate_k_values(k, working_dimensions)
        
        # Progressively filter candidates through dimensions
        for i, dim in enumerate(working_dimensions):
            level_start_time = time.time()
            k_level = k_values[i]
            
            # Get query vector for this dimension
            query_dim = self.query_cache[id(query)][dim]
            
            # If this is the last dimension or we have few enough candidates, do final search
            if i == len(working_dimensions) - 1 or len(candidates) <= k:
                if verbose:
                    logger.info(f"Final search in {dim}D with {len(candidates)} candidates")
                
                # Get results directly for final dimension
                if len(candidates) <= k:
                    # If we already have few enough candidates, return them all
                    results = candidates
                    if return_distances:
                        vectors = self.vector_dict[dim][candidates]
                        distances = self._compute_distances(query_dim, vectors)
                        sorted_indices = np.argsort(distances)
                        results = candidates[sorted_indices]
                        distances = distances[sorted_indices]
                else:
                    # Search among remaining candidates
                    results, distances = self._search_dimension(query_dim, dim, k, candidates)
                
                if verbose:
                    logger.info(f"Time for dimension {dim}: {time.time() - level_start_time:.4f}s")
                    logger.info(f"Total search time: {time.time() - start_time:.4f}s")
                
                # Map to external IDs if needed
                if self.id_mapping is not None:
                    results = self.id_mapping[results]
                
                if return_distances:
                    return results, distances
                else:
                    return results
            
            # Filter candidates for the next level
            if verbose:
                logger.info(f"Filtering in {dim}D: {len(candidates)} -> {k_level} candidates")
            
            # Search among current candidates
            new_candidates, _ = self._search_dimension(query_dim, dim, k_level, candidates)
            candidates = new_candidates
            
            if verbose:
                logger.info(f"Time for dimension {dim}: {time.time() - level_start_time:.4f}s")
        
        # If we somehow exit the loop without returning
        raise RuntimeError("Search loop exited without returning results")
    
    def _calculate_k_values(self, k: int, dimensions: List[int]) -> List[int]:
        """
        Calculate how many candidates to keep at each cascade level.
        
        Args:
            k: Final number of results to return
            dimensions: List of dimensions in the cascade
            
        Returns:
            List of k values for each level
        """
        k_values = []
        current_k = k
        
        # For all dimensions except the last, multiply k by filter_ratio
        for i in range(len(dimensions) - 1):
            current_k = max(k, int(current_k / self.filter_ratio))
            k_values.append(current_k)
        
        # For the last dimension, use k
        k_values.append(k)
        
        # Reverse order (largest k first, k at the end)
        return list(reversed(k_values))
    
    def _search_dimension(
        self,
        query: np.ndarray,
        dim: int,
        k: int,
        candidates: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors in a specific dimension.
        
        Args:
            query: Query vector of shape (dim,)
            dim: Dimension to search in
            k: Number of nearest neighbors to retrieve
            candidates: Optional candidate indices to search among
            
        Returns:
            Tuple of (indices, distances) of nearest neighbors
        """
        if len(query.shape) > 1:
            query = query.reshape(-1)
            
        # Adjust k to not exceed number of vectors
        k_adjusted = min(k, len(self.vector_dict[dim]) if candidates is None else len(candidates))
        
        if self.use_faiss and faiss is not None and isinstance(self.indices[dim], faiss.Index):
            # Prepare query
            if self.metric == "cosine":
                # Normalize query for cosine similarity
                query_norm = np.linalg.norm(query)
                if query_norm > 0:
                    query = query / query_norm
            
            # Convert to correct format
            query_f32 = query.astype(np.float32).reshape(1, -1)
            
            if candidates is None:
                # Search entire index
                distances, indices = self.indices[dim].search(query_f32, k_adjusted)
                return indices[0], distances[0]
            else:
                # Search only among candidates
                vectors = self.vector_dict[dim][candidates]
                
                if self.metric == "cosine":
                    # Normalize vectors for cosine similarity
                    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
                    mask = norms > 0
                    vectors[mask] = vectors[mask] / norms[mask]
                
                # Create temporary index with candidate vectors
                if self.metric == "cosine":
                    tmp_index = faiss.IndexFlatIP(dim)
                else:
                    tmp_index = faiss.IndexFlatL2(dim)
                
                tmp_index.add(vectors.astype(np.float32))
                
                # Search
                distances, local_indices = tmp_index.search(query_f32, k_adjusted)
                
                # Map local indices back to original indices
                global_indices = candidates[local_indices[0]]
                
                return global_indices, distances[0]
        else:
            # Use scikit-learn NearestNeighbors
            if candidates is None:
                # Search entire index
                distances, indices = self.indices[dim].kneighbors(
                    query.reshape(1, -1), n_neighbors=k_adjusted
                )
                return indices[0], distances[0]
            else:
                # Search only among candidates
                vectors = self.vector_dict[dim][candidates]
                tmp_nn = NearestNeighbors(metric=self.metric)
                tmp_nn.fit(vectors)
                distances, local_indices = tmp_nn.kneighbors(
                    query.reshape(1, -1), n_neighbors=k_adjusted
                )
                
                # Map local indices back to original indices
                global_indices = candidates[local_indices[0]]
                
                return global_indices, distances[0]
    
    def _compute_distances(self, query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        """
        Compute distances between query and vectors using the specified metric.
        
        Args:
            query: Query vector of shape (dim,)
            vectors: Document vectors of shape (n_docs, dim)
            
        Returns:
            Array of distances
        """
        if self.metric == "cosine":
            # Compute cosine distances (1 - cosine similarity)
            query_norm = np.linalg.norm(query)
            vector_norms = np.linalg.norm(vectors, axis=1)
            
            # Handle zero norms
            valid_mask = (query_norm > 0) & (vector_norms > 0)
            
            if not np.any(valid_mask):
                # All norms are zero, return zeros
                return np.zeros(len(vectors))
            
            # Initialize distances with ones (maximum cosine distance)
            distances = np.ones(len(vectors))
            
            # Compute similarities only for valid vectors
            if query_norm > 0:
                dot_products = np.dot(vectors[valid_mask], query)
                similarities = dot_products / (vector_norms[valid_mask] * query_norm)
                distances[valid_mask] = 1.0 - similarities
            
            return distances
        elif self.metric == "l2" or self.metric == "euclidean":
            # Compute Euclidean distances
            return np.sqrt(np.sum((vectors - query) ** 2, axis=1))
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")


class BruteForceSearch:
    """
    Brute force search implementation for baseline comparison.
    
    This class implements exact nearest neighbor search using either
    scikit-learn or FAISS for the entire vector set.
    """
    
    def __init__(
        self,
        vectors: np.ndarray,
        metric: str = "cosine",
        use_faiss: bool = True,
        id_mapping: Optional[np.ndarray] = None
    ):
        """
        Initialize the brute force search with document vectors.
        
        Args:
            vectors: Document vectors of shape (n_docs, dim)
            metric: Distance metric to use ('cosine', 'l2', etc.)
            use_faiss: Whether to use FAISS for indexing (faster for large datasets)
            id_mapping: Optional mapping from internal IDs to external IDs
        """
        self.vectors = vectors
        self.metric = metric
        self.use_faiss = use_faiss
        self.id_mapping = id_mapping
        
        # Create index
        self.index = self._create_index()
        
        logger.info(f"Initialized BruteForceSearch with {len(vectors)} vectors, metric={metric}")
    
    def _create_index(self) -> Any:
        """
        Create index for efficient search.
        
        Returns:
            Search index (FAISS or scikit-learn NearestNeighbors)
        """
        if self.use_faiss and faiss is not None:
            # Use FAISS for fast approximate search
            dim = self.vectors.shape[1]
            
            if self.metric == "cosine":
                # Normalize vectors for cosine similarity
                norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
                mask = norms > 0
                normalized_vectors = np.copy(self.vectors)
                normalized_vectors[mask] = normalized_vectors[mask] / norms[mask]
                
                # Create FAISS index
                index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
                index.add(normalized_vectors.astype(np.float32))
            else:
                # L2 distance (Euclidean)
                index = faiss.IndexFlatL2(dim)
                index.add(self.vectors.astype(np.float32))
            
            return index
        else:
            # Use scikit-learn for exact search
            nn_algo = 'brute' if len(self.vectors) < 10000 else 'auto'
            nn = NearestNeighbors(algorithm=nn_algo, metric=self.metric)
            nn.fit(self.vectors)
            return nn
    
    def search(
        self,
        query: np.ndarray,
        k: int,
        return_distances: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Perform brute force search for the nearest neighbors of a query vector.
        
        Args:
            query: Query vector of shape (dim,)
            k: Number of nearest neighbors to retrieve
            return_distances: Whether to return distances along with indices
            
        Returns:
            Indices of nearest neighbors, or tuple of (indices, distances) if return_distances=True
        """
        if len(query.shape) > 1:
            query = query.reshape(-1)
        
        # Adjust k to not exceed number of vectors
        k_adjusted = min(k, len(self.vectors))
        
        if self.use_faiss and faiss is not None and isinstance(self.index, faiss.Index):
            # Prepare query
            if self.metric == "cosine":
                # Normalize query for cosine similarity
                query_norm = np.linalg.norm(query)
                if query_norm > 0:
                    query = query / query_norm
            
            # Convert to correct format
            query_f32 = query.astype(np.float32).reshape(1, -1)
            
            # Search
            distances, indices = self.index.search(query_f32, k_adjusted)
            
            # Map to external IDs if needed
            if self.id_mapping is not None:
                indices = self.id_mapping[indices[0]]
            else:
                indices = indices[0]
            
            if return_distances:
                return indices, distances[0]
            else:
                return indices
        else:
            # Use scikit-learn NearestNeighbors
            distances, indices = self.index.kneighbors(
                query.reshape(1, -1), n_neighbors=k_adjusted
            )
            
            # Map to external IDs if needed
            if self.id_mapping is not None:
                indices = self.id_mapping[indices[0]]
            else:
                indices = indices[0]
            
            if return_distances:
                return indices, distances[0]
            else:
                return indices


def create_cascade_search(
    vector_dict: Dict[int, np.ndarray],
    dimensions: List[int],
    filter_ratio: float = 0.1,
    metric: str = "cosine",
    use_faiss: bool = True,
    id_mapping: Optional[np.ndarray] = None
) -> Callable:
    """
    Create a cascade search function for easy usage in benchmarks.
    
    Args:
        vector_dict: Dictionary mapping dimensions to document vectors
        dimensions: List of dimensions to use in the cascade
        filter_ratio: Ratio of candidates to keep at each level
        metric: Distance metric to use
        use_faiss: Whether to use FAISS for indexing
        id_mapping: Optional mapping from internal IDs to external IDs
        
    Returns:
        Search function that takes a query vector and k as input
    """
    # Create search object
    cascade = CascadeSearch(
        vector_dict=vector_dict,
        dimensions=dimensions,
        filter_ratio=filter_ratio,
        metric=metric,
        use_faiss=use_faiss,
        id_mapping=id_mapping
    )
    
    # Return search function
    def search_function(query, k, return_distances=False):
        return cascade.search(query, k, return_distances)
    
    return search_function


def create_brute_force_search(
    vectors: np.ndarray,
    metric: str = "cosine",
    use_faiss: bool = True,
    id_mapping: Optional[np.ndarray] = None
) -> Callable:
    """
    Create a brute force search function for baseline comparison.
    
    Args:
        vectors: Document vectors
        metric: Distance metric to use
        use_faiss: Whether to use FAISS for indexing
        id_mapping: Optional mapping from internal IDs to external IDs
        
    Returns:
        Search function that takes a query vector and k as input
    """
    # Create search object
    brute_force = BruteForceSearch(
        vectors=vectors,
        metric=metric,
        use_faiss=use_faiss,
        id_mapping=id_mapping
    )
    
    # Return search function
    def search_function(query, k, return_distances=False):
        return brute_force.search(query, k, return_distances)
    
    return search_function 