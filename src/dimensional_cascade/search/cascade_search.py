"""
Cascade search implementation for the Dimensional Cascade.
"""
from typing import Dict, List, Optional, Tuple, Any
import time

import numpy as np

from dimensional_cascade.models import ModelHierarchy
from dimensional_cascade.indexing import MultiResolutionIndex
from dimensional_cascade.core.cascade import CascadeConfig


class CascadeSearch:
    """Implements the progressive refinement search algorithm."""
    
    def __init__(
        self,
        model_hierarchy: ModelHierarchy,
        index: MultiResolutionIndex,
        config: CascadeConfig
    ):
        """Initialize the cascade search.
        
        Args:
            model_hierarchy: Model hierarchy for generating embeddings
            index: Multi-resolution index for searching
            config: Configuration for the cascade
        """
        self.model_hierarchy = model_hierarchy
        self.index = index
        self.config = config
        self._query_cache: Dict[str, Dict[int, np.ndarray]] = {}
    
    def search(
        self, 
        query: str,
        top_k: int = 100,
        min_dimension: Optional[int] = None,
        max_dimension: Optional[int] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search using the cascade approach.
        
        Args:
            query: Query text
            top_k: Number of results to return
            min_dimension: Minimum dimension to use (default: smallest in config)
            max_dimension: Maximum dimension to use (default: largest in config)
            
        Returns:
            List of (document, score) tuples
        """
        # Set default dimensions if not specified
        min_dimension = min_dimension or min(self.config.dimensions)
        max_dimension = max_dimension or max(self.config.dimensions)
        
        # Validate dimensions
        if min_dimension not in self.config.dimensions or max_dimension not in self.config.dimensions:
            valid_dims = ", ".join(map(str, self.config.dimensions))
            raise ValueError(f"Dimensions must be one of: {valid_dims}")
        
        if min_dimension > max_dimension:
            raise ValueError(f"min_dimension ({min_dimension}) must be <= max_dimension ({max_dimension})")
        
        # Get query embeddings at all dimensions
        query_embeddings = self._get_query_embeddings(query)
        
        # Get the cascade dimensions in ascending order (from smallest to largest)
        cascade_dimensions = [d for d in sorted(self.config.dimensions) 
                            if min_dimension <= d <= max_dimension]
        
        # Start with the smallest dimension
        start_dim = min(cascade_dimensions)
        candidates = self._search_dimension(query_embeddings[start_dim], start_dim, 
                                          self.config.transition_thresholds[start_dim])
        
        # Timing and metrics (for debug/analysis)
        timing = {start_dim: time.time()}
        n_candidates = {start_dim: len(candidates)}
        
        # Progress through dimensions
        current_dim = start_dim
        for next_dim in cascade_dimensions:
            if next_dim <= current_dim:
                continue
                
            # Skip to next dimension if we're below the configured threshold
            if len(candidates) <= top_k and self.config.early_termination:
                break
                
            # Determine how many candidates to keep
            limit = min(self.config.transition_thresholds[next_dim], len(candidates))
            
            # Refine with the next dimension
            candidates = self._refine_results(
                query_embeddings[next_dim], 
                next_dim,
                candidates, 
                limit
            )
            
            # Update timing and metrics
            timing[next_dim] = time.time()
            n_candidates[next_dim] = len(candidates)
            current_dim = next_dim
        
        # Get final results
        final_dim = max_dimension
        results = self._get_final_results(query_embeddings[final_dim], final_dim, candidates, top_k)
        
        # Calculate elapsed times (for debug/analysis)
        prev_time = 0
        for dim in sorted(timing.keys()):
            elapsed = timing[dim] - prev_time
            prev_time = timing[dim]
            # Could log these metrics or return them with results
        
        return results
    
    def _get_query_embeddings(self, query: str) -> Dict[int, np.ndarray]:
        """Get query embeddings at all dimensions.
        
        Args:
            query: Query text
            
        Returns:
            Dictionary mapping from dimension to embedding
        """
        # Check cache first
        if query in self._query_cache and self.config.cache_embeddings:
            return self._query_cache[query]
        
        # Generate embeddings at all dimensions
        embeddings = {}
        for dim in self.config.dimensions:
            embeddings[dim] = self.model_hierarchy.embed(query, dimension=dim)
        
        # Cache for future use
        if self.config.cache_embeddings:
            self._query_cache[query] = embeddings
            
            # Limit cache size (simple LRU-like approach)
            if len(self._query_cache) > 1000:
                # Remove oldest entry
                self._query_cache.pop(next(iter(self._query_cache)))
        
        return embeddings
    
    def _search_dimension(
        self, 
        query_embedding: np.ndarray, 
        dimension: int, 
        limit: int
    ) -> List[Tuple[int, float]]:
        """Search at a specific dimension.
        
        Args:
            query_embedding: Query embedding
            dimension: Dimension to search in
            limit: Number of results to return
            
        Returns:
            List of (document_id, score) tuples
        """
        doc_indices, distances = self.index.search(query_embedding, dimension, limit)
        
        # Convert distances to similarity scores (FAISS returns squared distances)
        # Lower distance = higher similarity
        similarity_scores = [1.0 / (1.0 + dist) for dist in distances]
        
        # Return document indices with their scores
        return list(zip(doc_indices, similarity_scores))
    
    def _refine_results(
        self, 
        query_embedding: np.ndarray, 
        dimension: int,
        candidates: List[Tuple[int, float]], 
        limit: int
    ) -> List[Tuple[int, float]]:
        """Refine results using a higher dimension.
        
        Args:
            query_embedding: Query embedding at the current dimension
            dimension: Current dimension
            candidates: List of (document_id, score) tuples from previous dimension
            limit: Number of results to return
            
        Returns:
            Refined list of (document_id, score) tuples
        """
        # Extract document IDs
        doc_ids = [doc_id for doc_id, _ in candidates]
        
        # Get documents
        documents = self.index.get_documents(doc_ids)
        
        # Re-score with current dimension
        new_scores = []
        for i, doc in enumerate(documents):
            # Get document text
            text = doc.get('text', '')
            if not text:
                continue
                
            # Generate embedding at current dimension
            doc_embedding = self.model_hierarchy.embed(text, dimension)
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            
            # Combine with previous score (with weight favoring higher dimensions)
            prev_score = candidates[i][1]
            combined_score = 0.2 * prev_score + 0.8 * similarity
            
            new_scores.append((doc_ids[i], combined_score))
        
        # Sort by score and return top results
        new_scores.sort(key=lambda x: x[1], reverse=True)
        return new_scores[:limit]
    
    def _get_final_results(
        self, 
        query_embedding: np.ndarray, 
        dimension: int,
        candidates: List[Tuple[int, float]], 
        limit: int
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Get final results with document objects.
        
        Args:
            query_embedding: Query embedding at the final dimension
            dimension: Final dimension
            candidates: List of (document_id, score) tuples from previous refinement
            limit: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        # Extract document IDs
        doc_ids = [doc_id for doc_id, _ in candidates]
        
        # Get documents
        documents = self.index.get_documents(doc_ids)
        
        # Construct result list
        results = []
        for i, doc in enumerate(documents):
            if i >= len(candidates):
                break
            results.append((doc, candidates[i][1]))
        
        # Sort by score and limit
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity score
        """
        # Ensure vectors are flattened
        a = a.flatten()
        b = b.flatten()
        
        # Calculate cosine similarity
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b) 