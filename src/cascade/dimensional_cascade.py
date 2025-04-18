"""
Dimensional Cascade implementation for efficient multi-dimensional vector retrieval.

This module provides the core functionality for creating and using a dimensional cascade
for efficient retrieval of high-dimensional vector embeddings.
"""

import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
import os
import json
from sklearn.decomposition import PCA
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DimensionalCascade:
    """
    Implements the Dimensional Cascade approach for efficient vector retrieval.
    
    This class manages a cascade of progressively more precise vector representations
    at different dimensionality levels.
    """
    
    def __init__(self, dimensions: List[int], model_dir: str = None):
        """
        Initialize the dimensional cascade with specified dimension levels.
        
        Args:
            dimensions: List of dimensions in descending order (e.g., [512, 256, 128, 64, 32, 16, 8, 4, 2, 1])
            model_dir: Directory containing pre-trained dimension reduction models
        """
        self.dimensions = sorted(dimensions, reverse=True)
        self.model_dir = model_dir
        self.dimension_models = {}
        self.indices = {}
        
        # Load models if directory is provided
        if model_dir and os.path.exists(model_dir):
            self._load_models()
    
    def _load_models(self):
        """Load pre-trained dimension reduction models from the model directory."""
        logger.info(f"Loading dimension reduction models from {self.model_dir}")
        
        for i, dim in enumerate(self.dimensions[1:], 1):  # Skip highest dimension
            model_path = os.path.join(self.model_dir, f"pca_{self.dimensions[i-1]}to{dim}.joblib")
            if os.path.exists(model_path):
                self.dimension_models[dim] = joblib.load(model_path)
                logger.info(f"Loaded PCA model for {self.dimensions[i-1]}→{dim}")
            else:
                logger.warning(f"Model not found: {model_path}")
    
    def create_cascade_models(self, high_dim_vectors: np.ndarray, output_dir: str = None):
        """
        Create cascade of dimension reduction models from high-dimensional vectors.
        
        Args:
            high_dim_vectors: High-dimensional vectors to use for training PCA models
            output_dir: Directory to save the trained models
        
        Returns:
            Dictionary of trained dimension reduction models
        """
        if high_dim_vectors.shape[1] != self.dimensions[0]:
            raise ValueError(f"Input vectors must have dimension {self.dimensions[0]}, got {high_dim_vectors.shape[1]}")
        
        logger.info(f"Creating cascade models from {self.dimensions[0]}d vectors")
        
        # Create output directory if needed
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.model_dir = output_dir
        
        # Start with the highest dimension vectors
        vectors = high_dim_vectors
        
        # Create models for each dimension level
        for i in range(len(self.dimensions) - 1):
            from_dim = self.dimensions[i]
            to_dim = self.dimensions[i+1]
            
            logger.info(f"Training PCA model: {from_dim}d → {to_dim}d")
            
            # Create and fit PCA model
            pca = PCA(n_components=to_dim, random_state=42)
            pca.fit(vectors)
            
            # Save model
            self.dimension_models[to_dim] = pca
            
            if output_dir:
                model_path = os.path.join(output_dir, f"pca_{from_dim}to{to_dim}.joblib")
                joblib.dump(pca, model_path)
                logger.info(f"Saved model to {model_path}")
            
            # Transform vectors for next level
            vectors = pca.transform(vectors)
        
        return self.dimension_models
    
    def reduce_dimensions(self, vectors: np.ndarray, target_dim: int) -> np.ndarray:
        """
        Reduce vectors to the specified target dimension.
        
        Args:
            vectors: Input vectors to reduce
            target_dim: Target dimension to reduce to
            
        Returns:
            Reduced vectors
        """
        if target_dim not in self.dimensions:
            raise ValueError(f"Target dimension {target_dim} not in cascade dimensions {self.dimensions}")
        
        # If already at target dimension
        input_dim = vectors.shape[1]
        if input_dim == target_dim:
            return vectors
        
        # Find position in dimension hierarchy
        input_pos = self.dimensions.index(input_dim) if input_dim in self.dimensions else -1
        target_pos = self.dimensions.index(target_dim)
        
        if input_pos == -1 or input_pos > target_pos:
            raise ValueError(f"Cannot reduce from {input_dim}d to {target_dim}d with available models")
        
        # Apply cascade of transformations
        result = vectors.copy()
        for i in range(input_pos, target_pos):
            from_dim = self.dimensions[i]
            to_dim = self.dimensions[i+1]
            
            if to_dim not in self.dimension_models:
                raise ValueError(f"No dimension reduction model for {from_dim}d → {to_dim}d")
            
            result = self.dimension_models[to_dim].transform(result)
        
        return result
    
    def create_index(self, ids: List[str], vectors_dict: Dict[int, np.ndarray], index_dir: str = None):
        """
        Create indices at each dimension level.
        
        Args:
            ids: Document IDs corresponding to the vectors
            vectors_dict: Dictionary mapping dimension to vectors array
            index_dir: Directory to save the indices
        """
        # Validate inputs
        for dim in vectors_dict:
            if dim not in self.dimensions:
                raise ValueError(f"Dimension {dim} not in cascade dimensions {self.dimensions}")
            if len(ids) != vectors_dict[dim].shape[0]:
                raise ValueError(f"Number of IDs ({len(ids)}) doesn't match number of vectors ({vectors_dict[dim].shape[0]})")
        
        # Create simple in-memory indices for demonstration
        # In a real implementation, this would use a vector database or optimized index
        for dim, vectors in vectors_dict.items():
            self.indices[dim] = {
                'ids': ids,
                'vectors': vectors
            }
            
            logger.info(f"Created {dim}d index with {len(ids)} vectors")
            
            # Save indices if directory provided
            if index_dir:
                os.makedirs(index_dir, exist_ok=True)
                index_path = os.path.join(index_dir, f"index_{dim}d.npz")
                np.savez(index_path, ids=ids, vectors=vectors)
                logger.info(f"Saved {dim}d index to {index_path}")
    
    def load_indices(self, index_dir: str):
        """
        Load indices from the specified directory.
        
        Args:
            index_dir: Directory containing saved indices
        """
        for dim in self.dimensions:
            index_path = os.path.join(index_dir, f"index_{dim}d.npz")
            if os.path.exists(index_path):
                with np.load(index_path, allow_pickle=True) as data:
                    self.indices[dim] = {
                        'ids': data['ids'].tolist(),
                        'vectors': data['vectors']
                    }
                logger.info(f"Loaded {dim}d index with {len(self.indices[dim]['ids'])} vectors")
            else:
                logger.warning(f"Index not found: {index_path}")
    
    def search(self, query_vector: np.ndarray, top_k: int = 10, 
               min_dimension: int = None, max_dimension: int = None,
               filter_ratio: float = 0.2) -> List[Dict[str, Any]]:
        """
        Search using the dimensional cascade approach.
        
        Args:
            query_vector: Query vector in the highest dimension
            top_k: Number of results to return
            min_dimension: Minimum dimension to use in cascade (default: lowest dimension)
            max_dimension: Maximum dimension to use in cascade (default: highest dimension)
            filter_ratio: Fraction of candidates to keep at each cascade level
            
        Returns:
            List of results with document IDs and similarity scores
        """
        # Set default dimensions if not specified
        if min_dimension is None:
            min_dimension = min(self.dimensions)
        if max_dimension is None:
            max_dimension = max(self.dimensions)
            
        # Validate dimensions
        if min_dimension not in self.dimensions or max_dimension not in self.dimensions:
            raise ValueError(f"Dimensions must be in {self.dimensions}")
        if min_dimension > max_dimension:
            raise ValueError(f"min_dimension ({min_dimension}) must be <= max_dimension ({max_dimension})")
        
        # Get sorted dimensions in our range
        cascade_dims = [d for d in self.dimensions if min_dimension <= d <= max_dimension]
        cascade_dims.sort()  # Ascending order for cascade
        
        # Check if indices exist
        for dim in cascade_dims:
            if dim not in self.indices:
                raise ValueError(f"No index available for {dim}d vectors")
        
        # Reduce query to each dimension level
        query_vectors = {}
        for dim in cascade_dims:
            if dim == query_vector.shape[1]:
                query_vectors[dim] = query_vector
            else:
                query_vectors[dim] = self.reduce_dimensions(query_vector.reshape(1, -1), dim).flatten()
        
        # Start with all documents as candidates
        candidates = set(self.indices[cascade_dims[0]]['ids'])
        logger.info(f"Starting cascade search with {len(candidates)} candidates")
        
        # Perform cascade search
        for i, dim in enumerate(cascade_dims):
            # Calculate number of candidates to keep
            if i == len(cascade_dims) - 1:
                # Final dimension: keep only top_k
                keep_k = min(top_k, len(candidates))
            else:
                # Intermediate dimension: apply filter ratio
                keep_k = max(top_k, int(len(candidates) * filter_ratio))
            
            # Get index for this dimension
            index = self.indices[dim]
            
            # Filter to candidates only
            candidate_indices = [index['ids'].index(doc_id) for doc_id in candidates]
            candidate_vectors = index['vectors'][candidate_indices]
            candidate_ids = [index['ids'][i] for i in candidate_indices]
            
            # Calculate similarities (dot product for normalized vectors)
            similarities = np.dot(candidate_vectors, query_vectors[dim])
            
            # Get top candidates for next level
            top_indices = np.argsort(similarities)[-keep_k:][::-1]
            candidates = set([candidate_ids[i] for i in top_indices])
            
            logger.info(f"Dimension {dim}d: filtered to {len(candidates)} candidates")
        
        # Prepare final results using highest dimension
        final_dim = cascade_dims[-1]
        index = self.indices[final_dim]
        final_results = []
        
        for doc_id in candidates:
            vec_index = index['ids'].index(doc_id)
            similarity = np.dot(index['vectors'][vec_index], query_vectors[final_dim])
            final_results.append({
                'id': doc_id,
                'score': float(similarity)
            })
        
        # Sort by score descending
        final_results.sort(key=lambda x: x['score'], reverse=True)
        return final_results[:top_k]


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Dimensional Cascade demo")
    parser.add_argument("--data_file", type=str, help="Path to file with sample vectors")
    parser.add_argument("--output_dir", type=str, default="models", help="Output directory for models")
    args = parser.parse_args()
    
    # Demo without actual data file
    if not args.data_file:
        print("No data file provided. Running with synthetic data.")
        
        # Create synthetic data
        dim = 512
        n_samples = 1000
        high_dim_vectors = np.random.randn(n_samples, dim)
        high_dim_vectors = high_dim_vectors / np.linalg.norm(high_dim_vectors, axis=1, keepdims=True)
        ids = [f"doc_{i}" for i in range(n_samples)]
        
        # Create cascade
        dimensions = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]
        cascade = DimensionalCascade(dimensions)
        
        # Create and save models
        cascade.create_cascade_models(high_dim_vectors, args.output_dir)
        
        # Create vectors at each dimension
        vectors_dict = {dim: high_dim_vectors}
        for d in dimensions[1:]:
            vectors_dict[d] = cascade.reduce_dimensions(high_dim_vectors, d)
        
        # Create indices
        cascade.create_index(ids, vectors_dict, args.output_dir)
        
        # Run sample search
        query = np.random.randn(dim)
        query = query / np.linalg.norm(query)
        
        results = cascade.search(query, top_k=5)
        print("\nSearch results:")
        for r in results:
            print(f"Document: {r['id']}, Score: {r['score']:.4f}") 