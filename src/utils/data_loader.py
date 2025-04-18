"""
Data loading and preprocessing utilities for dimensional cascade search.

This module provides functions to load vector data from files and perform
dimension reduction operations to create multi-dimensional embeddings.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.decomposition import PCA
import logging
import pickle
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_vectors_from_file(filepath: str) -> np.ndarray:
    """
    Load vectors from file based on extension.
    
    Supports .npy, .npz, .csv, .tsv, and .pickle formats.
    
    Args:
        filepath: Path to vector file
        
    Returns:
        Numpy array of vectors
    """
    extension = os.path.splitext(filepath)[1].lower()
    
    if extension == '.npy':
        return np.load(filepath)
    elif extension == '.npz':
        data = np.load(filepath)
        # Assume the first array in the npz file
        return data[list(data.keys())[0]]
    elif extension == '.csv':
        return pd.read_csv(filepath, header=None).values
    elif extension == '.tsv':
        return pd.read_csv(filepath, sep='\t', header=None).values
    elif extension in ['.pkl', '.pickle']:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file extension: {extension}")

def reduce_dimensions(vectors: np.ndarray, 
                     dimensions: List[int],
                     method: str = 'pca',
                     random_state: int = 42) -> Dict[int, np.ndarray]:
    """
    Reduce vectors to multiple dimensions using specified method.
    
    Args:
        vectors: Original vectors (n_samples, n_features)
        dimensions: List of target dimensions (in increasing order)
        method: Dimension reduction method ('pca' only for now)
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary mapping dimensions to reduced vectors
    """
    logger.info(f"Reducing dimensions using {method}...")
    
    # Validate input
    if not vectors.ndim == 2:
        raise ValueError("Input vectors must be 2D array")
        
    # Sort dimensions in ascending order
    dimensions = sorted(dimensions)
    
    # Check if dimensions are valid
    original_dim = vectors.shape[1]
    if dimensions[-1] > original_dim:
        raise ValueError(f"Target dimension {dimensions[-1]} exceeds original dimension {original_dim}")
    
    # Normalize vectors if not already
    norms = np.linalg.norm(vectors, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-5):
        logger.info("Normalizing input vectors...")
        vectors = vectors / norms[:, np.newaxis]
    
    # Prepare result dictionary with reduced vectors
    result = {}
    
    if method.lower() == 'pca':
        # Fit PCA on the original vectors with max dimension
        pca = PCA(n_components=dimensions[-1], random_state=random_state)
        pca.fit(vectors)
        
        # Transform vectors for each dimension
        for dim in dimensions:
            if dim == original_dim:
                # No need to reduce if same as original
                result[dim] = vectors
            else:
                # Take the first 'dim' components from PCA
                reduced = pca.transform(vectors)[:, :dim]
                
                # Normalize again after reduction
                reduced = reduced / np.linalg.norm(reduced, axis=1, keepdims=True)
                
                result[dim] = reduced
                logger.info(f"Reduced to {dim} dimensions, shape: {reduced.shape}")
    else:
        raise ValueError(f"Unsupported dimension reduction method: {method}")
    
    return result

def load_and_reduce(filepath: str, 
                   dimensions: List[int],
                   cache_dir: Optional[str] = None,
                   method: str = 'pca',
                   random_state: int = 42,
                   force_recompute: bool = False) -> Dict[int, np.ndarray]:
    """
    Load vectors from file and reduce to multiple dimensions.
    
    Caches reduced vectors to disk if cache_dir is specified.
    
    Args:
        filepath: Path to vector file
        dimensions: List of target dimensions (in increasing order)
        cache_dir: Directory to cache reduced vectors
        method: Dimension reduction method ('pca' only for now)
        random_state: Random seed for reproducibility
        force_recompute: Whether to force recomputation even if cache exists
        
    Returns:
        Dictionary mapping dimensions to reduced vectors
    """
    # Create cache directory if specified and not exists
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    # Load original vectors
    logger.info(f"Loading vectors from {filepath}...")
    vectors = load_vectors_from_file(filepath)
    logger.info(f"Loaded vectors with shape {vectors.shape}")
    
    # Sort dimensions
    dimensions = sorted(dimensions)
    
    # Check if original dimension is in the list
    original_dim = vectors.shape[1]
    if original_dim not in dimensions:
        # Add original dimension to the list
        dimensions = sorted(dimensions + [original_dim])
    
    result = {}
    dims_to_compute = []
    
    # Check if cache exists for each dimension
    if cache_dir:
        for dim in dimensions:
            cache_path = os.path.join(cache_dir, f"{os.path.basename(filepath)}.{dim}d.npy")
            
            if os.path.exists(cache_path) and not force_recompute:
                # Load from cache
                logger.info(f"Loading {dim}D vectors from cache: {cache_path}")
                result[dim] = np.load(cache_path)
            else:
                dims_to_compute.append(dim)
    else:
        dims_to_compute = dimensions
    
    # If original dimension is in dims_to_compute, add it directly
    if original_dim in dims_to_compute:
        result[original_dim] = vectors
        dims_to_compute.remove(original_dim)
    
    # Compute dimensions that are not cached
    if dims_to_compute:
        # Only reduce dimensions that are not already cached or available
        reduced = reduce_dimensions(
            vectors, 
            dims_to_compute,
            method=method,
            random_state=random_state
        )
        
        # Update result and cache
        for dim, reduced_vectors in reduced.items():
            result[dim] = reduced_vectors
            
            # Cache if directory specified
            if cache_dir:
                cache_path = os.path.join(cache_dir, f"{os.path.basename(filepath)}.{dim}d.npy")
                logger.info(f"Caching {dim}D vectors to {cache_path}")
                np.save(cache_path, reduced_vectors)
    
    return result

def generate_synthetic_data(n_samples: int,
                           n_features: int,
                           dimensions: List[int],
                           n_clusters: int = 10,
                           random_state: int = 42) -> Dict[int, np.ndarray]:
    """
    Generate synthetic data for testing.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features (original dimension)
        dimensions: List of target dimensions
        n_clusters: Number of clusters in the data
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary mapping dimensions to synthetic vectors
    """
    np.random.seed(random_state)
    
    # Generate cluster centers
    centers = np.random.randn(n_clusters, n_features)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    
    # Assign samples to clusters
    cluster_assignments = np.random.choice(n_clusters, size=n_samples)
    
    # Generate samples around cluster centers
    vectors = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        # Get assigned center
        center = centers[cluster_assignments[i]]
        
        # Add noise
        noise = np.random.randn(n_features) * 0.1
        
        # Combine and normalize
        vector = center + noise
        vector = vector / np.linalg.norm(vector)
        
        vectors[i] = vector
    
    # Reduce to multiple dimensions
    return reduce_dimensions(vectors, dimensions, random_state=random_state)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Loader Demo")
    parser.add_argument("--file", type=str, help="Path to vector file")
    parser.add_argument("--dims", type=str, default="32,64,128,256", help="Comma-separated dimensions")
    parser.add_argument("--cache_dir", type=str, default="cache", help="Cache directory")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic data instead of loading from file")
    parser.add_argument("--n_samples", type=int, default=1000, help="Number of samples for synthetic data")
    parser.add_argument("--n_features", type=int, default=768, help="Number of features for synthetic data")
    
    args = parser.parse_args()
    
    # Parse dimensions
    dimensions = [int(d) for d in args.dims.split(",")]
    
    if args.synthetic:
        # Generate synthetic data
        vector_dict = generate_synthetic_data(
            n_samples=args.n_samples,
            n_features=args.n_features,
            dimensions=dimensions
        )
        
        print(f"Generated synthetic data with {args.n_samples} samples")
        for dim, vectors in vector_dict.items():
            print(f"  {dim}D vectors shape: {vectors.shape}")
    elif args.file:
        # Load and reduce vectors from file
        vector_dict = load_and_reduce(
            filepath=args.file,
            dimensions=dimensions,
            cache_dir=args.cache_dir
        )
        
        print(f"Loaded and reduced vectors from {args.file}")
        for dim, vectors in vector_dict.items():
            print(f"  {dim}D vectors shape: {vectors.shape}")
    else:
        print("Either --file or --synthetic must be specified") 