#!/usr/bin/env python
"""
Test script for dimensional cascade.

This script tests the basic functionality of the dimensional cascade approach
by loading vectors, creating a cascade model, and performing searches.
"""

import os
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import from the project modules
try:
    from src.utils.data_loader import load_vectors_from_file, generate_synthetic_data
    from src.utils.evaluation import evaluate_recall, evaluate_search
except ImportError:
    logger.warning("Could not import from src, trying direct imports")
    try:
        from utils.data_loader import load_vectors_from_file, generate_synthetic_data
        from utils.evaluation import evaluate_recall, evaluate_search
    except ImportError:
        logger.error("Failed to import required modules. Make sure you're running from the correct directory.")
        raise

def create_sample_vectors(n_samples: int, n_features: int, n_clusters: int = 10, save_path: str = None) -> np.ndarray:
    """
    Create sample vectors for testing.
    
    Args:
        n_samples: Number of vectors to generate
        n_features: Dimensionality of vectors
        n_clusters: Number of clusters to generate
        save_path: Optional path to save the vectors
    
    Returns:
        Generated vectors as numpy array
    """
    logger.info(f"Generating {n_samples} sample vectors with {n_features} dimensions")
    
    # Generate clustered data
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)
    
    # Normalize the vectors
    from sklearn.preprocessing import normalize
    X = normalize(X)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, X)
        logger.info(f"Saved sample vectors to {save_path}")
    
    return X

def simple_vector_search(query_vectors: np.ndarray, index_vectors: np.ndarray, k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform a simple brute-force search.
    
    Args:
        query_vectors: Query vectors
        index_vectors: Index vectors to search against
        k: Number of nearest neighbors to return
    
    Returns:
        Tuple of (indices, distances)
    """
    # Compute dot product between queries and index vectors
    similarities = query_vectors @ index_vectors.T
    
    # Get top k indices and distances
    indices = np.argsort(-similarities, axis=1)[:, :k]
    distances = -np.sort(-similarities, axis=1)[:, :k]
    
    return indices, distances

def test_dimension_reduction():
    """Test the dimension reduction functionality."""
    logger.info("Testing dimension reduction...")
    
    # Generate sample vectors
    vectors = create_sample_vectors(1000, 768)
    
    # Define dimensions to test
    dimensions = [32, 64, 128, 256]
    
    try:
        # Reduce dimensions
        from src.utils.data_loader import reduce_dimensions
        reduced_vectors = reduce_dimensions(vectors, dimensions)
        
        # Verify shapes
        for dim, reduced in reduced_vectors.items():
            logger.info(f"Reduced to {dim} dimensions: shape = {reduced.shape}")
            assert reduced.shape == (vectors.shape[0], dim), f"Expected shape {(vectors.shape[0], dim)}, got {reduced.shape}"
        
        logger.info("✅ Dimension reduction test passed")
        return reduced_vectors
        
    except Exception as e:
        logger.error(f"❌ Dimension reduction test failed: {e}")
        raise

def test_cascade_search(vectors_by_dim: Dict[int, np.ndarray], num_queries: int = 100, k: int = 10):
    """
    Test cascade search performance.
    
    Args:
        vectors_by_dim: Dictionary of vectors by dimension
        num_queries: Number of query vectors to use
        k: Number of neighbors to retrieve
    """
    logger.info("Testing cascade search...")
    
    # Get dimensions in ascending order
    dimensions = sorted(vectors_by_dim.keys())
    
    # Use a subset of vectors as queries
    np.random.seed(42)
    query_indices = np.random.choice(vectors_by_dim[dimensions[0]].shape[0], num_queries, replace=False)
    
    # Get ground truth using the highest dimension
    highest_dim = max(dimensions)
    query_vectors = vectors_by_dim[highest_dim][query_indices]
    index_vectors = vectors_by_dim[highest_dim]
    
    # Compute ground truth
    gt_indices, _ = simple_vector_search(query_vectors, index_vectors, k)
    
    # Test each dimension
    results = {}
    for dim in dimensions:
        logger.info(f"Testing search at dimension {dim}...")
        
        # Get query and index vectors for this dimension
        query_vecs = vectors_by_dim[dim][query_indices]
        index_vecs = vectors_by_dim[dim]
        
        # Measure search time
        start_time = time.time()
        indices, _ = simple_vector_search(query_vecs, index_vecs, k)
        search_time = time.time() - start_time
        
        # Calculate recall
        recall = np.mean([
            len(set(indices[i]).intersection(set(gt_indices[i]))) / k 
            for i in range(num_queries)
        ])
        
        logger.info(f"Dimension {dim}: Recall@{k} = {recall:.4f}, Time = {search_time:.4f}s")
        results[dim] = {'recall': recall, 'time': search_time}
    
    # Visualize results
    plot_results(results, k)

def plot_results(results: Dict[int, Dict[str, float]], k: int):
    """
    Plot search results.
    
    Args:
        results: Dictionary of results by dimension
        k: Number of neighbors used
    """
    plt.figure(figsize=(12, 5))
    
    # Extract data
    dimensions = sorted(results.keys())
    recalls = [results[dim]['recall'] for dim in dimensions]
    times = [results[dim]['time'] for dim in dimensions]
    
    # Plot recall
    plt.subplot(1, 2, 1)
    plt.plot(dimensions, recalls, 'o-', markersize=8)
    plt.xlabel('Dimensions')
    plt.ylabel(f'Recall@{k}')
    plt.title(f'Recall@{k} vs. Dimensions')
    plt.grid(True)
    
    # Plot time
    plt.subplot(1, 2, 2)
    plt.plot(dimensions, times, 'o-', markersize=8, color='orange')
    plt.xlabel('Dimensions')
    plt.ylabel('Search Time (s)')
    plt.title('Search Time vs. Dimensions')
    plt.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/dimension_comparison.png')
    logger.info("Saved results plot to results/dimension_comparison.png")
    
    plt.show()

def main():
    """Main function to run the tests."""
    logger.info("Starting dimensional cascade tests")
    
    # Check if we have test vectors, otherwise create them
    test_vectors_path = 'data/test_vectors.npy'
    
    if os.path.exists(test_vectors_path):
        logger.info(f"Loading test vectors from {test_vectors_path}")
        vectors = np.load(test_vectors_path)
    else:
        logger.info(f"Test vectors not found at {test_vectors_path}")
        vectors = create_sample_vectors(1000, 768, save_path=test_vectors_path)
    
    logger.info(f"Loaded vectors with shape: {vectors.shape}")
    
    # Run dimension reduction tests
    try:
        vectors_by_dim = test_dimension_reduction()
        
        # Run cascade search tests
        test_cascade_search(vectors_by_dim)
        
        logger.info("✅ All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Tests failed: {e}")

if __name__ == "__main__":
    main() 