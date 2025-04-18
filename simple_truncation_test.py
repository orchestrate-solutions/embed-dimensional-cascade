#!/usr/bin/env python
"""
Simple truncation test for dimensional cascade.

This script demonstrates the basic concept of truncation for dimension reduction,
showing how a high-dimensional vector is truncated to create lower-dimensional versions.
"""

import os
import numpy as np
import logging
from typing import List, Dict
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a vector to unit length.
    
    Args:
        vector: Vector to normalize
    
    Returns:
        Normalized vector
    """
    norm = np.linalg.norm(vector)
    if norm > 0:
        return vector / norm
    return vector

def create_sample_vector(dim: int = 1024, seed: int = 42) -> np.ndarray:
    """
    Create a sample vector with random values.
    
    Args:
        dim: Dimensionality of the vector
        seed: Random seed for reproducibility
    
    Returns:
        Normalized random vector
    """
    np.random.seed(seed)
    vector = np.random.rand(dim)
    return normalize_vector(vector)

def truncate_vector(vector: np.ndarray, dimensions: List[int]) -> Dict[int, np.ndarray]:
    """
    Create truncated versions of a vector at different dimensions.
    
    Args:
        vector: High-dimensional vector to truncate
        dimensions: List of target dimensions
    
    Returns:
        Dictionary mapping from dimension to truncated vectors
    """
    logger.info(f"Creating truncated vectors from {vector.shape[0]}-dimensional vector")
    
    result = {}
    for dim in dimensions:
        logger.info(f"Truncating to {dim} dimensions: {vector[:dim]}")
        
        # Simple truncation - take first N dimensions
        truncated = vector[:dim].copy()
        
        # Normalize to ensure unit vector
        truncated = normalize_vector(truncated)
        
        result[dim] = truncated
        
    return result

def reduce_with_pca(vector: np.ndarray, dimensions: List[int]) -> Dict[int, np.ndarray]:
    """
    Create PCA-reduced versions of a vector at different dimensions.
    This is a simplified simulation since PCA normally works on datasets.
    
    Args:
        vector: High-dimensional vector to reduce
        dimensions: List of target dimensions
    
    Returns:
        Dictionary mapping from dimension to reduced vectors
    """
    logger.info(f"Simulating PCA reduction from {vector.shape[0]}-dimensional vector")
    
    result = {}
    for dim in dimensions:
        # In this simplified example, we'll just create a new random vector of the target dimension
        # This simulates what a PCA model might return
        np.random.seed(42 + dim)  # Different seed for each dimension
        reduced = np.random.rand(dim)
        reduced = normalize_vector(reduced)
        
        logger.info(f"PCA-reduced to {dim} dimensions: {reduced}")
        result[dim] = reduced
        
    return result

def print_vector_comparison(original: np.ndarray, truncated: Dict[int, np.ndarray], dimensions: List[int]):
    """
    Print a comparison of the original vector and its truncated versions.
    
    Args:
        original: Original high-dimensional vector
        truncated: Dictionary of truncated vectors by dimension
        dimensions: List of dimensions to print
    """
    print("\nVector Comparison:")
    print("-" * 80)
    
    # Print original vector (first few dimensions)
    print(f"Original ({len(original)} dim): [{', '.join([f'{v:.4f}' for v in original[:10]])}...]")
    
    # Print truncated vectors
    for dim in dimensions:
        trunc_vec = truncated[dim]
        print(f"Truncated ({dim} dim): [{', '.join([f'{v:.4f}' for v in trunc_vec])}]")
        
        # Calculate and print the cosine similarity with the original
        # (using only the first 'dim' components of the original)
        truncated_orig = normalize_vector(original[:dim])
        similarity = np.dot(truncated_orig, trunc_vec)
        print(f"  Cosine similarity with original (first {dim} dims): {similarity:.6f}")
        
        # Calculate how different the truncated vector is from simply taking first N dimensions
        difference = np.linalg.norm(truncated_orig - trunc_vec)
        print(f"  L2 difference from original (first {dim} dims): {difference:.6f}")
        
        # Calculate if truncated values match the original (after normalization)
        matches = np.isclose(truncated_orig, trunc_vec, rtol=1e-5, atol=1e-5)
        match_percent = np.sum(matches) / dim * 100
        print(f"  Match with original (first {dim} dims): {match_percent:.2f}%")
        
        print()

def main():
    """Main function."""
    logger.info("Starting simple truncation test")
    
    # Create a sample vector
    logger.info("Creating sample vector")
    original_vector = create_sample_vector(dim=1024)
    
    # Create another explicit vector for demonstration
    logger.info("Creating explicit sample vector")
    explicit_vector = np.array([0.123, 0.234, 0.259, 0.892, 0.475, 0.834, 0.294, 0.182])
    explicit_vector = normalize_vector(explicit_vector)
    
    # Dimensions to test
    dimensions = [2, 4, 6]
    
    # Truncate the vectors
    truncated_vectors = truncate_vector(explicit_vector, dimensions)
    
    # Print comparison
    print_vector_comparison(explicit_vector, truncated_vectors, dimensions)
    
    # Create a more realistic test with the random vector
    dimensions = [32, 64, 128, 256, 512]
    truncated_realistic = truncate_vector(original_vector, dimensions)
    
    # Print comparison for realistic test
    print_vector_comparison(original_vector, truncated_realistic, dimensions)
    
    # Save outputs
    os.makedirs('results/truncation_test', exist_ok=True)
    
    # Save originals and truncated
    np.save('results/truncation_test/original_vector.npy', original_vector)
    np.save('results/truncation_test/explicit_vector.npy', explicit_vector)
    
    for dim, vec in truncated_vectors.items():
        np.save(f'results/truncation_test/explicit_truncated_{dim}dim.npy', vec)
    
    for dim, vec in truncated_realistic.items():
        np.save(f'results/truncation_test/random_truncated_{dim}dim.npy', vec)
        
    logger.info("✅ Truncation test completed successfully!")
    logger.info("Files saved to results/truncation_test/")

if __name__ == "__main__":
    main() 