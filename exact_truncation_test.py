#!/usr/bin/env python
"""
Exact truncation test for dimensional cascade.

This script demonstrates the exact truncation behavior for dimension reduction,
showing explicitly how vectors are truncated and normalized during the process.
"""

import numpy as np
import logging

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

def demonstration():
    """
    Run a clear demonstration of vector truncation in dimensional cascade.
    """
    # Example vector - exactly as provided in the request
    original_vector = np.array([0.123, 0.234, 0.259, 0.892])
    
    print(f"Original vector: {original_vector}")
    
    # Normalized original vector
    normalized_vector = normalize_vector(original_vector)
    print(f"Normalized original vector: {normalized_vector}")
    
    # For 2D truncation
    target_dim = 2
    
    # Simple truncation - just take the first N dimensions
    truncated_vector = original_vector[:target_dim]
    print(f"Truncated to {target_dim}D (without normalization): {truncated_vector}")
    
    # Normalized truncated vector (this is the proper result)
    normalized_truncated = normalize_vector(truncated_vector)
    print(f"Truncated to {target_dim}D (with normalization): {normalized_truncated}")
    
    # Let's verify the properties
    print("\nVerification:")
    print("-" * 60)
    
    # Cosine similarity between original[:2] and truncated (normalized)
    truncated_original = normalize_vector(original_vector[:target_dim])
    similarity = np.dot(truncated_original, normalized_truncated)
    print(f"Cosine similarity between original[:2] and truncated (normalized): {similarity:.6f}")
    
    # Difference between original[:2] and truncated (both normalized)
    difference = np.linalg.norm(truncated_original - normalized_truncated)
    print(f"L2 difference between original[:2] and truncated (both normalized): {difference:.6f}")
    
    # Explain exactly what happens during truncation
    print("\nWhat happens during truncation:")
    print("-" * 60)
    print("1. We extract the first N elements of the vector")
    print(f"   From {original_vector} → {original_vector[:target_dim]}")
    print("2. We normalize the truncated vector to ensure it has unit length")
    print(f"   From {original_vector[:target_dim]} → {normalized_truncated}")
    print(f"   (Normalization factor: {np.linalg.norm(original_vector[:target_dim]):.6f})")
    
    # Show more examples
    print("\nMore examples:")
    print("-" * 60)
    
    examples = [
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
    ]
    
    for i, example in enumerate(examples):
        print(f"Example {i+1}: {example}")
        trunc = example[:target_dim]
        norm_trunc = normalize_vector(trunc)
        print(f"  Truncated to {target_dim}D: {trunc}")
        print(f"  Normalized result: {norm_trunc}")
        print()

if __name__ == "__main__":
    logger.info("Starting exact truncation demonstration")
    demonstration()
    logger.info("Demonstration completed") 