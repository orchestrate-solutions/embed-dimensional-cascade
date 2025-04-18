#!/usr/bin/env python
"""
Comprehensive test for dimensional cascade training.

This script provides a complete test of the dimensional cascade approach by:
1. Generating high-dimensional embeddings (1024-dim)
2. Creating both:
   - Truncated embeddings at various dimensions as "expected values"
   - PCA-reduced embeddings as "model outputs"
3. Comparing these against each other to simulate training process
"""

import os
import numpy as np
import logging
import matplotlib.pyplot as plt
import time
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_high_dim_vectors(
    n_samples: int, 
    n_features: int = 1024, 
    n_clusters: int = 20, 
    save_path: str = None
) -> np.ndarray:
    """
    Create high-dimensional vectors for testing.
    
    Args:
        n_samples: Number of vectors to generate
        n_features: Dimensionality of vectors
        n_clusters: Number of clusters to generate
        save_path: Optional path to save the vectors
    
    Returns:
        Generated vectors as numpy array
    """
    logger.info(f"Generating {n_samples} vectors with {n_features} dimensions")
    
    # Generate clustered data
    from sklearn.datasets import make_blobs
    X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)
    
    # Normalize the vectors
    from sklearn.preprocessing import normalize
    X = normalize(X)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, X)
        logger.info(f"Saved vectors to {save_path}")
    
    return X

def truncate_vectors(vectors: np.ndarray, dimensions: List[int]) -> Dict[int, np.ndarray]:
    """
    Create truncated versions of vectors at different dimensions.
    These serve as the "expected outputs" in a dimensional cascade.
    
    Args:
        vectors: High-dimensional vectors to truncate
        dimensions: List of target dimensions
    
    Returns:
        Dictionary mapping from dimension to truncated vectors
    """
    logger.info(f"Creating truncated vectors from {vectors.shape[1]}-dimensional vectors")
    
    result = {}
    for dim in dimensions:
        logger.info(f"Truncating to {dim} dimensions...")
        
        # Simple truncation - take first N dimensions
        truncated = vectors[:, :dim].copy()
        
        # Normalize to ensure unit vectors
        from sklearn.preprocessing import normalize
        truncated = normalize(truncated)
        
        result[dim] = truncated
        
    return result

def reduce_dimensions(vectors: np.ndarray, dimensions: List[int]) -> Dict[int, np.ndarray]:
    """
    Reduce dimensions of vectors using PCA.
    These serve as the "model outputs" in a dimensional cascade.
    
    Args:
        vectors: Vectors to reduce dimensions for
        dimensions: List of target dimensions
    
    Returns:
        Dictionary mapping from dimension to reduced vectors
    """
    logger.info(f"Reducing vectors from {vectors.shape[1]} to {dimensions} using PCA")
    
    from sklearn.decomposition import PCA
    
    result = {}
    for dim in dimensions:
        logger.info(f"Reducing to {dim} dimensions...")
        pca = PCA(n_components=dim, random_state=42)
        reduced = pca.fit_transform(vectors)
        
        # Normalize the vectors
        from sklearn.preprocessing import normalize
        reduced = normalize(reduced)
        
        result[dim] = reduced
        
    return result

def calculate_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def evaluate_model_performance(
    truncated_vecs: Dict[int, np.ndarray],
    pca_vecs: Dict[int, np.ndarray],
    num_samples: int = 500,
    k: int = 10
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate the performance of PCA reduction vs simple truncation.
    
    Args:
        truncated_vecs: Dictionary of truncated vectors by dimension
        pca_vecs: Dictionary of PCA-reduced vectors by dimension
        num_samples: Number of query vectors to use for evaluation
        k: Number of neighbors to retrieve for recall calculation
    
    Returns:
        Dictionary of performance metrics by dimension
    """
    logger.info(f"Evaluating model performance using {num_samples} query vectors")
    
    results = {}
    dimensions = sorted(truncated_vecs.keys())
    
    # Get indices for test cases
    np.random.seed(42)  
    n_vectors = truncated_vecs[dimensions[0]].shape[0]
    query_indices = np.random.choice(n_vectors, num_samples, replace=False)
    
    # Get ground truth (assuming highest dimension is the most accurate)
    highest_dim = max(dimensions)
    highest_dim_vectors = truncated_vecs[highest_dim]
    
    # For each dimension, evaluate performance
    for dim in dimensions:
        logger.info(f"Evaluating {dim}-dimensional vectors...")
        truncated = truncated_vecs[dim]
        pca_reduced = pca_vecs[dim]
        
        # 1. Similarity preservation
        similarities = []
        for idx in query_indices:
            # Get the same vector in both representations
            trunc_vec = truncated[idx]
            pca_vec = pca_reduced[idx]
            
            # Compare to original high-dimensional vector
            original_vec = highest_dim_vectors[idx]
            
            # Calculate similarity preservation
            trunc_sim = calculate_similarity(original_vec, trunc_vec)
            pca_sim = calculate_similarity(original_vec, pca_vec)
            
            similarities.append((trunc_sim, pca_sim))
        
        avg_trunc_sim = np.mean([s[0] for s in similarities])
        avg_pca_sim = np.mean([s[1] for s in similarities])
        
        # 2. Nearest neighbor recall
        trunc_recall = evaluate_recall(
            highest_dim_vectors, truncated, query_indices, k)
        pca_recall = evaluate_recall(
            highest_dim_vectors, pca_reduced, query_indices, k)
        
        # Store results
        results[dim] = {
            "truncated_similarity": avg_trunc_sim,
            "pca_similarity": avg_pca_sim,
            "truncated_recall": trunc_recall,
            "pca_recall": pca_recall,
            "similarity_improvement": avg_pca_sim - avg_trunc_sim,
            "recall_improvement": pca_recall - trunc_recall
        }
        
        logger.info(f"Dimension {dim}:")
        logger.info(f"  Similarity - Truncated: {avg_trunc_sim:.4f}, PCA: {avg_pca_sim:.4f}")
        logger.info(f"  Recall@{k} - Truncated: {trunc_recall:.4f}, PCA: {pca_recall:.4f}")
    
    return results

def evaluate_recall(
    ground_truth_vecs: np.ndarray,
    test_vecs: np.ndarray,
    query_indices: np.ndarray,
    k: int = 10
) -> float:
    """
    Evaluate recall@k for test vectors against ground truth.
    
    Args:
        ground_truth_vecs: Ground truth vectors (high dimensional)
        test_vecs: Test vectors (reduced dimension)
        query_indices: Indices of vectors to use as queries
        k: Number of neighbors for recall calculation
    
    Returns:
        Average recall@k
    """
    # Get queries from ground truth
    queries = ground_truth_vecs[query_indices]
    
    # Compute ground truth nearest neighbors
    gt_similarities = queries @ ground_truth_vecs.T
    gt_indices = np.argsort(-gt_similarities, axis=1)[:, :k]
    
    # Compute test nearest neighbors
    test_queries = test_vecs[query_indices]
    test_similarities = test_queries @ test_vecs.T
    test_indices = np.argsort(-test_similarities, axis=1)[:, :k]
    
    # Calculate recall
    recall = np.mean([
        len(set(test_indices[i].tolist()).intersection(set(gt_indices[i].tolist()))) / k 
        for i in range(len(query_indices))
    ])
    
    return recall

def plot_comparison_results(results: Dict[int, Dict[str, float]], save_dir: str = 'results'):
    """
    Plot performance comparison between truncation and PCA reduction.
    
    Args:
        results: Dictionary of results by dimension
        save_dir: Directory to save plots
    """
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    dimensions = sorted(results.keys())
    
    # Plot similarity comparison
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Similarity preservation
    plt.subplot(2, 2, 1)
    plt.plot(dimensions, [results[d]["truncated_similarity"] for d in dimensions], 'o-', label='Truncated')
    plt.plot(dimensions, [results[d]["pca_similarity"] for d in dimensions], 's-', label='PCA')
    plt.xlabel('Dimensions')
    plt.ylabel('Similarity to Original')
    plt.title('Similarity Preservation')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Recall comparison
    plt.subplot(2, 2, 2)
    plt.plot(dimensions, [results[d]["truncated_recall"] for d in dimensions], 'o-', label='Truncated')
    plt.plot(dimensions, [results[d]["pca_recall"] for d in dimensions], 's-', label='PCA')
    plt.xlabel('Dimensions')
    plt.ylabel('Recall@10')
    plt.title('Recall Performance')
    plt.grid(True)
    plt.legend()
    
    # Plot 3: Improvement in similarity
    plt.subplot(2, 2, 3)
    plt.bar(dimensions, [results[d]["similarity_improvement"] for d in dimensions], color='skyblue')
    plt.xlabel('Dimensions')
    plt.ylabel('Improvement')
    plt.title('PCA Similarity Improvement over Truncation')
    plt.grid(True)
    
    # Plot 4: Improvement in recall
    plt.subplot(2, 2, 4)
    plt.bar(dimensions, [results[d]["recall_improvement"] for d in dimensions], color='lightgreen')
    plt.xlabel('Dimensions')
    plt.ylabel('Improvement')
    plt.title('PCA Recall Improvement over Truncation')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/pca_vs_truncation_comparison.png')
    logger.info(f"Saved comparison plot to {save_dir}/pca_vs_truncation_comparison.png")
    
    # Save detailed results as CSV
    import csv
    with open(f'{save_dir}/pca_vs_truncation_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dimension', 'Truncated_Similarity', 'PCA_Similarity', 
                         'Truncated_Recall', 'PCA_Recall', 
                         'Similarity_Improvement', 'Recall_Improvement'])
        
        for dim in dimensions:
            r = results[dim]
            writer.writerow([
                dim,
                f"{r['truncated_similarity']:.4f}",
                f"{r['pca_similarity']:.4f}",
                f"{r['truncated_recall']:.4f}",
                f"{r['pca_recall']:.4f}",
                f"{r['similarity_improvement']:.4f}",
                f"{r['recall_improvement']:.4f}"
            ])
    
    logger.info(f"Saved detailed results to {save_dir}/pca_vs_truncation_results.csv")
    
    plt.show()

def simulate_cascade_test(
    high_dim_vectors: np.ndarray,
    dimensions: List[int],
    save_dir: str = 'data/cascade_training'
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Simulate a cascade test by generating both truncated and PCA-reduced vectors.
    
    Args:
        high_dim_vectors: High-dimensional vectors
        dimensions: Target dimensions to test
        save_dir: Directory to save vectors
    
    Returns:
        Tuple of (truncated_vectors, pca_vectors) dictionaries
    """
    # Create the output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create truncated vectors (expected outputs)
    truncated_vecs = truncate_vectors(high_dim_vectors, dimensions)
    
    # Save the truncated vectors
    for dim, vecs in truncated_vecs.items():
        out_path = f'{save_dir}/truncated_{dim}dim.npy'
        np.save(out_path, vecs)
        logger.info(f"Saved {dim}-dimensional truncated vectors to {out_path}")
    
    # Create PCA-reduced vectors (model outputs)
    pca_vecs = reduce_dimensions(high_dim_vectors, dimensions)
    
    # Save the PCA vectors
    for dim, vecs in pca_vecs.items():
        out_path = f'{save_dir}/pca_{dim}dim.npy'
        np.save(out_path, vecs)
        logger.info(f"Saved {dim}-dimensional PCA vectors to {out_path}")
    
    # Save original high-dimensional vectors
    high_dim_path = f'{save_dir}/original_{high_dim_vectors.shape[1]}dim.npy'
    np.save(high_dim_path, high_dim_vectors)
    logger.info(f"Saved original {high_dim_vectors.shape[1]}-dimensional vectors to {high_dim_path}")
    
    return truncated_vecs, pca_vecs

def main():
    """Main function to run the cascade training simulation test."""
    logger.info("Starting comprehensive dimensional cascade training test")
    
    # Create output directories
    data_dir = 'data/cascade_training'
    results_dir = 'results/cascade_training'
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Check if we have high-dimensional vectors, otherwise create them
    high_dim_path = f'{data_dir}/original_1024dim.npy'
    
    if os.path.exists(high_dim_path):
        logger.info(f"Loading high-dimensional vectors from {high_dim_path}")
        high_dim_vectors = np.load(high_dim_path)
    else:
        logger.info(f"High-dimensional vectors not found, creating new ones")
        high_dim_vectors = create_high_dim_vectors(
            n_samples=2000,  # More vectors for better training simulation
            n_features=1024,
            n_clusters=30,   # More clusters for diversity
            save_path=high_dim_path
        )
    
    logger.info(f"Using vectors with shape: {high_dim_vectors.shape}")
    
    # 2. Define dimensions to test
    dimensions = [32, 64, 128, 256, 512]
    
    # 3. Simulate cascade training data
    truncated_vecs, pca_vecs = simulate_cascade_test(
        high_dim_vectors, dimensions, data_dir)
    
    # 4. Evaluate performance
    results = evaluate_model_performance(
        truncated_vecs, pca_vecs, num_samples=500, k=10)
    
    # 5. Plot comparison results
    plot_comparison_results(results, results_dir)
    
    logger.info("✅ Cascade training simulation completed successfully!")

if __name__ == "__main__":
    main() 