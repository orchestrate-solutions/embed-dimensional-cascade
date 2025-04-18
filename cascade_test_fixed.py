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

def calculate_similarity_via_search(query_vectors: np.ndarray, index_vectors: np.ndarray, k: int = 10) -> float:
    """
    Calculate similarity via nearest neighbor search recall.
    
    Args:
        query_vectors: Query vectors
        index_vectors: Index vectors to search against
        k: Number of nearest neighbors to retrieve
    
    Returns:
        Average recall@k
    """
    # Ensure we have enough vectors
    k = min(k, index_vectors.shape[0] - 1)
    
    # Compute similarity matrix
    similarities = query_vectors @ index_vectors.T
    
    # Get top k indices
    indices = np.argsort(-similarities, axis=1)[:, :k]
    
    return indices

def evaluate_model_performance(
    original_vectors: np.ndarray,
    truncated_vecs: Dict[int, np.ndarray],
    pca_vecs: Dict[int, np.ndarray],
    num_samples: int = 500,
    k: int = 10
) -> Dict[int, Dict[str, float]]:
    """
    Evaluate the performance of PCA reduction vs simple truncation.
    
    Args:
        original_vectors: Original high-dimensional vectors
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
    
    # Get ground truth nearest neighbors using original high-dimensional vectors
    logger.info(f"Computing ground truth nearest neighbors using {original_vectors.shape[1]}-dimensional vectors")
    query_vecs_original = original_vectors[query_indices]
    gt_indices = calculate_similarity_via_search(query_vecs_original, original_vectors, k)
    
    # For each dimension, evaluate performance
    for dim in dimensions:
        logger.info(f"Evaluating {dim}-dimensional vectors...")
        truncated = truncated_vecs[dim]
        pca_reduced = pca_vecs[dim]
        
        # Get queries in each representation
        query_vecs_trunc = truncated[query_indices]
        query_vecs_pca = pca_reduced[query_indices]
        
        # Calculate nearest neighbors for each representation
        trunc_indices = calculate_similarity_via_search(query_vecs_trunc, truncated, k)
        pca_indices = calculate_similarity_via_search(query_vecs_pca, pca_reduced, k)
        
        # Calculate recall compared to ground truth
        trunc_recall = np.mean([
            len(set(trunc_indices[i]).intersection(set(gt_indices[i]))) / k 
            for i in range(num_samples)
        ])
        
        pca_recall = np.mean([
            len(set(pca_indices[i]).intersection(set(gt_indices[i]))) / k 
            for i in range(num_samples)
        ])
        
        # Store results
        results[dim] = {
            "truncated_recall": trunc_recall,
            "pca_recall": pca_recall,
            "recall_improvement": pca_recall - trunc_recall
        }
        
        logger.info(f"Dimension {dim}:")
        logger.info(f"  Recall@{k} - Truncated: {trunc_recall:.4f}, PCA: {pca_recall:.4f}")
        logger.info(f"  Improvement: {pca_recall - trunc_recall:.4f}")
    
    return results

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
    
    # Plot comparison
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Recall comparison
    plt.subplot(2, 1, 1)
    plt.plot(dimensions, [results[d]["truncated_recall"] for d in dimensions], 'o-', markersize=8, label='Truncated')
    plt.plot(dimensions, [results[d]["pca_recall"] for d in dimensions], 's-', markersize=8, label='PCA')
    plt.xlabel('Dimensions')
    plt.ylabel('Recall@10')
    plt.title('Recall Performance Comparison')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Improvement in recall
    plt.subplot(2, 1, 2)
    bars = plt.bar(dimensions, [results[d]["recall_improvement"] for d in dimensions], color='lightgreen')
    plt.xlabel('Dimensions')
    plt.ylabel('Recall Improvement')
    plt.title('PCA Improvement over Simple Truncation')
    plt.grid(True)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.annotate(f'{height:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/cascade_training/pca_vs_truncation_comparison.png')
    logger.info(f"Saved comparison plot to {save_dir}/cascade_training/pca_vs_truncation_comparison.png")
    
    # Save detailed results as CSV
    import csv
    csv_path = f'{save_dir}/cascade_training/pca_vs_truncation_results.csv'
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Dimension', 'Truncated_Recall', 'PCA_Recall', 'Recall_Improvement'])
        
        for dim in dimensions:
            r = results[dim]
            writer.writerow([
                dim,
                f"{r['truncated_recall']:.4f}",
                f"{r['pca_recall']:.4f}",
                f"{r['recall_improvement']:.4f}"
            ])
    
    logger.info(f"Saved detailed results to {csv_path}")
    
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
    results_dir = 'results'
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/cascade_training", exist_ok=True)
    
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
        high_dim_vectors, truncated_vecs, pca_vecs, num_samples=500, k=10)
    
    # 5. Plot comparison results
    plot_comparison_results(results, results_dir)
    
    logger.info("✅ Cascade training simulation completed successfully!")

if __name__ == "__main__":
    main() 