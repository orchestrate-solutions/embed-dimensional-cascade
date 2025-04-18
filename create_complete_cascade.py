#!/usr/bin/env python
# Create complete dimensional cascade from 512d down to 1d

import numpy as np
import torch
from sklearn.decomposition import PCA
import os
import argparse
from tqdm import tqdm
import json
import pickle
from model2vec import StaticModel
import logging
import time
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('dimensional_cascade')

# Define the cascade dimensions
CASCADE_DIMENSIONS = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]

def parse_args():
    parser = argparse.ArgumentParser(description='Create complete dimensional cascade from 512d to 1d')
    parser.add_argument('--input_model', type=str, default='models/arctic-embed-512d/model2vec-512d',
                        help='Path to the 512d Model2Vec model')
    parser.add_argument('--output_dir', type=str, default='models/dimensional-cascade',
                        help='Directory to save the cascade models')
    parser.add_argument('--sample_texts', type=str, default=None,
                        help='Optional JSON file with sample texts to validate embeddings')
    parser.add_argument('--eval_correlations', action='store_true',
                        help='Evaluate correlation between dimension levels')
    return parser.parse_args()

def load_base_model(model_path):
    """Load the base 512d Model2Vec model"""
    logger.info(f"Loading the base 512d model from {model_path}...")
    model = StaticModel.from_pretrained(model_path)
    return model

def get_vocabulary_embeddings(model):
    """Extract the vocabulary embeddings from the model"""
    logger.info("Extracting vocabulary embeddings...")
    vocab_embeddings = model.token_embeddings.weight.detach().numpy()
    return vocab_embeddings

def create_cascade_models(vocab_embeddings, output_dir):
    """Create cascade models for each dimension in the cascade"""
    logger.info("Creating cascade models for all dimensions...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Store results
    results = {
        "dimensions": {},
        "creation_time": datetime.now().isoformat(),
        "base_embedding_shape": vocab_embeddings.shape,
    }
    
    # Start with the full 512d embeddings
    current_embeddings = vocab_embeddings
    dim_models = {}
    
    # Create a model for each dimension in the cascade
    for i, dim in enumerate(CASCADE_DIMENSIONS):
        if i == 0:  # Skip the first dimension (512d) as it's our starting point
            dim_models[dim] = None  # No PCA needed for the base dimension
            results["dimensions"][dim] = {
                "explained_variance": 1.0,
                "is_base": True
            }
            continue
        
        prev_dim = CASCADE_DIMENSIONS[i-1]
        logger.info(f"Creating {dim}d model from {prev_dim}d embeddings...")
        
        start_time = time.time()
        
        # Train PCA model
        pca = PCA(n_components=dim)
        pca.fit(current_embeddings)
        
        # Project to the lower dimension
        reduced_embeddings = pca.transform(current_embeddings)
        
        # Store the model and results
        dim_models[dim] = pca
        
        # Calculate explained variance
        explained_variance = np.sum(pca.explained_variance_ratio_)
        logger.info(f"Explained variance with {dim} components: {explained_variance:.4f}")
        
        # Save the model
        model_path = os.path.join(output_dir, f"pca_{dim}d")
        os.makedirs(model_path, exist_ok=True)
        
        with open(os.path.join(model_path, 'pca_model.pkl'), 'wb') as f:
            pickle.dump(pca, f)
        
        np.save(os.path.join(model_path, 'embeddings.npy'), reduced_embeddings)
        
        # Update for next iteration
        current_embeddings = reduced_embeddings
        
        # Store results
        results["dimensions"][dim] = {
            "explained_variance": float(explained_variance),
            "creation_time": time.time() - start_time,
            "components_shape": pca.components_.shape,
        }
    
    # Save the results summary
    with open(os.path.join(output_dir, 'cascade_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return dim_models, results

def evaluate_correlation(base_model, cascade_models, sample_texts):
    """Evaluate correlation between different dimension levels"""
    logger.info("Evaluating correlation between dimension levels...")
    
    # Encode sample texts with the base model
    base_embeddings = base_model.encode(sample_texts, normalize=True)
    
    # Initialize correlation matrix
    dims = CASCADE_DIMENSIONS
    correlations = np.zeros((len(dims), len(dims)))
    
    # Compute correlations between all dimension pairs
    for i, dim_i in enumerate(dims):
        for j, dim_j in enumerate(dims):
            if i == j:
                correlations[i, j] = 1.0  # Self-correlation is 1.0
                continue
            
            if j > i:  # Only compute once per pair
                continue
            
            # Project to dimension i
            if i == 0:  # Base dimension
                emb_i = base_embeddings
            else:
                pca_i = cascade_models[dim_i]
                emb_i = pca_i.transform(base_embeddings)
                # Normalize
                emb_i = emb_i / np.linalg.norm(emb_i, axis=1, keepdims=True)
            
            # Project to dimension j
            if j == 0:  # Base dimension
                emb_j = base_embeddings
            else:
                pca_j = cascade_models[dim_j]
                emb_j = pca_j.transform(base_embeddings)
                # Normalize
                emb_j = emb_j / np.linalg.norm(emb_j, axis=1, keepdims=True)
            
            # Compute similarity matrices
            sim_i = np.zeros((len(sample_texts), len(sample_texts)))
            sim_j = np.zeros((len(sample_texts), len(sample_texts)))
            
            for a in range(len(sample_texts)):
                for b in range(len(sample_texts)):
                    if a != b:
                        sim_i[a, b] = np.dot(emb_i[a], emb_i[b])
                        sim_j[a, b] = np.dot(emb_j[a], emb_j[b])
            
            # Compute correlation
            flat_i = sim_i.flatten()
            flat_j = sim_j.flatten()
            corr = np.corrcoef(flat_i, flat_j)[0, 1]
            
            correlations[i, j] = corr
            correlations[j, i] = corr  # Symmetric
    
    return correlations

def main():
    args = parse_args()
    
    # Load sample texts if provided
    sample_texts = None
    if args.sample_texts:
        with open(args.sample_texts, 'r') as f:
            sample_texts = json.load(f)
            logger.info(f"Loaded {len(sample_texts)} sample texts for evaluation")
    
    # Load base model
    base_model = load_base_model(args.input_model)
    
    # Extract vocabulary embeddings
    vocab_embeddings = get_vocabulary_embeddings(base_model)
    logger.info(f"Extracted vocabulary embeddings with shape: {vocab_embeddings.shape}")
    
    # Create cascade models
    cascade_models, results = create_cascade_models(vocab_embeddings, args.output_dir)
    
    # Evaluate correlations if requested and sample texts provided
    if args.eval_correlations and sample_texts:
        correlations = evaluate_correlation(base_model, cascade_models, sample_texts)
        
        # Save correlation matrix
        np.save(os.path.join(args.output_dir, 'correlation_matrix.npy'), correlations)
        
        # Create a readable correlation table
        correlation_table = {}
        for i, dim_i in enumerate(CASCADE_DIMENSIONS):
            row = {}
            for j, dim_j in enumerate(CASCADE_DIMENSIONS):
                row[f"{dim_j}d"] = float(correlations[i, j])
            correlation_table[f"{dim_i}d"] = row
        
        with open(os.path.join(args.output_dir, 'correlation_table.json'), 'w') as f:
            json.dump(correlation_table, f, indent=2)
        
        logger.info("Correlation analysis complete! Results saved.")
    
    logger.info(f"Complete dimensional cascade created in {args.output_dir}")
    logger.info(f"Created models for dimensions: {', '.join([str(d) for d in CASCADE_DIMENSIONS])}")

if __name__ == "__main__":
    main() 