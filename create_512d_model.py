#!/usr/bin/env python
# Implementation of 1024d -> 512d dimension reduction for cascade

import numpy as np
import torch
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import os
import argparse
from tqdm import tqdm
import json
from model2vec import StaticModel
from model2vec.distill import distill
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('dimensional_cascade')

def parse_args():
    parser = argparse.ArgumentParser(description='Create 512d model from 1024d Arctic Embed model')
    parser.add_argument('--output_dir', type=str, default='models/arctic-embed-512d',
                        help='Directory to save the reduced model')
    parser.add_argument('--vocab_limit', type=int, default=30000,
                        help='Limit vocabulary size for efficiency')
    parser.add_argument('--sample_texts', type=str, default=None,
                        help='Optional JSON file with sample texts to validate embeddings')
    parser.add_argument('--use_model2vec', action='store_true',
                        help='Use Model2Vec library for distillation')
    return parser.parse_args()

def get_base_1024d_model():
    """Load the base 1024d Arctic Embed model"""
    logger.info("Loading the base 1024d Arctic Embed model...")
    model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0")
    return model

def extract_vocabulary_embeddings(model, vocab_limit=30000):
    """Extract embedding vectors for vocabulary tokens"""
    logger.info(f"Extracting vocabulary embeddings (limit: {vocab_limit})...")
    
    # Get the tokenizer from the model
    tokenizer = model._first_module().tokenizer
    
    # Get vocabulary
    vocab = list(tokenizer.get_vocab().items())
    vocab.sort(key=lambda x: x[1])  # Sort by token ID
    
    if vocab_limit and len(vocab) > vocab_limit:
        vocab = vocab[:vocab_limit]
    
    # Create a batch of single token inputs
    token_texts = [token for token, _ in vocab]
    
    # Get embeddings for all tokens
    logger.info(f"Generating embeddings for {len(token_texts)} tokens...")
    embeddings = []
    
    # Process in batches to avoid memory issues
    batch_size = 128
    for i in tqdm(range(0, len(token_texts), batch_size)):
        batch = token_texts[i:i+batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    
    embeddings = np.vstack(embeddings)
    
    return embeddings, token_texts

def create_pca_model(embeddings, n_components=512):
    """Create and train a PCA model to reduce dimensions"""
    logger.info(f"Creating PCA model to reduce from 1024d to {n_components}d...")
    pca = PCA(n_components=n_components)
    pca.fit(embeddings)
    
    # Log explained variance
    explained_variance = np.sum(pca.explained_variance_ratio_)
    logger.info(f"Explained variance with {n_components} components: {explained_variance:.4f}")
    
    return pca

def create_model2vec_distillation(target_dim=512):
    """Use the model2vec library to create a distilled model"""
    logger.info(f"Creating {target_dim}d Model2Vec distillation...")
    model_name = "Snowflake/snowflake-arctic-embed-l-v2.0"
    
    # Distill the model
    m2v_model = distill(model_name=model_name, pca_dims=target_dim)
    
    return m2v_model

def validate_embeddings(original_model, pca_model, sample_texts):
    """Validate the quality of dimension reduction using sample texts"""
    logger.info("Validating embeddings quality...")
    
    original_embeddings = original_model.encode(sample_texts)
    
    # Project original embeddings to reduced space
    reduced_embeddings = pca_model.transform(original_embeddings)
    
    # Compute similarities in original space
    original_sims = np.zeros((len(sample_texts), len(sample_texts)))
    for i in range(len(sample_texts)):
        for j in range(len(sample_texts)):
            if i != j:
                sim = np.dot(original_embeddings[i], original_embeddings[j]) / (
                    np.linalg.norm(original_embeddings[i]) * np.linalg.norm(original_embeddings[j])
                )
                original_sims[i, j] = sim
    
    # Compute similarities in reduced space
    reduced_sims = np.zeros((len(sample_texts), len(sample_texts)))
    for i in range(len(sample_texts)):
        for j in range(len(sample_texts)):
            if i != j:
                sim = np.dot(reduced_embeddings[i], reduced_embeddings[j]) / (
                    np.linalg.norm(reduced_embeddings[i]) * np.linalg.norm(reduced_embeddings[j])
                )
                reduced_sims[i, j] = sim
    
    # Compute correlation between similarity matrices
    flat_orig = original_sims.flatten()
    flat_reduced = reduced_sims.flatten()
    correlation = np.corrcoef(flat_orig, flat_reduced)[0, 1]
    
    logger.info(f"Similarity preservation correlation: {correlation:.4f}")
    return correlation

def main():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load sample texts if provided
    sample_texts = None
    if args.sample_texts:
        with open(args.sample_texts, 'r') as f:
            sample_texts = json.load(f)
    
    if args.use_model2vec:
        # Use Model2Vec for distillation
        m2v_model = create_model2vec_distillation(target_dim=512)
        
        # Save the model
        output_path = os.path.join(args.output_dir, 'model2vec-512d')
        logger.info(f"Saving Model2Vec distilled model to {output_path}...")
        m2v_model.save_pretrained(output_path)
        
        logger.info("Model2Vec distillation complete!")
    else:
        # Load base model
        model = get_base_1024d_model()
        
        # Extract vocabulary embeddings
        embeddings, token_texts = extract_vocabulary_embeddings(model, args.vocab_limit)
        
        # Create PCA model
        pca = create_pca_model(embeddings, n_components=512)
        
        # Validate if sample texts provided
        if sample_texts:
            validate_embeddings(model, pca, sample_texts)
        
        # Save PCA model and related data
        logger.info(f"Saving PCA model and data to {args.output_dir}...")
        np.save(os.path.join(args.output_dir, 'pca_components.npy'), pca.components_)
        np.save(os.path.join(args.output_dir, 'pca_mean.npy'), pca.mean_)
        with open(os.path.join(args.output_dir, 'explained_variance.json'), 'w') as f:
            json.dump({
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'explained_variance': pca.explained_variance_.tolist(),
                'total_explained_variance': np.sum(pca.explained_variance_ratio_)
            }, f, indent=2)
        
        logger.info("512d model creation complete!")

if __name__ == "__main__":
    main() 