#!/usr/bin/env python
"""
Test trained distillation models on sample queries.

This script loads a trained cascade distiller and tests it on sample queries
by comparing similarity search results between original and distilled embeddings.
"""

import os
import argparse
import logging
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel

from src.distillation.models import CascadeDistiller

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test distillation models")
    
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing trained distillation models")
    parser.add_argument("--embedding_model", type=str, 
                        default="Snowflake/snowflake-arctic-embed-s",
                        help="Model to use for generating embeddings")
    parser.add_argument("--test_queries", type=str, 
                        default="What is machine learning?,How does dimensional reduction work?,What are embeddings?",
                        help="Comma-separated list of test queries")
    parser.add_argument("--test_corpus", type=str, default=None,
                        help="Path to file with test corpus (one document per line)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda, mps, cpu, or None for auto)")
    parser.add_argument("--query_prefix", type=str, default="query: ",
                        help="Prefix to use for queries (if model requires it)")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of top results to show")
    
    return parser.parse_args()


def load_distillation_model(model_dir: str, device: str) -> CascadeDistiller:
    """
    Load trained distillation model.
    
    Args:
        model_dir: Directory containing the trained model
        device: Device to load the model on
        
    Returns:
        Loaded CascadeDistiller model
    """
    model_dir = Path(model_dir)
    
    # Look for common_corpus.pt or similar
    model_files = list(model_dir.glob("**/*.pt"))
    if not model_files:
        raise ValueError(f"No model files found in {model_dir}")
    
    # Prefer best_model.pt or final_model.pt
    for preferred in ["best_model.pt", "final_model.pt"]:
        for model_file in model_files:
            if model_file.name == preferred:
                checkpoint_path = model_file
                break
        else:
            continue
        break
    else:
        # Just use the first model file found
        checkpoint_path = model_files[0]
    
    logger.info(f"Loading model from {checkpoint_path}")
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create a new cascade distiller
    model = CascadeDistiller()
    
    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded model with supported dimensions: {model.supported_dimensions}")
    
    return model


def generate_embeddings(
    texts: List[str], 
    model_name: str, 
    device: str,
    query_prefix: str = ""
) -> np.ndarray:
    """
    Generate embeddings for texts using the specified model.
    
    Args:
        texts: List of text strings
        model_name: Name of the model to use
        device: Device to run inference on
        query_prefix: Prefix to add to queries
        
    Returns:
        NumPy array of embeddings
    """
    # Load model and tokenizer
    logger.info(f"Loading embedding model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
    model = model.to(device)
    model.eval()
    
    # Add query prefix if specified
    if query_prefix:
        texts = [f"{query_prefix}{text}" for text in texts]
    
    # Tokenize
    tokens = tokenizer(
        texts, 
        padding=True, 
        truncation=True, 
        return_tensors="pt", 
        max_length=512
    ).to(device)
    
    # Generate embeddings
    with torch.no_grad():
        outputs = model(**tokens)
        embeddings = outputs[0][:, 0]  # CLS token
        
        # Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    
    return embeddings.cpu().numpy()


def similarity_search(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    k: int = 5
) -> Tuple[List[int], List[float]]:
    """
    Perform similarity search.
    
    Args:
        query_embedding: Query embedding
        corpus_embeddings: Corpus embeddings
        k: Number of top results to return
        
    Returns:
        Tuple of (indices, scores)
    """
    # Calculate cosine similarities
    similarities = np.dot(corpus_embeddings, query_embedding)
    
    # Get top k indices and scores
    top_indices = np.argsort(-similarities)[:k]
    top_scores = similarities[top_indices]
    
    return top_indices, top_scores


def compare_search_results(
    original_results: List[Tuple[int, float]],
    distilled_results: List[Tuple[int, float]],
    corpus_texts: List[str]
) -> Dict[str, Any]:
    """
    Compare search results between original and distilled embeddings.
    
    Args:
        original_results: List of (index, score) tuples for original embeddings
        distilled_results: List of (index, score) tuples for distilled embeddings
        corpus_texts: List of corpus texts
        
    Returns:
        Dictionary with comparison metrics
    """
    orig_indices, orig_scores = zip(*original_results)
    dist_indices, dist_scores = zip(*distilled_results)
    
    # Calculate overlap (how many of the same documents are retrieved)
    overlap = len(set(orig_indices).intersection(set(dist_indices)))
    overlap_ratio = overlap / len(orig_indices)
    
    # Calculate rank correlation
    # We need to convert indices to ranks for both result sets
    orig_ranks = {idx: rank for rank, idx in enumerate(orig_indices)}
    dist_ranks = {idx: rank for rank, idx in enumerate(dist_indices)}
    
    # Get common indices
    common_indices = set(orig_indices).intersection(set(dist_indices))
    
    # Calculate Spearman rank correlation for common indices
    if common_indices:
        rank_diffs = [(orig_ranks[idx] - dist_ranks[idx])**2 for idx in common_indices]
        rank_correlation = 1 - (6 * sum(rank_diffs)) / (len(common_indices) * (len(common_indices)**2 - 1))
    else:
        rank_correlation = 0
    
    return {
        "overlap": overlap,
        "overlap_ratio": overlap_ratio,
        "rank_correlation": rank_correlation
    }


def main():
    """Main function."""
    args = get_args()
    
    # Determine device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else \
                "mps" if torch.backends.mps.is_available() else \
                "cpu"
    logger.info(f"Using device: {device}")
    
    # Load distillation model
    distiller = load_distillation_model(args.model_dir, device)
    
    # Parse test queries
    test_queries = [q.strip() for q in args.test_queries.split(",")]
    
    # Load test corpus
    if args.test_corpus:
        with open(args.test_corpus, "r", encoding="utf-8") as f:
            corpus_texts = [line.strip() for line in f if line.strip()]
    else:
        # Default test corpus if none provided
        corpus_texts = [
            "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
            "Dimensional reduction techniques reduce the number of features in a dataset.",
            "Principal Component Analysis (PCA) is a statistical procedure that uses an orthogonal transformation.",
            "Embeddings are vector representations of discrete variables in a continuous vector space.",
            "Neural networks are computing systems inspired by the biological neural networks in animal brains.",
            "Deep learning is part of a broader family of machine learning methods based on artificial neural networks.",
            "Word embeddings are a type of word representation that allows words with similar meaning to have similar representations.",
            "A transformer is a deep learning model that adopts the mechanism of attention.",
            "BERT is a transformer-based machine learning model for natural language processing.",
            "GPT (Generative Pre-trained Transformer) is an autoregressive language model that uses deep learning.",
            "Vector space models represent text documents as vectors of identifiers.",
            "Latent Semantic Analysis (LSA) is a technique in natural language processing for analyzing relationships between documents.",
            "t-SNE is a machine learning algorithm for visualization developed by Laurens van der Maaten and Geoffrey Hinton.",
            "UMAP (Uniform Manifold Approximation and Projection) is a dimension reduction technique.",
            "Word2Vec is a group of related models used to produce word embeddings.",
            "Doc2Vec is an unsupervised algorithm that generates vectors for sentences, paragraphs or documents.",
            "FastText is a library for learning word representations and text classification created by Facebook.",
            "GloVe is an unsupervised learning algorithm for obtaining vector representations for words."
        ]
    
    logger.info(f"Test corpus size: {len(corpus_texts)} documents")
    
    # Generate embeddings for corpus
    corpus_embeddings = generate_embeddings(
        corpus_texts, 
        args.embedding_model, 
        device,
        ""  # No prefix for corpus
    )
    
    # Test each query
    for query in test_queries:
        logger.info(f"\nTesting query: '{query}'")
        
        # Generate embedding for query
        query_embedding = generate_embeddings(
            [query], 
            args.embedding_model, 
            device,
            args.query_prefix
        )[0]
        
        # Perform search with original embedding
        original_indices, original_scores = similarity_search(
            query_embedding, 
            corpus_embeddings, 
            args.k
        )
        
        # Print original results
        logger.info("Top results with original embeddings:")
        for i, (idx, score) in enumerate(zip(original_indices, original_scores)):
            logger.info(f"  {i+1}. [{score:.4f}] {corpus_texts[idx][:100]}...")
        
        # For each target dimension, distill and compare
        for dim in distiller.target_dimensions:
            logger.info(f"\nDimension: {dim}")
            
            # Distill query
            distilled_query = distiller.distill(
                torch.tensor(query_embedding[np.newaxis, :], dtype=torch.float32).to(device),
                dim
            )
            
            if distilled_query is None:
                logger.warning(f"Could not distill to dimension {dim}")
                continue
            
            distilled_query = distilled_query.cpu().numpy()[0]
            
            # Distill corpus
            distilled_corpus = []
            batch_size = 32
            for i in range(0, len(corpus_embeddings), batch_size):
                batch = corpus_embeddings[i:i+batch_size]
                batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
                distilled_batch = distiller.distill(batch_tensor, dim)
                if distilled_batch is not None:
                    distilled_corpus.append(distilled_batch.cpu().numpy())
            
            distilled_corpus_embeddings = np.vstack(distilled_corpus)
            
            # Search with distilled embeddings
            distilled_indices, distilled_scores = similarity_search(
                distilled_query, 
                distilled_corpus_embeddings, 
                args.k
            )
            
            # Print distilled results
            logger.info(f"Top results with {dim}D embeddings:")
            for i, (idx, score) in enumerate(zip(distilled_indices, distilled_scores)):
                logger.info(f"  {i+1}. [{score:.4f}] {corpus_texts[idx][:100]}...")
            
            # Compare results
            metrics = compare_search_results(
                list(zip(original_indices, original_scores)),
                list(zip(distilled_indices, distilled_scores)),
                corpus_texts
            )
            
            logger.info(f"Comparison metrics:")
            logger.info(f"  Overlap: {metrics['overlap']}/{args.k} ({metrics['overlap_ratio']:.2f})")
            logger.info(f"  Rank correlation: {metrics['rank_correlation']:.4f}")
    
    logger.info("\nTesting complete!")


if __name__ == "__main__":
    main() 