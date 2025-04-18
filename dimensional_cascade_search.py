#!/usr/bin/env python
# Dimensional Cascade Search Implementation

import numpy as np
import torch
import os
import argparse
import json
import pickle
import time
from tqdm import tqdm
from model2vec import StaticModel
import logging
from typing import List, Dict, Any, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('dimensional_cascade')

# Define the cascade dimensions
CASCADE_DIMENSIONS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

class DimensionalCascade:
    """Implementation of the Dimensional Cascade search approach."""
    
    def __init__(self, models_dir: str, base_model_path: str):
        """
        Initialize the Dimensional Cascade.
        
        Args:
            models_dir: Directory containing PCA models for each dimension
            base_model_path: Path to the base Model2Vec model (512d)
        """
        self.models_dir = models_dir
        self.base_model_path = base_model_path
        
        # Load the base model
        logger.info(f"Loading base model from {base_model_path}...")
        self.base_model = StaticModel.from_pretrained(base_model_path)
        
        # Load PCA models for each dimension
        self.pca_models = {}
        for dim in CASCADE_DIMENSIONS:
            if dim == 512:  # Skip the base dimension
                continue
                
            model_path = os.path.join(models_dir, f"pca_{dim}d", "pca_model.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.pca_models[dim] = pickle.load(f)
                logger.info(f"Loaded {dim}d PCA model")
            else:
                logger.warning(f"Could not find {dim}d PCA model at {model_path}")
        
        # Verify all necessary models are loaded
        missing_dims = [dim for dim in CASCADE_DIMENSIONS if dim != 512 and dim not in self.pca_models]
        if missing_dims:
            logger.warning(f"Missing PCA models for dimensions: {missing_dims}")
        
        logger.info("Dimensional Cascade initialized successfully")
    
    def encode_documents(self, 
                         documents: List[str], 
                         output_path: str = None) -> Dict[int, np.ndarray]:
        """
        Encode documents at all dimension levels.
        
        Args:
            documents: List of document texts
            output_path: Optional path to save the encoded vectors
            
        Returns:
            Dictionary mapping dimensions to document embeddings
        """
        logger.info(f"Encoding {len(documents)} documents at all dimension levels...")
        
        # Encode documents with base model
        start_time = time.time()
        base_embeddings = self.base_model.encode(documents, normalize=True)
        logger.info(f"Base encoding completed in {time.time() - start_time:.2f}s")
        
        # Store embeddings for all dimensions
        embeddings = {512: base_embeddings}
        
        # Apply PCA models to get embeddings at each dimension level
        for dim in tqdm(CASCADE_DIMENSIONS, desc="Generating embeddings for all dimensions"):
            if dim == 512:  # Skip the base dimension
                continue
                
            pca_model = self.pca_models.get(dim)
            if pca_model:
                # Transform to lower dimension
                reduced_embeddings = pca_model.transform(base_embeddings)
                
                # Normalize
                norms = np.linalg.norm(reduced_embeddings, axis=1, keepdims=True)
                reduced_embeddings = reduced_embeddings / norms
                
                embeddings[dim] = reduced_embeddings
        
        # Save embeddings if output path provided
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            for dim, embs in embeddings.items():
                np.save(os.path.join(output_path, f"embeddings_{dim}d.npy"), embs)
            
            # Save document texts for reference
            with open(os.path.join(output_path, "documents.json"), "w") as f:
                json.dump(documents, f)
            
            logger.info(f"Saved document embeddings to {output_path}")
        
        return embeddings
    
    def encode_query(self, query: str) -> Dict[int, np.ndarray]:
        """
        Encode query at all dimension levels.
        
        Args:
            query: Query text
            
        Returns:
            Dictionary mapping dimensions to query embedding
        """
        logger.info(f"Encoding query at all dimension levels: '{query}'")
        
        # Encode query with base model
        base_embedding = self.base_model.encode(query, normalize=True)
        
        # If the model returns a 1D array, reshape to 2D for consistency
        if len(base_embedding.shape) == 1:
            base_embedding = base_embedding.reshape(1, -1)
        
        # Store embeddings for all dimensions
        embeddings = {512: base_embedding}
        
        # Apply PCA models to get embeddings at each dimension level
        for dim in CASCADE_DIMENSIONS:
            if dim == 512:  # Skip the base dimension
                continue
                
            pca_model = self.pca_models.get(dim)
            if pca_model:
                # Transform to lower dimension
                reduced_embedding = pca_model.transform(base_embedding)
                
                # Normalize
                norm = np.linalg.norm(reduced_embedding, axis=1, keepdims=True)
                reduced_embedding = reduced_embedding / norm
                
                embeddings[dim] = reduced_embedding
        
        return embeddings
    
    def search(self, 
               query: str, 
               doc_embeddings: Dict[int, np.ndarray],
               documents: List[str],
               min_dimension: int = 1,
               max_dimension: int = 512, 
               top_k: int = 10,
               search_factor: int = 4) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for query using dimensional cascade approach.
        
        Args:
            query: Query text
            doc_embeddings: Document embeddings at each dimension level
            documents: Original document texts
            min_dimension: Minimum dimension to start search with
            max_dimension: Maximum dimension to use
            top_k: Number of results to return
            search_factor: Factor to multiply top_k by at each cascade level
            
        Returns:
            List of (document, score, metadata) tuples
        """
        if min_dimension not in CASCADE_DIMENSIONS:
            min_dimension = min([d for d in CASCADE_DIMENSIONS if d >= min_dimension])
            logger.info(f"Adjusted min_dimension to {min_dimension}")
            
        if max_dimension not in CASCADE_DIMENSIONS:
            max_dimension = max([d for d in CASCADE_DIMENSIONS if d <= max_dimension])
            logger.info(f"Adjusted max_dimension to {max_dimension}")
        
        # Get the subset of dimensions to use
        search_dimensions = [d for d in CASCADE_DIMENSIONS if min_dimension <= d <= max_dimension]
        search_dimensions.sort()  # Ensure dimensions are in ascending order
        
        logger.info(f"Searching with dimensions: {search_dimensions}")
        
        # Encode query at all dimension levels
        query_embeddings = self.encode_query(query)
        
        # Start with smallest dimension
        current_dim = search_dimensions[0]
        logger.info(f"Starting search with {current_dim}d embeddings")
        
        # Compute initial similarities
        query_emb = query_embeddings[current_dim]
        doc_embs = doc_embeddings[current_dim]
        
        similarities = np.dot(doc_embs, query_emb.T).flatten()
        
        # Get candidate indices for first level
        candidate_count = top_k * search_factor
        if candidate_count > len(similarities):
            candidate_count = len(similarities)
            
        top_indices = np.argsort(similarities)[-candidate_count:][::-1]
        
        # For each level in the cascade
        for next_dim_idx in range(1, len(search_dimensions)):
            next_dim = search_dimensions[next_dim_idx]
            logger.info(f"Refining search with {next_dim}d embeddings, {len(top_indices)} candidates")
            
            # Get embeddings for the current candidates
            query_emb = query_embeddings[next_dim]
            doc_embs = doc_embeddings[next_dim][top_indices]
            
            # Compute similarities for candidates
            similarities = np.dot(doc_embs, query_emb.T).flatten()
            
            # Update candidate set for next level
            if next_dim_idx < len(search_dimensions) - 1:
                # Get top candidates for next level
                next_candidate_count = top_k * max(1, search_factor // (2 ** next_dim_idx))
                if next_candidate_count > len(similarities):
                    next_candidate_count = len(similarities)
                    
                # Get indices of top candidates within the current candidate set
                top_k_in_candidates = np.argsort(similarities)[-next_candidate_count:][::-1]
                
                # Map back to original indices
                top_indices = top_indices[top_k_in_candidates]
            else:
                # Final level - just sort the candidates
                top_k_in_candidates = np.argsort(similarities)[-top_k:][::-1]
                top_indices = top_indices[top_k_in_candidates]
                similarities = similarities[top_k_in_candidates]
        
        # Return final results
        results = []
        for i, idx in enumerate(top_indices):
            if i < len(similarities):
                doc = documents[idx]
                score = float(similarities[i])
                
                # Collect metadata about the search path
                metadata = {
                    "index": int(idx),
                    "dimensions_used": search_dimensions,
                    "final_dimension": search_dimensions[-1]
                }
                
                results.append((doc, score, metadata))
        
        return results


def parse_args():
    parser = argparse.ArgumentParser(description='Dimensional Cascade Search')
    parser.add_argument('--models_dir', type=str, default='models/dimensional-cascade',
                        help='Directory containing PCA models for the cascade')
    parser.add_argument('--base_model', type=str, default='models/arctic-embed-512d/model2vec-512d',
                        help='Path to the base 512d Model2Vec model')
    parser.add_argument('--documents', type=str, required=True,
                        help='JSON file containing documents to search')
    parser.add_argument('--index_dir', type=str, default=None,
                        help='Directory to save/load document embeddings')
    parser.add_argument('--query', type=str, default=None,
                        help='Query to search for')
    parser.add_argument('--min_dimension', type=int, default=1,
                        help='Minimum dimension to use for search')
    parser.add_argument('--max_dimension', type=int, default=512,
                        help='Maximum dimension to use for search')
    parser.add_argument('--top_k', type=int, default=10,
                        help='Number of results to return')
    parser.add_argument('--search_factor', type=int, default=4,
                        help='Factor to expand search at each level')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize the cascade
    cascade = DimensionalCascade(args.models_dir, args.base_model)
    
    # Load documents
    with open(args.documents, 'r') as f:
        documents = json.load(f)
    logger.info(f"Loaded {len(documents)} documents")
    
    # Encode or load document embeddings
    doc_embeddings = None
    if args.index_dir:
        # Check if embeddings already exist
        if os.path.exists(os.path.join(args.index_dir, "embeddings_512d.npy")):
            logger.info(f"Loading document embeddings from {args.index_dir}")
            doc_embeddings = {}
            for dim in CASCADE_DIMENSIONS:
                emb_path = os.path.join(args.index_dir, f"embeddings_{dim}d.npy")
                if os.path.exists(emb_path):
                    doc_embeddings[dim] = np.load(emb_path)
                    logger.info(f"Loaded {dim}d embeddings with shape {doc_embeddings[dim].shape}")
        
    # If embeddings weren't loaded, generate them
    if not doc_embeddings:
        logger.info("Generating document embeddings for all dimensions")
        doc_embeddings = cascade.encode_documents(documents, args.index_dir)
    
    # Perform search if query provided
    if args.query:
        start_time = time.time()
        results = cascade.search(
            query=args.query,
            doc_embeddings=doc_embeddings,
            documents=documents,
            min_dimension=args.min_dimension,
            max_dimension=args.max_dimension,
            top_k=args.top_k,
            search_factor=args.search_factor
        )
        search_time = time.time() - start_time
        
        print(f"\nResults for query: '{args.query}'")
        print(f"Search completed in {search_time:.3f} seconds using dimensions {args.min_dimension}d to {args.max_dimension}d")
        print("=" * 80)
        for i, (doc, score, metadata) in enumerate(results):
            print(f"{i+1}. Score: {score:.4f}")
            print(f"   Document: {doc[:100]}..." if len(doc) > 100 else f"   Document: {doc}")
            print(f"   Index: {metadata['index']}")
            print()
    else:
        logger.info("No query provided. Document embeddings have been generated/loaded.")
        logger.info(f"To search, run again with --query 'your search query'")

if __name__ == "__main__":
    main() 