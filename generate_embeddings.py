#!/usr/bin/env python
"""
Generate embeddings using Snowflake's Arctic-embed model.

This script loads documents from a JSONL file, generates embeddings using
Snowflake's Arctic-embed model, and saves them to a file for later use in
dimensional cascade training.
"""

import os
import json
import argparse
import logging
import time
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """
    Load documents from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of documents
    """
    documents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                documents.append(doc)
    return documents

def chunk_text(text: str, max_tokens: int = 250, overlap: int = 20) -> List[str]:
    """
    Split text into chunks of approximately max_tokens with some overlap.
    
    Args:
        text: The text to chunk
        max_tokens: Maximum tokens per chunk (approximate since we use word boundaries)
        overlap: Number of words to overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Rough approximation: consider each word as a token
    words = text.split()
    
    if len(words) <= max_tokens:
        return [text]
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(words):
        end_idx = start_idx + max_tokens
        
        # If we're at the end, just take the rest
        if end_idx >= len(words):
            chunks.append(" ".join(words[start_idx:]))
            break
        
        # Otherwise, create a chunk with the maximum tokens
        chunks.append(" ".join(words[start_idx:end_idx]))
        
        # Move start index for next chunk, including overlap
        start_idx = end_idx - overlap
    
    return chunks

class ArcticEmbedder:
    """Wrapper for Snowflake Arctic-embed model."""
    
    def __init__(
        self, 
        model_name: str = "Snowflake/snowflake-arctic-embed-l-v2.0",
        device: Optional[str] = None,
        is_query: bool = False
    ):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the model to use
            device: Device to use for inference
            is_query: Whether this embedder is for queries
        """
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load model and tokenizer
        logger.info(f"Loading model {model_name} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        self.model.to(self.device)
        self.model.eval()
        
        # Set query flag
        self.is_query = is_query
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.hidden_size
        logger.info(f"Model loaded with dimension {self.embedding_dim}")
    
    def embed(self, text: str, chunk_size: int = 250, chunk_overlap: int = 20) -> np.ndarray:
        """
        Embed a single text and return the embedding.
        For long texts, chunks the text and averages the embeddings.
        
        Args:
            text: Text to embed
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            
        Returns:
            The embedding vector (normalized)
        """
        # For short texts, embed directly
        if len(text.split()) <= chunk_size:
            # Add query prefix if embedding a query
            if self.is_query:
                text = f"query: {text}"
            
            inputs = self.tokenizer(text, padding=True, truncation=True, 
                                   return_tensors="pt", max_length=8192)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs[0][:, 0]  # CLS token embedding
                normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
            return normalized.cpu().numpy()[0]
        
        # For long texts, chunk and average embeddings
        chunks = chunk_text(text, max_tokens=chunk_size, overlap=chunk_overlap)
        
        # Embed each chunk
        chunk_embeddings = []
        for chunk in chunks:
            if self.is_query:
                chunk = f"query: {chunk}"
                
            inputs = self.tokenizer(chunk, padding=True, truncation=True, 
                                   return_tensors="pt", max_length=8192)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs[0][:, 0]  # CLS token embedding
                normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                chunk_embeddings.append(normalized.cpu().numpy()[0])
        
        # Average the chunk embeddings
        avg_embedding = np.mean(chunk_embeddings, axis=0)
        # Re-normalize the average embedding
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        return avg_embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 32, chunk_size: int = 250, chunk_overlap: int = 20) -> np.ndarray:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            chunk_size: Maximum tokens per chunk 
            chunk_overlap: Number of tokens to overlap between chunks
            
        Returns:
            Array of embeddings, shape (n_texts, embedding_dim)
        """
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding texts"):
            batch_texts = texts[i:i+batch_size]
            
            # Embed each text, handling chunking individually
            batch_embeddings = [self.embed(text, chunk_size, chunk_overlap) for text in batch_texts]
            embeddings.extend(batch_embeddings)
            
        return np.array(embeddings)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate embeddings with Snowflake Arctic-embed")
    
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file with documents")
    parser.add_argument("--output", type=str, required=True, help="Output file for embeddings (.npy)")
    parser.add_argument("--text-field", type=str, default="text", help="Field containing text to embed")
    parser.add_argument("--model", type=str, default="Snowflake/snowflake-arctic-embed-l-v2.0", 
                       help="Model name to use")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, cpu)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for embedding")
    parser.add_argument("--chunk-size", type=int, default=250, help="Maximum tokens per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=20, help="Overlap tokens between chunks")
    parser.add_argument("--save-texts", action="store_true", help="Save text along with embeddings")
    
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Load documents
    logger.info(f"Loading documents from {args.input}")
    documents = load_jsonl(args.input)
    logger.info(f"Loaded {len(documents)} documents")
    
    # Extract texts
    texts = [doc.get(args.text_field, "") for doc in documents]
    logger.info(f"Extracted text from field '{args.text_field}'")
    
    # Initialize embedder
    embedder = ArcticEmbedder(
        model_name=args.model,
        device=args.device
    )
    
    # Generate embeddings
    logger.info(f"Generating embeddings with {args.model}")
    start_time = time.time()
    
    embeddings = embedder.embed_batch(
        texts=texts,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    elapsed = time.time() - start_time
    logger.info(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]} in {elapsed:.2f} seconds")
    
    # Save embeddings
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    np.save(output_path, embeddings)
    logger.info(f"Saved embeddings to {output_path}")
    
    # Save texts if requested
    if args.save_texts:
        texts_path = output_path.with_suffix('.txt')
        with open(texts_path, 'w', encoding='utf-8') as f:
            for text in texts:
                f.write(text + '\n---\n')
        logger.info(f"Saved texts to {texts_path}")
    
    logger.info("Done!")

if __name__ == "__main__":
    main() 