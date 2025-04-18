"""
Vector embedding utilities for the Dimensional Cascade project.

This module provides tools for transforming text documents into vector embeddings
using various embedding models.
"""

from typing import List, Dict, Union, Optional, Any
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextEmbedder:
    """
    Text embedding class that can use various embedding models.
    
    Supports:
    - Sentence Transformers models
    - Hugging Face transformers models
    """
    
    def __init__(self, model_name_or_path: str, use_sentence_transformer: bool = True,
                 device: str = None, cache_dir: str = None):
        """
        Initialize the text embedder with a specific model.
        
        Args:
            model_name_or_path: Model name (from HuggingFace) or path to local model
            use_sentence_transformer: Whether to use SentenceTransformer (True) or regular transformers (False)
            device: Device to use ('cuda', 'cpu', etc.). If None, will use CUDA if available.
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name_or_path
        self.use_sentence_transformer = use_sentence_transformer
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing TextEmbedder with model: {model_name_or_path}")
        logger.info(f"Using device: {self.device}")
        
        # Initialize model
        if use_sentence_transformer:
            self.model = SentenceTransformer(model_name_or_path, device=self.device, cache_folder=cache_dir)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"Initialized SentenceTransformer with embedding dimension: {self.embedding_dim}")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=cache_dir)
            self.model = AutoModel.from_pretrained(model_name_or_path, cache_dir=cache_dir)
            self.model.to(self.device)
            
            # Run a test example to get the output dimension
            sample_text = "Sample text to determine embedding dimensions."
            with torch.no_grad():
                inputs = self.tokenizer(sample_text, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                self.embedding_dim = outputs.last_hidden_state.size(-1)
            
            logger.info(f"Initialized Transformers model with embedding dimension: {self.embedding_dim}")
    
    def embed_texts(self, texts: List[str], batch_size: int = 32, 
                    normalize: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            batch_size: Batch size for processing
            normalize: Whether to L2-normalize the vectors
            
        Returns:
            NumPy array of embeddings with shape (len(texts), embedding_dim)
        """
        logger.info(f"Embedding {len(texts)} texts with batch size {batch_size}")
        
        if self.use_sentence_transformer:
            # SentenceTransformer has built-in batching
            embeddings = self.model.encode(texts, batch_size=batch_size, 
                                          show_progress_bar=True, 
                                          normalize_embeddings=normalize)
        else:
            # Manual batching for transformers model
            embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                inputs = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                       return_tensors="pt").to(self.device)
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = self.model(**inputs)
                
                # Use pooled output if available, otherwise mean pool the last hidden state
                if hasattr(outputs, 'pooler_output'):
                    batch_embeddings = outputs.pooler_output
                else:
                    # Mean pooling
                    attention_mask = inputs['attention_mask']
                    last_hidden = outputs.last_hidden_state
                    
                    # Apply mask
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
                    sum_embeddings = torch.sum(last_hidden * input_mask_expanded, 1)
                    sum_mask = input_mask_expanded.sum(1)
                    sum_mask = torch.clamp(sum_mask, min=1e-9)
                    batch_embeddings = sum_embeddings / sum_mask
                
                # Move to CPU and convert to NumPy
                batch_embeddings = batch_embeddings.cpu().numpy()
                embeddings.append(batch_embeddings)
            
            # Combine batches
            embeddings = np.vstack(embeddings)
            
            # Normalize if requested
            if normalize:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms = np.where(norms > 0, norms, 1e-9)  # Avoid division by zero
                embeddings = embeddings / norms
        
        logger.info(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def embed_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text string to embed
            normalize: Whether to L2-normalize the vector
            
        Returns:
            NumPy array of embedding with shape (embedding_dim,)
        """
        return self.embed_texts([text], normalize=normalize)[0]


class DocumentEmbedder:
    """
    Utility for embedding documents with different strategies:
    - Whole document embedding
    - Paragraph-level embedding
    - Sliding window embedding
    """
    
    def __init__(self, text_embedder: TextEmbedder):
        """
        Initialize with a text embedder instance.
        
        Args:
            text_embedder: TextEmbedder instance to use for generating embeddings
        """
        self.text_embedder = text_embedder
        self.embedding_dim = text_embedder.embedding_dim
    
    def embed_document_whole(self, document: str, normalize: bool = True) -> np.ndarray:
        """
        Embed an entire document as a single text.
        
        Args:
            document: Document text
            normalize: Whether to normalize the vector
            
        Returns:
            Document embedding vector
        """
        return self.text_embedder.embed_text(document, normalize=normalize)
    
    def embed_document_paragraphs(self, document: str, normalize: bool = True) -> Dict[str, Any]:
        """
        Split document into paragraphs and embed each separately.
        
        Args:
            document: Document text
            normalize: Whether to normalize the vectors
            
        Returns:
            Dictionary with:
            - 'paragraphs': List of paragraph texts
            - 'embeddings': NumPy array of paragraph embeddings
            - 'avg_embedding': Average of all paragraph embeddings
        """
        # Split into paragraphs (non-empty lines)
        paragraphs = [p.strip() for p in document.split('\n') if p.strip()]
        
        if not paragraphs:
            logger.warning("Document has no paragraphs, returning zero vector")
            return {
                'paragraphs': [],
                'embeddings': np.array([]),
                'avg_embedding': np.zeros(self.text_embedder.embedding_dim)
            }
        
        # Embed paragraphs
        embeddings = self.text_embedder.embed_texts(paragraphs, normalize=normalize)
        
        # Calculate average embedding
        avg_embedding = np.mean(embeddings, axis=0)
        
        # Normalize average if requested
        if normalize:
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        return {
            'paragraphs': paragraphs,
            'embeddings': embeddings,
            'avg_embedding': avg_embedding
        }
    
    def embed_document_windows(self, document: str, window_size: int = 256, 
                             stride: int = 128, normalize: bool = True) -> Dict[str, Any]:
        """
        Split document into overlapping windows of tokens and embed each.
        
        Args:
            document: Document text
            window_size: Size of each window in tokens
            stride: Stride between windows in tokens
            normalize: Whether to normalize the vectors
            
        Returns:
            Dictionary with:
            - 'windows': List of window texts
            - 'embeddings': NumPy array of window embeddings
            - 'avg_embedding': Average of all window embeddings
        """
        # Tokenize document
        tokens = self.text_embedder.tokenizer.tokenize(document)
        
        if not tokens:
            logger.warning("Document has no tokens, returning zero vector")
            return {
                'windows': [],
                'embeddings': np.array([]),
                'avg_embedding': np.zeros(self.text_embedder.embedding_dim)
            }
        
        # Create windows
        windows = []
        for i in range(0, len(tokens), stride):
            window_tokens = tokens[i:i+window_size]
            if len(window_tokens) < window_size // 2:
                # Skip very small windows at the end
                break
                
            window_text = self.text_embedder.tokenizer.convert_tokens_to_string(window_tokens)
            windows.append(window_text)
        
        if not windows:
            logger.warning("No windows created, returning zero vector")
            return {
                'windows': [],
                'embeddings': np.array([]),
                'avg_embedding': np.zeros(self.text_embedder.embedding_dim)
            }
        
        # Embed windows
        embeddings = self.text_embedder.embed_texts(windows, normalize=normalize)
        
        # Calculate average embedding
        avg_embedding = np.mean(embeddings, axis=0)
        
        # Normalize average if requested
        if normalize:
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        return {
            'windows': windows,
            'embeddings': embeddings,
            'avg_embedding': avg_embedding
        }


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Text embedding demo")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", 
                       help="Model name or path")
    parser.add_argument("--text", type=str, 
                       help="Text to embed for testing")
    args = parser.parse_args()
    
    # Demo
    if not args.text:
        sample_text = """
        Dimensional Cascade is an approach to improve semantic search efficiency.
        It works by creating a cascade of progressively more precise vector representations.
        Search begins with lower-dimensional vectors and progressively refines results
        using higher-dimensional vectors, resulting in faster retrieval with minimal loss in accuracy.
        """
        args.text = sample_text
    
    # Create embedder
    embedder = TextEmbedder(args.model)
    doc_embedder = DocumentEmbedder(embedder)
    
    # Generate embeddings
    print(f"\nEmbedding text with model: {args.model}")
    vector = embedder.embed_text(args.text)
    print(f"Text embedding shape: {vector.shape}, norm: {np.linalg.norm(vector):.4f}")
    
    # Paragraph embeddings
    result = doc_embedder.embed_document_paragraphs(args.text)
    print(f"Found {len(result['paragraphs'])} paragraphs")
    print(f"Paragraph embeddings shape: {result['embeddings'].shape}")
    print(f"Average embedding shape: {result['avg_embedding'].shape}, norm: {np.linalg.norm(result['avg_embedding']):.4f}")
    
    # Window embeddings
    result = doc_embedder.embed_document_windows(args.text, window_size=128, stride=64)
    print(f"Created {len(result['windows'])} windows")
    print(f"Window embeddings shape: {result['embeddings'].shape}")
    print(f"Average embedding shape: {result['avg_embedding'].shape}, norm: {np.linalg.norm(result['avg_embedding']):.4f}") 