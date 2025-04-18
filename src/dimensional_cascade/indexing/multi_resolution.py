"""
Multi-resolution indexing for the Dimensional Cascade.
"""
import os
import pickle
import shutil
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import faiss


class MultiResolutionIndex:
    """Manages a set of indices at different dimensions."""
    
    def __init__(
        self,
        dimensions: List[int],
        index_path: Optional[str] = None
    ):
        """Initialize the multi-resolution index.
        
        Args:
            dimensions: List of dimensions to support
            index_path: Path to load/save indices (if None, uses in-memory indices)
        """
        self.dimensions = sorted(dimensions, reverse=True)
        self.index_path = index_path
        
        # Initialize indices
        self.indices: Dict[int, faiss.Index] = {}
        self.documents: List[Dict[str, Any]] = []
        
        # Load or create indices
        if index_path and os.path.exists(index_path):
            self._load_indices(index_path)
        else:
            self._create_indices()
    
    def _create_indices(self) -> None:
        """Create empty indices for each dimension."""
        for dim in self.dimensions:
            # Create HNSW index for each dimension
            # HNSW offers a good balance of speed and recall
            # Parameters can be tuned for specific use cases
            index = faiss.IndexHNSWFlat(dim, 32)  # 32 neighbors per layer
            
            # For smaller dimensions, we might want to use a more aggressive HNSW config
            # The smaller the dimension, the more approximate we can be
            if dim <= 64:
                index = faiss.IndexHNSWFlat(dim, 64)  # More neighbors for better recall in low dims
            
            self.indices[dim] = index
    
    def _load_indices(self, path: str) -> None:
        """Load indices from disk.
        
        Args:
            path: Path to load from
        """
        # Load documents
        docs_path = os.path.join(path, "documents.pkl")
        if os.path.exists(docs_path):
            with open(docs_path, 'rb') as f:
                self.documents = pickle.load(f)
        
        # Load indices for each dimension
        for dim in self.dimensions:
            index_file = os.path.join(path, f"index_{dim}d.faiss")
            if os.path.exists(index_file):
                self.indices[dim] = faiss.read_index(index_file)
            else:
                # Create a new index if not found
                index = faiss.IndexHNSWFlat(dim, 32)
                self.indices[dim] = index
    
    def add(
        self,
        embeddings: np.ndarray,
        documents: List[Union[str, Dict[str, Any]]],
        dimension: Optional[int] = None
    ) -> None:
        """Add embeddings and documents to the index.
        
        Args:
            embeddings: Embedding vectors (shape: n_samples, embedding_dim)
            documents: Documents corresponding to the embeddings
            dimension: Dimension of the embeddings (must match embeddings.shape[1])
        """
        dimension = dimension or embeddings.shape[1]
        
        if dimension not in self.indices:
            raise ValueError(f"No index available for dimension {dimension}")
        
        if embeddings.shape[1] != dimension:
            raise ValueError(f"Expected embeddings with {dimension} dimensions, got {embeddings.shape[1]}")
        
        # Convert embeddings to float32 (required by FAISS)
        embeddings = embeddings.astype(np.float32)
        
        # Add embeddings to index
        index = self.indices[dimension]
        
        # Store current index size as starting ID
        start_id = len(self.documents)
        
        # Add to index
        index.add(embeddings)
        
        # Store documents
        for doc in documents:
            if isinstance(doc, str):
                # Convert string to document dict
                self.documents.append({
                    'text': doc,
                    'id': len(self.documents)
                })
            else:
                # Ensure document has an ID
                doc_copy = doc.copy()
                if 'id' not in doc_copy:
                    doc_copy['id'] = len(self.documents)
                self.documents.append(doc_copy)
    
    def search(
        self,
        query_embedding: np.ndarray,
        dimension: int,
        k: int = 100
    ) -> Tuple[List[int], List[float]]:
        """Search the index for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            dimension: Dimension to search in
            k: Number of results to return
            
        Returns:
            Tuple of (indices, distances)
        """
        if dimension not in self.indices:
            raise ValueError(f"No index available for dimension {dimension}")
        
        # Ensure query_embedding has the right shape and type
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        if query_embedding.shape[1] != dimension:
            raise ValueError(f"Expected query with {dimension} dimensions, got {query_embedding.shape[1]}")
        
        query_embedding = query_embedding.astype(np.float32)
        
        # Search the index
        distances, indices = self.indices[dimension].search(query_embedding, k)
        
        return indices[0].tolist(), distances[0].tolist()
    
    def get_documents(self, indices: List[int]) -> List[Dict[str, Any]]:
        """Get documents by their indices.
        
        Args:
            indices: List of document indices
            
        Returns:
            List of document dictionaries
        """
        return [self.documents[i] for i in indices if 0 <= i < len(self.documents)]
    
    def save(self, path: str) -> None:
        """Save indices and documents to disk.
        
        Args:
            path: Path to save to
        """
        os.makedirs(path, exist_ok=True)
        
        # Save documents
        with open(os.path.join(path, "documents.pkl"), 'wb') as f:
            pickle.dump(self.documents, f)
        
        # Save indices
        for dim, index in self.indices.items():
            index_file = os.path.join(path, f"index_{dim}d.faiss")
            faiss.write_index(index, index_file)
    
    def clear(self) -> None:
        """Clear all indices and documents."""
        self.documents = []
        self._create_indices()
        
        # Clear any saved indices
        if self.index_path and os.path.exists(self.index_path):
            shutil.rmtree(self.index_path)
            os.makedirs(self.index_path, exist_ok=True) 