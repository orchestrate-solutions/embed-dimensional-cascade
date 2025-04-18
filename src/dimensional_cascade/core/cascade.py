"""
Core implementation of the Dimensional Cascade architecture.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np

from dimensional_cascade.models import ModelHierarchy
from dimensional_cascade.indexing import MultiResolutionIndex
from dimensional_cascade.search import CascadeSearch


@dataclass
class CascadeConfig:
    """Configuration for the Dimensional Cascade.
    
    Attributes:
        dimensions: List of dimensions in descending order (e.g., [1024, 512, 256, 128, 64, 32, 16])
        transition_thresholds: Dict mapping from dimension to number of candidates to consider
        early_termination: Whether to allow early termination if confidence is high
        parallel_search: Whether to perform parallel search across dimensions
        cache_embeddings: Whether to cache embeddings for queries
    """
    dimensions: List[int] = field(default_factory=lambda: [1024, 512, 256, 128, 64, 32, 16])
    transition_thresholds: Dict[int, int] = field(default_factory=lambda: {
        1024: 100,
        512: 250,
        256: 500,
        128: 1000,
        64: 2500, 
        32: 5000,
        16: 10000
    })
    early_termination: bool = True
    parallel_search: bool = False
    cache_embeddings: bool = True


class DimensionalCascade:
    """Main implementation of the Dimensional Cascade architecture.
    
    The Dimensional Cascade uses a hierarchy of progressively reduced-dimension models
    to enable ultra-fast initial retrieval with increasingly precise refinement.
    """
    
    def __init__(
        self,
        model_path: str,
        index_path: Optional[str] = None,
        config: Optional[CascadeConfig] = None
    ):
        """Initialize the Dimensional Cascade.
        
        Args:
            model_path: Path to the models or model base
            index_path: Path to stored indices (if None, indices will be created in memory)
            config: Configuration for the cascade
        """
        self.config = config or CascadeConfig()
        self.model_hierarchy = ModelHierarchy(model_path, self.config.dimensions)
        self.index = MultiResolutionIndex(self.config.dimensions, index_path)
        self.search_engine = CascadeSearch(self.model_hierarchy, self.index, self.config)
        
    def index_documents(
        self, 
        documents: List[Union[str, Dict[str, Any]]],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> None:
        """Index documents at all dimension levels.
        
        Args:
            documents: List of documents to index (strings or dicts with 'text' field)
            batch_size: Size of batches for processing
            show_progress: Whether to show a progress bar
        """
        # Extract text if documents are dictionaries
        texts = []
        for doc in documents:
            if isinstance(doc, dict):
                texts.append(doc.get('text', ''))
            else:
                texts.append(doc)
        
        # Generate embeddings at all dimension levels
        embeddings = {}
        for dim in self.config.dimensions:
            embeddings[dim] = self.model_hierarchy.embed_batch(texts, dimension=dim, batch_size=batch_size, show_progress=show_progress)
        
        # Add to indices
        for dim in self.config.dimensions:
            self.index.add(embeddings[dim], documents, dimension=dim)
            
    def search(
        self, 
        query: str,
        top_k: int = 100,
        min_dimension: Optional[int] = None,
        max_dimension: Optional[int] = None
    ) -> List[Tuple[Dict[str, Any], float]]:
        """Search the index using the cascade approach.
        
        Args:
            query: Query text
            top_k: Number of results to return
            min_dimension: Minimum dimension to use (default: smallest in config)
            max_dimension: Maximum dimension to use (default: largest in config)
            
        Returns:
            List of (document, score) tuples
        """
        min_dimension = min_dimension or min(self.config.dimensions)
        max_dimension = max_dimension or max(self.config.dimensions)
        
        return self.search_engine.search(query, top_k, min_dimension, max_dimension)
    
    def save(self, path: str) -> None:
        """Save the model and indices.
        
        Args:
            path: Directory path to save to
        """
        self.model_hierarchy.save(f"{path}/models")
        self.index.save(f"{path}/indices")
        
    @classmethod
    def load(cls, path: str, config: Optional[CascadeConfig] = None) -> 'DimensionalCascade':
        """Load a saved Dimensional Cascade.
        
        Args:
            path: Path to the saved cascade
            config: Optional config to override saved settings
            
        Returns:
            Loaded DimensionalCascade instance
        """
        cascade = cls(
            model_path=f"{path}/models",
            index_path=f"{path}/indices",
            config=config
        )
        return cascade 