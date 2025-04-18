"""
Model hierarchy for the Dimensional Cascade architecture.
"""
from typing import Dict, List, Optional, Union
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from dimensional_cascade.models.reduction import DimensionReducer


class ModelHierarchy:
    """Manages a hierarchy of progressively reduced-dimension models."""
    
    def __init__(
        self,
        model_path: str,
        dimensions: List[int],
        device: Optional[str] = None
    ):
        """Initialize the model hierarchy.
        
        Args:
            model_path: Path to models or base model
            dimensions: List of dimensions in descending order
            device: Device to use for inference ('cpu', 'cuda', etc.)
        """
        self.dimensions = sorted(dimensions, reverse=True)
        self.max_dimension = max(dimensions)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.models: Dict[int, Union[AutoModel, DimensionReducer]] = {}
        self._initialize_models(model_path)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path if os.path.isdir(model_path) else 'sentence-transformers/all-MiniLM-L6-v2'
        )
    
    def _initialize_models(self, model_path: str) -> None:
        """Initialize the base model and dimension reducers.
        
        Args:
            model_path: Path to models or base model
        """
        # Check if model_path contains individual models for each dimension
        if all(os.path.exists(os.path.join(model_path, f"model_{dim}d")) for dim in self.dimensions):
            # Load individual models
            for dim in self.dimensions:
                model_dir = os.path.join(model_path, f"model_{dim}d")
                if dim == self.max_dimension:
                    # Load base model
                    self.models[dim] = AutoModel.from_pretrained(model_dir).to(self.device)
                else:
                    # Load dimension reducer
                    with open(os.path.join(model_dir, "reducer.pkl"), 'rb') as f:
                        self.models[dim] = pickle.load(f)
        else:
            # Initialize from base model and create reducers
            # Load or initialize base model
            if os.path.isdir(model_path):
                base_model = AutoModel.from_pretrained(model_path).to(self.device)
            else:
                # Use a default model if path doesn't exist
                base_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2').to(self.device)
            
            self.models[self.max_dimension] = base_model
            
            # Create reducers for lower dimensions
            previous_dim = self.max_dimension
            for dim in self.dimensions:
                if dim == self.max_dimension:
                    continue
                
                self.models[dim] = DimensionReducer(
                    input_dim=previous_dim,
                    output_dim=dim,
                    method='pca'  # Default to PCA, could be configurable
                )
                previous_dim = dim
    
    def embed(self, text: str, dimension: Optional[int] = None) -> np.ndarray:
        """Generate embeddings for a single text.
        
        Args:
            text: Text to embed
            dimension: Dimension to use (default: max dimension)
            
        Returns:
            Embedding vector
        """
        dimension = dimension or self.max_dimension
        
        # Ensure we have a model for this dimension
        if dimension not in self.models:
            raise ValueError(f"No model available for dimension {dimension}")
        
        # Tokenize the text
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        # Generate base embedding if needed
        if dimension == self.max_dimension:
            # Use the base model directly
            with torch.no_grad():
                outputs = self.models[dimension](**inputs)
                embeddings = self._pool_embeddings(outputs)
            return embeddings.cpu().numpy()
        else:
            # First get the higher-dimension embedding, then reduce
            high_dim_embedding = self.embed(text, self.max_dimension)
            
            # Apply appropriate reduction transformations
            current_dim = self.max_dimension
            result = high_dim_embedding
            
            while current_dim > dimension:
                next_dim = max([d for d in self.dimensions if d < current_dim])
                result = self.models[next_dim].transform(result)
                current_dim = next_dim
                
            return result
    
    def embed_batch(
        self, 
        texts: List[str], 
        dimension: Optional[int] = None,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            dimension: Dimension to use (default: max dimension)
            batch_size: Size of batches for processing
            show_progress: Whether to show a progress bar
            
        Returns:
            Array of embedding vectors
        """
        dimension = dimension or self.max_dimension
        all_embeddings = []
        
        # Process in batches
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc=f"Embedding at {dimension}d", total=len(texts)//batch_size + 1)
        
        for i in iterator:
            batch_texts = texts[i:i+batch_size]
            
            if dimension == self.max_dimension:
                # Tokenize the batch
                inputs = self.tokenizer(
                    batch_texts, padding=True, truncation=True, return_tensors="pt"
                ).to(self.device)
                
                # Generate embeddings with the base model
                with torch.no_grad():
                    outputs = self.models[dimension](**inputs)
                    batch_embeddings = self._pool_embeddings(outputs)
                
                all_embeddings.append(batch_embeddings.cpu().numpy())
            else:
                # Get high-dimensional embeddings first
                high_dim_embeddings = self.embed_batch(
                    batch_texts, 
                    dimension=self.max_dimension, 
                    batch_size=batch_size,
                    show_progress=False
                )
                
                # Reduce dimensions
                current_dim = self.max_dimension
                result = high_dim_embeddings
                
                while current_dim > dimension:
                    next_dim = max([d for d in self.dimensions if d < current_dim])
                    result = self.models[next_dim].transform(result)
                    current_dim = next_dim
                
                all_embeddings.append(result)
        
        return np.vstack(all_embeddings)
    
    def _pool_embeddings(self, model_output) -> torch.Tensor:
        """Pool token embeddings into a single vector.
        
        Args:
            model_output: Output from the transformer model
            
        Returns:
            Pooled embedding tensor
        """
        # Use mean pooling as default strategy
        # This can be made more sophisticated based on the model
        token_embeddings = model_output.last_hidden_state
        attention_mask = model_output.get('attention_mask')
        
        if attention_mask is not None:
            # Apply attention mask before pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
                input_mask_expanded.sum(1), min=1e-9
            )
        else:
            # Simple mean pooling if no attention mask
            embeddings = torch.mean(token_embeddings, dim=1)
        
        # Normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    def save(self, path: str) -> None:
        """Save all models and reducers.
        
        Args:
            path: Directory path to save to
        """
        os.makedirs(path, exist_ok=True)
        
        # Save each model/reducer
        for dim, model in self.models.items():
            model_dir = os.path.join(path, f"model_{dim}d")
            os.makedirs(model_dir, exist_ok=True)
            
            if dim == self.max_dimension:
                # Save the transformer model
                model.save_pretrained(model_dir)
                self.tokenizer.save_pretrained(model_dir)
            else:
                # Save the reducer
                with open(os.path.join(model_dir, "reducer.pkl"), 'wb') as f:
                    pickle.dump(model, f)
    
    def get_dimension_embedding(self, embeddings: np.ndarray, target_dim: int) -> np.ndarray:
        """Convert embeddings from one dimension to another.
        
        Args:
            embeddings: Input embeddings
            target_dim: Target dimension
            
        Returns:
            Embeddings at the target dimension
        """
        input_dim = embeddings.shape[1]
        
        if input_dim == target_dim:
            return embeddings
        
        if input_dim < target_dim:
            raise ValueError(f"Cannot increase dimension from {input_dim} to {target_dim}")
        
        # Apply reduction cascade
        current_dim = input_dim
        result = embeddings
        
        while current_dim > target_dim:
            next_dim = max([d for d in self.dimensions if d < current_dim])
            result = self.models[next_dim].transform(result)
            current_dim = next_dim
            
            if current_dim == target_dim:
                break
        
        return result 