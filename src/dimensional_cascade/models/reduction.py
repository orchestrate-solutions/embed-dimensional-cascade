"""
Dimension reduction methods for the Dimensional Cascade.
"""
from typing import Literal, Optional, Union
import pickle
import os

import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.random_projection import GaussianRandomProjection


class DimensionReducer:
    """Reduces dimensions of embeddings using various methods."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        method: Literal['pca', 'svd', 'random_projection', 'autoencoder'] = 'pca',
        random_state: int = 42,
        pretrained_path: Optional[str] = None
    ):
        """Initialize the dimension reducer.
        
        Args:
            input_dim: Input dimension size
            output_dim: Output dimension size
            method: Reduction method ('pca', 'svd', 'random_projection', 'autoencoder')
            random_state: Random seed for reproducibility
            pretrained_path: Path to pretrained reducer (if None, will be trained on data)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.method = method
        self.random_state = random_state
        self.reducer = None
        self._is_fitted = False
        
        # Load pretrained reducer if provided
        if pretrained_path and os.path.exists(pretrained_path):
            with open(pretrained_path, 'rb') as f:
                loaded_reducer = pickle.load(f)
                self.reducer = loaded_reducer.reducer
                self._is_fitted = loaded_reducer._is_fitted
        else:
            # Initialize a new reducer
            self._initialize_reducer()
    
    def _initialize_reducer(self):
        """Initialize the appropriate reduction method."""
        if self.method == 'pca':
            self.reducer = PCA(
                n_components=self.output_dim,
                random_state=self.random_state
            )
        elif self.method == 'svd':
            self.reducer = TruncatedSVD(
                n_components=self.output_dim,
                random_state=self.random_state
            )
        elif self.method == 'random_projection':
            self.reducer = GaussianRandomProjection(
                n_components=self.output_dim,
                random_state=self.random_state
            )
        elif self.method == 'autoencoder':
            # For autoencoder, we'd normally use a neural network,
            # but that's more complex than we'll implement here.
            # For now, fall back to PCA with a warning
            import warnings
            warnings.warn(
                "Autoencoder method not fully implemented. Falling back to PCA.",
                UserWarning
            )
            self.reducer = PCA(
                n_components=self.output_dim,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown reduction method: {self.method}")
    
    def fit(self, X: np.ndarray) -> 'DimensionReducer':
        """Fit the reducer to the data.
        
        Args:
            X: Data to fit (shape: n_samples, input_dim)
            
        Returns:
            Self for chaining
        """
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {X.shape[1]}")
        
        self.reducer.fit(X)
        self._is_fitted = True
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to reduced dimensions.
        
        Args:
            X: Data to transform (shape: n_samples, input_dim)
            
        Returns:
            Transformed data (shape: n_samples, output_dim)
        """
        if not self._is_fitted and self.method not in ['random_projection']:
            raise RuntimeError("Reducer must be fitted before transform")
        
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dimension {self.input_dim}, got {X.shape[1]}")
        
        # Handle single vector case
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        
        reduced = self.reducer.transform(X)
        
        # Normalize vectors
        norms = np.linalg.norm(reduced, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        reduced = reduced / norms
        
        return reduced
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.
        
        Args:
            X: Data to fit and transform (shape: n_samples, input_dim)
            
        Returns:
            Transformed data (shape: n_samples, output_dim)
        """
        self.fit(X)
        return self.transform(X)
    
    def save(self, path: str) -> None:
        """Save the reducer to disk.
        
        Args:
            path: Path to save to
        """
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str) -> 'DimensionReducer':
        """Load a reducer from disk.
        
        Args:
            path: Path to load from
            
        Returns:
            Loaded reducer
        """
        with open(path, 'rb') as f:
            return pickle.load(f) 