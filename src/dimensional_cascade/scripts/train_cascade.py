#!/usr/bin/env python
"""
Train a Dimensional Cascade model hierarchy.

This script supports three approaches for dimensionality reduction:
1. Dimension Truncation with Fine-Tuning
2. Distillation Cascade
3. Autoencoder Dimensional Reduction

You can choose which approach to use via command-line arguments.
"""
import os
import argparse
import numpy as np
import torch
import random
from typing import List, Dict, Any, Tuple, Optional
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sentence_transformers import SentenceTransformer, InputExample, losses

from dimensional_cascade.models import ModelHierarchy, DimensionReducer
from dimensional_cascade.utils.io import load_jsonl, save_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description='Train a Dimensional Cascade model')
    
    parser.add_argument('--data', type=str, required=True, 
                        help='Path to training data (JSONL format with "text" field)')
    parser.add_argument('--output-dir', type=str, default='models',
                        help='Output directory for trained models')
    parser.add_argument('--base-model', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                        help='Base model to start from or path to existing model')
    parser.add_argument('--dimensions', type=str, default='1024,512,256,128,64,32,16',
                        help='Comma-separated list of dimensions to generate')
    parser.add_argument('--approach', type=str, default='truncation',
                        choices=['truncation', 'distillation', 'autoencoder'],
                        help='Dimension reduction approach to use')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Training epochs per dimension level')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--sample-size', type=int, default=10000,
                        help='Number of samples to use for training')
    parser.add_argument('--val-size', type=float, default=0.1,
                        help='Fraction of data to use for validation')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda, cpu, etc.)')
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_training_embeddings(
    model_path: str,
    documents: List[Dict[str, Any]],
    batch_size: int = 32,
    device: Optional[str] = None
) -> np.ndarray:
    """Generate embeddings for the training data.
    
    Args:
        model_path: Path to the model or model name
        documents: List of documents with 'text' field
        batch_size: Batch size for generating embeddings
        device: Device to use for inference
        
    Returns:
        Array of embeddings
    """
    # Create sentence transformer
    model = SentenceTransformer(model_path, device=device)
    
    # Extract texts
    texts = [doc['text'] for doc in documents if 'text' in doc]
    
    # Generate embeddings
    print(f"Generating embeddings for {len(texts)} documents...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    return embeddings


class EmbeddingDataset(Dataset):
    """Dataset for training dimension reduction models."""
    
    def __init__(self, embeddings: np.ndarray, texts: List[str] = None):
        self.embeddings = embeddings
        self.texts = texts
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        if self.texts:
            return {
                'embedding': self.embeddings[idx],
                'text': self.texts[idx]
            }
        else:
            return {
                'embedding': self.embeddings[idx]
            }


class DimensionReducerAutoencoder(nn.Module):
    """Autoencoder for dimensionality reduction."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: List[int] = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Default hidden layer sizes
        if hidden_layers is None:
            hidden_layers = [min(input_dim, output_dim * 4), output_dim * 2]
        
        # Build encoder layers
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Final encoder layer to target dimension
        encoder_layers.append(nn.Linear(prev_dim, output_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Build decoder layers
        decoder_layers = []
        prev_dim = output_dim
        for hidden_dim in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        # Final decoder layer back to input dimension
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded
    
    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(
    embeddings: np.ndarray,
    output_dim: int,
    batch_size: int = 32,
    epochs: int = 1,
    device: Optional[str] = None
) -> DimensionReducer:
    """Train an autoencoder for dimensional reduction.
    
    Args:
        embeddings: Input embeddings
        output_dim: Target output dimension
        batch_size: Training batch size
        epochs: Number of training epochs
        device: Device to use for training
        
    Returns:
        Trained DimensionReducer
    """
    input_dim = embeddings.shape[1]
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    # Create dataset and dataloader
    dataset = EmbeddingDataset(embeddings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create autoencoder
    autoencoder = DimensionReducerAutoencoder(input_dim, output_dim)
    autoencoder.to(device)
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)
    
    # Training loop
    autoencoder.train()
    for epoch in range(epochs):
        running_loss = 0.0
        
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
            for batch in pbar:
                # Get batch of embeddings
                inputs = batch['embedding'].float().to(device)
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                _, outputs = autoencoder(inputs)
                
                # Calculate loss
                loss = criterion(outputs, inputs)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Update running loss
                running_loss += loss.item()
                pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
    
    # Create DimensionReducer that uses the autoencoder
    class AutoencoderReducer(DimensionReducer):
        def __init__(self, autoencoder_model, input_dim, output_dim):
            super().__init__(input_dim, output_dim, method='autoencoder')
            self.autoencoder = autoencoder_model
            self._is_fitted = True
        
        def transform(self, X: np.ndarray) -> np.ndarray:
            # Handle single vector case
            if len(X.shape) == 1:
                X = X.reshape(1, -1)
            
            # Ensure we have the right input shape
            if X.shape[1] != self.input_dim:
                raise ValueError(f"Expected input dimension {self.input_dim}, got {X.shape[1]}")
            
            # Convert to torch tensor
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            
            # Encode the inputs
            with torch.no_grad():
                encoded = self.autoencoder.encode(X_tensor)
            
            # Convert back to numpy
            reduced = encoded.cpu().numpy()
            
            # Normalize vectors
            norms = np.linalg.norm(reduced, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            reduced = reduced / norms
            
            return reduced
    
    # Create the reducer
    reducer = AutoencoderReducer(autoencoder, input_dim, output_dim)
    
    return reducer


def create_distillation_dataset(
    teacher_model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 32
) -> List[InputExample]:
    """Create a dataset for knowledge distillation.
    
    Args:
        teacher_model: Teacher model to generate embeddings
        texts: List of texts to encode
        batch_size: Batch size for encoding
        
    Returns:
        List of InputExample objects for training
    """
    # Generate teacher embeddings
    teacher_embeddings = teacher_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # Create training examples
    examples = []
    for i in range(len(texts)):
        # Use the text and its teacher embedding as a training example
        examples.append(InputExample(texts=[texts[i]], label=teacher_embeddings[i]))
    
    return examples


def train_distillation_model(
    teacher_model_path: str,
    texts: List[str],
    output_dim: int,
    batch_size: int = 32,
    epochs: int = 1,
    device: Optional[str] = None
) -> SentenceTransformer:
    """Train a distilled model through knowledge distillation.
    
    Args:
        teacher_model_path: Path to the teacher model
        texts: List of texts for training
        output_dim: Dimension of the student model
        batch_size: Training batch size
        epochs: Number of training epochs
        device: Device to use
        
    Returns:
        Trained student model
    """
    # Load teacher model
    teacher_model = SentenceTransformer(teacher_model_path, device=device)
    
    # Create a student model with reduced dimension
    student_config = {
        'word_embedding_dimension': output_dim,
        'hidden_dim': output_dim
    }
    
    # Create the student model
    # We will use a BERT-based model with reduced word embedding dimension
    # For real implementation, you would likely modify the teacher model architecture
    # This is a simplified version for demonstration
    student_model = SentenceTransformer(teacher_model_path, device=device)
    
    # Create distillation dataset
    train_examples = create_distillation_dataset(teacher_model, texts, batch_size)
    
    # Use cosine similarity loss between student and teacher embeddings
    train_loss = losses.CosineSimilarityLoss(model=student_model)
    
    # Train the student model
    student_model.fit(
        train_objectives=[(train_examples, train_loss)],
        epochs=epochs,
        batch_size=batch_size,
        show_progress_bar=True
    )
    
    return student_model


def train_truncation_model(
    embeddings: np.ndarray,
    output_dim: int,
    method: str = 'pca'
) -> DimensionReducer:
    """Train a truncation-based dimension reducer.
    
    Args:
        embeddings: Input embeddings
        output_dim: Target output dimension
        method: Method to use ('pca', 'svd', etc.)
        
    Returns:
        Trained DimensionReducer
    """
    input_dim = embeddings.shape[1]
    
    # Create and fit the reducer
    reducer = DimensionReducer(
        input_dim=input_dim,
        output_dim=output_dim,
        method=method
    )
    
    reducer.fit(embeddings)
    
    return reducer


def train_cascade(
    documents: List[Dict[str, Any]],
    dimensions: List[int],
    base_model_path: str,
    output_dir: str,
    approach: str = 'truncation',
    batch_size: int = 32,
    epochs: int = 1,
    device: Optional[str] = None
) -> None:
    """Train a complete dimension cascade.
    
    Args:
        documents: List of training documents
        dimensions: List of dimensions to generate
        base_model_path: Path to base model
        output_dir: Output directory for trained models
        approach: Reduction approach to use
        batch_size: Training batch size
        epochs: Number of training epochs
        device: Device to use
    """
    # Ensure dimensions are sorted in descending order
    dimensions = sorted(dimensions, reverse=True)
    
    # Extract texts
    texts = [doc['text'] for doc in documents if 'text' in doc]
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each dimension level
    max_dim = dimensions[0]
    
    # Use the base model for the highest dimension
    if approach == 'distillation':
        # For distillation approach, we clone the base model for each dimension
        for dim in dimensions:
            dim_dir = os.path.join(output_dir, f"model_{dim}d")
            os.makedirs(dim_dir, exist_ok=True)
            
            if dim == max_dim:
                # For the highest dimension, just copy the base model
                base_model = SentenceTransformer(base_model_path, device=device)
                base_model.save(dim_dir)
                print(f"Saved base model for {dim}d")
            else:
                # Train a distilled model
                print(f"Training {dim}d model via distillation...")
                prev_dim = next(d for d in dimensions if d > dim)
                prev_model_dir = os.path.join(output_dir, f"model_{prev_dim}d")
                
                student_model = train_distillation_model(
                    teacher_model_path=prev_model_dir,
                    texts=texts,
                    output_dim=dim,
                    batch_size=batch_size,
                    epochs=epochs,
                    device=device
                )
                
                # Save the model
                student_model.save(dim_dir)
                print(f"Saved {dim}d distilled model")
    else:
        # For truncation and autoencoder approaches, we use the base model
        # only for the highest dimension and then generate reducers
        
        # First, save the base model for the highest dimension
        max_dim_dir = os.path.join(output_dir, f"model_{max_dim}d")
        os.makedirs(max_dim_dir, exist_ok=True)
        
        # Copy the base model
        base_model = SentenceTransformer(base_model_path, device=device)
        base_model.save(max_dim_dir)
        print(f"Saved base model for {max_dim}d")
        
        # Generate embeddings from the base model
        embeddings = generate_training_embeddings(
            model_path=base_model_path,
            documents=documents,
            batch_size=batch_size,
            device=device
        )
        
        # Create reducers for each lower dimension
        prev_dim = max_dim
        prev_embeddings = embeddings
        
        for dim in dimensions[1:]:  # Skip the highest dimension
            dim_dir = os.path.join(output_dir, f"model_{dim}d")
            os.makedirs(dim_dir, exist_ok=True)
            
            if approach == 'truncation':
                # Train a truncation-based reducer
                print(f"Training {dim}d model via truncation...")
                reducer = train_truncation_model(
                    embeddings=prev_embeddings,
                    output_dim=dim,
                    method='pca'
                )
            else:  # autoencoder
                # Train an autoencoder reducer
                print(f"Training {dim}d model via autoencoder...")
                reducer = train_autoencoder(
                    embeddings=prev_embeddings,
                    output_dim=dim,
                    batch_size=batch_size,
                    epochs=epochs,
                    device=device
                )
            
            # Save the reducer
            reducer_path = os.path.join(dim_dir, "reducer.pkl")
            reducer.save(reducer_path)
            print(f"Saved {dim}d reducer")
            
            # Generate reduced embeddings for the next level
            prev_embeddings = reducer.transform(prev_embeddings)
            prev_dim = dim


def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Parse dimensions
    dimensions = [int(dim) for dim in args.dimensions.split(',')]
    dimensions.sort(reverse=True)  # Ensure descending order
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load training data
    print(f"Loading data from {args.data}...")
    documents = load_jsonl(args.data)
    
    # Ensure documents have text field
    documents = [doc for doc in documents if 'text' in doc]
    
    # Sample documents if needed
    if args.sample_size and args.sample_size < len(documents):
        documents = random.sample(documents, args.sample_size)
    
    # Train the cascade
    train_cascade(
        documents=documents,
        dimensions=dimensions,
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        approach=args.approach,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device
    )
    
    print(f"Training complete. Models saved to {args.output_dir}")


if __name__ == '__main__':
    main() 