#!/usr/bin/env python
"""
Train dimensional cascade models using the Common Corpus dataset.

This script:
1. Downloads a subset of the Common Corpus dataset from Hugging Face
2. Generates embeddings using a specified model
3. Trains dimensional distillation models on these embeddings
"""

import os
import argparse
import logging
import torch
import numpy as np
import time
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from src.distillation.models import DimensionDistiller, CascadeDistiller
from src.distillation.trainer import DistillationTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train on Common Corpus dataset")
    
    # Dataset parameters
    parser.add_argument("--num_samples", type=int, default=10000,
                        help="Number of samples to use from the dataset")
    parser.add_argument("--collection", type=str, default=None,
                        help="Filter for specific collection (e.g., OpenScience, OpenSource)")
    parser.add_argument("--min_token_count", type=int, default=100,
                        help="Minimum token count for text samples")
    parser.add_argument("--languages", type=str, default="en",
                        help="Comma-separated list of language codes to include")
    
    # Embedding model parameters
    parser.add_argument("--embedding_model", type=str, 
                        default="Snowflake/snowflake-arctic-embed-s",
                        help="Model to use for generating embeddings")
    parser.add_argument("--embedding_dim", type=int, default=768,
                        help="Dimension of embeddings from the model")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum sequence length for tokenization")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for embedding generation")
    parser.add_argument("--query_prefix", type=str, default="query: ",
                        help="Prefix to use for queries (if model requires it)")
    
    # Distillation parameters
    parser.add_argument("--target_dims", type=str, default="768,512,256,128,64,32",
                        help="Target dimensions for distillation (comma-separated)")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for distillation")
    parser.add_argument("--train_batch_size", type=int, default=64,
                        help="Batch size for distillation training")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio")
    parser.add_argument("--output_dir", type=str, default="models/common_corpus",
                        help="Output directory for saving models")
    parser.add_argument("--save_embeddings", type=str, default=None,
                        help="Path to save generated embeddings (optional)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda, mps, cpu, or None for auto)")
    
    return parser.parse_args()


def filter_dataset(dataset, args):
    """Filter the dataset based on criteria."""
    # Convert languages to list
    languages = [lang.strip() for lang in args.languages.split(",")]
    
    # Set up filter function
    def filter_fn(example):
        # Filter by language
        language_match = example["language"] in languages
        
        # Filter by token count (if available)
        token_count_match = True
        if "token_count" in example:
            token_count_match = example["token_count"] >= args.min_token_count
        
        # Filter by collection (if specified)
        collection_match = True
        if args.collection and "collection" in example:
            collection_match = example["collection"] == args.collection
            
        # Check if text field exists and is non-empty
        text_valid = "text" in example and example["text"] and len(example["text"]) > 0
        
        return language_match and token_count_match and collection_match and text_valid
    
    # Apply filter
    filtered_dataset = dataset.filter(filter_fn)
    logger.info(f"Filtered dataset: {len(filtered_dataset)} samples")
    
    # Take subset if needed
    if args.num_samples > 0 and args.num_samples < len(filtered_dataset):
        filtered_dataset = filtered_dataset.select(range(args.num_samples))
        logger.info(f"Selected subset: {len(filtered_dataset)} samples")
    
    return filtered_dataset


def generate_embeddings(dataset, args):
    """Generate embeddings for the dataset texts."""
    # Load tokenizer and model
    logger.info(f"Loading embedding model: {args.embedding_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_model)
    model = AutoModel.from_pretrained(args.embedding_model, add_pooling_layer=False)
    
    # Determine device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else \
                "mps" if torch.backends.mps.is_available() else \
                "cpu"
    model = model.to(device)
    logger.info(f"Using device: {device}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Function to get embeddings in batches
    def get_batch_embeddings(texts):
        # Add query prefix if specified
        if args.query_prefix:
            texts = [args.query_prefix + text for text in texts]
        
        # Tokenize
        tokens = tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt", 
            max_length=args.max_length
        ).to(device)
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**tokens)
            embeddings = outputs[0][:, 0]  # Get CLS token embedding
            
            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        return embeddings.cpu().numpy()
    
    # Generate embeddings in batches
    all_embeddings = []
    batch_size = args.batch_size
    total_batches = (len(dataset) + batch_size - 1) // batch_size
    
    start_time = time.time()
    logger.info(f"Starting embedding generation for {len(dataset)} samples in {total_batches} batches")
    
    for i in tqdm(range(0, len(dataset), batch_size), desc="Generating embeddings", total=total_batches):
        batch_texts = dataset[i:i+batch_size]["text"]
        batch_start = time.time()
        batch_embeddings = get_batch_embeddings(batch_texts)
        all_embeddings.append(batch_embeddings)
        
        # Show occasional progress details
        if i % (10 * batch_size) == 0 and i > 0:
            elapsed = time.time() - start_time
            batch_time = time.time() - batch_start
            progress = i / len(dataset)
            eta = elapsed / progress * (1 - progress) if progress > 0 else 0
            logger.info(f"Progress: {i}/{len(dataset)} samples ({progress:.1%}), "
                        f"Batch time: {batch_time:.2f}s, ETA: {eta/60:.1f} min")
    
    elapsed = time.time() - start_time
    # Concatenate all embeddings
    embeddings = np.vstack(all_embeddings)
    logger.info(f"Generated embeddings: {embeddings.shape} in {elapsed/60:.1f} minutes")
    
    # Save embeddings if path provided
    if args.save_embeddings:
        save_path = Path(args.save_embeddings)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(save_path, embeddings)
        logger.info(f"Saved embeddings to {save_path}")
    
    return embeddings


def train_distillation_models(embeddings, args):
    """Train distillation models on the generated embeddings."""
    # Parse target dimensions
    target_dims = [int(dim.strip()) for dim in args.target_dims.split(",")]
    target_dims = sorted(target_dims, reverse=True)  # Sort in descending order
    
    # Determine device
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else \
                "mps" if torch.backends.mps.is_available() else \
                "cpu"
                
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert embeddings to torch tensor
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
    
    # Train cascade distiller
    logger.info(f"Training cascade distiller for dimensions: {target_dims}")
    
    # Initialize cascade model
    cascade = CascadeDistiller()
    cascade.add_distiller(args.embedding_dim, target_dims)
    
    # Create trainer
    trainer = DistillationTrainer(
        model=cascade,
        learning_rate=args.learning_rate,
        device=device,
        output_dir=str(output_dir),
        experiment_name="common_corpus"
    )
    
    # Prepare data
    train_loader, val_loader = trainer.prepare_data(
        vectors=embeddings,
        batch_size=args.train_batch_size,
        val_split=args.val_split
    )
    
    # Train model
    logger.info("Starting training...")
    metrics = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        patience=10  # Early stopping patience
    )
    
    # Evaluate model
    logger.info("Evaluating model...")
    results = trainer.evaluate_similarity_preservation(embeddings[:1000])
    
    # Print results
    logger.info("Training complete!")
    logger.info(f"Final training loss: {metrics['train_losses'][-1]:.6f}")
    
    if metrics['val_losses']:
        logger.info(f"Final validation loss: {metrics['val_losses'][-1]:.6f}")
        logger.info(f"Best validation loss: {metrics['best_val_loss']:.6f} (epoch {metrics['best_epoch']})")
    
    logger.info("Similarity preservation results:")
    for dim, result in results.items():
        logger.info(f"Dimension {dim}: MSE={result['mse']:.6f}, "
                   f"Compression ratio={result['compression_ratio']:.2f}x")
    
    return metrics, results


def main():
    """Main function."""
    args = get_args()
    
    # Load dataset
    logger.info("Loading Common Corpus dataset...")
    dataset = load_dataset("PleIAs/common_corpus", split="train", streaming=True)
    
    # Filter dataset
    filtered_dataset = filter_dataset(dataset, args)
    
    # Generate embeddings
    embeddings = generate_embeddings(filtered_dataset, args)
    
    # Train distillation models
    metrics, results = train_distillation_models(embeddings, args)
    
    logger.info("Process completed successfully!")


if __name__ == "__main__":
    main() 