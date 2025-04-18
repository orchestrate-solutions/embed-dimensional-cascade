#!/usr/bin/env python
"""
Basic example demonstrating how to use the Dimensional Cascade for semantic search.

This example:
1. Creates a simple dataset of documents
2. Initializes a Dimensional Cascade
3. Indexes the documents
4. Performs queries at different dimension levels
5. Compares results and performance
"""
import os
import time
import numpy as np
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt

from dimensional_cascade.core import DimensionalCascade, CascadeConfig
from dimensional_cascade.utils.metrics import calculate_precision_loss


# Create sample documents
def create_sample_documents(num_docs: int = 1000) -> List[Dict[str, Any]]:
    """Create a sample dataset for demonstration."""
    categories = [
        "Technology", "Science", "Health", "Business", 
        "Politics", "Entertainment", "Sports", "Education"
    ]
    
    documents = []
    for i in range(num_docs):
        # Assign a random category
        category = categories[i % len(categories)]
        
        # Create document with basic text
        doc = {
            "id": i,
            "title": f"{category} Document {i}",
            "text": f"This is a sample document about {category.lower()}. "
                   f"It contains information relevant to the {category.lower()} category. "
                   f"Document ID is {i}.",
            "category": category
        }
        documents.append(doc)
    
    return documents


def main():
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Create sample documents
    print("Creating sample documents...")
    documents = create_sample_documents(1000)
    
    # Configure the cascade
    dimensions = [768, 384, 192, 96, 48, 24]
    config = CascadeConfig(dimensions=dimensions)
    
    # Initialize the cascade
    print("Initializing Dimensional Cascade...")
    cascade = DimensionalCascade(
        model_path="sentence-transformers/all-MiniLM-L6-v2",
        index_path="output/index",
        config=config
    )
    
    # Index the documents
    print("Indexing documents...")
    start_time = time.time()
    cascade.index_documents(documents)
    indexing_time = time.time() - start_time
    print(f"Indexing completed in {indexing_time:.2f} seconds")
    
    # Define some queries
    queries = [
        "Latest technology innovations",
        "Scientific research on climate change",
        "Health and wellness tips",
        "Business strategies for startups",
        "Political developments in Europe",
        "Entertainment news and celebrity gossip",
        "Sports results and upcoming matches",
        "Education policy and learning techniques"
    ]
    
    # Compare search at different dimensions
    print("\nComparing search at different dimensions:")
    results = {}
    times = {}
    
    for dim in dimensions:
        times[dim] = []
        results[dim] = {}
        
        print(f"\nSearching at {dim}d dimension:")
        for query in queries:
            start_time = time.time()
            # Use the model_hierarchy directly to embed at specific dimension
            query_embedding = cascade.model_hierarchy.embed(query, dimension=dim)
            
            # Search the index directly
            doc_indices, distances = cascade.index.search(query_embedding, dim, 10)
            
            # Get the documents
            docs = cascade.index.get_documents(doc_indices)
            results[dim][query] = docs
            
            elapsed = time.time() - start_time
            times[dim].append(elapsed)
            print(f"  Query: '{query}' - {elapsed:.4f} seconds")
            
            # Print top 3 results
            print("  Top results:")
            for i, doc in enumerate(docs[:3]):
                print(f"    {i+1}. {doc.get('title')} (Score: {1.0 / (1.0 + distances[i]):.4f})")
    
    # Calculate precision loss compared to highest dimension
    print("\nCalculating precision loss...")
    max_dim = max(dimensions)
    precision_losses = {}
    
    for dim in dimensions:
        if dim == max_dim:
            continue
        
        precision_losses[dim] = []
        for query in queries:
            high_dim_results = [(doc, 1.0) for doc in results[max_dim][query]]
            low_dim_results = [(doc, 1.0) for doc in results[dim][query]]
            
            metrics = calculate_precision_loss(high_dim_results, low_dim_results)
            precision_losses[dim].append(metrics["precision_loss"])
            
            print(f"  {dim}d vs {max_dim}d for query '{query}': {metrics['precision_loss']:.4f} loss")
    
    # Now try the cascade approach
    print("\nTrying the cascade approach:")
    cascade_results = {}
    cascade_times = []
    
    for query in queries:
        start_time = time.time()
        results = cascade.search(query, top_k=10)
        elapsed = time.time() - start_time
        cascade_times.append(elapsed)
        cascade_results[query] = [doc for doc, _ in results]
        
        print(f"  Query: '{query}' - {elapsed:.4f} seconds")
        # Print top 3 results
        print("  Top results:")
        for i, (doc, score) in enumerate(results[:3]):
            print(f"    {i+1}. {doc.get('title')} (Score: {score:.4f})")
    
    # Calculate cascade precision loss
    cascade_precision_losses = []
    for query in queries:
        high_dim_results = [(doc, 1.0) for doc in results[max_dim][query]]
        cascade_results_with_score = [(doc, 1.0) for doc in cascade_results[query]]
        
        metrics = calculate_precision_loss(high_dim_results, cascade_results_with_score)
        cascade_precision_losses.append(metrics["precision_loss"])
    
    # Create summary plot
    plt.figure(figsize=(12, 6))
    
    # Plot 1: Average search time
    plt.subplot(1, 2, 1)
    avg_times = [np.mean(times[dim]) for dim in dimensions]
    avg_times.append(np.mean(cascade_times))
    
    labels = [f"{dim}d" for dim in dimensions] + ["Cascade"]
    plt.bar(labels, avg_times)
    plt.title('Average Search Time')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    
    # Plot 2: Average precision loss
    plt.subplot(1, 2, 2)
    avg_losses = [0] + [np.mean(precision_losses[dim]) for dim in dimensions if dim != max_dim]
    avg_losses.append(np.mean(cascade_precision_losses))
    
    plt.bar(labels, avg_losses)
    plt.title('Average Precision Loss')
    plt.ylabel('Precision Loss')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig("output/comparison.png")
    
    print(f"\nResults saved to output/comparison.png")
    print("Summary:")
    print(f"  Highest Dimension ({max_dim}d) Avg Time: {np.mean(times[max_dim]):.4f} seconds")
    print(f"  Lowest Dimension ({min(dimensions)}d) Avg Time: {np.mean(times[min(dimensions)]):.4f} seconds")
    print(f"  Cascade Avg Time: {np.mean(cascade_times):.4f} seconds")
    print(f"  Lowest Dimension Precision Loss: {np.mean(precision_losses[min(dimensions)]):.4f}")
    print(f"  Cascade Precision Loss: {np.mean(cascade_precision_losses):.4f}")


if __name__ == "__main__":
    main() 