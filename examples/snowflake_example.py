#!/usr/bin/env python3
"""
Example of using Snowflake Arctic Embed models for dimensional cascade search.

This example:
1. Creates a simple dataset of documents
2. Initializes a Snowflake cascade with pre-trained models
3. Indexes the documents
4. Performs searches with and without cascade
5. Compares performance and precision

The Snowflake Arctic Embed models are state-of-the-art embedding models 
(as of April 2024) that come in different sizes, making them perfect for 
dimensional cascade.
"""
import os
import sys
import time
import json
import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

try:
    from scripts.use_snowflake_models import SnowflakeCascade
except ImportError:
    print("Make sure the use_snowflake_models.py script is in the scripts directory")
    sys.exit(1)

# Create sample documents
def create_sample_documents(num_docs: int = 500) -> List[Dict[str, Any]]:
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
            "id": f"doc-{i}",
            "title": f"{category} Document {i}",
            "text": f"This is a sample document about {category.lower()}. "
                   f"It contains information relevant to the {category.lower()} category. "
                   f"Document ID is {i} and it's part of the {category} section. "
                   f"This example demonstrates using Snowflake Arctic Embed models "
                   f"for dimensional cascade searches.",
            "category": category
        }
        documents.append(doc)
    
    return documents


def main():
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Create sample documents
    print("Creating sample documents...")
    documents = create_sample_documents()
    
    # Save documents for later use
    docs_path = os.path.join("output", "sample_documents.jsonl")
    with open(docs_path, 'w') as f:
        for doc in documents:
            f.write(json.dumps(doc) + '\n')
    
    # Define models to use (from largest to smallest)
    model_sizes = ['335m', '137m', '33m', '22m']
    
    # Initialize cascade
    print(f"Initializing Snowflake cascade with models: {model_sizes}")
    cascade = SnowflakeCascade(
        model_sizes=model_sizes,
        search_factor=4  # Fetch 4x more candidates at each level
    )
    
    # Index documents
    print("Indexing documents...")
    index_path = os.path.join("output", "snowflake_index")
    
    # Check if index already exists
    if os.path.exists(os.path.join(index_path, "cascade_metadata.json")):
        print("Loading existing index...")
        cascade.load(index_path)
    else:
        print("Creating new index...")
        start_time = time.time()
        cascade.index_documents(documents, text_field="text", show_progress=True)
        indexing_time = time.time() - start_time
        print(f"Indexing completed in {indexing_time:.2f} seconds")
        
        # Save index
        print(f"Saving index to {index_path}")
        cascade.save(index_path)
    
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
    
    # Compare search methods
    print("\nComparing search methods:")
    
    # Prepare result containers
    cascade_times = []
    direct_times = []
    result_overlaps = []
    
    for query in queries:
        print(f"\nQuery: '{query}'")
        
        # Cascade search
        start_time = time.time()
        cascade_results = cascade.search(query, top_k=10)
        cascade_time = time.time() - start_time
        cascade_times.append(cascade_time)
        
        print(f"Cascade search: {cascade_time:.4f} seconds")
        print("Top 3 results:")
        for i, (doc, score) in enumerate(cascade_results[:3]):
            print(f"  {i+1}. {doc.get('title')} (Score: {score:.4f})")
        
        # Direct search (smallest model only)
        start_time = time.time()
        direct_results = cascade.search(query, top_k=10, smallest_only=True)
        direct_time = time.time() - start_time
        direct_times.append(direct_time)
        
        print(f"Direct search: {direct_time:.4f} seconds")
        print("Top 3 results:")
        for i, (doc, score) in enumerate(direct_results[:3]):
            print(f"  {i+1}. {doc.get('title')} (Score: {score:.4f})")
        
        # Calculate result overlap
        cascade_ids = set(doc.get("id") for doc, _ in cascade_results)
        direct_ids = set(doc.get("id") for doc, _ in direct_results)
        
        overlap = len(cascade_ids.intersection(direct_ids)) / len(cascade_ids)
        result_overlaps.append(overlap)
        
        print(f"Result overlap: {overlap:.2%}")
    
    # Calculate averages
    avg_cascade_time = sum(cascade_times) / len(cascade_times)
    avg_direct_time = sum(direct_times) / len(direct_times)
    avg_overlap = sum(result_overlaps) / len(result_overlaps)
    speedup = avg_direct_time / avg_cascade_time if avg_cascade_time > 0 else 0
    
    # Print summary
    print("\nPerformance Summary:")
    print(f"Average cascade search time: {avg_cascade_time:.4f} seconds")
    print(f"Average direct search time: {avg_direct_time:.4f} seconds")
    print(f"Average result overlap: {avg_overlap:.2%}")
    print(f"Average speedup: {speedup:.2f}x")
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    # Plot 1: Search times
    plt.subplot(1, 3, 1)
    plt.bar(['Cascade', 'Direct'], [avg_cascade_time, avg_direct_time])
    plt.ylabel('Time (seconds)')
    plt.title('Average Search Time')
    
    # Plot 2: Result overlap
    plt.subplot(1, 3, 2)
    plt.bar(['Overlap'], [avg_overlap])
    plt.ylim(0, 1)
    plt.ylabel('Ratio')
    plt.title('Average Result Overlap')
    
    # Plot 3: Individual query times
    plt.subplot(1, 3, 3)
    x = range(len(queries))
    width = 0.35
    plt.bar([i - width/2 for i in x], cascade_times, width, label='Cascade')
    plt.bar([i + width/2 for i in x], direct_times, width, label='Direct')
    plt.xticks(x, [f"Q{i+1}" for i in x])
    plt.ylabel('Time (seconds)')
    plt.title('Search Time by Query')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join("output", "snowflake_comparison.png"))
    
    print(f"\nVisualization saved to output/snowflake_comparison.png")
    
    # Explain the dimensions of each model
    print("\nSnowflake Model Dimensions:")
    for size in model_sizes:
        dim = cascade.dimensions[size]
        print(f"  snowflake-arctic-embed:{size} - {dim} dimensions")


if __name__ == "__main__":
    main() 