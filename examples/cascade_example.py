#!/usr/bin/env python3
"""
Example of using Snowflake Dimensional Cascade in a Python program

This example demonstrates how to:
1. Load documents from a JSONL file
2. Create a Snowflake Cascade
3. Index documents with the cascade
4. Perform searches with different approaches
5. Compare search results
"""
import os
import sys
import time
import json
from pprint import pprint

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

try:
    from dimensional_cascade.snowflake_cascade import (
        SnowflakeCascade,
        SnowflakeModel,
        load_jsonl,
        compare_search_methods
    )
except ImportError:
    print("Error: Could not import Snowflake Cascade module.")
    print("Make sure the dimensional_cascade package is properly installed.")
    sys.exit(1)


def main():
    """Run the example."""
    print("Snowflake Dimensional Cascade Example")
    print("-" * 40)

    # Example data path - change this to your data file
    data_path = os.path.join(project_root, 'data', 'sample_data.jsonl')
    
    # Check if sample data exists, if not create some dummy data
    if not os.path.exists(data_path):
        print(f"Sample data not found at {data_path}")
        print("Creating dummy data for the example...")
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        # Create some dummy documents
        docs = []
        for i in range(50):
            category = ["technology", "science", "health", "business"][i % 4]
            doc = {
                "id": f"doc-{i}",
                "title": f"{category.title()} Document {i}",
                "text": f"This is a sample document about {category}. "
                        f"It contains information relevant to the {category} category."
            }
            docs.append(doc)
        
        # Save to JSONL
        with open(data_path, 'w') as f:
            for doc in docs:
                f.write(json.dumps(doc) + '\n')
        
        print(f"Created {len(docs)} dummy documents at {data_path}")
    
    # Load documents
    print(f"Loading documents from {data_path}")
    documents = load_jsonl(data_path)
    print(f"Loaded {len(documents)} documents")
    
    # Create cascade with smaller models for faster example
    # For a real application, you might want to use larger models
    print("Initializing Snowflake Cascade...")
    cascade = SnowflakeCascade(
        model_sizes=['33m', '22m'],  # Using only smaller models for this example
        search_factor=4
    )
    
    # Index documents
    print("Indexing documents with all models in the cascade...")
    start_time = time.time()
    cascade.index_documents(
        documents=documents,
        text_field='text',
        show_progress=True
    )
    index_time = time.time() - start_time
    print(f"Indexing completed in {index_time:.2f} seconds")
    
    # Example queries
    queries = [
        "technology innovations",
        "scientific research",
        "health and wellness",
        "business strategies"
    ]
    
    # Search with cascade
    print("\nSearching with cascade approach...")
    for query in queries:
        start_time = time.time()
        results = cascade.search(query, top_k=3)
        search_time = time.time() - start_time
        
        print(f"\nResults for query: '{query}' (took {search_time:.4f}s)")
        for i, (doc, score) in enumerate(results):
            print(f"{i+1}. {doc.get('title', 'N/A')} (Score: {score:.4f})")
            text = doc.get('text', 'N/A')
            print(f"   {text[:100]}..." if len(text) > 100 else f"   {text}")
    
    # Compare with direct search using smallest model
    print("\n" + "-" * 40)
    print("Comparing cascade search with direct search...")
    comparison = compare_search_methods(cascade, queries, top_k=5)
    
    # Print summary
    summary = comparison["summary"]
    print("\nComparison Summary:")
    print(f"Average cascade search time: {summary['avg_cascade_time']:.6f} seconds")
    print(f"Average direct search time: {summary['avg_direct_time']:.6f} seconds")
    print(f"Average speedup: {summary['avg_speedup']:.2f}x")
    print(f"Average result overlap: {summary['avg_overlap_ratio']:.2%}")


if __name__ == "__main__":
    main() 