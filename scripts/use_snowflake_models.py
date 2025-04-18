#!/usr/bin/env python3
"""
Use Snowflake Arctic Embed Models in Dimensional Cascade

This script utilizes the pre-trained snowflake-arctic-embed model family
in a dimensional cascade approach for efficient semantic search.

Available models:
- snowflake-arctic-embed:335m (default)
- snowflake-arctic-embed:137m
- snowflake-arctic-embed:110m
- snowflake-arctic-embed:33m
- snowflake-arctic-embed:22m
"""
import os
import sys
import argparse
import json
import time
from typing import List, Dict, Any, Optional
import logging
import numpy as np
import torch
from tqdm import tqdm

# Add project root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, project_root)

from dimensional_cascade.utils.metrics import time_function
from dimensional_cascade.utils.io import load_jsonl, save_metrics, save_jsonl

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('dimensional_cascade.snowflake')

# Import transformers for loading models
try:
    from transformers import AutoTokenizer, AutoModel
    HAVE_TRANSFORMERS = True
except ImportError:
    HAVE_TRANSFORMERS = False
    logger.error("transformers package is required. Install with: pip install transformers")
    sys.exit(1)

try:
    import faiss
    HAVE_FAISS = True
except ImportError:
    HAVE_FAISS = False
    logger.error("faiss-cpu package is required. Install with: pip install faiss-cpu")
    sys.exit(1)


class SnowflakeModel:
    """Wrapper for snowflake-arctic-embed models."""
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize Snowflake model.
        
        Args:
            model_name: Snowflake model name (e.g., 'snowflake-arctic-embed:335m')
            device: Device to use for inference ('cpu', 'cuda', etc.)
        """
        self.model_name = model_name
        
        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading model {model_name} on {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        
        # Get embedding dimension
        with torch.no_grad():
            # Use a dummy input to get the output dimension
            inputs = self.tokenizer("This is a test", return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            self.embedding_dim = outputs.last_hidden_state.size(-1)
        
        logger.info(f"Model {model_name} loaded with dimension {self.embedding_dim}")
    
    def embed(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Normalized embedding vector
        """
        return self.embed_batch([text])[0]
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            
        Returns:
            Array of normalized embedding vectors
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch, 
                padding=True, 
                truncation=True, 
                return_tensors="pt"
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Use CLS token embeddings (first token)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                # Normalize
                norms = np.linalg.norm(batch_embeddings, axis=1, keepdims=True)
                batch_embeddings = batch_embeddings / norms
                
                embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)


class SnowflakeCascade:
    """Dimensional cascade using Snowflake models."""
    
    def __init__(
        self,
        model_sizes: List[str] = None,
        device: Optional[str] = None,
        search_factor: int = 4
    ):
        """
        Initialize Snowflake cascade.
        
        Args:
            model_sizes: List of model sizes to use, from largest to smallest
                         (e.g., ['335m', '137m', '33m', '22m'])
            device: Device to use for inference
            search_factor: Factor for cascade search (how many more results to fetch at each level)
        """
        # Default model sizes if not provided
        if model_sizes is None:
            model_sizes = ['335m', '137m', '33m', '22m']
        
        self.model_sizes = model_sizes
        self.search_factor = search_factor
        self.device = device
        
        # Initialize models
        self.models = {}
        self.dimensions = {}
        self.indices = {}
        self.document_store = None
        
        logger.info(f"Initializing Snowflake cascade with models: {model_sizes}")
        
        for size in model_sizes:
            model_name = f"snowflake-arctic-embed:{size}"
            self.models[size] = SnowflakeModel(model_name, device)
            self.dimensions[size] = self.models[size].embedding_dim
        
        logger.info(f"Cascade initialization complete. Dimensions: {self.dimensions}")
    
    def index_documents(
        self,
        documents: List[Dict[str, Any]],
        text_field: str = "text",
        batch_size: int = 32,
        show_progress: bool = True
    ):
        """
        Index documents using all models in the cascade.
        
        Args:
            documents: List of documents to index
            text_field: Field containing text to embed
            batch_size: Batch size for embedding
            show_progress: Whether to show progress bar
        """
        # Store documents
        self.document_store = {i: doc for i, doc in enumerate(documents)}
        
        # Get texts to embed
        texts = [doc.get(text_field, "") for doc in documents]
        
        # Index with each model
        for size in self.model_sizes:
            model = self.models[size]
            dim = self.dimensions[size]
            
            logger.info(f"Indexing {len(documents)} documents with model {size} (dim={dim})")
            
            # Create FAISS index
            index = faiss.IndexFlatIP(dim)  # Inner product (cosine similarity for normalized vectors)
            
            # Generate embeddings and add to index
            iterator = range(0, len(texts), batch_size)
            if show_progress:
                iterator = tqdm(iterator, desc=f"Indexing with {size} model")
            
            for i in iterator:
                batch = texts[i:i + batch_size]
                embeddings = model.embed_batch(batch)
                index.add(embeddings)
            
            self.indices[size] = index
            
            logger.info(f"Indexed {len(documents)} documents with model {size}")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        smallest_only: bool = False
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for query using cascade of models.
        
        Args:
            query: Query text
            top_k: Number of results to return
            smallest_only: If True, only use the smallest model
            
        Returns:
            List of (document, score) tuples
        """
        if smallest_only:
            # Use only the smallest model
            size = self.model_sizes[-1]
            model = self.models[size]
            index = self.indices[size]
            
            # Embed query
            query_embedding = model.embed(query)
            
            # Search
            scores, doc_indices = index.search(np.array([query_embedding]), top_k)
            
            # Get documents
            results = []
            for i, doc_idx in enumerate(doc_indices[0]):
                if doc_idx < 0:  # FAISS may return -1 for not enough results
                    continue
                doc = self.document_store[int(doc_idx)]
                score = float(scores[0][i])
                results.append((doc, score))
            
            return results
        
        # Use cascade approach
        # Start with the smallest model
        size = self.model_sizes[-1]
        model = self.models[size]
        index = self.indices[size]
        
        # Embed query with smallest model
        query_embedding = model.embed(query)
        
        # Search with wider net
        cascade_k = top_k * self.search_factor
        scores, candidate_indices = index.search(np.array([query_embedding]), cascade_k)
        
        # Filter out invalid indices
        candidates = [int(idx) for idx in candidate_indices[0] if idx >= 0]
        
        # If we have enough candidates, refine with larger models
        for size in reversed(self.model_sizes[:-1]):
            if len(candidates) <= top_k:
                break
                
            model = self.models[size]
            
            # Get candidate documents
            candidate_docs = [self.document_store[idx] for idx in candidates]
            candidate_texts = [doc.get("text", "") for doc in candidate_docs]
            
            # Re-embed query with current model
            query_embedding = model.embed(query)
            
            # Re-embed candidates
            candidate_embeddings = model.embed_batch(candidate_texts)
            
            # Compute similarities
            similarities = np.dot(candidate_embeddings, query_embedding)
            
            # Sort candidates by similarity
            sorted_indices = np.argsort(-similarities)
            candidates = [candidates[idx] for idx in sorted_indices[:top_k]]
        
        # Get final results
        results = []
        
        # Get the largest model for final scoring
        largest_model = self.models[self.model_sizes[0]]
        
        # Get candidate documents
        candidate_docs = [self.document_store[idx] for idx in candidates[:top_k]]
        candidate_texts = [doc.get("text", "") for doc in candidate_docs]
        
        # Re-embed query and candidates with largest model
        query_embedding = largest_model.embed(query)
        candidate_embeddings = largest_model.embed_batch(candidate_texts)
        
        # Compute final similarities
        similarities = np.dot(candidate_embeddings, query_embedding)
        
        # Create results
        for i, idx in enumerate(candidates[:top_k]):
            if i < len(similarities):
                doc = self.document_store[idx]
                score = float(similarities[i])
                results.append((doc, score))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def save(self, path: str):
        """
        Save cascade metadata.
        
        Args:
            path: Directory to save metadata
        """
        os.makedirs(path, exist_ok=True)
        
        metadata = {
            "model_sizes": self.model_sizes,
            "dimensions": self.dimensions,
            "search_factor": self.search_factor,
            "num_documents": len(self.document_store) if self.document_store else 0
        }
        
        # Save metadata
        with open(os.path.join(path, "cascade_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved cascade metadata to {path}")
        
        # Save indices
        for size, index in self.indices.items():
            index_path = os.path.join(path, f"index_{size}.faiss")
            faiss.write_index(index, index_path)
            logger.info(f"Saved index for model {size} to {index_path}")
        
        # Save document store
        if self.document_store:
            docs_path = os.path.join(path, "documents.jsonl")
            docs = [self.document_store[i] for i in range(len(self.document_store))]
            save_jsonl(docs, docs_path)
            logger.info(f"Saved {len(docs)} documents to {docs_path}")
    
    def load(self, path: str):
        """
        Load cascade from saved metadata.
        
        Args:
            path: Directory containing saved metadata
        """
        # Load metadata
        metadata_path = os.path.join(path, "cascade_metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        self.model_sizes = metadata["model_sizes"]
        self.dimensions = metadata["dimensions"]
        self.search_factor = metadata["search_factor"]
        
        # Load indices
        self.indices = {}
        for size in self.model_sizes:
            index_path = os.path.join(path, f"index_{size}.faiss")
            index = faiss.read_index(index_path)
            self.indices[size] = index
            logger.info(f"Loaded index for model {size} from {index_path}")
        
        # Load document store
        docs_path = os.path.join(path, "documents.jsonl")
        documents = load_jsonl(docs_path)
        self.document_store = {i: doc for i, doc in enumerate(documents)}
        logger.info(f"Loaded {len(documents)} documents from {docs_path}")
        
        logger.info(f"Loaded cascade from {path}")
        
        # Ensure models are loaded
        if not self.models:
            for size in self.model_sizes:
                model_name = f"snowflake-arctic-embed:{size}"
                self.models[size] = SnowflakeModel(model_name, self.device)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Use Snowflake Arctic Embed models in dimensional cascade')
    
    parser.add_argument(
        '--data', 
        type=str, 
        required=True,
        help='Path to documents file (JSONL format)'
    )
    
    parser.add_argument(
        '--text-field', 
        type=str, 
        default='text',
        help='Document field containing text to embed'
    )
    
    parser.add_argument(
        '--models', 
        type=str, 
        default='335m,137m,33m,22m',
        help='Comma-separated list of model sizes to use'
    )
    
    parser.add_argument(
        '--index-path', 
        type=str, 
        default='snowflake_index',
        help='Path to save/load index'
    )
    
    parser.add_argument(
        '--query', 
        type=str, 
        default=None,
        help='Query to search for'
    )
    
    parser.add_argument(
        '--query-file', 
        type=str, 
        default=None,
        help='File containing queries (one per line)'
    )
    
    parser.add_argument(
        '--top-k', 
        type=int, 
        default=10,
        help='Number of results to return'
    )
    
    parser.add_argument(
        '--search-factor', 
        type=int, 
        default=4,
        help='Factor for cascade search'
    )
    
    parser.add_argument(
        '--compare', 
        action='store_true',
        help='Compare cascade vs. direct search'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='snowflake_results.json',
        help='Path to save search results'
    )
    
    parser.add_argument(
        '--device', 
        type=str, 
        default=None,
        help='Device to use for inference (cpu, cuda, etc.)'
    )
    
    parser.add_argument(
        '--skip-indexing', 
        action='store_true',
        help='Skip indexing and load existing index'
    )
    
    return parser.parse_args()


def load_documents(file_path: str) -> List[Dict[str, Any]]:
    """Load documents from JSONL file."""
    logger.info(f"Loading documents from {file_path}")
    documents = load_jsonl(file_path)
    logger.info(f"Loaded {len(documents)} documents")
    return documents


def load_queries(file_path: str) -> List[str]:
    """Load queries from a file."""
    logger.info(f"Loading queries from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(queries)} queries")
    return queries


def compare_search_methods(
    cascade: SnowflakeCascade,
    queries: List[str],
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Compare cascade search with direct search using the smallest model.
    
    Args:
        cascade: SnowflakeCascade instance
        queries: List of queries to search for
        top_k: Number of results to return
        
    Returns:
        Dictionary with comparison results
    """
    results = {
        "queries": [],
        "summary": {}
    }
    
    cascade_times = []
    direct_times = []
    overlap_ratios = []
    
    for query in queries:
        query_result = {"query": query, "cascade": {}, "direct": {}, "comparison": {}}
        
        # Cascade search
        start_time = time.time()
        cascade_results = cascade.search(query, top_k=top_k)
        cascade_time = time.time() - start_time
        cascade_times.append(cascade_time)
        
        # Direct search with smallest model
        start_time = time.time()
        direct_results = cascade.search(query, top_k=top_k, smallest_only=True)
        direct_time = time.time() - start_time
        direct_times.append(direct_time)
        
        # Calculate overlap
        cascade_ids = [doc.get("id") for doc, _ in cascade_results]
        direct_ids = [doc.get("id") for doc, _ in direct_results]
        
        intersection = set(cascade_ids).intersection(set(direct_ids))
        overlap_ratio = len(intersection) / top_k if top_k > 0 else 0
        overlap_ratios.append(overlap_ratio)
        
        # Save results
        query_result["cascade"] = {
            "time": cascade_time,
            "results": [{"doc": doc, "score": score} for doc, score in cascade_results]
        }
        
        query_result["direct"] = {
            "time": direct_time,
            "results": [{"doc": doc, "score": score} for doc, score in direct_results]
        }
        
        query_result["comparison"] = {
            "overlap_ratio": overlap_ratio,
            "speedup": direct_time / cascade_time if cascade_time > 0 else 0,
            "cascade_time": cascade_time,
            "direct_time": direct_time
        }
        
        results["queries"].append(query_result)
    
    # Calculate summary
    avg_cascade_time = sum(cascade_times) / len(cascade_times) if cascade_times else 0
    avg_direct_time = sum(direct_times) / len(direct_times) if direct_times else 0
    avg_overlap = sum(overlap_ratios) / len(overlap_ratios) if overlap_ratios else 0
    avg_speedup = avg_direct_time / avg_cascade_time if avg_cascade_time > 0 else 0
    
    results["summary"] = {
        "avg_cascade_time": avg_cascade_time,
        "avg_direct_time": avg_direct_time,
        "avg_overlap_ratio": avg_overlap,
        "avg_speedup": avg_speedup,
        "num_queries": len(queries)
    }
    
    return results


def main():
    args = parse_arguments()
    
    # Parse model sizes
    model_sizes = args.models.split(',')
    
    # Create cascade
    cascade = SnowflakeCascade(
        model_sizes=model_sizes,
        device=args.device,
        search_factor=args.search_factor
    )
    
    # Load existing index or create new one
    if args.skip_indexing and os.path.exists(args.index_path):
        logger.info(f"Loading existing index from {args.index_path}")
        cascade.load(args.index_path)
    else:
        # Load documents
        documents = load_documents(args.data)
        
        # Index documents
        cascade.index_documents(
            documents=documents,
            text_field=args.text_field,
            show_progress=True
        )
        
        # Save index
        logger.info(f"Saving index to {args.index_path}")
        cascade.save(args.index_path)
    
    # Handle queries
    if args.query:
        # Single query
        results = cascade.search(args.query, top_k=args.top_k)
        
        # Print results
        print(f"\nResults for query: '{args.query}'")
        for i, (doc, score) in enumerate(results):
            print(f"{i+1}. Score: {score:.4f}")
            print(f"   ID: {doc.get('id', 'N/A')}")
            print(f"   Title: {doc.get('title', 'N/A')}")
            text = doc.get(args.text_field, 'N/A')
            print(f"   Text: {text[:200]}..." if len(text) > 200 else f"   Text: {text}")
            print()
    
    elif args.query_file:
        # Multiple queries
        queries = load_queries(args.query_file)
        
        if args.compare:
            # Compare cascade vs. direct search
            logger.info("Comparing cascade search with direct search")
            comparison_results = compare_search_methods(
                cascade=cascade,
                queries=queries,
                top_k=args.top_k
            )
            
            # Save results
            with open(args.output, 'w') as f:
                json.dump(comparison_results, f, indent=2)
            
            # Print summary
            summary = comparison_results["summary"]
            print("\nComparison Summary:")
            print(f"Average cascade search time: {summary['avg_cascade_time']:.6f} seconds")
            print(f"Average direct search time: {summary['avg_direct_time']:.6f} seconds")
            print(f"Average speedup: {summary['avg_speedup']:.2f}x")
            print(f"Average result overlap: {summary['avg_overlap_ratio']:.4f}")
            print(f"Results saved to {args.output}")
        
        else:
            # Just search with cascade
            all_results = []
            
            for query in queries:
                results = cascade.search(query, top_k=args.top_k)
                all_results.append({
                    "query": query,
                    "results": [{"doc": doc, "score": score} for doc, score in results]
                })
            
            # Save results
            with open(args.output, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            print(f"Search results for {len(queries)} queries saved to {args.output}")
    
    else:
        print("No query or query file provided. Use --query or --query-file")


if __name__ == "__main__":
    main() 