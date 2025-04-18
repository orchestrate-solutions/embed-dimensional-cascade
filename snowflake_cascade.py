#!/usr/bin/env python3
"""
Snowflake Arctic Embed Models Dimensional Cascade

A self-contained implementation of dimensional cascade using Snowflake Arctic Embed models.
This script provides efficient semantic search with a cascading approach using
progressively larger models.

Available models:
- snowflake-arctic-embed:335m (default/largest)
- snowflake-arctic-embed:137m
- snowflake-arctic-embed:110m
- snowflake-arctic-embed:33m
- snowflake-arctic-embed:22m (smallest)

Usage:
  python snowflake_cascade.py --data your_data.jsonl --query "your search query"
  python snowflake_cascade.py --data your_data.jsonl --query-file queries.txt --compare
"""
import os
import sys
import argparse
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger('snowflake_cascade')

# Check for required dependencies
try:
    import numpy as np
except ImportError:
    logger.error("numpy is required. Install with: pip install numpy")
    sys.exit(1)

try:
    import torch
    from torch.nn.functional import normalize
except ImportError:
    logger.error("torch is required. Install with: pip install torch")
    sys.exit(1)

try:
    from tqdm import tqdm
except ImportError:
    # Fallback to simple progress tracking if tqdm is not available
    class tqdm:
        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
            self.total = len(iterable) if iterable is not None else 0
            self.n = 0
            self.desc = kwargs.get('desc', '')
            
        def __iter__(self):
            for item in self.iterable:
                self.n += 1
                if self.n % 10 == 0:
                    print(f"{self.desc}: {self.n}/{self.total}", end='\r')
                yield item
            print()

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    logger.error("transformers is required. Install with: pip install transformers")
    sys.exit(1)

try:
    import faiss
except ImportError:
    logger.error("faiss-cpu is required. Install with: pip install faiss-cpu")
    sys.exit(1)


# ========== Utility Functions ==========

def time_function(func):
    """
    Decorator to measure execution time of a function.
    
    Args:
        func: The function to measure
        
    Returns:
        Wrapper function that returns (result, execution_time)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    return wrapper


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load documents from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of document dictionaries
    """
    documents = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    doc = json.loads(line)
                    documents.append(doc)
                except json.JSONDecodeError:
                    # Skip invalid lines
                    pass
                
    return documents


def save_jsonl(documents: List[Dict[str, Any]], file_path: str) -> None:
    """Save documents to a JSONL file.
    
    Args:
        documents: List of document dictionaries
        file_path: Path to save the JSONL file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for doc in documents:
            json_line = json.dumps(doc, ensure_ascii=False)
            f.write(json_line + '\n')


def save_metrics(metrics: Dict[str, Any], file_path: str) -> None:
    """
    Save metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics
        file_path: Path to save the metrics
    """
    with open(file_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {file_path}")


def load_queries(file_path: str) -> List[str]:
    """Load queries from a file.
    
    Args:
        file_path: Path to queries file (one query per line)
        
    Returns:
        List of query strings
    """
    logger.info(f"Loading queries from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        queries = [line.strip() for line in f if line.strip()]
    logger.info(f"Loaded {len(queries)} queries")
    return queries


def chunk_text(text: str, max_tokens: int = 250, overlap: int = 20) -> List[str]:
    """
    Split text into chunks of approximately max_tokens with some overlap.
    
    Args:
        text: The text to chunk
        max_tokens: Maximum tokens per chunk (approximate since we use word boundaries)
        overlap: Number of words to overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Rough approximation: consider each word as a token
    words = text.split()
    
    if len(words) <= max_tokens:
        return [text]
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(words):
        end_idx = start_idx + max_tokens
        
        # If we're at the end, just take the rest
        if end_idx >= len(words):
            chunks.append(" ".join(words[start_idx:]))
            break
        
        # Otherwise, create a chunk with the maximum tokens
        chunks.append(" ".join(words[start_idx:end_idx]))
        
        # Move start index for next chunk, including overlap
        start_idx = end_idx - overlap
    
    return chunks


# ========== Core Classes ==========

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
    
    def embed(self, text: str, chunk_size: int = 250, chunk_overlap: int = 20) -> np.ndarray:
        """
        Embed a single text and return the embedding.
        For long texts, chunks the text and averages the embeddings.
        
        Args:
            text: Text to embed
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
            
        Returns:
            The embedding vector (normalized)
        """
        # For short texts, embed directly
        if len(text.split()) <= chunk_size:
            # Add query prefix if embedding a query
            if self.is_query:
                text = f"query: {text}"
            
            inputs = self.tokenizer(text, padding=True, truncation=True, 
                                   return_tensors="pt", max_length=8192)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs[0][:, 0]  # CLS token embedding
                normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
            return normalized.cpu().numpy()[0]
        
        # For long texts, chunk and average embeddings
        chunks = chunk_text(text, max_tokens=chunk_size, overlap=chunk_overlap)
        
        # Embed each chunk
        chunk_embeddings = []
        for chunk in chunks:
            if self.is_query:
                chunk = f"query: {chunk}"
                
            inputs = self.tokenizer(chunk, padding=True, truncation=True, 
                                   return_tensors="pt", max_length=8192)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs[0][:, 0]  # CLS token embedding
                normalized = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                chunk_embeddings.append(normalized.cpu().numpy()[0])
        
        # Average the chunk embeddings
        avg_embedding = np.mean(chunk_embeddings, axis=0)
        # Re-normalize the average embedding
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        return avg_embedding

    def embed_batch(self, texts: List[str], batch_size: int = 32, chunk_size: int = 250, chunk_overlap: int = 20) -> np.ndarray:
        """
        Embed a batch of texts.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process at once
            chunk_size: Maximum tokens per chunk 
            chunk_overlap: Number of tokens to overlap between chunks
            
        Returns:
            Array of embeddings, shape (n_texts, embedding_dim)
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Embed each text, handling chunking individually
            batch_embeddings = [self.embed(text, chunk_size, chunk_overlap) for text in batch_texts]
            embeddings.extend(batch_embeddings)
            
        return np.array(embeddings)


class SnowflakeCascade:
    """Dimensional cascade using Snowflake models."""
    
    def __init__(
        self,
        model_sizes: List[str] = None,
        device: Optional[str] = None,
        search_factor: int = 4,
        chunk_size: int = 250,
        chunk_overlap: int = 20
    ):
        """
        Initialize Snowflake cascade.
        
        Args:
            model_sizes: List of model sizes to use, from largest to smallest
                         (e.g., ['335m', '137m', '33m', '22m'])
            device: Device to use for inference
            search_factor: Factor for cascade search (how many more results to fetch at each level)
            chunk_size: Maximum tokens per chunk
            chunk_overlap: Number of tokens to overlap between chunks
        """
        # Default model sizes if not provided
        if model_sizes is None:
            model_sizes = ['335m', '137m', '33m', '22m']
        
        self.model_sizes = model_sizes
        self.search_factor = search_factor
        self.device = device
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
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
        Index documents by embedding their text content.
        
        Args:
            documents: List of document dictionaries
            text_field: Key to access the text in each document
            batch_size: Batch size for embedding
            show_progress: Whether to show progress bar
        """
        self.documents = documents
        
        # Extract texts
        texts = [doc[text_field] for doc in documents]
        
        # Create progress bar if requested
        iterator = tqdm(texts, desc="Embedding documents") if show_progress else texts
        
        # Generate embeddings using the largest model
        embeddings = {}
        
        for size in self.model_sizes:
            model = self.models[size]
            
            # Convert to batch processing
            if show_progress:
                iterator.set_description(f"Embedding with {size} model")
            
            # Embed with chunking
            embeddings[size] = model.embed_batch(
                texts, 
                batch_size=batch_size,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            
        self.embeddings = embeddings
        
        # Build index for each model size
        self.indices = {}
        
        for size in self.model_sizes:
            if show_progress:
                iterator.set_description(f"Building index for {size} model")
            
            # Build index for this model size
            size_embeddings = embeddings[size]
            index = faiss.IndexFlatIP(size_embeddings.shape[1])
            index.add(size_embeddings)
            self.indices[size] = index
        
        return self
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        smallest_only: bool = False
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search documents by query.
        
        Args:
            query: Query text
            top_k: Number of results to return
            smallest_only: Whether to only use the smallest model
            
        Returns:
            List of (document, score) tuples
        """
        # Use hierarchy of models from smallest to largest
        sorted_sizes = sorted(self.model_sizes, key=lambda s: int(s.replace("s", "")))
        
        if smallest_only:
            # Only use the smallest model for search
            size = sorted_sizes[0]
            model = self.models[size]
            
            # Embed query with chunking if needed
            query_embedding = model.embed(query, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            
            # Search index
            scores, indices = self.indices[size].search(
                query_embedding.reshape(1, -1), top_k
            )
            
            # Return results
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx != -1:  # Skip invalid indices
                    results.append((self.documents[idx], float(score)))
            
            return results
            
        # Use cascade search
        candidates = set()
        final_results = []
        
        # Start with smallest model
        for i, size in enumerate(sorted_sizes):
            model = self.models[size]
            is_last = i == len(sorted_sizes) - 1
            
            # Number of candidates to retrieve with this model
            k = top_k if is_last else top_k * self.search_factor
            
            # Adjust k if we already have candidates
            if candidates and not is_last:
                # We need to retrieve enough candidates to ensure we have k * search_factor
                # unique candidates after union
                k = max(k - len(candidates), 0)
            
            # Skip if we already have enough candidates
            if k == 0 and not is_last:
                continue
                
            # Embed query with chunking if needed
            query_embedding = model.embed(query, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
            
            # Search index
            scores, indices = self.indices[size].search(
                query_embedding.reshape(1, -1), k
            )
            
            # Collect candidates
            model_candidates = [(idx, float(scores[0][i])) for i, idx in enumerate(indices[0]) if idx != -1]
            
            if is_last:
                # For the last (largest) model, create the final results
                # Start with existing candidates and their scores
                results_map = {idx: 0.0 for idx, _ in candidates}
                
                # Update with scores from the largest model
                for idx, score in model_candidates:
                    results_map[idx] = score
                
                # Sort by score
                sorted_results = sorted(results_map.items(), key=lambda x: x[1], reverse=True)
                
                # Return top_k results
                final_results = [(self.documents[idx], score) for idx, score in sorted_results[:top_k]]
            else:
                # Add to candidates for refinement
                candidates.update([(idx, score) for idx, score in model_candidates])
        
        return final_results
    
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


def compare_search_methods(
    cascade: SnowflakeCascade,
    queries: List[str],
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Compare different search methods using a common set of queries.
    
    Args:
        cascade: SnowflakeCascade instance
        queries: List of queries to use for comparison
        top_k: Number of results to return for each query
        
    Returns:
        Dictionary with comparison results
    """
    # Initialize results
    results = {
        "methods": {},
        "queries": {},
        "summary": {}
    }
    
    # List of methods to compare
    methods = [
        ("smallest", lambda q: cascade.search(q, top_k=top_k, smallest_only=True)),
        ("largest", lambda q: cascade.search(q, top_k=top_k * cascade.search_factor, smallest_only=True)),
        ("cascade", lambda q: cascade.search(q, top_k=top_k, smallest_only=False))
    ]
    
    # Track method stats
    for method_name, _ in methods:
        results["methods"][method_name] = {
            "avg_time": 0.0,
            "avg_results": 0.0,
            "total_time": 0.0,
            "query_times": []
        }
    
    # Run queries for each method
    for i, query in enumerate(queries):
        logger.info(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")
        results["queries"][query] = {}
        
        # Run each method
        for method_name, search_func in methods:
            # Time the search
            start_time = time.time()
            search_results = search_func(query)
            end_time = time.time()
            search_time = end_time - start_time
            
            # Record results
            results["queries"][query][method_name] = {
                "time": search_time,
                "num_results": len(search_results),
                "top_scores": [score for _, score in search_results[:min(3, len(search_results))]]
            }
            
            # Update method stats
            results["methods"][method_name]["total_time"] += search_time
            results["methods"][method_name]["query_times"].append(search_time)
    
    # Calculate averages
    for method_name in results["methods"]:
        method_data = results["methods"][method_name]
        method_data["avg_time"] = method_data["total_time"] / len(queries) if queries else 0
        
        # Calculate average number of results
        total_results = sum(results["queries"][q][method_name]["num_results"] for q in results["queries"])
        method_data["avg_results"] = total_results / len(queries) if queries else 0
    
    # Calculate recall (assuming largest model results are ground truth)
    if len(methods) > 1 and "largest" in results["methods"]:
        for method_name in results["methods"]:
            if method_name == "largest":
                continue
                
            # Calculate recall@k
            total_recall = 0.0
            total_queries = 0
            
            for query in results["queries"]:
                # Get results from this method and the largest model
                method_results = set(res[0] for res in cascade.search(query, top_k=top_k, smallest_only=(method_name == "smallest")))
                largest_results = set(res[0] for res in cascade.search(query, top_k=top_k, smallest_only=True))
                
                # Calculate recall
                if largest_results:
                    recall = len(method_results.intersection(largest_results)) / len(largest_results)
                    total_recall += recall
                    total_queries += 1
            
            # Calculate average recall
            if total_queries > 0:
                results["methods"][method_name]["recall"] = total_recall / total_queries
    
    # Calculate speedups
    if "largest" in results["methods"] and "cascade" in results["methods"]:
        largest_time = results["methods"]["largest"]["avg_time"]
        cascade_time = results["methods"]["cascade"]["avg_time"]
        
        if cascade_time > 0:
            results["summary"]["speedup"] = largest_time / cascade_time
    
    return results


# ========== Command Line Interface ==========

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Snowflake Cascade Search")
    
    # Command mode
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Index documents")
    index_parser.add_argument("--input", type=str, required=True, help="Input JSONL file")
    index_parser.add_argument("--output", type=str, required=True, help="Output index directory")
    index_parser.add_argument("--models", type=str, default="s,m,l", help="Comma-separated model sizes")
    index_parser.add_argument("--batch-size", type=int, default=32, help="Batch size for indexing")
    index_parser.add_argument("--text-field", type=str, default="text", help="Field containing text")
    index_parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, cpu)")
    index_parser.add_argument("--chunk-size", type=int, default=250, help="Maximum tokens per chunk")
    index_parser.add_argument("--chunk-overlap", type=int, default=20, help="Overlap tokens between chunks")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search documents")
    search_parser.add_argument("--index", type=str, required=True, help="Index directory")
    search_parser.add_argument("--query", type=str, help="Query text")
    search_parser.add_argument("--queries", type=str, help="File with queries (one per line)")
    search_parser.add_argument("--output", type=str, help="Output JSONL file for results")
    search_parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    search_parser.add_argument("--search-factor", type=int, default=4, help="Search factor for cascade search")
    search_parser.add_argument("--smallest-only", action="store_true", help="Only use smallest model")
    search_parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, cpu)")
    search_parser.add_argument("--chunk-size", type=int, default=250, help="Maximum tokens per chunk")
    search_parser.add_argument("--chunk-overlap", type=int, default=20, help="Overlap tokens between chunks")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare search methods")
    compare_parser.add_argument("--index", type=str, required=True, help="Index directory")
    compare_parser.add_argument("--queries", type=str, required=True, help="File with queries (one per line)")
    compare_parser.add_argument("--output", type=str, required=True, help="Output JSON file for results")
    compare_parser.add_argument("--top-k", type=int, default=10, help="Number of results to return")
    compare_parser.add_argument("--search-factor", type=int, default=4, help="Search factor for cascade search")
    compare_parser.add_argument("--device", type=str, default=None, help="Device to use (cuda, cpu)")
    compare_parser.add_argument("--chunk-size", type=int, default=250, help="Maximum tokens per chunk")
    compare_parser.add_argument("--chunk-overlap", type=int, default=20, help="Overlap tokens between chunks")
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_arguments()
    
    # Set up logging level
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if not args.command:
        logger.error("Please specify a command: index, search, or compare")
        return
    
    # Parse model sizes
    if args.command == "index":
        model_sizes = args.models.split(',')
        
        # Load documents
        logger.info(f"Loading documents from {args.input}")
        documents = load_jsonl(args.input)
        logger.info(f"Loaded {len(documents)} documents")
        
        # Create cascade
        cascade = SnowflakeCascade(
            model_sizes=model_sizes,
            device=args.device,
            search_factor=4,  # Default for indexing
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        # Index documents
        cascade.index_documents(
            documents=documents,
            text_field=args.text_field,
            batch_size=args.batch_size,
            show_progress=True
        )
        
        # Save index
        logger.info(f"Saving index to {args.output}")
        cascade.save(args.output)
        
    elif args.command == "search":
        # Load cascade
        logger.info(f"Loading index from {args.index}")
        cascade = SnowflakeCascade.load(args.index)
        
        # Update search parameters
        cascade.search_factor = args.search_factor
        cascade.chunk_size = args.chunk_size
        cascade.chunk_overlap = args.chunk_overlap
        
        # Process queries
        if args.query:
            # Search for a single query
            logger.info(f"Searching for: {args.query}")
            results = cascade.search(
                query=args.query,
                top_k=args.top_k,
                smallest_only=args.smallest_only
            )
            
            # Print results
            print(f"\nResults for query: {args.query}")
            print("-" * 80)
            for i, (doc, score) in enumerate(results):
                print(f"{i+1}. Score: {score:.4f}")
                text = doc.get("text", "")
                print(f"   {text[:200]}..." if len(text) > 200 else f"   {text}")
                print()
            
            # Save results if requested
            if args.output:
                search_results = {
                    "query": args.query,
                    "results": [
                        {"document": doc, "score": score}
                        for doc, score in results
                    ]
                }
                
                with open(args.output, 'w') as f:
                    json.dump(search_results, f, indent=2)
                logger.info(f"Results saved to {args.output}")
                
        elif args.queries:
            # Process multiple queries
            logger.info(f"Loading queries from {args.queries}")
            queries = load_queries(args.queries)
            logger.info(f"Loaded {len(queries)} queries")
            
            all_results = []
            
            for i, query in enumerate(queries):
                logger.info(f"Searching for query {i+1}/{len(queries)}: {query[:50]}...")
                results = cascade.search(
                    query=query,
                    top_k=args.top_k,
                    smallest_only=args.smallest_only
                )
                
                search_results = {
                    "query": query,
                    "results": [
                        {"document": doc, "score": score}
                        for doc, score in results
                    ]
                }
                
                all_results.append(search_results)
            
            # Save results if requested
            if args.output:
                with open(args.output, 'w') as f:
                    for result in all_results:
                        f.write(json.dumps(result) + '\n')
                logger.info(f"Results saved to {args.output}")
                
    elif args.command == "compare":
        # Load cascade
        logger.info(f"Loading index from {args.index}")
        cascade = SnowflakeCascade.load(args.index)
        
        # Update search parameters
        cascade.search_factor = args.search_factor
        cascade.chunk_size = args.chunk_size
        cascade.chunk_overlap = args.chunk_overlap
        
        # Load queries
        logger.info(f"Loading queries from {args.queries}")
        queries = load_queries(args.queries)
        logger.info(f"Loaded {len(queries)} queries")
        
        # Compare search methods
        logger.info("Comparing search methods")
        results = compare_search_methods(
            cascade=cascade,
            queries=queries,
            top_k=args.top_k
        )
        
        # Save results
        save_metrics(results, args.output)
        logger.info(f"Comparison results saved to {args.output}")
        
        # Print summary
        print("\nSearch Method Comparison Summary")
        print("-" * 40)
        for method, metrics in results["methods"].items():
            print(f"Method: {method}")
            print(f"  Average Time: {metrics['avg_time']:.4f} seconds")
            print(f"  Average Results: {metrics['avg_results']}")
            if "recall" in metrics:
                print(f"  Average Recall: {metrics['recall']:.4f}")
            print()
        
        # Print time comparison
        fastest = min(results["methods"].items(), key=lambda x: x[1]["avg_time"])
        print(f"Fastest method: {fastest[0]} ({fastest[1]['avg_time']:.4f} seconds)")
        
        if "cascade" in results["methods"] and "largest" in results["methods"]:
            speedup = results["methods"]["largest"]["avg_time"] / results["methods"]["cascade"]["avg_time"]
            print(f"Speedup of cascade vs. largest: {speedup:.2f}x")
    
    else:
        logger.error(f"Unknown command: {args.command}")
    
    logger.info("Done!")


if __name__ == "__main__":
    main() 