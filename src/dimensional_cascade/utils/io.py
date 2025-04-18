"""
I/O utilities for the Dimensional Cascade.
"""
import json
from typing import Dict, List, Any, Union, TextIO


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


def load_corpus(file_path: str) -> Dict[int, Dict[str, Any]]:
    """Load a corpus with document IDs as keys.
    
    Args:
        file_path: Path to the corpus file
        
    Returns:
        Dictionary mapping from document ID to document
    """
    documents = load_jsonl(file_path)
    corpus = {}
    
    for doc in documents:
        doc_id = doc.get('id')
        if doc_id is not None:
            corpus[doc_id] = doc
    
    return corpus


def save_results(
    results: List[Dict[str, Any]], 
    file_path: str, 
    include_fields: List[str] = None
) -> None:
    """Save search results or evaluation metrics to file.
    
    Args:
        results: List of result dictionaries
        file_path: Path to save the results
        include_fields: Fields to include (if None, include all)
    """
    if include_fields:
        # Filter fields
        filtered_results = []
        for result in results:
            filtered_result = {k: v for k, v in result.items() if k in include_fields}
            filtered_results.append(filtered_result)
        results = filtered_results
    
    save_jsonl(results, file_path)
    
    
def load_queries(file_path: str) -> List[str]:
    """Load a list of queries from a file.
    
    Args:
        file_path: Path to the queries file (one query per line)
        
    Returns:
        List of query strings
    """
    queries = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                queries.append(line)
                
    return queries


def save_queries(queries: List[str], file_path: str) -> None:
    """Save a list of queries to a file.
    
    Args:
        queries: List of query strings
        file_path: Path to save the queries
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for query in queries:
            f.write(query + '\n')
            
            
def save_metrics(metrics: Dict[str, Any], file_path: str) -> None:
    """Save evaluation metrics to a JSON file.
    
    Args:
        metrics: Dictionary of metrics
        file_path: Path to save the metrics
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False) 