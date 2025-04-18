#!/usr/bin/env python
"""
Basic tests for the Dimensional Cascade.

Tests include:
1. DimensionReducer functionality
2. ModelHierarchy embedding at different dimensions
3. MultiResolutionIndex operations
4. Full cascade search pipeline
"""
import unittest
import numpy as np
import tempfile
import os
import shutil
from typing import List, Dict, Any

from dimensional_cascade.core import (
    DimensionReducer, 
    ModelHierarchy, 
    MultiResolutionIndex,
    DimensionalCascade,
    CascadeConfig
)


class TestDimensionReducer(unittest.TestCase):
    """Test the DimensionReducer class."""
    
    def setUp(self):
        """Set up test data."""
        self.input_dim = 64
        self.output_dim = 32
        self.reducer = DimensionReducer(self.input_dim, self.output_dim)
        
        # Create random embeddings for testing
        self.num_samples = 10
        self.embeddings = np.random.randn(self.num_samples, self.input_dim).astype(np.float32)
    
    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.reducer.input_dim, self.input_dim)
        self.assertEqual(self.reducer.output_dim, self.output_dim)
        
    def test_fit(self):
        """Test fitting the reducer."""
        self.reducer.fit(self.embeddings)
        self.assertTrue(hasattr(self.reducer, 'pca'))
        
    def test_transform(self):
        """Test transforming embeddings."""
        self.reducer.fit(self.embeddings)
        reduced = self.reducer.transform(self.embeddings)
        
        # Check shape
        self.assertEqual(reduced.shape, (self.num_samples, self.output_dim))
        
    def test_fit_transform(self):
        """Test fit_transform method."""
        reduced = self.reducer.fit_transform(self.embeddings)
        
        # Check shape
        self.assertEqual(reduced.shape, (self.num_samples, self.output_dim))


class TestModelHierarchy(unittest.TestCase):
    """Test the ModelHierarchy class."""
    
    def setUp(self):
        """Set up test data."""
        # Use a small model for faster tests
        self.model_path = "sentence-transformers/paraphrase-MiniLM-L3-v2"
        self.dimensions = [384, 192, 96, 48]
        
        # Create temporary directory for the model
        self.temp_dir = tempfile.mkdtemp()
        
        # Initialize model hierarchy
        self.model_hierarchy = ModelHierarchy(
            model_path=self.model_path,
            dimensions=self.dimensions,
            cache_dir=self.temp_dir
        )
        
        # Sample text for embedding
        self.texts = [
            "This is a test document",
            "Another sample text for embedding"
        ]
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.model_hierarchy.dimensions, self.dimensions)
        self.assertEqual(self.model_hierarchy.model_path, self.model_path)
        
    def test_embed_batch(self):
        """Test batch embedding at different dimensions."""
        for dim in self.dimensions:
            embeddings = self.model_hierarchy.embed_batch(self.texts, dimension=dim)
            
            # Check shape
            self.assertEqual(embeddings.shape, (len(self.texts), dim))
            
            # Check normalization
            for emb in embeddings:
                # Approximately normalized (floating point precision issues)
                self.assertAlmostEqual(np.linalg.norm(emb), 1.0, places=6)
    
    def test_embed(self):
        """Test single text embedding."""
        for dim in self.dimensions:
            embedding = self.model_hierarchy.embed(self.texts[0], dimension=dim)
            
            # Check shape
            self.assertEqual(embedding.shape, (dim,))
            
            # Check normalization
            self.assertAlmostEqual(np.linalg.norm(embedding), 1.0, places=6)
    
    def test_dimension_reduction(self):
        """Test dimension reduction through the hierarchy."""
        # Get highest dimension embedding
        high_dim = max(self.dimensions)
        high_emb = self.model_hierarchy.embed(self.texts[0], dimension=high_dim)
        
        # Test reducing to each lower dimension
        for dim in self.dimensions:
            if dim < high_dim:
                reduced_emb = self.model_hierarchy.reduce_dimension(high_emb, high_dim, dim)
                
                # Check shape
                self.assertEqual(reduced_emb.shape, (dim,))
                
                # Should be approximately normalized
                self.assertAlmostEqual(np.linalg.norm(reduced_emb), 1.0, places=6)


class TestMultiResolutionIndex(unittest.TestCase):
    """Test the MultiResolutionIndex class."""
    
    def setUp(self):
        """Set up test data."""
        self.dimensions = [64, 32, 16]
        self.temp_dir = tempfile.mkdtemp()
        
        # Create index
        self.index = MultiResolutionIndex(
            dimensions=self.dimensions,
            index_path=self.temp_dir
        )
        
        # Create test documents and embeddings
        self.num_docs = 20
        self.docs = []
        for i in range(self.num_docs):
            self.docs.append({
                "id": i,
                "title": f"Document {i}",
                "text": f"This is test document number {i}"
            })
        
        # Create random embeddings for each dimension
        self.embeddings = {}
        for dim in self.dimensions:
            self.embeddings[dim] = np.random.randn(self.num_docs, dim).astype(np.float32)
            # Normalize
            for i in range(self.num_docs):
                self.embeddings[dim][i] = self.embeddings[dim][i] / np.linalg.norm(self.embeddings[dim][i])
    
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)
    
    def test_add_documents(self):
        """Test adding documents to index."""
        # Add documents with embeddings
        for dim in self.dimensions:
            self.index.add_documents(self.docs, self.embeddings[dim], dim)
        
        # Verify that documents were added
        self.assertEqual(len(self.index.get_document_ids()), self.num_docs)
    
    def test_save_load(self):
        """Test saving and loading the index."""
        # Add documents
        for dim in self.dimensions:
            self.index.add_documents(self.docs, self.embeddings[dim], dim)
        
        # Save the index
        self.index.save()
        
        # Create a new index and load
        new_index = MultiResolutionIndex(
            dimensions=self.dimensions,
            index_path=self.temp_dir
        )
        new_index.load()
        
        # Verify documents are present
        self.assertEqual(len(new_index.get_document_ids()), self.num_docs)
    
    def test_search(self):
        """Test searching the index."""
        # Add documents
        for dim in self.dimensions:
            self.index.add_documents(self.docs, self.embeddings[dim], dim)
        
        # Create a query embedding for each dimension
        for dim in self.dimensions:
            query_emb = np.random.randn(dim).astype(np.float32)
            query_emb = query_emb / np.linalg.norm(query_emb)
            
            # Search
            doc_indices, distances = self.index.search(query_emb, dim, top_k=5)
            
            # Verify results
            self.assertEqual(len(doc_indices), 5)
            self.assertEqual(len(distances), 5)
    
    def test_get_documents(self):
        """Test retrieving documents by ID."""
        # Add documents
        for dim in self.dimensions:
            self.index.add_documents(self.docs, self.embeddings[dim], dim)
        
        # Get specific documents
        ids_to_get = [3, 7, 12]
        retrieved_docs = self.index.get_documents(ids_to_get)
        
        # Verify
        self.assertEqual(len(retrieved_docs), len(ids_to_get))
        for doc in retrieved_docs:
            self.assertIn(doc["id"], ids_to_get)


class TestDimensionalCascade(unittest.TestCase):
    """Test the full DimensionalCascade."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Use a small model
        self.model_path = "sentence-transformers/paraphrase-MiniLM-L3-v2"
        self.dimensions = [384, 192, 96]
        
        # Create config
        self.config = CascadeConfig(
            dimensions=self.dimensions,
            candidate_multiplier=4,
            max_candidates=100
        )
        
        # Initialize cascade
        self.cascade = DimensionalCascade(
            model_path=self.model_path,
            index_path=os.path.join(self.temp_dir, "index"),
            config=self.config
        )
        
        # Create test documents
        self.num_docs = 30
        self.docs = []
        for i in range(self.num_docs):
            self.docs.append({
                "id": i,
                "title": f"Document {i}",
                "text": f"This is test document number {i}. It contains some test content."
            })
    
    def tearDown(self):
        """Clean up."""
        shutil.rmtree(self.temp_dir)
    
    def test_index_documents(self):
        """Test indexing documents."""
        # Index documents
        self.cascade.index_documents(self.docs)
        
        # Verify index was created
        for dim in self.dimensions:
            self.assertTrue(self.cascade.index.index_exists(dim))
    
    def test_search(self):
        """Test search functionality."""
        # Index documents
        self.cascade.index_documents(self.docs)
        
        # Perform search
        query = "test document"
        results = self.cascade.search(query, top_k=5)
        
        # Verify results
        self.assertEqual(len(results), 5)
        for doc, score in results:
            self.assertIn("id", doc)
            self.assertIn("title", doc)
            self.assertIn("text", doc)
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)
    
    def test_progressive_search(self):
        """Test the progressive search workflow."""
        # Index documents
        self.cascade.index_documents(self.docs)
        
        # Set a query
        query = "document with test content"
        
        # Embed query at highest dimension
        high_dim = max(self.dimensions)
        query_emb = self.cascade.model_hierarchy.embed(query, dimension=high_dim)
        
        # Search at highest dimension
        high_results, _ = self.cascade.index.search(query_emb, high_dim, top_k=10)
        
        # Create candidate set
        candidate_docs = self.cascade.index.get_document_ids(high_results)
        
        # Search at lowest dimension with candidates
        low_dim = min(self.dimensions)
        query_emb_low = self.cascade.model_hierarchy.embed(query, dimension=low_dim)
        
        # Search
        results = self.cascade.search(query, top_k=5)
        
        # Verify we have results
        self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main() 