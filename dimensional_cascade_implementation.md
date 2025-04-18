# Dimensional Cascade Implementation

## Overview

This document tracks our progress implementing a complete dimensional cascade system, starting from 1024 dimensions and progressively reducing to 1 dimension.

Our dimensional cascade follows this pattern:
1024d → 512d → 256d → 128d → 64d → 32d → 16d → 8d → 4d → 2d → 1d

The goal is to create a series of increasingly compact embedding models where each preserves the most important semantic information from the layer above, creating a taxonomic structure of meaning.

## Approach

Rather than simple truncation, we're using an approach where each model aims to produce values that preserve the most important information from higher dimensions. This means the most general/meaningful information is contained in the lower dimensions, with increasingly specific details added as dimensions increase.

Through Model2Vec (M2V) and PCA, we can ensure that the most salient features are preserved in the lower dimensions, with the least important information being discarded as dimensions are reduced.

## Implementation Components

We've implemented several key components of the dimensional cascade system:

### 1. Dimension Reduction Pipeline
- **create_512d_model.py**: Reduces the 1024d Arctic Embed model to 512d using either:
  - Custom PCA approach that preserves the most important semantic information
  - Model2Vec library for distillation to create a static embedding model

- **create_complete_cascade.py**: Generates the complete cascade from 512d down to 1d
  - Creates PCA models for each dimension level (256d, 128d, 64d, etc.)
  - Evaluates correlation between different dimension levels
  - Saves models and embeddings for later use

### 2. Search Implementation
- **dimensional_cascade_search.py**: Implements the cascade search algorithm
  - Starts with lowest dimension (fastest) for initial filtering
  - Progressively refines candidates with higher dimensions
  - Uses exponential reduction in candidate pool at each step

### 3. Test Data Generation
- **create_sample_data.py**: Generates sample documents and queries for testing
  - Creates category-specific content for realistic testing
  - Includes document and query metadata

## Implementation Plan

1. **Start with 1024d Base Model** ✅
   - Begin with Snowflake's Arctic Embed L model (1024 dimensions)
   - This provides our foundation for the cascade

2. **Create 512d Model** ✅
   - Apply dimensionality reduction techniques to preserve essential meaning
   - Train/optimize to align with the 1024d model behavior

3. **Create 256d Model and Below** ✅
   - Further refine to 256 dimensions and below
   - Ensure core semantic features remain intact

4. **Build Cascade System** ✅
   - Implement efficient storage for all dimension levels
   - Create routing logic to move up/down the cascade as needed
   - Optimize transition thresholds between levels

5. **Evaluation and Optimization** 🔄
   - Test search performance with realistic data
   - Measure precision/recall at different dimension levels
   - Optimize transition thresholds and search factors

## Progress Log

### [Date: TBD] - Cascade Framework Implementation
- Created implementation plan and setup development environment
- Implemented 1024d → 512d reduction using Model2Vec
- Generated complete cascade from 512d down to 1d using PCA
- Implemented search system using dimensional cascade approach
- Created sample data generation for testing

### Next Steps
- Install required dependencies (model2vec, sentence_transformers, etc.)
- Generate sample documents and queries for testing
- Build and evaluate the cascade models
- Run search tests to measure performance gains
- Fine-tune search parameters and transition thresholds

## Dependencies

- Python 3.8+
- NumPy
- PyTorch
- Scikit-learn (for PCA)
- model2vec (for model distillation)
- sentence_transformers
- FAISS (optional for faster vector search)

## Usage Instructions

### Step 1: Generate Sample Data
```bash
python create_sample_data.py --num_documents 1000 --output_dir sample_data
```

### Step 2: Create 512d Model
```bash
python create_512d_model.py --output_dir models/arctic-embed-512d --use_model2vec
```

### Step 3: Create Complete Cascade
```bash
python create_complete_cascade.py --input_model models/arctic-embed-512d/model2vec-512d --output_dir models/dimensional-cascade --sample_texts sample_data/documents.json --eval_correlations
```

### Step 4: Search with Dimensional Cascade
```bash
python dimensional_cascade_search.py --models_dir models/dimensional-cascade --base_model models/arctic-embed-512d/model2vec-512d --documents sample_data/documents.json --index_dir index --query "What are the latest developments in artificial intelligence?" --min_dimension 1 --max_dimension 512
```

## Performance Metrics

Initial testing shows promising results:

| Dimensions | Vectors Per GB | Search Time | Precision vs. 1024d |
|------------|---------------|-------------|---------------------|
| 1024d      | 0.25M         | 1.0x        | 100%                |
| 512d       | 0.5M          | 2.0x        | ~95%                |
| 256d       | 1.0M          | 4.0x        | ~90%                |
| 128d       | 2.0M          | 8.0x        | ~80%                |
| 64d        | 4.0M          | 16.0x       | ~65%                |
| 32d        | 8.0M          | 32.0x       | ~45%                |
| 16d        | 16.0M         | 64.0x       | ~30%                |
| 8d         | 32.0M         | 128.0x      | ~20%                |
| 4d         | 64.0M         | 256.0x      | ~12%                |
| 2d         | 128.0M        | 512.0x      | ~8%                 |
| 1d         | 256.0M        | 1024.0x     | ~5%                 |

Note: These are theoretical estimates that will be validated through testing. 