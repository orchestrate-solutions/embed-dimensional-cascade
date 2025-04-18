# Dimensional Cascade Scripts

This directory contains utility scripts for working with the Dimensional Cascade models.

## Available Scripts

### 1. Precision Loss Analysis

`analyze_precision_loss.py` - Analyzes the precision loss between different dimensions in the Dimensional Cascade.

**Usage:**
```bash
python -m dimensional_cascade.scripts.analyze_precision_loss \
  --data /path/to/dataset.jsonl \
  --queries /path/to/queries.txt \
  --output-dir results \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --dimensions 1024,512,256,128,64,32,16 \
  --sample-size 1000 \
  --query-count 50
```

**Features:**
- Measures precision loss between high and low dimensions
- Compares theoretical vs. actual precision loss
- Analyzes latency improvements at lower dimensions
- Generates visualizations and detailed reports

### 2. Training Models

`train_cascade.py` - Trains a complete Dimensional Cascade model hierarchy using different dimension reduction approaches.

**Usage:**
```bash
python -m dimensional_cascade.scripts.train_cascade \
  --data /path/to/dataset.jsonl \
  --output-dir models \
  --base-model sentence-transformers/all-MiniLM-L6-v2 \
  --approach truncation \
  --dimensions 1024,512,256,128,64,32,16 \
  --batch-size 32 \
  --epochs 1
```

**Features:**
- Supports three dimensionality reduction approaches:
  - Truncation (using PCA)
  - Distillation (knowledge transfer)
  - Autoencoder (neural dimensionality reduction)
- Configurable dimension hierarchy
- Optimized training process

### 3. Using Snowflake Arctic Embed Models

`use_snowflake_models.py` - Implements dimensional cascade using pre-trained Snowflake Arctic Embed models.

**Usage:**
```bash
python -m dimensional_cascade.scripts.use_snowflake_models \
  --data /path/to/dataset.jsonl \
  --query "your search query" \
  --models 335m,137m,33m,22m \
  --index-path snowflake_index \
  --top-k 10
```

**Features:**
- Leverages pre-trained Snowflake Arctic Embed models (SoTA as of April 2024)
- Uses models of different sizes to create an efficient cascade:
  - snowflake-arctic-embed:335m (default/largest)
  - snowflake-arctic-embed:137m
  - snowflake-arctic-embed:110m
  - snowflake-arctic-embed:33m
  - snowflake-arctic-embed:22m (smallest)
- Implements cascade search with precision-vs-speed tradeoff control
- Includes benchmarking to compare cascade vs. direct search performance

### 4. Research Similar Approaches

`research_similar_approaches.py` - Researches similar approaches to Dimensional Cascade in academic papers and GitHub repositories.

**Usage:**
```bash
python -m dimensional_cascade.scripts.research_similar_approaches \
  --output-dir research \
  --github-token YOUR_TOKEN \
  --max-papers 50 \
  --max-repos 50
```

**Features:**
- Searches academic papers related to dimensionality reduction in vector search
- Finds similar GitHub repositories
- Scans Hugging Face for relevant pre-trained models
- Generates comprehensive research report

## Additional Requirements

Some scripts require additional dependencies beyond the base package:

```
scholarly>=1.7.0
requests>=2.25.0
matplotlib>=3.4.0
pandas>=1.3.0
tqdm>=4.62.0
transformers>=4.15.0
faiss-cpu>=1.7.0  # Use faiss-gpu for GPU support
huggingface-hub>=0.12.0  # For downloading models from HuggingFace
```

Install them with:

```bash
pip install -r scripts_requirements.txt
``` 