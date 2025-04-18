# Dimensional Cascade

A library for progressive dimensionality reduction that preserves similarity relationships across multiple dimensions. Dimensional Cascade efficiently compresses high-dimensional vector embeddings into a cascade of lower-dimensional representations to accelerate similarity search while maintaining accuracy.

## Key Features

- **Progressive Dimensionality Reduction**: Cascade approach reduces dimensions while preserving similarity relationships
- **Neural Distillation**: Neural networks trained to distill high-dimensional embeddings into lower dimensions
- **Similarity Preservation**: Optimized for maintaining cosine similarity relationships
- **Flexible Architecture**: Support for both direct and sequential distillation strategies
- **Comprehensive Evaluation**: Tools for measuring recall, precision, and time efficiency

## Installation

```bash
# Clone the repository
git clone https://github.com/username/dimensional-cascade.git
cd dimensional-cascade

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training a Dimension Distiller

Train a model to distill high-dimensional embeddings into lower dimensions:

```bash
python src/scripts/train_distiller.py \
    --vectors_file path/to/vectors.npy \
    --dimensions 32,64,128,256 \
    --batch_size 64 \
    --epochs 50 \
    --strategy direct \
    --output_dir output
```

If no vectors file is provided, the script will generate synthetic data for demonstration purposes.

### Evaluating Search Performance

Evaluate the search performance of the dimensional cascade approach:

```bash
python src/evaluation/evaluator.py \
    --vectors_file path/to/vectors.npy \
    --query_file path/to/queries.npy \
    --gt_file path/to/ground_truth.npy \
    --k 10 \
    --metrics recall,precision,time \
    --output_dir results
```

## Project Structure

```
dimensional-cascade/
├── requirements.txt         # Project dependencies
├── src/                     # Source code
│   ├── distillation/        # Distillation models implementation
│   │   ├── models.py        # Dimension and Cascade distiller models
│   ├── evaluation/          # Evaluation utilities
│   │   ├── evaluator.py     # Search performance evaluation
│   ├── training/            # Training modules
│   │   ├── trainer.py       # Trainer class with callbacks
│   ├── utils/               # Utility functions
│   │   ├── data_loader.py   # Vector loading and dimension reduction
│   │   ├── evaluation.py    # Evaluation metrics
│   ├── scripts/             # Runnable scripts
│   │   ├── train_distiller.py # Script to train distillers
```

## How It Works

The Dimensional Cascade approach works by:

1. Training neural networks to compress high-dimensional vectors into lower dimensions
2. Preserving similarity relationships between vectors across dimensions
3. Using smaller dimensions for initial filtering and progressively larger ones for refinement
4. Achieving better recall-time trade-offs than traditional approaches

This approach is particularly effective for similarity search in large-scale vector databases, where both accuracy and efficiency are critical.

## License

[MIT License](LICENSE) 