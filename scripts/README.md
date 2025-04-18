# Training Scripts for Dimensional Cascade

This directory contains scripts for training and evaluating dimensional cascade models.

## Training on Common Corpus

The `train_on_common_corpus.py` script allows you to train dimensional distillation models using the [Common Corpus dataset](https://huggingface.co/datasets/PleIAs/common_corpus), a large collection of permissibly licensed text data (2 trillion tokens).

### Prerequisites

Ensure you have installed the required dependencies:

```bash
pip install -r ../requirements.txt
```

### Usage

Basic usage:

```bash
python train_on_common_corpus.py
```

This will download a subset of the Common Corpus, generate embeddings using Snowflake's arctic-embed model, and train a dimensional cascade distiller.

### Command Line Options

#### Dataset Parameters

- `--num_samples`: Number of samples to use from the dataset (default: 10000)
- `--collection`: Filter for specific collection (e.g., "OpenScience", "OpenSource")
- `--min_token_count`: Minimum token count for text samples (default: 100)
- `--languages`: Comma-separated list of language codes to include (default: "en")

#### Embedding Model Parameters

- `--embedding_model`: Model to use for generating embeddings (default: "Snowflake/snowflake-arctic-embed-s")
- `--embedding_dim`: Dimension of embeddings from the model (default: 768)
- `--max_length`: Maximum sequence length for tokenization (default: 512)
- `--batch_size`: Batch size for embedding generation (default: 32)
- `--query_prefix`: Prefix to use for queries (default: "query: ")

#### Distillation Parameters

- `--target_dims`: Target dimensions for distillation, comma-separated (default: "512,256,128,64,32")
- `--learning_rate`: Learning rate for distillation (default: 1e-3)
- `--train_batch_size`: Batch size for distillation training (default: 64)
- `--epochs`: Number of training epochs (default: 100)
- `--val_split`: Validation split ratio (default: 0.1)
- `--output_dir`: Output directory for saving models (default: "models/common_corpus")
- `--save_embeddings`: Path to save generated embeddings (optional)
- `--device`: Device to use ("cuda", "mps", "cpu", or None for auto-detection)

### Examples

**Training on OpenScience collection with 5000 samples:**

```bash
python train_on_common_corpus.py \
  --collection OpenScience \
  --num_samples 5000 \
  --min_token_count 200 \
  --target_dims 256,128,64 \
  --epochs 50 \
  --save_embeddings data/openscience_embeddings.npy
```

**Training on OpenSource (code) collection:**

```bash
python train_on_common_corpus.py \
  --collection OpenSource \
  --num_samples 10000 \
  --embedding_model all-MiniLM-L6-v2 \
  --embedding_dim 384 \
  --target_dims 256,128,64,32 \
  --output_dir models/code_distillation
```

**Using multi-language data:**

```bash
python train_on_common_corpus.py \
  --languages en,fr,de,es \
  --num_samples 20000 \
  --min_token_count 150 \
  --output_dir models/multilingual_distillation
``` 