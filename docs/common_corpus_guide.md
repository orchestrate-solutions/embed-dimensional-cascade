# Dimensional Cascade Training on Common Corpus
## Step-by-Step Guide

This guide provides detailed instructions for training dimensional cascade models on the Common Corpus dataset, including setup, execution, and monitoring progress.

## 1. Setup

### 1.1 Dependencies

First, ensure you have all the required dependencies installed:

```bash
# Make sure you're in the project root directory
pip install -r requirements.txt
```

This will install:
- PyTorch for deep learning
- Transformers for embedding models
- Datasets library for accessing Common Corpus
- Other required libraries

### 1.2 Prepare Environment

```bash
# Create output directories
mkdir -p data/embeddings
mkdir -p models/common_corpus
```

## 2. Training Process Overview

The training process consists of three main phases:
1. **Data Loading**: Downloading and filtering samples from Common Corpus
2. **Embedding Generation**: Creating embeddings using the specified model
3. **Distillation Training**: Training models to reduce embedding dimensions

## 3. Basic Training Run

### 3.1 Minimal Example

Run the following command to start a basic training run:

```bash
python scripts/train_on_common_corpus.py \
  --num_samples 5000 \
  --save_embeddings data/embeddings/common_corpus_sample.npy
```

This will:
- Download 5,000 samples from Common Corpus
- Generate embeddings using Snowflake's arctic-embed-s model
- Train a cascade distiller for dimensions: 768, 512, 256, 128, 64, 32
- Save the trained models to `models/common_corpus`
- Save the generated embeddings to `data/embeddings/common_corpus_sample.npy`

### 3.2 Customized Run

For a more customized run, you can specify additional parameters:

```bash
python scripts/train_on_common_corpus.py \
  --collection OpenScience \
  --num_samples 10000 \
  --min_token_count 200 \
  --languages en \
  --embedding_model Snowflake/snowflake-arctic-embed-s \
  --target_dims 768,512,256,128,64 \
  --epochs 50 \
  --train_batch_size 128 \
  --save_embeddings data/embeddings/science_corpus.npy \
  --output_dir models/science_distillation
```

## 4. Monitoring Progress

### 4.1 Understanding the Output Logs

The script provides detailed logging at each step:

1. **Data Loading Phase**:
   ```
   INFO - Loading Common Corpus dataset...
   INFO - Filtered dataset: X samples
   INFO - Selected subset: Y samples
   ```

2. **Embedding Generation Phase**:
   ```
   INFO - Loading embedding model: Snowflake/snowflake-arctic-embed-s
   INFO - Using device: cuda
   Generating embeddings: 100%|██████████| X/X [00:YY<00:00, Z.ZZit/s]
   INFO - Generated embeddings: (X, 768)
   INFO - Saved embeddings to data/embeddings/...
   ```

3. **Distillation Training Phase**:
   ```
   INFO - Training cascade distiller for dimensions: [768, 512, 256, 128, 64, 32]
   INFO - Added distiller: 768 → 512
   INFO - Added distiller: 768 → 256
   ...
   INFO - Starting training...
   INFO - Epoch 1/50: train_loss=X.XXXXX, val_loss=X.XXXXX
   ...
   INFO - Epoch N/50: train_loss=X.XXXXX, val_loss=X.XXXXX
   ```

4. **Evaluation Phase**:
   ```
   INFO - Evaluating model...
   INFO - Training complete!
   INFO - Final training loss: X.XXXXX
   INFO - Final validation loss: X.XXXXX
   INFO - Best validation loss: X.XXXXX (epoch N)
   INFO - Similarity preservation results:
   INFO - Dimension 512: MSE=X.XXXXX, Compression ratio=1.50x
   INFO - Dimension 256: MSE=X.XXXXX, Compression ratio=3.00x
   ...
   ```

### 4.2 Expected Runtime

Runtime varies based on your hardware, but here are typical expectations:
- Data loading: 1-5 minutes for 10,000 samples (depends on network speed)
- Embedding generation: 10-30 minutes for 10,000 samples on GPU
- Training: 30 minutes to several hours depending on:
  - Number of dimensions
  - Epochs
  - Batch size
  - Hardware (GPU/CPU)

### 4.3 Output Files

After successful training, you'll have:

1. **Trained Models**:
   - `models/common_corpus/common_corpus/best_model.pt`: Best model based on validation loss
   - `models/common_corpus/common_corpus/final_model.pt`: Model after final training epoch
   - `models/common_corpus/common_corpus/learning_curve.png`: Training/validation loss plot

2. **Embedding Files** (if `--save_embeddings` was specified):
   - The generated embeddings saved as a NumPy array

## 5. Testing Trained Models

After training, test your models using:

```bash
python scripts/test_distilled_embeddings.py \
  --model_dir models/common_corpus \
  --test_queries "What is machine learning?,How do embeddings work?"
```

This will:
1. Load your trained model
2. Run the test queries against a sample corpus
3. Compare search results between original and distilled embeddings
4. Report recall and rank correlation metrics

## 6. Common Issues and Solutions

### 6.1 Out of Memory Errors

If you encounter CUDA out of memory errors:
- Reduce `--batch_size` and `--train_batch_size`
- Reduce `--num_samples`
- Use a smaller embedding model

### 6.2 Slow Training

If training is too slow:
- Use a GPU if available
- Reduce the number of `--target_dims`
- Reduce the number of `--epochs`
- Increase `--train_batch_size` if memory allows

### 6.3 Dataset Issues

If you encounter dataset errors:
- Try a different `--collection` (OpenScience, OpenSource, OpenCulture, etc.)
- Reduce `--min_token_count`
- Check your internet connection for dataset downloading

## 7. Advanced Usage

### 7.1 Using Pre-Generated Embeddings

If you've already generated embeddings, you can use them directly:

```bash
python main.py train \
  --input_file data/embeddings/common_corpus_sample.npy \
  --output_dir models/distillation \
  --target_dims 768,512,256,128,64,32 \
  --cascade_strategy direct
```

### 7.2 Using Different Models

Try different embedding models to see how they affect distillation quality:

```bash
python scripts/train_on_common_corpus.py \
  --num_samples 5000 \
  --embedding_model sentence-transformers/all-mpnet-base-v2 \
  --embedding_dim 768 \
  --output_dir models/mpnet_distillation
```

## 8. Evaluating Model Quality

After training, look at:

1. **MSE values**: Lower is better, indicates how well similarity relationships are preserved
2. **Similarity preservation results**: Should show trade-off between dimension reduction and preservation quality
3. **Test results**: Show how well rankings are preserved in real search scenarios

A well-trained model should maintain high overlap (>80%) and rank correlation (>0.8) at 256 dimensions,
with graceful degradation as dimensions are reduced further. 