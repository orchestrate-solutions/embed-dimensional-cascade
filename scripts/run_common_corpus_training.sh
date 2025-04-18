#!/bin/bash
# Script to run Common Corpus training with various configurations

# Default values
NUM_SAMPLES=5000
COLLECTION="OpenScience"
TARGET_DIMS="768,512,256,128,64,32"
EPOCHS=50
BATCH_SIZE=32
TRAIN_BATCH=64
OUTPUT_DIR="models/common_corpus"
SAVE_EMBEDDINGS="data/embeddings/common_corpus.npy"

# Create directories
mkdir -p data/embeddings
mkdir -p "$OUTPUT_DIR"

# Parse command line options
while [[ $# -gt 0 ]]; do
  case $1 in
    --samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --collection)
      COLLECTION="$2"
      shift 2
      ;;
    --dims)
      TARGET_DIMS="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --train-batch)
      TRAIN_BATCH="$2"
      shift 2
      ;;
    --output)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --save-emb)
      SAVE_EMBEDDINGS="$2"
      shift 2
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --samples NUM       Number of samples to use (default: $NUM_SAMPLES)"
      echo "  --collection NAME   Collection to use (default: $COLLECTION)"
      echo "  --dims DIMENSIONS   Target dimensions (default: $TARGET_DIMS)"
      echo "  --epochs NUM        Number of epochs (default: $EPOCHS)"
      echo "  --batch SIZE        Batch size for embedding generation (default: $BATCH_SIZE)"
      echo "  --train-batch SIZE  Batch size for training (default: $TRAIN_BATCH)"
      echo "  --output DIR        Output directory (default: $OUTPUT_DIR)"
      echo "  --save-emb PATH     Path to save embeddings (default: $SAVE_EMBEDDINGS)"
      echo "  --help              Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Print configuration
echo "Running Common Corpus training with configuration:"
echo "  Collection: $COLLECTION"
echo "  Samples: $NUM_SAMPLES"
echo "  Target dimensions: $TARGET_DIMS"
echo "  Epochs: $EPOCHS"
echo "  Embedding batch size: $BATCH_SIZE"
echo "  Training batch size: $TRAIN_BATCH"
echo "  Output directory: $OUTPUT_DIR"
echo "  Save embeddings to: $SAVE_EMBEDDINGS"
echo ""
echo "Starting at: $(date)"
echo "======================================================="

# Run the training script
python scripts/train_on_common_corpus.py \
  --collection "$COLLECTION" \
  --num_samples "$NUM_SAMPLES" \
  --target_dims "$TARGET_DIMS" \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --train_batch_size "$TRAIN_BATCH" \
  --output_dir "$OUTPUT_DIR" \
  --save_embeddings "$SAVE_EMBEDDINGS"

# Check the exit status
if [ $? -eq 0 ]; then
  echo "======================================================="
  echo "Training completed successfully at: $(date)"
else
  echo "======================================================="
  echo "Training failed with error code $? at: $(date)"
fi 